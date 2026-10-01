import torch
import os
import sys
import cv2
import json
import numpy as np
from argparse import ArgumentParser

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel
from arguments import get_combined_args
from segmentation.projection import get_covariance_3d, project_gaussians
from segmentation.threshold_labels import target_fraction
from evaluation.analytics import QUANTILES
from evaluation.common import atomic_write

# Camera matrices are always stored on the GPU by the Camera class
DEVICE = "cuda"


def get_target_class_id(mask_dir, target_class):
    """
    Retrieve the stored detector ID for a detector name

    target_class is a detector name at this stage, not a main project name or dataset name
    """

    # Load the detector ID mapping
    with open(os.path.join(mask_dir, "classes.json"), 'r') as f:
        classes_map = json.load(f)

    # Invert the stored ID to detector name mapping
    name_to_id = {v: int(k) for k, v in classes_map.items()}

    # Resolve the requested detector name
    if target_class not in name_to_id:
        raise ValueError(f"Target class {target_class} not found in classes.json")

    return name_to_id[target_class]


def mask_name(camera):
    """ Name of the semantic and confidence PNG files of a camera """
    return os.path.splitext(os.path.basename(camera.image_name))[0] + ".png"


def load_masks(mask_dir, camera):
    """ Load the stored label map and the detector confidence map of one camera, at the camera resolution """
    images = []
    for folder in ("semantic", "confidence"):
        image = cv2.imread(os.path.join(mask_dir, folder, mask_name(camera)), cv2.IMREAD_UNCHANGED)  # (H, W)

        # Resize the mask to match camera dimensions
        if image.shape[:2] != (camera.image_height, camera.image_width):
            image = cv2.resize(image, (camera.image_width, camera.image_height), interpolation=cv2.INTER_NEAREST)
        images.append(image)

    detector_label_mask = torch.tensor(images[0], dtype=torch.long, device=DEVICE)

    # Normalize stored byte confidence values
    confidence_mask = torch.tensor(images[1], dtype=torch.float32, device=DEVICE) / 255.0
    return detector_label_mask, confidence_mask


def get_pixel_confidence(detector_label_mask, confidence_mask, background_confidence):
    """
    Return the confidence every pixel votes with

    A pixel that belongs to a detection keeps the confidence of that detection,
    whether its label is the target or not, and background_confidence is the
    fallback for the pixels that no detection claimed, which carry the stored identifier zero.
    """
    pixel_confidence = confidence_mask.clone()
    pixel_confidence[detector_label_mask == 0] = background_confidence
    return pixel_confidence


def accumulate_view(cam, gaussians, cov3D, target_confidence, background_confidence, block_size):
    """
    Accumulate the target and background votes of every Gaussian in one camera view

    target_confidence and background_confidence are the pixel confidence maps restricted
    to the target pixels and to the rest of the pixels, so both channels share the same
    visibility weights and only the mask separates them.

    Returns the original indices of the Gaussians that reach the image and their two votes
    """
    width, height = cam.image_width, cam.image_height

    # Projection of 3D Gaussians into 2D camera space
    # projection_results is a map with means2D, cov2D, depths, and indices of the Gaussians that are visible in this camera view
    projection_results = project_gaussians(cam, gaussians.get_xyz, cov3D)

    means2D = projection_results['means2D']
    cov2D = projection_results['cov2D']  # (M, 2, 2)
    depths = projection_results['depths']  # (M,)
    indices = projection_results['indices']  # (M,)

    opacities = gaussians.get_opacity[indices].squeeze(1)  # (M,)
    '''
    Equation 4 but projected in 2D
    Calculate the inverse matrix once per Gaussian instead of once per pixel
    Sigma is semidefinite positive, so simmetric (and with non negative eigenvalues)
    Store the inverse covariance as conic parameters for the ellipse formula
    As it's symmetric, we have only 3 unique values: [[A, B], [B, C]] where A = inv_cov2D[0,0], B = inv_cov2D[0,1], C = inv_cov2D[1,1]
    '''

    det = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1] * cov2D[:, 0, 1]
    det_inv = 1.0 / det
    conic = torch.stack([
        cov2D[:, 1, 1] * det_inv,
        -cov2D[:, 0, 1] * det_inv,
        cov2D[:, 0, 0] * det_inv
    ], dim=1)

    '''
    A Gaussian theoretically stretches to infinity, but in practice, its energy is negligible after 3 standard deviations.
    To find the "width" of the Gaussian, we need the lengths of its major and minor axes.
    These lengths are the square roots of the covariance eigenvalues
    '''

    # We can compute the largest eigenvalue of the 2D covariance matrix using the formula for 2x2 matrices:
    trace = cov2D[:, 0, 0] + cov2D[:, 1, 1]
    discriminant = torch.clamp(trace * trace - 4 * det, min=0.0)
    largest_eigenvalue = 0.5 * (trace + torch.sqrt(discriminant))
    radius = torch.ceil(3.0 * torch.sqrt(largest_eigenvalue))  # (M,)

    # Sorting the gaussians by depth, before focusing on any tile
    sort_indices = torch.argsort(depths)

    # Keep original indices so tile votes can be restored after depth sorting
    means2D, conic, opacities, radius, sorted_original_indices = (
        means2D[sort_indices], conic[sort_indices], opacities[sort_indices],
        radius[sort_indices], indices[sort_indices])

    # Frustum culling: filter out gaussians that don't overlap with the image
    # The projection only checks z > znear, so we need to filter the x and y bounds to match the view
    # Keep Gaussian centers inside the image to avoid bleed from outside the view
    in_frustum = ((means2D[:, 0] >= 0) & (means2D[:, 0] < width) &
                  (means2D[:, 1] >= 0) & (means2D[:, 1] < height))
    means2D, conic, opacities, radius, sorted_original_indices = (
        means2D[in_frustum], conic[in_frustum], opacities[in_frustum],
        radius[in_frustum], sorted_original_indices[in_frustum])

    # Initialize tensors to store the accumulated weights for each visible Gaussian
    num_visible = means2D.shape[0]
    view_target_weights_sorted = torch.zeros((num_visible,), device=DEVICE, dtype=torch.float32)
    view_background_weights_sorted = torch.zeros((num_visible,), device=DEVICE, dtype=torch.float32)

    # Rasterization:
    grid_columns = (width + block_size - 1) // block_size
    grid_rows = (height + block_size - 1) // block_size

    # Convert the Gaussian's 2D position and size from pixel coordinates to tile coordinates
    # They represent the bounding tiles for each Gaussian in terms of tile indices
    grid_min_x = ((means2D[:, 0] - radius).clamp(min=0) / block_size).int()
    grid_min_y = ((means2D[:, 1] - radius).clamp(min=0) / block_size).int()
    grid_max_x = ((means2D[:, 0] + radius).clamp(max=width-1) / block_size).int()
    grid_max_y = ((means2D[:, 1] + radius).clamp(max=height-1) / block_size).int()

    for row_tile in range(grid_rows):
        for column_tile in range(grid_columns):
            # Find which Gaussians have bounding tiles that include this tile
            in_tile = (grid_min_x <= column_tile) & (column_tile <= grid_max_x) & (grid_min_y <= row_tile) & (row_tile <= grid_max_y)
            # Check which projected Gaussian centers overlap this tile
            gaussians_in_tile = torch.nonzero(in_tile).squeeze(1)

            if gaussians_in_tile.shape[0] == 0:
                continue

            '''
            For each overlapping Gaussian, calculate its contribution to the tile pixels
            This applies the 2D Gaussian formula using precomputed conic parameters and opacities,
            accumulates pixel blending weights, checks the semantic mask against the target class,
            and accumulate votes using the original indices
            '''

            tile_means = means2D[gaussians_in_tile]
            tile_conics = conic[gaussians_in_tile]
            tile_opacities = opacities[gaussians_in_tile]

            # Obtain the boundaries of the current tile in pixel coordinates
            pix_min_x = column_tile * block_size
            pix_min_y = row_tile * block_size
            pix_max_x = min(pix_min_x + block_size, width)
            pix_max_y = min(pix_min_y + block_size, height)

            '''
            y comes before x because images are indexed by height then width

            If y_range is tensor([10, 11, 12]), then grid_y will be:
            tensor([[10, 10, 10],
                    [11, 11, 11],
                    [12, 12, 12]])
            And flat_y will be tensor([10, 10, 10, 11, 11, 11, 12, 12, 12])

            Similarly, if x_range is tensor([20, 21, 22]), then grid_x will be:
            tensor([[20, 21, 22],
                    [20, 21, 22],
                    [20, 21, 22]])
            And flat_x will be tensor([20, 21, 22, 20, 21, 22, 20, 21, 22])
            '''

            # Create a grid of pixel coordinates for the current tile
            y_range = torch.arange(pix_min_y, pix_max_y, device=DEVICE)
            x_range = torch.arange(pix_min_x, pix_max_x, device=DEVICE)

            # grid_y becomes a 2D grid where every row is identical, grid_x becomes a 2D grid where every column is identical
            grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')

            # Write the grid in a whole 1D array to make it easier to compute the Gaussian formula for all pixels in the tile at once
            flat_y = grid_y.flatten()
            flat_x = grid_x.flatten()

            '''
            Equation 4 in the paper, but using the conic parameters and opacities, and applied to the pixels in this tile
            For each pixel in the tile, calculate its distance to the Gaussian centers and apply the Gaussian formula using the conic parameters
            unsqueeze(0) is used to expand the dimensions of the pixel coordinates so that they can be broadcasted against the Gaussian parameters, (, N_Pixels_in_tile) -> (1, N_Pixels_in_tile)
            unsqueeze(1) is used to expand the dimensions of the Gaussian parameters so that they can be broadcasted against the pixel coordinates, (N_Gaussians_in_tile, ) -> (N_Gaussians_in_tile, 1)
            Now the shapes are compatible for broadcasting, resulting in a tensor of shape (N_Gaussians_in_tile, N_Pixels_in_tile)
            '''

            dx = flat_x.unsqueeze(0) - tile_means[:, 0].unsqueeze(1)
            dy = flat_y.unsqueeze(0) - tile_means[:, 1].unsqueeze(1)

            # This calculates how intense the Gaussian is at those specific distances
            gaussian_exponent = -0.5 * (tile_conics[:, 0].unsqueeze(1) * dx**2 +
                            tile_conics[:, 2].unsqueeze(1) * dy**2) - \
                            tile_conics[:, 1].unsqueeze(1) * dx * dy

            # The opacity of the Gaussian modulates its contributions
            # alpha shape: (N_Gaussians_in_tile, N_Pixels_in_tile)
            alpha = tile_opacities.view(-1, 1) * torch.exp(gaussian_exponent.clamp(max=0))

            transmission = 1.0 - alpha
            accumulated_transmission = torch.cumprod(transmission, dim=0)

            # We need to know how much light reached the current layer
            # Ones is one row of ones, being each column a pixel in the tile
            ones = torch.ones((1, alpha.shape[1]), device=DEVICE)

            # The rest of the rows are the accumulated transmission of the previous gaussians, which tells us how much light reaches the current layer
            T = torch.cat([ones, accumulated_transmission[:-1]], dim=0)  # (N_Gaussians_in_tile, N_Pixels_in_tile)

            '''
            First, alpha is multiplied by the accumulated transmission T to obtain the
            contribution of each Gaussian to each pixel:

            weights = alpha * T

            weights has shape (N_gaussians_in_tile, N_pixels_in_tile). Each row corresponds to one Gaussian and each column
            to one pixel in the tile, representing each Gaussian contribution
            '''

            weights = alpha * T

            '''
            The target and background confidences are then applied independently.
            Both confidence tensors have shape (N_pixels_in_tile,), and they are zero
            outside their own pixels, so a matrix product sums the contributions:

            Shape (K, P) @ Shape (P,) -> Shape (K,)

            where K is the number of Gaussians and P is the number of pixels
            '''

            # Vote Calculation: alpha * T * pixel_confidence
            # Sum all the pixel contributions for the target class and background to get the total vote for each Gaussian in this tile
            target_votes = weights @ target_confidence[flat_y, flat_x]
            background_votes = weights @ background_confidence[flat_y, flat_x]

            # Accumulate tile votes in depth order until the view is complete
            view_target_weights_sorted[gaussians_in_tile] += target_votes
            view_background_weights_sorted[gaussians_in_tile] += background_votes

    return sorted_original_indices, view_target_weights_sorted, view_background_weights_sorted


def main(args):
    # Define the gaussians, Scene loads the trained model at the requested iteration
    # Source images stay on args.data_device while Scene builds camera data
    gaussians = GaussianModel(sh_degree=args.sh_degree)
    scene = Scene(args, gaussians, load_iteration=args.loaded_iter, shuffle=False)
    cov3D = get_covariance_3d(gaussians)

    # Initialize a tensor to accumulate votes for each Gaussian across all camera views, and another one for background votes
    total_gaussians = gaussians.get_xyz.shape[0]
    global_target_weights = torch.zeros((total_gaussians,), device=DEVICE, dtype=torch.float32)
    global_background_weights = torch.zeros((total_gaussians,), device=DEVICE, dtype=torch.float32)

    # Read stored detector names and resolve the requested class
    target_id = get_target_class_id(args.mask_dir, args.target_class)

    # Cameras that have a corresponding 2D mask
    masked_cameras = [
        cam for cam in scene.getTrainCameras()
        if os.path.exists(os.path.join(args.mask_dir, "confidence", mask_name(cam)))
    ]
    print(f"Matched {len(masked_cameras)} cameras in the scene.")

    class_views = 0  # Views where the target class actually appears

    # Iterate through the matched cameras and accumulate votes for the target class
    for cam in masked_cameras:
        detector_label_mask, confidence_mask = load_masks(args.mask_dir, cam)

        # Check whether the target detector ID is present in this view
        target_pixels = detector_label_mask == target_id
        if target_pixels.any():
            class_views += 1
        elif args.background_view_policy == "target_views":
            # Empty detector views provide no positive evidence and would let uncertain background dominate the ratio
            continue

        # Every pixel outside the target class votes for the background channel
        pixel_confidence = get_pixel_confidence(detector_label_mask, confidence_mask, args.background_confidence)
        indices, target_votes, background_votes = accumulate_view(
            cam, gaussians, cov3D,
            pixel_confidence * target_pixels, pixel_confidence * ~target_pixels,
            args.raster_block_size)

        # After processing all tiles for this view, we add the votes from this view to the global weights tensor using the original indices of the Gaussians
        global_target_weights[indices] += target_votes
        global_background_weights[indices] += background_votes

    # Summarize the target evidence fraction over supported Gaussians, the distribution that beta cuts
    scores, supported = target_fraction(global_target_weights, global_background_weights)
    values = scores[supported].cpu().numpy()
    quantiles = np.quantile(values, list(QUANTILES.values())) if len(values) else [None] * len(QUANTILES)
    statistics = {
        'num_cameras': len(masked_cameras),
        'num_class_views': class_views,
        'num_gaussians': total_gaussians,
        'supported_gaussians': int(supported.sum().item()),
        'supported_fraction': float(supported.float().mean().item()),
        **{f'target_score_{name}': None if value is None else float(value)
           for name, value in zip(QUANTILES, quantiles)},
    }

    # Save the global votes for later use in thresholding, with their statistics beside them
    voting_data = {
        'target_weights': global_target_weights.cpu(),
        'background_weights': global_background_weights.cpu(),
        'num_cameras': len(masked_cameras),
        'num_class_views': class_views,
    }
    statistics_path = os.path.join(os.path.dirname(args.output_path), "vote_statistics.json")
    atomic_write(statistics_path, lambda path: path.write_text(json.dumps(statistics, indent=2) + "\n"))
    atomic_write(args.output_path, lambda path: torch.save(voting_data, path))
    print(f"Saved voting weights to {args.output_path}")


if __name__ == "__main__":
    parser = ArgumentParser()

    # Model, source data and target configuration
    parser.add_argument("--model_path", required=True, help="Path to trained 3DGS model output")
    parser.add_argument("--source_path", required=True, help="Prepared dataset directory used by Scene")
    parser.add_argument("--mask_dir", required=True, help="Directory containing semantic and confidence masks")
    parser.add_argument("--output_path", required=True, help="Voting data PT file to write, vote_statistics.json is written beside it")
    parser.add_argument("--target_class", required=True, help="Detector name of the segmented class; only one object at a time")
    parser.add_argument("--sh_degree", type=int, default=3)  # Spherical Harmonics degree for the Gaussian model
    parser.add_argument("--loaded_iter", type=int, default=30000, help="Iteration number to load from the model")

    # Device configuration and performance
    parser.add_argument("--data_device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for source images, camera matrices remain on the GPU")
    parser.add_argument("--raster_block_size", type=int, default=16, help="Block size for rasterization. Larger blocks are faster but less precise.")

    # Background handling parameters
    parser.add_argument("--background_confidence", type=float, default=0.25, help="Confidence assigned to pixels with stored semantic label zero")
    parser.add_argument("--background_view_policy", type=str, default="target_views", choices=["target_views", "all_views"],
        help="Use only views containing target pixels or every matched view")

    args = get_combined_args(parser)

    if args.raster_block_size <= 0:
        raise ValueError("--raster_block_size must be greater than zero")
    if not 0.0 <= args.background_confidence <= 1.0:
        raise ValueError("--background_confidence must be in [0, 1]")

    with torch.no_grad():
        main(args)
