# Render qualitative images with Gaussians recolored by prediction state

import os
import sys
import torch
import cv2
import numpy as np
from argparse import ArgumentParser

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args
from evaluation.transfer import map_subset_indices
from plyfile import PlyData


def _load_iteration_ply(model_path, loaded_iter):
    """ Return the full model PLY path used by every stage """
    return os.path.join(
        model_path, "point_cloud", f"iteration_{loaded_iter}", "point_cloud.ply",
    )


def _selected_colors(num_gaussians, selected_indices, target_color, show_base):
    """
    Build one RGB row per Gaussian for a labeled subset rendering

    The selected Gaussians receive the solid target color. The rest stays
    visible in light gray when context is requested, or fades into the white
    background otherwise.
    """
    base_color = 0.8 if show_base else 0.97
    colors = torch.full((num_gaussians, 3), base_color, dtype=torch.float32)
    if len(selected_indices):
        colors[torch.as_tensor(selected_indices, dtype=torch.long)] = torch.tensor(
            target_color, dtype=torch.float32,
        )
    return colors


def _score_colors(target_weights, background_weights, beta):
    """
    Build one RGB row per Gaussian colored by its target fraction rho

    Supported Gaussians use the colormap with beta as the boundary of the
    scale; unsupported Gaussians stay neutral gray. Returns the colors and the
    supported scores for the legend.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    evidence = target_weights + background_weights
    supported = evidence > 0
    scores = torch.zeros_like(evidence)
    scores[supported] = target_weights[supported] / evidence[supported]

    # beta splits the colour scale, matching the manuscript description
    norm = TwoSlopeNorm(vmin=0.0, vcenter=float(beta), vmax=1.0)
    rgba = plt.cm.turbo(norm(scores.numpy()))
    colors = torch.from_numpy(np.asarray(rgba[:, :3], dtype=np.float32))
    colors[~supported] = 0.85
    return colors, scores[supported]


def _save_score_legend(output_dir, beta):
    """ Save one colorbar image that documents the score scale and beta """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    fig, ax = plt.subplots(figsize=(4, 1.2))
    image = ax.imshow(
        np.linspace(0, 1, 256)[None, :], aspect="auto",
        norm=TwoSlopeNorm(vmin=0.0, vcenter=float(beta), vmax=1.0),
        cmap="turbo",
    )
    ax.axvline(int(beta * 255), color="black", linewidth=1.5)
    ax.text(int(beta * 255), -0.6, r"$\beta$", ha="center", fontsize=10)
    ax.set_yticks([])
    fig.colorbar(image, orientation="vertical", label=r"$\rho$")
    ax.set_axis_off()
    legend_path = os.path.join(output_dir, "score_colorbar.png")
    fig.savefig(legend_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved score legend to {legend_path}")


def main(args, pipe):
    # Define the gaussians and load the trained model
    gaussians = GaussianModel(sh_degree=args.sh_degree, use_labels=True)
    gaussians.load_ply(_load_iteration_ply(args.model_path, args.loaded_iter))

    # Build the camera set from the prepared dataset
    scene = Scene(args, gaussians, load_iteration=args.loaded_iter, shuffle=False)

    # Source images are never used by the rasterizer, so release them early
    if getattr(args, "data_device", "cuda") == "cpu":
        for camera in scene.getTrainCameras():
            for attribute in ("original_image", "alpha_mask", "gt_alpha_mask"):
                if hasattr(camera, attribute):
                    setattr(camera, attribute, None)

    total_gaussians = gaussians.get_xyz.shape[0]
    xyz = gaussians.get_xyz.detach().cpu().numpy()

    # Resolve the per-Gaussian colours for the requested mode
    if args.labeled_ply is not None:
        vertex = PlyData.read(str(args.labeled_ply))["vertex"]
        subset_xyz = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float64)
        selected_indices = map_subset_indices(xyz, subset_xyz)
        colors = _selected_colors(total_gaussians, selected_indices, args.color, args.show_base)
    else:
        voting_data = torch.load(args.voting_data, map_location="cpu")
        colors, _ = _score_colors(
            voting_data["target_weights"].float(),
            voting_data["background_weights"].float(),
            args.beta,
        )
        _save_score_legend(args.output_dir, args.beta)

    colors = colors.to(torch.device("cuda"))

    # Render the requested number of views with white background
    background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
    cameras = sorted(scene.getTrainCameras(), key=lambda item: item.image_name)
    os.makedirs(args.output_dir, exist_ok=True)

    for index, camera in enumerate(cameras[:max(1, args.num_views)]):
        output = render(camera, gaussians, pipe, background, override_color=colors)
        image = output.clamp(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
        stem = os.path.splitext(os.path.basename(camera.image_name))[0]
        image_path = os.path.join(args.output_dir, f"{stem}_qualitative.png")
        cv2.imwrite(image_path, (image * 255.0).astype(np.uint8)[:, :, ::-1])
        print(f"Saved {image_path}")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser()

    # Model and dataset configuration shared with training and fusion stages
    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)

    # Iteration to load and how many views to render
    parser.add_argument("--loaded_iter", type=int, default=30000, help="Iteration number to load from the model")
    parser.add_argument("--num_views", type=int, default=6, help="Number of first train cameras to render, in name order")

    # First mode, highlight a labeled Gaussian subset with one solid colour
    parser.add_argument("--labeled_ply", type=str, default=None,
        help="Labeled Gaussian PLY (prediction or GT reference subset) to recolour")
    parser.add_argument("--color", type=float, nargs=3, default=[1.0, 0.0, 0.0],
        help="RGB colour in [0, 1] applied to the labeled Gaussians")
    parser.add_argument("--show_base", action="store_true",
        help="Keep the unlabeled Gaussians visible in light gray instead of fading them out")

    # Second mode, colour every supported Gaussian by its target fraction rho
    parser.add_argument("--voting_data", type=str, default=None,
        help="Voting data PT file holding target/background weights for the same model")
    parser.add_argument("--beta", type=float, default=None,
        help="Operating point used as the boundary of the colour scale")

    parser.add_argument("--output_dir", required=True, help="Directory for the rendered PNG images")

    args = get_combined_args(parser)

    if (args.labeled_ply is None) == (args.voting_data is None):
        raise SystemExit("pass either --labeled_ply (subset highlight) or --voting_data plus --beta (score colours)")
    if args.voting_data is not None and not 0.0 < args.beta < 1.0:
        raise SystemExit("--beta must lie strictly between 0 and 1 to split the colour scale")
    if any(not 0.0 <= channel <= 1.0 for channel in args.color):
        raise SystemExit("--color channels must be in [0, 1]")

    with torch.no_grad():
        main(args, pipeline_params.extract(args))
