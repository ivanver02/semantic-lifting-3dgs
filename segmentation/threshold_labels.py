import torch
import os
import tempfile
import sys
from argparse import ArgumentParser
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, save_npz, load_npz
from scipy.sparse.csgraph import connected_components

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import GaussianModel


def _load_gaussians(args):
    """ Loads the Gaussian model from the specified path and iteration """

    print(f"Loading gaussian model from {args.model_path}")
    start_sh = args.sh_degree if hasattr(args, 'sh_degree') else 3
    gaussians = GaussianModel(sh_degree=start_sh, use_labels=True)
    loaded_iter = args.loaded_iter if hasattr(args, 'loaded_iter') else 30000
    ply_path = os.path.join(args.model_path, "point_cloud", f"iteration_{loaded_iter}", "point_cloud.ply")
    gaussians.load_ply(ply_path)
    return gaussians


def _graph_path(args):

    # Store derived graph data outside the read-only model mount
    token = str(float(args.hysteresis_radius)).replace(".", "_")
    ply_path = os.path.join(
        args.model_path, "point_cloud", f"iteration_{args.loaded_iter}",
        "point_cloud.ply",
    )

    # Read the model fingerprint
    stat = os.stat(ply_path)
    cache_dir = getattr(args, "cache_dir", None) or args.output_dir
    return os.path.join(
        cache_dir,
        f"hysteresis_graph_i{args.loaded_iter}_n{stat.st_size}_m{stat.st_mtime_ns}"
        f"_r{token}.npz",
    )


def _hysteresis_graph(args, xyz):
    """ Load or cache the radius graph shared by a batch """

    # Reuse a valid graph cache
    path = _graph_path(args)
    if os.path.exists(path):
        graph = load_npz(path).tocsr()
        if graph.shape == (len(xyz), len(xyz)):
            return graph

    # Build the radius graph when no valid cache exists
    pairs = cKDTree(xyz).query_pairs(
        args.hysteresis_radius, output_type="ndarray",
    )

    # Convert nearby pairs into a sparse adjacency matrix
    if len(pairs):
        rows = np.concatenate((pairs[:, 0], pairs[:, 1]))
        cols = np.concatenate((pairs[:, 1], pairs[:, 0]))
        graph = csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)),
                           shape=(len(xyz), len(xyz)))
    else:
        graph = csr_matrix((len(xyz), len(xyz)), dtype=np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(

    # Create the temporary graph archive
        dir=os.path.dirname(path), suffix=".tmp.npz",
    )
    os.close(fd)
    try:
        save_npz(temporary_name, graph)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):

    # Remove an incomplete graph archive
            os.unlink(temporary_name)
    return graph


def apply_threshold(args, gaussians=None, voting_data=None):
    """ Applies the threshold to the voting weights and saves the resulting segmented PLY file """
    if voting_data is None:
        voting_data = torch.load(args.voting_data_path, map_location=args.device)

    # Each tensor contains one accumulated evidence value per Gaussian
    target_weights = voting_data['target_weights']
    background_weights = voting_data['background_weights']

    # Stored ID of the target class represented by the selected Gaussians
    target_id = voting_data['target_id']

    # beta is the minimum target fraction of the total evidence
    if not 0.0 <= args.beta <= 1.0:
        raise ValueError("target evidence beta must be in [0, 1]")

    # Combine both types of evidence. Unsupported Gaussians have zero evidence
    evidence = target_weights + background_weights
    score = torch.zeros_like(target_weights)
    supported = evidence > 0

    # Compute the fraction of supported evidence assigned to the target class. Background competes here
    score[supported] = target_weights[supported] / evidence[supported]

    # Report the threshold, background mode and number of supported Gaussians.
    print(f"Applying target/background threshold beta={args.beta:.3f} "
          f"(mode={voting_data.get('background_mode', 'unknown')}, "
          f"supported={int(supported.sum().item())})")

    # Keep supported Gaussians whose target evidence ratio reaches beta
    final_mask = supported & (score >= args.beta)

    # Optionally expand high confidence seeds through nearby lower score Gaussians using hysteresis on the radius graph
    gamma = getattr(args, 'hysteresis_gamma', 0.0)
    if gamma > 0:

        # Hysteresis needs Gaussian positions, so load the model if necessary
        if gaussians is None:
            gaussians = _load_gaussians(args)
        xyz = gaussians.get_xyz
        if torch.is_tensor(xyz):
            xyz = xyz.detach().cpu().numpy()

        # Work on CPU copies while preserving the original tensors
        score_cpu = score.detach().cpu()
        seed = final_mask.detach().cpu()

        # The low threshold defines candidate bridge Gaussians around the seeds
        low_threshold_mask = supported.detach().cpu() & (score_cpu >= args.beta * gamma)
        seed_count = int(seed.sum().item())

        # Hysteresis requires both a nonempty bridge set and at least one seed
        if low_threshold_mask.sum().item() > 0 and seed_count > 0:

            # Load or cache the radius graph shared by this model
            hysteresis_graph = _hysteresis_graph(args, xyz)

            # Group bridge Gaussians into spatially connected components
            low_indices = np.flatnonzero(low_threshold_mask.numpy())
            component_labels = connected_components(
                hysteresis_graph[low_indices][:, low_indices],
                directed=False, return_labels=True,
            )[1]

            # Keep only components containing at least one high threshold seed
            seed_in_low_threshold_mask = seed[low_threshold_mask].numpy()
            keep_component = np.zeros(component_labels.max() + 1, dtype=bool)

            # Mark components that contain a seed
            np.logical_or.at(keep_component, component_labels, seed_in_low_threshold_mask)
            kept = keep_component[component_labels]

            # Reconstruct the full Gaussian mask from the retained components
            new_mask = torch.zeros_like(seed)
            new_mask[low_threshold_mask] = torch.from_numpy(kept)
            n_comps = component_labels.max() + 1

            # Report the hysteresis expansion and the retained components
            print(f"hysteresis phase: gamma={gamma} radius={args.hysteresis_radius} | "
                  f"seeds={seed_count} low_threshold_mask={int(low_threshold_mask.sum().item())} comps={n_comps} "
                  f"kept_comps={int(keep_component.sum())} | "
                  f"{seed_count} -> {int(new_mask.sum().item())} gaussians")

            # Return the final mask to the original device
            final_mask = new_mask.to(final_mask.device)
        else:
            # Keep the beta mask when hysteresis has no valid seed or bridge set
            print("hysteresis phase: degenerate set, keeping seed mask")

    # Count the Gaussians selected after thresholding and optional hysteresis
    count = final_mask.sum().item()
    print(f"Labeled {count} gaussians as {target_id}")

    # An empty selection is valid and is still saved as an empty PLY
    if count == 0:
        print("Warning: No Gaussians selected with this threshold; saving an empty PLY.")

    # Load the model if it was not already loaded for hysteresis
    if gaussians is None:
        gaussians = _load_gaussians(args)

    # Build a safe filesystem class name for the output path
    raw_class_name = args.target_class if hasattr(args, 'target_class') else str(target_id)
    safe_class_name = raw_class_name.replace(" ", "_")
    
    # Include beta so outputs from different thresholds can be distinguished
    filename = f"labeled_gaussians_{safe_class_name}"
    if hasattr(args, 'beta'):
        beta_str = str(args.beta).replace('.', '_')
        filename += f"_beta{beta_str}"
    filename += ".ply"
    
    # Store each target class in its own output directory
    target_class_dir = os.path.join(args.output_dir, safe_class_name)
    os.makedirs(target_class_dir, exist_ok=True)
    
    output_ply = os.path.join(target_class_dir, filename)

    # Select and save only the Gaussians contained in the final mask
    gaussians.set_mask_index(final_mask.nonzero(as_tuple=True)[0])
    gaussians.save_ply(output_ply)
    print(f"Saved labeled PLY to {output_ply}")

if __name__ == "__main__":
    parser = ArgumentParser()

    # Model and target configuration
    parser.add_argument("--model_path", required=True, help="Path to trained 3DGS model output")
    parser.add_argument("--sh_degree", type=int, default=3, help="SH degree")
    parser.add_argument("--loaded_iter", type=int, default=30000, help="Iteration of model to load")
    parser.add_argument("--target_class", type=str, default="object", help="Name of target class, for filename")

    # Input and output paths
    parser.add_argument("--voting_data_path", type=str, required=True, help="Path to .pt file containing voting weights")
    parser.add_argument("--output_dir", required=True, help="Directory to save labeled PLY")
    parser.add_argument("--cache_dir", default=None, help="Directory for derived hysteresis graph cache")

    # Target selection
    parser.add_argument("--beta", type=float, default=0.5, help="Minimum target evidence ratio in [0, 1]")

    # Device configuration
    parser.add_argument("--device", type=str, default="cuda", help="Device, either cuda or cpu")
    
    # Hysteresis expansion
    parser.add_argument("--hysteresis_gamma", type=float, default=0.8, help="Low-threshold factor. 0 disables hysteresis")
    parser.add_argument("--hysteresis_radius", type=float, default=0.05, help="Connectivity radius in meters for the bridge set")
    args = parser.parse_args()
    if args.hysteresis_gamma < 0.0:
        raise ValueError("--hysteresis_gamma must be non-negative")
    if args.hysteresis_radius <= 0.0:
        raise ValueError("--hysteresis_radius must be greater than zero")
    
    with torch.no_grad():
        apply_threshold(args)