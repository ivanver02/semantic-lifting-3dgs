import torch
import os
import tempfile
import sys
import json
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


def apply_threshold(args, gaussians, voting_data, hysteresis_graph, beta,
                    hysteresis_gamma, target_class, class_output_dir):
    """ Applies the threshold to the voting weights and saves the resulting segmented PLY file """

    # Skip existing outputs before loading votes or computing hysteresis
    safe_class_name = target_class.replace(" ", "_")
    target_class_dir = class_output_dir
    filename = f"labeled_gaussians_{safe_class_name}_beta{str(beta).replace('.', '_')}.ply"
    output_ply = os.path.join(target_class_dir, filename)
    if os.path.exists(output_ply) and not getattr(args, "force", False):
        return

    # Each tensor contains one accumulated evidence value per Gaussian
    target_weights = voting_data['target_weights']
    background_weights = voting_data['background_weights']

    # Stored ID of the target class represented by the selected Gaussians
    target_id = voting_data['target_id']

    # beta is the minimum target fraction of the total evidence
    if not 0.0 <= beta <= 1.0:
        raise ValueError("target evidence beta must be in [0, 1]")

    # Combine target and background evidence
    evidence = target_weights + background_weights
    score = torch.zeros_like(target_weights)
    supported = evidence > 0

    # Compute the target evidence fraction
    score[supported] = target_weights[supported] / evidence[supported]

    # Report threshold, background mode and supported Gaussian count
    print(f"Applying target/background threshold beta={beta:.3f} "
          f"(mode={voting_data.get('background_mode', 'unknown')}, "
          f"supported={int(supported.sum().item())})")

    # Keep supported Gaussians whose target evidence ratio reaches beta
    final_mask = supported & (score >= beta)

    # Optionally expand high confidence seeds through nearby lower score Gaussians using hysteresis on the radius graph
    gamma = hysteresis_gamma
    if gamma > 0:

        xyz = gaussians.get_xyz
        if torch.is_tensor(xyz):
            xyz = xyz.detach().cpu().numpy()

        # Work on CPU copies while preserving the original tensors
        score_cpu = score.detach().cpu()
        seed = final_mask.detach().cpu()

        # The low threshold defines candidate bridge Gaussians around the seeds
        low_threshold_mask = supported.detach().cpu() & (score_cpu >= beta * gamma)
        seed_count = int(seed.sum().item())

        # Hysteresis requires both a nonempty bridge set and at least one seed
        if low_threshold_mask.sum().item() > 0 and seed_count > 0:

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

    # Count selected Gaussians
    count = final_mask.sum().item()
    print(f"Labeled {count} gaussians as {target_id}")

    # An empty selection is valid and is still saved as an empty PLY
    if count == 0:
        print("Warning: No Gaussians selected with this threshold, saving an empty PLY")

    # Store each target class in its own output directory
    os.makedirs(target_class_dir, exist_ok=True)

    # Select the Gaussian rows
    gaussians.set_mask_index(final_mask.nonzero(as_tuple=True)[0])
    fd, temporary_name = tempfile.mkstemp(
        dir=target_class_dir, suffix=".ply.tmp",
    )

    # Write the labeled model
    os.close(fd)
    try:
        gaussians.save_ply(temporary_name)
        os.replace(temporary_name, output_ply)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    print(f"Saved labeled PLY to {output_ply}")


def apply_threshold_batch(args):
    """ Process every class, beta and gamma in this container invocation """

    # Load shared model and graph data
    gaussians = _load_gaussians(args)
    xyz = gaussians.get_xyz
    if torch.is_tensor(xyz):
        xyz = xyz.detach().cpu().numpy()
    graph = _hysteresis_graph(args, xyz)
    voting_paths = list(args.voting_data_path)
    target_classes = list(args.target_class)

    # Prepare batch threshold settings
    output_roots = list(args.class_output_dir)
    betas = list(args.beta)
    gammas = list(args.hysteresis_gamma)

    # Process every class and threshold combination
    for index, voting_path in enumerate(voting_paths):
        voting_data = torch.load(voting_path, map_location=args.device)
        target_class = target_classes[index]
        output_root = output_roots[index]
        for gamma in gammas:
            output_dir = os.path.join(
                output_root,
                f"g{str(gamma).replace('.', '_')}"

    # Add the hysteresis radius to the path
                f"_r{str(args.hysteresis_radius).replace('.', '_')}",
            )
            for beta in betas:
                apply_threshold(
                    args, gaussians=gaussians, voting_data=voting_data,
                    hysteresis_graph=graph, beta=beta,
                    hysteresis_gamma=gamma, target_class=target_class,
                    class_output_dir=output_dir)

if __name__ == "__main__":
    parser = ArgumentParser()

    # Model and target configuration
    parser.add_argument("--model_path", required=True, help="Path to trained 3DGS model output")
    parser.add_argument("--sh_degree", type=int, default=3, help="SH degree")
    parser.add_argument("--loaded_iter", type=int, default=30000, help="Iteration of model to load")
    parser.add_argument(
        "--class_spec", action="append", required=True,
        help="JSON object with target_class, voting_data_path and class_output_dir",
    )

    # Input and output paths
    parser.add_argument("--output_dir", required=True, help="Directory to save labeled PLY")
    parser.add_argument("--cache_dir", default=None, help="Directory for derived hysteresis graph cache")

    # Target selection
    parser.add_argument("--beta", nargs="+", type=float, default=[0.5], help="Minimum target evidence ratio(s) in [0, 1]")

    # Device configuration
    parser.add_argument("--device", type=str, default="cuda", help="Device, either cuda or cpu")
    
    # Hysteresis expansion
    parser.add_argument("--hysteresis_gamma", nargs="+", type=float, default=[0.8], help="Low-threshold factor(s). 0 disables hysteresis")
    parser.add_argument("--hysteresis_radius", type=float, default=0.05, help="Connectivity radius in meters for the bridge set")
    parser.add_argument("--force", action="store_true", help="Replace existing threshold outputs")

    args = parser.parse_args()
    if any(gamma < 0.0 for gamma in args.hysteresis_gamma):
        raise ValueError("--hysteresis_gamma needs to be non-negative")
    
    if args.hysteresis_radius <= 0.0:
        raise ValueError("--hysteresis_radius needs to be greater than zero")
    
    specs = [json.loads(value) for value in args.class_spec]
    required = {"target_class", "voting_data_path", "class_output_dir"}

    if any(set(spec) != required for spec in specs):
        raise ValueError("each --class_spec must contain exactly the three class fields")
    args.target_class = [spec["target_class"] for spec in specs]
    args.voting_data_path = [spec["voting_data_path"] for spec in specs]
    args.class_output_dir = [spec["class_output_dir"] for spec in specs]
    
    with torch.no_grad():
        apply_threshold_batch(args)