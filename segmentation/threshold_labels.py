import torch
import os
import tempfile
import sys
import json
from argparse import ArgumentParser
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import GaussianModel


def _load_gaussians(args):
    """ Loads the Gaussian model from the specified path and iteration """

    print(f"Loading gaussian model from {args.model_path}")
    gaussians = GaussianModel(sh_degree=args.sh_degree, use_labels=True)
    ply_path = os.path.join(args.model_path, "point_cloud", f"iteration_{args.loaded_iter}", "point_cloud.ply")
    gaussians.load_ply(ply_path)
    return gaussians


def target_fraction(target_weights, background_weights):
    """
    Compute the target evidence fraction rho = E+ / (E+ + E-) of every Gaussian

    The fraction only exists on the support set, the Gaussians with some evidence,
    and it stays zero outside it. Returns the fractions and the support mask.
    """

    # Combine target and background evidence
    evidence = target_weights + background_weights
    score = torch.zeros_like(target_weights)
    supported = evidence > 0

    # Compute the target evidence fraction
    score[supported] = target_weights[supported] / evidence[supported]
    return score, supported


def hysteresis(xyz, score, supported, beta, gamma, radius):
    """
    Select the seeds, whose fraction reaches beta, and every Gaussian above gamma * beta
    that is connected to a seed through Gaussians closer than radius

    Returns a boolean mask over the Gaussians
    """

    # Keep supported Gaussians whose target evidence ratio reaches beta
    seeds = supported & (score >= beta)

    # At gamma = 0 the expansion is skipped, and hysteresis requires at least one seed
    if gamma == 0 or not seeds.any():
        return seeds

    # The low threshold defines candidate bridge Gaussians around the seeds
    candidates = np.flatnonzero(supported & (score >= beta * gamma))

    # Connect the candidates closer than the radius and group them into spatially connected components
    pairs = cKDTree(xyz[candidates]).query_pairs(radius, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(candidates), len(candidates)))
    component_labels = connected_components(graph, directed=False)[1]

    # Keep only components containing at least one seed
    keep_component = np.zeros(component_labels.max() + 1, dtype=bool)
    keep_component[component_labels[seeds[candidates]]] = True

    # Reconstruct the full Gaussian mask from the retained components
    selected = np.zeros_like(seeds)
    selected[candidates] = keep_component[component_labels]
    print(f"hysteresis: {int(seeds.sum())} seeds -> {int(selected.sum())} gaussians")
    return selected


def apply_threshold(args, gaussians, xyz, voting_data, beta, target_class, class_output_dir):
    """ Applies the threshold to the voting weights and saves the resulting segmented PLY file """

    # Skip existing outputs before loading votes or computing hysteresis
    safe_class_name = target_class.replace(" ", "_")
    filename = f"labeled_gaussians_{safe_class_name}_beta{str(beta).replace('.', '_')}.ply"
    output_ply = os.path.join(class_output_dir, filename)
    if os.path.exists(output_ply) and not getattr(args, "force", False):
        return

    # Each tensor contains one accumulated evidence value per Gaussian
    score, supported = target_fraction(voting_data['target_weights'].cpu(), voting_data['background_weights'].cpu())
    print(f"beta={beta:.3f} supported={int(supported.sum().item())}")

    # Hysteresis expands high confidence seeds through nearby lower score Gaussians
    final_mask = hysteresis(xyz, score.numpy(), supported.numpy(), beta,
                            args.hysteresis_gamma, args.hysteresis_radius)

    if not final_mask.any():
        # An empty selection is valid and is still saved as an empty PLY
        print(f"Warning: no Gaussians selected for {target_class}, saving an empty PLY")

    # Select the Gaussian rows and write the labeled model atomically
    os.makedirs(class_output_dir, exist_ok=True)
    gaussians.set_mask_index(torch.from_numpy(np.flatnonzero(final_mask)).to(gaussians.get_xyz.device))
    fd, temporary_name = tempfile.mkstemp(
        dir=class_output_dir, suffix=".ply.tmp",
    )
    os.close(fd)
    try:
        gaussians.save_ply(temporary_name)
        os.replace(temporary_name, output_ply)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    print(f"Saved labeled PLY to {output_ply}")


def apply_threshold_batch(args):
    """ Process every class and beta in this container invocation """

    # Load shared model data once, hysteresis needs the Gaussian centers
    gaussians = _load_gaussians(args)
    xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float64)

    voting_paths = list(args.voting_data_path)
    target_classes = list(args.target_class)
    output_roots = list(args.class_output_dir)
    betas = list(args.beta)

    # Process every class and threshold combination
    for index, voting_path in enumerate(voting_paths):
        voting_data = torch.load(voting_path, map_location=args.device)
        target_class = target_classes[index]
        output_root = output_roots[index]

        # The hysteresis radius is part of the directory name because the graph depends on it
        output_dir = os.path.join(
            output_root,
            f"g{str(args.hysteresis_gamma).replace('.', '_')}"
            f"_r{str(args.hysteresis_radius).replace('.', '_')}",
        )
        for beta in betas:
            apply_threshold(
                args, gaussians=gaussians, xyz=xyz, voting_data=voting_data, beta=beta,
                target_class=target_class, class_output_dir=output_dir)


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

    # Threshold configuration
    parser.add_argument("--beta", nargs="+", type=float, default=[0.5], help="Minimum target evidence ratio(s) in [0, 1]")
    parser.add_argument("--device", type=str, default="cuda", help="Device, either cuda or cpu")
    parser.add_argument("--hysteresis_gamma", type=float, default=0.8, help="Low-threshold factor in [0, 1). 0 disables hysteresis")
    parser.add_argument("--hysteresis_radius", type=float, default=0.05, help="Connectivity radius in meters for the bridge set")
    parser.add_argument("--force", action="store_true", help="Replace existing threshold outputs")

    args = parser.parse_args()
    if any(not 0.0 <= beta <= 1.0 for beta in args.beta):
        raise ValueError("--beta values must be in [0, 1]")
    if not 0.0 <= args.hysteresis_gamma < 1.0:
        raise ValueError("--hysteresis_gamma must be in [0, 1)")
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
