import os
import sys
from argparse import ArgumentParser
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.common import atomic_write, selection_path
from evaluation.transfer import load_gaussian_ply


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


def main(args):
    """ Process every class and beta in this container invocation """

    # Hysteresis needs the Gaussian centers of the full model
    ply_path = os.path.join(args.model_path, "point_cloud", f"iteration_{args.loaded_iter}", "point_cloud.ply")
    xyz, _ = load_gaussian_ply(ply_path)

    # Process every class and threshold combination
    for voting_path in args.votes:
        voting_data = torch.load(voting_path, map_location="cpu")
        score, supported = target_fraction(voting_data['target_weights'], voting_data['background_weights'])
        score, supported = score.numpy(), supported.numpy()
        print(f"{voting_path}: supported={int(supported.sum())}")

        for beta in args.beta:
            selected = hysteresis(xyz, score, supported, beta, args.hysteresis_gamma, args.hysteresis_radius)
            if not selected.any():
                # An empty selection is valid and is still saved as an empty file
                print(f"Warning: no Gaussians selected at beta={beta}")

            # Save the indices of the selected Gaussians next to the votes they come from
            output_path = selection_path(os.path.dirname(voting_path), args.hysteresis_gamma,
                                         args.hysteresis_radius, beta)
            atomic_write(output_path, lambda path: np.save(path, np.flatnonzero(selected)))
            print(f"Saved {int(selected.sum())} selected Gaussians to {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser()

    # Model and votes of every target class
    parser.add_argument("--model_path", required=True, help="Path to trained 3DGS model output")
    parser.add_argument("--loaded_iter", type=int, default=30000, help="Iteration of model to load")
    parser.add_argument("--votes", nargs="+", required=True, help="Voting data PT file of each class")

    # Threshold configuration
    parser.add_argument("--beta", nargs="+", type=float, default=[0.5], help="Minimum target evidence ratio(s) in [0, 1]")
    parser.add_argument("--hysteresis_gamma", type=float, default=0.8, help="Low-threshold factor in [0, 1). 0 disables hysteresis")
    parser.add_argument("--hysteresis_radius", type=float, default=0.05, help="Connectivity radius in meters for the bridge set")

    args = parser.parse_args()
    if any(not 0.0 <= beta <= 1.0 for beta in args.beta):
        raise ValueError("--beta values must be in [0, 1]")
    if not 0.0 <= args.hysteresis_gamma < 1.0:
        raise ValueError("--hysteresis_gamma must be in [0, 1)")
    if args.hysteresis_radius <= 0.0:
        raise ValueError("--hysteresis_radius needs to be greater than zero")

    main(args)
