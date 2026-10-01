# Compare three scores for selecting the Gaussians of a class, at the level of the Gaussians and
# against the reference Gaussians that the mesh annotation gives: the target evidence fraction of
# the method, the raw target evidence, and the target evidence divided by the views with the class.
# For every class and scene it writes the IoU of each score along a grid of thresholds, from which
# the tables and figures take the best threshold of every item and the one shared by all of them.
# It reads the cached votes, so nothing is projected again. It runs inside lifting.sif through
# picasso/analysis.sbatch

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from evaluation.analytics import load_analytics, number, selected_operating_point
from evaluation.common import safe_name
from evaluation.scripts.experiment_common import token

GRIDS = {
    "fraction": np.unique(np.concatenate([np.linspace(0.05, 0.995, 96), [0.999]])),
    "evidence": 10.0 ** np.linspace(-6, 5, 221),
    "per_view": 10.0 ** np.linspace(-8, 3, 221),
}


def iou_curve(score, eligible, reference, grid):
    """ Gaussian-level IoU of {score >= t} against the reference set, for every t of the grid """
    selected_scores = np.sort(score[eligible])
    reference_scores = np.sort(score[eligible & reference])
    total_reference = int(reference.sum())
    tp = len(reference_scores) - np.searchsorted(reference_scores, grid, side="left")
    selected = len(selected_scores) - np.searchsorted(selected_scores, grid, side="left")
    union = selected + total_reference - tp
    return np.where(union > 0, tp / np.maximum(union, 1), 0.0)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="data root that holds analytics and both datasets")
    parser.add_argument("--selection", type=Path, default=None,
                        help="radius vote selection, analytics/selection.json by default")
    parser.add_argument("--output", type=Path, default=None,
                        help="CSV with the IoU of every scene, class, source, score and threshold")
    args = parser.parse_args(argv)
    output = args.output or args.data / "analysis" / "threshold_scores.csv"

    # The votes and the reference Gaussians do not depend on the transfer from Gaussians to mesh
    # nor on gamma, so the frozen radius vote runs at the gamma of their selection give every item
    view = load_analytics(args.data / "analytics")
    _, gamma = selected_operating_point(args.selection or args.data / "analytics" / "selection.json")
    classes = {row["class_id"]: row for row in view["classes"]}
    variant = f"frozen_g{token(gamma)}"

    # One item per scene, class and source, with the vote configuration it used
    items = {}
    for row in view["vote_statistics"]:
        if row["variant"] == variant:
            items[(row["scene_id"], row["class_id"], row["source"])] = (row["vote_id"], number(row["num_class_views"]))

    rows = []
    for (scene_id, class_id, source), (vote_id, views) in sorted(items.items()):
        dataset, scene = scene_id.split(":")
        run = args.data / dataset / "eval" / dataset / scene
        name = safe_name(classes[class_id]["detector_name"])
        votes_path = run / "segmentation" / source / name / vote_id / f"voting_data_{name}.pt"
        reference_path = run / "results" / variant / "reference" / f"{name}.npy"
        if not votes_path.exists() or not reference_path.exists():
            print(f"skip {scene_id} {name} {source}: missing {'votes' if not votes_path.exists() else 'reference'}")
            continue
        votes = torch.load(votes_path, map_location="cpu")
        target = votes["target_weights"].double().numpy()
        background = votes["background_weights"].double().numpy()
        reference = np.zeros(len(target), dtype=bool)
        reference[np.load(reference_path)] = True
        supported = (target + background) > 0

        scores = {
            "fraction": np.divide(target, target + background, out=np.zeros_like(target), where=supported),
            "evidence": target,
            "per_view": target / max(views or 1, 1),
        }
        for kind, score in scores.items():
            curve = iou_curve(score, supported, reference, GRIDS[kind])
            rows += [(dataset, scene, classes[class_id]["class_name"], source, kind, f"{t:.6g}", f"{v:.5f}")
                     for t, v in zip(GRIDS[kind], curve)]
        print(f"{scene_id} {name} {source}: {int(supported.sum())} supported, {int(reference.sum())} reference", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "scene", "class", "source", "score", "threshold", "iou"])
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
