# Run the validation, held-out test or contribution analysis experiment

import argparse
from pathlib import Path

from evaluation.analytics import BASELINE_PREFIX, SCANNET_TEST_PREFIX, SCANNET_VALIDATION_PREFIX, TRANSFER_PREFIX
from evaluation.scripts.experiment_common import (
    BETAS, DEVELOPMENT_SCENES, GAMMAS, PER_VIEW_BETAS, dump_plan, load_json, run_units, token, unit,
)

# The baseline thresholds the evidence per view and transfers with the nearest Gaussian, and the rest of
# the configuration is the one of the method
BASELINE_EXTRA = ("--threshold-score", "per_view", "--gaussian-to-mesh-transfer", "nearest_neighbor_label")


def transfer_extra(transfer):
    """ Arguments of evaluation.run for a transfer operator, none for the default radius vote """
    return () if transfer == "radius_vote" else ("--gaussian-to-mesh-transfer", transfer)


ROWS = (
    ("no_competition_gtom", ("--no-gaussian-to-mesh-background-competes",)),
    ("no_competition_mtog", ("--no-mesh-to-gaussian-background-competes",)),
    (
        "no_competition_both",
        (
            "--no-gaussian-to-mesh-background-competes",
            "--no-mesh-to-gaussian-background-competes",
        ),
    ),
    ("nearest", ("--gaussian-to-mesh-transfer", "nearest_neighbor_label")),
    ("no_opacity", ("--no-opacity-weighting",)),
    ("all_views", ("--background-view-policy", "all_views")),
)

EXPERIMENTS = {
    "validation": {"dataset": "replica", "split": "validation", "count": 7},
    "test": {"dataset": "scannetpp", "split": "test", "count": 10},
    "contribution_analysis": {"dataset": "replica", "split": "validation", "count": 7},
    "baseline_validation": {"dataset": "replica", "split": "validation", "count": 7},
    "baseline_test": {"dataset": "scannetpp", "split": "test", "count": 10},
    # The selection inside ScanNet++, only for the TFG: the validation grid on the ScanNet++ scenes that are
    # neither test nor development, and the test scenes at the point that the same rule selects there
    "scannet_validation": {"dataset": "scannetpp", "split": "validation", "count": 37},
    "scannet_test": {"dataset": "scannetpp", "split": "test", "count": 10},
}


def units(args, selection):
    """
    Build the execution units for the selected experiment

    The validation sweep runs the full beta and gamma grid, while test and
    contribution analysis evaluate only the selected operating point
    """
    settings = EXPERIMENTS[args.experiment]
    common = {
        "data_root": args.data_root, "output_root": args.output_root,
        "tau": selection["tau_star"], "theta": selection["theta_star"],
        "split": settings["split"],
    }

    # The validation runs the whole grid with the operator asked for, so the selection can compare operators.
    # Each operator writes under its own variant, so the runs of one never replace those of the other.
    # The ScanNet++ validation runs the grid of both operators in the same job, the radius vote first, because
    # the files of its scene are removed once the job has finished. Its runs are written under the prefixes
    # that the test summaries do not read
    if args.experiment in ("validation", "scannet_validation"):
        if args.experiment == "validation":
            prefixes, transfers = TRANSFER_PREFIX, [args.transfer]
        else:
            prefixes, transfers = SCANNET_VALIDATION_PREFIX, list(SCANNET_VALIDATION_PREFIX)
        return [
            unit(settings["dataset"], scene, f"{prefixes[transfer]}{token(gamma)}", betas=BETAS, gamma=gamma,
                 extra=transfer_extra(transfer), **common)
            for scene in args.scene
            for transfer in transfers
            for gamma in GAMMAS
        ]

    # The baseline goes through the same grid of gamma, with its own thresholds of the evidence per view,
    # and its test runs the point that the same rule selects for it on validation
    if args.experiment == "baseline_validation":
        return [
            unit("replica", scene, f"{BASELINE_PREFIX}{token(gamma)}", betas=PER_VIEW_BETAS, gamma=gamma,
                 extra=BASELINE_EXTRA, **common)
            for scene in args.scene
            for gamma in GAMMAS
        ]
    if args.experiment == "baseline_test":
        return [
            unit("scannetpp", scene, f"{BASELINE_PREFIX}{token(selection['gamma_star'])}",
                 betas=[selection["beta_star"]], gamma=selection["gamma_star"], extra=BASELINE_EXTRA, **common)
            for scene in args.scene
        ]

    # The test runs the operator that the validation selected, the radius vote in older selections. With the
    # point selected on ScanNet++, it writes under its own prefix, even when the point is the same one
    beta_star, gamma_star = [selection["beta_star"]], selection["gamma_star"]
    if args.experiment in ("test", "scannet_test"):
        transfer = selection.get("transfer", "radius_vote")
        prefixes = TRANSFER_PREFIX if args.experiment == "test" else SCANNET_TEST_PREFIX
        return [
            unit("scannetpp", scene, f"{prefixes[transfer]}{token(gamma_star)}", betas=beta_star,
                 gamma=gamma_star, extra=transfer_extra(transfer), **common)
            for scene in args.scene
        ]

    # The contribution analysis varies one factor at a time from the radius vote configuration
    return [
        unit("replica", scene, f"contribution_analysis_{name}", betas=beta_star, gamma=gamma_star,
             extra=extra, **common)
        for name, extra in ROWS
        for scene in args.scene
    ]


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)

    # Identify the experiment and the scenes that will be evaluated
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--scene", action="append", required=True)

    # Define the paths used by the launcher and by the container mounts
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path,
                        default=Path(__file__).resolve().parents[2])

    # Read the frozen configuration
    parser.add_argument("--selection", type=Path, required=True,
                        help="JSON with tau_star and theta_star from the development sweep for validation, "
                             "and the validation selection with beta_star and gamma_star too for test and contribution analysis")
    parser.add_argument("--transfer", choices=TRANSFER_PREFIX, default="radius_vote",
                        help="Transfer operator from Gaussians to mesh of the Replica validation grid; the ScanNet++ "
                             "validation runs both, and the tests take the one in the selection file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the unit plan without running anything")
    parser.add_argument("--only-scene", default=None,
                        help="Run only the units of this scene, so each scene can run in its own cluster job; "
                             "the full --scene list is still checked")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    settings = EXPERIMENTS[args.experiment]

    # Development scenes are never evaluated here, and every split has a fixed number of scenes
    if DEVELOPMENT_SCENES & set(args.scene):
        raise SystemExit(f"{args.experiment} scenes must exclude the development scenes")
    if len(set(args.scene)) != settings["count"]:
        raise SystemExit(f"{args.experiment} requires exactly {settings['count']} scenes")
    if args.only_scene is not None and args.only_scene not in args.scene:
        raise SystemExit("--only-scene must be one of the --scene values")

    planned = units(args, load_json(args.selection))
    if args.only_scene is not None:
        planned = [item for item in planned if item["scene"] == args.only_scene]
    if args.dry_run:
        dump_plan(args.experiment, planned)
        return 0

    run_units(planned, args.repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
