# Run the validation, held-out test or contribution analysis experiment

import argparse
import json
from pathlib import Path

from evaluation.scripts.experiment_common import BETAS, GAMMAS, dump_plan, run_units, token, unit


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
}


def _validate_scenes(args, settings):
    if "office_0" in args.scene:
        raise SystemExit(
            f"{args.experiment} scenes must exclude development scene office_0"
        )

    if args.experiment == "validation" and args.dry_run and len(args.scene) == 1:
        return

    if len(args.scene) != settings["count"]:
        raise SystemExit(f"{args.experiment} requires exactly {settings['count']} scenes")


def _load_selection(args):
    if args.selection is None:
        raise SystemExit(f"{args.experiment} requires --selection")

    selected = json.loads(args.selection.read_text(encoding="utf-8"))
    args.tau_star = selected.get("tau_star", selected.get("tau"))
    args.theta_star = selected.get("theta_star", selected.get("theta"))

    if args.tau_star is None or args.theta_star is None:
        raise SystemExit("selection must include tau_star and theta_star")

    args.beta_star = float(selected["beta_star"])
    args.gamma_star = float(selected["gamma_star"])


def units(args):
    """
    Build the execution units for the selected experiment

    The validation sweep runs the full beta and gamma grid, while test and
    contribution analysis evaluate only the selected operating point
    """
    settings = EXPERIMENTS[args.experiment]
    common = {
        "data_root": args.data_root, "output_root": args.output_root,
        "tau": args.tau_star, "theta": args.theta_star,
        "split": settings["split"],
    }

    if args.experiment == "validation":
        return [
            unit("replica", scene, f"frozen_g{token(gamma)}", betas=BETAS, gamma=gamma, **common)
            for scene in args.scene
            for gamma in GAMMAS
        ]

    beta_star, gamma_star = [args.beta_star], args.gamma_star
    if args.experiment == "test":
        return [
            unit("scannetpp", scene, f"frozen_g{token(gamma_star)}", betas=beta_star, gamma=gamma_star, **common)
            for scene in args.scene
        ]

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

    # Define the paths used by the launcher and by the Docker mounts
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path,
                        default=Path(__file__).resolve().parents[2])

    # Read the operating point selected on the validation scenes
    parser.add_argument("--selection", type=Path,
                        help="JSON holding beta_star, gamma_star, tau_star and theta_star")
    parser.add_argument("--tau-star", type=float)
    parser.add_argument("--theta-star", type=float)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the unit plan without running anything")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    settings = EXPERIMENTS[args.experiment]
    _validate_scenes(args, settings)

    if args.experiment in {"test", "contribution_analysis"}:
        _load_selection(args)
    elif args.tau_star is None or args.theta_star is None:
        raise SystemExit("validation requires --tau-star and --theta-star")

    planned = units(args)
    if args.dry_run:
        dump_plan(args.experiment, planned)
        return 0

    run_units(planned, args.repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
