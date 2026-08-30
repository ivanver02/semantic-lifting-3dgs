# Run the validation, held-out test or contribution analysis experiment

import argparse
import json
from pathlib import Path

from evaluation.scripts.experiment_common import BETAS, GAMMAS, command, dump_plan, run_units, token


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
    """ Build the execution units for the selected experiment """
    if args.experiment == "validation":
        return [
            {
                "dataset": "replica",
                "scene": scene,
                "variant": f"frozen_g{token(gamma)}",
                "gamma": gamma,
            }
            for scene in args.scene
            for gamma in GAMMAS
        ]

    if args.experiment == "test":
        return [
            {
                "dataset": "scannetpp",
                "scene": scene,
                "variant": f"frozen_g{token(args.gamma_star)}",
            }
            for scene in args.scene
        ]

    return [
        {
            "dataset": "replica",
            "scene": scene,
            "variant": f"contribution_analysis_{name}",
            "gamma": args.gamma_star,
            "extra": list(extra),
        }
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

    # Select the source of the 2D masks
    parser.add_argument("--mask-source", choices=["both"], default="both")
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
        return dump_plan(args, f"{args.experiment}_plan", planned) or 0

    # The validation sweep runs the full beta grid
    # Test and contribution analysis evaluate only the selected operating point
    betas = BETAS if args.experiment == "validation" else [args.beta_star]

    run_units(
        args,
        planned,
        lambda current, unit: command(
            current,
            unit,
            betas=betas,
            gamma=unit.get("gamma", args.gamma_star),
            split=settings["split"],
            extra=unit.get("extra", ()),
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
