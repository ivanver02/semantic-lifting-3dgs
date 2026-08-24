# Run the validation, held-out test or contribution analysis experiment

import argparse
import json
from pathlib import Path

from evaluation.common import atomic_write_text
from evaluation.state_store import unit_id
from evaluation.scripts.experiment_common import (
    BETAS,
    GAMMAS,
    command,
    dump_plan,
    run_units,
    token,
)


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
                "gamma_invocations": list(GAMMAS),
                "miss": position == 0,
                "vote_accumulation": "one per class/source" if position == 0 else 0,
                "threshold_containers": 2,
                "unit_id": unit_id(
                    args.experiment, "replica", scene, f"frozen_g{token(gamma)}"
                ),
            }
            for scene in args.scene
            for position, gamma in enumerate(GAMMAS)
        ]
    
    if args.experiment == "test":
        return [
            {
                "dataset": "scannetpp",
                "scene": scene,
                "variant": f"frozen_g{token(args.gamma_star)}",
                "unit_id": unit_id(
                    args.experiment,
                    "scannetpp",
                    scene,
                    f"frozen_g{token(args.gamma_star)}",
                ),
            }
            for scene in args.scene
        ]
    
    return [
        {
            "dataset": "replica",
            "scene": scene,
            "variant": f"contribution_analysis_{name}",
            "unit_id": unit_id(args.experiment, "replica", scene, f"contribution_analysis_{name}"),
            "row": name,
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

    # Record experiment state and the analytics written by every unit
    parser.add_argument("--state-store", type=Path, required=True,
                        help="Experiment state store used for resumability")
    parser.add_argument("--analytics", type=Path, required=True)
    parser.add_argument("--reuse-manifest", type=Path,
                        help="Where the contribution analysis reuse rows are written")

    # Read the operating point selected on the validation scenes
    parser.add_argument("--selection", type=Path,
                        help="JSON holding beta_star, gamma_star, tau_star and theta_star")
    parser.add_argument("--tau-star", type=float)
    parser.add_argument("--theta-star", type=float)

    # Select the source of the 2D masks
    parser.add_argument("--mask-source", choices=["both"], default="both")

    # Resume a previous experiment instead of starting from an empty state store
    parser.add_argument("--retry-failed", action="store_true",
                        help="Retry state store units previously marked failed")
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
    
    args.mask_source = "gt2d" if args.experiment == "contribution_analysis" else args.mask_source
    planned = units(args)

    hits = []
    if args.experiment == "contribution_analysis":
        for scene in args.scene:
            frozen = f"frozen_g{token(args.gamma_star)}"
            no_hyst = f"frozen_g{token(GAMMAS[0])}"
            hits.extend(
                [
                    {
                        "row": "frozen",
                        "dataset": "replica",
                        "scene": scene,
                        "variant": frozen,
                        "unit_id": unit_id("validation", "replica", scene, frozen),
                        "hit_from": "validation",
                    },
                    {
                        "row": "no_hysteresis",
                        "dataset": "replica",
                        "scene": scene,
                        "variant": no_hyst,
                        "unit_id": unit_id("validation", "replica", scene, no_hyst),
                        "hit_from": "validation gamma=0",
                    },
                ]
            )

    if args.experiment == "validation":
        betas = BETAS
        notes = {
            "gamma_invocations_per_scene": 5,
            "vote_accumulation": "one per class/source on the miss invocation",
        }

    elif args.experiment == "test":
        betas = [args.beta_star]
        notes = {
            "operating_point": [args.beta_star, args.gamma_star],
            "tau_star": args.tau_star,
            "theta_star": args.theta_star,
        }

    else:
        betas = [args.beta_star]
        notes = {"hit_rows": hits, "all_views_is_the_only_new_vote_pass": True}

    if args.dry_run:
        return (
            dump_plan(
                args,
                f"{args.experiment}_plan",
                hits + planned,
                notes=notes,
                invocation_units=planned,
            )
            or 0
        )

    manifest_path = args.reuse_manifest or args.output_root / "contribution_analysis_reuse.json"
    if args.experiment == "contribution_analysis":
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            manifest_path,
            json.dumps(
                {
                    "experiment": args.experiment,
                    "source_experiment": "validation",
                    "rows": hits,
                    "rule": "Frozen and Hysteresis disabled are read from validation at beta_star",
                },indent=2, sort_keys=True) + "\n")
        
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