# Run the development sweep

import argparse
import json
import re
from pathlib import Path

from evaluation.analytics import deduplicate_analytics
from evaluation.common import atomic_write_text
from evaluation.state_store import unit_id
from evaluation.scripts.experiment_common import (
    command,
    observed_vote_counts,
    run_units,
    sources,
    token,
)


DEFAULT_GAMMA = 0.8
DEFAULT_BETA = 0.975


def _units(args):
    units = []
    for dataset, scene in (
        ("replica", args.replica_scene),
        ("scannetpp", args.scannetpp_scene),
    ):
        
        # Build tau candidates for each scene
        for tau in args.tau_grid:
            variant = f"development_tau_{token(tau)}"
            units.append(
                {
                    "phase": "tau",
                    "dataset": dataset,
                    "scene": scene,
                    "tau": tau,
                    "theta": args.default_theta,
                    # Build the sweep settings
                    "variant": variant,
                }
            )

        for theta in args.theta_grid:
            tau_label = (
                token(args.tau_star) if args.tau_star is not None else "selected"
            )
            units.append(
                {
                    "phase": "theta",
                    "dataset": dataset,
                    "scene": scene,
                    "tau": None,
                    "theta": theta,
                    "variant": f"development_theta_tau{tau_label}_theta{token(theta)}",
                }
            )
    return units


def _plan(args, units):

    # Describe miss preparation work and hit stages for each unit
    mask_sources = sources(args.mask_source)
    plan = []
    miss_scenes = set()

    for unit in units:
        scene_key = (unit["dataset"], unit["scene"])
        miss = unit["phase"] == "tau" and scene_key not in miss_scenes

        if miss:
            # Add the model settings
            miss_scenes.add(scene_key)
        planned = {
            **unit,
            "unit_id": unit_id(
                args.experiment,
                unit["dataset"],
                unit["scene"],
                unit["variant"],
            ),
            "tau_dependency": (
                "selected tau_star from tau phase"
                if unit["phase"] == "theta"
                else "command-line candidate"
            ),
            "stages": {
                "dataset_preparation": 1 if miss else 0,
                "training": 1 if miss else 0,
                "mask_generation": {
                    source: 1 if miss else 0 for source in mask_sources
                },
                "ground_truth_build": 1,
                "hysteresis_graph_build": 1 if miss else 0,
                "threshold_containers": len(mask_sources) if miss else 0,
                "gamma_invocations": [args.gamma],
            },
            "vote_accumulation": "one per class/source on the miss unit" if miss else 0,
        }

        # Mark theta units until tau selection is available
        if unit["phase"] == "theta" and args.tau_star is None:
            planned["variant"] = None
            planned["unit_id"] = None
            planned["blocked_on"] = "tau_star"
        plan.append(planned)
    return plan


def select_candidate(rows, parameter):
    """Apply a candidate selection rule"""

    # Aggregate one candidate value across both development scenes
    grouped = {}
    for row in rows:
        value = float(row[parameter])
        grouped.setdefault(value, {})[row["scene_id"]] = float(row["mIoU"])

    if not grouped or any(len(values) != 2 for values in grouped.values()):
        raise RuntimeError(
            f"cannot select {parameter}: expected one metric per candidate and "
            "both development scenes"
           
        )

    # Record the sweep result
    summaries = []
    for value, scenes in grouped.items():
        values = list(scenes.values())
        summaries.append(
            {
                parameter: value,
                "mean_mIoU": sum(values) / len(values),
                "scene_difference": abs(values[0] - values[1]),
                # Add the experiment identifier
            }
        )

    best = max(item["mean_mIoU"] for item in summaries)
    eligible = [item for item in summaries if item["mean_mIoU"] >= best - 0.01]
    return min(eligible, key=lambda item: (item["scene_difference"], item[parameter]))


def _analytics_rows(args, prefix):

    # Filter completed analytics rows for one sweep phase
    view = deduplicate_analytics(args.analytics)
    rows = []
    for row in view.get("aggregate_beta_metrics", []):
        if not row.get("variant", "").startswith(prefix):
            continue
        if row.get("source") != "gt2d" or float(row["beta"]) != args.beta:
            continue
        row = dict(row)

        # Decode the sweep parameter from the variant
        if prefix == "development_tau_":
            parameter_text = row["variant"].removeprefix(prefix)
            row["tau"] = float(parameter_text.replace("_", "."))
        else:
            match = re.search(r"_theta([0-9_]+)$", row["variant"])
            if match is None:
                continue
            row["theta"] = float(match.group(1).replace("_", "."))

        # Build the dry run command
        rows.append(row)
    return rows


def main(argv=None):

    # Parse sweep inputs and restore a persisted selection when available
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state-store", type=Path, required=True)
    parser.add_argument("--analytics", type=Path, default=None)
    parser.add_argument("--selection-output", type=Path, default=None)

    # Record the completed run
    parser.add_argument("--replica-scene", default="office_0")
    parser.add_argument("--scannetpp-scene", required=True)
    parser.add_argument("--tau-grid", nargs="+", type=float, required=True)
    parser.add_argument("--theta-grid", nargs="+", type=float, required=True)
    parser.add_argument("--default-theta", type=float, default=0.5)

    # Add sweep operating point defaults
    parser.add_argument("--tau-star", type=float, default=None)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument(
        "--mask-source",
        choices=["gt2d", "yolo", "both"],
        default="gt2d",
        help="Development selection defaults to annotation-derived masks to isolate transfer",
    )
    parser.add_argument("--experiment", default="development")

    # Add the next sweep value
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Pass --retry-failed to run.py on resumed units",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.mask_source not in {"gt2d", "both"}:
        raise SystemExit("development selection requires annotation-derived gt2d masks")
    args.analytics = args.analytics or args.data_root.parent / "analytics"

    # Add the beta values
    args.analytics = Path(args.analytics)
    selection_path = (
        args.selection_output or args.output_root / "tau_theta_selection.json"
    )

    if args.dry_run and selection_path.exists():
        persisted = json.loads(selection_path.read_text(encoding="utf-8"))
        args.tau_star = float(persisted["tau_star"])

    # Build the current sweep plan
    sweep_units = _units(args)
    plan = _plan(args, sweep_units)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "experiment": args.experiment,
                    "driver": "development_sweep",
                    "units": plan,
                    "state_store_owner": "evaluation.run",
                    "state_store_reader": "read_state_store before every decision",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    # Run the tau phase before selecting the theta dependency
    def build_command(current, unit):
        return command(
            current,
            unit,
            betas=[current.beta],
            gamma=current.gamma,
            split="validation",
            tau=(unit["tau"] if unit["tau"] is not None else current.tau_star),
            theta=unit["theta"],
        )

    tau_units = [unit for unit in sweep_units if unit["phase"] == "tau"]
    run_units(args, tau_units, build_command)
    if selection_path.exists():
        persisted = json.loads(selection_path.read_text(encoding="utf-8"))
        args.tau_star = float(persisted["tau_star"])
        tau_selection = persisted.get("tau_selection", {"tau": args.tau_star})
    else:
        tau_rows = _analytics_rows(args, "development_tau_")

        # Add the transfer settings
        tau_selection = select_candidate(tau_rows, "tau")
        args.tau_star = tau_selection["tau"]

        # Persist the tau operating point
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            selection_path,
            json.dumps(
                {
                    "tau_star": args.tau_star,
                    "tau_selection": tau_selection,
                    "rule": "within 0.01 of best mean, then smallest scene_difference, then smallest parameter",
                },
                indent=2,
            )
            + "\n",
        )
        
    # Run the theta phase after tau selection
    theta_units = _units(args)
    theta_units = [unit for unit in theta_units if unit["phase"] == "theta"]
    run_units(args, theta_units, build_command)
    theta_rows = _analytics_rows(args, "development_theta_")
    theta_selection = select_candidate(theta_rows, "theta")

    selection = {
        "tau_star": tau_selection["tau"],
        "theta_star": theta_selection["theta"],
        "tau_selection": tau_selection,
        "theta_selection": theta_selection,
        "rule": "within 0.01 of best mean, then smallest scene_difference, then smallest parameter",
    }
    
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(selection_path, json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "observed_vote_containers": observed_vote_counts(
                    args.analytics,
                    sweep_units,
                )
            },
            sort_keys=True,
        )
    )

    # Return the sweep results
    return 0


if __name__ == "__main__":
    raise SystemExit(main())