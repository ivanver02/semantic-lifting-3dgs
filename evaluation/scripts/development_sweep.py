# Run the development sweep that selects tau and theta on the two development scenes

import argparse
import json
import re
from pathlib import Path

from evaluation.analytics import deduplicate_analytics
from evaluation.common import atomic_write_text
from evaluation.scripts.experiment_common import command, dump_plan, run_units, token


DEFAULT_GAMMA = 0.8
DEFAULT_BETA = 0.975


def _units(args):
    """ Build the tau and theta sweep units for both development scenes """
    units = []
    # The theta variant label records the tau it depends on
    tau_label = (
        token(args.tau_star) if args.tau_star is not None else "selected"
    )
    for dataset, scene in (
        ("replica", args.replica_scene),
        ("scannetpp", args.scannetpp_scene),
    ):
        for tau in args.tau_grid:
            units.append(
                {
                    "phase": "tau",
                    "dataset": dataset,
                    "scene": scene,
                    "tau": tau,
                    "theta": args.default_theta,
                    "variant": f"development_tau_{token(tau)}",
                }
            )

        for theta in args.theta_grid:
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

    summaries = []
    for value, scenes in grouped.items():
        values = list(scenes.values())
        summaries.append(
            {
                parameter: value,
                "mean_mIoU": sum(values) / len(values),
                "scene_difference": abs(values[0] - values[1]),
            }
        )

    best = max(item["mean_mIoU"] for item in summaries)
    eligible = [item for item in summaries if item["mean_mIoU"] >= best - 0.01]
    return min(eligible, key=lambda item: (item["scene_difference"], item[parameter]))


def _analytics_rows(args, prefix):

    # Filter completed analytics rows for one sweep phase and decode its parameter
    view = deduplicate_analytics(args.analytics)
    rows = []
    for row in view.get("aggregate_beta_metrics", []):
        if not row.get("variant", "").startswith(prefix):
            continue
        if row.get("source") != "gt2d" or float(row["beta"]) != args.beta:
            continue
        row = dict(row)

        # Decode the sweep parameter from the variant name
        if prefix == "development_tau_":
            parameter_text = row["variant"].removeprefix(prefix)
            row["tau"] = float(parameter_text.replace("_", "."))
        else:
            match = re.search(r"_theta([0-9_]+)$", row["variant"])
            if match is None:
                continue
            row["theta"] = float(match.group(1).replace("_", "."))
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--analytics", type=Path, default=None)
    parser.add_argument("--selection-output", type=Path, default=None)
    parser.add_argument("--replica-scene", default="office_0")
    parser.add_argument("--scannetpp-scene", required=True)
    parser.add_argument("--tau-grid", nargs="+", type=float, required=True)
    parser.add_argument("--theta-grid", nargs="+", type=float, required=True)
    parser.add_argument("--default-theta", type=float, default=0.5)
    parser.add_argument("--tau-star", type=float, default=None)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument(
        "--mask-source",
        choices=["gt2d", "yolo", "both"],
        default="gt2d",
        help="Development selection defaults to annotation-derived masks to isolate transfer",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    # This driver always plans the development experiment
    args.experiment = "development"

    if args.mask_source not in {"gt2d", "both"}:
        raise SystemExit("development selection requires annotation-derived gt2d masks")
    args.analytics = Path(args.analytics or args.data_root.parent / "analytics")
    selection_path = (
        args.selection_output or args.output_root / "tau_theta_selection.json"
    )

    # Restore a persisted tau so theta variants are labelled consistently
    if selection_path.exists():
        persisted = json.loads(selection_path.read_text(encoding="utf-8"))
        args.tau_star = float(persisted["tau_star"])
    sweep_units = _units(args)

    if args.dry_run:
        dump_plan(args, "development_sweep", sweep_units)
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

    # Select tau from analytics unless it was already persisted
    if selection_path.exists():
        persisted = json.loads(selection_path.read_text(encoding="utf-8"))
        args.tau_star = float(persisted["tau_star"])
        tau_selection = persisted.get("tau_selection", {"tau": args.tau_star})
    else:
        tau_rows = _analytics_rows(args, "development_tau_")
        tau_selection = select_candidate(tau_rows, "tau")
        args.tau_star = tau_selection["tau"]

        # Persist the tau operating point before running the theta phase
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

    # Relabel and run the theta phase after tau selection
    theta_units = [
        unit for unit in _units(args) if unit["phase"] == "theta"
    ]
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
