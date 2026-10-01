# Run the development sweep that selects tau and theta on the two development scenes

import argparse
import json
from pathlib import Path

from evaluation.analytics import close, load_analytics, number
from evaluation.scripts.experiment_common import dump_plan, run_units, token, unit


DEFAULT_GAMMA = 0.8
DEFAULT_BETA = 0.975
TOLERANCE = 0.01
RULE = "within 0.01 of best mean, then smallest scene_difference, then smallest parameter"


def _units(args, phase, tau_star=None):
    """
    Build the tau or theta sweep units for both development scenes

    The two development scenes belong to different datasets, and each dataset
    resolves its scenes under its own root. The output root is derived from the
    data root because evaluation.run requires the output to live inside it.
    """
    units = []
    for dataset, scene, data_root in (
        ("replica", args.replica_scene, args.replica_data_root),
        ("scannetpp", args.scannetpp_scene, args.scannetpp_data_root),
    ):
        # The theta variant label records the tau it depends on
        if phase == "tau":
            candidates = [(f"development_tau_{token(tau)}", tau, args.default_theta) for tau in args.tau_grid]
        else:
            candidates = [(f"development_theta_tau{token(tau_star)}_theta{token(theta)}", tau_star, theta)
                          for theta in args.theta_grid]
        for variant, tau, theta in candidates:
            units.append(unit(
                dataset, scene, variant, data_root, data_root / args.output_subdir,
                betas=[args.beta], gamma=args.gamma, tau=tau, theta=theta, mask_source="gt2d",
            ))
    return units


def select_candidate(view, prefix, parameter, beta):
    """
    Apply a candidate selection rule

    Every candidate is scored by the mean mIoU of the two development scenes, and among those
    within the tolerance of the best mean, the one where the two scenes differ least is picked
    """

    # The run parameters say which tau and theta each development run used
    parameters = {row["run_id"]: row for row in view["run_parameters"]}

    # Aggregate one candidate value across both development scenes
    grouped = {}
    for row in view["aggregate_beta_metrics"]:
        if not row["variant"].startswith(prefix) or row["source"] != "gt2d":
            continue
        if not close(number(row["beta"]), beta):
            continue
        value = float(parameters[row["run_id"]][parameter])
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
    eligible = [item for item in summaries if item["mean_mIoU"] >= best - TOLERANCE]
    return min(eligible, key=lambda item: (item["scene_difference"], item[parameter]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--replica-data-root", type=Path, required=True,
                        help="Replica root, holding one directory per scene")
    parser.add_argument("--scannetpp-data-root", type=Path, required=True,
                        help="ScanNet++ root, holding validation_data and metadata")
    parser.add_argument("--output-subdir", default="eval",
                        help="Run directory created inside each data root")
    parser.add_argument("--selection-output", type=Path, default=None)
    parser.add_argument("--replica-scene", default="office_0")
    parser.add_argument("--scannetpp-scene", required=True)
    parser.add_argument("--tau-grid", nargs="+", type=float, required=True)
    parser.add_argument("--theta-grid", nargs="+", type=float, required=True)
    parser.add_argument("--default-theta", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the tau phase plan, the theta phase depends on the selected tau")
    parser.add_argument("--only-scene", default=None,
                        help="Run only the tau phase of this development scene, so both scenes train and vote "
                             "in parallel jobs; a later run without it finishes the sweep")
    args = parser.parse_args(argv)
    if args.only_scene not in (None, args.replica_scene, args.scannetpp_scene):
        raise SystemExit("--only-scene must be one of the development scenes")
    args.replica_data_root = args.replica_data_root.resolve()
    args.scannetpp_data_root = args.scannetpp_data_root.resolve()

    # Both datasets write their metrics into the same store, which is what lets
    # the rule below compare the two development scenes
    analytics = args.replica_data_root.parent / "analytics"
    if args.scannetpp_data_root.parent / "analytics" != analytics:
        raise SystemExit("the two data roots must share a parent, which holds the analytics store")
    selection_path = args.selection_output or analytics / "tau_theta_selection.json"

    # Development selection uses annotation-derived masks to isolate transfer
    tau_units = _units(args, "tau")
    if args.only_scene is not None:
        tau_units = [item for item in tau_units if item["scene"] == args.only_scene]
    if args.dry_run:
        dump_plan("development_tau", tau_units)
        return 0

    # Run the tau phase and select tau before running the theta phase that depends on it
    run_units(tau_units, args.repo_root)
    if args.only_scene is not None:
        return 0
    tau_selection = select_candidate(load_analytics(analytics), "development_tau_", "tau", args.beta)

    run_units(_units(args, "theta", tau_selection["tau"]), args.repo_root)
    theta_selection = select_candidate(
        load_analytics(analytics), f"development_theta_tau{token(tau_selection['tau'])}_",
        "min_fraction", args.beta,
    )

    selection = {
        "tau_star": tau_selection["tau"],
        "theta_star": theta_selection["min_fraction"],
        "tau_selection": tau_selection,
        "theta_selection": theta_selection,
        "rule": RULE,
    }
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selection, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
