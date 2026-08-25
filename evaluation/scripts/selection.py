# Select a validation operating point

import argparse
import json
import statistics
from pathlib import Path

from evaluation.analytics import deduplicate_analytics
from evaluation.common import atomic_write_text
from evaluation.scripts.make_tables import require_complete_state_store


def select(rows):
    # Group candidate operating points by scene metrics
    grouped = {}

    for row in rows:
        grouped.setdefault((float(row["beta"]), float(row["hysteresis_gamma"])), {})[
            row["scene_id"]] = float(row["mIoU"])
        
    if not grouped or any(len(values) != 7 for values in grouped.values()):
        raise RuntimeError("selection requires one validation mIoU per candidate and scene")
    
    # Summarize each candidate across scenes
    summaries = []
    for (beta, gamma), values in grouped.items():
        numbers = list(values.values())
        summaries.append({"beta": beta, "gamma": gamma,
                          "mean": statistics.mean(numbers),
                          "std": statistics.stdev(numbers)})
        
    # Apply the score margin and deterministic tie rule
    best = max(item["mean"] for item in summaries)
    eligible = [item for item in summaries if item["mean"] >= best - 0.01]
    selected = min(eligible, key=lambda item: (item["std"], item["beta"], item["gamma"]))
    return {"beta_star": selected["beta"], "gamma_star": selected["gamma"],
            "m_star": best, "best_scene_std": min(item["std"] for item in summaries
                                                     if abs(item["mean"] - best) <= 1e-12),
            "selected_mean": selected["mean"], "selected_scene_std": selected["std"],
            "eligible_count": len(eligible),

    # Load candidate records
            "tie_rule": "smaller beta, then smaller gamma after scene standard deviation"}


def main(argv=None):
    # Parse inputs and reject incomplete validation experiments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analytics", type=Path, required=True)
    parser.add_argument("--state-store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument("--source", default="gt2d")
    parser.add_argument("--experiment", default="validation")

    # Read optional transfer thresholds
    parser.add_argument("--tau-star", type=float)
    parser.add_argument("--theta-star", type=float)
    args = parser.parse_args(argv)
    try:
        require_complete_state_store(args.state_store, args.experiment)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    # Filter analytics to the selected scenes
    # Select only rows from the requested validation scenes and source
    view = deduplicate_analytics(args.analytics)
    allowed = set(args.scene)
    rows = [row for row in view["aggregate_beta_metrics"]
            if row.get("source") == args.source and
            row.get("scene_id", "").split(":")[-1] in allowed]
    result = select(rows)

    if args.tau_star is not None:
        result["tau_star"] = args.tau_star

    # Return selected records
    if args.theta_star is not None:
        result["theta_star"] = args.theta_star
        
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())