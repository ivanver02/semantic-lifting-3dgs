# Select a validation operating point

import argparse
import json
import statistics
from pathlib import Path

from evaluation.analytics import deduplicate_analytics
from evaluation.common import atomic_write_text

TOLERANCE = 0.01

FLOAT_SLACK = 1e-12


def number(value):
    # Convert an optional analytics cell
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_ablation(row):
    # Contribution analysis runs share the validation scenes
    return (row.get("variant") or "").startswith("contribution_analysis_")


def summarise(rows, expected):
    """ Group the candidates by operating point and summarise them over scenes """
    grouped = {}
    for row in rows:
        beta, gamma = number(row.get("beta")), number(row.get("hysteresis_gamma"))
        miou = number(row.get("mIoU"))
        if beta is None or gamma is None or miou is None:
            continue
        grouped.setdefault((beta, gamma), {})[row.get("scene_id")] = miou

    incomplete = [key for key, values in grouped.items() if len(values) != expected]
    if not grouped or incomplete:
        raise RuntimeError(
            "selection requires one validation mIoU per candidate and scene: "
            f"{len(incomplete)} of {len(grouped)} candidates do not have "
            f"{expected} scenes"
        )

    # Summarize each candidate across scenes
    summaries = []
    for (beta, gamma), values in grouped.items():
        numbers = list(values.values())
        summaries.append({
            "beta": beta,
            "gamma": gamma,
            "mean": statistics.mean(numbers),
            "std": statistics.pstdev(numbers),
        })
    return summaries


def select(rows, expected, tolerance=TOLERANCE):
    """ Return the operating point and the numbers the rule produced with it """
    summaries = summarise(rows, expected)

    # The candidate with the best mean is a row of the manuscript table
    best_mean = max(item["mean"] for item in summaries)
    best = min(
        (item for item in summaries if item["mean"] >= best_mean - FLOAT_SLACK),
        key=lambda item: (item["beta"], item["gamma"]),
    )

    # Apply the score margin and the deterministic tie rule
    eligible = [
        item for item in summaries
        if item["mean"] >= best_mean - tolerance - FLOAT_SLACK
    ]
    selected = min(eligible, key=lambda item: (item["std"], item["beta"], item["gamma"]))
    return {
        "beta_star": selected["beta"],
        "gamma_star": selected["gamma"],
        "best_mean": best["mean"],
        "best_mean_sd": best["std"],
        "best_beta": best["beta"],
        "best_gamma": best["gamma"],
        "selected_mean": selected["mean"],
        "selected_sd": selected["std"],
        "eligible_count": len(eligible),
        "candidate_count": len(summaries),
        "selected_is_best": (
            (selected["beta"], selected["gamma"]) == (best["beta"], best["gamma"])
        ),
        "tolerance": tolerance,
        "dispersion": "population standard deviation over scenes",

    # Record the rule alongside its result
        "tie_rule": "smaller beta, then smaller gamma after scene standard deviation",
    }


def main(argv=None):
    # Parse inputs and filter analytics to the selected scenes
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analytics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument("--source", default="gt2d")
    parser.add_argument("--experiment", default="validation")

    # Read optional transfer thresholds
    parser.add_argument("--tau-star", type=float)
    parser.add_argument("--theta-star", type=float)
    args = parser.parse_args(argv)

    # Select only completed rows from the requested validation scenes and source
    view = deduplicate_analytics(args.analytics)
    completed = {
        row.get("run_id")
        for row in view["runs"]
        if row.get("status") == "completed"
    }
    allowed = set(args.scene)
    rows = [
        row for row in view["aggregate_beta_metrics"]
        if row.get("source") == args.source
        and row.get("run_id") in completed
        and not is_ablation(row)
        and row.get("scene_id", "").split(":")[-1] in allowed
    ]
    result = select(rows, len(allowed))
    result["scenes"] = sorted(allowed)
    result["source"] = args.source

    if args.tau_star is not None:
        result["tau_star"] = args.tau_star

    # Return selected records
    if args.theta_star is not None:
        result["theta_star"] = args.theta_star

    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["selected_is_best"]:
        print(
            "note: the rule returned the candidate with the best mean, so the two "
            "rows of the operating-point table carry the same numbers"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())