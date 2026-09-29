# Select a validation operating point

import argparse
import json
import statistics
from pathlib import Path

from evaluation.analytics import is_frozen, load_analytics, number

TOLERANCE = 0.01

FLOAT_SLACK = 1e-12


def summarise(rows, expected):
    """ Group the candidates by operating point and summarise them over scenes """
    grouped = {}
    for row in rows:
        beta, gamma = number(row["beta"]), number(row["hysteresis_gamma"])
        grouped.setdefault((beta, gamma), {})[row["scene_id"]] = number(row["mIoU"])

    incomplete = [key for key, values in grouped.items() if len(values) != expected]
    if not grouped or incomplete:
        raise RuntimeError(
            "selection requires one validation mIoU per candidate and scene: "
            f"{len(incomplete)} of {len(grouped)} candidates do not have "
            f"{expected} scenes"
        )

    # Summarize each candidate across scenes with the population standard deviation
    return [
        {
            "beta": beta,
            "gamma": gamma,
            "mean": statistics.mean(list(values.values())),
            "std": statistics.pstdev(list(values.values())),
        }
        for (beta, gamma), values in grouped.items()
    ]


def select(rows, expected, tolerance=TOLERANCE):
    """ Return the operating point and the numbers the rule produced with it """
    summaries = summarise(rows, expected)

    # The candidate with the best mean is a row of the manuscript table, so ties
    # are resolved with the same deterministic rule as the selection: smaller beta, then smaller gamma
    best = min(summaries, key=lambda item: (-item["mean"], item["beta"], item["gamma"]))

    # Apply the score margin and keep the candidate with the smallest scene standard deviation
    eligible = [
        item for item in summaries
        if item["mean"] >= best["mean"] - tolerance - FLOAT_SLACK
    ]
    selected = min(eligible, key=lambda item: (item["std"], item["beta"], item["gamma"]))
    return {
        "beta_star": selected["beta"],
        "gamma_star": selected["gamma"],
        "best_mean": best["mean"],
        "best_mean_sd": best["std"],
        "selected_mean": selected["mean"],
        "selected_sd": selected["std"],
        "eligible_count": len(eligible),
    }


def main(argv=None):
    # Parse inputs and filter analytics to the selected scenes
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analytics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument("--development-selection", type=Path, required=True,
                        help="JSON with tau_star and theta_star written by the development sweep")
    args = parser.parse_args(argv)

    # The rule is applied to the frozen validation rows with annotation-derived masks
    allowed = set(args.scene)
    rows = [
        row for row in load_analytics(args.analytics)["aggregate_beta_metrics"]
        if row["source"] == "gt2d"
        and is_frozen(row)
        and row["scene_id"].split(":")[-1] in allowed
    ]
    result = select(rows, len(allowed))
    result["scenes"] = sorted(allowed)

    # Record the transfer thresholds with the operating point, so one file holds the frozen configuration
    development = json.loads(args.development_selection.read_text(encoding="utf-8"))
    result["tau_star"] = development["tau_star"]
    result["theta_star"] = development["theta_star"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
