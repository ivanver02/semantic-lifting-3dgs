# Select a validation operating point

import argparse
import json
import statistics
from pathlib import Path

from evaluation.analytics import (
    BASELINE_PREFIX, SCANNET_VALIDATION_PREFIX, TRANSFER_PREFIX, is_frozen, load_analytics, number, transfer_of,
)

TOLERANCE = 0.01

FLOAT_SLACK = 1e-12

# Ties go to the operator listed first, the radius vote, as they go to the smaller beta and gamma
TRANSFER_ORDER = {transfer: index for index, transfer in enumerate(TRANSFER_PREFIX)}


def summarise(rows, expected):
    """ Group the candidates by transfer operator and operating point and summarise them over scenes """
    grouped = {}
    for row in rows:
        key = (transfer_of(row), number(row["beta"]), number(row["hysteresis_gamma"]))
        grouped.setdefault(key, {})[row["scene_id"]] = number(row["mIoU"])

    # With no rows at all the validation step has not run, which is worth saying apart from a partial sweep
    if not grouped:
        raise RuntimeError("selection found no frozen validation results in the analytics, run the validation step first")
    incomplete = [key for key, values in grouped.items() if len(values) != expected]
    if incomplete:
        raise RuntimeError(
            "selection requires one validation mIoU per candidate and scene: "
            f"{len(incomplete)} of {len(grouped)} candidates do not have "
            f"{expected} scenes"
        )

    # Summarize each candidate across scenes with the population standard deviation
    return [
        {
            "transfer": transfer,
            "beta": beta,
            "gamma": gamma,
            "mean": statistics.mean(list(values.values())),
            "std": statistics.pstdev(list(values.values())),
        }
        for (transfer, beta, gamma), values in grouped.items()
    ]


def select(rows, expected, tolerance=TOLERANCE):
    """ Return the operating point and the numbers the rule produced with it """
    summaries = summarise(rows, expected)

    def order(item):
        return TRANSFER_ORDER[item["transfer"]], item["beta"], item["gamma"]

    # The candidate with the best mean is a row of the manuscript table, so ties
    # are resolved with the same deterministic rule as the selection
    best = min(summaries, key=lambda item: (-item["mean"],) + order(item))

    # Apply the score margin and keep the candidate with the smallest scene standard deviation
    eligible = [
        item for item in summaries
        if item["mean"] >= best["mean"] - tolerance - FLOAT_SLACK
    ]
    selected = min(eligible, key=lambda item: (item["std"],) + order(item))

    # The best candidate of every operator, so the manuscript can compare them at their own best
    per_transfer = {}
    for item in summaries:
        current = per_transfer.get(item["transfer"])
        if current is None or (-item["mean"],) + order(item) < (-current["mean"],) + order(current):
            per_transfer[item["transfer"]] = item

    return {
        "transfer": selected["transfer"],
        "beta_star": selected["beta"],
        "gamma_star": selected["gamma"],
        "best_mean": best["mean"],
        "best_mean_sd": best["std"],
        "best_transfer": best["transfer"],
        "selected_mean": selected["mean"],
        "selected_sd": selected["std"],
        "eligible_count": len(eligible),
        "candidate_count": len(summaries),
        "best_per_transfer": {
            transfer: {"beta": item["beta"], "gamma": item["gamma"], "mean": item["mean"], "std": item["std"]}
            for transfer, item in per_transfer.items()
        },
    }


def main(argv=None):
    # Parse inputs and filter analytics to the selected scenes
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analytics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument("--development-selection", type=Path, required=True,
                        help="JSON with tau_star and theta_star written by the development sweep")
    parser.add_argument("--transfer", choices=TRANSFER_PREFIX, action="append", default=None,
                        help="Transfer operators the rule compares, every operator with validation results by default")
    parser.add_argument("--baseline", action="store_true",
                        help="Apply the rule to the evidence per view baseline instead of the method")
    parser.add_argument("--scannet", action="store_true",
                        help="Apply the rule to the grid of the ScanNet++ validation scenes instead of the Replica one")
    args = parser.parse_args(argv)

    def candidate(row):
        if args.baseline:
            return row["variant"].startswith(BASELINE_PREFIX)
        if args.scannet:
            return row["variant"].startswith(tuple(SCANNET_VALIDATION_PREFIX.values()))
        return is_frozen(row, transfer=None)

    def scene_of(row):
        return row["scene_id"].split(":")[-1]

    # The rule is applied to the validation rows with annotation-derived masks
    allowed = set(args.scene)
    rows = [
        row for row in load_analytics(args.analytics)["aggregate_beta_metrics"]
        if row["source"] == "gt2d" and candidate(row)
        and (args.transfer is None or transfer_of(row) in args.transfer)
        and scene_of(row) in allowed
    ]

    # A scene where no target class appears in the annotation masks gives no mIoU to any candidate. For this
    # reason, it is left out for all of them in the same way, and the selection records it
    empty = sorted(
        scene for scene in allowed
        if any(scene_of(row) == scene for row in rows)
        and all(number(row["mIoU"]) is None for row in rows if scene_of(row) == scene)
    )
    rows = [row for row in rows if scene_of(row) not in empty]
    result = select(rows, len(allowed) - len(empty))
    result["scenes"] = sorted(allowed - set(empty))
    result["scenes_without_classes"] = empty

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
