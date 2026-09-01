# Generate appendix table bodies from analytics

import argparse
import json
from collections import defaultdict
from pathlib import Path

from evaluation.analytics import deduplicate_analytics
from evaluation.common import atomic_write_text

DATASETS = {"replica": "Replica", "scannetpp": "ScanNet++"}

# The manuscript names the two conditions after their masks.
SOURCES = (("gt2d", "annotation-derived"), ("yolo", "YOLO"))


def number(value):
    # Convert optional numeric text
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values):
    # Average available values
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def cell(value, digits=2):
    # Format one table cell
    return "--" if value is None else f"{value:.{digits}f}"


def strip_last(lines):
    # Remove the final table separator
    lines = list(lines)
    while lines and lines[-1] == "\\addlinespace[3pt]":
        lines.pop()
    if not lines:
        return "\n"
    text = "\n".join(lines)
    return text[: text.rindex("\\\\")] + "\n" if "\\\\" in text else text + "\n"


def selected_operating_point(path, tolerance=1e-9):
    # Load the selected validation point
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        return float(data["beta_star"]), float(data["gamma_star"]), tolerance
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "selection JSON must contain beta_star and gamma_star"
        ) from error


def _filtered(view, runs, beta, gamma, tolerance):
    # Yield completed rows at the requested point
    for row in view.get("class_beta_metrics", []):
        b, g = number(row.get("beta")), number(row.get("hysteresis_gamma"))
        if beta is not None and (b is None or abs(b - beta) > tolerance):
            continue
        if gamma is not None and (g is None or abs(g - gamma) > tolerance):
            continue
        run = runs.get(row.get("run_id"))
        if run and run.get("status") == "completed":
            # Select the table rows
            yield row, run


def per_class(out, view, beta=None, gamma=None, tolerance=1e-9):
    # Gather per class values
    runs = {r["run_id"]: r for r in view["runs"]}
    names = {r["class_id"]: r["class_name"] for r in view["classes"]}

    # Collect values by dataset, class and source
    values = defaultdict(list)
    refs = defaultdict(dict)
    scenes = defaultdict(set)
    for row, run in _filtered(view, runs, beta, gamma, tolerance):
        key = (run["dataset"], row["class_id"])
        values[key + (row["source"],)].append(number(row.get("iou")))
        refs[key][run["scene_id"]] = number(row.get("ground_truth_transfer_iou"))
        scenes[key].add(run["scene_id"])

    # Build the table header
    lines = []
    for dataset in ("replica", "scannetpp"):
        keys = sorted(k for k in scenes if k[0] == dataset)
        for position, key in enumerate(keys):
            lines.append(
                f"{DATASETS[dataset] if position == 0 else ''} & {names.get(key[1], key[1])} & "
                f"{cell(mean(values[key + ('gt2d',)]))} & {cell(mean(values[key + ('yolo',)]))} & "
                f"{cell(mean(refs[key].values()))} & {len(scenes[key])} \\\\"
            )
        if dataset == "replica" and keys:
            # Add table metrics
            lines.append("\\addlinespace[3pt]")
    atomic_write_text(out / "per_class.tex", strip_last(lines))
    return len(lines)


def quantiles(out, view):
    # Gather vote score quantiles
    runs = {r["run_id"]: r for r in view["runs"]}
    columns = [
        "target_score_p25",
        "target_score_median",
        "target_score_p75",
        "target_score_p95",
        "target_score_p99",
        "supported_fraction",
    ]
    gathered = defaultdict(lambda: defaultdict(list))
    for row in view["vote_statistics"]:
        run = runs.get(row.get("run_id"))
        if not run or run.get("status") != "completed":
            continue

        # Add aggregate values
        for column in columns:
            gathered[(run["dataset"], row["source"])][column].append(
                number(row.get(column))
            )
    lines = []

    for dataset in ("replica", "scannetpp"):
        for source, label in SOURCES:
            values = gathered.get((dataset, source))

            # Emit quantiles for available sources
            if values:
                lines.append(
                    f"{DATASETS[dataset] if source == 'gt2d' else ''} & {label} & {' & '.join(cell(mean(values[c]), 3) for c in columns)} \\\\"
                )
    atomic_write_text(out / "quantiles.tex", strip_last(lines))
    return len(lines)


def main():
    # Parse inputs and write each appendix table from the deduplicated analytics view
    p = argparse.ArgumentParser()
    p.add_argument("--analytics", type=Path, default=Path("analytics"))
    p.add_argument("--out", type=Path, default=Path("tables"))
    p.add_argument("--selection", type=Path, default=Path("selection.json"))
    args = p.parse_args()

    view = deduplicate_analytics(args.analytics)
    beta = gamma = None
    if args.selection.exists():
        beta, gamma, _ = selected_operating_point(args.selection)

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"per_class.tex: {per_class(args.out, view, beta, gamma)} rows")
    print(f"quantiles.tex: {quantiles(args.out, view)} rows")


if __name__ == "__main__":
    main()