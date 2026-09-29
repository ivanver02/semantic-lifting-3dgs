# Generate appendix table bodies from analytics

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

from evaluation.analytics import close, dataset_of, is_frozen, load_analytics, number, selected_operating_point
from evaluation.common import atomic_write_text

DATASETS = {"replica": "Replica", "scannetpp": "ScanNet++"}

# The manuscript names the two conditions after their masks.
SOURCES = (("gt2d", "annotation-derived"), ("yolo", "YOLO"))


def cell(values, digits=2):
    # Format the mean of the available values as one table cell
    values = [value for value in values if value is not None]
    return f"{statistics.mean(values):.{digits}f}" if values else "--"


def table_body(groups):
    """
    Join the rows of every dataset into a LaTeX table body, with a small space between datasets

    main.tex ends the body with its own line break, so the last row has none
    """
    return " \\\\\n\\addlinespace[3pt]\n".join(
        " \\\\\n".join(rows) for rows in groups if rows
    ) + "\n"


def per_class(view, beta, gamma):
    """ IoU per class at the selected point under both mask sources, its reference and the scenes that contain it """
    names = {row["class_id"]: row["class_name"] for row in view["classes"]}

    # Collect values by dataset, class and source
    values = defaultdict(list)
    references = defaultdict(dict)
    for row in view["class_beta_metrics"]:
        if not (is_frozen(row) and close(number(row["beta"]), beta)
                and close(number(row["hysteresis_gamma"]), gamma)):
            continue
        key = (dataset_of(row), row["class_id"])
        values[key + (row["source"],)].append(number(row["iou"]))

        # The reference only depends on the mesh annotation, so both sources share it per scene
        references[key][row["scene_id"]] = number(row["ground_truth_transfer_iou"])

    groups = []
    for dataset in DATASETS:
        keys = sorted(key for key in references if key[0] == dataset)
        groups.append([
            f"{DATASETS[dataset] if position == 0 else ''} & {names[key[1]]} & "
            f"{cell(values[key + ('gt2d',)])} & {cell(values[key + ('yolo',)])} & "
            f"{cell(references[key].values())} & {len(references[key])}"
            for position, key in enumerate(keys)
        ])
    return table_body(groups)


def quantiles(view):
    """ Mean quantiles of the target evidence fraction and supported portion per dataset and source """
    columns = [
        "target_score_p25",
        "target_score_median",
        "target_score_p75",
        "target_score_p95",
        "target_score_p99",
        "supported_fraction",
    ]
    groups = []
    for dataset in DATASETS:
        rows = []
        for source, label in SOURCES:
            selected = [
                row for row in view["vote_statistics"]
                if is_frozen(row) and dataset_of(row) == dataset and row["source"] == source
            ]
            if selected:
                rows.append(
                    f"{DATASETS[dataset] if not rows else ''} & {label} & "
                    + " & ".join(cell([number(row[column]) for row in selected], 3) for column in columns)
                )
        groups.append(rows)
    return table_body(groups)


def main():
    # Parse inputs and write each appendix table from the analytics view
    p = argparse.ArgumentParser()
    p.add_argument("--analytics", type=Path, default=Path("analytics"))
    p.add_argument("--out", type=Path, default=Path("tables"))
    p.add_argument("--selection", type=Path, default=Path("selection.json"))
    args = p.parse_args()

    view = load_analytics(args.analytics)
    beta, gamma = selected_operating_point(args.selection)

    args.out.mkdir(parents=True, exist_ok=True)
    atomic_write_text(args.out / "per_class.tex", per_class(view, beta, gamma))
    atomic_write_text(args.out / "quantiles.tex", quantiles(view))
    print(f"Wrote per_class.tex and quantiles.tex to {args.out}")


if __name__ == "__main__":
    main()
