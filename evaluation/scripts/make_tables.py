# Generate the table bodies of the manuscripts from the analytics and the analyses

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

from evaluation.analytics import dataset_of, is_frozen, load_analytics, number
from evaluation.summaries import (
    DATASETS, SOURCES, TRANSFERS, class_summary, load_baseline, load_points, mask_rows, mean, ordered_classes,
    scene_summary, threshold_summary,
)
from evaluation.scripts.make_macros import ABLATIONS

ABLATION_LABELS = {
    "ablFrozen": "Radius vote at its selected point",
    "ablNoHyst": "Hysteresis disabled",
    "ablNoGtoM": "No competition, Gaussian to mesh",
    "ablNoMtoG": "No competition, mesh to Gaussian",
    "ablNoBoth": "No competition, both directions",
    "ablNoOpac": "Opacity weighting disabled",
    "ablAllViews": "Every matched camera contributes",
}


def cell(value, digits=2):
    return "--" if value is None else f"{value:.{digits}f}"


def pm(value, spread):
    """ A mean with its standard deviation in the small size of the manuscripts """
    return cell(value) if spread is None else f"{cell(value)} {{\\footnotesize$\\pm${cell(spread)}}}"


def table_body(groups):
    """
    Join the rows of every group into a LaTeX table body, with a small space between groups

    The manuscripts end the body with their own line break, so the last row has none
    """
    return " \\\\\n\\addlinespace[3pt]\n".join(" \\\\\n".join(rows) for rows in groups if rows) + "\n"


def per_class(view, selected):
    """ IoU per class at the selected point under both mask sources, its reference and the scenes that contain it """
    summary = class_summary(view, selected)
    groups = []
    for dataset in DATASETS:
        keys = [key for key in ordered_classes(summary) if key[0] == dataset]
        groups.append([
            f"{DATASETS[dataset] if position == 0 else ''} & {key[1]} & {cell(summary[key]['gt2d:iou'])} & "
            f"{cell(summary[key].get('yolo:iou'))} & {cell(summary[key]['reference'])} & {summary[key]['scenes']}"
            for position, key in enumerate(keys)
        ])
    return table_body(groups)


def per_scene(view, selected, radius):
    """ mIoU of every scene with both sources and its reference, for the selected operator and the radius vote """
    groups = []
    for dataset in DATASETS:
        columns = [scene_summary(view, point, dataset, source) for point in (selected, radius) for source in SOURCES]
        if any(column is None for column in columns):
            continue
        rows = []
        for position, scene in enumerate(sorted(columns[0]["scenes"])):
            # Annotation-derived and YOLO mIoU of each operator, followed by its reference
            cells = []
            for annotation, detector in (columns[:2], columns[2:]):
                item = annotation["scenes"].get(scene, {})
                cells += [cell(item.get("miou")), cell(detector["scenes"].get(scene, {}).get("miou")),
                          cell(item.get("reference"))]
            name = scene.replace("_", "\\_")
            rows.append(f"{DATASETS[dataset] if position == 0 else ''} & \\texttt{{{name}}} & " + " & ".join(cells))
        groups.append(rows)
    return table_body(groups)


def operators(view, selected, radius):
    """ Each transfer operator at its own point on validation and test """
    rows = []
    for point in (selected, radius):
        cells = [f"{TRANSFERS[point.transfer]} & $({point.beta:g},\\,{point.gamma:g})$"]
        for dataset in DATASETS:
            for source in SOURCES:
                summary = scene_summary(view, point, dataset, source)
                cells.append(pm(summary["miou"], summary["sd"]) if summary else "--")
            summary = scene_summary(view, point, dataset, "gt2d")
            cells.append(cell(summary["reference"]) if summary else "--")
        rows.append(" & ".join(cells))
    return table_body([rows])


def interval(summary):
    """ The bootstrap interval of a mean, as the manuscripts write it """
    return "--" if summary is None or summary["low"] is None else f"[{cell(summary['low'])}, {cell(summary['high'])}]"


def main_results(view, selected):
    """ The main table: mean and deviation over scenes, bootstrap interval, precision, recall and reference """
    groups = []
    for dataset in DATASETS:
        rows = []
        for source, label in SOURCES.items():
            summary = scene_summary(view, selected, dataset, source)
            if summary is None:
                continue
            rows.append(" & ".join([
                DATASETS[dataset] if not rows else "", "annotation" if source == "gt2d" else label,
                str(summary["count"]), pm(summary["miou"], summary["sd"]), interval(summary),
                cell(summary["precision"]), cell(summary["recall"]), cell(summary["reference"]),
                cell(summary["relative"], 3),
            ]))
        groups.append(rows)
    return table_body(groups)


def baseline(view, selected, analytics):
    """ The evidence per view baseline and the method, each at its selected point, on both splits """
    point, _ = load_baseline(analytics)
    if point is None:
        return None
    rows = []
    for label, current in (("Target evidence per view", point), ("Target evidence fraction", selected)):
        cells = [label, f"$({current.beta:g},\\,{current.gamma:g})$"]
        for dataset in DATASETS:
            for source in SOURCES:
                summary = scene_summary(view, current, dataset, source)
                cells.append(pm(summary["miou"], summary["sd"]) if summary else "--")
        rows.append(" & ".join(cells))
    return table_body([rows])


def ablations(view, radius):
    """ One row per factor with its paired difference to the radius vote, scene by scene """
    def scenes_of(variant, gamma):
        return {
            row["scene_id"]: row for row in view["aggregate_beta_metrics"]
            if row["variant"].startswith(variant) and dataset_of(row) == "replica" and row["source"] == "gt2d"
            and abs(number(row["beta"]) - radius.beta) < 1e-9 and abs(number(row["hysteresis_gamma"]) - gamma) < 1e-9
        }

    base = scenes_of("frozen_g", radius.gamma)
    rows = []
    for prefix, variant, fixed_gamma in ABLATIONS:
        if prefix not in ABLATION_LABELS:
            continue
        gamma = radius.gamma if fixed_gamma is None else fixed_gamma
        scenes = scenes_of(variant, gamma)
        if not scenes:
            continue
        miou = [number(row["mIoU"]) for row in scenes.values()]
        differences = [number(scenes[s]["mIoU"]) - number(base[s]["mIoU"]) for s in scenes if s in base]
        delta = "--" if prefix == "ablFrozen" else f"{statistics.mean(differences):+.3f}"
        rows.append(" & ".join([
            ABLATION_LABELS[prefix], pm(statistics.mean(miou), statistics.pstdev(miou)), delta,
            cell(mean(number(row["ground_truth_transfer_mIoU"]) for row in scenes.values())),
        ]))
    return table_body([rows[:1], rows[1:]])


def scores(path):
    """ The threshold analysis: mean best IoU per item and at one shared threshold, and the spread, per score """
    summary = threshold_summary(path)
    groups = []
    for dataset in DATASETS:
        rows = []
        for source, label in SOURCES.items():
            if (dataset, source, "fraction") not in summary:
                continue
            cells = [DATASETS[dataset] if not rows else "", label]
            for score in ("fraction", "evidence", "per_view"):
                item = summary[(dataset, source, score)]
                cells += [cell(item["oracle"]), cell(item["shared"]),
                          cell(item["spread"]) if score == "fraction" else f"{item['spread']:.1f}"]
            rows.append(" & ".join(cells))
        groups.append(rows)
    return table_body(groups)


def mask_agreement(path):
    """ 2D agreement of YOLO with the annotation masks per class, next to the 3D IoU with each source """
    rows = mask_rows(path)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["name"])].append(row)
    groups = []
    for dataset in DATASETS:
        keys = [key for key in ordered_classes(grouped) if key[0] == dataset]
        groups.append([
            f"{DATASETS[dataset] if position == 0 else ''} & {key[1]} & {len(grouped[key])} & "
            + " & ".join(cell(mean(row[field] for row in grouped[key])) for field in ("p2d", "r2d", "iou2d", "yolo3d", "gt3d"))
            for position, key in enumerate(keys)
        ])
    return table_body(groups)


def quantiles(view):
    """ Mean quantiles of the target evidence fraction and supported portion per dataset and source """
    columns = ["target_score_p25", "target_score_median", "target_score_p75",
               "target_score_p95", "target_score_p99", "supported_fraction"]
    groups = []
    for dataset in DATASETS:
        rows = []
        for source, label in SOURCES.items():
            # The votes do not depend on the transfer nor on gamma, so one frozen variant gives them all
            selected = [row for row in view["vote_statistics"]
                        if is_frozen(row, None) and dataset_of(row) == dataset and row["source"] == source]
            if selected:
                rows.append(f"{DATASETS[dataset] if not rows else ''} & {label} & "
                            + " & ".join(cell(mean(number(row[c]) for row in selected), 3) for c in columns))
        groups.append(rows)
    return table_body(groups)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--analytics", type=Path, required=True,
                   help="analytics directory, with selection_transfer.json and selection.json")
    p.add_argument("--analysis", type=Path, default=None,
                   help="directory with mask_agreement.csv and threshold_scores.csv, skipped when missing")
    p.add_argument("--out", type=Path, default=Path("tables"))
    args = p.parse_args(argv)

    view = load_analytics(args.analytics)
    selected, radius, _ = load_points(args.analytics)
    tables = {
        "main_results": main_results(view, selected),
        "per_class": per_class(view, selected),
        "per_scene": per_scene(view, selected, radius),
        "operators": operators(view, selected, radius),
        "ablations": ablations(view, radius),
        "quantiles": quantiles(view),
    }
    base = baseline(view, selected, args.analytics)
    if base is not None:
        tables["baseline"] = base
    if args.analysis is not None and (args.analysis / "threshold_scores.csv").exists():
        tables["scores"] = scores(args.analysis / "threshold_scores.csv")
    if args.analysis is not None and (args.analysis / "mask_agreement.csv").exists():
        tables["mask_agreement"] = mask_agreement(args.analysis / "mask_agreement.csv")

    args.out.mkdir(parents=True, exist_ok=True)
    for name, body in tables.items():
        (args.out / f"{name}.tex").write_text(body, encoding="utf-8")
    print(f"wrote {', '.join(sorted(tables))} to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
