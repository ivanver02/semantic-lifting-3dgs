# Generate the measured manuscript macros without changing it

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from evaluation.analytics import (
    QUANTILES, close, dataset_of, is_frozen, load_analytics, number, selected_operating_point,
)
from evaluation.common import atomic_write_text


NAMES = """betaStar gammaStar tauStar thetaStar
replicaGTmIoU replicaYOLOmIoU scannetGTmIoU scannetYOLOmIoU
replicaGTmIoUSd replicaGTPrec replicaGTRec
replicaYOLOmIoUSd replicaYOLOPrec replicaYOLORec
scannetGTmIoUSd scannetGTPrec scannetGTRec
scannetYOLOmIoUSd scannetYOLOPrec scannetYOLORec
replicaRef scannetRef
sdBetaNoHyst sdBetaHyst sdClassNoHyst sdClassHyst bestMean bestMeanSd selectedMean selectedSd eligibleCount quantileAtBeta
replicaDetectorGap replicaLiftingGap replicaRepGap scannetDetectorGap scannetLiftingGap scannetRepGap classIoUmin classIoUmax
replicaGTrel replicaYOLOrel scannetGTrel scannetYOLOrel
ablFrozenmIoU ablFrozenSd ablFrozenRef ablFrozenCount ablNoHystmIoU ablNoHystSd ablNoHystRef ablNoHystCount
ablNoGtoMmIoU ablNoGtoMSd ablNoGtoMRef ablNoGtoMCount ablNoMtoGmIoU ablNoMtoGSd ablNoMtoGRef ablNoMtoGCount
ablNoBothmIoU ablNoBothSd ablNoBothRef ablNoBothCount ablNearestmIoU ablNearestSd ablNearestRef ablNearestCount
ablNoOpacmIoU ablNoOpacSd ablNoOpacRef ablNoOpacCount ablAllViewsmIoU ablAllViewsSd ablAllViewsRef ablAllViewsCount
timeMasks timeVotes timeThreshold timeTransfer timeSweepWarm timeMissTotal
memMasks memVotes memThreshold memTransfer memSweepWarm hardwareDescription qualScene qualClass""".split()


DATASETS = (("replica", "replica"), ("scannetpp", "scannet"))
SOURCES = (("gt2d", "GT"), ("yolo", "YOLO"))

# Rows of the ablation table: the frozen configuration and hysteresis disabled
# reuse the validation runs, and the other rows are contribution analysis variants
ABLATIONS = (
    ("ablFrozen", "frozen_", None),
    ("ablNoHyst", "frozen_", 0.0),
    ("ablNoGtoM", "contribution_analysis_no_competition_gtom", None),
    ("ablNoMtoG", "contribution_analysis_no_competition_mtog", None),
    ("ablNoBoth", "contribution_analysis_no_competition_both", None),
    ("ablNearest", "contribution_analysis_nearest", None),
    ("ablNoOpac", "contribution_analysis_no_opacity", None),
    ("ablAllViews", "contribution_analysis_all_views", None),
)

# The stage groups of the cost table
COST_STAGES = {
    "Masks": lambda stage: stage in {"generate_gt_masks", "generate_yolo_masks"},
    "Votes": lambda stage: stage.endswith(":votes"),
    "Threshold": lambda stage: stage.endswith(":threshold_hysteresis"),
    "Transfer": lambda stage: (
        stage == "ground_truth_transfer" or stage.endswith(":evaluation_transfer")
    ),
}


def numbers(rows, field):
    # Keep incomplete metric cells out of aggregate statistics
    return [value for value in (number(row[field]) for row in rows) if value is not None]


def mean_of_spreads(groups):
    """ Mean of the population standard deviations of the groups with more than one value """
    spreads = [statistics.pstdev(values) for values in groups if len(values) > 1]
    return f"{statistics.mean(spreads):.2f}" if spreads else "--"


def at_point(row, beta, gamma):
    # Whether a metric row belongs to one operating point
    return close(number(row["beta"]), beta) and close(number(row["hysteresis_gamma"]), gamma)


def main_results(values, view, beta, gamma):
    """ Fill the main table, the relative mIoU and the three gaps from the frozen rows at the selected point """
    for dataset, dataset_prefix in DATASETS:
        miou = {}
        for source, source_prefix in SOURCES:
            rows = [
                row for row in view["aggregate_beta_metrics"]
                if is_frozen(row) and dataset_of(row) == dataset and row["source"] == source
                and at_point(row, beta, gamma)
            ]
            prefix = dataset_prefix + source_prefix
            if not rows:
                continue

            # Means and standard deviations are taken over scenes
            scene_miou = numbers(rows, "mIoU")
            miou[source_prefix] = statistics.mean(scene_miou)
            values[prefix + "mIoU"] = f"{miou[source_prefix]:.2f}"
            values[prefix + "mIoUSd"] = f"{statistics.pstdev(scene_miou):.2f}"
            values[prefix + "Prec"] = f"{statistics.mean(numbers(rows, 'macro_precision')):.2f}"
            values[prefix + "Rec"] = f"{statistics.mean(numbers(rows, 'macro_recall')):.2f}"

            # Reference-relative mIoU is a mean of ratios over the classes whose
            # reference has positive IoU, so it is read from the same pass that
            # produced it rather than recomputed from the class table.
            relative = numbers(rows, "relative_mIoU")
            if relative:
                values[prefix + "rel"] = f"{statistics.mean(relative):.3f}"

            # The reference only depends on the mesh annotation, so it is read under annotation-derived masks
            if source == "gt2d":
                miou["R"] = statistics.mean(numbers(rows, "ground_truth_transfer_mIoU"))
                values[dataset_prefix + "Ref"] = f"{miou['R']:.2f}"

        # Derive the three gaps from the means already computed
        if "GT" in miou and "YOLO" in miou:
            values[dataset_prefix + "DetectorGap"] = f"{miou['GT'] - miou['YOLO']:.2f}"
        if "R" in miou:
            values[dataset_prefix + "LiftingGap"] = f"{miou['R'] - miou['GT']:.2f}"
            values[dataset_prefix + "RepGap"] = f"{1.0 - miou['R']:.2f}"


def stability(values, view, beta, gamma):
    """ Fill the two dispersions, the per-class range and the quantile at beta, on the validation split """
    curves, at_selected = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    per_class = defaultdict(list)
    for row in view["class_beta_metrics"]:
        if not (is_frozen(row) and dataset_of(row) == "replica" and row["source"] == "gt2d"):
            continue
        iou, row_beta, row_gamma = number(row["iou"]), number(row["beta"]), number(row["hysteresis_gamma"])
        for setting, setting_gamma in (("Hyst", gamma), ("NoHyst", 0.0)):
            if close(row_gamma, setting_gamma):
                # For a fixed class, across the beta grid, and between classes at the selected beta
                curves[setting][(row["scene_id"], row["class_id"])].append(iou)
                if close(row_beta, beta):
                    at_selected[setting][row["scene_id"]].append(iou)
                    if setting == "Hyst":
                        per_class[row["class_id"]].append(iou)

    for setting in ("Hyst", "NoHyst"):
        values["sdBeta" + setting] = mean_of_spreads(curves[setting].values())
        values["sdClass" + setting] = mean_of_spreads(at_selected[setting].values())

    means = [statistics.mean(observed) for observed in per_class.values()]
    if means:
        values["classIoUmin"] = f"{min(means):.2f}"
        values["classIoUmax"] = f"{max(means):.2f}"

    # The largest recorded quantile of the target evidence fraction that still lies below beta
    rows = [
        row for row in view["vote_statistics"]
        if is_frozen(row) and dataset_of(row) == "replica" and row["source"] == "gt2d"
    ]
    below = [
        100 * level for name, level in QUANTILES.items()
        if numbers(rows, f"target_score_{name}")
        and statistics.mean(numbers(rows, f"target_score_{name}")) <= beta
    ]
    if below:
        values["quantileAtBeta"] = f"P{max(below):g}"


def ablations(values, view, beta, gamma):
    """ Fill the eight rows of the ablation table on the validation split with annotation-derived masks """
    for prefix, variant, fixed_gamma in ABLATIONS:
        row_gamma = gamma if fixed_gamma is None else fixed_gamma

        def selected(row):
            return (row["variant"].startswith(variant) and dataset_of(row) == "replica"
                    and row["source"] == "gt2d" and at_point(row, beta, row_gamma))

        rows = [row for row in view["aggregate_beta_metrics"] if selected(row)]
        if not rows:
            continue
        miou = numbers(rows, "mIoU")
        values[prefix + "mIoU"] = f"{statistics.mean(miou):.2f}"
        values[prefix + "Sd"] = f"{statistics.pstdev(miou):.2f}"
        values[prefix + "Ref"] = f"{statistics.mean(numbers(rows, 'ground_truth_transfer_mIoU')):.2f}"

        # A scene selects the sum of the Gaussians of its classes, and the cell is the mean over scenes
        per_scene = defaultdict(float)
        for row in view["class_beta_metrics"]:
            if selected(row):
                per_scene[row["scene_id"]] += number(row["gaussian_count"])
        if per_scene:
            values[prefix + "Count"] = f"{int(round(statistics.mean(per_scene.values())))}"


def qualitative_pair(values, view, beta, gamma):
    """ Name the scene and class whose IoU is the median of the test split """
    names = {row["class_id"]: row["class_name"] for row in view["classes"]}
    pairs = sorted(
        (number(row["iou"]), row["scene_id"].split(":")[-1], names[row["class_id"]])
        for row in view["class_beta_metrics"]
        if is_frozen(row) and dataset_of(row) == "scannetpp" and row["source"] == "gt2d"
        and at_point(row, beta, gamma)
    )
    if pairs:
        _, values["qualScene"], values["qualClass"] = pairs[len(pairs) // 2]


def stage_costs(view, run_id):
    """ Sum elapsed time and take the peak memory per cost group for one run """
    times, memories = defaultdict(float), defaultdict(float)
    for row in view["run_stages"]:
        if row["run_id"] != run_id:
            continue
        for group, predicate in COST_STAGES.items():
            if predicate(row["stage"]):
                times[group] += number(row["elapsed_seconds"])
                memories[group] = max(memories[group], number(row["peak_cuda_memory_bytes"]) or 0.0)
    return times, memories


def costs(values, view, gamma):
    """
    Fill the cost table from one Replica run on a cache miss, and the sweep row
    from one Replica run at the selected gamma that read its votes from the cache

    A dash is kept for the stages that allocate no CUDA memory
    """
    vote_modes = defaultdict(set)
    for row in view["run_stages"]:
        if row["stage"].endswith(":votes") and row["dataset"] == "replica":
            vote_modes[row["run_id"]].add(row["cache_mode"])
    parameters = {row["run_id"]: row for row in view["run_parameters"]}
    runs = sorted(
        (run for run in view["runs"].values() if is_frozen(parameters[run["run_id"]])),
        key=lambda row: row["created_at"],
    )

    miss = next((run["run_id"] for run in runs if vote_modes.get(run["run_id"]) == {"miss"}), None)
    hit = next((run["run_id"] for run in runs if vote_modes.get(run["run_id"]) == {"hit"}
                and close(number(parameters[run["run_id"]]["hysteresis_gamma"]), gamma)), None)

    if miss is not None:
        times, memories = stage_costs(view, miss)
        for group, seconds in times.items():
            values["time" + group] = f"{seconds:.1f}~s"
            if memories[group] > 0:
                values["mem" + group] = f"{memories[group] / 1e9:.1f}~GB"
        values["timeMissTotal"] = f"{sum(times.values()):.1f}~s"

    if hit is not None:
        times, memories = stage_costs(view, hit)
        values["timeSweepWarm"] = f"{times['Threshold']:.1f}~s"
        if memories["Threshold"] > 0:
            values["memSweepWarm"] = f"{memories['Threshold'] / 1e9:.1f}~GB"


def hardware(values, view):
    """
    Describe the GPUs from the run metadata.

    collect_run_metadata stores gpu_name as a JSON list with one entry per
    visible device, so identical devices are collapsed into a count and the
    manuscript reads "2x NVIDIA ..." rather than a JSON array.
    """
    for row in view["run_parameters"]:
        names = json.loads(row["gpu_name"] or "[]")
        if names:
            values["hardwareDescription"] = ", ".join(
                f"{names.count(name)}x {name}" if names.count(name) > 1 else name
                for name in sorted(set(names))
            )
            return


def main(argv=None):
    # Parse inputs and load the selected operating point
    p = argparse.ArgumentParser()
    p.add_argument("--analytics", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("preprint/macros_measured.tex"))
    args = p.parse_args(argv)
    point = json.loads(args.selection.read_text(encoding="utf-8"))
    beta, gamma = selected_operating_point(args.selection)

    # The operating point and the numbers the selection rule produced beside it
    values = {name: "--" for name in NAMES}
    values.update(betaStar=f"{beta:g}", gammaStar=f"{gamma:g}")
    for macro, key in (("tauStar", "tau_star"), ("thetaStar", "theta_star")):
        if point.get(key) is not None:
            values[macro] = f"{float(point[key]):g}"
    for macro, key in (("bestMean", "best_mean"), ("bestMeanSd", "best_mean_sd"),
                       ("selectedMean", "selected_mean"), ("selectedSd", "selected_sd")):
        if point.get(key) is not None:
            values[macro] = f"{point[key]:.2f}"
    if point.get("eligible_count") is not None:
        values["eligibleCount"] = f"{point['eligible_count']}"

    # Aggregate the analytics rows into manuscript macro values
    view = load_analytics(args.analytics)
    main_results(values, view, beta, gamma)
    stability(values, view, beta, gamma)
    ablations(values, view, beta, gamma)
    qualitative_pair(values, view, beta, gamma)
    costs(values, view, gamma)
    hardware(values, view)

    # Render the macro file and report missing measurements
    args.output.parent.mkdir(parents=True, exist_ok=True)
    missing = [n for n in NAMES if values[n] == "--"]
    atomic_write_text(
        args.output,
        "% Generated by evaluation/scripts/make_macros.py.\n"
        "% Uses renewcommand because main.tex declares the same macro names\n"
        "% as placeholders before this file is input.\n"
        + "".join(f"\\renewcommand\\{name}{{{values[name]}}}\n" for name in NAMES),
    )
    print(f"unfilled macros ({len(missing)}/{len(NAMES)}): " + ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
