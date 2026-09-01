# Generate the measured manuscript macros without changing it

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from evaluation.analytics import deduplicate_analytics
from evaluation.common import atomic_write_text
from evaluation.scripts.make_tables import selected_operating_point

NAMES = """betaStar gammaStar tauStar thetaStar
replicaGTmIoU replicaYOLOmIoU scannetGTmIoU scannetYOLOmIoU
replicaGTmIoUSd replicaGTPrec replicaGTRec
replicaYOLOmIoUSd replicaYOLOPrec replicaYOLORec
scannetGTmIoUSd scannetGTPrec scannetGTRec
scannetYOLOmIoUSd scannetYOLOPrec scannetYOLORec
replicaRef scannetRef
sdBetaNoHyst sdBetaHyst sdClassNoHyst sdClassHyst bestMean bestMeanSd selectedMean selectedSd eligibleCount quantileAtBeta
    replicaDetectorGap replicaLiftingGap replicaRepGap scannetDetectorGap scannetLiftingGap scannetRepGap classIoUmin classIoUmax
replicaGTrel replicaYOLOrel scannetGTrel scannetYOLOrel replicaRelExcl scannetRelExcl
ablFrozenmIoU ablFrozenSd ablFrozenRef ablFrozenCount ablNoHystmIoU ablNoHystSd ablNoHystRef ablNoHystCount
ablNoGtoMmIoU ablNoGtoMSd ablNoGtoMRef ablNoGtoMCount ablNoMtoGmIoU ablNoMtoGSd ablNoMtoGRef ablNoMtoGCount
ablNoBothmIoU ablNoBothSd ablNoBothRef ablNoBothCount ablNearestmIoU ablNearestSd ablNearestRef ablNearestCount
ablNoOpacmIoU ablNoOpacSd ablNoOpacRef ablNoOpacCount ablAllViewsmIoU ablAllViewsSd ablAllViewsRef ablAllViewsCount
timeMasks timeVotes timeThreshold timeTransfer timeSweepWarm timeMissTotal timeMissSweepEquivalent
memMasks memVotes memThreshold memTransfer memSweepWarm hardwareDescription qualScene qualClass""".split()


REFERENCE_TOLERANCE = 0.005

# The eight ablation rows, in the order of the table
REUSED_ROWS = (("ablFrozen", None), ("ablNoHyst", 0.0))
ABLATION_VARIANTS = (
    ("ablNoGtoM", "contribution_analysis_no_competition_gtom"),
    ("ablNoMtoG", "contribution_analysis_no_competition_mtog"),
    ("ablNoBoth", "contribution_analysis_no_competition_both"),
    ("ablNearest", "contribution_analysis_nearest"),
    ("ablNoOpac", "contribution_analysis_no_opacity"),
    ("ablAllViews", "contribution_analysis_all_views"),
)

# Ablations and the stability diagnostics are read on the validation split under masks derived from the dataset
ABLATION_DATASET = "replica"
ABLATION_SOURCE = "gt2d"
TEST_DATASET = "scannetpp"

# The count the ablation table reports is the number of Gaussians the method selected
PREDICTED_SET = "predicted"

# The stage groups of the cost table
COST_STAGES = {
    "Masks": lambda stage: stage in {"generate_gt_masks", "generate_yolo_masks"},
    "Votes": lambda stage: stage.endswith(":votes"),
    "Threshold": lambda stage: stage.endswith(":threshold_hysteresis"),
    "Transfer": lambda stage: (
        stage in {"ground_truth_transfer"} or stage.endswith(":evaluation_transfer")
    ),
}


def num(value):
    # Convert optional analytics values
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def close(left, right, tolerance=1e-9):
    # Compare two optional operating-point coordinates
    return left is not None and right is not None and abs(left - right) <= tolerance


def selected_row(row, beta, gamma):
    # Ignore rows with incomplete operating-point metadata
    return close(num(row.get("beta")), beta) and close(
        num(row.get("hysteresis_gamma")), gamma
    )


def numeric_values(rows, field):
    # Keep incomplete metric cells out of aggregate statistics
    return [
        value for value in (num(row.get(field)) for row in rows) if value is not None
    ]


def spread(values):
    """ Population standard deviation, or None when there is nothing to spread """
    return statistics.pstdev(values) if len(values) > 1 else None


def collapse_over_sources(per_source, dataset, quantity, tolerance):
    """ Return the single value a mask-source-independent quantity must have """
    observed = [value for value in per_source.values() if value is not None]
    if not observed:
        return None
    if max(observed) - min(observed) > tolerance:
        detail = ", ".join(f"{source}={value:g}" for source, value in per_source.items())
        raise RuntimeError(
            f"{quantity} differs between mask sources on {dataset} ({detail}). "
            "It is computed from the mesh annotation alone, so a difference "
            "means the evaluated class set depends on the mask source."
        )
    return statistics.mean(observed)


def is_ablation(row):
    """ Whether a metric row belongs to a contribution analysis variant """
    return (row.get("variant") or "").startswith("contribution_analysis_")


def selected_counts(view, runs, beta):
    """
    Sum the selected Gaussians of each scene, per variant.

    gaussian_statistics holds one row per class, so a scene is the sum over its
    classes, and the table cell is then the mean over scenes.
    """
    per_scene = defaultdict(lambda: defaultdict(float))
    for row in view.get("gaussian_statistics", []):
        if row.get("set_type") != PREDICTED_SET or row.get("source") != ABLATION_SOURCE:
            continue
        run = runs.get(row.get("run_id"))
        if not run or run.get("dataset") != ABLATION_DATASET:
            continue
        if not close(num(row.get("beta")), beta):
            continue
        count = num(row.get("gaussian_count"))
        if count is not None:
            per_scene[row.get("variant") or ""][row.get("scene_id")] += count
    return per_scene


def cache_run_ids(view):
    """
    Return the run identified as a cache miss and the one identified as a hit.

    The vote container count is the signal that does not depend on cache_mode:
    a miss launches one vote container per class and source, a hit launches none
    because the vote artifact is already on disk. When cache_mode is populated
    it is preferred, since it states the condition directly.
    """
    completed = {
        row.get("run_id")
        for row in view.get("runs", [])
        if row.get("status") == "completed"
    }
    containers = defaultdict(int)
    modes = defaultdict(set)
    for row in view.get("run_stages", []):
        run_id = row.get("run_id")
        if run_id not in completed or not row.get("stage", "").endswith(":votes"):
            continue
        containers[run_id] += int(row.get("container_count") or 0)
        if row.get("cache_mode"):
            modes[run_id].add(row["cache_mode"])
    if not containers:
        return None, None
    miss = max(containers, key=lambda run_id: (containers[run_id], run_id or ""))
    hits = [
        run_id
        for run_id, count in containers.items()
        if modes.get(run_id) == {"hit"} or (count == 0 and run_id != miss)
    ]
    return miss, (min(hits, key=lambda run_id: run_id or "") if hits else None)


def stage_totals(view, run_id):
    """ Sum elapsed time and collect peak memory per cost group for one run """
    times, memories, seen = defaultdict(float), defaultdict(list), set()
    for row in view.get("run_stages", []):
        if row.get("run_id") != run_id:
            continue
        stage = row.get("stage", "")
        for group, predicate in COST_STAGES.items():
            if not predicate(stage):
                continue
            seconds = num(row.get("elapsed_seconds"))
            if seconds is not None:
                times[group] += seconds
                seen.add(group)
            memory = num(row.get("peak_cuda_memory_bytes"))
            if memory is not None:
                memories[group].append(memory)
    return {group: times[group] for group in seen}, memories


def measured_costs(values, view, candidate_count):
    """ Fill the cost table from the miss run and the sweep row from the hit """
    miss, hit = cache_run_ids(view)
    if miss is None:
        return
    times, memories = stage_totals(view, miss)
    for group, seconds in times.items():
        values["time" + group] = f"{seconds:.1f}~s"
    for group, observed in memories.items():
        if observed:
            values["mem" + group] = f"{max(observed) / 1e9:.1f}~GB"
    if times:
        values["timeMissTotal"] = f"{sum(times.values()):.1f}~s"

    if "Votes" in times and candidate_count:
        values["timeMissSweepEquivalent"] = f"{times['Votes'] * candidate_count:.1f}~s"

    if hit is None:
        return
    hit_times, hit_memories = stage_totals(view, hit)
    if "Threshold" in hit_times:
        values["timeSweepWarm"] = f"{hit_times['Threshold']:.1f}~s"
    if hit_memories.get("Threshold"):
        values["memSweepWarm"] = f"{max(hit_memories['Threshold']) / 1e9:.1f}~GB"


def error_decomposition(values, miou, reference):
    """ Derive the three gaps from the means already computed """
    for dataset in ("replica", "scannet"):
        annotation = miou.get(dataset, {}).get("GT")
        detector = miou.get(dataset, {}).get("YOLO")
        clean = reference.get(dataset)
        if annotation is not None and detector is not None:
            values[dataset + "DetectorGap"] = f"{annotation - detector:.2f}"
        if annotation is not None and clean is not None:
            values[dataset + "LiftingGap"] = f"{clean - annotation:.2f}"
        if clean is not None:
            values[dataset + "RepGap"] = f"{1.0 - clean:.2f}"


def selection_extras(values, point):
    """ Read the numbers the selection rule produced beside the point itself """
    for macro, key, digits in (
        ("bestMean", "best_mean", 2),
        ("bestMeanSd", "best_mean_sd", 2),
        ("selectedMean", "selected_mean", 2),
        ("selectedSd", "selected_sd", 2),
    ):
        observed = num(point.get(key))
        if observed is not None:
            values[macro] = f"{observed:.{digits}f}"
    eligible = num(point.get("eligible_count"))
    if eligible is not None:
        values["eligibleCount"] = f"{int(round(eligible))}"


def ablation_group(values, prefix, rows, counts):
    """ Fill the four cells of one ablation row """
    miou = numeric_values(rows, "mIoU")
    if miou:
        values[prefix + "mIoU"] = f"{statistics.mean(miou):.2f}"
        deviation = spread(miou)
        if deviation is not None:
            values[prefix + "Sd"] = f"{deviation:.2f}"
    reference = numeric_values(rows, "ground_truth_transfer_mIoU")
    if reference:
        values[prefix + "Ref"] = f"{statistics.mean(reference):.2f}"
    if counts:
        values[prefix + "Count"] = f"{int(round(statistics.mean(list(counts))))}"


def ablations(values, view, runs, beta, gamma):
    """ Fill the eight rows of the ablation table """
    by_variant = defaultdict(list)
    reused = defaultdict(list)
    reused_variants = defaultdict(set)
    for row in view.get("aggregate_beta_metrics", []):
        run = runs.get(row.get("run_id"))
        if not run or run.get("dataset") != ABLATION_DATASET:
            continue
        if row.get("source") != ABLATION_SOURCE:
            continue
        if not close(num(row.get("beta")), beta):
            continue
        row_gamma = num(row.get("hysteresis_gamma"))
        if is_ablation(row):
            if close(row_gamma, gamma):
                by_variant[row.get("variant")].append(row)
            continue
        for prefix, fixed_gamma in REUSED_ROWS:
            if close(row_gamma, gamma if fixed_gamma is None else fixed_gamma):
                reused[prefix].append(row)
                reused_variants[prefix].add(row.get("variant") or "")

    counts = selected_counts(view, runs, beta)
    for prefix, rows in reused.items():
        gathered = [
            value
            for variant in reused_variants[prefix]
            for value in counts.get(variant, {}).values()
        ]
        ablation_group(values, prefix, rows, gathered)
    for prefix, variant in ABLATION_VARIANTS:
        if by_variant.get(variant):
            ablation_group(
                values, prefix, by_variant[variant], counts.get(variant, {}).values()
            )


def quantile_columns(rows):
    """ Discover the recorded quantile columns and the percentile of each """
    columns = {}
    for row in rows:
        for key in row:
            if key == "target_score_median":
                columns[key] = 50.0
                continue
            match = re.fullmatch(r"target_score_p(\d+(?:_\d+)?)", key or "")
            if match:
                columns[key] = float(match.group(1).replace("_", "."))
    return columns


def stability(values, view, runs, beta, gamma):
    """ Fill the two dispersions, the per-class range and the quantile at beta """
    curves, curves_no_hyst = defaultdict(dict), defaultdict(dict)
    at_point, at_point_no_hyst = defaultdict(dict), defaultdict(dict)
    per_class = defaultdict(list)
    for row in view.get("class_beta_metrics", []):
        run = runs.get(row.get("run_id"))
        if not run or run.get("dataset") != ABLATION_DATASET:
            continue
        if row.get("source") != ABLATION_SOURCE:
            continue
        if is_ablation(row):
            continue
        iou, row_beta = num(row.get("iou")), num(row.get("beta"))
        row_gamma = num(row.get("hysteresis_gamma"))
        if iou is None or row_beta is None:
            continue
        scene, class_id = row.get("scene_id"), row.get("class_id")
        if close(row_gamma, gamma):
            curves[(scene, class_id)][row_beta] = iou
            if close(row_beta, beta):
                at_point[scene][class_id] = iou
                per_class[class_id].append(iou)
        elif close(row_gamma, 0.0):
            curves_no_hyst[(scene, class_id)][row_beta] = iou
            if close(row_beta, beta):
                at_point_no_hyst[scene][class_id] = iou

    for macro, gathered in (
        ("sdBetaHyst", curves),
        ("sdBetaNoHyst", curves_no_hyst),
        ("sdClassHyst", at_point),
        ("sdClassNoHyst", at_point_no_hyst),
    ):
        spreads = [
            value
            for value in (spread(list(inner.values())) for inner in gathered.values())
            if value is not None
        ]
        if spreads:
            values[macro] = f"{statistics.mean(spreads):.2f}"

    means = [statistics.mean(observed) for observed in per_class.values() if observed]
    if means:
        values["classIoUmin"] = f"{min(means):.2f}"
        values["classIoUmax"] = f"{max(means):.2f}"

    rows = [
        row
        for row in view.get("vote_statistics", [])
        if (runs.get(row.get("run_id")) or {}).get("dataset") == ABLATION_DATASET
        and row.get("source") == ABLATION_SOURCE
        and not is_ablation(row)
    ]
    below = [
        percentile
        for column, percentile in quantile_columns(rows).items()
        if numeric_values(rows, column)
        and statistics.mean(numeric_values(rows, column)) <= beta
    ]
    if below:
        values["quantileAtBeta"] = f"P{max(below):g}"


def qualitative_pair(values, view, runs, beta, gamma):
    """ Name the scene and class whose IoU is the median of the test split """
    names = {row.get("class_id"): row.get("class_name") for row in view.get("classes", [])}
    pairs = []
    for row in view.get("class_beta_metrics", []):
        run = runs.get(row.get("run_id"))
        if not run or run.get("dataset") != TEST_DATASET:
            continue
        if row.get("source") != ABLATION_SOURCE or is_ablation(row):
            continue
        if not (
            close(num(row.get("beta")), beta)
            and close(num(row.get("hysteresis_gamma")), gamma)
        ):
            continue
        iou = num(row.get("iou"))
        if iou is not None:
            pairs.append(
                (
                    iou,
                    str(run.get("scene_name") or run.get("scene_id")),
                    str(names.get(row.get("class_id"), row.get("class_id"))),
                )
            )
    if not pairs:
        return
    pairs.sort()
    _, scene, class_name = pairs[len(pairs) // 2]
    values["qualScene"] = scene
    values["qualClass"] = class_name


def hardware(values, view):
    """
    Describe the GPUs from the run metadata.

    collect_run_metadata stores gpu_name as a JSON list with one entry per
    visible device, so identical devices are collapsed into a count and the
    manuscript reads "2x NVIDIA ..." rather than a JSON array.
    """
    for row in view.get("run_parameters", []):
        raw = row.get("gpu_name")
        if not raw:
            continue
        try:
            names = json.loads(raw)
        except (TypeError, ValueError):
            names = [raw]
        names = [str(name).strip() for name in names if str(name).strip()]
        if not names:
            continue
        unique = sorted(set(names))
        values["hardwareDescription"] = ", ".join(
            f"{names.count(name)}x {name}" if names.count(name) > 1 else name
            for name in unique
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
    beta, gamma, _ = selected_operating_point(args.selection)
    values = {name: "--" for name in NAMES}
    values.update(betaStar=f"{beta:g}", gammaStar=f"{gamma:g}")

    # Load the metric rows
    for macro, key in (("tauStar", "tau_star"), ("thetaStar", "theta_star")):
        if point.get(key) is not None:
            values[macro] = f"{float(point[key]):g}"
    selection_extras(values, point)

    # Aggregate completed analytics rows
    view = deduplicate_analytics(args.analytics)

    # Aggregate completed rows into manuscript macro values
    runs = {r["run_id"]: r for r in view["runs"] if r.get("status") == "completed"}
    groups = defaultdict(list)

    for row in view["aggregate_beta_metrics"]:
        if not selected_row(row, beta, gamma):
            continue
        run = runs.get(row.get("run_id"))
        if run and not is_ablation(row):
            # Select the macro values
            groups[(run.get("dataset"), row.get("source"))].append(row)

    # Quantities that depend on the mask source are written per group
    miou = defaultdict(dict)
    references = defaultdict(dict)
    excluded = defaultdict(dict)

    for (dataset, source), rows in groups.items():
        dataset_prefix = "replica" if dataset == "replica" else "scannet"
        source_prefix = "GT" if source == "gt2d" else "YOLO"
        prefix = dataset_prefix + source_prefix

        for macro, field in (
            ("mIoU", "mIoU"),
            ("Prec", "macro_precision"),
            ("Rec", "macro_recall"),
        ):
            # Format the macro record
            metric_values = numeric_values(rows, field)
            if metric_values:
                values[prefix + macro] = f"{statistics.mean(metric_values):.2f}"
        miou_values = numeric_values(rows, "mIoU")

        if miou_values:
            values[prefix + "mIoUSd"] = f"{statistics.pstdev(miou_values):.2f}"
            miou[dataset_prefix][source_prefix] = statistics.mean(miou_values)

        # Reference-relative mIoU is a mean of ratios over the classes whose reference has positive IoU
        relative_values = numeric_values(rows, "relative_mIoU")
        if relative_values:
            values[prefix + "rel"] = f"{statistics.mean(relative_values):.3f}"

        reference_values = numeric_values(rows, "ground_truth_transfer_mIoU")
        if reference_values:
            references[dataset_prefix][source_prefix] = statistics.mean(
                reference_values
            )
        excluded_values = numeric_values(rows, "zero_reference_class_count")
        if excluded_values:
            excluded[dataset_prefix][source_prefix] = sum(excluded_values)

    # Emit the two mask-source-independent quantities, once per dataset
    collapsed_reference = {}
    for dataset_prefix in ("replica", "scannet"):
        reference = collapse_over_sources(
            references.get(dataset_prefix, {}),
            dataset_prefix,
            "clean-label transfer reference mIoU",
            REFERENCE_TOLERANCE,
        )
        if reference is not None:
            collapsed_reference[dataset_prefix] = reference
            values[dataset_prefix + "Ref"] = f"{reference:.2f}"
        count = collapse_over_sources(
            excluded.get(dataset_prefix, {}),
            dataset_prefix,
            "count of classes without a positive reference",
            0.0,
        )
        if count is not None:
            values[dataset_prefix + "RelExcl"] = f"{int(round(count))}"

    error_decomposition(values, miou, collapsed_reference)
    ablations(values, view, runs, beta, gamma)
    stability(values, view, runs, beta, gamma)
    qualitative_pair(values, view, runs, beta, gamma)

    # The sweep the manuscript compares against would recompute the projection once per candidate
    candidate_count = len(
        {
            num(row.get("beta"))
            for row in view.get("aggregate_beta_metrics", [])
            if num(row.get("beta")) is not None
        }
    )
    measured_costs(values, view, candidate_count)
    hardware(values, view)

    # Render the macro file and report missing measurements
    args.output.parent.mkdir(parents=True, exist_ok=True)
    missing = [n for n in NAMES if values[n] == "--"]
    text = (
        "% Generated by evaluation/scripts/make_macros.py.\n"
        "% Uses renewcommand because main.tex declares the same macro names\n"
        "% as placeholders before this file is input.\n"
        + "".join(f"\\renewcommand\\{name}{{{values[name]}}}\n" for name in NAMES)
    )
    atomic_write_text(args.output, text)
    print(f"unfilled macros ({len(missing)}/{len(NAMES)}): " + ", ".join(missing))

    # Write the macro file
    return 0


if __name__ == "__main__":
    raise SystemExit(main())