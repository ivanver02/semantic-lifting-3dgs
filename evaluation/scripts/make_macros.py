# Generate the measured manuscript macros without changing the manuscripts

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from evaluation.analytics import QUANTILES, close, dataset_of, is_frozen, load_analytics, number
from evaluation.summaries import (
    at_point, class_summary, frozen_rows, gaps, load_points, mask_rows, mask_summary, mean, paired, reachable,
    scene_summary, threshold_summary, validation_grid,
)

DATASETS = (("replica", "replica"), ("scannetpp", "scannet"))
SOURCES = (("gt2d", "GT"), ("yolo", "YOLO"))
SCORES = (("fraction", "Frac"), ("evidence", "Evid"), ("per_view", "View"))
DATASETS_TEXT = (("replica", "Replica"), ("scannetpp", "ScanNet++"))
TRANSFER_TEXT = {"nearest_neighbor_label": "nearest Gaussian", "radius_vote": "radius vote"}

# Rows of the ablation table around the radius vote point: hysteresis disabled reuses the
# validation runs at gamma zero, and the other rows are contribution analysis variants
ABLATIONS = (
    ("ablFrozen", "frozen_g", None),
    ("ablNoHyst", "frozen_g", 0.0),
    ("ablNoGtoM", "contribution_analysis_no_competition_gtom", None),
    ("ablNoMtoG", "contribution_analysis_no_competition_mtog", None),
    ("ablNoBoth", "contribution_analysis_no_competition_both", None),
    ("ablNearest", "contribution_analysis_nearest", None),
    ("ablNoOpac", "contribution_analysis_no_opacity", None),
    ("ablAllViews", "contribution_analysis_all_views", None),
)

# The stage groups of the cost table, in the order they run
COST_STAGES = {
    "Prepare": lambda stage: stage == "prepare_dataset",
    "Masks": lambda stage: stage in {"generate_gt_masks", "generate_yolo_masks"},
    "Training": lambda stage: stage == "train_gaussians",
    "Votes": lambda stage: stage.endswith(":votes"),
    "Threshold": lambda stage: stage.endswith(":threshold_hysteresis"),
    "Transfer": lambda stage: (
        stage == "ground_truth_transfer" or stage.endswith(":evaluation_transfer")
    ),
}


def two(value):
    return "--" if value is None else f"{value:.2f}"


def signed(value):
    return "--" if value is None else f"{value:+.2f}"


def capital(name):
    """ Macro-safe form of a class or dataset name, as in clsScannetTable """
    return "".join(part.capitalize() for part in name.replace("+", "").split())


def selection_values(values, selections, selected, radius):
    """ The two operating points and the numbers of the selection rule beside them """
    chosen, only_radius = selections["selected"], selections["radius"]
    values.update(
        betaStar=f"{selected.beta:g}", gammaStar=f"{selected.gamma:g}",
        transferStar=TRANSFER_TEXT[selected.transfer],
        tauStar=f"{float(chosen['tau_star']):.2f}", thetaStar=f"{float(chosen['theta_star']):g}",
        candidateCount=f"{chosen.get('candidate_count', '--')}", eligibleCount=f"{chosen['eligible_count']}",
        bestMean=two(chosen["best_mean"]), bestMeanSd=two(chosen["best_mean_sd"]),
        selectedMean=two(chosen["selected_mean"]), selectedSd=two(chosen["selected_sd"]),
        radiusBetaStar=f"{radius.beta:g}", radiusGammaStar=f"{radius.gamma:g}",
        radiusCandidateCount=f"{only_radius.get('candidate_count', '--')}",
        radiusEligibleCount=f"{only_radius['eligible_count']}",
        radiusSelectedMean=two(only_radius["selected_mean"]), radiusSelectedSd=two(only_radius["selected_sd"]),
        radiusBestMean=two(only_radius["best_mean"]),
    )


def main_results(values, view, point, prefix=""):
    """ Main table, relative mIoU and the three gaps of one operator at its point """
    for dataset, name in DATASETS:
        stem = prefix + (name.capitalize() if prefix else name)
        for source, label in SOURCES:
            summary = scene_summary(view, point, dataset, source)
            if summary is None:
                continue
            values[stem + label + "mIoU"] = two(summary["miou"])
            values[stem + label + "mIoUSd"] = two(summary["sd"])
            values[stem + label + "mIoUMedian"] = two(summary["median"])
            values[stem + label + "mIoUAll"] = two(summary["miou_all"])
            values[stem + "Pairs"], values[stem + "Unreachable"] = str(summary["pairs"]), str(summary["unreachable"])
            scene_miou = [item["miou"] for item in summary["scenes"].values()]
            values[stem + label + "SceneMin"], values[stem + label + "SceneMax"] = two(min(scene_miou)), two(max(scene_miou))
            values[stem + label + "Prec"] = two(summary["precision"])
            values[stem + label + "Rec"] = two(summary["recall"])
            if summary["relative"] is not None:
                values[stem + label + "rel"] = f"{summary['relative']:.3f}"
            if source == "gt2d":
                values[stem + "Ref"] = two(summary["reference"])
        split_gaps = gaps(view, point, dataset)
        if split_gaps:
            values[stem + "DetectorGap"] = two(split_gaps["detector"])
            values[stem + "LiftingGap"] = two(split_gaps["lifting"])
            values[stem + "RepGap"] = two(split_gaps["representation"])


def operators(values, view, selected, radius):
    """ The selected operator against the radius vote, each at its own point, scene by scene """
    for dataset, name in DATASETS:
        for source, label in SOURCES:
            first, second = scene_summary(view, selected, dataset, source), scene_summary(view, radius, dataset, source)
            if first is None or second is None:
                continue
            comparison = paired(first, second)
            stem = name.capitalize() + label
            values["opDelta" + stem] = signed(comparison["delta"])
            values["opBetter" + stem] = str(comparison["better"])
            values["opWorse" + stem] = str(comparison["worse"])
            values["opScenes" + name.capitalize()] = str(comparison["count"])
            if source == "gt2d":
                values["opRefDelta" + name.capitalize()] = signed(first["reference"] - second["reference"])


def stability(values, view, selected, radius):
    """ Dispersions along beta and between classes, and the range of the validation mean along beta """
    curves, at_selected = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    per_class = defaultdict(list)
    for row in view["class_beta_metrics"]:
        if not (is_frozen(row, selected.transfer) and dataset_of(row) == "replica" and row["source"] == "gt2d"):
            continue
        iou, row_beta, row_gamma = number(row["iou"]), number(row["beta"]), number(row["hysteresis_gamma"])
        for setting, setting_gamma in (("Hyst", selected.gamma), ("NoHyst", 0.0)):
            if close(row_gamma, setting_gamma):
                # For a fixed class, across the beta grid, and between classes at the selected beta
                curves[setting][(row["scene_id"], row["class_id"])].append(iou)
                if close(row_beta, selected.beta):
                    at_selected[setting][row["scene_id"]].append(iou)
                    if setting == "Hyst":
                        per_class[row["class_id"]].append(iou)

    for setting in ("Hyst", "NoHyst"):
        spreads = [statistics.pstdev(v) for v in curves[setting].values() if len(v) > 1]
        values["sdBeta" + setting] = two(mean(spreads))
        spreads = [statistics.pstdev(v) for v in at_selected[setting].values() if len(v) > 1]
        values["sdClass" + setting] = two(mean(spreads))
    means = [statistics.mean(observed) for observed in per_class.values()]
    if means:
        values["classIoUmin"], values["classIoUmax"] = two(min(means)), two(max(means))

    # Range of the mean validation mIoU along the beta grid, with hysteresis and without it
    for prefix, point in (("", selected), ("radius", radius)):
        grid = {key: statistics.mean(scenes.values()) for key, scenes in validation_grid(view, point.transfer).items()}
        for setting, gamma in (("Hyst", point.gamma), ("NoHyst", 0.0)):
            line = {beta: value for (beta, g), value in grid.items() if close(g, gamma)}
            if not line:
                continue
            name = (prefix + setting) if prefix else setting[0].lower() + setting[1:]
            values[name + "RangeLow"], values[name + "RangeHigh"] = two(min(line.values())), two(max(line.values()))
            if setting == "NoHyst":
                best = max(line, key=line.get)
                values[name + "Best"], values[name + "BestBeta"] = two(line[best]), f"{best:g}"
                at_beta = next((v for b, v in line.items() if close(b, point.beta)), None)
                values[name + "AtBeta"] = two(at_beta)

    # The largest recorded quantile of the target evidence fraction that still lies below beta
    rows = [row for row in view["vote_statistics"]
            if is_frozen(row, None) and dataset_of(row) == "replica" and row["source"] == "gt2d"]
    below = []
    for name, level in QUANTILES.items():
        observed = [number(row[f"target_score_{name}"]) for row in rows]
        if mean(observed) is not None and mean(observed) <= selected.beta:
            below.append(100 * level)
    if below:
        values["quantileAtBeta"] = f"P{max(below):g}"


def ablations(values, view, radius):
    """ One row per factor on the validation split with annotation-derived masks, paired with the radius vote """
    def scenes_of(variant, gamma):
        rows = [row for row in view["aggregate_beta_metrics"]
                if row["variant"].startswith(variant) and dataset_of(row) == "replica"
                and row["source"] == "gt2d" and at_point(row, radius.beta, gamma)]
        return {row["scene_id"]: row for row in rows}

    base = scenes_of("frozen_g", radius.gamma)
    for prefix, variant, fixed_gamma in ABLATIONS:
        gamma = radius.gamma if fixed_gamma is None else fixed_gamma
        rows = scenes_of(variant, gamma)
        if not rows:
            continue
        miou = [number(row["mIoU"]) for row in rows.values()]
        values[prefix + "mIoU"], values[prefix + "Sd"] = two(mean(miou)), two(statistics.pstdev(miou))
        values[prefix + "Ref"] = two(mean(number(row["ground_truth_transfer_mIoU"]) for row in rows.values()))
        differences = [number(rows[s]["mIoU"]) - number(base[s]["mIoU"]) for s in rows if s in base]
        values[prefix + "Delta"] = signed(mean(differences))
        values[prefix + "Better"] = str(sum(d > 0.0005 for d in differences))
        values[prefix + "Worse"] = str(sum(d < -0.0005 for d in differences))

        # A scene selects the sum of the Gaussians of its classes, and the cell is the mean over scenes
        per_scene = defaultdict(float)
        for row in view["class_beta_metrics"]:
            if (row["variant"].startswith(variant) and dataset_of(row) == "replica" and row["source"] == "gt2d"
                    and at_point(row, radius.beta, gamma)):
                per_scene[row["scene_id"]] += number(row["gaussian_count"])
        if per_scene:
            values[prefix + "Count"] = f"{int(round(statistics.mean(per_scene.values())))}"


def classes(values, view, point, prefix=""):
    """ IoU of every class with each source, its precision and recall with each source, and its reference """
    for (dataset, name), item in class_summary(view, point).items():
        stem = prefix + ("Cls" if prefix else "cls") + capital(dict(DATASETS)[dataset]) + capital(name)
        values[stem + "GT"], values[stem + "YOLO"] = two(item["gt2d:iou"]), two(item.get("yolo:iou"))
        values[stem + "GTPrec"], values[stem + "GTRec"] = two(item["gt2d:precision"]), two(item["gt2d:recall"])
        values[stem + "YOLOPrec"], values[stem + "YOLORec"] = two(item.get("yolo:precision")), two(item.get("yolo:recall"))
        values[stem + "Ref"], values[stem + "Scenes"] = two(item["reference"]), str(item["scenes"])
        values[stem + "Zero"] = str(item["zero_yolo"])


def development(values, view, chosen):
    """
    mIoU and reference of each development scene at the selected theta and at the default 0.5 of the
    theta phase, and at the selected tau and at the smallest one of the tau phase
    """
    parameters = {row["run_id"]: row for row in view["run_parameters"]}
    points = defaultdict(dict)
    for row in view["aggregate_beta_metrics"]:
        if not row["variant"].startswith("development_") or row["source"] != "gt2d":
            continue
        run = parameters[row["run_id"]]
        phase = "Tau" if row["variant"].startswith("development_tau_") else "Theta"
        value = number(run["tau"]) if phase == "Tau" else number(run["min_fraction"])
        points[(dataset_of(row), phase)][value] = (number(row["mIoU"]), number(row["ground_truth_transfer_mIoU"]))
    for dataset, name in DATASETS:
        stem = "dev" + name.capitalize()
        theta = points.get((dataset, "Theta"), {})
        for label, value in (("AtTheta", float(chosen["theta_star"])), ("AtDefaultTheta", 0.5)):
            match = next((v for k, v in theta.items() if close(k, value)), None)
            if match:
                values[stem + "Pred" + label], values[stem + "Ref" + label] = two(match[0]), two(match[1])
        tau = points.get((dataset, "Tau"), {})
        for label, value in (("AtTau", float(chosen["tau_star"])), ("AtSmallTau", min(tau) if tau else None)):
            match = next((v for k, v in tau.items() if value is not None and close(k, value)), None)
            if match:
                values[stem + "Pred" + label], values[stem + "Ref" + label] = two(match[0]), two(match[1])
    tau = sorted({k for (d, phase), items in points.items() if phase == "Tau" for k in items})
    if tau:
        values["devTauSmall"], values["devTauLarge"] = f"{tau[0]:.2f}", f"{tau[-1]:.2f}"


def worst_scene(values, view, selected):
    """ The test scene with the lowest mIoU under annotation masks, and the test mean without it """
    summary = scene_summary(view, selected, "scannetpp", "gt2d")
    if summary is None:
        return
    scene = min(summary["scenes"], key=lambda s: summary["scenes"][s]["miou"])
    values["worstScene"] = scene
    values["worstSceneGT"] = two(summary["scenes"][scene]["miou"])
    values["worstSceneRef"] = two(summary["scenes"][scene]["reference"])
    values["scannetGTmIoUNoWorst"] = two(mean(v["miou"] for s, v in summary["scenes"].items() if s != scene))


def scores(values, path):
    """ The threshold analysis: best threshold per item, one shared threshold, and its spread """
    for (dataset, source, score), item in threshold_summary(path).items():
        stem = dict(DATASETS)[dataset] + dict(SOURCES)[source] + dict(SCORES)[score]
        values[stem + "Oracle"], values[stem + "Shared"] = two(item["oracle"]), two(item["shared"])
        values[stem + "At"] = f"{item['threshold']:.2f}" if score == "fraction" else f"{item['threshold']:.2g}"
        values[stem + "Spread"] = f"{item['spread']:.1f}" if score != "fraction" else two(item["spread"])
        values[stem + "SpreadFull"] = f"{item['full_spread']:.1f}" if score != "fraction" else two(item["full_spread"])
        if "cross" in item:
            values[stem + "Cross"] = two(item["cross"])


def masks(values, path):
    """ The 2D agreement of YOLO with the annotation masks next to the 3D result, per dataset and per class """
    grouped = defaultdict(list)
    for row in mask_rows(path):
        grouped[(row["dataset"], row["name"])].append(row)
    for (dataset, name), rows in grouped.items():
        stem = "mask" + capital(dict(DATASETS)[dataset]) + capital(name)
        values[stem + "TwoD"] = two(mean(row["iou2d"] for row in rows))
        values[stem + "TwoDPrec"] = two(mean(row["p2d"] for row in rows))
        values[stem + "TwoDRec"] = two(mean(row["r2d"] for row in rows))
        values[stem + "ThreeD"] = two(mean(row["yolo3d"] for row in rows))
    for dataset, item in mask_summary(path).items():
        stem = dict(DATASETS)[dataset] + "Mask"
        values[stem + "Pairs"], values[stem + "Wins"] = str(item["pairs"]), str(item["wins"])
        values[stem + "TwoD"], values[stem + "ThreeD"] = two(item["iou2d"]), two(item["yolo3d"])
        values[stem + "TwoDPrec"], values[stem + "TwoDRec"] = two(item["p2d"]), two(item["r2d"])
        values[stem + "Pearson"] = two(item["pearson"])


def qualitative_pair(values, view, selected):
    """ Name the scene and class whose IoU is the median of the test split """
    names = {row["class_id"]: row["class_name"] for row in view["classes"]}
    pairs = sorted(
        (number(row["iou"]), row["scene_id"].split(":")[-1], names[row["class_id"]])
        for row in frozen_rows(view, "class_beta_metrics", selected, "scannetpp", "gt2d") if reachable(row)
    )
    if pairs:
        _, values["qualScene"], values["qualClass"] = pairs[len(pairs) // 2]


def seconds(value):
    return f"{value:.0f}~s" if value >= 100 else f"{value:.1f}~s"


def stage_costs(stages):
    """ Sum elapsed time and take the peak memory per cost group over the stages of one run """
    times, memories = defaultdict(float), defaultdict(float)
    for row in stages:
        for group, predicate in COST_STAGES.items():
            if predicate(row["stage"]):
                times[group] += number(row["elapsed_seconds"])
                memories[group] = max(memories[group], number(row["peak_cuda_memory_bytes"]) or 0.0)
    return times, memories


def costs(values, view):
    """
    Fill the cost table with the median over the frozen runs that computed every stage, which are
    test scenes run from scratch, both mask sources included. The sweep row is the median time of
    thresholding and transferring the whole beta grid from cached votes, over the validation runs

    A dash is kept for the stages that allocate no CUDA memory
    """
    stages = defaultdict(list)
    for row in view["run_stages"]:
        stages[row["run_id"]].append(row)
    parameters = {row["run_id"]: row for row in view["run_parameters"]}
    frozen = [run_id for run_id in stages if is_frozen(parameters[run_id], None)]

    # Runs recorded with every stage computed, which only happens when a scene runs from scratch in one part
    complete = [run_id for run_id in frozen if all(row["cache_mode"] == "miss" for row in stages[run_id])]
    if complete:
        measured = [stage_costs(stages[run_id]) for run_id in complete]
        values["costScenes"] = str(len(complete))
        values["costDataset"] = ", ".join(sorted({dict(DATASETS_TEXT)[parameters[r]["dataset"]] for r in complete}))
        for group in COST_STAGES:
            observed = [times[group] for times, _ in measured if group in times]
            if observed:
                values["time" + group] = seconds(statistics.median(observed))
                peak = max(memories[group] for _, memories in measured)
                values["mem" + group] = f"{peak / 1e9:.1f}~GB" if peak > 0 else "--"
        values["timeMissTotal"] = seconds(statistics.median(sum(times.values()) for times, _ in measured))

    # Validation runs that read the votes from the cache and thresholded and transferred the beta grid
    # again, with one source: the annotation-derived masks
    sweep = []
    for run_id in frozen:
        stage = {row["stage"]: row for row in stages[run_id]}
        if (parameters[run_id]["dataset"] == "replica" and stage.get("gt2d:votes", {}).get("cache_mode") == "hit"
                and stage.get("gt2d:threshold_hysteresis", {}).get("cache_mode") == "miss"):
            sweep.append(number(stage["gt2d:threshold_hysteresis"]["elapsed_seconds"])
                         + number(stage["gt2d:evaluation_transfer"]["elapsed_seconds"]))
    if sweep:
        values["timeSweepWarm"] = seconds(statistics.median(sweep))


def hardware(values, view):
    """
    Describe the GPUs from the run metadata

    collect_run_metadata stores gpu_name as a JSON list with one entry per visible device,
    so identical devices are collapsed into a count, as in "2x NVIDIA ..."
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
    p = argparse.ArgumentParser()
    p.add_argument("--analytics", type=Path, required=True,
                   help="analytics directory, with selection_transfer.json and selection.json")
    p.add_argument("--analysis", type=Path, default=None,
                   help="directory with mask_agreement.csv and threshold_scores.csv, skipped when missing")
    p.add_argument("--output", type=Path, default=Path("preprint/macros_measured.tex"))
    args = p.parse_args(argv)

    view = load_analytics(args.analytics)
    selected, radius, selections = load_points(args.analytics)

    values = {}
    selection_values(values, selections, selected, radius)
    main_results(values, view, selected)
    main_results(values, view, radius, prefix="radius")
    operators(values, view, selected, radius)
    stability(values, view, selected, radius)
    ablations(values, view, radius)
    classes(values, view, selected)
    classes(values, view, radius, prefix="radius")
    development(values, view, selections["selected"])
    worst_scene(values, view, selected)
    qualitative_pair(values, view, selected)
    costs(values, view)
    hardware(values, view)
    if args.analysis is not None and (args.analysis / "threshold_scores.csv").exists():
        scores(values, args.analysis / "threshold_scores.csv")
    if args.analysis is not None and (args.analysis / "mask_agreement.csv").exists():
        masks(values, args.analysis / "mask_agreement.csv")

    # providecommand first, so the file works whether or not the manuscript declares the names
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "% Generated by evaluation/scripts/make_macros.py from the analytics and the analyses\n"
        + "".join(f"\\providecommand{{\\{name}}}{{}}\\renewcommand{{\\{name}}}{{{values[name]}}}\n"
                  for name in sorted(values)),
        encoding="utf-8",
    )
    missing = sorted(name for name, value in values.items() if value == "--")
    print(f"wrote {len(values)} macros to {args.output}; without a value: {', '.join(missing) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
