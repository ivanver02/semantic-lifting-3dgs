# Generate the measured manuscript macros without changing it

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from evaluation.analytics import deduplicate_analytics
from evaluation.common import atomic_write_text
from evaluation.scripts.make_tables import (
    require_complete_state_store,
    selected_operating_point,
)

NAMES = """betaStar gammaStar tauStar thetaStar
replicaGTmIoU replicaYOLOmIoU scannetGTmIoU scannetYOLOmIoU
replicaGTmIoUSd replicaGTPrec replicaGTRec replicaGTRef
replicaYOLOmIoUSd replicaYOLOPrec replicaYOLORec replicaYOLORef
scannetGTmIoUSd scannetGTPrec scannetGTRec scannetGTRef
scannetYOLOmIoUSd scannetYOLOPrec scannetYOLORec scannetYOLORef
sdBetaNoHyst sdBetaHyst sdClassNoHyst sdClassHyst bestMean bestMeanSd selectedMean selectedSd eligibleCount quantileAtBeta
    replicaDetectorGap replicaLiftingGap replicaRepGap scannetDetectorGap scannetLiftingGap scannetRepGap classIoUmin classIoUmax
replicaGTrel replicaGTrelExcl replicaYOLOrel replicaYOLOrelExcl scannetGTrel scannetGTrelExcl scannetYOLOrel scannetYOLOrelExcl
ablFrozenmIoU ablFrozenSd ablFrozenRef ablFrozenCount ablNoHystmIoU ablNoHystSd ablNoHystRef ablNoHystCount
ablNoGtoMmIoU ablNoGtoMSd ablNoGtoMRef ablNoGtoMCount ablNoMtoGmIoU ablNoMtoGSd ablNoMtoGRef ablNoMtoGCount
ablNoBothmIoU ablNoBothSd ablNoBothRef ablNoBothCount ablNearestmIoU ablNearestSd ablNearestRef ablNearestCount
ablNoOpacmIoU ablNoOpacSd ablNoOpacRef ablNoOpacCount ablAllViewsmIoU ablAllViewsSd ablAllViewsRef ablAllViewsCount
timeMasks timeVotes timeThreshold timeTransfer timeSweepWarm timeMissTotal timeMissSweepEquivalent
memMasks memVotes memThreshold memTransfer memSweepWarm hardwareDescription qualScene qualClass""".split()


def num(value):
    # Convert optional analytics values
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def selected_row(row, beta, gamma):
    # Ignore rows with incomplete operating-point metadata
    row_beta = num(row.get("beta"))
    row_gamma = num(row.get("hysteresis_gamma"))
    return (
        row_beta is not None
        and row_gamma is not None
        and abs(row_beta - beta) <= 1e-9
        and abs(row_gamma - gamma) <= 1e-9
    )


def numeric_values(rows, field):
    # Keep incomplete metric cells out of aggregate statistics
    return [
        value for value in (num(row.get(field)) for row in rows) if value is not None
    ]


def miss_run_id(view):
    """ Return the completed run with the most observed vote containers """
    completed = {
        row.get("run_id")
        for row in view.get("runs", [])
        if row.get("status") == "completed"
    }
    counts = defaultdict(int)
    for row in view.get("run_stages", []):
        if row.get("run_id") not in completed:
            continue
        if not row.get("stage", "").endswith(":votes"):
            continue
        counts[row.get("run_id")] += int(row.get("container_count") or 0)
    if not counts:
        return None
    return max(
        counts,
        key=lambda run_id: (counts[run_id], run_id or ""),
    )


def stage_total(view, run_id, predicate, field):
    """ Sum one numeric stage field for a run """
    values = [
        num(row.get(field))
        for row in view.get("run_stages", [])
        if row.get("run_id") == run_id and predicate(row.get("stage", ""))
    ]
    return sum(value for value in values if value is not None)


def measured_costs(values, view):
    """ Fill cost macros from the run identified as miss by observed votes """
    run_id = miss_run_id(view)
    if run_id is None:
        return
    mappings = {
        "timeMasks": lambda stage: (
            stage in {"generate_gt_masks", "generate_yolo_masks"}
        ),
        "timeVotes": lambda stage: stage.endswith(":votes"),
        "timeThreshold": lambda stage: stage.endswith(":threshold_hysteresis"),
        "timeTransfer": lambda stage: (
            stage in {"ground_truth_transfer"} or stage.endswith(":evaluation_transfer")
        ),
    }
    for macro, predicate in mappings.items():
        seconds = stage_total(view, run_id, predicate, "elapsed_seconds")
        if seconds:
            values[macro] = f"{seconds:.1f}~s"


def main(argv=None):
    # Parse inputs and load the selected operating point
    p = argparse.ArgumentParser()
    p.add_argument("--analytics", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("preprint/macros_measured.tex"))

    # Parse optional experiment completeness inputs
    p.add_argument("--state-store", type=Path)
    p.add_argument("--experiment")
    args = p.parse_args(argv)
    require_complete_state_store(args.state_store, args.experiment)
    point = json.loads(args.selection.read_text(encoding="utf-8"))
    beta, gamma, _ = selected_operating_point(args.selection)
    values = {name: "--" for name in NAMES}
    values.update(betaStar=f"{beta:g}", gammaStar=f"{gamma:g}")

    # Load the metric rows
    for macro, key in (("tauStar", "tau_star"), ("thetaStar", "theta_star")):
        if point.get(key) is not None:
            values[macro] = f"{float(point[key]):g}"

    # Aggregate completed analytics rows
    view = deduplicate_analytics(args.analytics)

    # Aggregate completed rows into manuscript macro values
    runs = {r["run_id"]: r for r in view["runs"] if r.get("status") == "completed"}
    groups = defaultdict(list)

    for row in view["aggregate_beta_metrics"]:
        if not selected_row(row, beta, gamma):
            continue
        run = runs.get(row.get("run_id"))
        if run:
            # Select the macro values
            groups[(run.get("dataset"), row.get("source"))].append(row)

    for (dataset, source), rows in groups.items():
        prefix = ("replica" if dataset == "replica" else "scannet") + (
            "GT" if source == "gt2d" else "YOLO"
        )

        for macro, field in (
            ("mIoU", "mIoU"),
            ("Prec", "macro_precision"),
            ("Rec", "macro_recall"),
            ("Ref", "ground_truth_transfer_mIoU"),
        ):
            # Format the macro record
            metric_values = numeric_values(rows, field)
            if metric_values:
                values[prefix + macro] = f"{statistics.mean(metric_values):.2f}"
        miou_values = numeric_values(rows, "mIoU")

        if miou_values:
            values[prefix + "mIoUSd"] = f"{statistics.pstdev(miou_values):.2f}"
    measured_costs(values, view)
    
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
    print("unfilled macros: " + ", ".join(missing))

    # Write the macro file
    return 0


if __name__ == "__main__":
    raise SystemExit(main())