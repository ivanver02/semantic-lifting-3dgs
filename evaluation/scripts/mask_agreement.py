# Compare the YOLO masks with the annotation masks in 2D, pixel by pixel, and put that agreement
# next to the 3D IoU that the lifting reaches from each of them, for every class and scene of the
# frozen configuration at the selected operating point. It reads the masks, so it runs inside
# lifting.sif through picasso/analysis.sbatch

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import cv2

from evaluation.analytics import close, is_frozen, load_analytics, number, selected_operating_point, selected_transfer


def pixel_counts(eval_dir, stored_ids):
    """ Pixel TP, FP and FN of YOLO against the annotation masks, and view counts, per stored class id """
    gt_dir, yolo_dir = eval_dir / "masks_gt2d" / "semantic", eval_dir / "masks_yolo" / "semantic"
    stems = sorted({p.stem for p in gt_dir.glob("*.png")} & {p.stem for p in yolo_dir.glob("*.png")})
    counts = {i: dict(tp=0, fp=0, fn=0, views_gt=0, views_yolo=0, views_both=0) for i in stored_ids}
    for stem in stems:
        gt = cv2.imread(str(gt_dir / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        yolo = cv2.imread(str(yolo_dir / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        if yolo.shape != gt.shape:
            # The pipeline resamples detector masks to the camera resolution with nearest neighbour
            yolo = cv2.resize(yolo, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        for i in stored_ids:
            g, y = gt == i, yolo == i
            c = counts[i]
            c["tp"] += int((g & y).sum())
            c["fp"] += int((~g & y).sum())
            c["fn"] += int((g & ~y).sum())
            c["views_gt"] += int(g.any())
            c["views_yolo"] += int(y.any())
            c["views_both"] += int(g.any() and y.any())
    return len(stems), counts


def ratio(a, b):
    return a / b if b else 0.0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="data root that holds analytics and both datasets")
    parser.add_argument("--selection", type=Path, default=None,
                        help="selection with the transfer operator, analytics/selection_transfer.json by default")
    parser.add_argument("--output", type=Path, default=None, help="CSV with one row per scene and class")
    args = parser.parse_args(argv)
    selection = args.selection or args.data / "analytics" / "selection_transfer.json"
    output = args.output or args.data / "analysis" / "mask_agreement.csv"

    view = load_analytics(args.data / "analytics")
    beta, gamma = selected_operating_point(selection)
    transfer = selected_transfer(selection)
    classes = {row["class_id"]: row for row in view["classes"]}
    print(f"3D results of the {transfer} at beta={beta:g}, gamma={gamma:g}")

    # 3D IoU at the selected point per scene, class and source, and the reference of the same operator
    iou3d = defaultdict(dict)
    for row in view["class_beta_metrics"]:
        if is_frozen(row, transfer) and close(number(row["beta"]), beta) and close(number(row["hysteresis_gamma"]), gamma):
            iou3d[(row["scene_id"], row["class_id"])][row["source"]] = number(row["iou"])
            iou3d[(row["scene_id"], row["class_id"])]["ref"] = number(row["ground_truth_transfer_iou"])

    rows = []
    for scene_id in sorted({key[0] for key in iou3d}):
        dataset, scene = scene_id.split(":")
        class_ids = sorted({key[1] for key in iou3d if key[0] == scene_id})
        stored = {int(classes[c]["detector_stored_id"]): c for c in class_ids}
        frames, counts = pixel_counts(args.data / dataset / "eval" / dataset / scene, list(stored))
        print(f"{scene_id}: {frames} frames with both masks", flush=True)
        for stored_id, class_id in stored.items():
            c = counts[stored_id]
            values = iou3d[(scene_id, class_id)]
            rows.append(dict(
                dataset=dataset, scene=scene, name=classes[class_id]["class_name"],
                iou2d=ratio(c["tp"], c["tp"] + c["fp"] + c["fn"]),
                p2d=ratio(c["tp"], c["tp"] + c["fp"]), r2d=ratio(c["tp"], c["tp"] + c["fn"]),
                views_gt=c["views_gt"], views_yolo=c["views_yolo"], views_both=c["views_both"],
                yolo3d=values.get("yolo"), gt3d=values.get("gt2d"), ref=values.get("ref"),
            ))
    if not rows:
        raise SystemExit(f"no frozen {transfer} results at beta={beta:g}, gamma={gamma:g}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")

    # A short summary per dataset, the tables and figures read the CSV
    for dataset in sorted({r["dataset"] for r in rows}):
        d = [r for r in rows if r["dataset"] == dataset]
        print(f"{dataset:10s} pairs={len(d):3d}  3D yolo > 2D IoU in {sum(r['yolo3d'] > r['iou2d'] for r in d)}  "
              f"mean 2D={statistics.mean(r['iou2d'] for r in d):.3f}  "
              f"mean 3D yolo={statistics.mean(r['yolo3d'] for r in d):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
