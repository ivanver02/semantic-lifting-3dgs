# Append only CSV tables for validation analysis

import csv
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .common import atomic_write_text, safe_name, threshold_path, vote_class_dir


# Keep each relation in its own CSV
SCHEMA = {
    "runs": [
        "run_id", "created_at", "status", "dataset", "scene_id",
        "scene_name", "split", "source", "output_root", "model_root",
        "elapsed_seconds", "peak_cuda_memory_bytes",
        "peak_cuda_memory_reserved_bytes",
    ],

    "run_parameters": [
        "run_id", "variant", "vote_id", "evaluation_scope_version", "dataset",
        "scene", "split", "data_root", "sequence_name", "frame_step",
        "iterations", "resolution", "train_data_device", "vote_data_device",
        "replica_vertex_label_min_fraction", "replica_visibility_slop",
        "scannetpp_mask_version", "scannetpp_mask_bands",
        "yolo_conf", "hysteresis_gamma", "hysteresis_radius",
        "background_mode", "background_confidence", "background_view_policy",
        "betas", "tau", "min_fraction", "mesh_to_gaussian_transfer",
        "gaussian_to_mesh_transfer", "min_opacity",
        "gaussian_to_mesh_background_competes",
        "mesh_to_gaussian_background_competes", "opacity_weighting",
        "raster_block_size", "code_commit", "gpu_name", "command",
    ],

    "classes": ["class_id", "dataset", "class_name", "detector_name", "detector_stored_id"],

    # Per scene ground-truth support, consumed by the per-class figure
    "scene_classes": [
        "scene_id", "class_id", "gt_vertex_count", "gt_visible_vertex_count",
        "gt_evaluated_vertex_count",
    ],

    "run_stages": [
        "run_id", "dataset", "scene_id", "stage", "cache_mode",
        "container_count",
        "elapsed_seconds", "peak_cuda_memory_bytes",
        "peak_cuda_memory_reserved_bytes",
    ],

    "vote_statistics": [
        "run_id", "variant", "scene_id", "source", "vote_id", "class_id", "num_cameras", "num_class_views",
        "num_gaussians", "target_weight_sum", "background_weight_sum",
        "supported_gaussians", "target_score_mean", "target_score_std",
        "target_score_min", "target_score_p05", "target_score_p25",
        "target_score_median", "target_score_p75", "target_score_p90",
        "target_score_p92_5", "target_score_p95", "target_score_p97_5",
        "target_score_p99", "target_score_p99_9",
        "target_score_max", "supported_fraction",
    ],

    # One row per class and beta with the number of selected Gaussians
    "gaussian_statistics": [
        "run_id", "variant", "scene_id", "source", "vote_id", "class_id",
        "beta_id", "beta", "set_type", "gaussian_count",
    ],

    "class_beta_metrics": [
        "run_id", "variant", "scene_id", "source", "vote_id", "class_id", "beta_id", "beta", "hysteresis_gamma", "tp", "fp", "fn",
        "gt_count", "pred_count", "precision", "recall", "iou",
        "ground_truth_transfer_tp", "ground_truth_transfer_fp",
        "ground_truth_transfer_fn", "ground_truth_transfer_gt_count",
        "ground_truth_transfer_pred_count",
        "ground_truth_transfer_precision", "ground_truth_transfer_recall",
        "ground_truth_transfer_iou", "relative_iou",
    ],

    "aggregate_beta_metrics": [
        "run_id", "variant", "scene_id", "source", "beta_id", "beta", "hysteresis_gamma", "mIoU", "global_iou",
        "macro_precision", "macro_recall", "global_precision", "global_recall",
        "ground_truth_transfer_mIoU", "ground_truth_transfer_macro_precision",
        "ground_truth_transfer_macro_recall",
        "ground_truth_transfer_global_precision",
        "ground_truth_transfer_global_recall", "relative_mIoU",
        "evaluated_classes", "relative_classes", "zero_reference_class_count",
    ],
}


def _command_output(command):
    """ Return command output"""
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _gpu_names():
    """ Return the names of the visible NVIDIA devices """
    output = _command_output([
        "nvidia-smi", "--query-gpu=name", "--format=csv,noheader",
    ])
    if output is None:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def utc_now():
    """ Return a timestamp for CSV records """
    return datetime.now(timezone.utc).isoformat()


def collect_run_metadata(repo_root, command):
    """ Collect the reproducibility metadata recorded with every run """
    return {
        "code_commit": _command_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"]),
        "gpu_name": json.dumps(_gpu_names()),
        "command": shlex.join(command),
    }


class AnalyticsStore:
    """ Write append rows for analytical relations """

    def __init__(self, root):
        """ Create the analytics directory and table headers """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._unique_keys = {}

        for table, fields in SCHEMA.items():
            path = self.root / f"{table}.csv"
            if not path.exists() or path.stat().st_size == 0:
                with path.open("w", newline="", encoding="utf-8") as handle:
                    csv.DictWriter(handle, fieldnames=fields).writeheader()

    def append(self, table, row):
        """Append one row using the table schema"""
        fields = SCHEMA[table]
        values = {field: row.get(field) for field in fields}
        with (self.root / f"{table}.csv").open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(values)

    def append_unique(self, table, row, key_fields):
        """Append a row when its key is new"""
        path = self.root / f"{table}.csv"
        key = tuple(str(row.get(field, "")) for field in key_fields)
        cache_key = (table, tuple(key_fields))
        keys = self._unique_keys.setdefault(cache_key, set())
        if not keys and path.exists():
            with path.open("r", newline="", encoding="utf-8") as handle:
                keys.update(
                    tuple(existing.get(field, "") for field in key_fields)
                    for existing in csv.DictReader(handle)
                )
        if key in keys:
            return
        self.append(table, row)
        keys.add(key)


def gaussian_count(path):
    """ Count Gaussians from the PLY header without loading vertex arrays """
    path = Path(path)
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        for line in handle:
            text = line.strip()
            if text.startswith(b"element vertex"):
                return int(text.split()[-1])
            if text == b"end_header":
                break
    raise ValueError(f"PLY file has no vertex element: {path}")


def deduplicate_analytics(root):
    """
    Return one analytical view that keeps only the latest completed data

    Scenes may be evaluated repeatedly over time, so metric tables can hold
    several rows per key. Rows from completed runs win, and among them the
    latest created_at wins.
    """
    root = Path(root)
    view = {}
    for table in SCHEMA:
        path = root / f"{table}.csv"
        if path.exists():
            with path.open("r", newline="", encoding="utf-8") as handle:
                view[table] = list(csv.DictReader(handle))

    # Represent missing tables as empty lists
        else:
            view[table] = []

    # One row per run is written, at completion; last occurrence wins
    runs = {}
    for row in view["runs"]:
        runs[row.get("run_id")] = row
    view["runs"] = list(runs.values())
    completed = {
        run_id: row.get("created_at", "")
        for run_id, row in runs.items()
        if row.get("status") == "completed"
    }

    keys = {
        "aggregate_beta_metrics": (
            "scene_id", "variant", "source", "beta", "hysteresis_gamma",
        ),
        "class_beta_metrics": (
            "scene_id", "variant", "source", "beta", "hysteresis_gamma", "class_id",
        ),
        "vote_statistics": ("scene_id", "variant", "source", "vote_id", "class_id"),
        "gaussian_statistics": (
            "scene_id", "variant", "source", "vote_id", "class_id",
            "beta", "set_type",
        ),
        "scene_classes": ("scene_id", "class_id"),
    }
    for table, key_fields in keys.items():
        selected = {}
        for row in view.get(table, []):
            key = tuple(row.get(field, "") for field in key_fields)
            rank = completed.get(row.get("run_id", ""), "")
            previous = selected.get(key)
            if previous is None or rank >= previous[0]:
                selected[key] = (rank, row)
        view[table] = [item[1] for item in selected.values()]
    return view


def record_class_inventory(store, scene, scene_id):
    """ Record once the target classes and their ground-truth support per scene """
    evaluation_mask = scene.evaluation_mask
    for class_id, spec in enumerate(scene.classes):
        store.append_unique("classes", {
            "class_id": f"{scene.dataset}:{class_id}",
            "dataset": scene.dataset,
            "class_name": spec.name,
            "detector_name": spec.name_by_detector,
            "detector_stored_id": spec.detector_stored_id,
        }, ["class_id"])

        # Ground-truth vertex support of this class in this scene
        class_mask = scene.semantic_labels == class_id
        store.append_unique("scene_classes", {
            "scene_id": scene_id,
            "class_id": f"{scene.dataset}:{class_id}",
            "gt_vertex_count": int(class_mask.sum()),
            "gt_visible_vertex_count": int((class_mask & scene.visible).sum()),
            "gt_evaluated_vertex_count": int((class_mask & evaluation_mask).sum()),
        }, ["scene_id", "class_id"])


def record_source_analytics(store, run_id, source, scene, scene_id, classes, betas,
                            source_dir, result, vote_identifier,
                            hysteresis_gamma, hysteresis_radius):
    """ Record votes, Gaussian counts and metrics for one mask source """
    for spec in classes:
        class_id = scene.class_id(spec.name)
        analytics_class_id = f"{scene.dataset}:{class_id}"
        safe = safe_name(spec.name_by_detector)

        # Vote statistics come from the JSON written by the accumulation container
        class_dir = source_dir / safe
        vote_dir = vote_class_dir(source_dir, spec, vote_identifier)
        vote_stats_path = vote_dir / "vote_statistics.json"
        if vote_stats_path.exists():
            vote_stats = json.loads(vote_stats_path.read_text())
            vote_stats.update({
                "run_id": run_id,
                "variant": result.get("variant"),
                "scene_id": scene_id,
                "source": source,
                "vote_id": vote_identifier,
                "class_id": analytics_class_id,
            })
            store.append("vote_statistics", vote_stats)

        item = result["per_class"].get(spec.name, {})
        for beta_order, beta in enumerate(betas, start=1):
            beta_id = f"{run_id}:{source}:{beta_order}"
            beta_key = str(beta)
            sweep = item.get("sweep", {}).get(beta_key)

            # Number of Gaussians selected by this threshold, read from the PLY header
            predicted_path = threshold_path(
                source_dir, spec, vote_identifier,
                hysteresis_gamma, hysteresis_radius, beta,
            )
            store.append("gaussian_statistics", {
                "run_id": run_id,
                "variant": result.get("variant"),
                "scene_id": scene_id,
                "source": source,
                "vote_id": vote_identifier,
                "class_id": analytics_class_id,
                "beta_id": beta_id,
                "beta": beta,
                "set_type": "predicted",
                "gaussian_count": gaussian_count(predicted_path),
            })
            if sweep is None:
                continue
            prediction = sweep["iou"]
            ground_truth_transfer_metrics = sweep["ground_truth_transfer_iou"]
            store.append("class_beta_metrics", {
                "run_id": run_id,
                "variant": result.get("variant"),
                "scene_id": scene_id,
                "source": source,
                "vote_id": vote_identifier,
                "class_id": analytics_class_id,
                "beta_id": beta_id,
                "beta": beta,
                "hysteresis_gamma": hysteresis_gamma,
                "tp": prediction["tp"],
                "fp": prediction["fp"],
                "fn": prediction["fn"],
                "gt_count": prediction["gt_count"],
                "pred_count": prediction["pred_count"],
                "precision": prediction["precision"],
                "recall": prediction["recall"],
                "iou": prediction["iou"],
                "ground_truth_transfer_tp": ground_truth_transfer_metrics["tp"],
                "ground_truth_transfer_fp": ground_truth_transfer_metrics["fp"],
                "ground_truth_transfer_fn": ground_truth_transfer_metrics["fn"],
                "ground_truth_transfer_gt_count": ground_truth_transfer_metrics["gt_count"],
                "ground_truth_transfer_pred_count": ground_truth_transfer_metrics["pred_count"],
                "ground_truth_transfer_precision": ground_truth_transfer_metrics["precision"],
                "ground_truth_transfer_recall": ground_truth_transfer_metrics["recall"],
                "ground_truth_transfer_iou": ground_truth_transfer_metrics["iou"],
                "relative_iou": sweep["relative_iou"],
            })

    for beta_order, beta in enumerate(betas, start=1):
        aggregate = result["metrics_by_beta"].get(str(beta))
        if aggregate is None:
            continue
        store.append("aggregate_beta_metrics", {
            "run_id": run_id,
            "variant": result.get("variant"),
            "scene_id": scene_id,
            "source": source,
            "beta_id": f"{run_id}:{source}:{beta_order}",
            "beta": beta,
            "hysteresis_gamma": hysteresis_gamma,
            **aggregate,
        })
