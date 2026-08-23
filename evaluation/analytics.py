# Append only CSV tables for validation analysis

import csv
import hashlib
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .common import atomic_write_text, safe_name, threshold_path, vote_class_dir
from .reporting import gaussian_statistics


# Keep each relation in its own CSV
SCHEMA = {
    "runs": [
        "run_id", "created_at", "status", "dataset", "scene_id",
        "scene_name", "split", "source", "output_root", "model_root",
        "elapsed_seconds", "peak_cuda_memory_bytes",
        "peak_cuda_memory_reserved_bytes",
    ],

    "run_parameters": [
        "run_id", "variant", "vote_id", "evaluation_scope_version", "dataset", "scene", "split",
        "data_root", "iterations", "resolution", "train_data_device",
        "replica_vertex_label_min_fraction", "replica_visibility_slop",
        "scannetpp_mask_version", "scannetpp_mask_bands",
        "vote_data_device", "sequence_name", "frame_step", "yolo_conf",
        "hysteresis_gamma", "hysteresis_radius",
        "background_mode", "background_confidence", "background_view_policy",
        "betas",
        "tau", "min_fraction", "mesh_to_gaussian_transfer",
        "gaussian_to_mesh_transfer", "min_opacity",
        "gaussian_to_mesh_background_competes", "mesh_to_gaussian_background_competes",
        "opacity_weighting",
        "raster_block_size", "mask_source", "code_commit", "image_digest",
        "detector_sha256", "code_dirty", "gpu_name", "driver_version",
        "cuda_visible_devices", "command",
    ],

    "scenes": [
        "scene_id", "dataset", "scene_name", "split", "scene_path",
        "num_vertices", "num_images",
    ],

    "camera_statistics": [
        "scene_id", "dataset", "num_images",
        "width_min", "width_mean", "width_max",
        "height_min", "height_mean", "height_max",
        "fx_min", "fx_mean", "fx_max", "fy_min", "fy_mean", "fy_max",
        "cx_min", "cx_mean", "cx_max", "cy_min", "cy_mean", "cy_max",
    ],

    "classes": ["class_id", "dataset", "class_name", "detector_name", "detector_stored_id"],

    "run_sources": ["run_id", "source", "mask_directory", "segmentation_directory"],

    "run_stages": [
        "run_id", "dataset", "scene_id", "stage", "cache_mode",
        "container_count",
        "elapsed_seconds", "peak_cuda_memory_bytes",
        "peak_cuda_memory_reserved_bytes",
    ],

    "scene_classes": [
        "scene_id", "class_id", "gt_vertex_count", "gt_visible_vertex_count",
        "gt_evaluated_vertex_count",
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

    "gaussian_statistics": [
        "run_id", "variant", "scene_id", "source", "vote_id", "class_id", "beta_id", "beta", "set_type", "gaussian_count",
        "size_min", "size_mean", "size_std", "size_max", "opacity_min",
        "opacity_mean", "opacity_std", "opacity_max", "file_path",
    ],

    "model_statistics": [
        "run_id", "variant", "scene_id", "file_path", "gaussian_count",
        "size_min", "size_mean", "size_std", "size_max", "opacity_min",
        "opacity_mean", "opacity_std", "opacity_max",
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


def _sha256(path):
    """ Return a file digest """
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_cached(path, cache_root=None):
    """ Reuse a detector digest when its signature is unchanged """
    if cache_root is None:
        return _sha256(path)
    cache_root = Path(cache_root)
    cache_path = cache_root / "run_metadata_cache.json"
    cache = {}
    if cache_path.exists():
        try:

    # Load the run metadata cache
            cache = json.loads(cache_path.read_text())
        except (OSError, ValueError):
            cache = {}
    key = str(path.resolve())
    signature = None
    if path.exists():
        stat = path.stat()
        signature = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    # Check the cached file signature
    previous = cache.get(key)
    if previous is not None and previous.get("signature") == signature:
        return previous.get("digest")
    digest = _sha256(path)
    cache[key] = {"signature": signature, "digest": digest}
    atomic_write_text(cache_path, json.dumps(cache, indent=2, sort_keys=True) + "\n")
    return digest


def _git_dirty(repo_root):
    """ Return whether the repository has uncommitted changes """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    # Return the repository state
    return bool(result.stdout.strip())


def _image_digest(image_name):
    """ Return an image digest or local image ID """
    output = _command_output([
        "docker", "image", "inspect", "--format={{json .RepoDigests}}\t{{.Id}}",
        image_name,
    ])
    if output is None:
        return None
    repo_digests, separator, image_id = output.partition("\t")

    # Require a separated image response
    if not separator:
        return None
    try:
        repo_digests = json.loads(repo_digests)
    except json.JSONDecodeError:
        repo_digests = []
    if repo_digests:
        return {"kind": "repo_digest", "value": repo_digests}

    # Return the local image identifier
    return {"kind": "image_id", "value": image_id}


def collect_run_metadata(repo_root, detector_path, image_names, command, cache_root=None):
    """ Collect reproducibility metadata """
    commit = _command_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    image_digest = {name: _image_digest(name) for name in image_names}

    gpu_output = _command_output([
        "nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader",
    ])
    gpu_names = []
    driver_versions = []
    if gpu_output is not None:
        for line in gpu_output.splitlines():
            name, separator, driver = line.partition(",")

    # Record detected GPU details
            if separator:
                gpu_names.append(name.strip())
                driver_versions.append(driver.strip())

    return {
        "code_commit": commit,
        "image_digest": json.dumps(image_digest, sort_keys=True),
        "detector_sha256": _sha256_cached(detector_path, cache_root),
        "code_dirty": _git_dirty(repo_root),
        "gpu_name": json.dumps(gpu_names),
        "driver_version": json.dumps(driver_versions),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),

    # Add the command run metadata
        "command": shlex.join(command),
    }


def utc_now():
    """ Return a timestamp for CSV records """
    return datetime.now(timezone.utc).isoformat()


def _summary(values):
    """ Return summary statistics for a numeric camera field """
    values = [float(value) for value in values]
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
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
                continue

            with path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames == fields:
                    continue
                rows = list(reader)
            temporary_path = path.with_suffix(".csv.tmp")
            try:
                with temporary_path.open("w", newline="", encoding="utf-8") as handle:

    # Rewrite the table with its schema
                    writer = csv.DictWriter(
                        handle, fieldnames=fields, extrasaction="ignore"
                    )
                    writer.writeheader()
                    writer.writerows(rows)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_path, path)

    # Remove the temporary table
            finally:
                temporary_path.unlink(missing_ok=True)


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

    # Load existing unique keys
                    tuple(existing.get(field, "") for field in key_fields)
                    for existing in csv.DictReader(handle)
                )
        if key in keys:
            return
        self.append(table, row)
        keys.add(key)


def deduplicate_analytics(root):
    """ Return a deduplicated analytics view """
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

    runs = {}
    for row in view["runs"]:
        current = runs.get(row.get("run_id"))
        if current is None:
            runs[row.get("run_id")] = dict(row)
            continue
        current_completed = current.get("status") == "completed"
        row_completed = row.get("status") == "completed"

    # Select the latest run record
        if (row_completed and not current_completed) or (
                row_completed == current_completed and
                row.get("created_at", "") >= current.get("created_at", "")
        ):
            runs[row.get("run_id")] = dict(row)
    for row in runs.values():
        if row.get("status") != "completed":
            row["status"] = "incomplete"

    # Replace incomplete run records
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

    # Store the selected run records
        "gaussian_statistics": (
            "scene_id", "variant", "source", "vote_id", "class_id",
            "beta", "set_type",
        ),
        "model_statistics": ("scene_id",),
    }
    for table, key_fields in keys.items():
        selected = {}

    # Add Gaussian table keys
        for row in view.get(table, []):
            key = tuple(row.get(field, "") for field in key_fields)
            rank = completed.get(row.get("run_id", ""), "")
            previous = selected.get(key)
            if previous is None or rank >= previous[0]:
                selected[key] = (rank, row)
        view[table] = [item[1] for item in selected.values()]
    return view


def record_scene_analytics(store, args, scene, scene_id):
    """ Record the scene and its class level ground truth support once """
    evaluation_mask = scene.evaluation_mask
    store.append_unique("scenes", {
        "scene_id": scene_id,
        "dataset": scene.dataset,
        "scene_name": scene.scene,
        "split": args.split,
        "scene_path": str(getattr(scene, "scene_root", "")),

    # Record scene size fields
        "num_vertices": len(scene.vertices),
        "num_images": scene.num_images,
    }, ["scene_id"])

    camera_fields = {
        "width": [row["width"] for row in scene.camera_intrinsics],
        "height": [row["height"] for row in scene.camera_intrinsics],
        "fx": [row["fx"] for row in scene.camera_intrinsics],
        "fy": [row["fy"] for row in scene.camera_intrinsics],
        "cx": [row["cx"] for row in scene.camera_intrinsics],
        "cy": [row["cy"] for row in scene.camera_intrinsics],
    }

    # Build camera summary fields
    camera_row = {
        "scene_id": scene_id,
        "dataset": scene.dataset,
        "num_images": scene.num_images,
    }
    for field, values in camera_fields.items():
        summary = _summary(values)
        camera_row.update({

    # Add camera summary values
            f"{field}_min": summary["min"],
            f"{field}_mean": summary["mean"],
            f"{field}_max": summary["max"],
        })
    store.append_unique("camera_statistics", camera_row, ["scene_id"])
    for class_id, spec in enumerate(scene.classes):
        class_mask = scene.semantic_labels == class_id
        store.append_unique("classes", {

    # Record class metadata
            "class_id": f"{scene.dataset}:{class_id}",
            "dataset": scene.dataset,
            "class_name": spec.name,
            "detector_name": spec.name_by_detector,
            "detector_stored_id": spec.detector_stored_id,
        }, ["class_id"])
        store.append_unique("scene_classes", {
            "scene_id": scene_id,

    # Record scene class support
            "class_id": f"{scene.dataset}:{class_id}",
            "gt_vertex_count": int(class_mask.sum()),
            "gt_visible_vertex_count": int((class_mask & scene.visible).sum()),
            "gt_evaluated_vertex_count": int((class_mask & evaluation_mask).sum()),
        }, ["scene_id", "class_id"])


def record_source_analytics(store, run_id, source, scene, scene_id, classes, betas,
                            source_dir, result, vote_identifier,
                            hysteresis_gamma, hysteresis_radius, full_model_stats):
    """ Record votes, Gaussian summaries and metrics for one mask source """
    store.append_unique("model_statistics", {
        "run_id": run_id,
        "variant": result.get("variant"),
        "scene_id": scene_id,
        **full_model_stats,
    }, ["scene_id"])
    for spec in classes:
        class_id = scene.class_id(spec.name)
        analytics_class_id = f"{scene.dataset}:{class_id}"
        safe = safe_name(spec.name_by_detector)
        class_dir = source_dir / safe
        vote_dir = vote_class_dir(source_dir, spec, vote_identifier)
        vote_stats_path = vote_dir / "vote_statistics.json"
        if vote_stats_path.exists():

    # Complete vote statistics fields
            vote_stats = json.loads(vote_stats_path.read_text())
            vote_stats.update({
                "run_id": run_id,
                "variant": result.get("variant"),
                "scene_id": scene_id,
                "source": source,
                "vote_id": vote_identifier,
                "class_id": analytics_class_id,

    # Record transferred Gaussian statistics
            })
            store.append("vote_statistics", vote_stats)

        ground_truth_transfer_path = class_dir / "ground_truth_gaussians.ply"
        ground_truth_transfer_stats = gaussian_statistics(ground_truth_transfer_path)
        store.append("gaussian_statistics", {
            "run_id": run_id,
            "variant": result.get("variant"),
            "scene_id": scene_id,
            "source": source,
            "vote_id": vote_identifier,

    # Read the beta sweep entry
            "class_id": analytics_class_id,
            "beta": None,
            "set_type": "ground_truth_transfer",
            **ground_truth_transfer_stats,
        })
        item = result["per_class"].get(spec.name, {})
        for beta_order, beta in enumerate(betas, start=1):
            beta_id = f"{run_id}:{source}:{beta_order}"

    # Add predicted Gaussian metadata
            beta_key = str(beta)
            sweep = item.get("sweep", {}).get(beta_key)
            predicted_path = threshold_path(
                source_dir, spec, vote_identifier,
                hysteresis_gamma, hysteresis_radius, beta,
            )
            stats = gaussian_statistics(predicted_path)
            store.append("gaussian_statistics", {

    # Complete predicted Gaussian fields
                "run_id": run_id,
                "variant": result.get("variant"),
                "scene_id": scene_id,
                "source": source,
                "vote_id": vote_identifier,
                "class_id": analytics_class_id,
                "beta_id": beta_id,
                "beta": beta,

    # Add class metric metadata
                "set_type": "predicted",
                **stats,
            })
            if sweep is None:
                continue
            prediction = sweep["iou"]
            ground_truth_transfer_metrics = sweep["ground_truth_transfer_iou"]
            store.append("class_beta_metrics", {

    # Add transfer metric values
                "run_id": run_id,
                "variant": result.get("variant"),
                "scene_id": scene_id,
                "source": source,
                "vote_id": vote_identifier,
                "class_id": analytics_class_id,
                "beta_id": beta_id,
                "beta": beta,

    # Add aggregate metric metadata
                "hysteresis_gamma": hysteresis_gamma,
                "tp": prediction["tp"],
                "fp": prediction["fp"],
                "fn": prediction["fn"],
                "gt_count": prediction["gt_count"],
                "pred_count": prediction["pred_count"],
                "precision": prediction["precision"],
                "recall": prediction["recall"],

    # Complete metric fields
                "iou": prediction["iou"],
                "ground_truth_transfer_tp": ground_truth_transfer_metrics["tp"],
                "ground_truth_transfer_fp": ground_truth_transfer_metrics["fp"],
                "ground_truth_transfer_fn": ground_truth_transfer_metrics["fn"],
                "ground_truth_transfer_gt_count": ground_truth_transfer_metrics["gt_count"],
                "ground_truth_transfer_pred_count": ground_truth_transfer_metrics["pred_count"],
                "ground_truth_transfer_precision": ground_truth_transfer_metrics["precision"],
                "ground_truth_transfer_recall": ground_truth_transfer_metrics["recall"],

    # Complete metric fields
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

    # Complete metric fields
            "source": source,
            "beta_id": f"{run_id}:{source}:{beta_order}",
            "beta": beta,
            "hysteresis_gamma": hysteresis_gamma,
            **aggregate,
        })