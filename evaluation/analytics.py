# Append only CSV tables for validation analysis

import csv
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path


# Quantiles recorded for the target evidence fraction distribution, from P05 to P99.9.
# They are consumed by the analytics CSV and reported in the manuscript appendix.
QUANTILES = {
    "p05": 0.05, "p25": 0.25, "median": 0.50, "p75": 0.75, "p90": 0.90,
    "p92_5": 0.925, "p95": 0.95, "p97_5": 0.975, "p99": 0.99, "p99_9": 0.999,
}

# Keep each relation in its own CSV
SCHEMA = {
    "runs": [
        "run_id", "created_at", "dataset", "scene_id", "scene_name", "split",
        "source", "output_root", "model_root", "elapsed_seconds",
        "peak_cuda_memory_bytes",
    ],

    "run_parameters": [
        "run_id", "variant", "vote_id", "dataset", "scene", "split", "data_root",
        "iterations", "resolution", "train_data_device", "vote_data_device",
        "raster_block_size", "hysteresis_gamma", "hysteresis_radius",
        "background_confidence", "background_view_policy",
        "betas", "tau", "min_fraction",
        "gaussian_to_mesh_transfer", "min_opacity", "opacity_weighting",
        "gaussian_to_mesh_background_competes",
        "mesh_to_gaussian_background_competes",
        "code_commit", "gpu_name", "driver_version", "command",
    ],

    "classes": ["class_id", "dataset", "class_name", "detector_name", "detector_stored_id"],

    # Per scene ground-truth support, consumed by the per-class figure
    "scene_classes": [
        "scene_id", "class_id", "gt_vertex_count", "gt_visible_vertex_count",
        "gt_evaluated_vertex_count",
    ],

    "run_stages": [
        "run_id", "dataset", "scene_id", "stage", "cache_mode",
        "elapsed_seconds", "peak_cuda_memory_bytes",
    ],

    "vote_statistics": [
        "run_id", "variant", "scene_id", "source", "vote_id", "class_id",
        "num_cameras", "num_class_views", "num_gaussians",
        "supported_gaussians", "supported_fraction",
    ] + [f"target_score_{name}" for name in QUANTILES],

    # One row per class and beta, with the number of selected Gaussians
    "class_beta_metrics": [
        "run_id", "variant", "scene_id", "source", "class_id", "beta", "hysteresis_gamma",
        "gaussian_count", "tp", "fp", "fn", "precision", "recall", "iou",
        "ground_truth_transfer_tp", "ground_truth_transfer_fp",
        "ground_truth_transfer_fn", "ground_truth_transfer_precision",
        "ground_truth_transfer_recall", "ground_truth_transfer_iou", "relative_iou",
    ],

    "aggregate_beta_metrics": [
        "run_id", "variant", "scene_id", "source", "beta", "hysteresis_gamma",
        "mIoU", "macro_precision", "macro_recall",
        "ground_truth_transfer_mIoU", "relative_mIoU",
    ],
}

# A scene may be evaluated repeatedly over time, so these tables keep the latest row per key
KEYS = {
    "classes": ("class_id",),
    "scene_classes": ("scene_id", "class_id"),
    "vote_statistics": ("scene_id", "variant", "source", "vote_id", "class_id"),
    "class_beta_metrics": ("scene_id", "variant", "source", "class_id", "beta", "hysteresis_gamma"),
    "aggregate_beta_metrics": ("scene_id", "variant", "source", "beta", "hysteresis_gamma"),
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


def utc_now():
    """ Return a timestamp for CSV records """
    return datetime.now(timezone.utc).isoformat()


def collect_run_metadata(repo_root, command):
    """ Collect the reproducibility metadata recorded with every run """

    # One line per visible NVIDIA device with its name and driver version
    output = _command_output([
        "nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader",
    ]) or ""
    devices = [line.split(",") for line in output.splitlines() if line.strip()]
    return {
        "code_commit": _command_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"]),
        "gpu_name": json.dumps([device[0].strip() for device in devices]),
        "driver_version": ";".join(sorted({device[-1].strip() for device in devices})),
        "command": shlex.join(command),
    }


class AnalyticsStore:
    """ Write append rows for analytical relations """

    def __init__(self, root):
        """ Create the analytics directory and table headers """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        for table, fields in SCHEMA.items():
            path = self.root / f"{table}.csv"
            if not path.exists():
                with path.open("w", newline="", encoding="utf-8") as handle:
                    csv.DictWriter(handle, fieldnames=fields).writeheader()
                continue

            # Rows are written by position, so a table with other columns cannot receive them
            with path.open("r", newline="", encoding="utf-8") as handle:
                if next(csv.reader(handle), None) != fields:
                    raise RuntimeError(f"{path} has other columns, use a new analytics directory")

    def append(self, table, row):
        """Append one row using the table schema"""
        fields = SCHEMA[table]
        values = {field: row.get(field) for field in fields}
        with (self.root / f"{table}.csv").open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(values)


def load_analytics(root):
    """
    Return one analytical view that keeps only the latest data of recorded runs

    A run is recorded in runs.csv when it completes, so rows written by an
    interrupted run are dropped. Among rows with the same key, the one from the
    latest run wins.
    """
    root = Path(root)
    view = {}
    for table in SCHEMA:
        path = root / f"{table}.csv"
        with path.open("r", newline="", encoding="utf-8") as handle:
            view[table] = list(csv.DictReader(handle))

    runs = {row["run_id"]: row for row in view["runs"]}
    view["runs"] = runs
    for table in SCHEMA:
        if table != "runs" and "run_id" in SCHEMA[table]:
            view[table] = [row for row in view[table] if row["run_id"] in runs]

    for table, key_fields in KEYS.items():
        selected = {}
        for row in view[table]:
            key = tuple(row[field] for field in key_fields)
            rank = runs[row["run_id"]]["created_at"] if "run_id" in row else ""
            if key not in selected or rank >= selected[key][0]:
                selected[key] = (rank, row)
        view[table] = [item[1] for item in selected.values()]
    return view


def number(value):
    """ Convert an optional analytics cell; None keeps empty cells out """
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def close(left, right, tolerance=1e-9):
    """ Compare two optional operating-point coordinates """
    return left is not None and right is not None and abs(left - right) <= tolerance


def dataset_of(row):
    """ Return the dataset of a row from its scene ID """
    return row["scene_id"].split(":")[0]


def is_frozen(row):
    """
    Whether a row belongs to the frozen configuration

    Development sweeps and contribution analysis runs share scenes and operating
    points with it, so every summary of the frozen configuration keeps only these rows
    """
    return row["variant"].startswith("frozen_")


def selected_operating_point(path):
    """ Load the (beta, gamma) pair selected on the validation scenes """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return float(data["beta_star"]), float(data["gamma_star"])


def record_class_inventory(store, scene):
    """ Record the target classes and their ground-truth support per scene """
    evaluation_mask = scene.evaluation_mask
    for class_id, spec in enumerate(scene.classes):
        store.append("classes", {
            "class_id": f"{scene.dataset}:{class_id}",
            "dataset": scene.dataset,
            "class_name": spec.name,
            "detector_name": spec.name_by_detector,
            "detector_stored_id": spec.detector_stored_id,
        })

        # Ground-truth vertex support of this class in this scene
        class_mask = scene.semantic_labels == class_id
        store.append("scene_classes", {
            "scene_id": scene.scene_id,
            "class_id": f"{scene.dataset}:{class_id}",
            "gt_vertex_count": int(class_mask.sum()),
            "gt_visible_vertex_count": int((class_mask & scene.visible).sum()),
            "gt_evaluated_vertex_count": int((class_mask & evaluation_mask).sum()),
        })


def record_source_analytics(store, run_id, scene, result):
    """
    Record votes, Gaussian counts and metrics for one mask source

    The vote statistics of the result come from the JSON written by the accumulation container
    """
    common = {
        "run_id": run_id,
        "variant": result["variant"],
        "scene_id": scene.scene_id,
        "source": result["mask_source"],
    }
    gamma = result["parameters"]["hysteresis_gamma"]

    for name, statistics in result["vote_statistics"].items():
        store.append("vote_statistics", {
            **common, **statistics,
            "vote_id": result["vote_id"],
            "class_id": f"{scene.dataset}:{scene.class_id(name)}",
        })

    for name, item in result["per_class"].items():
        for sweep in item["sweep"].values():
            prediction = sweep["iou"]
            reference = sweep["ground_truth_transfer_iou"]
            store.append("class_beta_metrics", {
                **common,
                "class_id": f"{scene.dataset}:{scene.class_id(name)}",
                "beta": sweep["beta"],
                "hysteresis_gamma": gamma,
                "gaussian_count": sweep["gaussian_count"],
                **prediction,
                **{f"ground_truth_transfer_{key}": value for key, value in reference.items()},
                "relative_iou": sweep["relative_iou"],
            })

    for beta_key, aggregate in result["metrics_by_beta"].items():
        store.append("aggregate_beta_metrics", {
            **common,
            "beta": float(beta_key),
            "hysteresis_gamma": gamma,
            **aggregate,
        })
