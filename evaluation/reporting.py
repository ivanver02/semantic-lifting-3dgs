# JSON reports for evaluation results

import json
from pathlib import Path

import numpy as np
from plyfile import PlyData

from .common import atomic_write_text, ensure_dir


def _stat(values, name):
    """ Return a scalar statistic from a numeric array or None """
    if values is None or len(values) == 0:
        return None
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "std": float(values.std()),

    # Add the maximum statistic
        "max": float(values.max()),
    }[name]


def gaussian_statistics(path):
    """ Read compact size and opacity statistics from a Gaussian PLY """
    path = Path(path)
    if not path.exists():
        return {"gaussian_count": 0, "file_path": str(path)}
    vertex = PlyData.read(str(path))["vertex"]
    names = set(vertex.data.dtype.names or ())
    scales = [
        np.asarray(vertex[name], dtype=np.float64)

    # Collect the available scale fields
        for name in ("scale_0", "scale_1", "scale_2")
        if name in names
    ]
    size = np.linalg.norm(np.column_stack(scales), axis=1) if len(scales) == 3 else None
    opacity = np.asarray(vertex["opacity"], dtype=np.float64) if "opacity" in names else None
    return {
        "gaussian_count": len(vertex),
        "size_min": _stat(size, "min"),

    # Add size and opacity statistics
        "size_mean": _stat(size, "mean"),
        "size_std": _stat(size, "std"),
        "size_max": _stat(size, "max"),
        "opacity_min": _stat(opacity, "min"),
        "opacity_mean": _stat(opacity, "mean"),
        "opacity_std": _stat(opacity, "std"),
        "opacity_max": _stat(opacity, "max"),
        "file_path": str(path),

    # Finish the result payload
    }


def write_result(results_dir, result):
    """ Write one source result as JSON """
    ensure_dir(results_dir)
    tag = result["mask_source"] # Either "yolo" or "gt2d"
    json_path = results_dir / f"results_{tag}.json"
    if json_path.exists():
        previous = json.loads(json_path.read_text())
        previous_betas = previous.get("parameters", {}).get("betas")
        current_betas = result.get("parameters", {}).get("betas")

    # Validate the beta sweep
        if previous_betas != current_betas:
            raise RuntimeError(
                f"{json_path} was computed with betas {previous_betas!r}, "
                f"but this run uses {current_betas!r}, use a distinct variant"
            )
    atomic_write_text(json_path, json.dumps(result, indent=2, default=str) + "\n")