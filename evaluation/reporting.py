# JSON reports for evaluation results

import json

from .common import atomic_write_text, ensure_dir


def write_result(results_dir, result):
    """ Write one source result as JSON """
    ensure_dir(results_dir)
    tag = result["mask_source"]  # Either "yolo" or "gt2d"
    json_path = results_dir / f"results_{tag}.json"
    atomic_write_text(json_path, json.dumps(result, indent=2, default=str) + "\n")
