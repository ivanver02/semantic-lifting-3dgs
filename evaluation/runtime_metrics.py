# Run a Python stage and persist its CUDA peak memory

import argparse
import json
import runpy
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

# Running this file as a script only puts evaluation/ on the path, and the
# stages import the repository packages, so the repository root is added too
sys.path.append(str(Path(__file__).resolve().parents[1]))


def _parser():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--script")
    group.add_argument("--module")
    parser.add_argument("--metrics-path", required=True, type=Path)
    parser.add_argument("stage_args", nargs=argparse.REMAINDER)
    return parser


def _cuda_available():
    return torch is not None and torch.cuda.is_available()


def main():
    args = _parser().parse_args()
    stage_args = list(args.stage_args)
    if stage_args[:1] == ["--"]:
        stage_args = stage_args[1:]

    # Initialize CUDA and reset allocation counters on every device
    if _cuda_available():
        torch.cuda.init()
        for device_index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device_index)

    try:
        sys.argv = [args.script or args.module] + stage_args
        if args.script:
            runpy.run_path(args.script, run_name="__main__")
        else:
            runpy.run_module(args.module, run_name="__main__")

    finally:
        # Keep the largest allocated peak across visible devices
        peak = None
        if _cuda_available():
            peak = max(
                torch.cuda.max_memory_allocated(device_index)
                for device_index in range(torch.cuda.device_count())
            )
        args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_path.write_text(json.dumps({"peak_cuda_memory_bytes": peak}), encoding="utf-8")


if __name__ == "__main__":
    main()
