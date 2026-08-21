# Run a Python stage and persist its CUDA peak memory

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


def _parser():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--script")
    group.add_argument("--module")
    parser.add_argument("--metrics-path", required=True, type=Path)
    parser.add_argument("stage_args", nargs=argparse.REMAINDER)
    return parser


def _reset_cuda_peaks():
    """ Initialize CUDA and reset allocation counters on every device """
    if torch is None or not torch.cuda.is_available():
        return
    
    torch.cuda.init()
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device_index)


def _cuda_peaks():
    """ Return the largest allocated and reserved peaks across visible devices """
    if torch is None or not torch.cuda.is_available():
        return None, None
    
    allocated = max(
        torch.cuda.max_memory_allocated(device_index)
        for device_index in range(torch.cuda.device_count())
    )

    # Read reserved memory peaks
    reserved = max(
        torch.cuda.max_memory_reserved(device_index)
        for device_index in range(torch.cuda.device_count())
    )
    return int(allocated), int(reserved)


def main():
    args = _parser().parse_args()
    stage_args = list(args.stage_args)
    if stage_args[:1] == ["--"]:
        stage_args = stage_args[1:]

    _reset_cuda_peaks()

    try:
        sys.argv = [args.script or args.module] + stage_args
        if args.script:
            runpy.run_path(args.script, run_name="__main__")
        else:
            runpy.run_module(args.module, run_name="__main__")

    finally:
        peak_allocated, peak_reserved = _cuda_peaks()
        args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = args.metrics_path.with_suffix(".tmp")
        
        try:
            with temporary_path.open("w", encoding="utf-8") as handle:
                json.dump({
                    "peak_cuda_memory_bytes": peak_allocated,
                    "peak_cuda_memory_reserved_bytes": peak_reserved,
                }, handle)

    # Flush the metrics file
                handle.flush()
                os.fsync(handle.fileno())

            # Replace the metrics file atomically
            os.replace(temporary_path, args.metrics_path)
        finally:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()