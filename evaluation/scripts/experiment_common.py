# Helpers for experiment drivers

import json
import subprocess
import sys
from pathlib import Path

from evaluation.state_store import read_state_store, unit_id


BETAS = (0.90, 0.94, 0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 0.999)
GAMMAS = (0.0, 0.5, 0.7, 0.8, 0.9)


def token(value):
    # Format a numeric value for an identifier
    return str(value).replace(".", "_")


def sources(mask_source):
    # Expand the selected mask source option
    return ["gt2d", "yolo"] if mask_source == "both" else [mask_source]


def run_units(args, units, command_builder):
    """ Launch units not marked done in the state store """
    # Refresh the state store before deciding whether to launch each unit
    for unit in units:
        identifier = unit.setdefault(
            "unit_id",
            unit_id(args.experiment, unit["dataset"], unit["scene"], unit["variant"]),
        )
        state_store = read_state_store(args.state_store)
        if state_store.is_done(identifier):
            continue

        # Resolve the experiment path or execute an injected test runner
        result = command_builder(args, unit)
        if result is not None:
            subprocess.run(result, check=True, cwd=args.repo_root)


def command(args, unit, *, betas, gamma, split, extra=(), tau=None, theta=None):
    # Build the common evaluation command and optional retry flag
    scene_root = args.output_root / unit["dataset"] / unit["scene"]
    result = [
        sys.executable, "-m",
        "evaluation.run", "--dataset", unit["dataset"],
        "--scene", unit["scene"],
        "--data-root", str(args.data_root),
        "--output-root", str(scene_root),
        "--split", split,

        # Add mask and threshold options
        "--mask-source", args.mask_source,
        "--betas", *map(str, betas),
        "--hysteresis-gamma", str(gamma),
        "--tau", str(args.tau_star if tau is None else tau),
        "--min-fraction", str(args.theta_star if theta is None else theta),
        "--variant", unit["variant"],
        "--state-store", str(args.state_store),
        "--experiment", args.experiment,
        "--save_results_to_csv",
    ]
    result.extend(extra)
    if getattr(args, "retry_failed", False):
        # Add the retry flag
        result.append("--retry-failed")
    return result


def dump_plan(args, driver, units, *, notes=None, invocation_units=None):
    # Print a serializable experiment plan
    print(
        json.dumps(
            {
                "experiment": args.experiment,
                "driver": driver,
                "units": units,
                "invocations": len(
                    units if invocation_units is None else invocation_units
                ),
                "state_store_owner": "evaluation.run",
                "state_store_reader": "read_state_store before every decision",
                **({"notes": notes} if notes else {})}, indent=2, sort_keys=True)
    )