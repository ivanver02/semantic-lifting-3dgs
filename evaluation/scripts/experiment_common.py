# Helpers for experiment drivers

import json
import subprocess
import sys
from pathlib import Path


# The validation beta grid of the manuscript, dense above 0.95
BETAS = (0.50, 0.70, 0.90, 0.94, 0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 0.999)
GAMMAS = (0.0, 0.5, 0.7, 0.8, 0.9)


def token(value):
    # Format a numeric value for an identifier
    return str(value).replace(".", "_")


def sources(mask_source):
    # Expand the selected mask source option
    return ["gt2d", "yolo"] if mask_source == "both" else [mask_source]


def unit_results(args, unit):
    """ Return the result files that mark one unit as complete """
    results = (
        Path(args.output_root) / unit["dataset"] / unit["scene"] /
        "results" / unit["variant"]
    )
    return [results / f"results_{source}.json" for source in sources(args.mask_source)]


def run_units(args, units, command_builder):
    """ Launch units whose result files do not exist yet """
    for unit in units:
        if all(path.exists() for path in unit_results(args, unit)):
            print(f"skip: {unit['dataset']}/{unit['scene']} ({unit['variant']})")
            continue

        # Resolve the experiment path or execute an injected test runner
        result = command_builder(args, unit)
        if result is not None:
            subprocess.run(result, check=True, cwd=str(args.repo_root))


def command(args, unit, *, betas, gamma, split, extra=(), tau=None, theta=None):
    # Build the common evaluation command
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
        "--save_results_to_csv",
    ]
    result.extend(extra)
    return result


def dump_plan(args, driver, units):
    # Print the experiment plan
    print(
        json.dumps(
            {
                "experiment": args.experiment,
                "driver": driver,
                "invocations": len(units),
                "units": units,
            },
            indent=2, sort_keys=True)
    )
