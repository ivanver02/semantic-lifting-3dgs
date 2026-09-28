# Helpers for experiment drivers

import json
import subprocess
import sys
from pathlib import Path


# The validation beta grid of the manuscript, dense above 0.95
BETAS = (0.50, 0.70, 0.90, 0.94, 0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 0.999)
GAMMAS = (0.0, 0.5, 0.7, 0.8, 0.9)

# Development scenes of Replica and Scannet++, never part of a validation or test summary
DEVELOPMENT_SCENES = {"office_0", "7831862f02"}


def token(value):
    # Format a numeric value for an identifier
    return str(value).replace(".", "_")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unit(dataset, scene, variant, data_root, output_root, *, betas, gamma, tau, theta,
         split="validation", mask_source="both", extra=()):
    """
    Describe one invocation of evaluation.run

    Each unit writes into output_root/dataset/scene, which must live inside its data root
    """
    return {
        "dataset": dataset, "scene": scene, "variant": variant,
        "data_root": str(data_root), "output_root": str(Path(output_root) / dataset / scene),
        "betas": list(betas), "gamma": gamma, "tau": tau, "theta": theta,
        "split": split, "mask_source": mask_source, "extra": list(extra),
    }


def evaluation_command(unit):
    # Build the evaluation command of one unit
    return [
        sys.executable, "-m",
        "evaluation.run", "--dataset", unit["dataset"],
        "--scene", unit["scene"],
        "--data-root", unit["data_root"],
        "--output-root", unit["output_root"],
        "--split", unit["split"],

        # Add mask and threshold options
        "--mask-source", unit["mask_source"],
        "--betas", *map(str, unit["betas"]),
        "--hysteresis-gamma", str(unit["gamma"]),
        "--tau", str(unit["tau"]),
        "--min-fraction", str(unit["theta"]),
        "--variant", unit["variant"],
        "--save_results_to_csv",
        *unit["extra"],
    ]


def run_units(units, repo_root):
    """ Launch units whose result files do not exist yet """
    for unit in units:
        sources = ["gt2d", "yolo"] if unit["mask_source"] == "both" else [unit["mask_source"]]
        results = Path(unit["output_root"]) / "results" / unit["variant"]
        if all((results / f"results_{source}.json").exists() for source in sources):
            print(f"skip: {unit['dataset']}/{unit['scene']} ({unit['variant']})")
            continue
        subprocess.run(evaluation_command(unit), check=True, cwd=str(repo_root))


def dump_plan(experiment, units):
    # Print the experiment plan
    print(json.dumps({"experiment": experiment, "invocations": len(units), "units": units},
                     indent=2, sort_keys=True))
