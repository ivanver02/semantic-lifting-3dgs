# The workflow that evaluates both Scannet++ and Replica datasets

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from . import metrics, transfer
from .analytics import (
    AnalyticsStore,
    collect_run_metadata,
    record_class_inventory,
    record_source_analytics,
    utc_now,
)
from .common import (
    atomic_write,
    digest,
    safe_name,
    target_classes_by_detector,
    selection_path,
    vote_dir,
    vote_id,
    vote_path,
)
from .runtime import Runtime
from .replica.scene import ReplicaScene
from .scannetpp.scene import ScannetScene


DEFAULT_DATA_ROOT = Path("/mnt/hddb/dataTFGIvanVerdugo")

# Protocol values frozen in the manuscript configuration (Table of the
# experimental design). They are not command line options on purpose: changing
# them means changing the documented experiment.
YOLO_CONF = 0.75
RASTER_BLOCK_SIZE = 16

# The configuration fields whose results belong to one variant
VARIANT_DEFAULTS = {
    "hysteresis_gamma": 0.8,
    "hysteresis_radius": 0.05,
    "tau": 0.05,
    "min_fraction": 0.5,
    "gaussian_to_mesh_background_competes": True,
    "mesh_to_gaussian_background_competes": True,
    "gaussian_to_mesh_transfer": "radius_vote",
    "opacity_weighting": True,
    "min_opacity": 0.1,
    "background_confidence": 0.25,
    "background_view_policy": "target_views",
}


def _progress(message):
    """ Print a progress message immediately, even when stdout is buffered """
    print(f"progress: {message}", flush=True)


def _measure_stage(stage_records, runtime, name, function, computed=True):
    """
    Run one stage and retain elapsed time plus container CUDA peak memory

    computed is False when every output of the stage is already in the cache,
    and then the stage is recorded as a cache hit without running
    """
    started = time.perf_counter()
    result = function() if computed else None
    stage_records.append({
        "stage": name,
        "cache_mode": "miss" if computed else "hit",
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": runtime.end_stage(),
    })
    return result


def _parser():
    """ Build the parser for the evaluation workflow """
    parser = argparse.ArgumentParser(description=__doc__)

    # Identify the dataset and scene that will be evaluated
    parser.add_argument("--dataset", choices=["replica", "scannetpp"], required=True)
    parser.add_argument("--scene", required=True)

    # Define the paths used by the launcher and by the Docker mounts
    parser.add_argument("--data-root", type=Path, default=None, help="dataset path root, something like .../scannetpp")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--variant", default=None,
                        help="Result variant identity, defaults to a configuration digest")
    parser.add_argument("--model-root", type=Path, default=None, help="Gaussian model directory to reuse, if exists")

    # Select the source of the 2D masks
    parser.add_argument("--mask-source", choices=["yolo", "gt2d", "both"], default="yolo")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--resolution", type=int, default=None,
        help="training image scale: 1 is original, 2 is half width and height")
    parser.add_argument("--train-data-device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--vote-data-device", choices=["cuda", "cpu"], default="cpu")

    # Configure threshold selection and transfer
    parser.add_argument("--hysteresis-gamma", type=float,
                        default=VARIANT_DEFAULTS["hysteresis_gamma"])
    parser.add_argument("--hysteresis-radius", type=float,
                        default=VARIANT_DEFAULTS["hysteresis_radius"])
    parser.add_argument("--background-confidence", type=float,
        default=VARIANT_DEFAULTS["background_confidence"],
        help="Confidence assigned to pixels with semantic label zero")
    parser.add_argument(
        "--background-view-policy", choices=["target_views", "all_views"],
        default=VARIANT_DEFAULTS["background_view_policy"],
        help="Use only views containing target pixels or every matched view",
    )
    parser.add_argument("--betas", nargs="+", type=float, required=True,
        help="Beta values to evaluate for every target class")
    parser.add_argument("--tau", type=float, default=VARIANT_DEFAULTS["tau"])
    parser.add_argument("--min-fraction", type=float, default=VARIANT_DEFAULTS["min_fraction"])
    parser.add_argument(
        "--gaussian-to-mesh-transfer",
        choices=["radius_vote", "nearest_neighbor_label"],
        default=VARIANT_DEFAULTS["gaussian_to_mesh_transfer"],
    )
    parser.add_argument("--min-opacity", type=float,
                        default=VARIANT_DEFAULTS["min_opacity"])

    # Background competition and opacity weighting switches used by ablations
    parser.add_argument(
        "--no-gaussian-to-mesh-background-competes",
        dest="gaussian_to_mesh_background_competes",
        action="store_false",
        help="Disable background competition in predicted mesh labels",
    )
    parser.add_argument("--no-mesh-to-gaussian-background-competes", dest="mesh_to_gaussian_background_competes", action="store_false",
        help="Do not use background votes when assigning GT labels to Gaussians")
    parser.add_argument("--no-opacity-weighting", dest="opacity_weighting", action="store_false")

    # Rebuild cached data instead of reusing files from an earlier run
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--save_results_to_csv", action="store_true", default=False,
        help="Append validation results and summaries to the analytics directory beside the data root")
    return parser


def run_parameters(args, data_root):
    """ Prepare the full parameter record stored with the results and the analytics """
    return {
        "dataset": args.dataset,
        "scene": args.scene,
        "split": args.split,
        "data_root": str(data_root),
        "iterations": args.iterations,
        "resolution": args.resolution,
        "train_data_device": args.train_data_device,
        "vote_data_device": args.vote_data_device,
        "raster_block_size": RASTER_BLOCK_SIZE,
        "betas": list(args.betas),
        **{key: getattr(args, key) for key in VARIANT_DEFAULTS},
    }


def _source_names(mask_source):
    """ Determine the list of mask sources """
    if mask_source == "both":
        return ["yolo", "gt2d"]
    return [mask_source]


def _pending_sources(results_dir, sources, parameters, force):
    """
    Return sources whose result JSON does not exist yet

    Existing results are only reused when they were computed with the same
    parameters, so one variant never mixes two configurations
    """
    pending = []
    for source in sources:
        path = results_dir / f"results_{source}.json"
        if force or not path.exists():
            pending.append(source)
        elif json.loads(path.read_text())["parameters"] != parameters:
            raise RuntimeError(
                f"{path} was computed with other parameters. Use another "
                "--variant, or pass --force to discard those results."
            )
    return pending


def _resolve_model_dir(args, data_root, output_root):
    """
    Determine the Gaussian model directory to use for evaluation

    --model-root reuses an explicit model; otherwise an already trained model
    is reused when it exists, and only then does training write into the run's
    own output directory.
    """
    def has_model(model_dir):
        return (model_dir / "point_cloud" / f"iteration_{args.iterations}" / "point_cloud.ply").exists()

    if args.model_root is not None:
        if not has_model(args.model_root.resolve()):
            raise FileNotFoundError(
                f"Gaussian model missing for iteration {args.iterations}: {args.model_root}"
            )
        return args.model_root.resolve()

    # Reuse an existing Gaussian model from earlier training when it exists
    if args.dataset == "replica":
        conventional_model = data_root / args.scene / "eval_output" / "gs_model"
    else:
        conventional_model = args.repo_root / "output" / args.scene
    output_model = output_root / "model"
    if not has_model(output_model) and has_model(conventional_model):
        print(f"model: Using existing Gaussian model: {conventional_model}")
        return conventional_model
    return output_model


def _is_prepared(dataset_dir):
    """ A prepared dataset holds a COLMAP model in sparse/0, in binary or text form """
    sparse = dataset_dir / "sparse" / "0"
    return (sparse / "points3D.bin").exists() or (sparse / "points3D.txt").exists()


def _mask_classes(mask_dir, classes):
    """Select target class records present in a generated mask directory.

    classes is the collection of TargetClassInfo supported by the scene.
    classes.json maps stored detector IDs to detector names:

    {
        "73": "refrigerator",
        "63": "tv"
    }

    The returned list contains only records whose detector name appears in the
    mask metadata.
    """
    classes_path = mask_dir / "classes.json"
    if not classes_path.exists():
        raise FileNotFoundError(f"mask class metadata not found: {classes_path}")

    # Read detector names from classes json
    names = set(json.loads(classes_path.read_text()).values())

    # Map detector names to main class records and keep only supported classes
    mapping = target_classes_by_detector(classes)
    selected = []
    for name in sorted(names):
        spec = mapping.get(name)
        if spec is not None:
            selected.append(spec)
    return selected


def _classes_with_gt2d_views(mask_dir, classes):
    """ Return classes that occur in at least one generated GT2D view """
    present_ids = set()
    for path in (mask_dir / "semantic").glob("*.png"):
        semantic = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if semantic is not None:
            present_ids.update(np.unique(semantic).tolist())
    return [
        spec for spec in classes
        if spec.detector_stored_id in present_ids
    ]


def _generate_yolo_masks(args, runtime, dataset_dir, output_dir):
    """ Run the detector in the lifting container on the prepared dataset images """
    runtime.run_lifting(
        "segmentation/generate_mask.py",
        [
            "--images_dir", dataset_dir / "images",
            "--output_root", output_dir,
            "--model", runtime.repo_root / "yolo26x-seg.pt",
            "--conf", YOLO_CONF,
        ],
    )


def _run_votes(args, runtime, dataset_dir, model_dir, mask_dir, segmentation_dir, classes, identifier):
    """
    Accumulate 2D votes for every target class present in the source masks.

    classes contains only classes that the source can represent in its mask
    metadata and whose votes are not cached yet. Classes absent from the source
    are handled as empty predictions by the evaluation stage.
    """
    for spec in classes:
        # Each selected main class, identified here by its detector name,
        # receives its own vote directory and cache file
        runtime.run_lifting(
            "segmentation/accumulate_votes.py",
            [
                "--model_path", model_dir,
                "--source_path", dataset_dir,
                "--mask_dir", mask_dir,
                "--output_path", vote_path(segmentation_dir, spec, identifier),
                "--target_class", spec.name_by_detector,
                "--loaded_iter", args.iterations,
                "--raster_block_size", RASTER_BLOCK_SIZE,
                "--data_device", args.vote_data_device,
                "--background_confidence", args.background_confidence,
                "--background_view_policy", args.background_view_policy,
            ],
        )


def _run_thresholds(args, runtime, model_dir, segmentation_dir, classes, identifier):
    """ Select the Gaussians of every class and beta value in one container """
    runtime.run_lifting(
        "segmentation/threshold_labels.py",
        [
            "--model_path", model_dir,
            "--loaded_iter", args.iterations,
            "--hysteresis_gamma", args.hysteresis_gamma,
            "--hysteresis_radius", args.hysteresis_radius,
            "--beta", *args.betas,
            "--votes", *[vote_path(segmentation_dir, spec, identifier) for spec in classes],
        ],
    )


def _transfer_to_mesh(args, neighbors, scene, selected, opacity):
    """ Transfer a binary Gaussian selection to the mesh vertices with the configured operator """
    return transfer.predict_vertex_labels(
        neighbors, len(scene.vertices), selected, opacity, args.min_fraction,
        args.opacity_weighting, args.min_opacity,
        args.gaussian_to_mesh_background_competes, args.gaussian_to_mesh_transfer,
    )


def _ground_truth_transfer(args, scene, classes, full_xyz, full_opacity, reference_dir):
    """
    Build the ground-truth transfer reference of every class

    The mesh annotation reaches the Gaussians through the same radius vote that
    carries the prediction back to the mesh, and then travels back to the mesh.
    The reference Gaussians of every class are saved for the qualitative renderer.
    Returns the vertex and Gaussian neighborhoods and the reference metrics by class.
    """
    neighbors = transfer.build_radius_neighbors(scene.vertices, full_xyz, args.tau)
    gaussian_labels = transfer.ground_truth_gaussian_labels(
        neighbors, len(full_xyz), scene, args.min_fraction,
        args.mesh_to_gaussian_background_competes,
    )

    references = {}
    reference_dir.mkdir(parents=True, exist_ok=True)
    for spec in classes:
        class_id = scene.class_id(spec.name)
        selected = gaussian_labels == class_id
        np.save(reference_dir / f"{safe_name(spec.name_by_detector)}.npy", np.flatnonzero(selected))
        references[spec.name] = metrics.class_iou(
            _transfer_to_mesh(args, neighbors, scene, selected, full_opacity), scene, class_id,
        )
    return neighbors, references


def _evaluate_source(args, scene, classes, vote_classes, neighbors, full_opacity,
                     references, segmentation_dir, identifier):
    """
    Evaluate one mask source for every class and beta

    Returns the per class sweeps and the aggregate metrics of each beta
    """
    per_class = {}
    for spec in classes:
        class_id = scene.class_id(spec.name)
        reference = references[spec.name]

        sweep = {}
        for beta in args.betas:

            # A class without votes represents an empty prediction for this
            # class and beta, so its Ground Truth instances still contribute
            # false negatives
            selected = np.zeros(len(full_opacity), dtype=bool)
            if spec in vote_classes:
                selected[np.load(selection_path(
                    vote_dir(segmentation_dir, spec, identifier),
                    args.hysteresis_gamma, args.hysteresis_radius, beta,
                ))] = True

            # Evaluate the predicted Gaussian mesh including empty predictions
            prediction = metrics.class_iou(
                _transfer_to_mesh(args, neighbors, scene, selected, full_opacity), scene, class_id,
            )
            sweep[str(beta)] = {
                "beta": beta,
                "gaussian_count": int(selected.sum()),
                "iou": prediction,
                "ground_truth_transfer_iou": reference,
                "relative_iou": (
                    prediction["iou"] / reference["iou"] if reference["iou"] > 0 else None
                ),
            }

        # Store the complete beta sweep for this class
        per_class[spec.name] = {
            "name_by_detector": spec.name_by_detector,
            "sweep": sweep,
        }

    # Aggregate each requested beta independently
    metrics_by_beta = {
        str(beta): metrics.aggregate({
            name: item["sweep"][str(beta)] for name, item in per_class.items()
        })
        for beta in args.betas
    }
    return per_class, metrics_by_beta


def main():
    """ Run preparation, mask generation, voting, thresholding and evaluation """
    args = _parser().parse_args()

    # Validate the operating point ranges used by the host side calculations
    if any(beta < 0.0 or beta > 1.0 for beta in args.betas):
        raise ValueError("all --betas must be in [0, 1]")
    if args.tau <= 0:
        raise ValueError("--tau must be greater than zero")
    if not 0.0 <= args.min_fraction <= 1.0:
        raise ValueError("--min-fraction must be in [0, 1]")
    if not 0.0 <= args.hysteresis_gamma < 1.0:
        raise ValueError("--hysteresis-gamma must be in [0, 1)")
    if args.hysteresis_radius <= 0.0:
        raise ValueError("--hysteresis-radius must be greater than zero")

    # Resolve the data root and output root directories
    data_root = (
        args.data_root
        if args.data_root is not None
        else DEFAULT_DATA_ROOT / args.dataset
    ).resolve()

    output_root = (
        args.output_root
        if args.output_root is not None
        else data_root / "evaluation" / args.scene
    ).resolve()

    # Check if the output root is within the data root
    try:
        output_root.relative_to(data_root)
    except ValueError:
        raise ValueError("--output-root must be inside --data-root")

    # Resolve training defaults before checking existing reports
    if args.resolution is None:
        args.resolution = 2 if args.dataset == "scannetpp" else 1
    if args.train_data_device is None:
        args.train_data_device = "cpu" if args.dataset == "scannetpp" else "cuda"

    parameters = run_parameters(args, data_root)
    variant = args.variant or "v" + digest({key: parameters[key] for key in VARIANT_DEFAULTS})
    identifier = vote_id(parameters)
    results_dir = output_root / "results" / variant

    # Decide which mask sources still need a run
    pending_sources = _pending_sources(
        results_dir, _source_names(args.mask_source), parameters, args.force,
    )
    if not pending_sources:
        print("skip: All requested sources already have results")
        return

    run_id = str(uuid.uuid4())
    analytics_store = (
        AnalyticsStore(data_root.parent / "analytics")
        if args.save_results_to_csv else None
    )
    run_metadata = (
        collect_run_metadata(args.repo_root, sys.argv)
        if analytics_store is not None else {}
    )
    run_started = time.perf_counter()
    stage_records = []

    # Initialize the Docker runtime and create the selected dataset scene
    runtime = Runtime(args.repo_root, data_root)
    scene_type = ReplicaScene if args.dataset == "replica" else ScannetScene
    scene_instance = scene_type(data_root, args.scene, output_root)

    # Replica writes its COLMAP model into the run directory, while Scannet++
    # keeps an undistorted one beside the scene
    dataset_dir = scene_instance.prepared_dir
    _measure_stage(
        stage_records, runtime, "prepare_dataset",
        lambda: scene_instance.prepare_dataset(runtime),
        computed=not _is_prepared(dataset_dir),
    )
    model_dir = _resolve_model_dir(args, data_root, output_root)

    # Generate reference masks; they are always produced or hit because they
    # define which instances are observable and therefore evaluable
    # Every mask stage writes classes.json last, so it marks complete masks
    mask_dirs = {"gt2d": scene_instance.masks_dir, "yolo": output_root / "masks_yolo"}
    _measure_stage(
        stage_records, runtime, "generate_gt_masks",
        lambda: scene_instance.generate_gt_masks(runtime),
        computed=args.force or not (mask_dirs["gt2d"] / "classes.json").exists(),
    )
    if "yolo" in pending_sources:
        _measure_stage(
            stage_records, runtime, "generate_yolo_masks",
            lambda: _generate_yolo_masks(args, runtime, dataset_dir, mask_dirs["yolo"]),
            computed=args.force or not (mask_dirs["yolo"] / "classes.json").exists(),
        )

    # Load scene data and train when no model exists
    scene = scene_instance.load_data()
    evaluation_classes = _classes_with_gt2d_views(mask_dirs["gt2d"], scene.classes)
    model_ply = model_dir / "point_cloud" / f"iteration_{args.iterations}" / "point_cloud.ply"
    if not model_ply.exists():
        _measure_stage(
            stage_records, runtime, "train_gaussians",
            lambda: runtime.run_train(
                dataset_dir, model_dir, args.iterations,
                args.resolution, args.train_data_device,
            ),
        )
    if not model_ply.exists():
        raise FileNotFoundError(f"trained Gaussian model missing: {model_ply}")

    # The clean-label reference per class is shared by every mask source
    full_xyz, full_opacity = transfer.load_gaussian_ply(model_ply)
    neighbors, references = _measure_stage(
        stage_records, runtime, "ground_truth_transfer",
        lambda: _ground_truth_transfer(
            args, scene, evaluation_classes, full_xyz, full_opacity, results_dir / "reference",
        ),
    )

    # Process every source that still needs results
    results = {}
    for source in pending_sources:
        _progress(f"Evaluation {source}: {len(evaluation_classes)} classes, {len(args.betas)} beta value(s)")
        segmentation_dir = output_root / "segmentation" / source

        # Only source classes absent from its mask metadata are excluded from vote generation
        vote_classes = _mask_classes(mask_dirs[source], evaluation_classes)

        # Accumulate votes for the classes whose votes are not cached yet
        missing_votes = [
            spec for spec in vote_classes
            if args.force
            or not vote_path(segmentation_dir, spec, identifier).exists()
            or not (vote_dir(segmentation_dir, spec, identifier) / "vote_statistics.json").exists()
        ]
        _measure_stage(
            stage_records, runtime, f"{source}:votes",
            lambda: _run_votes(
                args, runtime, dataset_dir, model_dir, mask_dirs[source],
                segmentation_dir, missing_votes, identifier,
            ),
            computed=bool(missing_votes),
        )

        # Threshold the votes and select the Gaussians of every beta
        missing_selections = [
            spec for spec in vote_classes
            if args.force or not all(
                selection_path(
                    vote_dir(segmentation_dir, spec, identifier),
                    args.hysteresis_gamma, args.hysteresis_radius, beta,
                ).exists()
                for beta in args.betas
            )
        ]
        _measure_stage(
            stage_records, runtime, f"{source}:threshold_hysteresis",
            lambda: _run_thresholds(
                args, runtime, model_dir, segmentation_dir, missing_selections, identifier,
            ),
            computed=bool(missing_selections),
        )

        # Evaluate every beta for the selected source
        per_class, metrics_by_beta = _measure_stage(
            stage_records, runtime, f"{source}:evaluation_transfer",
            lambda: _evaluate_source(
                args, scene, evaluation_classes, vote_classes, neighbors,
                full_opacity, references, segmentation_dir, identifier,
            ),
        )

        # Save the scene name, parameters, vote statistics and metrics
        results[source] = {
            "dataset": scene.dataset,
            "scene": scene.scene,
            "mask_source": source,
            "variant": variant,
            "vote_id": identifier,
            "parameters": parameters,
            "vote_statistics": {
                spec.name: json.loads(
                    (vote_dir(segmentation_dir, spec, identifier) / "vote_statistics.json").read_text()
                )
                for spec in vote_classes
            },
            "metrics_by_beta": metrics_by_beta,
            "per_class": per_class,
        }

    # Record the completed run, its parameters and its stages
    if analytics_store is not None:
        record_class_inventory(analytics_store, scene)
        for result in results.values():
            record_source_analytics(analytics_store, run_id, scene, result)
        analytics_store.append("runs", {
            "run_id": run_id,
            "created_at": utc_now(),
            "dataset": scene.dataset,
            "scene_id": scene.scene_id,
            "scene_name": scene.scene,
            "split": args.split,
            "source": args.mask_source,
            "output_root": str(output_root),
            "model_root": str(model_dir),
            "elapsed_seconds": time.perf_counter() - run_started,
            "peak_cuda_memory_bytes": max(
                (record["peak_cuda_memory_bytes"] for record in stage_records
                 if record["peak_cuda_memory_bytes"] is not None),
                default=None,
            ),
        })
        analytics_store.append("run_parameters", {
            "run_id": run_id,
            "variant": variant,
            "vote_id": identifier,
            **parameters,
            **run_metadata,
        })
        for record in stage_records:
            analytics_store.append("run_stages", {
                "run_id": run_id,
                "dataset": scene.dataset,
                "scene_id": scene.scene_id,
                **record,
            })

    # The results are written last, so a unit with results is always a complete one
    for source, result in results.items():
        atomic_write(
            results_dir / f"results_{source}.json",
            lambda path: path.write_text(json.dumps(result, indent=2) + "\n"),
        )
    _progress(f"Run finished in {time.perf_counter() - run_started:.1f}s")
    print(json.dumps({source: result["metrics_by_beta"]
                      for source, result in results.items()}, indent=2))


if __name__ == "__main__":
    main()
