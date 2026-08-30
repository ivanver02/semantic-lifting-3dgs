# The workflow that evaluates both Scannet++ and Replica datasets

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from . import cache, ground_truth, metrics, reporting, transfer
from .analytics import (
    AnalyticsStore,
    collect_run_metadata,
    record_class_inventory,
    record_source_analytics,
    utc_now,
)
from .common import (
    atomic_write_text,
    main_digest,
    safe_name,
    target_classes_by_detector,
    threshold_path,
    vote_class_dir,
    vote_id,
)
from .runtime import Runtime
from .replica.scene import ReplicaScene
from .scannetpp.scene import MASKS_CACHE_VERSION, ScannetScene


DEFAULT_DATA_ROOT = Path("/mnt/hddb/dataTFGIvanVerdugo")

# Protocol values frozen in the manuscript configuration (Table of the
# experimental design). They are not command line options on purpose: changing
# them means changing the documented experiment.
SEQUENCE_NAME = "Sequence_2"
FRAME_STEP = 5
REPLICA_VERTEX_LABEL_MIN_FRACTION = 0.6
REPLICA_VISIBILITY_SLOP = 0.05
SCANNETPP_MASK_BANDS = 4
YOLO_CONF = 0.75
RASTER_BLOCK_SIZE = 16

VARIANT_DEFAULTS = {
    "hysteresis_gamma": 0.8,
    "hysteresis_radius": 0.05,
    "tau": 0.05,
    "min_fraction": 0.5,
    "gaussian_to_mesh_background_competes": True,
    "mesh_to_gaussian_background_competes": True,
    "mesh_to_gaussian_transfer": "radius_vote",
    "gaussian_to_mesh_transfer": "radius_vote",
    "opacity_weighting": True,
    "min_opacity": 0.1,
    "background_mode": "confidence_weighted",
    "background_confidence": 0.25,
    "background_view_policy": "target_views",
}


def _variant_parameters(args):
    """ Return the configuration fields whose results belong to one variant """
    return {
        "hysteresis_gamma": args.hysteresis_gamma,
        "hysteresis_radius": args.hysteresis_radius,
        "tau": args.tau,
        "min_fraction": args.min_fraction,
        "gaussian_to_mesh_background_competes": args.gaussian_to_mesh_background_competes,
        "mesh_to_gaussian_background_competes": args.mesh_to_gaussian_background_competes,

    # Add transfer and weighting settings
        "mesh_to_gaussian_transfer": args.mesh_to_gaussian_transfer,
        "gaussian_to_mesh_transfer": args.gaussian_to_mesh_transfer,
        "opacity_weighting": not args.no_opacity_weighting,
        "min_opacity": args.min_opacity,
        "background_mode": args.background_mode,
        "background_confidence": args.background_confidence,
        "background_view_policy": args.background_view_policy,
    }


def _resolve_variant(args):
    """
    Resolve the result variant label

    A readable label such as frozen_g0_8 can be passed explicitly, otherwise
    the configuration digest identifies the variant on its own.
    """
    if args.variant is not None:
        return args.variant
    return "v" + main_digest(_variant_parameters(args))


def _progress(message):
    """ Print a progress message immediately, even when stdout is buffered """
    print(f"progress: {message}", flush=True)


def _artifact_stamp(path):
    """
    Identify the current state of a stage artifact.

    A stage that reuses its output leaves the file alone, so an unchanged stamp
    across the call means the work came from the cache. Reading the state
    instead of restating each reuse condition keeps the label and the caching
    rule from drifting apart, and it is the only thing that works for the
    stages whose real work happens inside a container and returns nothing.
    """
    try:
        status = path.stat()
    except OSError:
        return None
    return status.st_mtime_ns, status.st_size, status.st_ino


def _measure_stage(stage_records, name, function, artifact=None, runtime=None):
    """ Run one stage and retain elapsed time plus container CUDA peak memory """
    if runtime is not None:
        runtime.begin_stage()
    before = _artifact_stamp(artifact) if artifact is not None else None
    started = time.perf_counter()
    try:
        return function()
    finally:
        memory = runtime.end_stage() if runtime is not None else {
            "allocated": None,
            "reserved": None,
        }
        after = _artifact_stamp(artifact) if artifact is not None else None
        stage_records.append({
            "stage": name,
            "cache_mode": (
                "hit" if before is not None and after == before else "miss"
            ),
            "container_count": None,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": memory["allocated"],
            "peak_cuda_memory_reserved_bytes": memory["reserved"],
        })


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
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the source plan without initializing evaluation")
    parser.add_argument("--model-root", type=Path, default=None, help="Gaussian model directory to reuse, if exists")

    # Select the source of the 2D masks
    parser.add_argument("--mask-source", choices=["yolo", "gt2d", "both"], default="yolo")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
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
    parser.add_argument(
        "--background-mode",
        choices=["all_non_target", "explicit_background", "confidence_weighted"],
        default=VARIANT_DEFAULTS["background_mode"],
        help="How 2D non-target evidence is constructed",
    )
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
        "--mesh-to-gaussian-transfer",
        choices=["radius_vote", "nearest_neighbor_label"],
        default=VARIANT_DEFAULTS["mesh_to_gaussian_transfer"],
    )
    parser.add_argument(
        "--gaussian-to-mesh-transfer",
        choices=["radius_vote", "nearest_neighbor_label"],
        default=VARIANT_DEFAULTS["gaussian_to_mesh_transfer"],
    )
    parser.add_argument("--min-opacity", type=float,
                        default=VARIANT_DEFAULTS["min_opacity"])

    # Background competition and opacity weighting switches used by ablations
    parser.add_argument(
        "--gaussian-to-mesh-background-competes",
        dest="gaussian_to_mesh_background_competes",
        action="store_true",
        default=VARIANT_DEFAULTS["gaussian_to_mesh_background_competes"],
        help="Include background votes in predicted mesh labels",
    )
    parser.add_argument(
        "--no-gaussian-to-mesh-background-competes",
        dest="gaussian_to_mesh_background_competes",
        action="store_false",
        help="Disable background competition in predicted mesh labels",
    )
    parser.add_argument(
        "--mesh-to-gaussian-background-competes",
        dest="mesh_to_gaussian_background_competes", action="store_true",
        default=VARIANT_DEFAULTS["mesh_to_gaussian_background_competes"],
        help="Use background votes when assigning GT labels to Gaussians",
    )
    parser.add_argument("--no-mesh-to-gaussian-background-competes", dest="mesh_to_gaussian_background_competes", action="store_false",
        help="Do not use background votes when assigning GT labels to Gaussians")
    parser.add_argument("--no-opacity-weighting", action="store_true")

    # Rebuild cached data instead of reusing files from an earlier run
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--save_results_to_csv", action="store_true", default=False,
        help="Append validation results and summaries to dataTFGIvanVerdugo/analytics")

    # Frozen protocol values; they are part of the documented experiment and
    # therefore not exposed as options
    parser.set_defaults(
        sequence_name=SEQUENCE_NAME,
        frame_step=FRAME_STEP,
        replica_vertex_label_min_fraction=REPLICA_VERTEX_LABEL_MIN_FRACTION,
        replica_visibility_slop=REPLICA_VISIBILITY_SLOP,
        scannetpp_mask_version=MASKS_CACHE_VERSION,
        scannetpp_mask_bands=SCANNETPP_MASK_BANDS,
        yolo_conf=YOLO_CONF,
        raster_block_size=RASTER_BLOCK_SIZE,
    )
    return parser


def _source_names(mask_source):
    """ Determine the list of mask sources """
    if mask_source == "both":
        return ["yolo", "gt2d"]
    return [mask_source]


def _pending_sources(results_dir, sources, force):
    """ Return sources whose result JSON does not exist yet """
    return [
        source for source in sources
        if force or not (results_dir / f"results_{source}.json").exists()
    ]


def _validate_existing_beta_grids(results_dir, sources, betas, force):
    """ Reject an overwrite with a different beta sweep """
    if force:
        return
    for source in sources:
        path = results_dir / f"results_{source}.json"
        if not path.exists():
            continue
        previous = json.loads(path.read_text())
        previous_betas = previous.get("parameters", {}).get("betas")
        if previous_betas != list(betas):
            raise RuntimeError(
                f"{path} already contains results for betas {previous_betas!r}, "
                f"while this invocation uses {list(betas)!r}. The caller must "
                "read the existing results or pass --force to discard them."
            )


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


def _prepare_scene(args, scene, runtime, dataset_dir):
    """ Prepare the dataset in the format expected by training and projection """
    # Replica prepares its images and COLMAP text files locally
    if args.dataset == "replica":
        return scene.prepare_dataset(dataset_dir)

    # Scannet++ prepares its COLMAP model in the container
    elif args.dataset == "scannetpp":
        return scene.prepare_dataset(runtime)


def _generate_gt_masks(args, scene, runtime, output_dir):
    """
    Generate or reuse the dataset specific 2D masks.

    args.force controls whether existing masks are regenerated.
    """

    # Replica can generate its actual GT masks directly from the semantic image sequence
    force = args.force
    if args.dataset == "replica":
        runtime.run_lifting_module(
            "evaluation.replica.gt_masks",
            [
                "--data_root", str(scene.data_root),
                "--scene", scene.scene,
                "--sequence_name", scene.sequence.name,
                "--frame_step", str(scene.frame_step),
                "--vertex_label_min_fraction", str(scene.vertex_label_min_fraction),
                "--visibility_slop", str(scene.visibility_slop),
                "--resolution", str(args.resolution),
                "--output_dir", str(output_dir),
            ] + (["--force"] if force else []),
        )

    # Scannet++ renders its masks from the mesh through the lifting container, they will be considered our "GT"
    elif args.dataset == "scannetpp":
        scene.generate_gt_masks(
            runtime, output_dir, bands=args.scannetpp_mask_bands, force=force,
            resolution=args.resolution, mask_version=args.scannetpp_mask_version,
        )


def _generate_yolo_masks(args, runtime, dataset_dir, output_dir):
    """ Generate or reuse YOLO masks for the prepared dataset images """
    # The classes file says whether the mask directory exists
    if (output_dir / "classes.json").exists() and not args.force:
        return

    # Run the detector in the lifting container
    runtime.run_lifting(
        "segmentation/generate_mask.py",
        [
            "--images_dir", str(dataset_dir / "images"),
            "--output_root", str(output_dir),
            "--model", str(runtime.repo_root / "yolo26x-seg.pt"),
            "--conf", str(args.yolo_conf),
        ],
    )


def _export_gt_gaussians(args, runtime, model_dir, gt_dir,
                         segmentation_dir, scene, classes):
    """ Export the reference transferred Gaussians into each source class directory """
    class_specs = [
        f"{safe_name(spec.name_by_detector)}:{scene.class_id(spec.name)}"
        for spec in classes
    ]
    if not class_specs:
        return
    runtime.run_lifting(
        "segmentation/export_gt_gaussians.py",
        [
            "--model_path", str(model_dir),
            "--gt_labels_path", str(gt_dir / "gt_gaussian_labels.npz"),
            "--output_dir", str(segmentation_dir),
            "--loaded_iter", str(args.iterations),
        ] + sum((["--class_spec", item] for item in class_specs), []),
    )


def _run_votes(args, runtime, dataset_dir, model_dir, mask_dir,
               segmentation_dir, classes,
               save_statistics=False, vote_identifier=None):
    """
    Accumulate 2D votes for every target class present in the source masks.

    classes contains only classes that the source can represent in its mask
    metadata. Classes absent from the source are handled as empty predictions
    by the evaluation stage. Existing vote files are hit unless args.force
    is true.
    """
    launched = 0
    for spec in classes:
        # Each selected main class, identified here by its detector name,
        # receives its own vote directory and cache file
        vote_identifier = vote_identifier or vote_id(vars(args))
        class_dir = vote_class_dir(segmentation_dir, spec, vote_identifier)
        safe = safe_name(spec.name_by_detector)
        vote_path = class_dir / f"voting_data_{safe}.pt"
        statistics_path = class_dir / "vote_statistics.json"
        if (vote_path.exists() and
                not args.force and
                (not save_statistics or statistics_path.exists())):
            continue

        # Accumulate votes for this main class using masks whose pixels contain
        # stored detector IDs
        command = [
            "--model_path", str(model_dir),
            "--mask_dir", str(mask_dir),
            "--output_dir", str(segmentation_dir),
            "--vote_id", vote_identifier,
            "--target_class", spec.name_by_detector,
            "--loaded_iter", str(args.iterations),
            "--raster_block_size", str(args.raster_block_size),
            "--source_path", str(dataset_dir),
            "--data_device", str(args.vote_data_device),
        ]
        if save_statistics:
            command += [
                "--statistics_path",
                str(statistics_path),
            ]

        runtime.run_lifting(
            "segmentation/accumulate_votes.py", command + [
                "--background_mode", str(args.background_mode),
                "--background_confidence", str(args.background_confidence),
                "--background_view_policy", str(args.background_view_policy),
            ],
        )
        launched += 1

    return launched


def _run_thresholds(args, runtime, model_dir, segmentation_dir, classes,
                    vote_identifier=None):
    """
    Create labeled Gaussian files for every class and beta value

    The returned tuple contains the beta values used and container count.
    Existing labeled files are hit unless args.force is true.
    """

    betas = list(args.betas)
    pending = []
    for spec in classes:
        # Start thresholding after vote accumulation
        vote_identifier = vote_identifier or vote_id(vars(args))
        safe = safe_name(spec.name_by_detector)
        vote_path = vote_class_dir(segmentation_dir, spec, vote_identifier) / (
            f"voting_data_{safe}.pt"
        )
        if not vote_path.exists():
            continue
        if any(not threshold_path(
                segmentation_dir, spec, vote_identifier,
                args.hysteresis_gamma, args.hysteresis_radius, beta,
        ).exists() for beta in betas) or args.force:
            pending.append((spec, vote_path))
    if not pending:
        return betas, 0
    command = [
        "--model_path", str(model_dir),
        "--output_dir", str(segmentation_dir),
        "--cache_dir", str(segmentation_dir),
        "--beta", *[str(beta) for beta in betas],
        "--loaded_iter", str(args.iterations),
        "--hysteresis_gamma", str(args.hysteresis_gamma),
        "--hysteresis_radius", str(args.hysteresis_radius),
    ]
    for spec, vote_path in pending:
        command += [
            "--class_spec", json.dumps({
                "target_class": spec.name_by_detector,
                "voting_data_path": str(vote_path),
                "class_output_dir": str(
                    vote_class_dir(segmentation_dir, spec, vote_identifier)
                ),
            }, separators=(",", ":")),
        ]

    if args.force:
        command.append("--force")
    runtime.run_lifting("segmentation/threshold_labels.py", command)
    return betas, 1


def _evaluate_scene(args, scene, gaussians_near_a_vertex, gaussian_labels,
                      full_xyz, full_opacity,
                      classes,
                       available_classes,
                       ground_truth_transfer_by_class,
                       vote_identifier,
                       variant,
                       segmentation_dir, betas, results_dir, source):
    """
    Evaluate one mask source and write its JSON results

    betas is the beta threshold grid
    """

    per_class = {}
    per_class_by_beta = {}
    scene_started = time.perf_counter()
    _progress(
        f"Evaluation {source}: {len(classes)} classes, "
        f"{len(betas)} beta value(s)"
    )
    available_names = {spec.name for spec in available_classes}

    for spec in classes:
        ground_truth_transfer_metrics = ground_truth_transfer_by_class[spec.name]

        sweep = {}
        for beta_index, beta in enumerate(betas, start=1):

            # A missing labeled file represents an empty prediction for this
            # class and beta, so its Ground Truth instances still contribute to false negatives
            path = threshold_path(
                segmentation_dir, spec, vote_identifier,
                args.hysteresis_gamma, args.hysteresis_radius, beta,
            )

            if spec.name not in available_names or not path.exists():
                predicted_xyz = np.empty((0, 3), dtype=np.float64)
            else:
                predicted_xyz, _ = transfer.load_gaussian_ply(path)

            # Evaluate the predicted Gaussian mesh including empty predictions
            result = metrics.evaluate_class(
                scene, gaussians_near_a_vertex, gaussian_labels, full_xyz,
                full_opacity, spec,
                predicted_xyz, args.tau, args.min_fraction,
                not args.no_opacity_weighting, args.min_opacity,
                args.gaussian_to_mesh_background_competes,
                args.gaussian_to_mesh_transfer, ground_truth_transfer_metrics,
            )

            score = result["iou"]["iou"]
            sweep[str(beta)] = {
                "beta": beta,
                "iou": result["iou"],
                "ground_truth_transfer_iou": result["ground_truth_transfer_iou"],
                "relative_iou": (
                    result["iou"]["iou"] /
                    result["ground_truth_transfer_iou"]["iou"]
                    if result["ground_truth_transfer_iou"]["iou"] else 0.0
                ),
                "score": score,
            }
            per_class_by_beta.setdefault(str(beta), {})[spec.name] = result

        # Store the complete beta sweep for this class
        per_class[spec.name] = {
            "name_by_detector": spec.name_by_detector,
            "sweep": sweep,
        }

    # Aggregate each requested beta independently
    metrics_by_beta = {
        beta: metrics.aggregate(beta_classes)
        for beta, beta_classes in per_class_by_beta.items()
    }

    # Save the scene name, evaluation masks, parameters and metrics to JSON
    result = {
        "dataset": scene.dataset,
        "scene": scene.scene,
        "mask_source": source,
        "variant": variant,
        "support": {
            "vertices_evaluated": int(scene.evaluation_mask.sum()),
        },
        "parameters": {
            "hysteresis_gamma": args.hysteresis_gamma,
            "hysteresis_radius": args.hysteresis_radius,
            "background_mode": args.background_mode,
            "background_confidence": args.background_confidence,
            "background_view_policy": args.background_view_policy,
            "betas": betas,
            "tau": args.tau,
            "min_fraction": args.min_fraction,
            "mesh_to_gaussian_transfer": args.mesh_to_gaussian_transfer,
            "gaussian_to_mesh_transfer": args.gaussian_to_mesh_transfer,
            "opacity_weighted": not args.no_opacity_weighting,
            "gaussian_to_mesh_background_competes": args.gaussian_to_mesh_background_competes,
            "mesh_to_gaussian_background_competes": args.mesh_to_gaussian_background_competes,
        },
        "metrics_by_beta": metrics_by_beta,
        "per_class": per_class,
    }
    reporting.write_result(results_dir, result)
    _progress(
        f"Evaluation {source} finished in "
        f"{time.perf_counter() - scene_started:.1f}s"
    )
    return result


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
    if args.hysteresis_gamma < 0.0:
        raise ValueError("--hysteresis-gamma must be non-negative")
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

    variant = _resolve_variant(args)
    results_dir = output_root / "results" / variant
    parameters = cache.run_parameters(args, data_root)
    vote_identifier = vote_id(parameters)
    requested_sources = _source_names(args.mask_source)
    _validate_existing_beta_grids(
        results_dir, requested_sources, args.betas, args.force,
    )

    # Decide which mask sources still need a run
    pending_sources = _pending_sources(results_dir, requested_sources, args.force)
    if args.dry_run:
        skipped = [source for source in requested_sources if source not in pending_sources]
        print(json.dumps({"run": False, "pending": pending_sources, "skipped": skipped}))
        return

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
    masks_gt = output_root / "masks_gt2d"
    scene_instance = _make_scene(args, data_root, masks_gt)

    # Prepare metadata contracts and reuse caches within this output root
    cache.prepare_run_metadata(output_root, parameters, args.force, pending_sources)

    # Replica writes its COLMAP model into the run directory, while Scannet++ ignores it and keeps an undistorted one beside the scene
    # Therefore, the artifact of dataset preparation does not live in the same place for the two
    dataset_dir = output_root / "dataset"
    prepared_root = (
        dataset_dir if args.dataset == "replica" else scene_instance.prepared_dir
    )
    dataset_dir = _measure_stage(
        stage_records, "prepare_dataset",
        lambda: _prepare_scene(args, scene_instance, runtime, dataset_dir),
        artifact=prepared_root / "sparse" / "0",
        runtime=runtime,
    )
    model_dir = cache.resolve_model_dir(args, data_root, output_root)

    # Generate reference masks: they are always produced or hit because they define which instances are observable and therefore evaluable
    _measure_stage(
        stage_records, "generate_gt_masks",
        lambda: _generate_gt_masks(args, scene_instance, runtime, masks_gt),
        artifact=masks_gt / "classes.json",
        runtime=runtime,
    )
    if "yolo" in pending_sources:
        masks_yolo = output_root / "masks_yolo"
        _measure_stage(
            stage_records, "generate_yolo_masks",
            lambda: _generate_yolo_masks(args, runtime, dataset_dir, masks_yolo),
            artifact=masks_yolo / "classes.json",
            runtime=runtime,
        )

    # Load scene data and train when no model exists
    scene = scene_instance.load_data()
    if analytics_store is not None:
        record_class_inventory(analytics_store, scene, f"{scene.dataset}:{scene.scene}")
    evaluation_classes = _classes_with_gt2d_views(masks_gt, scene.classes)
    model_ply = model_dir / "point_cloud" / f"iteration_{args.iterations}" / "point_cloud.ply"
    if not model_ply.exists():
        _measure_stage(
            stage_records, "train_gaussians",
            lambda: runtime.run_train(
                dataset_dir, model_dir, args.iterations,
                args.resolution, args.train_data_device,
            ),
            runtime=runtime,
        )
    if not model_ply.exists():
        raise FileNotFoundError(f"trained Gaussian model missing: {model_ply}")

    # Build or reuse transfer neighborhoods and labels
    segmentation_root = output_root / "segmentation"
    gt_dir = output_root / "gt"
    gaussians_near_a_vertex, gaussian_labels = _measure_stage(
        stage_records, "ground_truth_transfer",
        lambda: ground_truth.build(
            scene, model_ply, gt_dir, args.tau, args.min_fraction,
            args.mesh_to_gaussian_background_competes,
            args.mesh_to_gaussian_transfer, args.force,
            evaluation_scope_version=parameters["evaluation_scope_version"],
        ),
        artifact=gt_dir / "gt_gaussian_labels.npz",
        runtime=runtime,
    )
    full_xyz, full_opacity = transfer.load_gaussian_ply(model_ply)

    # The ground truth reference per class is shared by every mask source
    ground_truth_transfer_by_class = {}
    for spec in evaluation_classes:
        reference = metrics.evaluate_class(
            scene, gaussians_near_a_vertex, gaussian_labels, full_xyz,
            full_opacity, spec, None,
            args.tau, args.min_fraction, not args.no_opacity_weighting,
            args.min_opacity, args.gaussian_to_mesh_background_competes,
            args.gaussian_to_mesh_transfer,
        )
        ground_truth_transfer_by_class[spec.name] = reference[
            "ground_truth_transfer_iou"
        ]

    # Process every source that still needs results
    scene_results = {}
    results_dir_for = {
        source: results_dir for source in pending_sources
    }
    for source in pending_sources:
        # Select the mask directory and the segmentation directory for this mask source
        mask_dir = output_root / ("masks_yolo" if source == "yolo" else "masks_gt2d")
        source_dir = segmentation_root / source

        # Only source classes absent from its mask metadata are excluded from vote generation
        vote_classes = _mask_classes(mask_dir, evaluation_classes)

        # Export the clean reference transferred Gaussians into prediction directories
        _measure_stage(
            stage_records, f"{source}:export_gt_gaussians",
            lambda: _export_gt_gaussians(
                args, runtime, model_dir, gt_dir, source_dir, scene,
                evaluation_classes,
            ),
            runtime=runtime,
        )

        # Accumulate votes
        vote_launches = _measure_stage(
            stage_records, f"{source}:votes",
            lambda: _run_votes(
                args, runtime, dataset_dir, model_dir, mask_dir, source_dir,
                vote_classes,
                analytics_store is not None,
                vote_identifier,
            ),
            runtime=runtime,
        )
        stage_records[-1]["container_count"] = vote_launches
        stage_records[-1]["cache_mode"] = "hit" if vote_launches == 0 else "miss"

        # Threshold the votes and produce labeled Gaussian files
        betas, threshold_containers = _measure_stage(
            stage_records, f"{source}:threshold_hysteresis",
            lambda: _run_thresholds(
                args, runtime, model_dir, source_dir, vote_classes,
                vote_identifier,
            ),
            runtime=runtime,
        )
        stage_records[-1]["container_count"] = threshold_containers
        stage_records[-1]["cache_mode"] = (
            "hit" if threshold_containers == 0 else "miss"
        )

        # Evaluate every beta for the selected source
        scene_results[source] = _measure_stage(
            stage_records, f"{source}:evaluation_transfer",
            lambda: _evaluate_scene(
                args, scene, gaussians_near_a_vertex, gaussian_labels,
                full_xyz, full_opacity, evaluation_classes, vote_classes,
                ground_truth_transfer_by_class, vote_identifier, variant,
                source_dir, betas, results_dir_for[source], source,
            ),
            runtime=runtime,
        )
        if analytics_store is not None:
            record_source_analytics(
                analytics_store, run_id, source, scene,
                f"{scene.dataset}:{scene.scene}", evaluation_classes, betas,
                source_dir, scene_results[source],
                vote_identifier, args.hysteresis_gamma, args.hysteresis_radius,
            )

    # Update the source summary without dropping other results
    summary_path = results_dir / "results.json"
    previous_results = {}
    if summary_path.exists():
        previous_results = json.loads(summary_path.read_text())
    previous_results.update(scene_results)
    atomic_write_text(
        summary_path, json.dumps(previous_results, indent=2, default=str) + "\n",
    )

    # Record the completed run, its parameters and its stages
    if analytics_store is not None:
        elapsed_seconds = time.perf_counter() - run_started
        peak_memory_values = [
            record["peak_cuda_memory_bytes"]
            for record in stage_records
            if record["peak_cuda_memory_bytes"] is not None
        ]
        peak_reserved_memory_values = [
            record["peak_cuda_memory_reserved_bytes"]
            for record in stage_records
            if record["peak_cuda_memory_reserved_bytes"] is not None
        ]
        analytics_store.append("runs", {
            "run_id": run_id,
            "created_at": utc_now(),
            "status": "completed",
            "dataset": scene.dataset,
            "scene_id": f"{scene.dataset}:{scene.scene}",
            "scene_name": scene.scene,
            "split": args.split,
            "source": args.mask_source,
            "output_root": str(output_root),
            "model_root": str(model_dir),
            "elapsed_seconds": elapsed_seconds,
            "peak_cuda_memory_bytes": max(peak_memory_values, default=None),
            "peak_cuda_memory_reserved_bytes": max(
                peak_reserved_memory_values, default=None,
            ),
        })
        analytics_store.append("run_parameters", {
            "run_id": run_id,
            "variant": variant,
            "vote_id": vote_identifier,
            **parameters,
            **run_metadata,
        })
        for record in stage_records:
            analytics_store.append("run_stages", {
                "run_id": run_id,
                "dataset": scene.dataset,
                "scene_id": f"{scene.dataset}:{scene.scene}",
                "stage": record["stage"],
                "cache_mode": record["cache_mode"],
                "container_count": record["container_count"],
                "elapsed_seconds": record["elapsed_seconds"],
                "peak_cuda_memory_bytes": record["peak_cuda_memory_bytes"],
                "peak_cuda_memory_reserved_bytes": record[
                    "peak_cuda_memory_reserved_bytes"
                ],
            })
    _progress(f"Run finished in {time.perf_counter() - run_started:.1f}s")
    print(json.dumps({name: value["metrics_by_beta"]
                      for name, value in scene_results.items()}, indent=2))


def _make_scene(args, data_root, support_dir):
    """
    Create a scene instance based on the dataset type and provided arguments

    The returned instance loads the dataset specific mesh, labels and visibility
    information into the common scene representation
    """
    if args.dataset == "replica":
        return ReplicaScene(
            data_root, args.scene, args.sequence_name, args.frame_step, seed=3,
            vertex_label_min_fraction=args.replica_vertex_label_min_fraction,
            visibility_slop=args.replica_visibility_slop,
        )
    elif args.dataset == "scannetpp":
        return ScannetScene(data_root, args.scene, support_dir)


if __name__ == "__main__":
    main()
