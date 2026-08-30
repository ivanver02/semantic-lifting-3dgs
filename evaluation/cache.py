# Run metadata contracts that decide which artifacts are reused

import json

from .common import atomic_write_text, ensure_dir, vote_id


# Each scope lists the parameters that invalidate one kind of cached artifact
# When any of these values changes, the corresponding artifact is rebuilt
VOTE_CACHE_KEYS = [
    "evaluation_scope_version", "dataset", "scene", "data_root",
    "sequence_name", "frame_step",
    "iterations", "resolution", "background_mode", "background_confidence",
    "background_view_policy", "raster_block_size", "vote_data_device",
]

GT_MASK_CACHE_KEYS = [
    "evaluation_scope_version", "dataset", "scene", "data_root",
    "sequence_name", "frame_step", "resolution",
    "replica_vertex_label_min_fraction", "replica_visibility_slop",
    "scannetpp_mask_version", "scannetpp_mask_bands",
]

DATASET_METADATA_KEYS = [
    "evaluation_scope_version", "dataset", "scene", "data_root",
    "sequence_name", "frame_step",
    "resolution", "iterations", "train_data_device",
]

YOLO_MASK_METADATA_KEYS = [
    "evaluation_scope_version", "dataset", "scene", "data_root",
    "sequence_name", "frame_step",
    "resolution", "yolo_conf",
]

GT_METADATA_KEYS = [
    "evaluation_scope_version",
    "tau", "min_fraction", "mesh_to_gaussian_background_competes",
    "mesh_to_gaussian_transfer",
]


def _has_model(model_dir, iterations):
    """ Check whether the Gaussian model exists """
    return (model_dir / "point_cloud" / f"iteration_{iterations}" /
            "point_cloud.ply").exists()


def resolve_model_dir(args, data_root, output_root):
    """
    Determine the Gaussian model directory to use for evaluation

    --model-root reuses an explicit model; otherwise an already trained model
    is reused when it exists, and only then does training write into the run's
    own output directory.
    """
    if args.model_root is not None:
        model_dir = args.model_root.resolve()
        if not _has_model(model_dir, args.iterations):
            raise FileNotFoundError(
                f"Gaussian model missing for iteration {args.iterations}: "
                f"{model_dir}"
            )
        return model_dir

    # Check the run's own output model directory
    output_model = output_root / "model"
    if _has_model(output_model, args.iterations):
        return output_model

    # Reuse an existing Gaussian model from earlier training when it exists
    if args.dataset == "replica":
        conventional_models = [
            data_root / args.scene / "eval_output" / "gs_model",
        ]
    else:
        conventional_models = [args.repo_root / "output" / args.scene]
    for conventional_model in conventional_models:
        if _has_model(conventional_model, args.iterations):
            print(f"model: Using existing Gaussian model: {conventional_model}")
            return conventional_model

    return output_model


def run_parameters(args, data_root):
    """ Prepare the full parameter record used by every cache contract """
    return {
        "evaluation_scope_version": 6,
        "dataset": args.dataset,
        "scene": args.scene,
        "split": args.split,
        "data_root": str(data_root),
        "sequence_name": args.sequence_name,
        "frame_step": args.frame_step,
        "replica_vertex_label_min_fraction": args.replica_vertex_label_min_fraction,
        "replica_visibility_slop": args.replica_visibility_slop,
        "scannetpp_mask_version": args.scannetpp_mask_version,
        "scannetpp_mask_bands": args.scannetpp_mask_bands,
        "iterations": args.iterations,
        "resolution": args.resolution,
        "train_data_device": args.train_data_device,
        "yolo_conf": args.yolo_conf,
        "hysteresis_gamma": args.hysteresis_gamma,
        "hysteresis_radius": args.hysteresis_radius,
        "background_mode": args.background_mode,
        "background_confidence": args.background_confidence,
        "background_view_policy": args.background_view_policy,
        "betas": list(args.betas),
        "tau": args.tau,
        "min_fraction": args.min_fraction,
        "mesh_to_gaussian_transfer": args.mesh_to_gaussian_transfer,
        "gaussian_to_mesh_transfer": args.gaussian_to_mesh_transfer,
        "min_opacity": args.min_opacity,
        "gaussian_to_mesh_background_competes": args.gaussian_to_mesh_background_competes,
        "mesh_to_gaussian_background_competes": args.mesh_to_gaussian_background_competes,
        "opacity_weighting": not args.no_opacity_weighting,
        "raster_block_size": args.raster_block_size,
        "vote_data_device": args.vote_data_device,
    }


def _validate_scope_metadata(path, expected, artifact, force):
    """
    Validate one artifact contract against its stored metadata

    A mismatch means the directory already contains artifacts produced with
    different parameters, so reusing them would silently mix experiments.
    """
    if not path.exists():
        return
    try:
        previous = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"{path} is unreadable or truncated, rebuild this artifact scope"
        ) from error
    if previous == expected or force:
        return
    mismatches = [
        (key, previous.get(key), expected.get(key))
        for key in expected
        if previous.get(key) != expected.get(key)
    ]
    changes = "\n".join(
        f"  {key}: {old!r} -> {new!r}" for key, old, new in mismatches
    )
    raise RuntimeError(
        f"{path.parent} contains incompatible metadata for {artifact}:\n"
        f"{changes}\n"
        "use a new --output-root or pass --force"
    )


def prepare_run_metadata(output_root, parameters, force, sources):
    """
    Validate and write one metadata contract per reusable artifact scope

    Existing contracts are checked before any stage runs, so an incompatible
    invocation stops instead of mixing cached files from other settings.
    """
    scopes = [
        ("meta_dataset.json", DATASET_METADATA_KEYS, "dataset/model"),
        ("meta_masks_gt2d.json", GT_MASK_CACHE_KEYS, "GT2D masks"),
        ("meta_gt.json", GT_METADATA_KEYS, "ground-truth transfer"),
    ]
    if "yolo" in sources:
        scopes.append(("meta_masks_yolo.json", YOLO_MASK_METADATA_KEYS, "YOLO masks"))
    identifier = vote_id(parameters)
    for source in sources:
        scopes.append((
            f"meta_votes_{source}_{identifier}.json",
            VOTE_CACHE_KEYS,
            f"{source} votes ({identifier})",
        ))

    ensure_dir(output_root)
    for filename, keys, artifact in scopes:
        expected = {key: parameters[key] for key in keys}
        _validate_scope_metadata(output_root / filename, expected, artifact, force)

    # Persist the validated contracts and the full parameter snapshot
    for filename, keys, _artifact in scopes:
        expected = {key: parameters[key] for key in keys}
        atomic_write_text(
            output_root / filename, json.dumps(expected, indent=2) + "\n",
        )
    atomic_write_text(
        output_root / "run_parameters.json",
        json.dumps(parameters, indent=2) + "\n",
    )
