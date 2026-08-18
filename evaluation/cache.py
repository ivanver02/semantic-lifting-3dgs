# Run metadata and evaluation caches

import json
import os
import shutil
import tempfile

from .common import atomic_write_text, ensure_dir


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
    """ Determine the Gaussian model directory to use for evaluation """

    if args.model_root is not None:
        model_dir = args.model_root.resolve()
        if not _has_model(model_dir, args.iterations):
            raise FileNotFoundError(
                f"Gaussian model missing for iteration {args.iterations}: "
                f"{model_dir}"
            )
        return model_dir

    # Check the output model directory
    output_model = output_root / "model"
    if _has_model(output_model, args.iterations):
        return output_model

    if args.dataset == "replica":
        conventional_models = [
            data_root / args.scene / "eval_output" / "gs_model",
        ]
    else:
        conventional_models = [args.repo_root / "output" / args.scene]
    for conventional_model in conventional_models:
        if _has_model(conventional_model, args.iterations):

    # Reuse the conventional model directory
            print(f"model: Using existing Gaussian model: {conventional_model}")
            return conventional_model

    return output_model


def run_parameters(args, data_root):
    """ Prepare parameters for cache validation """
    return {
        "evaluation_scope_version": 5,
        "dataset": args.dataset,
        "scene": args.scene,
        "split": args.split,
        "data_root": str(data_root),
        "sequence_name": args.sequence_name,

    # Add frame sampling settings
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


def _scope_metadata(parameters, keys):
    """ Return parameters for one artifact scope """
    return {key: parameters[key] for key in keys}


def _validate_scope_metadata(path, expected, artifact, force):
    """ Validate one artifact contract """
    if not path.exists():
        return

    try:
        previous = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"{path} is unreadable or truncated, rebuild this artifact scope"
        ) from error
    _validate_scope_values(previous, expected, artifact, force, path)


def _validate_scope_values(previous, expected, artifact, force, path):
    """ Validate scope metadata values """
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

    # Raise the metadata mismatch
    raise RuntimeError(
        f"{path.parent} contains incompatible metadata for {artifact}:\n"
        f"{changes}\n"
        "use a new --output-root or pass --force"
    )


def _write_scope_metadata(path, metadata):
    """ Write a scope contract """
    ensure_dir(path.parent)
    temporary_fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
    )
    try:
        with os.fdopen(temporary_fd, "w", encoding="utf-8") as temporary:
            json.dump(metadata, temporary, indent=2)

    # Flush the metadata file
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def prepare_run_metadata(output_root, parameters, force, sources, force_delete=False):
    """ Prepare the output directory and write run parameters """
    common_metadata_path = output_root / "run_parameters.json"
    common_previous = None
    if common_metadata_path.exists():
        common_previous = json.loads(common_metadata_path.read_text())
    other_sources_have_results = any(
        (output_root / "results" / f"results_{source}.md").exists()
        for source in {"yolo", "gt2d"} - set(sources)
    )

    scopes = [
        ("meta_dataset.json", DATASET_METADATA_KEYS, "dataset/model"),
        ("meta_masks_gt2d.json", GT_MASK_CACHE_KEYS, "GT2D masks"),
        ("meta_gt.json", GT_METADATA_KEYS, "ground-truth transfer"),
    ]
    if "yolo" in sources:
        scopes.append(("meta_masks_yolo.json", YOLO_MASK_METADATA_KEYS, "YOLO masks"))
    for source in sources:
        scopes.append((
            f"meta_votes_{source}.json",
            VOTE_CACHE_KEYS,
            f"{source} votes",
        ))

    for filename, keys, artifact in scopes:
        metadata_path = output_root / filename
        expected = _scope_metadata(parameters, keys)
        if metadata_path.exists():
            _validate_scope_metadata(metadata_path, expected, artifact, force)
        elif common_previous is not None and not filename.startswith("meta_votes_"):
            _validate_scope_values(
                common_previous, expected, artifact, force, common_metadata_path,
            )

    for source in sources:
        if force_delete:
            mask_dir = output_root / ("masks_gt2d" if source == "gt2d" else "masks_yolo")
            shutil.rmtree(mask_dir, ignore_errors=True)
            shutil.rmtree(output_root / "segmentation" / source, ignore_errors=True)
            for suffix in ["json", "md"]:
                (output_root / "results" / f"results_{source}.{suffix}").unlink(
                    missing_ok=True,
                )

    if (force_delete and common_previous is not None and not other_sources_have_results and
            (common_previous.get("iterations") != parameters["iterations"] or
             common_previous.get("resolution") != parameters["resolution"])):
        shutil.rmtree(output_root / "model", ignore_errors=True)
        shutil.rmtree(output_root / "dataset", ignore_errors=True)

    # Create the output root directory and record parameters for later runs.
    ensure_dir(output_root)
    for filename, keys, _artifact in scopes:
        _write_scope_metadata(
            output_root / filename, _scope_metadata(parameters, keys),
        )
    atomic_write_text(
        output_root / "run_parameters.json",
        json.dumps(parameters, indent=2) + "\n",
    )