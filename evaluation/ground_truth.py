# Ground truth caches and label transfers

import json
import os
import tempfile

import numpy as np
from scipy.spatial import cKDTree

from .common import atomic_write_text, ensure_dir
from . import transfer


def _neighborhood_metadata(scene, gaussian_ply, tau, evaluation_scope_version, gaussian_count):
    """
    Describe the neighborhood arrays

    The Gaussian model signature (size and mtime) is part of the contract, so a
    retrained model rebuilds the neighborhoods instead of mixing them.
    """
    stat = gaussian_ply.stat()
    return {
        "evaluation_scope_version": evaluation_scope_version,
        "dataset": scene.dataset,
        "scene": scene.scene,
        "gaussians": int(gaussian_count),
        "ply_size": int(stat.st_size),
        "ply_mtime_ns": int(stat.st_mtime_ns),
        "tau": float(tau),
    }


def _labels_metadata(neighborhood_metadata, min_fraction,
                     mesh_to_gaussian_background_competes, mesh_to_gaussian_transfer):
    """ Describe labels derived from neighborhood data """
    return {
        **neighborhood_metadata,
        "min_fraction": float(min_fraction),
        "mesh_to_gaussian_background_competes": bool(
            mesh_to_gaussian_background_competes
        ),
        "mesh_to_gaussian_transfer": mesh_to_gaussian_transfer,
    }


def _needs_rebuild(meta_path, expected, force):
    """ Return whether the cache needs a rebuild """
    if force or not meta_path.exists():
        return True
    try:
        return json.loads(meta_path.read_text()) != expected
    except (OSError, ValueError):
        return True


def build(scene, gaussian_ply, gt_dir, tau, min_fraction, mesh_to_gaussian_background_competes,
          mesh_to_gaussian_transfer="radius_vote", force=False,
          evaluation_scope_version=6):
    """
    Build or reuse the neighborhoods and the Gaussians GT local semantic labels used for evaluation

    - The cache contains both directions of the radius neighborhoods and semantic labels for the Gaussian model
    - The transfer method chooses between radius voting and nearest-neighbor label assignment

    mesh_to_gaussian_background_competes controls whether non-target mesh labels
    participate when transferring GT labels from the mesh to Gaussians.
    force makes cached files rebuild even when their metadata matches.
    """

    # Create the cache directory before reading or writing any cache file
    ensure_dir(gt_dir)
    neighborhood_meta_path = gt_dir / "neighborhood_meta.json"
    labels_meta_path = gt_dir / "labels_meta.json"

    full_xyz, _ = transfer.load_gaussian_ply(gaussian_ply)
    neighborhood_expected = _neighborhood_metadata(
        scene, gaussian_ply, tau, evaluation_scope_version, len(full_xyz),
    )
    labels_expected = _labels_metadata(
        neighborhood_expected, min_fraction,
        mesh_to_gaussian_background_competes, mesh_to_gaussian_transfer,
    )
    neighborhoods_rebuild = _needs_rebuild(
        neighborhood_meta_path, neighborhood_expected, force,
    )
    labels_rebuild = neighborhoods_rebuild or _needs_rebuild(
        labels_meta_path, labels_expected, force,
    )

    gaussians_near_a_vertex_path = gt_dir / "gaussians_near_a_vertex_neighbors.npz"
    vertices_near_a_gaussian_path = gt_dir / "vertices_near_a_gaussian_neighbors.npz"
    gaussian_labels_path = gt_dir / "gt_gaussian_labels.npz"

    if neighborhoods_rebuild or not gaussians_near_a_vertex_path.exists():
        gaussians_near_a_vertex = transfer.build_radius_neighbors(
            scene.vertices, cKDTree(full_xyz), tau,
        )
        transfer.save_neighbors(gaussians_near_a_vertex_path, gaussians_near_a_vertex)

    else:
        gaussians_near_a_vertex = transfer.load_neighbors(gaussians_near_a_vertex_path)

    if neighborhoods_rebuild or not vertices_near_a_gaussian_path.exists():
        vertices_near_a_gaussian = transfer.build_radius_neighbors(
            full_xyz, cKDTree(scene.vertices), tau,
        )
        transfer.save_neighbors(vertices_near_a_gaussian_path, vertices_near_a_gaussian)

    else:
        vertices_near_a_gaussian = transfer.load_neighbors(vertices_near_a_gaussian_path)

    if labels_rebuild or not gaussian_labels_path.exists():
        reference_labels = np.where(scene.semantic_labels >= 0, scene.semantic_labels, -1).astype(np.int64)
        classes = np.arange(len(scene.classes), dtype=np.int64)

        if mesh_to_gaussian_transfer == "nearest_neighbor_label":
            # Find the nearest mesh vertex for every Gaussian center
            distances, nearest = cKDTree(scene.vertices).query(full_xyz, k=1)
            gaussian_labels = np.where((distances <= tau) & (reference_labels[nearest] >= 0), reference_labels[nearest], -1,).astype(np.int64)

        else:
            gaussian_labels = transfer.radius_label_vote(
                len(full_xyz), vertices_near_a_gaussian, reference_labels,
                np.ones(len(reference_labels), dtype=np.float64), classes,
                min_fraction, mesh_to_gaussian_background_competes,
            )

        # Write the label archive atomically
        fd, temporary_name = tempfile.mkstemp(
            dir=gaussian_labels_path.parent, suffix=".npz",
        )
        try:
            with os.fdopen(fd, "wb") as temporary:
                np.savez_compressed(temporary, labels=gaussian_labels)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, gaussian_labels_path)
        finally:
            # Remove an incomplete label file
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    else:
        gaussian_labels = np.load(gaussian_labels_path)["labels"]

    atomic_write_text(
        neighborhood_meta_path,
        json.dumps(neighborhood_expected, indent=2) + "\n",
    )
    atomic_write_text(
        labels_meta_path, json.dumps(labels_expected, indent=2) + "\n",
    )
    return gaussians_near_a_vertex, gaussian_labels
