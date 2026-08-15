# Semantic IoU, precision and recall metrics

import numpy as np

from . import transfer


def class_iou(predicted, ground_truth, mask, class_id):
    """ Compute metrics for a target class """

    # Identify target vertices
    predicted_positive = predicted[mask] == class_id
    ground_truth_positive = ground_truth[mask] == class_id

    # Count error matrix entries
    tp = int((predicted_positive & ground_truth_positive).sum())
    fp = int((predicted_positive & ~ground_truth_positive).sum())
    fn = int((~predicted_positive & ground_truth_positive).sum())
    union = tp + fp + fn

    # Return scalar metrics
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_count": tp + fn,
        "pred_count": tp + fp,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,

    # Add the intersection over union score
        "iou": float(tp / union) if union else 0.0,
    }


def evaluate_class(scene, gaussians_near_a_vertex, gaussian_labels, full_xyz, full_opacity, spec, predicted_xyz, tau, min_fraction, opacity_weighted,
                   min_opacity, gaussian_to_mesh_background_competes, gaussian_to_mesh_transfer,
                   ground_truth_transfer_metrics=None):
    """Evaluate one target class and its GT transfer reference"""

    # Resolve the local class and evaluation mask
    class_id = scene.class_id(spec.name)
    eval_mask = scene.evaluation_mask

    # Evaluate an empty or populated prediction
    if predicted_xyz is None or len(predicted_xyz) == 0:

        prediction = class_iou(np.full(len(scene.vertices), -1, dtype=np.int64), scene.semantic_labels, eval_mask, class_id)

    else:
        indices = transfer.map_subset_indices(full_xyz, predicted_xyz)
        predicted_labels = np.full(len(full_xyz), -1, dtype=np.int64)
        predicted_labels[indices] = class_id

        vertex_labels = transfer.predict_vertex_labels(
            scene.vertices, gaussians_near_a_vertex, predicted_labels,
            full_opacity, tau, min_fraction, opacity_weighted, min_opacity,
            gaussian_to_mesh_background_competes,
            gaussian_to_mesh_transfer,
        )

        prediction = class_iou(vertex_labels, scene.semantic_labels, eval_mask, class_id)

    # Build the reference transfer metrics when not supplied
    if ground_truth_transfer_metrics is None:

        ground_truth_transfer_mask = gaussian_labels == class_id
        ground_truth_transfer_xyz = full_xyz[ground_truth_transfer_mask]

        if len(ground_truth_transfer_xyz):
            ground_truth_transfer_class_labels = np.where(
                ground_truth_transfer_mask, class_id, -1,
            )
            ground_truth_transfer_labels = transfer.predict_vertex_labels(
                scene.vertices, gaussians_near_a_vertex,
                ground_truth_transfer_class_labels,
                full_opacity, tau, min_fraction, opacity_weighted, min_opacity,

    # Pass background transfer settings
                gaussian_to_mesh_background_competes,
                gaussian_to_mesh_transfer,
            )
            ground_truth_transfer_metrics = class_iou(
                ground_truth_transfer_labels, scene.semantic_labels,
                eval_mask, class_id,
            )

        else:
            ground_truth_transfer_metrics = class_iou(
                np.full(len(scene.vertices), -1, dtype=np.int64),
                scene.semantic_labels, eval_mask, class_id,
            )

    # Return prediction and reference metrics
    return {
        "class": spec.name,
        "name_by_detector": spec.name_by_detector,
        "iou": prediction,
        "ground_truth_transfer_iou": ground_truth_transfer_metrics,
    }


def _mean(values):
    """ Return the mean or zero for an empty sequence """
    return float(np.mean(values)) if values else 0.0


def aggregate(per_class):
    """ Aggregate semantic metrics using macro and global or micro averages """

    # Classes with neither GT nor predictions do not contribute to the averages
    evaluated = [
        (name, item) for name, item in per_class.items()
        if item["iou"]["gt_count"] > 0 or item["iou"]["pred_count"] > 0
    ]

    # Compute relative IoU only where the GT-transfer reference is non-zero.
    relative = [
        item["iou"]["iou"] / item["ground_truth_transfer_iou"]["iou"]
        for _, item in evaluated
        if item["ground_truth_transfer_iou"]["iou"] > 0
    ]

    # Classes with a non-zero GT-transfer IoU contribute to relative metrics.
    relative_classes = [
        name for name, item in evaluated
        if item["ground_truth_transfer_iou"]["iou"] > 0
    ]

    # Summing confusion counts gives the global or micro metrics across classes
    tp = sum(item["iou"]["tp"] for _, item in evaluated)
    fp = sum(item["iou"]["fp"] for _, item in evaluated)
    fn = sum(item["iou"]["fn"] for _, item in evaluated)
    union = tp + fp + fn
    ground_truth_transfer_tp = sum(
        item["ground_truth_transfer_iou"]["tp"] for _, item in evaluated
    )
    ground_truth_transfer_fp = sum(
        item["ground_truth_transfer_iou"]["fp"] for _, item in evaluated
    )
    ground_truth_transfer_fn = sum(
        item["ground_truth_transfer_iou"]["fn"] for _, item in evaluated
    )

    # Macro metrics average class level values, while global metrics use the accumulated confusion counts and therefore weight classes by their pixels
    return {
        "mIoU": _mean([item["iou"]["iou"] for _, item in evaluated]),
        "ground_truth_transfer_mIoU": _mean([
            item["ground_truth_transfer_iou"]["iou"] for _, item in evaluated
            if item["ground_truth_transfer_iou"]["iou"] > 0
        ]),

        "relative_mIoU": _mean(relative),
        "global_iou": float(tp / union) if union else 0.0,
        "macro_precision": _mean([
            item["iou"]["precision"] for _, item in evaluated
        ]),

        "macro_recall": _mean([
            item["iou"]["recall"] for _, item in evaluated
        ]),

        "global_precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "global_recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "ground_truth_transfer_macro_precision": _mean([
            item["ground_truth_transfer_iou"]["precision"] for _, item in evaluated
        ]),

        "ground_truth_transfer_macro_recall": _mean([
            item["ground_truth_transfer_iou"]["recall"] for _, item in evaluated
        ]),

        "ground_truth_transfer_global_precision": (
            float(ground_truth_transfer_tp /
                  (ground_truth_transfer_tp + ground_truth_transfer_fp))
            if ground_truth_transfer_tp + ground_truth_transfer_fp else 0.0
        ),

        "ground_truth_transfer_global_recall": (
            float(ground_truth_transfer_tp /
                  (ground_truth_transfer_tp + ground_truth_transfer_fn))
            if ground_truth_transfer_tp + ground_truth_transfer_fn else 0.0
        ),
        
        "evaluated_classes": [name for name, _ in evaluated],
        "relative_classes": relative_classes,
    }