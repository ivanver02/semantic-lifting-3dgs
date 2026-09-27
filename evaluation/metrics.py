# Semantic IoU, precision and recall metrics

import numpy as np


def class_iou(predicted, scene, class_id):
    """ Compute metrics for a target class from the boolean vertex prediction """

    # Identify target vertices
    mask = scene.evaluation_mask
    predicted_positive = predicted[mask]
    ground_truth_positive = scene.semantic_labels[mask] == class_id

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


def _mean(values):
    """ Return the mean or zero for an empty sequence """
    return float(np.mean(values)) if values else 0.0


def aggregate(per_class):
    """ Aggregate semantic metrics """

    # Average over the mask-level evaluation universe for every mask source.
    evaluated = list(per_class.items())

    # Compute relative scores against transferred reference metrics
    relative = []
    relative_classes = []
    for name, item in evaluated:
        reference_iou = item["ground_truth_transfer_iou"]["iou"]
        if reference_iou > 0:
            relative.append(item["iou"]["iou"] / reference_iou)
            relative_classes.append(name)
    zero_reference_class_count = sum(
        item["ground_truth_transfer_iou"]["iou"] == 0
        for _, item in evaluated
    )

    # Pool error counts for global metrics
    tp = sum(item["iou"]["tp"] for _, item in evaluated)
    fp = sum(item["iou"]["fp"] for _, item in evaluated)
    fn = sum(item["iou"]["fn"] for _, item in evaluated)
    union = tp + fp + fn
    ground_truth_transfer_tp = sum(
        item["ground_truth_transfer_iou"]["tp"] for _, item in evaluated
    )
    ground_truth_transfer_fp = sum(

    # Compute global transfer metrics
        item["ground_truth_transfer_iou"]["fp"] for _, item in evaluated
    )
    ground_truth_transfer_fn = sum(
        item["ground_truth_transfer_iou"]["fn"] for _, item in evaluated
    )

    # Return aggregate metrics and class inventories
    return {
        "mIoU": _mean([item["iou"]["iou"] for _, item in evaluated]),
        "ground_truth_transfer_mIoU": _mean([
            item["ground_truth_transfer_iou"]["iou"] for _, item in evaluated
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
        "zero_reference_class_count": int(zero_reference_class_count),
    }
