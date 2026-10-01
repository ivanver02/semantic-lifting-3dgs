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
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,

    # Add the intersection over union score
        "iou": float(tp / union) if union else 0.0,
    }


def _mean(values):
    """ Return the mean or None for an empty sequence """
    return float(np.mean(values)) if values else None


def aggregate(per_class):
    """ Aggregate semantic metrics """

    # Average over the mask-level evaluation universe for every mask source.
    evaluated = list(per_class.values())

    # Compute relative scores against transferred reference metrics, only for classes with a positive reference
    relative = [
        item["iou"]["iou"] / item["ground_truth_transfer_iou"]["iou"]
        for item in evaluated
        if item["ground_truth_transfer_iou"]["iou"] > 0
    ]

    # Return aggregate metrics
    return {
        "mIoU": _mean([item["iou"]["iou"] for item in evaluated]),
        "macro_precision": _mean([item["iou"]["precision"] for item in evaluated]),
        "macro_recall": _mean([item["iou"]["recall"] for item in evaluated]),
        "ground_truth_transfer_mIoU": _mean([
            item["ground_truth_transfer_iou"]["iou"] for item in evaluated
        ]),
        "relative_mIoU": _mean(relative),
    }
