import os
import sys
import argparse
import numpy as np
import cv2
import json
from pathlib import Path
import torch
from ultralytics import YOLO
from tqdm import tqdm

# Add project root to path so repository imports work inside the container
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_segmentation_masks(image_path, model, conf):
    """
    Runs YOLO 2D segmentation on a single image file

    Returns:
        detector_label_mask as a np.array of shape (H, W) whose values are stored
        detector IDs: 0 for background and detector model ID + 1 for a
        detected class
        confidence_mask as a np.array of shape (H, W) with confidence values in [0, 1]
        names_map as a dict mapping stored detector IDs to detector-
        vocabulary names
    """

    # Inference at the original image resolution
    results = model(str(image_path), verbose=False, conf=conf, save=False, retina_masks=True)
    result = results[0]

    # Allocate masks at the source image resolution
    original_height, original_width = result.orig_shape
    device = result.masks.data.device if result.masks is not None else 'cpu'

    detector_label_mask = torch.zeros((original_height, original_width), dtype=torch.int32, device=device)
    confidence_mask = torch.zeros((original_height, original_width), dtype=torch.float32, device=device)
    names_map = {}

    if result.masks is not None:
        masks = result.masks.data
        boxes = result.boxes

        # Sort detections from low to high confidence, so higher confidence
        # detections overwrite weaker ones in the mask
        confidences = boxes.conf
        sort_idx = torch.argsort(confidences)

        class_ids = boxes.cls

        for idx in sort_idx:
            soft_mask = masks[idx]
            mask_bool = soft_mask > 0.5

            cls_id = int(class_ids[idx])
            det_conf = confidences[idx]
            stored_id = cls_id + 1
            names_map[stored_id] = result.names[cls_id]

            # Combine pixel mask probability with detection confidence, which
            # reduces edge confidence
            pixel_conf = soft_mask * det_conf

            # Because we iterate low to high, pixels already painted by a weaker
            # detection are overwritten here
            detector_label_mask[mask_bool] = stored_id
            confidence_mask[mask_bool] = pixel_conf[mask_bool]

    return (
        detector_label_mask.cpu().numpy(),
        confidence_mask.cpu().numpy(),
        names_map,
    )


def save_single_mask(key, semantic, confidence, output_dir):
    # Write semantic and confidence images
    sem_path = output_dir / "semantic" / f"{key}.png"
    conf_path = output_dir / "confidence" / f"{key}.png"

    cv2.imwrite(str(sem_path), semantic.astype(np.uint8))
    cv2.imwrite(str(conf_path), (confidence * 255).astype(np.uint8))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True, help="Directory of images")
    parser.add_argument("--model", default="./yolo26x-seg.pt", help="Path to YOLO model")
    parser.add_argument("--conf", type=float, default=0.75, help="Confidence threshold")
    parser.add_argument("--output_root", required=True, help="Root directory for output masks")

    args = parser.parse_args()

    output_root = Path(args.output_root)
    (output_root / "semantic").mkdir(parents=True, exist_ok=True)
    (output_root / "confidence").mkdir(parents=True, exist_ok=True)

    img_dir = Path(args.images_dir)
    exts = ["*.jpg", "*.png", "*.JPG", "*.PNG", "*.jpeg"]
    images_to_process = []
    for ext in exts:
        images_to_process.extend(img_dir.glob(ext))

    images_to_process.sort()
    if not images_to_process:
        print(f"No images found in {args.images_dir}")
        sys.exit(0)

    print(f"Loading model {args.model}")
    model = YOLO(args.model)

    global_names = {}
    print(f"Processing {len(images_to_process)} images")

    for img_path in tqdm(images_to_process):
        sem, conf, names = get_segmentation_masks(img_path, model, conf=args.conf)

        # Update the global detector name map
        global_names.update(names)
        save_single_mask(img_path.stem, sem, conf, output_root)

    '''
     In classes.json, the keys are stored detector IDs and the values are
     detector names returned by YOLO. The keys are detector model IDs shifted
     by one (stored_id = cls_id + 1), so zero remains the background ID.
    '''

    serializable_map = {str(k): v for k, v in global_names.items()}
    with open(output_root / "classes.json", 'w') as f:
        json.dump(serializable_map, f, indent=4)
