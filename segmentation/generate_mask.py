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

# Add project root to path to ensure imports work if needed
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

    # Prepare the image path for inference
    img_path_str = str(image_path)
    
    # Inference
    # Request masks at the original image resolution
    results = model(img_path_str, verbose=False, conf=conf, save=False, retina_masks=True)
    result = results[0]
    
    # Allocate masks at the source image resolution
    original_height, original_width = result.orig_shape
    
    # Pre allocate tensors on the same device as the model output
    device = result.masks.data.device if result.masks is not None else 'cpu'
    
    detector_label_mask = torch.zeros((original_height, original_width), dtype=torch.int32, device=device)
    confidence_mask = torch.zeros((original_height, original_width), dtype=torch.float32, device=device)
    names_map = {}
    
    # Paint detections from low to high confidence
    if result.masks is not None:
        masks = result.masks.data
        boxes = result.boxes
        
        # Get sorted indexes by confidence, from low to high, so that higher confidence detections will overwrite lower ones in the mask
        confidences = boxes.conf # There is one confidence per object detected
        sort_idx = torch.argsort(confidences)
        
        class_ids = boxes.cls
        
        for idx in sort_idx:
            # Use soft mask values combined with detection confidence for pixel confidence
            soft_mask = masks[idx]
            
            # Binary mask for assignment
            mask_bool = soft_mask > 0.5
            
            # Keep the detector ID before applying the storage offset
            cls_id = int(class_ids[idx])
            det_conf = confidences[idx]
            
            stored_id = cls_id + 1
            names_map[stored_id] = result.names[cls_id]
            
            # Combine pixel mask probability with detection confidence to reduce edge confidence
            pixel_conf = soft_mask * det_conf
            
            # Because we are looping from low to high, if this pixel was already painted by a weaker detection, it gets overwritten
            detector_label_mask[mask_bool] = stored_id
            confidence_mask[mask_bool] = pixel_conf[mask_bool]
            
    # Return CPU arrays for image writing
    return (
        detector_label_mask.cpu().numpy(),
        confidence_mask.cpu().numpy(), 
        names_map
    )

def save_single_mask(key, semantic, confidence, output_dir):
    # Write semantic and confidence images
    sem_path = output_dir / "semantic" / f"{key}.png"
    conf_path = output_dir / "confidence" / f"{key}.png"
    
    cv2.imwrite(str(sem_path), semantic.astype(np.uint8))
    cv2.imwrite(str(conf_path), (confidence * 255).astype(np.uint8))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to single input image", default=None)
    parser.add_argument("--images_dir", help="Directory of images", default="example_data/data/tandt/truck/images")
    parser.add_argument("--model", default="./yolo26x-seg.pt", help="Path to YOLO model")
    parser.add_argument("--conf", type=float, default=0.75, help="Confidence threshold")
    parser.add_argument("--output_root", required=True, help="Root directory for output masks")
    
    args = parser.parse_args()
    
    output_root = Path(args.output_root)
    (output_root / "semantic").mkdir(parents=True, exist_ok=True)
    (output_root / "confidence").mkdir(parents=True, exist_ok=True)

    images_to_process = []
    
    if args.image:
        images_to_process.append(Path(args.image))
    else:
        # Use pathlib to find images
        img_dir = Path(args.images_dir)
        exts = ["*.jpg", "*.png", "*.JPG", "*.PNG", "*.jpeg"]
        for ext in exts:
            images_to_process.extend(img_dir.glob(ext))
        
        images_to_process.sort()
        
        if not images_to_process:
            print(f"No images found in {args.images_dir}")
            sys.exit(0)

    # Load the model
    print(f"Loading model {args.model}")
    model = YOLO(args.model)

    global_names = {}
    print(f"Processing {len(images_to_process)} images")

    for img_path in tqdm(images_to_process):
        # Process
        sem, conf, names = get_segmentation_masks(img_path, model, conf=args.conf)
        
        # Update global names
        global_names.update(names)
        
        # Save that scene's masks
        key = img_path.stem
        save_single_mask(key, sem, conf, output_root)

    '''
     In classes.json, the keys are stored detector IDs and the values are
     detector names returned by YOLO. These are not main project
     names rather than dataset names

     The keys are detector model IDs shifted by 1, stored_id = cls_id + 1
     The values are the corresponding detector names, result.names[cls_id]
    '''

    serializable_map = {str(k): v for k, v in global_names.items()}
    with open(output_root / "classes.json", 'w') as f:
        json.dump(serializable_map, f, indent=4)