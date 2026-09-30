import json
import os
import sys
import traceback
from argparse import ArgumentParser

import cv2
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.sh_utils import SH2RGB

# Colours of the agreement panels: both sets, only the prediction and only the reference
AGREEMENT = {"both": (0.17, 0.63, 0.17), "prediction": (0.84, 0.15, 0.16), "reference": (0.12, 0.47, 0.71)}
FRACTION_COLOUR = (0.84, 0.15, 0.16)


def pale_colours(gaussians):
    """ The view-independent colour of every Gaussian, desaturated and lightened towards white """
    rgb = SH2RGB(gaussians.get_features_dc[:, 0, :]).clamp(0.0, 1.0)
    grey = rgb.mean(dim=1, keepdim=True)
    return 0.45 * (0.6 * grey + 0.4 * rgb) + 0.55


def indices(path):
    return torch.as_tensor(np.load(path), dtype=torch.long)


def panel_colours(panel, base):
    """ Per-Gaussian colours of one rendered panel """
    colours = base.clone()
    if panel["kind"] == "fraction":
        votes = torch.load(panel["votes"], map_location="cpu")
        target, background = votes["target_weights"].float(), votes["background_weights"].float()
        total = target + background
        rho = torch.where(total > 0, target / total.clamp_min(1e-12), torch.zeros_like(total)).to(base.device)
        strong = torch.tensor(FRACTION_COLOUR, device=base.device)
        colours = base * (1.0 - rho[:, None]) + strong * rho[:, None]
    elif panel["kind"] == "labels":
        for path, colour in panel["layers"]:
            colours[indices(path).to(base.device)] = torch.tensor(colour, device=base.device)
    elif panel["kind"] == "agreement":
        # One pair of prediction and reference per class; a class without prediction has only its reference.
        # A Gaussian correct for some class is green, and otherwise an extra selection is red before a miss
        found = {key: torch.zeros(len(base), dtype=torch.bool, device=base.device) for key in AGREEMENT}
        for prediction_path, reference_path in panel["pairs"]:
            prediction = torch.zeros_like(found["both"])
            reference = torch.zeros_like(found["both"])
            if prediction_path:
                prediction[indices(prediction_path).to(base.device)] = True
            reference[indices(reference_path).to(base.device)] = True
            found["both"] |= prediction & reference
            found["prediction"] |= prediction & ~reference
            found["reference"] |= reference & ~prediction
        for key in ("reference", "prediction", "both"):
            colours[found[key]] = torch.tensor(AGREEMENT[key], device=base.device)
    return colours


def photograph(camera):
    """ The photograph of the camera as an 8-bit BGR image """
    image = camera.original_image.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return (image * 255.0).astype(np.uint8)[:, :, ::-1].copy()


def mask_overlay(camera, panel):
    """ The photograph with the pixels of each listed class painted in its colour and outlined """
    image = photograph(camera)
    mask = cv2.imread(panel["mask"], cv2.IMREAD_UNCHANGED)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    painted = image.copy()
    for stored_id, colour in panel["classes"]:
        region = mask == stored_id
        bgr = np.array(colour[::-1]) * 255.0
        painted[region] = (0.45 * painted[region] + 0.55 * bgr).astype(np.uint8)
        contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(painted, contours, -1, tuple(int(c) for c in bgr), 2)
    return painted


def main(args, dataset, pipe):
    spec = json.loads(open(args.spec, encoding="utf-8").read())
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.loaded_iter, shuffle=False)
    cameras = {os.path.splitext(os.path.basename(camera.image_name))[0]: camera for camera in scene.getTrainCameras()}
    base = pale_colours(gaussians)
    background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
    os.makedirs(args.output_dir, exist_ok=True)

    # A view that fails is reported and skipped, so the other views of the spec are still saved
    failed = []
    for view in spec["views"]:
        try:
            camera = cameras[view["camera"]]
            for panel in view["panels"]:
                path = os.path.join(args.output_dir, f"{view['name']}_{panel['name']}.png")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                if panel["kind"] == "photo":
                    image = photograph(camera)
                elif panel["kind"] == "mask":
                    image = mask_overlay(camera, panel)
                else:
                    colours = panel_colours(panel, base)
                    output = render(camera, gaussians, pipe, background, override_color=colours)["render"]
                    image = (output.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)[:, :, ::-1]
                cv2.imwrite(path, image)
                print(f"saved {path}")
        except Exception:
            traceback.print_exc()
            failed.append(view["name"])
    if failed:
        raise SystemExit(f"failed views: {', '.join(failed)}")


if __name__ == "__main__":
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--loaded_iter", type=int, default=30000)
    parser.add_argument("--spec", required=True, help="JSON with the views and the panels of each one")
    parser.add_argument("--output_dir", required=True)
    args = get_combined_args(parser)
    with torch.no_grad():
        main(args, model.extract(args), pipeline.extract(args))
