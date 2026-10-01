import argparse
import json
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2

from evaluation.analytics import load_analytics
from evaluation.common import safe_name, selection_path, vote_dir
from evaluation.runtime import Runtime
from evaluation.scannetpp.scene import CLASSES
from evaluation.scripts.experiment_common import token
from evaluation.summaries import frozen_rows, load_points, reachable, scene_summary

# The colours of the classes in every figure of the manuscripts
CLASS_COLOURS = {
    "chair": "#0072B2", "sofa": "#E69F00", "table": "#009E73", "tv": "#CC79A7",
    "laptop": "#56B4E9", "sink": "#D55E00", "plant": "#999933", "clock": "#000000", "bench": "#882255",
}
FRACTION_CLASS = "chair"
MIN_SHARE = 0.005
CANDIDATE_CAMERAS = 4
SOURCES = ("gt2d", "yolo")


def rgb(colour):
    return [int(colour[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]


def class_shares(mask_dir, stored_ids):
    """ For every camera, the share of the image that each class covers in a mask directory """
    shares = {}
    for path in sorted(Path(mask_dir).glob("*.png")):
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        shares[path.stem] = {stored: float((mask == stored).mean()) for stored in stored_ids}
    return shares


class SceneFiles:
    """ The paths of one test scene: model, masks, votes, selections and references at the selected point """

    def __init__(self, data_root, view, point, scene):
        self.scene = scene
        self.root = Path(data_root) / "scannetpp" / "eval" / "scannetpp" / scene
        self.point = point
        self.variant = f"frozen_nearest_g{token(point.gamma)}"
        runs = sorted((row for row in view["runs"].values() if row["scene_id"] == f"scannetpp:{scene}"),
                      key=lambda row: row["created_at"])
        model = [Path(row["model_root"]) for row in runs if row.get("model_root")]
        self.model = next((m for m in reversed(model) if (m / "cfg_args").exists()), self.root / "model")
        self.vote_ids = {(row["class_id"], row["source"]): row["vote_id"] for row in view["vote_statistics"]
                         if row["scene_id"] == f"scannetpp:{scene}" and row["variant"] == self.variant}
        self.class_ids = {row["class_name"]: row["class_id"] for row in view["classes"] if row["dataset"] == "scannetpp"}

    def spec(self, name):
        return next(item for item in CLASSES if item.name == name)

    def votes_dir(self, name, source):
        # A class that a source never proposes in the scene has no votes, and its paths do not exist
        identifier = self.vote_ids.get((self.class_ids[name], source), "missing")
        return vote_dir(self.root / "segmentation" / source, self.spec(name), identifier)

    def votes(self, name, source):
        return self.votes_dir(name, source) / f"voting_data_{safe_name(self.spec(name).name_by_detector)}.pt"

    def prediction(self, name, source):
        return selection_path(self.votes_dir(name, source), self.point.gamma, 0.05, self.point.beta)

    def reference(self, name):
        return self.root / "results" / self.variant / "reference" / f"{safe_name(self.spec(name).name_by_detector)}.npy"

    def masks(self, source):
        return self.root / f"masks_{source}" / "semantic"


def ranked_cameras(files, classes, both_sources):
    """
    The cameras ordered by the number of classes that cover at least MIN_SHARE of the image, then by their
    total share. With both_sources the class has to be visible in the annotation and in the YOLO masks
    """
    ids = {name: files.spec(name).detector_stored_id for name in classes}
    gt = class_shares(files.masks("gt2d"), ids.values())
    yolo = class_shares(files.masks("yolo"), ids.values()) if both_sources else gt

    def score(stem):
        visible = [name for name in classes if gt[stem][ids[name]] >= MIN_SHARE
                   and yolo.get(stem, {}).get(ids[name], 0.0) >= MIN_SHARE]
        return len(visible), sum(gt[stem][ids[name]] for name in visible)

    return sorted(gt, key=score, reverse=True), sorted(gt)


def model_cameras(files):
    """ The image stems of the cameras of the trained model, or None when the model does not list them """
    path = files.model / "cameras.json"
    if not path.exists():
        return None
    return {Path(item["img_name"]).stem for item in json.loads(path.read_text(encoding="utf-8"))}


def spread_cameras(files, classes, count):
    """ The best cameras of ranked_cameras that are apart from each other in the capture order """
    ranked, order = ranked_cameras(files, classes, both_sources=False)
    known = model_cameras(files)
    if known is not None:
        ranked = [stem for stem in ranked if stem in known]
    position = {stem: i for i, stem in enumerate(order)}
    gap = max(1, len(order) // (2 * count))
    chosen = []
    for stem in ranked:
        if all(abs(position[stem] - position[other]) >= gap for other in chosen):
            chosen.append(stem)
        if len(chosen) == count:
            break
    return chosen


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True, help="the data root with analytics and scannetpp")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=None, help="scannetpp/figures under the data root by default")
    args = parser.parse_args(argv)
    output = args.output or args.data_root / "scannetpp" / "figures"
    output.mkdir(parents=True, exist_ok=True)

    view = load_analytics(args.data_root / "analytics")
    selected, _, _ = load_points(args.data_root / "analytics")
    names = {row["class_id"]: row["class_name"] for row in view["classes"]}

    # Reachable classes of every test scene, and the scene means with the annotation masks
    classes = defaultdict(set)
    for row in frozen_rows(view, "class_beta_metrics", selected, "scannetpp", "gt2d"):
        if reachable(row):
            classes[row["scene_id"].split(":")[1]].add(names[row["class_id"]])
    miou = {source: scene_summary(view, selected, "scannetpp", source)["scenes"] for source in SOURCES}
    scenes = miou["gt2d"]
    overview_scene = max(scenes, key=lambda s: (len(classes[s]), scenes[s]["miou"]))

    runtime = Runtime(args.repo_root, args.data_root / "scannetpp")
    to_container = runtime.container_path
    manifest = {"point": selected._asdict(), "views": [], "candidates": []}

    # Overview: from the YOLO masks of one view to the labels of every class
    files = SceneFiles(args.data_root, view, selected, overview_scene)
    scene_classes = [name for name in CLASS_COLOURS if name in classes[overview_scene]]
    camera = ranked_cameras(files, scene_classes, both_sources=True)[0][0]
    yolo_classes = [name for name in scene_classes if files.prediction(name, "yolo").exists()]
    overview = {"name": "overview", "camera": camera, "panels": [
        {"name": "photo", "kind": "photo"},
        {"name": "masks", "kind": "mask", "mask": to_container(files.masks("yolo") / f"{camera}.png"),
         "classes": [[files.spec(n).detector_stored_id, rgb(CLASS_COLOURS[n])] for n in scene_classes]},
        {"name": "fraction", "kind": "fraction", "votes": to_container(files.votes(FRACTION_CLASS, "yolo"))},
        {"name": "labels", "kind": "labels",
         "layers": [[to_container(files.prediction(n, "yolo")), rgb(CLASS_COLOURS[n])] for n in yolo_classes]},
    ]}
    manifest["views"].append({"name": "overview", "scene": overview_scene, "camera": camera,
                              "classes": scene_classes, "fraction_class": FRACTION_CLASS})
    json_path = output / "spec_overview.json"
    json_path.write_text(json.dumps({"views": [overview]}, indent=2), encoding="utf-8")
    runtime.run_train_script("segmentation/render_figures.py", [
        "-m", files.model, "--spec", json_path, "--output_dir", output])

    # Candidates of the qualitative figure: every test scene, all its classes, both mask sources.
    # A class that a source never proposes enters the agreement with its reference only
    median = statistics.median(s["miou"] for s in scenes.values())
    for scene in sorted(scenes, key=lambda s: scenes[s]["miou"]):
        files = SceneFiles(args.data_root, view, selected, scene)
        scene_classes = [name for name in CLASS_COLOURS if name in classes[scene] and files.reference(name).exists()]
        views, entries = [], []
        for camera in spread_cameras(files, scene_classes, CANDIDATE_CAMERAS):
            panels = [{"name": "photo", "kind": "photo"}]
            for source in SOURCES:
                predicted = [n for n in scene_classes if files.prediction(n, source).exists()]
                panels.append({"name": f"labels_{source}", "kind": "labels",
                               "layers": [[to_container(files.prediction(n, source)), rgb(CLASS_COLOURS[n])]
                                          for n in predicted]})
                panels.append({"name": f"agreement_{source}", "kind": "agreement",
                               "pairs": [[to_container(files.prediction(n, source)) if n in predicted else None,
                                          to_container(files.reference(n))] for n in scene_classes]})
            views.append({"name": f"candidates/{scene}_{camera}", "camera": camera, "panels": panels})
            entries.append({"scene": scene, "camera": camera, "classes": scene_classes,
                            "miou": {s: miou[s].get(scene, {}).get("miou") for s in SOURCES}})
        json_path = output / "candidates" / f"spec_{scene}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps({"views": views}, indent=2), encoding="utf-8")
        # A scene that fails does not stop the others; the renderer saves every view it can before failing
        try:
            runtime.run_train_script("segmentation/render_figures.py", [
                "-m", files.model, "--spec", json_path, "--output_dir", output])
        except subprocess.CalledProcessError:
            print(f"some candidate views of {scene} failed, see the traceback above")
        manifest["candidates"] += [entry for entry in entries if all(
            (output / "candidates" / f"{scene}_{entry['camera']}_{key}.png").exists()
            for key in ("photo", "agreement_gt2d", "agreement_yolo"))]
    manifest["median_miou"] = median

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
