# Summaries of the analytics and of the two analyses, shared by the manuscript macros, tables and figures,
# so every number of the manuscripts comes from one place

import csv
import json
import math
import statistics
from collections import defaultdict, namedtuple
from pathlib import Path

from evaluation.analytics import close, dataset_of, is_frozen, number

DATASETS = {"replica": "Replica", "scannetpp": "ScanNet++"}
SOURCES = {"gt2d": "annotation-derived", "yolo": "YOLO"}
TRANSFERS = {"nearest_neighbor_label": "Nearest Gaussian", "radius_vote": "Radius vote"}
SCORES = {"fraction": "Evidence fraction", "evidence": "Raw evidence", "per_view": "Evidence per view"}

# The class order of the tables and figures, the four shared classes first
CLASS_ORDER = ("chair", "sofa", "table", "tv", "laptop", "sink", "plant", "clock", "bench")

# One operating point of one transfer operator
Point = namedtuple("Point", "transfer beta gamma")


def load_points(analytics):
    """
    The selection over both operators, which the test used, and the selection over the radius vote
    alone, which the contribution analysis varies one factor at a time, with their JSON contents
    """
    analytics = Path(analytics)
    selections = {}
    for key, name in (("selected", "selection_transfer.json"), ("radius", "selection.json")):
        path = analytics / name
        if path.exists():
            selections[key] = json.loads(path.read_text(encoding="utf-8"))
    if "selected" not in selections:
        # An older store has one selection only, over the radius vote
        selections["selected"] = selections["radius"]
    selections.setdefault("radius", selections["selected"])

    def point(data):
        return Point(data.get("transfer", "radius_vote"), float(data["beta_star"]), float(data["gamma_star"]))

    return point(selections["selected"]), point(selections["radius"]), selections


def at_point(row, beta, gamma):
    """ Whether a metric row belongs to one operating point """
    return close(number(row["beta"]), beta) and close(number(row["hysteresis_gamma"]), gamma)


def frozen_rows(view, table, point, dataset=None, source=None, gamma=None):
    """ Rows of the frozen configuration of one operator at its point, or at another gamma """
    gamma = point.gamma if gamma is None else gamma
    return [
        row for row in view[table]
        if is_frozen(row, point.transfer) and at_point(row, point.beta, gamma)
        and (dataset is None or dataset_of(row) == dataset) and (source is None or row["source"] == source)
    ]


def mean(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def sd(values):
    values = [value for value in values if value is not None]
    return statistics.pstdev(values) if len(values) > 1 else None


def reachable(row):
    """
    Whether a class of a scene can be reached through the Gaussian model, that is, whether the
    ground-truth transfer reference has a positive IoU for it. A pair that not even the annotation
    reaches measures the capture and not the method, so the summaries leave it out
    """
    return (number(row["ground_truth_transfer_iou"]) or 0.0) > 0.0


def scene_summary(view, point, dataset, source, gamma=None):
    """
    Scene values of one split and source, and their means: mIoU, precision, recall, reference and
    relative mIoU over the reachable classes of each scene, and the mIoU over every class as miou_all
    """
    classes = defaultdict(list)
    for row in frozen_rows(view, "class_beta_metrics", point, dataset, source, gamma):
        classes[row["scene_id"].split(":")[1]].append(row)
    scenes = {}
    for scene, rows in classes.items():
        kept = [row for row in rows if reachable(row)]
        if not kept:
            continue
        scenes[scene] = dict(
            miou=mean(number(row["iou"]) for row in kept),
            precision=mean(number(row["precision"]) for row in kept),
            recall=mean(number(row["recall"]) for row in kept),
            reference=mean(number(row["ground_truth_transfer_iou"]) for row in kept),
            relative=mean(number(row["iou"]) / number(row["ground_truth_transfer_iou"]) for row in kept),
            miou_all=mean(number(row["iou"]) for row in rows),
        )
    if not scenes:
        return None
    summary = {key: mean(item[key] for item in scenes.values()) for key in next(iter(scenes.values()))}
    summary.update(scenes=scenes, count=len(scenes), sd=sd(item["miou"] for item in scenes.values()),
                   median=statistics.median(item["miou"] for item in scenes.values()),
                   pairs=sum(len(rows) for rows in classes.values()),
                   unreachable=sum(not reachable(row) for rows in classes.values() for row in rows))
    return summary


def gaps(view, point, dataset):
    """ The three gaps of the error decomposition from the means of the split """
    annotation, detector = scene_summary(view, point, dataset, "gt2d"), scene_summary(view, point, dataset, "yolo")
    if annotation is None or detector is None:
        return None
    return dict(
        detector=annotation["miou"] - detector["miou"],
        lifting=annotation["reference"] - annotation["miou"],
        representation=1.0 - annotation["reference"],
    )


def paired(first, second):
    """ Mean difference of two scene summaries over their common scenes, and the scenes where the first is better """
    common = sorted(set(first["scenes"]) & set(second["scenes"]))
    differences = [first["scenes"][s]["miou"] - second["scenes"][s]["miou"] for s in common]
    return dict(
        delta=mean(differences), better=sum(d > 0 for d in differences),
        worse=sum(d < 0 for d in differences), count=len(common),
    )


def class_summary(view, point):
    """ IoU, precision and recall of every class per dataset and source, with its reference, over the scenes """
    names = {row["class_id"]: row["class_name"] for row in view["classes"]}
    values = defaultdict(lambda: defaultdict(list))
    for row in frozen_rows(view, "class_beta_metrics", point):
        if not reachable(row):
            continue
        key = (dataset_of(row), names[row["class_id"]])
        source = row["source"]
        values[key][source + ":iou"].append(number(row["iou"]))
        values[key][source + ":precision"].append(number(row["precision"]))
        values[key][source + ":recall"].append(number(row["recall"]))
        values[key][source + ":scenes"].append(row["scene_id"].split(":")[1])
        if source == "gt2d":
            # The reference only depends on the mesh annotation, so it is read once per scene
            values[key]["reference"].append(number(row["ground_truth_transfer_iou"]))
    return {
        key: dict(
            {field: mean(items) for field, items in item.items() if not field.endswith(":scenes")},
            scenes=len(item["gt2d:scenes"]),
            zero_yolo=sum(value == 0 for value in item["yolo:iou"]),
            per_scene=dict(zip(item["gt2d:scenes"], zip(item["gt2d:iou"], item["reference"]))),
        )
        for key, item in values.items()
    }


def ordered_classes(keys):
    """ Sort (dataset, class) keys by dataset and the class order of the manuscripts """
    rank = {name: index for index, name in enumerate(CLASS_ORDER)}
    return sorted(keys, key=lambda key: (list(DATASETS).index(key[0]), rank.get(key[1], len(rank)), key[1]))


def validation_grid(view, transfer, dataset="replica", source="gt2d"):
    """ Scene mIoU of every (beta, gamma) of the validation grid of one operator """
    grid = defaultdict(dict)
    for row in view["aggregate_beta_metrics"]:
        if is_frozen(row, transfer) and dataset_of(row) == dataset and row["source"] == source:
            key = (number(row["beta"]), number(row["hysteresis_gamma"]))
            grid[key][row["scene_id"]] = number(row["mIoU"])
    return grid


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def threshold_curves(path):
    """ IoU curves of the threshold analysis, as {(dataset, source, score): {(scene, class): [(t, iou)]}} """
    curves = defaultdict(lambda: defaultdict(list))
    for row in read_csv(path):
        key = (row["dataset"], row["source"], row["score"])
        curves[key][(row["scene"], row["class"])].append((float(row["threshold"]), float(row["iou"])))
    for key, items in curves.items():
        # A class without reference Gaussians has a zero curve for any score, so it compares nothing
        curves[key] = {item: sorted(curve) for item, curve in items.items() if max(iou for _, iou in curve) > 0}
    return curves


def _shared(items):
    """ The threshold with the best mean IoU over the items, and that mean """
    thresholds = [t for t, _ in next(iter(items.values()))]
    means = [statistics.mean(curve[i][1] for curve in items.values()) for i in range(len(thresholds))]
    best = max(range(len(thresholds)), key=means.__getitem__)
    return thresholds[best], means[best]


def _at(items, threshold):
    """ The mean IoU of the items at one threshold of the grid """
    return statistics.mean(dict(curve)[threshold] for curve in items.values())


def threshold_summary(path):
    """
    For every dataset, source and score: the mean of the best IoU of every item, which needs one
    threshold per class and scene, the mean IoU at the best threshold shared by all the items, the
    spread of the best thresholds, and, on ScanNet++, the mean IoU at the shared threshold of Replica
    """
    curves = threshold_curves(path)
    summary = {}
    for (dataset, source, score), items in curves.items():
        best = [max(curve, key=lambda point: point[1]) for curve in items.values()]
        threshold, shared = _shared(items)
        # The fraction is spread linearly, the two unbounded scores over orders of magnitude
        scale = (lambda t: t) if score == "fraction" else math.log10
        positions = sorted(scale(t) for t, _ in best)
        quartiles = statistics.quantiles(positions, n=4) if len(positions) > 1 else [positions[0]] * 3
        summary[(dataset, source, score)] = dict(
            items=len(items), oracle=statistics.mean(iou for _, iou in best),
            shared=shared, threshold=threshold,
            # The interquartile range of the best thresholds, and their whole range, which one item can set
            spread=quartiles[2] - quartiles[0], full_spread=positions[-1] - positions[0],
        )
    for (dataset, source, score), item in summary.items():
        replica = summary.get(("replica", source, score))
        if dataset == "scannetpp" and replica is not None:
            item["cross"] = _at(curves[(dataset, source, score)], replica["threshold"])
    return summary


def pearson(xs, ys):
    """ Pearson correlation of two samples, None when it is not defined """
    if len(xs) < 3:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy) if sx and sy else None


def mask_rows(path):
    """ Rows of the 2D mask analysis with numeric cells, for the reachable classes of each scene """
    rows = read_csv(path)
    for row in rows:
        for key, value in row.items():
            if key not in ("dataset", "scene", "name"):
                row[key] = number(value)
    return [row for row in rows if (row["ref"] or 0.0) > 0.0]


def mask_summary(path):
    """ Per dataset: pairs of scene and class, how many improve from 2D to 3D, the means and the correlation """
    rows = mask_rows(path)
    summary = {}
    for dataset in DATASETS:
        d = [row for row in rows if row["dataset"] == dataset and row["yolo3d"] is not None]
        if not d:
            continue
        summary[dataset] = dict(
            pairs=len(d), wins=sum(row["yolo3d"] > row["iou2d"] for row in d),
            iou2d=mean(row["iou2d"] for row in d), p2d=mean(row["p2d"] for row in d),
            r2d=mean(row["r2d"] for row in d), yolo3d=mean(row["yolo3d"] for row in d),
            pearson=pearson([row["iou2d"] for row in d], [row["yolo3d"] for row in d]),
        )
    return summary
