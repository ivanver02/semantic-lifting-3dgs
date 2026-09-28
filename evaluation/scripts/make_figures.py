# Create the manuscript metric PDFs from analytics

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

from evaluation.analytics import close, dataset_of, is_frozen, load_analytics, number, selected_operating_point


DATASETS = {"replica": "Replica", "scannetpp": "ScanNet++"}
DATASET_COLORS = {"replica": "tab:blue", "scannetpp": "tab:orange"}


def _pdf(path, title, draw):
    # Render one PDF figure
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title(title)
    draw(ax)
    fig.tight_layout()
    fig.savefig(path, format="pdf")
    plt.close(fig)


def main(argv=None):
    import matplotlib.pyplot as plt

    # Parse inputs and load the selected operating point
    p = argparse.ArgumentParser()
    p.add_argument("--analytics", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("preprint/figures"))
    args = p.parse_args(argv)

    beta, gamma = selected_operating_point(args.selection)
    view = load_analytics(args.analytics)
    names = {row["class_id"]: row["class_name"] for row in view["classes"]}

    # Both figures read the frozen configuration under annotation-derived masks
    metrics = [
        row for row in view["class_beta_metrics"]
        if is_frozen(row) and row["source"] == "gt2d"
    ]
    args.out.mkdir(parents=True, exist_ok=True)

    # Figure 1: one line per class under ground-truth masks, with hysteresis
    # disabled and at the selected gamma, matching the manuscript caption
    def curves(ax):
        series = defaultdict(lambda: defaultdict(list))
        for row in metrics:
            if dataset_of(row) != "replica":
                continue
            row_gamma = number(row["hysteresis_gamma"])
            if close(row_gamma, gamma):
                setting = "hysteresis"
            elif close(row_gamma, 0.0):
                setting = "no_hysteresis"
            else:
                continue
            series[(row["class_id"], setting)][number(row["beta"])].append(number(row["iou"]))

        # Average the scenes for every beta and draw one color per class,
        # solid at gammaStar and dashed with hysteresis disabled
        classes = sorted({class_id for class_id, _ in series})
        for index, class_id in enumerate(classes):
            color = plt.cm.tab10(index % 10)
            for setting, style in (("hysteresis", "-"), ("no_hysteresis", "--")):
                points = sorted(
                    (b, statistics.mean(values))
                    for b, values in series.get((class_id, setting), {}).items()
                )
                if points:
                    ax.plot(
                        [point[0] for point in points], [point[1] for point in points],
                        style, color=color,
                        label=names[class_id] if setting == "hysteresis" else None,
                    )
        if series:
            ax.legend(fontsize=6, ncol=2)
        ax.axvline(beta, color="black", linestyle=":")
        ax.set_xlabel("beta")
        ax.set_ylabel("IoU")

    _pdf(args.out / "beta_curves.pdf", "Validation beta curves", curves)

    # Figure 2: per-class IoU across the scenes of each dataset against the
    # number of annotated vertices of the class
    def per_class(ax):
        counts = {
            (row["scene_id"], row["class_id"]): number(row["gt_evaluated_vertex_count"])
            for row in view["scene_classes"]
        }

        # Gather the IoU values at the selected operating point, and the
        # annotated vertices of the class in the same scenes
        values, vertices = defaultdict(list), defaultdict(list)
        for row in metrics:
            if close(number(row["beta"]), beta) and close(number(row["hysteresis_gamma"]), gamma):
                key = (dataset_of(row), row["class_id"])
                values[key].append(number(row["iou"]))
                vertices[key].append(counts[(row["scene_id"], row["class_id"])])

        # Each box sits at the mean number of annotated vertices of its class,
        # on a logarithmic axis so that small and large classes remain readable
        for (dataset, class_id), iou_values in values.items():
            position = max(statistics.mean(vertices[(dataset, class_id)]), 1.0)
            box = ax.boxplot([iou_values], positions=[position], widths=[0.3 * position],
                             patch_artist=True, manage_ticks=False)
            box["boxes"][0].set_facecolor(DATASET_COLORS[dataset])
            ax.text(position, 1.01, names[class_id], transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=6, rotation=90)

        ax.set_xscale("log")
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color) for color in DATASET_COLORS.values()]
        ax.legend(handles, list(DATASETS.values()), fontsize=6)
        ax.set_xlabel("annotated vertices of the class")
        ax.set_ylabel("IoU")

    _pdf(args.out / "per_class.pdf", "Per-class IoU at selected point", per_class)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
