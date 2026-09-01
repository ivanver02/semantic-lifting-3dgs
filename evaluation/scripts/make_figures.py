# Create the manuscript metric PDFs from analytics

import argparse
from collections import defaultdict
from pathlib import Path

from evaluation.analytics import deduplicate_analytics
from evaluation.scripts.make_tables import selected_operating_point


FIGURES = (
    "beta_curves.pdf",
    "per_class.pdf",
)


def _number(value):
    # Convert an optional analytics cell; None keeps malformed rows out
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_ablation(row):
    return (row.get("variant") or "").startswith("contribution_analysis_")


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

    beta, gamma, tolerance = selected_operating_point(args.selection)
    view = deduplicate_analytics(args.analytics)
    runs = {r["run_id"]: r for r in view["runs"] if r.get("status") == "completed"}
    metrics = view.get("class_beta_metrics", [])
    args.out.mkdir(parents=True, exist_ok=True)

    # Figure 1: one line per class under ground-truth masks, with hysteresis
    # disabled and at the selected gamma, matching the manuscript caption
    def curves(ax):
        series = defaultdict(lambda: defaultdict(list))
        for row in metrics:
            run = runs.get(row.get("run_id"))
            if not run or run.get("dataset") != "replica":
                continue
            if row.get("source") != "gt2d" or _is_ablation(row):
                continue
            b, g, iou = (
                _number(row.get("beta")),
                _number(row.get("hysteresis_gamma")),
                _number(row.get("iou")),
            )
            if b is None or g is None or iou is None:
                continue
            if abs(g - gamma) <= tolerance:
                setting = "hysteresis"
            elif g == 0.0:
                setting = "no_hysteresis"
            else:
                continue
            series[(row.get("class_id"), setting)][b].append(iou)

        # Average the scenes for every beta and draw one color per class,
        # solid at gammaStar and dashed with hysteresis disabled
        classes = sorted({cls for cls, _ in series})
        for index, cls in enumerate(classes):
            color = plt.cm.tab10(index % 10)
            for setting, style in (("hysteresis", "-"), ("no_hysteresis", "--")):
                points = sorted(
                    (b, sum(v) / len(v))
                    for b, v in series.get((cls, setting), {}).items()
                )
                if points:
                    ax.plot(
                        [p[0] for p in points], [p[1] for p in points],
                        style, color=color,
                        label=cls if setting == "hysteresis" else None,
                    )
        if series:
            ax.legend(fontsize=6, ncol=2)
        ax.axvline(beta, color="black", linestyle=":")
        ax.set_xlabel("beta")
        ax.set_ylabel("IoU")

    _pdf(args.out / FIGURES[0], "Validation beta curves", curves)

    # Figure 2: per-class IoU across the scenes of each dataset against the
    # number of annotated vertices of the class
    def per_class(ax):
        counts = {}
        for row in view.get("scene_classes", []):
            scene_id, class_id = row.get("scene_id"), row.get("class_id")
            vertices = _number(row.get("gt_evaluated_vertex_count"))
            if vertices is not None:
                counts[(scene_id, class_id)] = vertices

        # Gather the IoU values at the selected operating point
        values = defaultdict(list)
        for row in metrics:
            run = runs.get(row.get("run_id"))
            if not run:
                continue
            b, g, iou = (
                _number(row.get("beta")),
                _number(row.get("hysteresis_gamma")),
                _number(row.get("iou")),
            )
            if b is None or g is None or iou is None:
                continue
            if abs(b - beta) > tolerance or abs(g - gamma) > tolerance:
                continue
            values[(run.get("dataset"), row.get("class_id"))].append(iou)

        def mean_count(class_id):
            observed = [
                vertices for (_, cid), vertices in counts.items()
                if cid == class_id
            ]
            return sum(observed) / len(observed) if observed else 0.0

        items = sorted(values.items(), key=lambda kv: mean_count(kv[0][1]))
        dataset_colors = {"replica": "tab:blue", "scannetpp": "tab:orange"}
        all_values, positions, tick_labels, box_colors = [], [], [], []
        for (dataset, class_id), iou_values in items:
            all_values.append(iou_values)
            positions.append(mean_count(class_id))
            tick_labels.append(str(class_id))
            box_colors.append(dataset_colors.get(dataset, "gray"))

        if all_values:
            boxes = ax.boxplot(
                all_values, positions=positions, widths=None,
                tick_labels=[f"{position:.0f}" for position in positions],
                patch_artist=True,
            )
            for box, color in zip(boxes["boxes"], box_colors):
                box.set_facecolor(color)
            handles = [
                plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.5)
                for color in dict.fromkeys(box_colors)
            ]
            ax.legend(handles, list(dict.fromkeys(box_colors)), fontsize=6)
        ax.set_xlabel("annotated vertices of the class")
        ax.set_ylabel("IoU")

    _pdf(args.out / FIGURES[1], "Per-class IoU at selected point", per_class)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
