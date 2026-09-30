# Create the manuscript figures from the analytics and the analyses, as PDFs sized for a column
# of the preprint (3.4 in) or for its full width (7 in)

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

from evaluation.analytics import close, dataset_of, is_frozen, load_analytics, number
from evaluation.summaries import (
    CLASS_ORDER, DATASETS, TRANSFERS, class_summary, load_points, mask_rows, ordered_classes,
    threshold_summary, validation_grid,
)

# Okabe-Ito colours, which stay distinct for colour-blind readers, one per class of both datasets
CLASS_COLORS = {
    "chair": "#0072B2", "sofa": "#E69F00", "table": "#009E73", "tv": "#CC79A7",
    "laptop": "#56B4E9", "sink": "#D55E00", "plant": "#999933", "clock": "#000000", "bench": "#882255",
}
SOURCE_STYLE = {
    "yolo": dict(color="#E69F00", marker="o", label="YOLO masks"),
    "gt2d": dict(color="#0072B2", marker="o", label="annotation-derived masks"),
    "reference": dict(color="#555555", marker="D", label="reference"),
}
DATASET_MARKERS = {"replica": "o", "scannetpp": "^"}
SCORE_AXES = {
    "fraction": ("target evidence fraction $\\rho$", "linear"),
    "evidence": ("raw target evidence $E^{+}$", "log"),
    "per_view": ("$E^{+}$ per view with the class", "log"),
}


def style():
    import logging
    import matplotlib

    matplotlib.use("Agg")
    # The PDF backend warns about the timestamps of some font files, which does not affect the figures
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.1,
        "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def save(fig, path):
    import matplotlib.pyplot as plt

    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"wrote {path}")


def beta_gamma(view, selected, radius, path):
    """ Mean validation mIoU of every (beta, gamma) for both operators, with the eligible and selected cells """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle

    means = {transfer: {key: statistics.mean(scenes.values()) for key, scenes in validation_grid(view, transfer).items()}
             for transfer in TRANSFERS}
    betas = sorted({beta for grid in means.values() for beta, _ in grid})
    gammas = sorted({gamma for grid in means.values() for _, gamma in grid})
    best = max(value for grid in means.values() for value in grid.values())
    low = 0.80

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.3), sharex=True)
    for ax, point in zip(axes, (selected, radius) if selected.transfer != radius.transfer else (selected,)):
        grid = means[point.transfer]
        matrix = np.array([[grid.get((b, g), np.nan) for b in betas] for g in gammas])
        image = ax.imshow(matrix, cmap="viridis", vmin=low, vmax=best, aspect="auto", origin="lower")
        for i, gamma in enumerate(gammas):
            for j, beta in enumerate(betas):
                value = matrix[i, j]
                if np.isnan(value):
                    continue
                ax.text(j, i, f"{value:.2f}"[1:], ha="center", va="center", fontsize=6,
                        color="white" if value < (low + best) / 2 else "black")
                # Cells within the tolerance of the best mean of both operators
                if value >= best - 0.01:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="white", linewidth=0.8))
                if close(beta, point.beta) and close(gamma, point.gamma):
                    ax.add_patch(Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fill=False,
                                           edgecolor="#D55E00", linewidth=1.8))
        ax.set_yticks(range(len(gammas)), [f"{g:g}" for g in gammas])
        ax.set_ylabel("$\\gamma$")
        title = TRANSFERS[point.transfer] + (", selected" if point == selected else ", best point of its own rule")
        ax.set_title(title, loc="left")
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[-1].set_xticks(range(len(betas)), [f"{b:g}" for b in betas])
    axes[-1].set_xlabel("$\\beta$")
    bar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.015)
    bar.set_label("mean validation mIoU")
    bar.ax.tick_params(labelsize=6)
    save(fig, path)


def beta_curves(view, selected, path):
    """ IoU of each class along beta with the selected operator, at the selected gamma and without hysteresis """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    names = {row["class_id"]: row["class_name"] for row in view["classes"]}
    series = defaultdict(lambda: defaultdict(list))
    for row in view["class_beta_metrics"]:
        if not (is_frozen(row, selected.transfer) and dataset_of(row) == "replica" and row["source"] == "gt2d"):
            continue
        gamma = number(row["hysteresis_gamma"])
        setting = "hysteresis" if close(gamma, selected.gamma) else "none" if close(gamma, 0.0) else None
        if setting:
            series[(names[row["class_id"]], setting)][number(row["beta"])].append(number(row["iou"]))

    betas = sorted({beta for values in series.values() for beta in values})
    position = {beta: index for index, beta in enumerate(betas)}
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    classes = [name for name in CLASS_ORDER if (name, "hysteresis") in series]
    for name in classes:
        for setting, line in (("hysteresis", "-"), ("none", "--")):
            points = sorted((position[b], statistics.mean(v)) for b, v in series[(name, setting)].items())
            ax.plot(*zip(*points), line, color=CLASS_COLORS[name], linewidth=1.0 if setting == "hysteresis" else 0.8)
    ax.axvline(position[min(betas, key=lambda b: abs(b - selected.beta))], color="#777777", linestyle=":", linewidth=0.8)
    ax.set_xticks(range(len(betas)), [f"{b:g}" for b in betas], rotation=60)
    ax.set_xlabel("$\\beta$")
    ax.set_ylabel("IoU, validation")
    ax.set_ylim(0, 1)
    handles = [Line2D([], [], color=CLASS_COLORS[name], label=name) for name in classes]
    handles += [Line2D([], [], color="#444444", linestyle="-", label=f"$\\gamma={selected.gamma:g}$"),
                Line2D([], [], color="#444444", linestyle="--", label="$\\gamma=0$")]
    ax.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.36), fontsize=6,
              handlelength=1.6, columnspacing=0.8)
    save(fig, path)


def score_bars(path_csv, path):
    """
    For each score, the IoU of the Gaussians with the best threshold of every class and scene next to
    the IoU with one threshold shared by all of them, with the annotation-derived masks
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    summary = threshold_summary(path_csv)
    labels = {"fraction": "fraction", "evidence": "raw\nevidence", "per_view": "evidence\nper view"}
    colours = {"oracle": "#9ecae1", "shared": "#08519c"}
    fig, axes = plt.subplots(1, 2, figsize=(3.6, 2.2), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        for position, score in enumerate(labels):
            item = summary.get((dataset, "gt2d", score))
            if item is None:
                continue
            for offset, key in ((-0.19, "oracle"), (0.19, "shared")):
                ax.bar(position + offset, item[key], width=0.36, color=colours[key])
                ax.text(position + offset, item[key] + 0.015, f"{item[key]:.2f}", ha="center", va="bottom", fontsize=4.8)
        ax.set_xticks(range(len(labels)), list(labels.values()), fontsize=6.5)
        ax.set_ylim(0, 1.05)
        ax.set_title(DATASETS[dataset], fontsize=7.5)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("IoU of the Gaussians")
    handles = [Patch(color=colours["oracle"], label="its own threshold per class and scene"),
               Patch(color=colours["shared"], label="one threshold for all")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=1, fontsize=6)
    fig.tight_layout()
    save(fig, path)


def mask_2d_3d(path_csv, path):
    """ Pixel IoU of YOLO against the annotation masks in 2D against the 3D IoU that lifting reaches from it """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = mask_rows(path_csv)
    fig, ax = plt.subplots(figsize=(3.4, 3.1))
    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=0.7)
    for row in rows:
        if row["yolo3d"] is None:
            continue
        ax.scatter(row["iou2d"], row["yolo3d"], s=16, marker=DATASET_MARKERS[row["dataset"]],
                   facecolor=CLASS_COLORS.get(row["name"], "#777777"), edgecolor="white", linewidth=0.4, alpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("2D IoU of the YOLO masks, pixels")
    ax.set_ylabel("3D IoU with YOLO masks, vertices")
    names = [name for name in CLASS_ORDER if any(row["name"] == name for row in rows)]
    handles = [Line2D([], [], marker="s", linestyle="", color=CLASS_COLORS[name], label=name) for name in names]
    handles += [Line2D([], [], marker=DATASET_MARKERS[d], linestyle="", color="#555555", label=DATASETS[d])
                for d in DATASETS]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=5, fontsize=6,
              handletextpad=0.2, columnspacing=0.8)
    save(fig, path)


def class_gaps(view, selected, path):
    """ For every class, the IoU with YOLO masks, with annotation-derived masks and of the reference """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    summary = class_summary(view, selected)
    keys = ordered_classes(summary)
    fig, ax = plt.subplots(figsize=(3.4, 3.3))
    labels, y, positions = [], 0.0, []
    previous = None
    for key in keys:
        if previous is not None and key[0] != previous:
            y += 0.8
        previous = key[0]
        positions.append(y)
        labels.append(key[1])
        item = summary[key]
        values = [item.get("yolo:iou"), item["gt2d:iou"], item["reference"]]
        ax.plot([min(values), max(values)], [y, y], color="#BBBBBB", linewidth=1.0, zorder=1)
        for source, value in zip(("yolo", "gt2d", "reference"), values):
            s = SOURCE_STYLE[source]
            ax.scatter(value, y, marker=s["marker"], color=s["color"], s=18 if source != "reference" else 14, zorder=2)
        y += 1.0
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("IoU at the selected point")
    # Dataset names beside their groups
    for dataset in DATASETS:
        rows = [p for p, key in zip(positions, keys) if key[0] == dataset]
        if rows:
            ax.text(-0.26, statistics.mean(rows), DATASETS[dataset], transform=ax.get_yaxis_transform(),
                    rotation=90, ha="center", va="center", fontsize=7)
    handles = [Line2D([], [], marker=SOURCE_STYLE[s]["marker"], linestyle="", color=SOURCE_STYLE[s]["color"],
                      label=SOURCE_STYLE[s]["label"]) for s in ("yolo", "gt2d", "reference")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=3, fontsize=6,
              handletextpad=0.2, columnspacing=0.8)
    ax.grid(axis="x", color="#EEEEEE", linewidth=0.6)
    ax.set_axisbelow(True)
    save(fig, path)


def development(view, development_selection, path):
    """ mIoU and reference of the two development scenes along the tau grid and then along the theta grid """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    parameters = {row["run_id"]: row for row in view["run_parameters"]}
    phases = {"tau": defaultdict(list), "theta": defaultdict(list)}
    for row in view["aggregate_beta_metrics"]:
        if not row["variant"].startswith("development_") or row["source"] != "gt2d":
            continue
        run = parameters[row["run_id"]]
        phase, value = ("tau", number(run["tau"])) if row["variant"].startswith("development_tau_") \
            else ("theta", number(run["min_fraction"]))
        phases[phase][dataset_of(row)].append((value, number(row["mIoU"]), number(row["ground_truth_transfer_mIoU"])))
    if not any(phases.values()):
        return

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.2), sharey=True)
    colors = {"replica": "#0072B2", "scannetpp": "#D55E00"}
    for ax, (phase, label) in zip(axes, (("tau", "transfer radius $\\tau$ (m)"), ("theta", "minimum vote fraction $\\theta$"))):
        for dataset, points in phases[phase].items():
            points.sort()
            xs = [p[0] for p in points]
            ax.plot(xs, [p[1] for p in points], "-o", color=colors[dataset], markersize=2.5)
            ax.plot(xs, [p[2] for p in points], "--", color=colors[dataset], linewidth=0.8)
        if phase in development_selection:
            ax.axvline(development_selection[phase], color="#777777", linestyle=":", linewidth=0.8)
        ax.set_xlabel(label)
        ax.grid(axis="y", color="#EEEEEE", linewidth=0.6)
    axes[0].set_ylabel("mIoU, development scenes")
    handles = [Line2D([], [], color=colors[d], label=DATASETS[d]) for d in colors]
    handles += [Line2D([], [], color="#444444", linestyle="-", label="prediction"),
                Line2D([], [], color="#444444", linestyle="--", label="reference")]
    axes[1].legend(handles=handles, loc="lower left", fontsize=6)
    save(fig, path)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--analytics", type=Path, required=True,
                   help="analytics directory, with selection_transfer.json and selection.json")
    p.add_argument("--analysis", type=Path, default=None,
                   help="directory with mask_agreement.csv and threshold_scores.csv, skipped when missing")
    p.add_argument("--out", type=Path, default=Path("preprint/figures"))
    args = p.parse_args(argv)
    style()

    view = load_analytics(args.analytics)
    selected, radius, selections = load_points(args.analytics)
    chosen = selections["selected"]
    args.out.mkdir(parents=True, exist_ok=True)
    beta_gamma(view, selected, radius, args.out / "beta_gamma.pdf")
    beta_curves(view, selected, args.out / "beta_curves.pdf")
    class_gaps(view, selected, args.out / "class_gaps.pdf")
    development(view, {"tau": chosen["tau_star"], "theta": chosen["theta_star"]}, args.out / "development.pdf")
    if args.analysis is not None and (args.analysis / "threshold_scores.csv").exists():
        score_bars(args.analysis / "threshold_scores.csv", args.out / "scores.pdf")
    if args.analysis is not None and (args.analysis / "mask_agreement.csv").exists():
        mask_2d_3d(args.analysis / "mask_agreement.csv", args.out / "mask_2d_3d.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
