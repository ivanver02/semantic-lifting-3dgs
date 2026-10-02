# Create the manuscript figures from the analytics and the analyses, as PDFs sized for a column
# of the preprint (3.4 in) or for its full width (7 in)

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from evaluation.analytics import close, dataset_of, is_frozen, load_analytics, number
from evaluation.summaries import (
    CLASS_ORDER, DATASETS, TRANSFERS, belongs, class_summary, load_baseline, load_points, mask_rows,
    ordered_classes, threshold_summary, validation_grid,
)


CLASS_COLORS = {
    "chair": "#0072B2", "sofa": "#E69F00", "table": "#009E73", "tv": "#CC79A7",
    "laptop": "#56B4E9", "sink": "#D55E00", "plant": "#999933", "clock": "#000000", "bench": "#882255",
}
DATASET_MARKERS = {"replica": "o", "scannetpp": "^"}
# The views of the qualitative figure, as (scene, camera), taken from the candidates of figure_renders.py:
# two test scenes next to the median mIoU, one with a dining table that YOLO finds and one with a desk that it misses
QUALITATIVE_VIEWS = [("3db0a1c8f3", "DSC09808"), ("0d2ee665be", "DSC00097")]
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
    """
    Mean validation mIoU along beta, one curve per gamma, for both operators, with the band of the candidates
    within 0.01 of the best mean and the point that the rule chooses for each operator
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    means = {transfer: {key: statistics.mean(scenes.values()) for key, scenes in validation_grid(view, transfer).items()}
             for transfer in TRANSFERS}
    betas = sorted({beta for grid in means.values() for beta, _ in grid})
    gammas = sorted({gamma for grid in means.values() for _, gamma in grid})
    best = max(value for grid in means.values() for value in grid.values())
    low = 0.6
    shades = dict(zip([g for g in gammas if g > 0], ["#7AD151", "#22A884", "#2A788E", "#414487"]))

    points = (selected, radius) if selected.transfer != radius.transfer else (selected,)
    fig, axes = plt.subplots(len(points), 1, figsize=(3.4, 1.45 * len(points) + 0.55), sharex=True, sharey=True)
    axes = list(axes) if len(points) > 1 else [axes]
    for ax, point in zip(axes, points):
        grid = means[point.transfer]
        ax.axhspan(best - 0.01, best, color="#DDDDDD", linewidth=0, zorder=0)
        for gamma in gammas:
            xs = [i for i, b in enumerate(betas) if (b, gamma) in grid]
            ys = [grid[(betas[i], gamma)] for i in xs]
            clipped = [(x, y) for x, y in zip(xs, ys) if y < low]
            style = dict(color="#777777", linestyle="--") if gamma == 0 else dict(color=shades[gamma], linestyle="-")
            ax.plot(xs, ys, marker="o", markersize=2, linewidth=1.0, **style)
            if clipped:
                # The curve without hysteresis leaves the axis, and its last value is written at the border
                x, y = clipped[-1]
                ax.annotate(f"{y:.2f}", (x, low), xytext=(0, 3), textcoords="offset points", ha="center",
                            fontsize=5.5, color="#555555")
        star = next(i for i, b in enumerate(betas) if close(b, point.beta))
        ax.plot([star], [grid[(betas[star], point.gamma)]], marker="o", markersize=7, markerfacecolor="none",
                markeredgecolor="#D55E00", markeredgewidth=1.4, linestyle="none")
        ax.set_ylim(low, best + 0.02)
        ax.grid(axis="y", color="#EEEEEE", linewidth=0.6)
        ax.set_title(TRANSFERS[point.transfer], loc="left", fontsize=7)
    axes[-1].set_xticks(range(len(betas)), [f"{b:g}" for b in betas], rotation=90, fontsize=6)
    axes[-1].set_xlabel("$\\beta$")
    fig.supylabel("Mean validation mIoU", fontsize=7, x=0.02)
    handles = [Line2D([], [], color="#777777", linestyle="--", label="$\\gamma=0$")]
    handles += [Line2D([], [], color=shades[g], label=f"$\\gamma={g:g}$") for g in gammas if g > 0]
    handles += [Patch(color="#DDDDDD", label="Within 0.01 of the best"),
                Line2D([], [], marker="o", markersize=6, markerfacecolor="none", markeredgecolor="#D55E00",
                       linestyle="none", label="Chosen by the rule")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=6,
               handlelength=1.4, columnspacing=0.8)
    fig.tight_layout(rect=(0.02, 0.07, 1, 1))
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

    rows = [row for row in mask_rows(path_csv) if row["yolo3d"] is not None]
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=0.7)
    for colour, dataset in zip(("C0", "C1"), DATASETS):
        points = [(row["iou2d"], row["yolo3d"]) for row in rows if row["dataset"] == dataset]
        if points:
            ax.scatter(*zip(*points), s=12, marker=DATASET_MARKERS[dataset], color=colour, label=DATASETS[dataset])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("2D IoU of the YOLO masks")
    ax.set_ylabel("3D IoU")
    ax.legend(loc="lower right", fontsize=6.5)
    save(fig, path)


def class_gaps(view, selected, path):
    """ For every class, the IoU with YOLO masks, with annotation-derived masks and of the reference """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    summary = class_summary(view, selected)
    keys = ordered_classes(summary)
    # The same colours as in the rest of the figures, and a different marker for the reference
    sources = (("yolo:iou", "C1", "o", "YOLO"), ("gt2d:iou", "C0", "o", "Annotation"), ("reference", "0.4", "D", "Reference"))
    fig, ax = plt.subplots(figsize=(3.4, 3.3))
    labels, positions, y, previous = [], [], 0.0, None
    for key in keys:
        # A small gap between the classes of both datasets
        if previous is not None and key[0] != previous:
            y += 0.8
        previous = key[0]
        positions.append(y)
        labels.append(key[1])
        values = [summary[key].get(field) for field, _, _, _ in sources]
        present = [v for v in values if v is not None]
        ax.plot([min(present), max(present)], [y, y], color="#BBBBBB", linewidth=1.0, zorder=1)
        for value, (_, colour, marker, _) in zip(values, sources):
            if value is not None:
                ax.scatter(value, y, marker=marker, color=colour, s=14 if marker == "D" else 18, zorder=2)
        y += 1.0
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("IoU")
    # The name of each dataset beside its group of classes
    for dataset in DATASETS:
        rows = [p for p, key in zip(positions, keys) if key[0] == dataset]
        if rows:
            ax.text(-0.26, statistics.mean(rows), DATASETS[dataset], transform=ax.get_yaxis_transform(),
                    rotation=90, ha="center", va="center", fontsize=7)
    handles = [Line2D([], [], marker=marker, linestyle="", color=colour, label=label)
               for _, colour, marker, label in sources]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=3, fontsize=6.5,
              handletextpad=0.2, columnspacing=0.8)
    ax.grid(axis="x", color="#EEEEEE", linewidth=0.6)
    ax.set_axisbelow(True)
    save(fig, path)


def thresholds(view, selected, baseline, path):
    """
    Mean validation mIoU along the threshold grid, for the evidence fraction of the method and for the
    evidence per view of the baseline, without hysteresis and at the gamma selected for each one
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    def line(point, gamma):
        values = defaultdict(list)
        for row in view["aggregate_beta_metrics"]:
            if (belongs(row, point.transfer) and dataset_of(row) == "replica" and row["source"] == "gt2d"
                    and close(number(row["hysteresis_gamma"]), gamma)):
                values[number(row["beta"])].append(number(row["mIoU"]))
        return sorted((beta, statistics.mean(v)) for beta, v in values.items())

    fig, axes = plt.subplots(1, 2, figsize=(3.6, 1.9), sharey=True)
    panels = ((axes[0], selected, "Evidence fraction $\\rho$"), (axes[1], baseline, "Evidence per view"))
    for ax, point, label in panels:
        for gamma, style in ((point.gamma, "-"), (0.0, "--")):
            points = line(point, gamma)
            if not points:
                continue
            ax.plot(range(len(points)), [v for _, v in points], style, color="#0072B2" if point is selected else "#D55E00",
                    marker="o", markersize=2)
            ax.set_xticks(range(len(points)), [f"{b:g}" for b, _ in points], rotation=90, fontsize=5)
        star = [b for b, _ in line(point, point.gamma)]
        if star:
            ax.axvline(min(range(len(star)), key=lambda i: abs(star[i] - point.beta)), color="#777777", linestyle=":", linewidth=0.7)
        ax.set_xlabel(label, fontsize=7)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color="#EEEEEE", linewidth=0.6)
    axes[0].set_ylabel("Mean mIoU, validation")
    handles = [Line2D([], [], color="#444444", linestyle="-", label="With hysteresis"),
               Line2D([], [], color="#444444", linestyle="--", label="Without hysteresis")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=6)
    fig.tight_layout()
    save(fig, path)


def overview(renders, path):
    """ The photograph, the YOLO masks, the evidence fraction and the labelled Gaussians of one view """
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch

    manifest = json.loads((renders / "manifest.json").read_text(encoding="utf-8"))
    info = next(item for item in manifest["views"] if item["name"] == "overview")
    titles = {"photo": "Photograph", "masks": "YOLO masks of this view",
              "fraction": f"Evidence fraction of the {info['fraction_class']}", "labels": "Labelled Gaussians"}
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.55))
    for ax, (key, title) in zip(axes, titles.items()):
        ax.imshow(mpimg.imread(renders / f"overview_{key}.png"), interpolation="none")
        ax.set_axis_off()
        ax.set_title(title, fontsize=7, pad=3)
    handles = [Patch(color=CLASS_COLORS[name], label=name) for name in info["classes"]]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.25, -0.02), ncol=len(handles), fontsize=6,
               handlelength=1.0, columnspacing=0.8)
    # The colour scale of the fraction panel, from the pale colour of the room to the strongest red
    scale = LinearSegmentedColormap.from_list("fraction", ["#E6E6E6", "#D62728"])
    fig.subplots_adjust(wspace=0.03, left=0.005, right=0.995, top=0.9, bottom=0.12)
    bar = fig.add_axes([axes[2].get_position().x0 + 0.03, 0.06, axes[2].get_position().width - 0.06, 0.035])
    bar.imshow([np.linspace(0, 1, 256)], aspect="auto", cmap=scale)
    bar.set_yticks([])
    bar.set_xticks([0, 255], ["$\\rho=0$", "$\\rho=1$"], fontsize=5)
    bar.tick_params(length=1.5, pad=1)
    save(fig, path)


def qualitative(renders, path, views=QUALITATIVE_VIEWS):
    """
    One row per view of the candidates of figure_renders.py: the photograph and the agreement between the
    prediction and the reference over all the classes of the scene, with each mask source
    """
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    manifest = json.loads((renders / "manifest.json").read_text(encoding="utf-8"))
    candidates = {(item["scene"], item["camera"]): item for item in manifest.get("candidates", [])}
    views = [v for v in views if v in candidates]
    if not views:
        return
    columns = (("photo", "Photograph"), ("agreement_gt2d", "Annotation masks"), ("agreement_yolo", "YOLO masks"))
    fig, axes = plt.subplots(len(views), 3, figsize=(7.0, 1.62 * len(views) + 0.2), squeeze=False)
    for row, (scene, camera) in zip(axes, views):
        info = candidates[(scene, camera)]
        for ax, (key, title) in zip(row, columns):
            ax.imshow(mpimg.imread(renders / "candidates" / f"{scene}_{camera}_{key}.png"), interpolation="none")
            ax.set_axis_off()
            if key == "photo":
                title = f"Scene {scene}"
            else:
                title = f"{title}, mIoU {info['miou'][key.split('_')[1]]:.2f}"
            ax.set_title(title, fontsize=7, pad=3)
    colours = {"both": "#2CA02C", "prediction": "#D62728", "reference": "#1F77B4"}
    labels = {"both": "Prediction and reference", "prediction": "Only the prediction", "reference": "Only the reference"}
    fig.legend(handles=[Patch(color=colours[k], label=labels[k]) for k in colours], loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=6.5)
    fig.subplots_adjust(wspace=0.03, hspace=0.14, left=0.005, right=0.995, top=0.95,
                        bottom=0.24 / (1.62 * len(views) + 0.2))
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
    for ax, (phase, label) in zip(axes, (("tau", "Transfer radius $\\tau$ (m)"), ("theta", "Minimum vote fraction $\\theta$"))):
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
    handles += [Line2D([], [], color="#444444", linestyle="-", label="Prediction"),
                Line2D([], [], color="#444444", linestyle="--", label="Reference")]
    axes[1].legend(handles=handles, loc="lower left", fontsize=6)
    save(fig, path)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--analytics", type=Path, required=True,
                   help="analytics directory, with selection_transfer.json and selection.json")
    p.add_argument("--analysis", type=Path, default=None,
                   help="directory with mask_agreement.csv and threshold_scores.csv, skipped when missing")
    p.add_argument("--out", type=Path, default=Path("preprint/figures"))
    p.add_argument("--renders", type=Path, default=None,
                   help="directory written by figure_renders.py, for the overview and qualitative figures")
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
    baseline, _ = load_baseline(args.analytics)
    if baseline is not None:
        thresholds(view, selected, baseline, args.out / "thresholds.pdf")
    if args.renders is not None and (args.renders / "manifest.json").exists():
        overview(args.renders, args.out / "overview.pdf")
        qualitative(args.renders, args.out / "qualitative.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
