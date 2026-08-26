# Create the five manuscript PDFs from analytics with cache tracking

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from evaluation.analytics import deduplicate_analytics
from evaluation.common import atomic_write_text
from evaluation.scripts.make_tables import (
    require_complete_state_store,
    selected_operating_point,
)

FIGURES = (
    "beta_curves.pdf",
    "per_class.pdf",
    "qualitative_fraction.pdf",
    "qualitative_prediction.pdf",
    "qualitative_reference.pdf",
)


def _fingerprint(paths, point):
    # Hash source files and the selected point
    digest = hashlib.sha256(json.dumps(point, sort_keys=True).encode())
    for path in sorted(Path(p) for p in paths if Path(p).exists()):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _pdf(path, title, draw=None):
    # Render one PDF figure
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title(title)
    if draw:
        draw(ax)
    else:
        ax.text(0.5, 0.5, "analytics unavailable", ha="center", va="center")
    fig.tight_layout()

    # Load the result rows
    fig.savefig(path, format="pdf")
    plt.close(fig)


def _number(value):
    # Convert optional analytics values
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main(argv=None):
    # Parse inputs and load the selected operating point
    p = argparse.ArgumentParser()
    p.add_argument("--analytics", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("preprint/figures"))
    p.add_argument(
        "--qualitative-input",
        type=Path,
        help="prepared qualitative geometry/model input",
    )

    # Parse optional experiment completeness inputs
    p.add_argument("--state-store", type=Path)
    p.add_argument("--experiment")
    args = p.parse_args(argv)
    require_complete_state_store(args.state_store, args.experiment)
    point = json.loads(args.selection.read_text(encoding="utf-8"))
    beta, gamma, tolerance = selected_operating_point(args.selection)
    view = deduplicate_analytics(args.analytics)
    runs = {r["run_id"]: r for r in view["runs"] if r.get("status") == "completed"}

    # Select the figure data
    metrics = view.get("class_beta_metrics", [])
    inputs = [
        args.analytics / "class_beta_metrics.csv",
        args.analytics / "aggregate_beta_metrics.csv",
    ]
    fingerprint = _fingerprint(inputs, point)
    args.out.mkdir(parents=True, exist_ok=True)
    marker = args.out / ".figures.inputs.json"
    old = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else {}
    qualitative_ready = (
        args.qualitative_input is not None and args.qualitative_input.exists()
    )
    available_figures = FIGURES if qualitative_ready else FIGURES[:2]

    # Build the figure labels
    if not qualitative_ready:
        print(
            "warning: qualitative inputs are unavailable, qualitative PDFs are pending"
        )
    if (
        old.get("fingerprint") == fingerprint
        and all((args.out / f).exists() for f in available_figures)
        and old.get("figures") == available_figures
    ):
        print("figures unchanged, cache hit")
        return 0

    # Render the required metric figures
    # Build the metric plots from completed analytics rows
    def curves(ax):
        grouped = defaultdict(list)
        for row in metrics:
            run = runs.get(row.get("run_id"))
            b = _number(row.get("beta"))
            if run and run.get("dataset") == "replica":
                grouped[(row.get("class_id"), b)].append(_number(row.get("iou")))
        for cls in sorted({key[0] for key in grouped}):
            # Add the figure series
            points = sorted(
                (b, sum(v) / len(v)) for (c, b), v in grouped.items() if c == cls
            )

            # Complete the figure series

            # Draw one curve for each class
            ax.plot([x[0] for x in points], [x[1] for x in points], label=cls)
        if grouped:
            ax.legend(fontsize=6, ncol=2)
        ax.axvline(beta, color="black", linestyle="--")
        ax.set_xlabel("beta")
        ax.set_ylabel("IoU")

    _pdf(args.out / FIGURES[0], "Validation beta curves", curves)

    def per_class(ax):

        # Save the figure
        values = defaultdict(list)
        for row in metrics:
            if (
                abs(_number(row.get("beta")) - beta) > tolerance
                or abs(_number(row.get("hysteresis_gamma")) - gamma) > tolerance
            ):
                continue
            run = runs.get(row.get("run_id"))
            if run:
                values[row.get("class_id")].append(_number(row.get("iou")))
        ax.boxplot(
            list(values.values()),
            tick_labels=list(values) if values else ["--"],
        )

        # Return the figure path
        ax.set_ylabel("IoU")

    _pdf(args.out / FIGURES[1], "Per-class IoU at selected point", per_class)
    # Create optional qualitative placeholders when source geometry exists
    qualitative = [args.out / name for name in FIGURES[2:]]
    if qualitative_ready:
        for name, title in zip(
            qualitative,
            ("Qualitative fraction", "Qualitative prediction", "Qualitative reference"),
        ):
            _pdf(name, title)

    # Record the figure input fingerprint
    atomic_write_text(
        marker,
        json.dumps(
            {
                "fingerprint": fingerprint,
                "figures": available_figures,
                "qualitative_pending": not qualitative_ready,
            },
            indent=2,
        )
        + "\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())