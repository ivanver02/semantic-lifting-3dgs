# Generate appendix table bodies from analytics

import argparse
import json
from collections import defaultdict
from pathlib import Path

from evaluation.analytics import deduplicate_analytics
from evaluation.common import atomic_write_text

CORPUS = {"replica": "Replica", "scannetpp": "ScanNet++"}


def number(value):
    # Convert optional numeric text
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values):
    # Average available values
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def cell(value, digits=2):
    # Format one table cell
    return "--" if value is None else f"{value:.{digits}f}"


def strip_last(lines):
    # Remove the final table separator
    lines = list(lines)
    while lines and lines[-1] == "\\addlinespace[3pt]":
        lines.pop()
    if not lines:
        return "\n"
    text = "\n".join(lines)
    return text[: text.rindex("\\\\")] + "\n" if "\\\\" in text else text + "\n"


def selected_operating_point(path, tolerance=1e-9):
    # Load the selected validation point
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        return float(data["beta_star"]), float(data["gamma_star"]), tolerance
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "selection JSON must contain beta_star and gamma_star"
        ) from error


def require_complete_state_store(path, experiment=None):
    # Reject incomplete experiment units
    if path is None:
        return
    from evaluation.state_store import read_state_store

    bad = [
        row
        for row in read_state_store(path).rows.values()
        if (experiment is None or row.get("experiment") == experiment)
        and row.get("state") in {"pending", "running", "failed"}
    ]
    if bad:
        raise RuntimeError(f"state store contains {len(bad)} incomplete units")


def _filtered(view, runs, beta, gamma, tolerance):
    # Yield completed rows at the requested point
    for row in view.get("class_beta_metrics", []):
        b, g = number(row.get("beta")), number(row.get("hysteresis_gamma"))
        if beta is not None and (b is None or abs(b - beta) > tolerance):
            continue
        if gamma is not None and (g is None or abs(g - gamma) > tolerance):
            continue
        run = runs.get(row.get("run_id"))
        if run and run.get("status") == "completed":
            # Select the table rows
            yield row, run