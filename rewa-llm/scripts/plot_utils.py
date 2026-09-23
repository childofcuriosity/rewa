"""Shared plotting style for ReWA-LLM result figures."""

from __future__ import annotations

import matplotlib as mpl


PALETTE = {
    "rewa": "#0072B2",
    "dense": "#666666",
    "l1": "#D55E00",
    "l1_sparse": "#CC79A7",
    "k3_m0": "#009E73",
    "k3_m1": "#E69F00",
    "k9_m2": "#0072B2",
}


def apply_paper_style() -> None:
    """Apply a compact, colour-blind-safe style suitable for papers and GitHub."""

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )
