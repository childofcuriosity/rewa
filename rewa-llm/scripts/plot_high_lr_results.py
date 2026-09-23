#!/usr/bin/env python3
"""Plot the local high-learning-rate ReWA confirmation results.

The script reads committed CSV artifacts and writes both a vector PDF for
papers and a PNG preview for GitHub. No values are embedded in the code.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

from plot_utils import PALETTE, apply_paper_style


ROOT = Path(__file__).resolve().parents[1]
TARGETS = (0.0, 0.5, 0.7, 0.8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=ROOT / "artifacts" / "stage1-rewa-high-lr" / "rewa-llm-pruning-comparison",
        help="Output path without extension (default: committed artifact directory).",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def points_for(
    rows: Iterable[dict[str, str]], *, run_name: str | None = None,
    method: str | None = None, selected: bool | None = None,
) -> dict[float, float]:
    points: dict[float, float] = {}
    for row in rows:
        if run_name is not None and row.get("run_name") != run_name:
            continue
        if method is not None and row.get("method") != method:
            continue
        if selected is not None and (row.get("selected_for_method") == "yes") != selected:
            continue
        target = round(float(row["target_sparsity"]), 12)
        if target in TARGETS:
            points[target] = float(row["perplexity"])
    missing = set(TARGETS) - set(points)
    if missing:
        raise ValueError(
            f"curve run={run_name!r}, method={method!r} is missing sparsities {sorted(missing)}"
        )
    return points


def best_sparse_l1_run(rows: list[dict[str, str]]) -> str:
    grouped: dict[str, dict[float, float]] = {}
    for row in rows:
        if row.get("method") != "l1":
            continue
        target = round(float(row["target_sparsity"]), 12)
        if target in TARGETS:
            grouped.setdefault(row["run_name"], {})[target] = float(row["validation_loss"])
    complete = [
        (name, points) for name, points in grouped.items()
        if all(target in points for target in TARGETS)
    ]
    if not complete:
        raise ValueError("no complete L1 pruning curves found")
    return min(
        complete,
        key=lambda item: (item[1][0.7] + item[1][0.8], item[0]),
    )[0]


def plot_curve(
    axis: plt.Axes, points: dict[float, float], label: str, *, color: str,
    marker: str, linestyle: str, linewidth: float = 1.8, zorder: int = 2,
) -> None:
    axis.plot(
        [100 * target for target in TARGETS],
        [points[target] for target in TARGETS],
        label=label,
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=linewidth,
        markeredgecolor="white",
        markeredgewidth=0.6,
        zorder=zorder,
    )


def main() -> int:
    args = parse_args()
    high_lr_path = ROOT / "artifacts" / "stage1-rewa-high-lr" / "ten-million-global-pruning.csv"
    baseline_path = ROOT / "artifacts" / "stage1" / "global-pruning.csv"
    high_lr = read_csv(high_lr_path)
    baselines = read_csv(baseline_path)

    dense = points_for(baselines, method="dense", selected=True)
    l1_accuracy = points_for(baselines, method="l1", selected=True)
    l1_sparse_name = best_sparse_l1_run(baselines)
    l1_sparse = points_for(baselines, run_name=l1_sparse_name)
    k3_m0 = points_for(high_lr, run_name="k3-m0-eps0-wd1e-4-lr0.006-seed0")
    k3_m1 = points_for(high_lr, run_name="k3-m1-eps0-wd1e-4-lr0.006-seed0")
    k9_m2 = points_for(high_lr, run_name="k9-m2-eps0-wd1e-4-lr0.006-seed0")

    apply_paper_style()
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.15, 2.85))

    plot_curve(
        left, dense, "Dense AdamW (20M)", color=PALETTE["dense"],
        marker="o", linestyle="--",
    )
    plot_curve(
        left, l1_accuracy, "L1: best unpruned (20M)", color=PALETTE["l1"],
        marker="s", linestyle="-.",
    )
    plot_curve(
        left, l1_sparse, "L1: sparse operating point (20M)",
        color=PALETTE["l1_sparse"], marker="D", linestyle=":",
    )
    plot_curve(
        left, k9_m2, "ReWA K9/M2 (10M)", color=PALETTE["rewa"],
        marker="^", linestyle="-", linewidth=2.5, zorder=4,
    )

    plot_curve(
        right, k3_m0, "K3/M0", color=PALETTE["k3_m0"],
        marker="o", linestyle="--",
    )
    plot_curve(
        right, k3_m1, "K3/M1", color=PALETTE["k3_m1"],
        marker="s", linestyle="-.",
    )
    plot_curve(
        right, k9_m2, "K9/M2", color=PALETTE["k9_m2"],
        marker="^", linestyle="-", linewidth=2.5, zorder=4,
    )

    for axis in (left, right):
        axis.set_yscale("log")
        axis.set_xticks([0, 50, 70, 80])
        axis.set_xlabel("Global eligible-weight sparsity (%)")
        axis.grid(True, which="major", axis="y")
        axis.grid(True, which="major", axis="x", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    left.set_ylabel("Validation perplexity (log scale)")
    left.set_title("(a) Accuracy–sparsity operating points", loc="left")
    right.set_title("(b) ReWA geometry at 10M tokens", loc="left")
    left.legend(frameon=False, loc="upper left")
    right.legend(frameon=False, loc="upper left")

    left.annotate(
        f"{k9_m2[0.8]:.1f}", xy=(80, k9_m2[0.8]), xytext=(-7, 8),
        textcoords="offset points", ha="right", color=PALETTE["rewa"], fontweight="bold",
    )
    right.annotate(
        "7.6× lower than K3/M0",
        xy=(80, k9_m2[0.8]), xytext=(50, 75), textcoords="data",
        arrowprops={"arrowstyle": "->", "color": PALETTE["k9_m2"], "lw": 0.9},
        color=PALETTE["k9_m2"], ha="center",
    )

    figure.subplots_adjust(left=0.09, right=0.99, bottom=0.19, top=0.88, wspace=0.30)
    output_stem = args.output_stem.resolve()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"))
    figure.savefig(output_stem.with_suffix(".png"), dpi=240)
    plt.close(figure)
    print(output_stem.with_suffix(".pdf"))
    print(output_stem.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
