#!/usr/bin/env python3
"""Summarize full-budget ReWA runs against a fixed strong-L1 checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

try:
    from .plot_utils import PALETTE, apply_paper_style
except ImportError:  # Direct script execution.
    from plot_utils import PALETTE, apply_paper_style


TARGETS = (0.0, 0.5, 0.7, 0.8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--l1-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--l1-ppl70", type=float, default=13.973123700231621)
    parser.add_argument("--l1-ppl80", type=float, default=14.693895971726272)
    parser.add_argument("--max-curves", type=int, default=6)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_from_manifest(path: str, manifest: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    project_root = manifest.resolve().parents[2]
    return project_root / candidate


def pruning_points(path: Path) -> dict[float, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        round(float(point["target_sparsity"]), 12): point
        for point in payload["results"]
    }


def collect_runs(
    manifest: Path, l1_ppl70: float, l1_ppl80: float
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for row in read_csv(manifest):
        if row.get("status") != "completed":
            continue
        pruning_path = resolve_from_manifest(row["pruning_json"], manifest)
        if not pruning_path.exists():
            continue
        try:
            points = pruning_points(pruning_path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if any(target not in points for target in TARGETS):
            continue
        ppl = {target: float(points[target]["perplexity"]) for target in TARGETS}
        ratio70 = ppl[0.7] / l1_ppl70
        ratio80 = ppl[0.8] / l1_ppl80
        collected.append(
            {
                "run_id": row["run_id"],
                "seed": int(row["seed"]),
                "rewa_k": float(row["rewa_k"]),
                "rewa_m": float(row["rewa_m"]),
                "learning_rate": float(row["learning_rate"]),
                "rewa_eps": float(row["rewa_eps"]),
                "rewa_weight_decay": float(row["rewa_weight_decay"]),
                "train_tokens": int(row["train_tokens"]),
                "best_val_ppl": float(row["best_val_ppl"]),
                "ppl_0": ppl[0.0],
                "ppl_50": ppl[0.5],
                "ppl_70": ppl[0.7],
                "ppl_80": ppl[0.8],
                "ratio_to_l1_70": ratio70,
                "ratio_to_l1_80": ratio80,
                "worst_l1_ratio": max(ratio70, ratio80),
                "beats_l1_both": ratio70 < 1.0 and ratio80 < 1.0,
                "pruning_json": str(pruning_path),
            }
        )
    return sorted(
        collected,
        key=lambda item: (
            not item["beats_l1_both"],
            item["worst_l1_ratio"],
            item["ppl_0"],
            item["run_id"],
        ),
    )


def l1_curve(path: Path) -> dict[float, float]:
    points: dict[float, float] = {}
    for row in read_csv(path):
        target = round(float(row["target_sparsity"]), 12)
        if target in TARGETS:
            points[target] = float(row["perplexity"])
    missing = set(TARGETS) - set(points)
    if missing:
        raise ValueError(f"L1 curve is missing sparsities: {sorted(missing)}")
    return points


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "rank", "run_id", "seed", "rewa_k", "rewa_m", "learning_rate",
        "rewa_eps", "rewa_weight_decay", "train_tokens", "best_val_ppl",
        "ppl_0", "ppl_50", "ppl_70", "ppl_80", "ratio_to_l1_70",
        "ratio_to_l1_80", "worst_l1_ratio", "beats_l1_both", "pruning_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            writer.writerow({"rank": rank, **row})


def label(row: dict[str, Any]) -> str:
    return (
        f"K{row['rewa_k']:g}/M{row['rewa_m']:g}, "
        f"lr={row['learning_rate']:g}, eps={row['rewa_eps']:g}, "
        f"wd={row['rewa_weight_decay']:g}"
    )


def make_plot(
    path: Path, rows: list[dict[str, Any]], l1: dict[float, float], max_curves: int
) -> None:
    apply_paper_style()
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    xs = [100 * target for target in TARGETS]
    axis.plot(
        xs, [l1[target] for target in TARGETS], color=PALETTE["l1_sparse"],
        marker="D", linestyle="--", linewidth=2.4, label="Strong L1 reference",
        zorder=5,
    )
    colors = plt.get_cmap("viridis")
    selected = rows[:max_curves]
    for index, row in enumerate(selected):
        ys = [row[f"ppl_{int(100 * target)}"] for target in TARGETS]
        axis.plot(
            xs, ys, marker="o", linewidth=1.7,
            color=colors(index / max(1, len(selected) - 1)), label=label(row),
        )
    axis.set_yscale("log")
    axis.set_xlabel("Eligible global sparsity (%)")
    axis.set_ylabel("Validation perplexity (log scale)")
    axis.set_xticks(xs)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=7.2, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def write_results(
    path: Path,
    rows: list[dict[str, Any]],
    l1: dict[float, float],
    fixed70: float,
    fixed80: float,
) -> None:
    lines = [
        "# Full-budget ReWA regularization search", "",
        "Every listed ReWA configuration completed its declared token budget and",
        "was ranked using the same checkpoint at 70% and 80% global sparsity.", "",
        f"Fixed strong-L1 targets: PPL {fixed70:.4f} at 70% and {fixed80:.4f} at 80%.",
        f"Supplied L1 curve: PPL {l1[0.0]:.4f} unpruned, {l1[0.7]:.4f} at 70%, "
        f"and {l1[0.8]:.4f} at 80%.", "",
        "| Rank | Configuration | PPL 0% | PPL 70% | PPL 80% | Worst/L1 | Beats both |",
        "| ---: | --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for rank, row in enumerate(rows, 1):
        lines.append(
            f"| {rank} | `{label(row)}` | {row['ppl_0']:.4f} | "
            f"{row['ppl_70']:.4f} | {row['ppl_80']:.4f} | "
            f"{row['worst_l1_ratio']:.4f} | "
            f"{'yes' if row['beats_l1_both'] else 'no'} |"
        )
    if not rows:
        lines.extend(["| - | No completed full-budget ReWA runs yet | - | - | - | - | - |"])
    winners = [row for row in rows if row["beats_l1_both"]]
    lines.extend(["", "## Decision", ""])
    if winners:
        lines.append(
            f"{len(winners)} run(s) beat the fixed strong-L1 thresholds at both target "
            "sparsities. The first table row is the minimax winner."
        )
    else:
        lines.append(
            "No completed run yet beats the fixed strong-L1 thresholds at both target "
            "sparsities; this is not a positive result."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not all(math.isfinite(value) and value > 0 for value in (args.l1_ppl70, args.l1_ppl80)):
        raise ValueError("L1 thresholds must be finite and positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_runs(args.manifest, args.l1_ppl70, args.l1_ppl80)
    l1 = l1_curve(args.l1_csv)
    write_summary_csv(args.output_dir / "full-budget-comparison.csv", rows)
    write_results(
        args.output_dir / "FULL_BUDGET_RESULTS.md", rows, l1,
        args.l1_ppl70, args.l1_ppl80,
    )
    if rows:
        make_plot(
            args.output_dir / "full-budget-pruning-comparison.png",
            rows, l1, args.max_curves,
        )
    print(f"summarized {len(rows)} completed full-budget ReWA runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
