#!/usr/bin/env python3
"""Aggregate stage-one pilot artifacts into tables, a plot, and RESULTS.md."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("dense", "l1", "rewa")
METHOD_LABELS = {"dense": "Dense AdamW", "l1": "AdamW + L1", "rewa": "ReWA-AdamW"}

# Coarse, explicit reporting thresholds keep the generated interpretation
# deterministic.  They are diagnostics for this pilot, not substitutes for
# fresh-seed uncertainty estimates.
UNPRUNED_CLOSE_RELATIVE_TOLERANCE = 0.05
COLLAPSED_PPL_THRESHOLD = 1000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/stage1/run-manifest.csv")
    )
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/stage1"))
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--layer-csv", type=Path, default=None)
    parser.add_argument("--training-csv", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--envelope-csv", type=Path, default=None)
    parser.add_argument("--envelope-plot", type=Path, default=None)
    parser.add_argument("--results", type=Path, default=None)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_recorded_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = (PROJECT_ROOT / path, manifest_path.parent / path, Path.cwd() / path)
    return next((candidate.resolve() for candidate in candidates if candidate.exists()), candidates[0].resolve())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def config_text(row: dict[str, Any]) -> str:
    lr = as_float(row.get("learning_rate"))
    if row["method"] == "dense":
        return f"lr={lr:g}"
    if row["method"] == "l1":
        return f"lr={lr:g}, alpha={as_float(row.get('l1_alpha')):g}"
    return (
        f"lr={lr:g}, K={as_float(row.get('rewa_k')):g}, "
        f"M={as_float(row.get('rewa_m')):g}, "
        f"wd={as_float(row.get('rewa_weight_decay')):g}"
    )


def collect_pruning_rows(
    manifest: list[dict[str, str]], manifest_path: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for run in manifest:
        if run.get("status") != "completed":
            continue
        pruning_value = run.get("pruning_csv", "")
        if not pruning_value:
            warnings.append(f"{run.get('run_name', '<unknown>')}: pruning CSV is absent from manifest")
            continue
        pruning_path = resolve_recorded_path(pruning_value, manifest_path)
        if not pruning_path.exists():
            warnings.append(f"{run['run_name']}: missing {pruning_path}")
            continue
        for item in read_csv(pruning_path):
            merged: dict[str, Any] = {
                "run_name": run["run_name"],
                "method": run["method"],
                "seed": run.get("seed", ""),
                "learning_rate": run.get("learning_rate", ""),
                "l1_alpha": run.get("l1_alpha", ""),
                "rewa_k": run.get("rewa_k", ""),
                "rewa_m": run.get("rewa_m", ""),
                "rewa_weight_decay": run.get("rewa_weight_decay", ""),
                "rewa_eps": run.get("rewa_eps", ""),
                "checkpoint_best_val_loss": run.get("best_val_loss", ""),
            }
            merged.update(item)
            rows.append(merged)
    rows.sort(
        key=lambda row: (
            METHODS.index(row["method"]) if row["method"] in METHODS else 99,
            row["run_name"],
            as_float(row["target_sparsity"]),
        )
    )
    return rows, warnings


def collect_layer_rows(
    manifest: list[dict[str, str]], manifest_path: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in manifest:
        if run.get("status") != "completed" or not run.get("layer_csv"):
            continue
        path = resolve_recorded_path(run["layer_csv"], manifest_path)
        if not path.exists():
            continue
        for item in read_csv(path):
            rows.append(
                {
                    "run_name": run["run_name"],
                    "method": run["method"],
                    "seed": run.get("seed", ""),
                    **item,
                }
            )
    rows.sort(
        key=lambda row: (
            METHODS.index(row["method"]) if row["method"] in METHODS else 99,
            row["run_name"],
            as_float(row["target_sparsity"]),
            row["name"],
        )
    )
    return rows


def collect_training_rows(
    manifest: list[dict[str, str]], manifest_path: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in manifest:
        if run.get("status") == "superseded":
            continue
        value = run.get("metrics_jsonl", "")
        if not value:
            continue
        path = resolve_recorded_path(value, manifest_path)
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append(
                    {
                        "run_name": run["run_name"],
                        "method": run["method"],
                        "seed": run.get("seed", ""),
                        **item,
                    }
                )
    rows.sort(key=lambda row: (row["run_name"], int(row.get("iter", 0))))
    return rows


def select_best_runs(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    dense_rows = [
        row
        for row in rows
        if math.isclose(as_float(row["target_sparsity"]), 0.0, abs_tol=1e-12)
        and math.isfinite(as_float(row["validation_loss"]))
    ]
    selected: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        candidates = [row for row in dense_rows if row["method"] == method]
        if candidates:
            selected[method] = min(
                candidates,
                key=lambda row: (as_float(row["validation_loss"]), row["run_name"]),
            )
    return selected


def selected_curves(
    rows: list[dict[str, Any]], selected: dict[str, dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    curves: dict[str, list[dict[str, Any]]] = {}
    for method, best in selected.items():
        curves[method] = sorted(
            (row for row in rows if row["run_name"] == best["run_name"]),
            key=lambda row: as_float(row["target_sparsity"]),
        )
    return curves


def select_candidate_grid_envelope(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Select the lowest-PPL completed candidate at every method/sparsity point.

    ``collect_pruning_rows`` is the provenance boundary: it supplies only runs
    whose manifest status is exactly ``completed``, so superseded candidates do
    not enter this optimistic, per-point pilot selection.
    """

    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        method = row.get("method", "")
        target = as_float(row.get("target_sparsity"))
        perplexity = as_float(row.get("perplexity"))
        if method not in METHODS or not math.isfinite(target) or not math.isfinite(perplexity):
            continue
        grouped.setdefault((method, round(target, 12)), []).append(row)

    envelope: dict[str, list[dict[str, Any]]] = {}
    for (method, _target), candidates in grouped.items():
        best = min(
            candidates,
            key=lambda row: (
                as_float(row["perplexity"]),
                as_float(row.get("validation_loss"), math.inf),
                row["run_name"],
            ),
        )
        envelope.setdefault(method, []).append(best)
    for points in envelope.values():
        points.sort(key=lambda row: as_float(row["target_sparsity"]))
    return envelope


def flatten_candidate_grid_envelope(
    envelope: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Return explicit provenance rows for the candidate-grid envelope CSV."""

    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for point in envelope.get(method, []):
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "target_sparsity": point["target_sparsity"],
                    "perplexity": point["perplexity"],
                    "validation_loss": point.get("validation_loss", ""),
                    "selected_run_name": point["run_name"],
                    "selected_configuration": config_text(point),
                    "seed": point.get("seed", ""),
                    "learning_rate": point.get("learning_rate", ""),
                    "l1_alpha": point.get("l1_alpha", ""),
                    "rewa_k": point.get("rewa_k", ""),
                    "rewa_m": point.get("rewa_m", ""),
                    "rewa_weight_decay": point.get("rewa_weight_decay", ""),
                    "rewa_eps": point.get("rewa_eps", ""),
                    "checkpoint_best_val_loss": point.get(
                        "checkpoint_best_val_loss", ""
                    ),
                    "selected_count": point.get("selected_count", ""),
                    "total_eligible": point.get("total_eligible", ""),
                    "selection_sparsity": point.get("selection_sparsity", ""),
                    "eligible_sparsity": point.get("eligible_sparsity", ""),
                    "whole_model_sparsity": point.get("whole_model_sparsity", ""),
                    "attention_sparsity": point.get("attention_sparsity", ""),
                    "mlp_sparsity": point.get("mlp_sparsity", ""),
                }
            )
    rows.sort(
        key=lambda row: (
            as_float(row["target_sparsity"]),
            METHODS.index(row["method"]),
        )
    )
    return rows


def make_plot(
    path: Path,
    curves: dict[str, list[dict[str, Any]]],
    selected: dict[str, dict[str, Any]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required; install requirements.txt") from error

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    colors = {"dense": "#4C78A8", "l1": "#F58518", "rewa": "#54A24B"}
    markers = {"dense": "o", "l1": "s", "rewa": "^"}
    all_ppl: list[float] = []
    for method in METHODS:
        points = curves.get(method, [])
        if not points:
            continue
        x = [100.0 * as_float(point["target_sparsity"]) for point in points]
        y = [as_float(point["perplexity"]) for point in points]
        finite = [(x_value, y_value) for x_value, y_value in zip(x, y) if math.isfinite(y_value)]
        if not finite:
            continue
        x, y = map(list, zip(*finite))
        all_ppl.extend(y)
        label = f"{METHOD_LABELS[method]} ({config_text(selected[method])})"
        axis.plot(x, y, marker=markers[method], color=colors[method], linewidth=2, label=label)
    axis.set_xlabel("Eligible global sparsity (%)")
    axis.set_ylabel("Validation perplexity")
    axis.set_title("TinyStories stage-one single-seed pilot")
    axis.grid(True, alpha=0.25)
    if all_ppl and min(all_ppl) > 0 and max(all_ppl) / min(all_ppl) > 20:
        axis.set_yscale("log")
    if curves:
        axis.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_envelope_plot(
    path: Path, envelope: dict[str, list[dict[str, Any]]]
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required; install requirements.txt") from error

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    colors = {"dense": "#4C78A8", "l1": "#F58518", "rewa": "#54A24B"}
    markers = {"dense": "o", "l1": "s", "rewa": "^"}
    all_ppl: list[float] = []
    for method in METHODS:
        points = envelope.get(method, [])
        finite = [
            (100.0 * as_float(point["target_sparsity"]), as_float(point["perplexity"]))
            for point in points
            if math.isfinite(as_float(point["perplexity"]))
        ]
        if not finite:
            continue
        x, y = map(list, zip(*finite))
        all_ppl.extend(y)
        axis.plot(
            x,
            y,
            marker=markers[method],
            color=colors[method],
            linewidth=2,
            label=METHOD_LABELS[method],
        )
    axis.set_xlabel("Eligible global sparsity (%)")
    axis.set_ylabel("Validation perplexity")
    axis.set_title(
        "Candidate-grid oracle/Pareto envelope\n"
        "Per-point selection on the same validation set (optimistic pilot view)"
    )
    axis.grid(True, alpha=0.25)
    if all_ppl and min(all_ppl) > 0 and max(all_ppl) / min(all_ppl) > 20:
        axis.set_yscale("log")
    if envelope:
        axis.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def fmt(value: Any, digits: int = 4) -> str:
    number = as_float(value)
    if math.isnan(number):
        return "n/a"
    if math.isinf(number):
        return "inf"
    return f"{number:.{digits}f}"


def write_results(
    path: Path,
    manifest: list[dict[str, str]],
    rows: list[dict[str, Any]],
    selected: dict[str, dict[str, Any]],
    curves: dict[str, list[dict[str, Any]]],
    warnings: list[str],
    csv_path: Path,
    plot_path: Path,
    envelope: dict[str, list[dict[str, Any]]] | None = None,
    envelope_csv_path: Path | None = None,
    envelope_plot_path: Path | None = None,
) -> None:
    completed = [row for row in manifest if row.get("status") == "completed"]
    failed = [row for row in manifest if row.get("status") == "failed"]
    superseded = [row for row in manifest if row.get("status") == "superseded"]
    seeds = sorted({row.get("seed", "") for row in completed if row.get("seed", "") != ""})
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")

    lines = [
        "# Stage-one single-seed pilot results",
        "",
        (
            f"This exploratory TinyStories pilot contains {len(completed)} completed configuration(s) "
            f"at seed(s) {', '.join(seeds) if seeds else 'n/a'}. The curves compare exact global "
            "absolute-magnitude pruning over eligible attention and MLP matrices."
        ),
        "",
        f"Generated: `{generated}`.",
        "",
    ]
    dense_choice = next(
        (
            row
            for row in manifest
            if row.get("status") == "completed"
            and row.get("method") == "dense"
            and row.get("dense_lr_selected") == "yes"
        ),
        None,
    )
    if dense_choice is not None:
        lines.extend(
            [
                (
                    "The dense sweep selected "
                    f"`lr={as_float(dense_choice.get('learning_rate')):g}` from training-time "
                    "`best_val_loss`; every L1 and ReWA candidate used that learning rate."
                ),
                "",
            ]
        )
    lines.extend(["## Best unpruned configuration per method", ""])
    selection_rows: list[list[str]] = []
    for method in METHODS:
        row = selected.get(method)
        if row is None:
            selection_rows.append([METHOD_LABELS[method], "n/a", "n/a", "n/a"])
        else:
            selection_rows.append(
                [
                    METHOD_LABELS[method],
                    f"`{row['run_name']}`",
                    config_text(row),
                    fmt(row["perplexity"]),
                ]
            )
    lines.extend(
        [
            markdown_table(
                ["Method", "Selected run", "Configuration", "Unpruned PPL"], selection_rows
            ),
            "",
            (
                "Selection uses the common unpruned (`target_sparsity = 0`) validation evaluation. "
                "The same validation protocol then supplies every point on each selected curve."
            ),
            "",
            "## Selected global-pruning curves",
            "",
        ]
    )

    common_targets: set[float] | None = None
    maps: dict[str, dict[float, dict[str, Any]]] = {}
    for method, curve in curves.items():
        maps[method] = {round(as_float(row["target_sparsity"]), 12): row for row in curve}
        targets = set(maps[method])
        common_targets = targets if common_targets is None else common_targets & targets
    curve_rows: list[list[str]] = []
    for target in sorted(common_targets or set()):
        curve_rows.append(
            [
                f"{100 * target:.0f}%",
                *(fmt(maps[method][target]["perplexity"]) if method in maps else "n/a" for method in METHODS),
            ]
        )
    if curve_rows:
        lines.extend(
            [
                markdown_table(
                    ["Eligible sparsity", "Dense PPL", "L1 PPL", "ReWA PPL"], curve_rows
                ),
                "",
            ]
        )
    else:
        lines.extend(["No common sparsity grid is available across the selected methods.", ""])

    envelope_rows = flatten_candidate_grid_envelope(envelope or {})
    lines.extend(
        [
            "## Candidate-grid oracle/Pareto envelope (optimistic)",
            "",
            (
                "The candidate grid's lower envelope records the lowest observed validation "
                "perplexity at each tested sparsity within each method. Every point below names "
                "the run and configuration that produced it."
            ),
            "",
            (
                "Selection and reporting use the same validation set, and a method's envelope "
                "may switch runs between sparsity levels. The envelope is therefore an optimistic "
                "pilot diagnostic, not a held-out estimate or a single-run pruning curve."
            ),
            "",
        ]
    )
    if envelope_rows:
        lines.extend(
            [
                markdown_table(
                    [
                        "Eligible sparsity",
                        "Method",
                        "Lowest PPL",
                        "Chosen run",
                        "Chosen configuration",
                    ],
                    [
                        [
                            f"{100 * as_float(row['target_sparsity']):.0f}%",
                            row["method_label"],
                            fmt(row["perplexity"]),
                            f"`{row['selected_run_name']}`",
                            row["selected_configuration"],
                        ]
                        for row in envelope_rows
                    ],
                ),
                "",
            ]
        )
    else:
        lines.extend(["No finite candidate-grid envelope points are available.", ""])

    if all(method in maps for method in METHODS):
        nonzero = sorted(target for target in maps["rewa"] if target > 0 and target in maps["dense"] and target in maps["l1"])
        better_dense = sum(
            as_float(maps["rewa"][target]["perplexity"])
            < as_float(maps["dense"][target]["perplexity"])
            for target in nonzero
        )
        better_l1 = sum(
            as_float(maps["rewa"][target]["perplexity"])
            < as_float(maps["l1"][target]["perplexity"])
            for target in nonzero
        )
        selected_joint_wins = [
            target
            for target in nonzero
            if as_float(maps["rewa"][target]["perplexity"])
            < as_float(maps["dense"][target]["perplexity"])
            and as_float(maps["rewa"][target]["perplexity"])
            < as_float(maps["l1"][target]["perplexity"])
        ]
        collapsed_selected_wins = [
            target
            for target in selected_joint_wins
            if all(
                as_float(maps[method][target]["perplexity"])
                > COLLAPSED_PPL_THRESHOLD
                for method in METHODS
            )
        ]

        dense_unpruned = as_float(maps["dense"].get(0.0, {}).get("perplexity"))
        rewa_unpruned = as_float(maps["rewa"].get(0.0, {}).get("perplexity"))
        unpruned_gap = (
            rewa_unpruned / dense_unpruned - 1.0
            if math.isfinite(dense_unpruned)
            and dense_unpruned > 0
            and math.isfinite(rewa_unpruned)
            else math.nan
        )

        envelope_maps = {
            method: {
                round(as_float(row["target_sparsity"]), 12): row
                for row in (envelope or {}).get(method, [])
            }
            for method in METHODS
        }
        envelope_nonzero = sorted(
            target
            for target in envelope_maps["rewa"]
            if target > 0
            and target in envelope_maps["dense"]
            and target in envelope_maps["l1"]
        )
        envelope_joint_wins = [
            target
            for target in envelope_nonzero
            if as_float(envelope_maps["rewa"][target]["perplexity"])
            < as_float(envelope_maps["dense"][target]["perplexity"])
            and as_float(envelope_maps["rewa"][target]["perplexity"])
            < as_float(envelope_maps["l1"][target]["perplexity"])
        ]
        envelope_comparison_available = bool(envelope_nonzero) and all(
            (envelope or {}).get(method) for method in METHODS
        )

        if math.isfinite(unpruned_gap):
            direction = "higher" if unpruned_gap >= 0 else "lower"
            unpruned_observation = (
                f"The selected ReWA run's unpruned PPL is {abs(100.0 * unpruned_gap):.2f}% "
                f"{direction} than the selected dense run ({fmt(rewa_unpruned)} versus "
                f"{fmt(dense_unpruned)}). For this deterministic pilot summary, 'close' means "
                f"no more than {100.0 * UNPRUNED_CLOSE_RELATIVE_TOLERANCE:.0f}% above dense."
            )
        else:
            unpruned_observation = (
                "The relative unpruned ReWA-versus-dense PPL difference is unavailable."
            )
        envelope_observation = (
            f"On the optimistic candidate-grid envelope, ReWA is lower than both dense "
            f"and L1 at {len(envelope_joint_wins)} of {len(envelope_nonzero)} shared "
            "nonzero sparsity level(s)."
            if envelope_comparison_available
            else "A three-method candidate-grid envelope comparison is unavailable."
        )

        lines.extend(
            [
                "## Pilot observation",
                "",
                unpruned_observation,
                "",
                (
                    f"Across {len(nonzero)} shared nonzero sparsity levels, the selected ReWA run "
                    f"has lower perplexity than the selected dense run at {better_dense} level(s) "
                    f"and lower perplexity than the selected L1 run at {better_l1} level(s)."
                ),
                "",
                envelope_observation,
                "",
                *(
                    [
                        (
                            "At selected-curve joint-win target(s) "
                            + ", ".join(f"{100.0 * target:.0f}%" for target in collapsed_selected_wins)
                            + f", all three methods have PPL above {COLLAPSED_PPL_THRESHOLD:g}. "
                            "These collapsed-regime orderings are not treated as practical pruning "
                            "advantages."
                        ),
                        "",
                    ]
                    if collapsed_selected_wins
                    else []
                ),
                *(
                    [
                        (
                            "The frozen stage-one scale-up rule is not met: ReWA is not close to "
                            "dense before pruning and the candidate-grid envelope improves on both "
                            "baselines at no more than one shared nonzero sparsity level. The current "
                            "evidence therefore does not support scale-up."
                        ),
                        "",
                    ]
                    if math.isfinite(unpruned_gap)
                    and unpruned_gap > UNPRUNED_CLOSE_RELATIVE_TOLERANCE
                    and envelope_comparison_available
                    and len(envelope_joint_wins) <= 1
                    else []
                ),
                (
                    "This is a single-seed hyperparameter-selection pilot. A follow-up with fresh seeds "
                    "is the next measurement needed to estimate variance and confirm the observed ordering."
                ),
                "",
            ]
        )
    else:
        missing = [METHOD_LABELS[method] for method in METHODS if method not in maps]
        lines.extend(
            [
                "## Pilot status",
                "",
                f"A three-method comparison awaits completed results for: {', '.join(missing)}.",
                "",
            ]
        )

    lines.extend(
        [
            "## Artifacts",
            "",
            f"- Full configuration-by-sparsity table: `{csv_path.name}`",
            f"- Selected-curve plot: `{plot_path.name}`",
            *(
                [f"- Candidate-grid oracle/Pareto envelope table: `{envelope_csv_path.name}`"]
                if envelope_csv_path is not None
                else []
            ),
            *(
                [f"- Candidate-grid oracle/Pareto envelope plot: `{envelope_plot_path.name}`"]
                if envelope_plot_path is not None
                else []
            ),
            "- Per-layer table: `layer-sparsity.csv`",
            "- Aggregated training log: `training-curves.csv`",
            "- Run provenance and failures: `run-manifest.csv`",
            "",
        ]
    )
    if failed:
        lines.extend(["## Failed configurations", ""])
        for row in failed:
            lines.append(f"- `{row['run_name']}`: {row.get('error') or 'unspecified failure'}")
        lines.append("")
    if superseded:
        lines.extend(["## Superseded configurations", ""])
        lines.append(
            "These preserved runs were excluded from every aggregate table, selection, and curve."
        )
        lines.append("")
        for row in superseded:
            detail = row.get("status_detail") or (
                f"run learning rate {row.get('learning_rate', 'unknown')} does not match "
                "the final dense selection"
            )
            lines.append(f"- `{row['run_name']}`: {detail}")
        lines.append("")
    if warnings:
        lines.extend(["## Artifact warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    manifest_path = project_path(args.manifest)
    artifact_dir = project_path(args.artifact_dir)
    output_csv = project_path(args.output_csv) if args.output_csv else artifact_dir / "global-pruning.csv"
    layer_csv = project_path(args.layer_csv) if args.layer_csv else artifact_dir / "layer-sparsity.csv"
    training_csv = project_path(args.training_csv) if args.training_csv else artifact_dir / "training-curves.csv"
    plot_path = project_path(args.plot) if args.plot else artifact_dir / "ppl-vs-sparsity.png"
    envelope_csv_path = (
        project_path(args.envelope_csv)
        if args.envelope_csv
        else artifact_dir / "candidate-grid-oracle-envelope.csv"
    )
    envelope_plot_path = (
        project_path(args.envelope_plot)
        if args.envelope_plot
        else artifact_dir / "candidate-grid-oracle-envelope.png"
    )
    results_path = project_path(args.results) if args.results else artifact_dir / "RESULTS.md"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    manifest = read_csv(manifest_path)
    pruning_rows, warnings = collect_pruning_rows(manifest, manifest_path)
    if not pruning_rows:
        raise RuntimeError("no completed global-pruning CSVs were found")
    layer_rows = collect_layer_rows(manifest, manifest_path)
    training_rows = collect_training_rows(manifest, manifest_path)
    selected = select_best_runs(pruning_rows)
    curves = selected_curves(pruning_rows, selected)
    envelope = select_candidate_grid_envelope(pruning_rows)
    selected_names = {method: row["run_name"] for method, row in selected.items()}
    for row in pruning_rows:
        row["selected_for_method"] = (
            "yes" if selected_names.get(row["method"]) == row["run_name"] else "no"
        )

    pruning_fields = [
        "run_name", "method", "selected_for_method", "seed", "learning_rate", "l1_alpha", "rewa_k", "rewa_m",
        "rewa_weight_decay", "rewa_eps", "checkpoint_best_val_loss", "target_sparsity",
        "selected_count", "total_eligible", "selection_sparsity", "eligible_zeros",
        "eligible_nonzeros", "eligible_sparsity", "whole_model_zeros", "whole_model_nonzeros",
        "whole_model_sparsity", "attention_sparsity", "mlp_sparsity", "validation_loss",
        "perplexity",
    ]
    write_csv(output_csv, pruning_rows, pruning_fields)
    envelope_fields = [
        "method", "method_label", "target_sparsity", "perplexity", "validation_loss",
        "selected_run_name", "selected_configuration", "seed", "learning_rate", "l1_alpha",
        "rewa_k", "rewa_m", "rewa_weight_decay", "rewa_eps", "checkpoint_best_val_loss",
        "selected_count", "total_eligible", "selection_sparsity", "eligible_sparsity",
        "whole_model_sparsity", "attention_sparsity", "mlp_sparsity",
    ]
    write_csv(
        envelope_csv_path,
        flatten_candidate_grid_envelope(envelope),
        envelope_fields,
    )
    write_csv(
        layer_csv,
        layer_rows,
        ["run_name", "method", "seed", "target_sparsity", "name", "zeros", "nonzeros", "numel", "sparsity"],
    )
    training_fields = [
        "run_name", "method", "seed", "tokens_seen", "iter", "train_loss", "train_objective",
        "val_loss", "ppl", "lr", "elapsed_sec", "max_cuda_memory", "tokens_per_sec",
    ]
    write_csv(training_csv, training_rows, training_fields)
    make_plot(plot_path, curves, selected)
    make_envelope_plot(envelope_plot_path, envelope)
    write_results(
        results_path,
        manifest,
        pruning_rows,
        selected,
        curves,
        warnings,
        output_csv,
        plot_path,
        envelope,
        envelope_csv_path,
        envelope_plot_path,
    )
    print(f"wrote {output_csv}")
    print(f"wrote {layer_csv}")
    print(f"wrote {training_csv}")
    print(f"wrote {plot_path}")
    print(f"wrote {envelope_csv_path}")
    print(f"wrote {envelope_plot_path}")
    print(f"wrote {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
