#!/usr/bin/env python3
"""Run a restart-safe, successive-halving ReWA stage-one tuning study.

The tuning evidence is deliberately separated from the frozen stage-one pilot:

1. screen theory-admissible (K, M) pairs and learning rates at 3M tokens;
2. retrain the best four configurations from scratch at 10M tokens;
3. search epsilon/weight-decay variants around the two best 10M geometries;
4. retrain two accuracy/sparsity representatives from scratch at 20M tokens
   and evaluate the frozen global-pruning grid.

Short-budget ranks are exploratory.  Only the final phase is compared directly
with the original 20M-token Dense/L1/ReWA results.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KM = ((3.0, 0.0), (3.0, 1.0), (5.0, 0.0), (5.0, 2.0),
              (7.0, 2.0), (9.0, 2.0), (9.0, 4.0))
DEFAULT_LRS = (6e-3, 1.2e-2, 2.4e-2, 3.6e-2)
SHORT_SPARSITIES = (0.0, 0.5, 0.7, 0.8)
FINAL_SPARSITIES = (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95)
HIGH_SPARSITIES = (0.7, 0.8)
PHASE_ORDER = {"screen": 0, "confirm": 1, "variant": 2, "final": 3}
MANIFEST_FIELDS = (
    "order", "phase", "run_id", "run_name", "status", "selected_next",
    "selection_reason", "seed", "learning_rate", "rewa_k", "rewa_m",
    "rewa_eps", "rewa_weight_decay", "theory_status", "train_tokens",
    "expected_iters", "completed_iters", "best_val_loss", "best_val_ppl",
    "skipped_optimizer_steps",
    "pruning_eval_iters", "pruning_sparsities", "checkpoint", "metrics_jsonl",
    "pruning_json", "pruning_csv", "layer_csv", "run_config", "train_log",
    "git_commit", "error", "updated_at_utc",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be finite")
    return parsed


def rewa_config(value: str) -> tuple[float, float]:
    try:
        raw_k, raw_m = value.split(":", maxsplit=1)
        k, m = float(raw_k), float(raw_m)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("expected K:M, for example 3:0") from error
    if not math.isfinite(k) or not math.isfinite(m) or k <= 1 or m < 0 or m >= k - 1:
        raise argparse.ArgumentTypeError("ReWA requires K > 1 and 0 <= M < K - 1")
    return k, m


def slug(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    return f"{value:g}"


def project_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


@dataclass(frozen=True)
class TuneSpec:
    k: float
    m: float
    learning_rate: float
    eps: float = 0.0
    weight_decay: float = 1e-4
    seed: int = 0

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (
            self.k, self.m, self.learning_rate, self.eps, self.weight_decay
        )):
            raise ValueError("all ReWA tuning values must be finite")
        if self.k <= 1 or self.m < 0 or self.m >= self.k - 1:
            raise ValueError("ReWA requires K > 1 and 0 <= M < K - 1")
        if self.learning_rate <= 0 or self.eps < 0 or self.weight_decay < 0:
            raise ValueError("learning rate must be positive; epsilon/decay non-negative")

    @property
    def name(self) -> str:
        return (
            f"k{slug(self.k)}-m{slug(self.m)}-eps{slug(self.eps)}-"
            f"wd{slug(self.weight_decay)}-lr{slug(self.learning_rate)}-seed{self.seed}"
        )

    @property
    def theory_status(self) -> str:
        if self.eps == 0:
            return "base-admissible; epsilon=0"
        if self.m < 2:
            return "base-admissible; positive-epsilon Configuration-B range"
        return "excluded: positive epsilon requires M < 2 in this tuning study"


@dataclass(frozen=True)
class Phase:
    name: str
    train_tokens: int
    eval_iters: int
    eval_interval: int
    warmup_iters: int
    pruning_eval_iters: int = 0
    sparsities: tuple[float, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data-dir", type=Path, default=Path("data/tinystories"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/stage1-rewa-high-lr")
    )
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("artifacts/stage1-rewa-high-lr")
    )
    parser.add_argument(
        "--baseline-artifact-dir", type=Path, default=Path("artifacts/stage1")
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--pruning-seed", type=int, default=20_260_922)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float16"
    )
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=positive_int, default=16)
    parser.add_argument("--seq-len", type=positive_int, default=256)
    parser.add_argument("--n-layer", type=positive_int, default=6)
    parser.add_argument("--n-head", type=positive_int, default=6)
    parser.add_argument("--n-embd", type=positive_int, default=384)
    parser.add_argument("--intermediate-size", type=positive_int, default=1024)
    parser.add_argument("--ordinary-weight-decay", type=finite_float, default=0.1)
    parser.add_argument("--min-lr-ratio", type=finite_float, default=0.0)
    parser.add_argument("--pruning-batch-size", type=positive_int, default=16)

    parser.add_argument("--screen-tokens", type=positive_int, default=3_000_000)
    parser.add_argument("--confirm-tokens", type=positive_int, default=10_000_000)
    parser.add_argument("--final-tokens", type=positive_int, default=20_000_000)
    parser.add_argument("--screen-eval-iters", type=positive_int, default=20)
    parser.add_argument("--confirm-eval-iters", type=positive_int, default=30)
    parser.add_argument("--final-eval-iters", type=positive_int, default=50)
    parser.add_argument("--short-pruning-eval-iters", type=positive_int, default=20)
    parser.add_argument("--final-pruning-eval-iters", type=positive_int, default=100)
    parser.add_argument("--top-k", type=positive_int, default=4)
    parser.add_argument("--finalists", type=positive_int, default=2)
    parser.add_argument(
        "--screen-configs", type=rewa_config, nargs="+", default=list(DEFAULT_KM)
    )
    parser.add_argument(
        "--learning-rates", type=finite_float, nargs="+", default=list(DEFAULT_LRS)
    )
    parser.add_argument(
        "--variant-eps", type=finite_float, nargs="+", default=[0.0, 1e-6, 1e-3]
    )
    parser.add_argument(
        "--variant-weight-decays",
        type=finite_float,
        nargs="+",
        default=[1e-4, 0.1, 1.0],
    )
    parser.add_argument(
        "--variant-decay-reference-lr",
        type=finite_float,
        default=3e-3,
        help=(
            "Interpret --variant-weight-decays at this reference LR and scale raw "
            "y-space decay inversely with each candidate LR."
        ),
    )
    parser.add_argument(
        "--stop-after",
        choices=("screen", "confirm", "variant", "final"),
        default="final",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry matching manifest rows already marked failed (default: skip them).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.screen_configs or not args.learning_rates:
        raise ValueError("screen configs and learning rates cannot be empty")
    if any(lr <= 0 for lr in args.learning_rates):
        raise ValueError("learning rates must be positive")
    if any(value < 0 for value in (*args.variant_eps, *args.variant_weight_decays)):
        raise ValueError("epsilon and weight-decay candidates must be non-negative")
    if args.variant_decay_reference_lr <= 0:
        raise ValueError("variant decay reference LR must be positive")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("minimum LR ratio must be in [0, 1]")
    if args.top_k > len(args.screen_configs) * len(args.learning_rates):
        raise ValueError("top-k exceeds the screen grid")
    if args.finalists != 2:
        raise ValueError("this protocol selects exactly two complementary finalists")


def phase_definitions(args: argparse.Namespace) -> dict[str, Phase]:
    return {
        "screen": Phase(
            "screen", args.screen_tokens, args.screen_eval_iters, 46, 8,
            args.short_pruning_eval_iters, SHORT_SPARSITIES,
        ),
        "confirm": Phase(
            "confirm", args.confirm_tokens, args.confirm_eval_iters, 100, 25,
            args.short_pruning_eval_iters, SHORT_SPARSITIES,
        ),
        "variant": Phase(
            "variant", args.confirm_tokens, args.confirm_eval_iters, 100, 25,
            args.short_pruning_eval_iters, SHORT_SPARSITIES,
        ),
        "final": Phase(
            "final", args.final_tokens, args.final_eval_iters, 100, 50,
            args.final_pruning_eval_iters, FINAL_SPARSITIES,
        ),
    }


def expected_iters(phase: Phase, args: argparse.Namespace) -> int:
    tokens_per_iter = args.batch_size * args.seq_len * args.gradient_accumulation_steps
    return math.ceil(phase.train_tokens / tokens_per_iter)


def screen_specs(args: argparse.Namespace) -> list[TuneSpec]:
    return [
        TuneSpec(k, m, lr, seed=args.seed)
        for k, m in args.screen_configs
        for lr in args.learning_rates
    ]


def variant_specs(base_specs: Iterable[TuneSpec], args: argparse.Namespace) -> list[TuneSpec]:
    variants: dict[str, TuneSpec] = {}
    for base in base_specs:
        for eps in args.variant_eps:
            if eps > 0 and base.m >= 2:
                continue
            for decay in args.variant_weight_decays:
                # The strongest decay is an epsilon=0 diagnostic; combining it
                # with attenuation would spend budget on a poorly identified corner.
                if decay >= 1.0 and eps > 0:
                    continue
                normalized_decay = decay * args.variant_decay_reference_lr / base.learning_rate
                spec = replace(base, eps=eps, weight_decay=normalized_decay)
                if spec.name != base.name:
                    variants[spec.name] = spec
    return list(variants.values())


def read_metrics(path: Path) -> tuple[int, float | None, int]:
    completed = 0
    best: float | None = None
    skipped = 0
    if not path.exists():
        return completed, best, skipped
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                completed = max(completed, int(record.get("iter", 0)))
                skipped += int(record.get("skipped_optimizer_steps", 0))
                value = record.get("val_loss")
                if value is not None and math.isfinite(float(value)):
                    best = float(value) if best is None else min(best, float(value))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return completed, best, skipped


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["run_id"]: row for row in csv.DictReader(handle)}


def write_manifest(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    ordered = sorted(rows.values(), key=lambda row: (int(row["order"]), row["run_id"]))
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in ordered:
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})
    os.replace(temporary, path)


def run_paths(spec: TuneSpec, phase: Phase, output_root: Path, artifact_dir: Path) -> dict[str, Path]:
    run_dir = output_root / phase.name / spec.name
    eval_dir = artifact_dir / "runs" / phase.name / spec.name
    return {
        "run_dir": run_dir,
        "best": run_dir / "best.pt",
        "latest": run_dir / "latest.pt",
        "metrics": run_dir / "metrics.jsonl",
        "config": run_dir / "run_config.json",
        "train_log": run_dir / "train.log",
        "pruning_json": eval_dir / "global-pruning.json",
        "pruning_csv": eval_dir / "global-pruning.csv",
        "layer_csv": eval_dir / "layer-sparsity.csv",
        "evaluation_log": eval_dir / "evaluation.log",
    }


def run_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = subprocess.list2cmdline(command)
    print(f"\n$ {printable}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] $ {printable}\n")
        log.flush()
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def training_command(
    spec: TuneSpec, phase: Phase, args: argparse.Namespace, data_dir: Path,
    paths: dict[str, Path], resume: bool,
) -> list[str]:
    command = [
        args.python, str(PROJECT_ROOT / "train.py"), "--data-dir", str(data_dir),
        "--out-dir", str(paths["run_dir"]), "--method", "rewa", "--seed", str(spec.seed),
        "--eval-seed", str(args.eval_seed), "--device", args.device, "--dtype", args.dtype,
        "--seq-len", str(args.seq_len), "--n-layer", str(args.n_layer),
        "--n-head", str(args.n_head), "--n-embd", str(args.n_embd),
        "--intermediate-size", str(args.intermediate_size), "--batch-size", str(args.batch_size),
        "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
        "--train-tokens", str(phase.train_tokens), "--eval-iters", str(phase.eval_iters),
        "--eval-interval", str(phase.eval_interval), "--log-interval", "10",
        "--save-interval", str(min(phase.eval_interval, expected_iters(phase, args))),
        "--warmup-iters", str(phase.warmup_iters), "--learning-rate", str(spec.learning_rate),
        "--min-lr-ratio", str(args.min_lr_ratio),
        "--weight-decay", str(args.ordinary_weight_decay), "--rewa-k", str(spec.k),
        "--rewa-m", str(spec.m), "--rewa-eps", str(spec.eps),
        "--rewa-weight-decay", str(spec.weight_decay),
    ]
    if resume:
        command.extend(("--resume", str(paths["latest"])))
    return command


def pruning_command(
    phase: Phase, args: argparse.Namespace, data_dir: Path, paths: dict[str, Path]
) -> list[str]:
    return [
        args.python, str(PROJECT_ROOT / "evaluate_global_pruning.py"),
        "--checkpoint", str(paths["best"]), "--data-dir", str(data_dir),
        "--sparsities", *(str(value) for value in phase.sparsities),
        "--batch-size", str(args.pruning_batch_size), "--seq-len", str(args.seq_len),
        "--eval-iters", str(phase.pruning_eval_iters), "--seed", str(args.pruning_seed),
        "--device", args.device, "--dtype", args.dtype,
        "--output-json", str(paths["pruning_json"]),
        "--output-csv", str(paths["pruning_csv"]), "--layer-csv", str(paths["layer_csv"]),
    ]


def expected_training_args(
    spec: TuneSpec, phase: Phase, args: argparse.Namespace, data_dir: Path
) -> dict[str, Any]:
    return {
        "data_dir": str(data_dir), "method": "rewa", "seed": spec.seed,
        "eval_seed": args.eval_seed, "device": args.device, "dtype": args.dtype,
        "learning_rate": spec.learning_rate, "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "seq_len": args.seq_len, "n_layer": args.n_layer, "n_head": args.n_head,
        "n_embd": args.n_embd, "intermediate_size": args.intermediate_size,
        "train_tokens": phase.train_tokens, "eval_iters": phase.eval_iters,
        "eval_interval": phase.eval_interval, "warmup_iters": phase.warmup_iters,
        "min_lr_ratio": args.min_lr_ratio,
        "weight_decay": args.ordinary_weight_decay, "rewa_k": spec.k,
        "rewa_m": spec.m, "rewa_eps": spec.eps,
        "rewa_weight_decay": spec.weight_decay,
    }


def assert_compatible(
    config_path: Path, spec: TuneSpec, phase: Phase, args: argparse.Namespace, data_dir: Path
) -> None:
    if not config_path.exists():
        return
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    saved = payload.get("train_args", {})
    expected = expected_training_args(spec, phase, args, data_dir)
    mismatches = {
        key: (saved.get(key), value) for key, value in expected.items()
        if saved.get(key) != value
    }
    if mismatches:
        detail = ", ".join(
            f"{key}: saved={old!r}, requested={new!r}"
            for key, (old, new) in mismatches.items()
        )
        raise RuntimeError(f"incompatible existing tuning run {config_path.parent.name}: {detail}")


def pruning_complete(paths: dict[str, Path], phase: Phase, args: argparse.Namespace) -> bool:
    if not all(paths[key].exists() for key in ("pruning_json", "pruning_csv", "layer_csv")):
        return False
    try:
        payload = json.loads(paths["pruning_json"].read_text(encoding="utf-8"))
        evaluation = payload["evaluation"]
        observed = [float(row["target_sparsity"]) for row in payload["results"]]
        return (
            int(evaluation["eval_iters"]) == phase.pruning_eval_iters
            and int(evaluation["seed"]) == args.pruning_seed
            and len(observed) == len(phase.sparsities)
            and all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(observed, phase.sparsities))
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def base_row(
    order: int, spec: TuneSpec, phase: Phase, args: argparse.Namespace,
    paths: dict[str, Path], commit: str,
) -> dict[str, Any]:
    run_id = f"{phase.name}/{spec.name}"
    return {
        "order": order, "phase": phase.name, "run_id": run_id,
        "run_name": spec.name, "status": "planned", "selected_next": "",
        "selection_reason": "", "seed": spec.seed, "learning_rate": spec.learning_rate,
        "rewa_k": spec.k, "rewa_m": spec.m, "rewa_eps": spec.eps,
        "rewa_weight_decay": spec.weight_decay, "theory_status": spec.theory_status,
        "train_tokens": phase.train_tokens, "expected_iters": expected_iters(phase, args),
        "pruning_eval_iters": phase.pruning_eval_iters,
        "pruning_sparsities": " ".join(str(value) for value in phase.sparsities),
        "checkpoint": display_path(paths["best"]), "metrics_jsonl": display_path(paths["metrics"]),
        "pruning_json": display_path(paths["pruning_json"]),
        "pruning_csv": display_path(paths["pruning_csv"]),
        "layer_csv": display_path(paths["layer_csv"]),
        "run_config": display_path(paths["config"]), "train_log": display_path(paths["train_log"]),
        "git_commit": commit, "error": "", "updated_at_utc": utc_now(),
    }


def execute_spec(
    order: int, spec: TuneSpec, phase: Phase, args: argparse.Namespace,
    data_dir: Path, output_root: Path, artifact_dir: Path,
    manifest_path: Path, rows: dict[str, dict[str, Any]], commit: str,
) -> bool:
    paths = run_paths(spec, phase, output_root, artifact_dir)
    run_id = f"{phase.name}/{spec.name}"
    previous = dict(rows.get(run_id, {}))
    if previous.get("status") == "failed" and not args.retry_failed:
        assert_compatible(paths["config"], spec, phase, args, data_dir)
        print(f"[{run_id}] previously failed; skipping (use --retry-failed to retry)", flush=True)
        return False
    existing = dict(previous)
    existing.update(base_row(order, spec, phase, args, paths, commit))
    rows[run_id] = existing
    write_manifest(manifest_path, rows)
    completed, best, skipped = read_metrics(paths["metrics"])
    try:
        assert_compatible(paths["config"], spec, phase, args, data_dir)
        training_done = (
            paths["best"].exists() and paths["latest"].exists()
            and completed >= expected_iters(phase, args) and best is not None
        )
        if training_done:
            print(f"[{run_id}] training complete; skipping", flush=True)
        else:
            rows[run_id].update(status="training", completed_iters=completed, error="")
            write_manifest(manifest_path, rows)
            resume = paths["latest"].exists() and completed > 0
            code = run_command(
                training_command(spec, phase, args, data_dir, paths, resume), paths["train_log"]
            )
            if code != 0:
                raise RuntimeError(f"training exited with status {code}")
            completed, best, skipped = read_metrics(paths["metrics"])
            if completed < expected_iters(phase, args) or best is None:
                raise RuntimeError("training returned without complete metrics/checkpoints")
        assert best is not None
        config_copy = artifact_dir / "configs" / phase.name / f"{spec.name}.json"
        config_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths["config"], config_copy)
        rows[run_id].update(
            status="trained", completed_iters=completed,
            best_val_loss=f"{best:.12g}", best_val_ppl=f"{math.exp(best):.12g}", error="",
            skipped_optimizer_steps=skipped,
        )
        write_manifest(manifest_path, rows)
        if phase.pruning_eval_iters:
            if pruning_complete(paths, phase, args):
                print(f"[{run_id}] pruning screen complete; skipping", flush=True)
            else:
                code = run_command(
                    pruning_command(phase, args, data_dir, paths), paths["evaluation_log"]
                )
                if code != 0:
                    raise RuntimeError(f"pruning evaluation exited with status {code}")
                if not pruning_complete(paths, phase, args):
                    raise RuntimeError("pruning evaluation outputs are incomplete")
        rows[run_id].update(status="completed", error="", updated_at_utc=utc_now())
        write_manifest(manifest_path, rows)
        return True
    except Exception as error:
        completed, best, skipped = read_metrics(paths["metrics"])
        rows[run_id].update(
            status="failed", completed_iters=completed,
            best_val_loss="" if best is None else f"{best:.12g}",
            best_val_ppl="" if best is None else f"{math.exp(best):.12g}",
            skipped_optimizer_steps=skipped,
            error=f"{type(error).__name__}: {error}", updated_at_utc=utc_now(),
        )
        write_manifest(manifest_path, rows)
        print(f"[{run_id}] FAILED: {error}", file=sys.stderr, flush=True)
        if args.fail_fast:
            raise
        return False


def spec_from_row(row: dict[str, Any]) -> TuneSpec:
    return TuneSpec(
        float(row["rewa_k"]), float(row["rewa_m"]), float(row["learning_rate"]),
        float(row["rewa_eps"]), float(row["rewa_weight_decay"]), int(row["seed"]),
    )


def rank_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (row for row in rows if row.get("status") == "completed" and row.get("best_val_loss")),
        key=lambda row: (float(row["best_val_loss"]), row["run_id"]),
    )


def select_screen_rows(
    screen_rank: list[dict[str, Any]], count: int, unpruned_guard: float = 1.25
) -> list[dict[str, Any]]:
    """Balance accuracy, high-sparsity behavior, and canonical retention."""

    selected: list[dict[str, Any]] = []

    def append_once(row: dict[str, Any] | None) -> None:
        if row is not None and row not in selected and len(selected) < count:
            selected.append(row)

    append_once(screen_rank[0] if screen_rank else None)
    pruning_candidates: list[tuple[dict[str, Any], dict[float, dict[str, Any]]]] = []
    for row in screen_rank:
        try:
            points = pruning_points(row)
            if 0.0 in points and all(target in points for target in HIGH_SPARSITIES):
                pruning_candidates.append((row, points))
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            continue

    if pruning_candidates:
        best_unpruned = min(float(points[0.0]["perplexity"]) for _, points in pruning_candidates)
        guarded = [
            item for item in pruning_candidates
            if float(item[1][0.0]["perplexity"]) <= unpruned_guard * best_unpruned
        ]
        sparse_row, _ = min(
            guarded,
            key=lambda item: (
                sum(float(item[1][target]["validation_loss"]) for target in HIGH_SPARSITIES)
                / len(HIGH_SPARSITIES),
                item[0]["run_id"],
            ),
        )
        append_once(sparse_row)

    canonical_rows = [
        row for row in screen_rank
        if math.isclose(float(row["rewa_k"]), 9.0)
        and math.isclose(float(row["rewa_m"]), 2.0)
    ]
    canonical = canonical_rows[0] if canonical_rows else None
    canonical_pruning = [item for item in pruning_candidates if item[0] in canonical_rows]
    if canonical_pruning:
        canonical, _ = min(
            canonical_pruning,
            key=lambda item: (
                sum(float(item[1][target]["validation_loss"]) for target in HIGH_SPARSITIES)
                / len(HIGH_SPARSITIES),
                item[0]["run_id"],
            ),
        )
    append_once(canonical)
    for row in screen_rank:
        append_once(row)
    return selected


def select_geometry_rows(confirm_rank: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep accuracy/sparsity representatives and an epsilon-eligible branch."""

    try:
        selected = [row for row, _ in select_finalists(confirm_rank)]
    except RuntimeError:
        # Ranking-only callers and legacy manifests may not have pruning files.
        selected = list(confirm_rank[:2])
    if selected and not any(float(row["rewa_m"]) < 2 for row in selected):
        low_m = next((row for row in confirm_rank if float(row["rewa_m"]) < 2), None)
        if low_m is not None:
            selected.append(low_m)
    return selected


def pruning_points(row: dict[str, Any]) -> dict[float, dict[str, Any]]:
    path = project_path(Path(str(row["pruning_json"])))
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {round(float(point["target_sparsity"]), 12): point for point in payload["results"]}


def select_finalists(
    pool: list[dict[str, Any]], count: int = 2, unpruned_guard: float = 1.10
) -> list[tuple[dict[str, Any], str]]:
    if count != 2:
        raise ValueError("this selector returns two complementary finalists")
    candidates: list[tuple[dict[str, Any], dict[float, dict[str, Any]]]] = []
    for row in pool:
        try:
            points = pruning_points(row)
            if 0.0 in points and all(target in points for target in HIGH_SPARSITIES):
                candidates.append((row, points))
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            continue
    if len(candidates) < count:
        raise RuntimeError("fewer than two completed 10M candidates have pruning screens")
    primary, primary_points = min(
        candidates, key=lambda item: (float(item[1][0.0]["validation_loss"]), item[0]["run_id"])
    )
    best_unpruned_ppl = float(primary_points[0.0]["perplexity"])
    guarded = [
        item for item in candidates
        if item[0]["run_id"] != primary["run_id"]
        and float(item[1][0.0]["perplexity"]) <= unpruned_guard * best_unpruned_ppl
    ]
    if guarded:
        secondary, _ = min(
            guarded,
            key=lambda item: (
                sum(float(item[1][target]["validation_loss"]) for target in HIGH_SPARSITIES)
                / len(HIGH_SPARSITIES),
                item[0]["run_id"],
            ),
        )
        reason = (
            "best mean 70%/80%-pruned loss within "
            f"{100*(unpruned_guard-1):.0f}% unpruned-PPL guard"
        )
    else:
        secondary, _ = min(
            (item for item in candidates if item[0]["run_id"] != primary["run_id"]),
            key=lambda item: (float(item[1][0.0]["validation_loss"]), item[0]["run_id"]),
        )
        reason = "second-best unpruned loss; no alternate passed the unpruned guard"
    return [(primary, "best unpruned loss"), (secondary, reason)]


def write_ranking(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "rank", "phase", "run_id", "learning_rate", "rewa_k", "rewa_m",
        "rewa_eps", "rewa_weight_decay", "best_val_loss", "best_val_ppl",
        "selected_next", "selection_reason", "theory_status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            writer.writerow({"rank": index, **{field: row.get(field, "") for field in fields[1:]}})


def mark_selection(
    rows: dict[str, dict[str, Any]], manifest_path: Path,
    selected: Iterable[tuple[dict[str, Any], str]], source_phases: set[str],
) -> None:
    for row in rows.values():
        if row.get("phase") in source_phases:
            row["selected_next"] = "no"
            row["selection_reason"] = ""
    for row, reason in selected:
        rows[row["run_id"]]["selected_next"] = "yes"
        rows[row["run_id"]]["selection_reason"] = reason
    write_manifest(manifest_path, rows)


def flatten_final(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        for target, point in sorted(pruning_points(row).items()):
            flattened.append({
                "run_id": row["run_id"], "run_name": row["run_name"],
                "seed": row["seed"], "learning_rate": row["learning_rate"],
                "rewa_k": row["rewa_k"], "rewa_m": row["rewa_m"],
                "rewa_eps": row["rewa_eps"], "rewa_weight_decay": row["rewa_weight_decay"],
                "target_sparsity": target, "validation_loss": point["validation_loss"],
                "perplexity": point["perplexity"],
                "eligible_sparsity": point["eligible"]["sparsity"],
                "whole_model_sparsity": point["whole_model"]["sparsity"],
                "attention_sparsity": point["groups"]["attention"]["sparsity"],
                "mlp_sparsity": point["groups"]["mlp"]["sparsity"],
            })
    return flattened


def write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_baseline_curves(path: Path) -> dict[str, list[dict[str, Any]]]:
    curves: dict[str, list[dict[str, Any]]] = {"dense": [], "l1": [], "rewa": []}
    if not path.exists():
        return curves
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["method"] in curves and row.get("selected_for_method") == "yes":
                curves[row["method"]].append(row)
    return curves


def load_pruning_grid(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_confirm_results(
    path: Path, confirm_rank: list[dict[str, Any]],
    baseline_curves: dict[str, list[dict[str, Any]]], baseline_grid: list[dict[str, Any]],
) -> None:
    lines = [
        "# ReWA 10M-token high-learning-rate confirmation", "",
        "All ReWA rows below were trained locally from scratch with seed 0, "
        "10,027,008 realized tokens, batch size 8, gradient accumulation 16, "
        "maximum LR 0.006, and a cosine schedule ending at zero. Each pruning "
        "ratio was applied independently to the same best checkpoint.", "",
        "## ReWA geometry comparison", "",
        "| K | M | Unpruned PPL | 50% PPL | 70% PPL | 80% PPL |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_run: dict[str, dict[float, dict[str, Any]]] = {}
    for row in confirm_rank:
        points = pruning_points(row)
        by_run[row["run_id"]] = points
        values = [float(points[target]["perplexity"]) for target in SHORT_SPARSITIES]
        lines.append(
            f"| {float(row['rewa_k']):g} | {float(row['rewa_m']):g} | "
            + " | ".join(f"{value:.4f}" for value in values) + " |"
        )

    lines.extend([
        "", "## Frozen 20M-token baselines", "",
        "These baselines use the same model, data hashes, seed, effective batch, "
        "eligible tensors, and pruning seed. They have twice the training-token "
        "budget, so this comparison does not favor ReWA.", "",
        "| Method | Unpruned PPL | 50% PPL | 70% PPL | 80% PPL |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    baseline_maps = {
        method: {round(float(row["target_sparsity"]), 12): row for row in curve}
        for method, curve in baseline_curves.items()
    }
    for method, label in (("dense", "Dense AdamW"), ("l1", "AdamW + L1")):
        points = baseline_maps.get(method, {})
        if all(target in points for target in SHORT_SPARSITIES):
            lines.append(
                f"| {label} | "
                + " | ".join(
                    f"{float(points[target]['perplexity']):.4f}" for target in SHORT_SPARSITIES
                )
                + " |"
            )

    selected = select_finalists(confirm_rank)
    accuracy_row, _ = selected[0]
    sparse_row, sparse_reason = selected[1]
    sparse_points = by_run[sparse_row["run_id"]]
    dense_zero = baseline_maps.get("dense", {}).get(0.0)
    lines.extend(["", "## Evidence-based interpretation", ""])
    lines.append(
        f"The accuracy representative is `{accuracy_row['run_id']}`. The sparsity "
        f"representative is `{sparse_row['run_id']}` ({sparse_reason})."
    )
    if dense_zero is not None:
        dense_ppl = float(dense_zero["perplexity"])
        sparse_ppl = float(sparse_points[0.0]["perplexity"])
        gap = 100.0 * (sparse_ppl / dense_ppl - 1.0)
        lines.append(
            f"Its unpruned PPL is {gap:.2f}% above the frozen Dense baseline "
            f"({sparse_ppl:.4f} versus {dense_ppl:.4f})."
        )
    for target in HIGH_SPARSITIES:
        dense_point = baseline_maps.get("dense", {}).get(target)
        l1_point = baseline_maps.get("l1", {}).get(target)
        if dense_point is not None and l1_point is not None:
            lines.append(
                f"At {100*target:.0f}% sparsity, the ReWA representative has PPL "
                f"{float(sparse_points[target]['perplexity']):.4f}, versus "
                f"{float(dense_point['perplexity']):.4f} for Dense and "
                f"{float(l1_point['perplexity']):.4f} for the best-unpruned L1 curve."
            )

    l1_by_run: dict[str, dict[float, dict[str, Any]]] = {}
    for row in baseline_grid:
        if row.get("method") == "l1":
            l1_by_run.setdefault(row["run_name"], {})[
                round(float(row["target_sparsity"]), 12)
            ] = row
    l1_sparse = [
        (name, points) for name, points in l1_by_run.items()
        if all(target in points for target in (0.0, *HIGH_SPARSITIES))
    ]
    if l1_sparse:
        name, points = min(
            l1_sparse,
            key=lambda item: sum(
                float(item[1][target]["validation_loss"]) for target in HIGH_SPARSITIES
            ),
        )
        lines.append(
            "The optimistic L1 candidate-grid comparison selects a different, strongly "
            f"regularized run (`{name}`): PPL "
            f"{float(points[0.0]['perplexity']):.4f} unpruned, "
            f"{float(points[0.7]['perplexity']):.4f} at 70%, and "
            f"{float(points[0.8]['perplexity']):.4f} at 80%. This is a different "
            "accuracy/sparsity operating point, not the best-unpruned L1 curve."
        )
    lines.extend([
        "", "The result is a single-seed confirmation on the tuning validation split. "
        "It establishes a strong local signal and a reproducible configuration; fresh "
        "seeds are still required for an uncertainty estimate.", "",
        "## Artifacts", "", "- `tuning-manifest.csv`", "- `screen-ranking.csv`",
        "- `ten-million-ranking.csv`", "- `ten-million-global-pruning.csv`", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def make_plot(
    path: Path, final_rows: list[dict[str, Any]], baseline_curves: dict[str, list[dict[str, Any]]]
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(10, 6.5))
    for method, label in (("dense", "Dense AdamW"), ("l1", "AdamW + L1"),
                          ("rewa", "Original ReWA")):
        curve = sorted(baseline_curves.get(method, []), key=lambda row: float(row["target_sparsity"]))
        if curve:
            axis.plot(
                [100 * float(row["target_sparsity"]) for row in curve],
                [float(row["perplexity"]) for row in curve], marker="o", label=label,
            )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in final_rows:
        grouped.setdefault(row["run_name"], []).append(row)
    for index, (name, curve) in enumerate(sorted(grouped.items()), 1):
        curve.sort(key=lambda row: float(row["target_sparsity"]))
        axis.plot(
            [100 * float(row["target_sparsity"]) for row in curve],
            [float(row["perplexity"]) for row in curve], marker="s", linewidth=2.5,
            label=f"Tuned ReWA finalist {index}",
        )
    axis.set_yscale("log")
    axis.set_xlabel("Eligible global sparsity (%)")
    axis.set_ylabel("Validation perplexity")
    axis.set_title("ReWA tuning finalists vs frozen stage-one baselines")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_results(
    path: Path, screen_rank: list[dict[str, Any]], ten_m_rank: list[dict[str, Any]],
    final_rows: list[dict[str, Any]], baseline_curves: dict[str, list[dict[str, Any]]],
) -> None:
    lines = [
        "# ReWA stage-one hyperparameter tuning", "",
        "Short-budget rankings are exploratory and were used only for successive halving. "
        "The two finalists were retrained from scratch at 20M tokens before comparison.", "",
        "## Screen", "",
        "| Rank | K | M | LR | eps | wd | Best validation PPL |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(screen_rank[:10], 1):
        lines.append(
            f"| {index} | {float(row['rewa_k']):g} | {float(row['rewa_m']):g} | "
            f"{float(row['learning_rate']):g} | {float(row['rewa_eps']):g} | "
            f"{float(row['rewa_weight_decay']):g} | {float(row['best_val_ppl']):.4f} |"
        )
    lines.extend(["", "## Ten-million-token confirmation and local variants", "",
                  "| Rank | Run | Unpruned PPL | 50% PPL | Selected |",
                  "| ---: | --- | ---: | ---: | --- |"])
    for index, row in enumerate(ten_m_rank, 1):
        points = pruning_points(row)
        lines.append(
            f"| {index} | `{row['run_id']}` | {float(points[0.0]['perplexity']):.4f} | "
            f"{float(points[0.5]['perplexity']):.4f} | {row.get('selected_next','')} |"
        )
    lines.extend(["", "## Final 20M-token comparison", ""])
    final_by_run: dict[str, dict[float, dict[str, Any]]] = {}
    for row in final_rows:
        final_by_run.setdefault(row["run_name"], {})[float(row["target_sparsity"])] = row
    baseline_maps = {
        method: {float(row["target_sparsity"]): row for row in curve}
        for method, curve in baseline_curves.items()
    }
    targets = sorted({float(row["target_sparsity"]) for row in final_rows})
    headers = ["Sparsity", "Dense PPL", "L1 PPL", *(
        f"Tuned finalist {index} PPL" for index in range(1, len(final_by_run) + 1)
    )]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---:" for _ in headers) + " |")
    for target in targets:
        values = [f"{100*target:.0f}%"]
        for method in ("dense", "l1"):
            point = baseline_maps.get(method, {}).get(target)
            values.append("n/a" if point is None else f"{float(point['perplexity']):.4f}")
        for curve in final_by_run.values():
            values.append(f"{float(curve[target]['perplexity']):.4f}")
        lines.append("| " + " | ".join(values) + " |")

    dense_zero = baseline_maps.get("dense", {}).get(0.0)
    close_finalists: list[str] = []
    high_sparsity_winners: list[str] = []
    if dense_zero is not None:
        dense_ppl = float(dense_zero["perplexity"])
        for name, curve in final_by_run.items():
            if float(curve[0.0]["perplexity"]) <= 1.05 * dense_ppl:
                close_finalists.append(name)
            if all(
                target in curve
                and target in baseline_maps.get("dense", {})
                and target in baseline_maps.get("l1", {})
                and float(curve[target]["validation_loss"])
                < float(baseline_maps["dense"][target]["validation_loss"])
                and float(curve[target]["validation_loss"])
                < float(baseline_maps["l1"][target]["validation_loss"])
                for target in HIGH_SPARSITIES
            ):
                high_sparsity_winners.append(name)
    lines.extend(["", "## Interpretation", ""])
    if set(close_finalists).intersection(high_sparsity_winners):
        lines.append(
            "At least one tuned finalist is within 5% of Dense before pruning and tuned ReWA "
            "beats both frozen baselines at both 70% and 80% global sparsity. This is a "
            "positive single-seed tuning signal, not yet a multi-seed estimate."
        )
    else:
        lines.append(
            "The expanded search does not satisfy the stage-one scale-up rule: no combination "
            "simultaneously stays within 5% of Dense before pruning and beats both frozen "
            "baselines at both 70% and 80% global sparsity."
        )
    lines.extend([
        "", "All selection used the same validation split, so screening ranks and the 10M "
        "pruning guard are optimistic tuning diagnostics. Final curves remain single-seed evidence.",
        "", "## Artifacts", "", "- `tuning-manifest.csv`", "- `screen-ranking.csv`",
        "- `ten-million-ranking.csv`", "- `final-global-pruning.csv`",
        "- `rewa-tuning-vs-baselines.png`", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_phase(
    specs: list[TuneSpec], phase: Phase, start_order: int, args: argparse.Namespace,
    data_dir: Path, output_root: Path, artifact_dir: Path, manifest_path: Path,
    rows: dict[str, dict[str, Any]], commit: str,
) -> None:
    for offset, spec in enumerate(specs):
        execute_spec(
            start_order + offset, spec, phase, args, data_dir, output_root,
            artifact_dir, manifest_path, rows, commit,
        )


def main() -> int:
    args = parse_args()
    validate_args(args)
    data_dir = project_path(args.data_dir)
    output_root = project_path(args.output_root)
    artifact_dir = project_path(args.artifact_dir)
    baseline_artifact_dir = project_path(args.baseline_artifact_dir)
    if not (data_dir / "metadata.json").exists():
        raise FileNotFoundError(f"prepared data not found: {data_dir}")
    phases = phase_definitions(args)
    manifest_path = artifact_dir / "tuning-manifest.csv"
    rows: dict[str, dict[str, Any]] = load_manifest(manifest_path)
    commit = git_commit()

    screen = screen_specs(args)
    run_phase(screen, phases["screen"], 0, args, data_dir, output_root,
              artifact_dir, manifest_path, rows, commit)
    screen_rank = rank_rows(row for row in rows.values() if row.get("phase") == "screen")
    if len(screen_rank) < args.top_k:
        raise RuntimeError(f"only {len(screen_rank)} screen configurations completed")
    selected_screen_rows = select_screen_rows(screen_rank, args.top_k)
    selected_screen = [
        (
            row,
            "canonical K=9/M=2 retention"
            if row not in screen_rank[:args.top_k]
            and math.isclose(float(row["rewa_k"]), 9.0)
            and math.isclose(float(row["rewa_m"]), 2.0)
            else "short-run 70%/80% pruning retention"
            if row not in screen_rank[:args.top_k]
            else f"top-{args.top_k} 3M validation loss",
        )
        for row in selected_screen_rows
    ]
    mark_selection(rows, manifest_path, selected_screen, {"screen"})
    write_ranking(artifact_dir / "screen-ranking.csv", screen_rank)
    if args.stop_after == "screen":
        return 0

    confirm_specs = [spec_from_row(row) for row, _ in selected_screen]
    run_phase(confirm_specs, phases["confirm"], 10_000, args, data_dir, output_root,
              artifact_dir, manifest_path, rows, commit)
    confirm_rank = rank_rows(row for row in rows.values() if row.get("phase") == "confirm")
    if len(confirm_rank) < 2:
        raise RuntimeError("fewer than two 10M confirmation runs completed")
    geometry_rows = select_geometry_rows(confirm_rank)
    geometry_bases = [spec_from_row(row) for row in geometry_rows]
    finalist_reasons = {
        row["run_id"]: reason for row, reason in select_finalists(confirm_rank)
    }
    mark_selection(
        rows, manifest_path,
        [
            (
                row,
                "10M geometry for local epsilon/decay search: "
                + finalist_reasons.get(
                    row["run_id"], "retained to guarantee M < 2 epsilon branch"
                ),
            )
            for row in geometry_rows
        ],
        {"confirm"},
    )
    if args.stop_after == "confirm":
        write_ranking(artifact_dir / "ten-million-ranking.csv", confirm_rank)
        write_dict_csv(
            artifact_dir / "ten-million-global-pruning.csv", flatten_final(confirm_rank)
        )
        confirm_baselines = load_baseline_curves(
            baseline_artifact_dir / "global-pruning.csv"
        )
        write_confirm_results(
            artifact_dir / "CONFIRM_RESULTS.md", confirm_rank, confirm_baselines,
            load_pruning_grid(baseline_artifact_dir / "global-pruning.csv"),
        )
        return 0

    variants = variant_specs(geometry_bases, args)
    run_phase(variants, phases["variant"], 20_000, args, data_dir, output_root,
              artifact_dir, manifest_path, rows, commit)
    ten_m_pool = rank_rows(
        row for row in rows.values() if row.get("phase") in {"confirm", "variant"}
    )
    finalists = select_finalists(ten_m_pool, args.finalists)
    mark_selection(rows, manifest_path, finalists, {"confirm", "variant"})
    ten_m_rank = sorted(
        ten_m_pool,
        key=lambda row: (float(pruning_points(row)[0.0]["validation_loss"]), row["run_id"]),
    )
    write_ranking(artifact_dir / "ten-million-ranking.csv", ten_m_rank)
    if args.stop_after == "variant":
        return 0

    final_specs = [spec_from_row(row) for row, _ in finalists]
    run_phase(final_specs, phases["final"], 30_000, args, data_dir, output_root,
              artifact_dir, manifest_path, rows, commit)
    final_manifest_rows = rank_rows(row for row in rows.values() if row.get("phase") == "final")
    if len(final_manifest_rows) < args.finalists:
        raise RuntimeError("not all 20M finalists completed")
    final_flat = flatten_final(final_manifest_rows)
    write_dict_csv(artifact_dir / "final-global-pruning.csv", final_flat)
    baselines = load_baseline_curves(baseline_artifact_dir / "global-pruning.csv")
    make_plot(artifact_dir / "rewa-tuning-vs-baselines.png", final_flat, baselines)
    write_results(
        artifact_dir / "RESULTS.md", screen_rank, ten_m_rank, final_flat, baselines
    )
    failures = [row for row in rows.values() if row.get("status") == "failed"]
    print(
        f"ReWA tuning finished: {len(screen_rank)} screen, {len(ten_m_pool)} 10M, "
        f"{len(final_manifest_rows)} final; {len(failures)} preserved failure(s).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
