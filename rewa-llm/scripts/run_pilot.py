#!/usr/bin/env python3
"""Run the complete single-seed stage-one pilot and its pruning evaluations.

The runner executes the two dense learning-rate candidates first, selects the
one with the lowest training-time ``best_val_loss``, and uses that learning
rate for the L1 and ReWA sweeps.  Completed training and pruning work is
detected from artifacts, so rerunning the command is safe.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
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
DEFAULT_SPARSITIES = (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95)
METHOD_ORDER = {"dense": 0, "l1": 1, "rewa": 2}
MANIFEST_FIELDS = (
    "order",
    "run_name",
    "method",
    "status",
    "seed",
    "learning_rate",
    "dense_lr_selected",
    "l1_alpha",
    "rewa_k",
    "rewa_m",
    "rewa_weight_decay",
    "rewa_eps",
    "best_val_loss",
    "best_val_ppl",
    "train_tokens",
    "expected_iters",
    "completed_iters",
    "batch_size",
    "gradient_accumulation_steps",
    "seq_len",
    "eval_iters",
    "pruning_eval_iters",
    "git_commit",
    "checkpoint",
    "metrics_jsonl",
    "pruning_csv",
    "layer_csv",
    "pruning_json",
    "run_config",
    "train_log",
    "superseded_from_status",
    "status_detail",
    "error",
    "updated_at_utc",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be finite and in [0, 1]")
    return parsed


def finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be finite")
    return parsed


def rewa_config(value: str) -> tuple[float, float]:
    """Parse a theory-conforming ``K:M`` ReWA configuration."""

    try:
        raw_k, raw_m = value.split(":", maxsplit=1)
        k, m = float(raw_k), float(raw_m)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            "ReWA configurations must use K:M syntax, for example 3:0 or 9:2"
        ) from error
    if not math.isfinite(k) or not math.isfinite(m):
        raise argparse.ArgumentTypeError("ReWA K and M must be finite")
    if k <= 1.0 or m < 0.0 or m >= k - 1.0:
        raise argparse.ArgumentTypeError(
            "ReWA requires K > 1 and 0 <= M < K - 1"
        )
    return k, m


def project_path(value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def display_path(path: Path) -> str:
    """Prefer portable project-relative paths in the manifest."""

    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def slug_number(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        text = f"{value:.0e}"
        return text.replace("e-0", "e-").replace("e+0", "e+")
    return f"{value:g}"


@dataclass(frozen=True)
class RunSpec:
    order: int
    method: str
    seed: int
    learning_rate: float
    l1_alpha: float = 0.0
    rewa_k: float = 3.0
    rewa_m: float = 2.0
    rewa_weight_decay: float = 1e-3
    rewa_eps: float = 1e-8

    @property
    def name(self) -> str:
        lr = slug_number(self.learning_rate)
        if self.method == "dense":
            return f"dense-lr{lr}-seed{self.seed}"
        if self.method == "l1":
            alpha = slug_number(self.l1_alpha)
            return f"l1-a{alpha}-lr{lr}-seed{self.seed}"
        k = slug_number(self.rewa_k)
        m = slug_number(self.rewa_m)
        decay = slug_number(self.rewa_weight_decay)
        return f"rewa-k{k}-m{m}-wd{decay}-lr{lr}-seed{self.seed}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Python used for child scripts")
    parser.add_argument("--data-dir", type=Path, default=Path("data/tinystories"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/stage1-pilot"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/stage1"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--pruning-seed", type=int, default=20_260_922)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float16"
    )

    parser.add_argument("--train-tokens", type=positive_int, default=20_000_000)
    parser.add_argument(
        "--max-iters",
        type=positive_int,
        default=None,
        help="Override the token-derived iteration count, primarily for dry smoke runs.",
    )
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--eval-batch-size", type=positive_int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=positive_int, default=16)
    parser.add_argument("--seq-len", type=positive_int, default=256)
    parser.add_argument("--n-layer", type=positive_int, default=6)
    parser.add_argument("--n-head", type=positive_int, default=6)
    parser.add_argument("--n-embd", type=positive_int, default=384)
    parser.add_argument("--intermediate-size", type=positive_int, default=1024)
    parser.add_argument("--eval-iters", type=positive_int, default=50)
    parser.add_argument("--eval-interval", type=positive_int, default=100)
    parser.add_argument("--log-interval", type=positive_int, default=10)
    parser.add_argument("--save-interval", type=positive_int, default=100)
    parser.add_argument("--warmup-iters", type=nonnegative_int, default=50)
    parser.add_argument("--weight-decay", type=finite_float, default=0.1)

    parser.add_argument(
        "--dense-learning-rates", type=finite_float, nargs="+", default=[3e-4, 6e-4]
    )
    parser.add_argument(
        "--l1-alphas", type=finite_float, nargs="+", default=[1e-7, 1e-6, 1e-5]
    )
    parser.add_argument(
        "--rewa-configs",
        type=rewa_config,
        nargs="+",
        default=[(9.0, 2.0)],
        metavar="K:M",
        help="Theory-conforming ReWA (K, M) pairs; requires 0 <= M < K - 1",
    )
    parser.add_argument(
        "--rewa-weight-decays",
        type=finite_float,
        nargs="+",
        default=[1e-4, 1e-1, 1.0],
    )
    parser.add_argument(
        "--rewa-eps",
        type=finite_float,
        default=0.0,
        help="ReWA attenuation epsilon; the canonical K=9, M=2 AdamW recipe uses 0",
    )

    parser.add_argument(
        "--sparsities",
        type=probability,
        nargs="+",
        default=list(DEFAULT_SPARSITIES),
    )
    parser.add_argument("--pruning-eval-iters", type=positive_int, default=100)
    parser.add_argument("--pruning-batch-size", type=positive_int, default=16)
    parser.add_argument("--force-evaluation", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--no-summarize",
        action="store_true",
        help="Do not invoke summarize_stage1.py after the sweep.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for label, values in (
        ("dense learning rates", args.dense_learning_rates),
        ("L1 alphas", args.l1_alphas),
        ("ReWA weight decays", args.rewa_weight_decays),
    ):
        if not values or any(value <= 0 for value in values):
            raise ValueError(f"{label} must contain positive values")
    if not args.rewa_configs:
        raise ValueError("at least one ReWA K:M configuration is required")
    for k, m in args.rewa_configs:
        if k <= 1.0 or m < 0.0 or m >= k - 1.0:
            raise ValueError("ReWA requires K > 1 and 0 <= M < K - 1")
    if args.rewa_eps < 0 or args.weight_decay < 0:
        raise ValueError("epsilon and weight decay must be non-negative")


def expected_iterations(args: argparse.Namespace) -> int:
    if args.max_iters is not None:
        return args.max_iters
    tokens_per_iter = args.batch_size * args.seq_len * args.gradient_accumulation_steps
    return math.ceil(args.train_tokens / tokens_per_iter)


def read_metrics(path: Path) -> tuple[int, float | None]:
    max_iter = 0
    best_val: float | None = None
    if not path.exists():
        return max_iter, best_val
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            max_iter = max(max_iter, int(record.get("iter", 0)))
            value = record.get("val_loss")
            if value is not None:
                value = float(value)
                if math.isfinite(value) and (best_val is None or value < best_val):
                    best_val = value
    return max_iter, best_val


def assert_compatible_existing_run(
    config_path: Path, spec: RunSpec, args: argparse.Namespace, data_dir: Path
) -> None:
    if not config_path.exists():
        return
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    saved = payload.get("train_args", {})
    expected: dict[str, Any] = {
        "method": spec.method,
        "seed": spec.seed,
        "eval_seed": args.eval_seed,
        "device": args.device,
        "dtype": args.dtype,
        "learning_rate": spec.learning_rate,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "seq_len": args.seq_len,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_embd": args.n_embd,
        "intermediate_size": args.intermediate_size,
        "train_tokens": args.train_tokens,
        "max_iters": args.max_iters,
        "eval_iters": args.eval_iters,
        "eval_interval": args.eval_interval,
        "warmup_iters": args.warmup_iters,
        "weight_decay": args.weight_decay,
    }
    if spec.method == "l1":
        expected["l1_alpha"] = spec.l1_alpha
    elif spec.method == "rewa":
        expected.update(
            rewa_k=spec.rewa_k,
            rewa_m=spec.rewa_m,
            rewa_weight_decay=spec.rewa_weight_decay,
            rewa_eps=spec.rewa_eps,
        )
    mismatches = {
        key: (saved.get(key), value)
        for key, value in expected.items()
        if saved.get(key) != value
    }
    saved_data_dir = Path(str(saved.get("data_dir", "")))
    if not saved_data_dir.is_absolute():
        saved_data_dir = PROJECT_ROOT / saved_data_dir
    if saved_data_dir.resolve() != data_dir.resolve():
        mismatches["data_dir"] = (str(saved_data_dir.resolve()), str(data_dir.resolve()))
    if mismatches:
        details = ", ".join(
            f"{key}: saved={old!r}, requested={new!r}"
            for key, (old, new) in mismatches.items()
        )
        raise RuntimeError(
            f"existing run {config_path.parent.name!r} has incompatible settings ({details}); "
            "choose a different --output-root to preserve the existing run"
        )


def run_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = subprocess.list2cmdline(command)
    print(f"\n$ {printable}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] $ {printable}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["run_name"]: row for row in csv.DictReader(handle)}


def write_manifest(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    ordered = sorted(
        rows.values(), key=lambda row: (int(row.get("order", 1_000_000)), row["run_name"])
    )
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in ordered:
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})
    os.replace(temporary, path)


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"


def paths_for_run(
    spec: RunSpec, output_root: Path, artifact_dir: Path
) -> dict[str, Path]:
    run_dir = output_root / spec.name
    evaluation_dir = artifact_dir / "runs" / spec.name
    return {
        "run_dir": run_dir,
        "checkpoint": run_dir / "best.pt",
        "latest": run_dir / "latest.pt",
        "metrics": run_dir / "metrics.jsonl",
        "run_config": run_dir / "run_config.json",
        "train_log": run_dir / "train.log",
        "pruning_json": evaluation_dir / "global-pruning.json",
        "pruning_csv": evaluation_dir / "global-pruning.csv",
        "layer_csv": evaluation_dir / "layer-sparsity.csv",
    }


def base_manifest_row(
    spec: RunSpec, args: argparse.Namespace, paths: dict[str, Path], commit: str
) -> dict[str, Any]:
    return {
        "order": spec.order,
        "run_name": spec.name,
        "method": spec.method,
        "status": "planned",
        "seed": spec.seed,
        "learning_rate": spec.learning_rate,
        "dense_lr_selected": "",
        "l1_alpha": spec.l1_alpha if spec.method == "l1" else "",
        "rewa_k": spec.rewa_k if spec.method == "rewa" else "",
        "rewa_m": spec.rewa_m if spec.method == "rewa" else "",
        "rewa_weight_decay": spec.rewa_weight_decay if spec.method == "rewa" else "",
        "rewa_eps": spec.rewa_eps if spec.method == "rewa" else "",
        "train_tokens": args.train_tokens,
        "expected_iters": expected_iterations(args),
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "seq_len": args.seq_len,
        "eval_iters": args.eval_iters,
        "pruning_eval_iters": args.pruning_eval_iters,
        "git_commit": commit,
        "checkpoint": display_path(paths["checkpoint"]),
        "metrics_jsonl": display_path(paths["metrics"]),
        "pruning_csv": display_path(paths["pruning_csv"]),
        "layer_csv": display_path(paths["layer_csv"]),
        "pruning_json": display_path(paths["pruning_json"]),
        "run_config": display_path(paths["run_config"]),
        "train_log": display_path(paths["train_log"]),
        "superseded_from_status": "",
        "status_detail": "",
        "error": "",
        "updated_at_utc": utc_now(),
    }


def update_row(
    manifest_path: Path,
    rows: dict[str, dict[str, Any]],
    name: str,
    **updates: Any,
) -> None:
    rows[name].update(updates)
    rows[name]["updated_at_utc"] = utc_now()
    write_manifest(manifest_path, rows)


def train_command(
    spec: RunSpec,
    args: argparse.Namespace,
    data_dir: Path,
    paths: dict[str, Path],
    resume: bool,
) -> list[str]:
    command = [
        args.python,
        str(PROJECT_ROOT / "train.py"),
        "--data-dir",
        str(data_dir),
        "--out-dir",
        str(paths["run_dir"]),
        "--method",
        spec.method,
        "--seed",
        str(spec.seed),
        "--eval-seed",
        str(args.eval_seed),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--seq-len",
        str(args.seq_len),
        "--n-layer",
        str(args.n_layer),
        "--n-head",
        str(args.n_head),
        "--n-embd",
        str(args.n_embd),
        "--intermediate-size",
        str(args.intermediate_size),
        "--batch-size",
        str(args.batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--train-tokens",
        str(args.train_tokens),
        "--eval-iters",
        str(args.eval_iters),
        "--eval-interval",
        str(args.eval_interval),
        "--log-interval",
        str(args.log_interval),
        "--save-interval",
        str(args.save_interval),
        "--warmup-iters",
        str(args.warmup_iters),
        "--learning-rate",
        str(spec.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
    ]
    if args.eval_batch_size is not None:
        command.extend(("--eval-batch-size", str(args.eval_batch_size)))
    if args.max_iters is not None:
        command.extend(("--max-iters", str(args.max_iters)))
    if spec.method == "l1":
        command.extend(("--l1-alpha", str(spec.l1_alpha)))
    elif spec.method == "rewa":
        command.extend(
            (
                "--rewa-k",
                str(spec.rewa_k),
                "--rewa-m",
                str(spec.rewa_m),
                "--rewa-eps",
                str(spec.rewa_eps),
                "--rewa-weight-decay",
                str(spec.rewa_weight_decay),
            )
        )
    if resume:
        command.extend(("--resume", str(paths["latest"])))
    return command


def normalize_sparsities(values: Iterable[float]) -> list[float]:
    result: list[float] = []
    for value in (0.0, *values):
        if not any(math.isclose(value, prior, abs_tol=1e-12) for prior in result):
            result.append(float(value))
    return result


def evaluation_is_complete(
    paths: dict[str, Path], args: argparse.Namespace, data_dir: Path
) -> bool:
    if not all(paths[key].exists() for key in ("pruning_json", "pruning_csv", "layer_csv")):
        return False
    try:
        payload = json.loads(paths["pruning_json"].read_text(encoding="utf-8"))
        evaluation = payload["evaluation"]
        acceptable_dtypes = (
            {args.dtype, "float32"} if args.device == "auto" else
            {args.dtype if args.device.startswith("cuda") else "float32"}
        )
        if (
            int(evaluation["batch_size"]) != args.pruning_batch_size
            or int(evaluation["eval_iters"]) != args.pruning_eval_iters
            or int(evaluation["seed"]) != args.pruning_seed
            or evaluation["dtype"] not in acceptable_dtypes
            or Path(payload["data_dir"]).resolve() != data_dir.resolve()
            or Path(payload["checkpoint"]).resolve() != paths["checkpoint"].resolve()
        ):
            return False
        observed = [float(row["target_sparsity"]) for row in payload["results"]]
        requested = normalize_sparsities(args.sparsities)
        return len(observed) == len(requested) and all(
            any(math.isclose(value, item, abs_tol=1e-12) for item in observed)
            for value in requested
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def evaluate_command(
    args: argparse.Namespace,
    data_dir: Path,
    paths: dict[str, Path],
) -> list[str]:
    sparsities = normalize_sparsities(args.sparsities)
    return [
        args.python,
        str(PROJECT_ROOT / "evaluate_global_pruning.py"),
        "--checkpoint",
        str(paths["checkpoint"]),
        "--data-dir",
        str(data_dir),
        "--sparsities",
        *(str(value) for value in sparsities),
        "--batch-size",
        str(args.pruning_batch_size),
        "--seq-len",
        str(args.seq_len),
        "--eval-iters",
        str(args.pruning_eval_iters),
        "--seed",
        str(args.pruning_seed),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--output-json",
        str(paths["pruning_json"]),
        "--output-csv",
        str(paths["pruning_csv"]),
        "--layer-csv",
        str(paths["layer_csv"]),
    ]


def execute_run(
    spec: RunSpec,
    args: argparse.Namespace,
    data_dir: Path,
    output_root: Path,
    artifact_dir: Path,
    manifest_path: Path,
    rows: dict[str, dict[str, Any]],
    commit: str,
) -> bool:
    paths = paths_for_run(spec, output_root, artifact_dir)
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    paths["pruning_csv"].parent.mkdir(parents=True, exist_ok=True)
    current = base_manifest_row(spec, args, paths, commit)
    current.update(rows.get(spec.name, {}))
    current.update(base_manifest_row(spec, args, paths, commit))
    rows[spec.name] = current
    write_manifest(manifest_path, rows)

    try:
        assert_compatible_existing_run(paths["run_config"], spec, args, data_dir)
        completed_iters, best_val = read_metrics(paths["metrics"])
        training_complete = (
            paths["checkpoint"].exists()
            and paths["latest"].exists()
            and completed_iters >= expected_iterations(args)
            and best_val is not None
        )
        if training_complete:
            print(f"[{spec.name}] training already complete; skipping", flush=True)
        else:
            resume = paths["latest"].exists() and completed_iters > 0
            update_row(
                manifest_path,
                rows,
                spec.name,
                status="training",
                completed_iters=completed_iters,
                error="",
            )
            code = run_command(
                train_command(spec, args, data_dir, paths, resume), paths["train_log"]
            )
            if code != 0:
                raise RuntimeError(f"training exited with status {code}")
            completed_iters, best_val = read_metrics(paths["metrics"])
            if (
                not paths["checkpoint"].exists()
                or completed_iters < expected_iterations(args)
                or best_val is None
            ):
                raise RuntimeError("training returned success without complete checkpoint/metrics")

        assert best_val is not None
        config_copy = artifact_dir / "configs" / f"{spec.name}.json"
        config_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths["run_config"], config_copy)
        update_row(
            manifest_path,
            rows,
            spec.name,
            status="trained",
            completed_iters=completed_iters,
            best_val_loss=f"{best_val:.12g}",
            best_val_ppl=f"{math.exp(best_val):.12g}" if best_val < 709 else "inf",
            error="",
        )

        if args.force_evaluation or not evaluation_is_complete(paths, args, data_dir):
            update_row(manifest_path, rows, spec.name, status="evaluating")
            eval_log = paths["pruning_csv"].parent / "evaluation.log"
            code = run_command(evaluate_command(args, data_dir, paths), eval_log)
            if code != 0:
                raise RuntimeError(f"global-pruning evaluation exited with status {code}")
            if not evaluation_is_complete(paths, args, data_dir):
                raise RuntimeError("evaluation returned success without the requested artifacts")
        else:
            print(f"[{spec.name}] pruning evaluation already complete; skipping", flush=True)

        update_row(manifest_path, rows, spec.name, status="completed", error="")
        return True
    except Exception as error:
        completed_iters, best_val = read_metrics(paths["metrics"])
        update_row(
            manifest_path,
            rows,
            spec.name,
            status="failed",
            completed_iters=completed_iters,
            best_val_loss="" if best_val is None else f"{best_val:.12g}",
            best_val_ppl=(
                ""
                if best_val is None
                else (f"{math.exp(best_val):.12g}" if best_val < 709 else "inf")
            ),
            error=str(error),
        )
        print(f"[{spec.name}] FAILED: {error}", file=sys.stderr, flush=True)
        if args.fail_fast:
            raise
        return False


def select_dense_learning_rate(
    dense_specs: list[RunSpec], rows: dict[str, dict[str, Any]]
) -> float:
    candidates: list[tuple[float, float, str]] = []
    for spec in dense_specs:
        row = rows.get(spec.name, {})
        if row.get("status") != "completed" or not row.get("best_val_loss"):
            continue
        loss = float(row["best_val_loss"])
        if math.isfinite(loss):
            candidates.append((loss, spec.learning_rate, spec.name))
    if not candidates:
        raise RuntimeError("no dense candidate completed; an L1/ReWA learning rate cannot be selected")
    loss, learning_rate, name = min(candidates)
    print(
        f"Selected dense learning rate {learning_rate:g} from {name} "
        f"(best_val_loss={loss:.6f})",
        flush=True,
    )
    return learning_rate


def reconcile_selected_learning_rate(
    rows: dict[str, dict[str, Any]], selected_lr: float
) -> list[str]:
    """Exclude stale regularized runs created with a superseded dense LR.

    An interrupted invocation can leave completed L1/ReWA rows in the
    manifest before a later invocation finishes the dense sweep and selects a
    different learning rate.  Keep those rows and artifact paths as
    provenance, but make their exclusion from the active sweep explicit.
    """

    superseded: list[str] = []
    for name, row in rows.items():
        if row.get("method") not in {"l1", "rewa"}:
            continue
        try:
            run_lr = float(row.get("learning_rate", ""))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(run_lr) or math.isclose(
            run_lr, selected_lr, rel_tol=1e-12, abs_tol=0.0
        ):
            continue

        current_status = str(row.get("status", "") or "unknown")
        previous_status = str(
            row.get("superseded_from_status", "")
            or (current_status if current_status != "superseded" else "unknown")
        )
        detail = (
            f"Superseded from status={previous_status}: run learning_rate={run_lr:g} "
            f"differs from selected dense learning_rate={selected_lr:g}."
        )
        changed = (
            current_status != "superseded"
            or row.get("superseded_from_status") != previous_status
            or row.get("status_detail") != detail
        )
        row.update(
            status="superseded",
            dense_lr_selected="",
            superseded_from_status=previous_status,
            status_detail=detail,
        )
        if changed:
            row["updated_at_utc"] = utc_now()
            superseded.append(name)
    return sorted(superseded)


def reconcile_active_regularized_runs(
    rows: dict[str, dict[str, Any]], active_run_names: set[str]
) -> list[str]:
    """Mark L1/ReWA rows outside the current frozen grid as superseded."""

    superseded: list[str] = []
    for name, row in rows.items():
        if row.get("method") not in {"l1", "rewa"} or name in active_run_names:
            continue
        current_status = str(row.get("status", "") or "unknown")
        previous_status = str(
            row.get("superseded_from_status", "")
            or (current_status if current_status != "superseded" else "unknown")
        )
        detail = (
            f"Superseded from status={previous_status}: run is not part of the "
            "current frozen L1/ReWA candidate grid."
        )
        changed = (
            current_status != "superseded"
            or row.get("superseded_from_status") != previous_status
            or row.get("status_detail") != detail
        )
        row.update(
            status="superseded",
            dense_lr_selected="",
            superseded_from_status=previous_status,
            status_detail=detail,
        )
        if changed:
            row["updated_at_utc"] = utc_now()
            superseded.append(name)
    return sorted(superseded)


def main() -> int:
    args = parse_args()
    validate_args(args)
    data_dir = project_path(args.data_dir)
    output_root = project_path(args.output_root)
    artifact_dir = project_path(args.artifact_dir)
    manifest_path = artifact_dir / "run-manifest.csv"
    for required in ("metadata.json", "train.bin", "validation.bin"):
        if not (data_dir / required).exists():
            raise FileNotFoundError(
                f"prepared TinyStories file missing: {data_dir / required}; "
                "run scripts/prepare_tinystories.py first"
            )
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict[str, Any]] = load_manifest(manifest_path)
    commit = git_commit()
    failures = 0

    dense_specs = [
        RunSpec(order=index, method="dense", seed=args.seed, learning_rate=lr)
        for index, lr in enumerate(args.dense_learning_rates)
    ]
    for spec in dense_specs:
        if not execute_run(
            spec,
            args,
            data_dir,
            output_root,
            artifact_dir,
            manifest_path,
            rows,
            commit,
        ):
            failures += 1

    selected_lr = select_dense_learning_rate(dense_specs, rows)
    for spec in dense_specs:
        update_row(
            manifest_path,
            rows,
            spec.name,
            dense_lr_selected="yes" if spec.learning_rate == selected_lr else "no",
        )
    superseded = reconcile_selected_learning_rate(rows, selected_lr)
    if superseded:
        write_manifest(manifest_path, rows)
        print(
            "Marked stale non-dense run(s) as superseded after dense LR selection: "
            + ", ".join(superseded),
            flush=True,
        )
    order = len(dense_specs)
    remaining_specs: list[RunSpec] = []
    for alpha in args.l1_alphas:
        remaining_specs.append(
            RunSpec(
                order=order,
                method="l1",
                seed=args.seed,
                learning_rate=selected_lr,
                l1_alpha=alpha,
            )
        )
        order += 1
    for k, m in args.rewa_configs:
        for decay in args.rewa_weight_decays:
            remaining_specs.append(
                RunSpec(
                    order=order,
                    method="rewa",
                    seed=args.seed,
                    learning_rate=selected_lr,
                    rewa_k=k,
                    rewa_m=m,
                    rewa_weight_decay=decay,
                    rewa_eps=args.rewa_eps,
                )
            )
            order += 1

    off_grid = reconcile_active_regularized_runs(
        rows, {spec.name for spec in remaining_specs}
    )
    if off_grid:
        write_manifest(manifest_path, rows)
        print(
            "Marked non-active L1/ReWA run(s) as superseded: "
            + ", ".join(off_grid),
            flush=True,
        )

    for spec in remaining_specs:
        if not execute_run(
            spec,
            args,
            data_dir,
            output_root,
            artifact_dir,
            manifest_path,
            rows,
            commit,
        ):
            failures += 1

    if not args.no_summarize:
        summary_log = artifact_dir / "summarize.log"
        code = run_command(
            [
                args.python,
                str(PROJECT_ROOT / "scripts" / "summarize_stage1.py"),
                "--manifest",
                str(manifest_path),
                "--artifact-dir",
                str(artifact_dir),
            ],
            summary_log,
        )
        if code != 0:
            failures += 1
            print(f"summary generation failed with status {code}", file=sys.stderr)

    completed = sum(row.get("status") == "completed" for row in rows.values())
    print(
        f"Stage-one pilot finished: {completed} completed run(s), {failures} failure(s). "
        f"Manifest: {manifest_path}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
