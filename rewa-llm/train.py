#!/usr/bin/env python3
"""Train the stage-one TinyStories model with Dense, L1, or ReWA AdamW."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from rewa_llm.data import TokenDataset
from rewa_llm.model import ModelConfig, ReWALanguageModel, is_rewa_parameter
from rewa_llm.optim import ReWAAdamW


METHODS = ("dense", "l1", "rewa")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float16"
    )

    # The defaults instantiate the approximately 14M-parameter stage-one model.
    parser.add_argument("--seq-len", type=positive_int, default=256)
    parser.add_argument("--n-layer", type=positive_int, default=6)
    parser.add_argument("--n-head", type=positive_int, default=6)
    parser.add_argument("--n-embd", type=positive_int, default=384)
    parser.add_argument("--intermediate-size", type=positive_int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--eval-batch-size", type=positive_int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=positive_int, default=16)
    parser.add_argument("--train-tokens", type=positive_int, default=20_000_000)
    parser.add_argument(
        "--max-iters",
        type=positive_int,
        default=None,
        help="Override the iteration count derived from --train-tokens (for smoke tests).",
    )
    parser.add_argument("--eval-iters", type=positive_int, default=50)
    parser.add_argument("--eval-interval", type=positive_int, default=100)
    parser.add_argument("--log-interval", type=positive_int, default=10)
    parser.add_argument("--save-interval", type=positive_int, default=100)
    parser.add_argument("--eval-at-start", action="store_true")

    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-iters", type=nonnegative_int, default=50)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--l1-alpha", type=float, default=0.0)
    parser.add_argument("--rewa-k", type=float, default=9.0)
    parser.add_argument("--rewa-m", type=float, default=2.0)
    parser.add_argument("--rewa-eps", type=float, default=0.0)
    parser.add_argument("--rewa-weight-decay", type=float, default=1e-4)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    if args.n_embd % args.n_head:
        raise ValueError("--n-embd must be divisible by --n-head")
    if (args.n_embd // args.n_head) % 2:
        raise ValueError("attention head dimension must be even for RoPE")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("--min-lr-ratio must be in [0, 1]")
    if not 0.0 <= args.beta1 < 1.0 or not 0.0 <= args.beta2 < 1.0:
        raise ValueError("Adam betas must be in [0, 1)")
    if min(
        args.adam_eps,
        args.weight_decay,
        args.grad_clip,
        args.l1_alpha,
        args.rewa_eps,
        args.rewa_weight_decay,
    ) < 0:
        raise ValueError("epsilons, decays, clipping, and L1 alpha must be non-negative")
    if args.method == "rewa" and not (
        math.isfinite(args.rewa_k)
        and math.isfinite(args.rewa_m)
        and args.rewa_k > 1.0
        and 0.0 <= args.rewa_m < args.rewa_k - 1.0
    ):
        raise ValueError("ReWA requires K > 1 and 0 <= M < K - 1")
    if args.method == "l1" and args.l1_alpha == 0:
        raise ValueError("the l1 method requires a positive --l1-alpha")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", 0)
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parameter_groups(
    model: nn.Module,
    method: str,
    weight_decay: float,
    rewa_weight_decay: float,
    rewa_k: float,
    rewa_m: float,
    rewa_eps: float,
) -> tuple[list[dict[str, Any]], list[nn.Parameter], list[str]]:
    """Create the same structural groups for every method.

    Attention/MLP matrices are the eligible group. Other matrix-valued
    parameters (the tied embedding/head) receive ordinary AdamW decay, while
    vectors such as RMSNorm weights receive no decay.
    """
    eligible: list[nn.Parameter] = []
    eligible_names: list[str] = []
    ordinary_decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if is_rewa_parameter(name, parameter):
            eligible.append(parameter)
            eligible_names.append(name)
        elif parameter.ndim >= 2:
            ordinary_decay.append(parameter)
        else:
            no_decay.append(parameter)

    groups: list[dict[str, Any]] = [
        {
            "params": eligible,
            "weight_decay": rewa_weight_decay if method == "rewa" else weight_decay,
            "rewa": method == "rewa",
            "rewa_k": rewa_k,
            "rewa_m": rewa_m,
            "rewa_eps": rewa_eps,
        },
        {"params": ordinary_decay, "weight_decay": weight_decay, "rewa": False},
        {"params": no_decay, "weight_decay": 0.0, "rewa": False},
    ]

    grouped = [parameter for group in groups for parameter in group["params"]]
    expected = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if len({id(parameter) for parameter in grouped}) != len(grouped):
        raise RuntimeError("a trainable parameter appears in more than one optimizer group")
    if {id(parameter) for parameter in grouped} != {id(parameter) for parameter in expected}:
        raise RuntimeError("optimizer groups do not cover exactly the trainable parameters")
    return groups, eligible, eligible_names


def build_optimizer(
    model: nn.Module, args: argparse.Namespace
) -> tuple[torch.optim.Optimizer, list[nn.Parameter], list[str]]:
    groups, eligible, eligible_names = parameter_groups(
        model=model,
        method=args.method,
        weight_decay=args.weight_decay,
        rewa_weight_decay=args.rewa_weight_decay,
        rewa_k=args.rewa_k,
        rewa_m=args.rewa_m,
        rewa_eps=args.rewa_eps,
    )
    common = {
        "lr": args.learning_rate,
        "betas": (args.beta1, args.beta2),
        "eps": args.adam_eps,
    }
    if args.method == "rewa":
        optimizer: torch.optim.Optimizer = ReWAAdamW(
            groups,
            **common,
            rewa_k=args.rewa_k,
            rewa_m=args.rewa_m,
            rewa_eps=args.rewa_eps,
        )
    else:
        optimizer = torch.optim.AdamW(groups, **common)
    return optimizer, eligible, eligible_names


def iterations_for_budget(args: argparse.Namespace) -> int:
    if args.max_iters is not None:
        return args.max_iters
    tokens_per_iter = args.batch_size * args.seq_len * args.gradient_accumulation_steps
    return math.ceil(args.train_tokens / tokens_per_iter)


def learning_rate_at(iter_num: int, max_iters: int, args: argparse.Namespace) -> float:
    if args.warmup_iters > 0 and iter_num < args.warmup_iters:
        return args.learning_rate * (iter_num + 1) / args.warmup_iters
    if max_iters <= args.warmup_iters:
        return args.learning_rate
    progress = min(1.0, (iter_num - args.warmup_iters) / (max_iters - args.warmup_iters))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.learning_rate * (args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine)


def set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda" or dtype_name == "float32":
        return nullcontext()
    dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def perplexity(loss: float | None) -> float | None:
    if loss is None:
        return None
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


@torch.no_grad()
def evaluate(
    model: ReWALanguageModel,
    dataset: TokenDataset,
    batch_size: int,
    seq_len: int,
    eval_iters: int,
    eval_seed: int,
    device: torch.device,
    dtype_name: str,
) -> float:
    was_training = model.training
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(eval_seed)
    losses: list[float] = []
    for _ in range(eval_iters):
        x, y = dataset.batch(batch_size, seq_len, device, generator)
        with autocast_context(device, dtype_name):
            _, loss = model(x, y)
        assert loss is not None
        losses.append(float(loss.detach()))
    model.train(was_training)
    return sum(losses) / len(losses)


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch before weights_only was added
        return torch.load(path, map_location=device)


def save_checkpoint(
    path: Path,
    model: ReWALanguageModel,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    iter_num: int,
    best_val_loss: float,
    tokens_seen: int,
    scaler: torch.amp.GradScaler,
    train_generator: torch.Generator,
) -> None:
    # The first six keys are the stable evaluator-facing checkpoint contract.
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": model.config.to_dict(),
        "train_args": jsonable_args(args),
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        # Runtime state below allows a faithful training resume.
        "tokens_seen": tokens_seen,
        "scaler": scaler.state_dict(),
        "train_generator_state": train_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def metric_record(
    *,
    iter_num: int,
    tokens_seen: int,
    train_loss: float | None,
    train_objective: float | None,
    val_loss: float | None,
    learning_rate: float,
    elapsed_sec: float,
    device: torch.device,
    tokens_per_sec: float | None,
    skipped_optimizer_steps: int,
) -> dict[str, Any]:
    max_cuda_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    return {
        "tokens_seen": tokens_seen,
        "iter": iter_num,
        "train_loss": train_loss,
        "train_objective": train_objective,
        "val_loss": val_loss,
        "ppl": perplexity(val_loss),
        "lr": learning_rate,
        "elapsed_sec": elapsed_sec,
        "max_cuda_memory": max_cuda_memory,
        "tokens_per_sec": tokens_per_sec,
        "skipped_optimizer_steps": skipped_optimizer_steps,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    train_data = TokenDataset(args.data_dir, "train")
    validation_data = TokenDataset(args.data_dir, "validation")
    if train_data.vocab_size != validation_data.vocab_size:
        raise ValueError("training and validation vocabulary sizes differ")

    resume_checkpoint = load_checkpoint(args.resume, device) if args.resume else None
    if resume_checkpoint is not None:
        saved_method = resume_checkpoint.get("train_args", {}).get("method")
        if saved_method is not None and saved_method != args.method:
            raise ValueError(
                f"checkpoint method is {saved_method!r}, but --method is {args.method!r}"
            )
        model_config = ModelConfig(**resume_checkpoint["model_config"])
    else:
        model_config = ModelConfig(
            vocab_size=train_data.vocab_size,
            max_seq_len=args.seq_len,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            intermediate_size=args.intermediate_size,
            dropout=args.dropout,
        )
    if model_config.vocab_size != train_data.vocab_size:
        raise ValueError("checkpoint/model vocabulary does not match prepared data")
    if args.seq_len > model_config.max_seq_len:
        raise ValueError("--seq-len exceeds the model checkpoint's maximum sequence length")

    model = ReWALanguageModel(model_config).to(device)
    optimizer, eligible_parameters, eligible_names = build_optimizer(model, args)
    use_grad_scaler = device.type == "cuda" and args.dtype == "float16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    train_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)

    iter_num = 0
    tokens_seen = 0
    best_val_loss = float("inf")
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        iter_num = int(resume_checkpoint["iter_num"])
        best_val_loss = float(resume_checkpoint["best_val_loss"])
        tokens_seen = int(resume_checkpoint.get("tokens_seen", 0))
        if "scaler" in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint["scaler"])
        if "train_generator_state" in resume_checkpoint:
            # ``torch.load(..., map_location=cuda)`` also moves serialized CPU
            # RNG byte tensors.  CPU generators require their state on CPU.
            train_generator.set_state(
                resume_checkpoint["train_generator_state"].cpu()
            )
        if "torch_rng_state" in resume_checkpoint:
            torch.set_rng_state(resume_checkpoint["torch_rng_state"].cpu())
        cuda_states = resume_checkpoint.get("cuda_rng_state_all")
        if cuda_states is not None and device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "metrics.jsonl"
    if resume_checkpoint is None:
        metrics_path.write_text("", encoding="utf-8")
    run_description = {
        "train_args": jsonable_args(args),
        "model_config": model_config.to_dict(),
        "num_parameters": model.num_parameters(),
        "eligible_parameters": sum(parameter.numel() for parameter in eligible_parameters),
        "eligible_parameter_names": eligible_names,
        "data_metadata": json.loads(
            (args.data_dir / "metadata.json").read_text(encoding="utf-8")
        ),
    }
    (args.out_dir / "run_config.json").write_text(
        json.dumps(run_description, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    max_iters = iterations_for_budget(args)
    if iter_num >= max_iters:
        raise ValueError(f"checkpoint is already at iteration {iter_num} >= {max_iters}")
    eval_batch_size = args.eval_batch_size or args.batch_size
    tokens_per_iter = args.batch_size * args.seq_len * args.gradient_accumulation_steps
    print(
        json.dumps(
            {
                "device": str(device),
                "method": args.method,
                "parameters": model.num_parameters(),
                "eligible_parameters": run_description["eligible_parameters"],
                "tokens_per_iter": tokens_per_iter,
                "max_iters": max_iters,
                "target_tokens": args.train_tokens,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    started = time.perf_counter()
    last_log_time = started
    loss_since_log = 0.0
    objective_since_log = 0.0
    steps_since_log = 0
    skipped_steps_since_log = 0
    model.train()

    if args.eval_at_start and iter_num == 0:
        val_loss = evaluate(
            model,
            validation_data,
            eval_batch_size,
            args.seq_len,
            args.eval_iters,
            args.eval_seed,
            device,
            args.dtype,
        )
        record = metric_record(
            iter_num=0,
            tokens_seen=0,
            train_loss=None,
            train_objective=None,
            val_loss=val_loss,
            learning_rate=0.0,
            elapsed_sec=time.perf_counter() - started,
            device=device,
            tokens_per_sec=None,
            skipped_optimizer_steps=0,
        )
        append_jsonl(metrics_path, record)
        print(json.dumps(record, sort_keys=True), flush=True)

    while iter_num < max_iters:
        current_lr = learning_rate_at(iter_num, max_iters, args)
        set_learning_rate(optimizer, current_lr)
        optimizer.zero_grad(set_to_none=True)
        step_ce_loss = 0.0
        step_objective = 0.0
        for _ in range(args.gradient_accumulation_steps):
            x, y = train_data.batch(
                args.batch_size, args.seq_len, device, train_generator
            )
            with autocast_context(device, args.dtype):
                _, ce_loss = model(x, y)
                assert ce_loss is not None
                objective = ce_loss
                if args.method == "l1":
                    l1_penalty = sum(
                        parameter.abs().sum() for parameter in eligible_parameters
                    )
                    objective = objective + args.l1_alpha * l1_penalty
            if not torch.isfinite(objective.detach()):
                raise FloatingPointError(
                    f"non-finite training objective at iteration {iter_num}"
                )
            step_ce_loss += float(ce_loss.detach())
            step_objective += float(objective.detach())
            scaler.scale(objective / args.gradient_accumulation_steps).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.is_enabled() and scaler.get_scale() < scale_before_step:
            skipped_steps_since_log += 1

        iter_num += 1
        tokens_seen += tokens_per_iter
        step_ce_loss /= args.gradient_accumulation_steps
        step_objective /= args.gradient_accumulation_steps
        loss_since_log += step_ce_loss
        objective_since_log += step_objective
        steps_since_log += 1

        should_eval = iter_num % args.eval_interval == 0 or iter_num == max_iters
        should_log = iter_num % args.log_interval == 0 or should_eval
        val_loss: float | None = None
        if should_eval:
            val_loss = evaluate(
                model,
                validation_data,
                eval_batch_size,
                args.seq_len,
                args.eval_iters,
                args.eval_seed,
                device,
                args.dtype,
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    args.out_dir / "best.pt",
                    model,
                    optimizer,
                    args,
                    iter_num,
                    best_val_loss,
                    tokens_seen,
                    scaler,
                    train_generator,
                )

        if should_log:
            now = time.perf_counter()
            interval_seconds = now - last_log_time
            interval_tokens = steps_since_log * tokens_per_iter
            record = metric_record(
                iter_num=iter_num,
                tokens_seen=tokens_seen,
                train_loss=loss_since_log / steps_since_log,
                train_objective=objective_since_log / steps_since_log,
                val_loss=val_loss,
                learning_rate=current_lr,
                elapsed_sec=now - started,
                device=device,
                tokens_per_sec=interval_tokens / interval_seconds,
                skipped_optimizer_steps=skipped_steps_since_log,
            )
            append_jsonl(metrics_path, record)
            print(json.dumps(record, sort_keys=True), flush=True)
            last_log_time = now
            loss_since_log = 0.0
            objective_since_log = 0.0
            steps_since_log = 0
            skipped_steps_since_log = 0

        if iter_num % args.save_interval == 0 or iter_num == max_iters:
            save_checkpoint(
                args.out_dir / "latest.pt",
                model,
                optimizer,
                args,
                iter_num,
                best_val_loss,
                tokens_seen,
                scaler,
                train_generator,
            )

    print(
        f"completed {iter_num} iterations and {tokens_seen:,} training tokens; "
        f"best validation loss {best_val_loss:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
