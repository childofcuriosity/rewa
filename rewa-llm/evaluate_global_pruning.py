#!/usr/bin/env python3
"""Evaluate a trained checkpoint under exact global magnitude pruning."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
from dataclasses import fields, is_dataclass, asdict
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch

from rewa_llm.data import TokenDataset
from rewa_llm.model import ModelConfig, ReWALanguageModel
from rewa_llm.pruning import GlobalMagnitudePruner, sparsity_report


def _torch_load(path: str | Path) -> Any:
    """Load full training checkpoints across PyTorch default changes."""

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before the weights_only keyword.
        return torch.load(path, map_location="cpu")


def _strip_wrapper_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = dict(state_dict)
    for prefix in ("module.", "_orig_mod."):
        if result and all(key.startswith(prefix) for key in result):
            result = {key[len(prefix) :]: value for key, value in result.items()}
    return result


def _as_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return vars(value)
    return None


def load_checkpoint_model(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[ReWALanguageModel, ModelConfig, dict[str, Any]]:
    """Load the standard training checkpoint, with a few legacy aliases."""

    checkpoint = _torch_load(checkpoint_path)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a mapping")

    state_dict: Any = None
    for key in ("model", "model_state_dict", "state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            state_dict = candidate
            break
    if state_dict is None and checkpoint and all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        state_dict = checkpoint
    if state_dict is None:
        raise ValueError("checkpoint has no model/model_state_dict/state_dict")

    config_mapping = _as_mapping(checkpoint.get("model_config"))
    if config_mapping is None:
        config_mapping = _as_mapping(checkpoint.get("config"))
    if config_mapping is None:
        config_mapping = _as_mapping(checkpoint.get("train_args"))
    if config_mapping is None:
        raise ValueError("checkpoint has no model_config")
    if "model_config" in config_mapping:
        nested = _as_mapping(config_mapping["model_config"])
        if nested is not None:
            config_mapping = nested

    allowed = {field.name for field in fields(ModelConfig)}
    config_values = {key: value for key, value in config_mapping.items() if key in allowed}
    config = ModelConfig(**config_values)
    model = ReWALanguageModel(config)
    model.load_state_dict(_strip_wrapper_prefixes(state_dict), strict=True)
    model.to(device)
    model.eval()
    return model, config, checkpoint


def _autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda" or dtype_name == "float32":
        return nullcontext()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.inference_mode()
def evaluate_validation(
    model: ReWALanguageModel,
    dataset: TokenDataset,
    *,
    batch_size: int,
    seq_len: int,
    eval_iters: int,
    seed: int,
    device: torch.device,
    dtype_name: str,
) -> tuple[float, float]:
    """Evaluate fixed validation batches and return mean loss and perplexity."""

    if eval_iters <= 0:
        raise ValueError("eval_iters must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    total_loss = 0.0
    model.eval()
    for _ in range(eval_iters):
        tokens, targets = dataset.batch(batch_size, seq_len, device, generator)
        with _autocast_context(device, dtype_name):
            _, loss = model(tokens, targets)
        if loss is None:
            raise RuntimeError("model did not return validation loss")
        loss_value = float(loss.detach().float().item())
        if not math.isfinite(loss_value):
            return loss_value, math.inf
        total_loss += loss_value
    mean_loss = total_loss / eval_iters
    perplexity = math.exp(mean_loss) if mean_loss < 709.0 else math.inf
    return mean_loss, perplexity


def _unique_sparsities(values: Iterable[float], include_dense: bool = True) -> list[float]:
    result: list[float] = []
    candidates = ([0.0] if include_dense else []) + [float(value) for value in values]
    for value in candidates:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid sparsity {value}; expected a value in [0, 1]")
        if value not in result:
            result.append(value)
    return result


def evaluate_sparsities(
    model: ReWALanguageModel,
    dataset: TokenDataset,
    sparsities: Iterable[float],
    *,
    batch_size: int,
    seq_len: int,
    eval_iters: int,
    seed: int,
    device: torch.device,
    dtype_name: str,
) -> list[dict[str, Any]]:
    """Evaluate each ratio from the same unpruned snapshot."""

    pruner = GlobalMagnitudePruner(model)
    results: list[dict[str, Any]] = []
    try:
        for target_sparsity in _unique_sparsities(sparsities):
            with pruner.pruned(target_sparsity) as selection:
                stats = sparsity_report(model)
                validation_loss, perplexity = evaluate_validation(
                    model,
                    dataset,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    eval_iters=eval_iters,
                    seed=seed,
                    device=device,
                    dtype_name=dtype_name,
                )
                results.append(
                    {
                        "target_sparsity": target_sparsity,
                        "selected_count": selection.selected_count,
                        "total_eligible": selection.total_eligible,
                        "selection_sparsity": selection.realized_selection_sparsity,
                        "validation_loss": validation_loss,
                        "perplexity": perplexity,
                        **stats,
                    }
                )
    finally:
        # This also protects callers if validation raises outside the context.
        pruner.restore()
    return results


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return vars(value)
    return str(value)


def write_results(
    payload: dict[str, Any], json_path: str | Path, csv_path: str | Path,
    layer_csv_path: str | Path | None = None,
) -> None:
    """Write a lossless JSON report plus summary and per-layer CSV tables."""

    json_path = Path(json_path)
    csv_path = Path(csv_path)
    if layer_csv_path is None:
        layer_csv_path = csv_path.with_name(f"{csv_path.stem}.layers{csv_path.suffix}")
    layer_csv_path = Path(layer_csv_path)
    for path in (json_path, csv_path, layer_csv_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )

    summary_fields = [
        "target_sparsity",
        "selected_count",
        "total_eligible",
        "selection_sparsity",
        "eligible_zeros",
        "eligible_nonzeros",
        "eligible_sparsity",
        "whole_model_zeros",
        "whole_model_nonzeros",
        "whole_model_sparsity",
        "attention_sparsity",
        "mlp_sparsity",
        "validation_loss",
        "perplexity",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for result in payload["results"]:
            writer.writerow(
                {
                    "target_sparsity": result["target_sparsity"],
                    "selected_count": result["selected_count"],
                    "total_eligible": result["total_eligible"],
                    "selection_sparsity": result["selection_sparsity"],
                    "eligible_zeros": result["eligible"]["zeros"],
                    "eligible_nonzeros": result["eligible"]["nonzeros"],
                    "eligible_sparsity": result["eligible"]["sparsity"],
                    "whole_model_zeros": result["whole_model"]["zeros"],
                    "whole_model_nonzeros": result["whole_model"]["nonzeros"],
                    "whole_model_sparsity": result["whole_model"]["sparsity"],
                    "attention_sparsity": result["groups"]["attention"]["sparsity"],
                    "mlp_sparsity": result["groups"]["mlp"]["sparsity"],
                    "validation_loss": result["validation_loss"],
                    "perplexity": result["perplexity"],
                }
            )

    layer_fields = [
        "target_sparsity", "name", "zeros", "nonzeros", "numel", "sparsity"
    ]
    with layer_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=layer_fields)
        writer.writeheader()
        for result in payload["results"]:
            for layer in result["layers"]:
                writer.writerow({"target_sparsity": result["target_sparsity"], **layer})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--sparsities", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
        help="target eligible sparsities; the dense 0.0 point is added automatically",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--eval-iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float16"
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--layer-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype_name = args.dtype if device.type == "cuda" else "float32"

    model, config, checkpoint = load_checkpoint_model(args.checkpoint, device)
    dataset = TokenDataset(args.data_dir, "validation")
    if dataset.vocab_size != config.vocab_size:
        raise ValueError(
            f"dataset vocab_size={dataset.vocab_size} does not match model "
            f"vocab_size={config.vocab_size}"
        )
    seq_len = config.max_seq_len if args.seq_len is None else args.seq_len
    if not 1 <= seq_len <= config.max_seq_len:
        raise ValueError(f"seq_len must be in [1, {config.max_seq_len}]")

    results = evaluate_sparsities(
        model,
        dataset,
        args.sparsities,
        batch_size=args.batch_size,
        seq_len=seq_len,
        eval_iters=args.eval_iters,
        seed=args.seed,
        device=device,
        dtype_name=dtype_name,
    )

    output_json = args.output_json or args.checkpoint.parent / "global_pruning.json"
    output_csv = args.output_csv or args.checkpoint.parent / "global_pruning.csv"
    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "model_config": config.to_dict(),
        "checkpoint_iter_num": checkpoint.get("iter_num"),
        "checkpoint_best_val_loss": checkpoint.get("best_val_loss"),
        "evaluation": {
            "batch_size": args.batch_size,
            "seq_len": seq_len,
            "eval_iters": args.eval_iters,
            "seed": args.seed,
            "device": str(device),
            "dtype": dtype_name,
            "rule": "global absolute-magnitude ranking over attention/MLP matrices",
        },
        "results": results,
    }
    write_results(payload, output_json, output_csv, args.layer_csv)

    for result in results:
        print(
            f"target={result['target_sparsity']:.2%} "
            f"actual={result['eligible']['sparsity']:.2%} "
            f"loss={result['validation_loss']:.6f} "
            f"ppl={result['perplexity']:.4f}"
        )
    print(f"wrote {output_json}")
    print(f"wrote {output_csv}")


if __name__ == "__main__":
    main()
