#!/usr/bin/env python3
"""Prepare a reproducible TinyStories token corpus for stage-one experiments.

The script trains one 8,192-token byte-level BPE on the training split, appends
an end-of-document token to every story, and streams the encoded ids to uint16
binary files.  Metadata records corpus sizes and SHA-256 checksums so a run can
be tied to the exact tokenizer and token stream it used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Iterator

import numpy as np


VOCAB_SIZE = 8192
UNK_TOKEN = "<|unk|>"
EOS_TOKEN = "<|endoftext|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="roneneldan/TinyStories")
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face dataset revision. Pin a commit for archival runs.",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--max-train-docs", type=positive_int, default=None)
    parser.add_argument("--max-validation-docs", type=positive_int, default=None)
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream examples instead of first materializing Arrow shards.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def iter_texts(
    records: Iterable[dict[str, Any]], text_column: str, max_docs: int | None
) -> Iterator[str]:
    """Yield non-empty documents in source order, up to ``max_docs``."""
    yielded = 0
    for record in records:
        if max_docs is not None and yielded >= max_docs:
            break
        if text_column not in record:
            raise KeyError(f"dataset record has no {text_column!r} column")
        text = record[text_column]
        if text is None:
            continue
        text = str(text).strip()
        if not text:
            continue
        yield text
        yielded += 1


def load_split(
    dataset_name: str, split: str, revision: str, streaming: bool
) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - exercised on the server
        raise SystemExit("Install dependencies with: pip install -r requirements.txt") from error
    return load_dataset(
        dataset_name,
        split=split,
        revision=revision,
        streaming=streaming,
    )


def train_tokenizer(
    texts: Iterable[str], output_path: Path, vocab_size: int = VOCAB_SIZE
) -> Any:
    if vocab_size != VOCAB_SIZE:
        raise ValueError(f"stage one uses a fixed {VOCAB_SIZE}-token vocabulary")
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    except ImportError as error:  # pragma: no cover - exercised on the server
        raise SystemExit("Install dependencies with: pip install -r requirements.txt") from error

    # Byte-level BPE avoids an out-of-vocabulary path for arbitrary story text.
    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[UNK_TOKEN, EOS_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    if tokenizer.get_vocab_size() != vocab_size:
        raise RuntimeError(
            f"tokenizer produced {tokenizer.get_vocab_size()} tokens, expected {vocab_size}; "
            "use more training documents"
        )
    tokenizer.save(str(output_path))
    return tokenizer


def encode_split(
    tokenizer: Any,
    texts: Iterable[str],
    output_path: Path,
    chunk_docs: int = 1024,
) -> dict[str, int]:
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    if eos_id is None:
        raise RuntimeError(f"tokenizer is missing {EOS_TOKEN}")

    documents = 0
    tokens = 0
    pending: list[int] = []
    with output_path.open("wb") as handle:
        for text in texts:
            pending.extend(tokenizer.encode(text, add_special_tokens=False).ids)
            pending.append(eos_id)
            documents += 1
            if documents % chunk_docs == 0:
                encoded = np.asarray(pending, dtype="<u2")
                encoded.tofile(handle)
                tokens += int(encoded.size)
                pending.clear()
        if pending:
            encoded = np.asarray(pending, dtype="<u2")
            encoded.tofile(handle)
            tokens += int(encoded.size)

    if documents == 0:
        raise RuntimeError(f"no non-empty documents were written to {output_path}")
    return {"documents": documents, "tokens": tokens}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(dataset: Any) -> str | None:
    value = getattr(dataset, "_fingerprint", None)
    return str(value) if value is not None else None


def ensure_output_directory(path: Path, overwrite: bool) -> None:
    expected = [path / "tokenizer.json", path / "train.bin", path / "validation.bin"]
    if any(candidate.exists() for candidate in expected) and not overwrite:
        raise FileExistsError(
            f"{path} already contains prepared data; pass --overwrite to replace it"
        )
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    if args.vocab_size != VOCAB_SIZE:
        raise SystemExit(f"--vocab-size must be {VOCAB_SIZE} for stage-one comparability")
    ensure_output_directory(args.output_dir, args.overwrite)

    # Avoid tokenizer worker scheduling becoming another uncontrolled variable.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tokenizer_path = args.output_dir / "tokenizer.json"

    print(f"Loading {args.dataset}:{args.train_split} for BPE training", flush=True)
    tokenizer_train_data = load_split(
        args.dataset, args.train_split, args.revision, args.streaming
    )
    tokenizer = train_tokenizer(
        iter_texts(tokenizer_train_data, args.text_column, args.max_train_docs),
        tokenizer_path,
        args.vocab_size,
    )

    split_specs = {
        "train": (args.train_split, args.max_train_docs),
        "validation": (args.validation_split, args.max_validation_docs),
    }
    split_metadata: dict[str, dict[str, Any]] = {}
    for output_name, (source_split, max_docs) in split_specs.items():
        print(f"Encoding {args.dataset}:{source_split} -> {output_name}.bin", flush=True)
        dataset = load_split(args.dataset, source_split, args.revision, args.streaming)
        binary_path = args.output_dir / f"{output_name}.bin"
        counts = encode_split(
            tokenizer,
            iter_texts(dataset, args.text_column, max_docs),
            binary_path,
        )
        split_metadata[output_name] = {
            "source_split": source_split,
            "max_documents": max_docs,
            "dataset_fingerprint": dataset_fingerprint(dataset),
            "path": binary_path.name,
            "dtype": "uint16-le",
            **counts,
            "bytes": binary_path.stat().st_size,
            "sha256": sha256_file(binary_path),
        }

    metadata = {
        "format_version": 1,
        "dataset": {
            "name": args.dataset,
            "revision": args.revision,
            "text_column": args.text_column,
            "streaming": args.streaming,
        },
        "vocab_size": tokenizer.get_vocab_size(),
        "tokenizer": {
            "type": "byte-level BPE",
            "path": tokenizer_path.name,
            "vocab_size": tokenizer.get_vocab_size(),
            "unk_token": UNK_TOKEN,
            "unk_token_id": tokenizer.token_to_id(UNK_TOKEN),
            "eos_token": EOS_TOKEN,
            "eos_token_id": tokenizer.token_to_id(EOS_TOKEN),
            "sha256": sha256_file(tokenizer_path),
        },
        "splits": split_metadata,
        "producer": {
            "script": Path(__file__).name,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    print(f"Prepared dataset at {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
