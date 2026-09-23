from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from rewa_llm.data import TokenDataset
from rewa_llm.model import ModelConfig, ReWALanguageModel, is_rewa_parameter
from scripts.prepare_tinystories import encode_split, iter_texts, sha256_file
from unittest import mock

from train import (
    iterations_for_budget,
    parameter_groups,
    parse_args,
    resolve_device,
    validate_args,
)


class TokenDatasetTests(unittest.TestCase):
    def test_uint16_memmap_and_seeded_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "metadata.json").write_text(
                json.dumps({"vocab_size": 128}), encoding="utf-8"
            )
            values = np.arange(100, dtype=np.uint16)
            values.tofile(root / "train.bin")

            dataset = TokenDataset(root, "train")
            self.assertEqual(dataset.vocab_size, 128)
            first_generator = torch.Generator().manual_seed(7)
            second_generator = torch.Generator().manual_seed(7)
            x1, y1 = dataset.batch(4, 8, torch.device("cpu"), first_generator)
            x2, y2 = dataset.batch(4, 8, torch.device("cpu"), second_generator)
            torch.testing.assert_close(x1, x2)
            torch.testing.assert_close(y1, y2)
            torch.testing.assert_close(x1[:, 1:], y1[:, :-1])
            self.assertEqual(x1.dtype, torch.int64)
            del dataset
            gc.collect()

    def test_rejects_too_short_stream(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "metadata.json").write_text(
                json.dumps({"vocab_size": 8}), encoding="utf-8"
            )
            np.arange(5, dtype=np.uint16).tofile(root / "validation.bin")
            dataset = TokenDataset(root, "validation")
            with self.assertRaisesRegex(ValueError, "shorter than one sequence"):
                dataset.batch(
                    1, 4, torch.device("cpu"), torch.Generator().manual_seed(0)
                )
            del dataset
            gc.collect()


class PreparationHelpersTests(unittest.TestCase):
    def test_iter_texts_filters_empty_and_honors_limit(self) -> None:
        records = [{"text": " first "}, {"text": ""}, {"text": None}, {"text": "second"}]
        self.assertEqual(list(iter_texts(records, "text", 2)), ["first", "second"])

    def test_encode_split_writes_uint16_and_checksum(self) -> None:
        class Encoding:
            def __init__(self, ids: list[int]):
                self.ids = ids

        class FakeTokenizer:
            @staticmethod
            def token_to_id(token: str) -> int | None:
                return 7 if token == "<|endoftext|>" else None

            @staticmethod
            def encode(text: str, add_special_tokens: bool = False) -> Encoding:
                del add_special_tokens
                return Encoding([len(part) for part in text.split()])

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.bin"
            counts = encode_split(FakeTokenizer(), ["aa b", "cccc"], path, chunk_docs=1)
            written = np.fromfile(path, dtype=np.uint16)
            np.testing.assert_array_equal(written, np.asarray([2, 1, 7, 4, 7]))
            self.assertEqual(counts, {"documents": 2, "tokens": 5})
            self.assertEqual(len(sha256_file(path)), 64)


class ModelTests(unittest.TestCase):
    def test_cuda_device_gets_explicit_default_index(self) -> None:
        with mock.patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(resolve_device("cuda"), torch.device("cuda", 0))
            self.assertEqual(resolve_device("auto"), torch.device("cuda", 0))

    def test_default_model_is_approximately_14m_and_tied(self) -> None:
        model = ReWALanguageModel(ModelConfig())
        self.assertGreater(model.num_parameters(), 13_000_000)
        self.assertLess(model.num_parameters(), 15_000_000)
        self.assertIs(model.lm_head.weight, model.tok_embeddings.weight)
        eligible_names = [
            name
            for name, parameter in model.named_parameters()
            if is_rewa_parameter(name, parameter)
        ]
        self.assertEqual(len(eligible_names), 6 * 7)
        self.assertFalse(any("tok_embeddings" in name for name in eligible_names))
        self.assertFalse(any("norm" in name for name in eligible_names))

    def test_small_model_forward_and_loss(self) -> None:
        config = ModelConfig(
            vocab_size=64,
            max_seq_len=16,
            n_layer=2,
            n_head=2,
            n_embd=32,
            intermediate_size=64,
        )
        model = ReWALanguageModel(config)
        tokens = torch.randint(0, config.vocab_size, (2, 8))
        logits, loss = model(tokens, tokens)
        self.assertEqual(logits.shape, (2, 8, config.vocab_size))
        self.assertIsNotNone(loss)
        assert loss is not None
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.blocks[0].attn.q_proj.weight.grad)

    def test_parameter_partition_is_method_invariant(self) -> None:
        config = ModelConfig(
            vocab_size=32,
            max_seq_len=8,
            n_layer=1,
            n_head=2,
            n_embd=16,
            intermediate_size=32,
        )
        model = ReWALanguageModel(config)
        partitions = {}
        for method in ("dense", "l1", "rewa"):
            groups, eligible, names = parameter_groups(
                model,
                method,
                weight_decay=0.1,
                rewa_weight_decay=1e-3,
                rewa_k=3.0,
                rewa_m=2.0,
                rewa_eps=1e-8,
            )
            partitions[method] = [[id(parameter) for parameter in group["params"]] for group in groups]
            self.assertEqual(len(eligible), 7)
            self.assertEqual(len(names), 7)
            self.assertEqual(groups[0]["rewa"], method == "rewa")
            self.assertEqual(groups[1]["weight_decay"], 0.1)
            self.assertEqual(groups[2]["weight_decay"], 0.0)
        self.assertEqual(partitions["dense"], partitions["l1"])
        self.assertEqual(partitions["dense"], partitions["rewa"])

    def test_token_budget_rounds_up(self) -> None:
        args = argparse.Namespace(
            max_iters=None,
            train_tokens=20_000_000,
            batch_size=8,
            seq_len=256,
            gradient_accumulation_steps=16,
        )
        self.assertEqual(iterations_for_budget(args), 611)
        args.max_iters = 3
        self.assertEqual(iterations_for_budget(args), 3)


class TrainingArgumentTests(unittest.TestCase):
    @staticmethod
    def parse_training_args(method: str = "rewa") -> argparse.Namespace:
        argv = [
            "train.py",
            "--data-dir",
            "data",
            "--out-dir",
            "outputs/test",
            "--method",
            method,
        ]
        with mock.patch("sys.argv", argv):
            return parse_args()

    def test_rewa_cli_defaults_match_canonical_adamw_recipe(self) -> None:
        args = self.parse_training_args()

        self.assertEqual(args.rewa_k, 9.0)
        self.assertEqual(args.rewa_m, 2.0)
        self.assertEqual(args.rewa_eps, 0.0)
        self.assertEqual(args.rewa_weight_decay, 1e-4)
        validate_args(args)

    def test_rewa_validation_enforces_theory_regime(self) -> None:
        args = self.parse_training_args()
        invalid_pairs = (
            (1.0, 0.0),
            (3.0, -1.0),
            (3.0, 2.0),
            (float("nan"), 0.0),
            (float("inf"), 0.0),
        )

        for k, m in invalid_pairs:
            with self.subTest(k=k, m=m):
                args.rewa_k = k
                args.rewa_m = m
                with self.assertRaisesRegex(
                    ValueError, r"K > 1 and 0 <= M < K - 1"
                ):
                    validate_args(args)

    def test_non_rewa_method_ignores_rewa_theory_parameters(self) -> None:
        args = self.parse_training_args(method="dense")
        args.rewa_k = 1.0
        args.rewa_m = -1.0

        validate_args(args)

    def test_rewa_eps_and_decay_remain_non_negative(self) -> None:
        for field in ("rewa_eps", "rewa_weight_decay"):
            with self.subTest(field=field):
                args = self.parse_training_args()
                setattr(args, field, -1.0)
                with self.assertRaisesRegex(ValueError, "must be non-negative"):
                    validate_args(args)


if __name__ == "__main__":
    unittest.main()
