from __future__ import annotations

import csv
import gc
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from evaluate_global_pruning import (
    evaluate_sparsities,
    load_checkpoint_model,
    write_results,
)
from rewa_llm.data import TokenDataset
from rewa_llm.model import ModelConfig, ReWALanguageModel
from rewa_llm.pruning import (
    GlobalMagnitudePruner,
    eligible_named_parameters,
    sparsity_report,
)


class _ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.q_proj = nn.Linear(2, 2, bias=False)
        self.mlp = nn.Module()
        self.mlp.up_proj = nn.Linear(2, 2, bias=False)


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = _ToyBlock()
        self.embedding = nn.Embedding(3, 2)
        self.norm = nn.LayerNorm(2)
        with torch.no_grad():
            self.block.attn.q_proj.weight.copy_(
                torch.tensor([[0.4, 0.1], [0.2, 0.3]])
            )
            self.block.mlp.up_proj.weight.copy_(
                torch.tensor([[0.8, 0.7], [0.6, 0.5]])
            )
            self.embedding.weight.fill_(2.0)
            self.norm.weight.fill_(1.0)
            self.norm.bias.fill_(1.0)


class GlobalPruningTests(unittest.TestCase):
    def test_only_attention_and_mlp_matrices_are_eligible(self):
        model = _ToyModel()
        names = [name for name, _ in eligible_named_parameters(model)]
        self.assertEqual(
            names,
            ["block.attn.q_proj.weight", "block.mlp.up_proj.weight"],
        )

    def test_global_ranking_is_not_layerwise(self):
        model = _ToyModel()
        pruner = GlobalMagnitudePruner(model)
        selection = pruner.apply(0.25)

        self.assertEqual(selection.selected_count, 2)
        self.assertEqual(
            torch.count_nonzero(model.block.attn.q_proj.weight == 0).item(), 2
        )
        self.assertEqual(
            torch.count_nonzero(model.block.mlp.up_proj.weight == 0).item(), 0
        )

    def test_ties_select_exact_count_and_context_restores(self):
        model = _ToyModel()
        with torch.no_grad():
            model.block.attn.q_proj.weight.fill_(1.0)
            model.block.mlp.up_proj.weight.fill_(1.0)
        originals = {
            name: parameter.detach().clone()
            for name, parameter in eligible_named_parameters(model)
        }
        pruner = GlobalMagnitudePruner(model)

        with pruner.pruned(0.375) as selection:
            report = sparsity_report(model)
            self.assertEqual(selection.selected_count, 3)
            self.assertEqual(report["eligible"]["zeros"], 3)
            # Stable tie breaking selects only the first three flattened values,
            # rather than thresholding all eight equal-magnitude entries.
            self.assertEqual(
                torch.count_nonzero(model.block.attn.q_proj.weight == 0).item(), 3
            )

        for name, parameter in eligible_named_parameters(model):
            torch.testing.assert_close(parameter, originals[name])
        self.assertIsNone(pruner.active_selection)

    def test_each_apply_starts_from_original_snapshot(self):
        model = _ToyModel()
        originals = {
            name: parameter.detach().clone()
            for name, parameter in eligible_named_parameters(model)
        }
        pruner = GlobalMagnitudePruner(model)
        pruner.apply(0.25)
        self.assertEqual(sparsity_report(model)["eligible"]["zeros"], 2)
        pruner.apply(0.5)
        self.assertEqual(sparsity_report(model)["eligible"]["zeros"], 4)
        pruner.restore()
        for name, parameter in eligible_named_parameters(model):
            torch.testing.assert_close(parameter, originals[name])

    def test_sparsity_report_includes_per_layer_and_whole_model(self):
        model = _ToyModel()
        pruner = GlobalMagnitudePruner(model)
        pruner.apply(0.25)
        report = sparsity_report(model)

        self.assertEqual(report["eligible"]["numel"], 8)
        self.assertEqual(report["eligible"]["zeros"], 2)
        self.assertAlmostEqual(report["eligible"]["sparsity"], 0.25)
        self.assertEqual(len(report["layers"]), 2)
        self.assertAlmostEqual(report["groups"]["attention"]["sparsity"], 0.5)
        self.assertAlmostEqual(report["groups"]["mlp"]["sparsity"], 0.0)
        self.assertGreater(report["whole_model"]["numel"], 8)
        self.assertEqual(report["whole_model"]["zeros"], 2)

    def test_invalid_sparsity_is_rejected(self):
        pruner = GlobalMagnitudePruner(_ToyModel())
        for value in (-0.1, 1.1, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pruner.apply(value)


class EvaluatorTests(unittest.TestCase):
    def _small_model(self) -> ReWALanguageModel:
        torch.manual_seed(7)
        return ReWALanguageModel(
            ModelConfig(
                vocab_size=16,
                max_seq_len=8,
                n_layer=1,
                n_head=2,
                n_embd=8,
                intermediate_size=16,
            )
        )

    def test_standard_training_checkpoint_loads(self):
        model = self._small_model()
        config = model.config
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "checkpoint.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": {},
                    "model_config": config.to_dict(),
                    "train_args": {"method": "dense"},
                    "iter_num": 12,
                    "best_val_loss": 2.5,
                },
                path,
            )
            loaded, loaded_config, checkpoint = load_checkpoint_model(
                path, torch.device("cpu")
            )

        self.assertEqual(loaded_config, config)
        self.assertEqual(checkpoint["iter_num"], 12)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key], value)

    def test_evaluation_ratios_are_independent_and_restore_model(self):
        model = self._small_model()
        originals = {
            name: parameter.detach().clone()
            for name, parameter in eligible_named_parameters(model)
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            (data_dir / "metadata.json").write_text(
                json.dumps({"vocab_size": 16}), encoding="utf-8"
            )
            (np.arange(128, dtype=np.uint16) % 16).tofile(
                data_dir / "validation.bin"
            )
            dataset = TokenDataset(data_dir, "validation")
            results = evaluate_sparsities(
                model,
                dataset,
                [0.25, 0.5],
                batch_size=2,
                seq_len=4,
                eval_iters=2,
                seed=9,
                device=torch.device("cpu"),
                dtype_name="float32",
            )
            # NumPy memmaps hold an open Windows file handle until collected.
            del dataset
            gc.collect()

        # Dense is automatically prepended; every requested ratio is exact.
        self.assertEqual([result["target_sparsity"] for result in results], [0.0, 0.25, 0.5])
        for result in results:
            self.assertEqual(
                result["selected_count"],
                int(result["total_eligible"] * result["target_sparsity"] + 0.5),
            )
            self.assertTrue(np.isfinite(result["perplexity"]))
        for name, parameter in eligible_named_parameters(model):
            torch.testing.assert_close(parameter, originals[name])

    def test_json_and_csv_outputs_include_summary_and_layers(self):
        payload = {
            "results": [
                {
                    "target_sparsity": 0.5,
                    "selected_count": 2,
                    "total_eligible": 4,
                    "selection_sparsity": 0.5,
                    "eligible": {"zeros": 2, "nonzeros": 2, "numel": 4, "sparsity": 0.5},
                    "whole_model": {"zeros": 2, "nonzeros": 6, "numel": 8, "sparsity": 0.25},
                    "groups": {
                        "attention": {"zeros": 2, "nonzeros": 2, "numel": 4, "sparsity": 0.5},
                        "mlp": {"zeros": 0, "nonzeros": 0, "numel": 0, "sparsity": 0.0},
                    },
                    "validation_loss": 2.0,
                    "perplexity": 7.389,
                    "layers": [
                        {"name": "block.attn.q_proj.weight", "zeros": 2, "nonzeros": 2, "numel": 4, "sparsity": 0.5}
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            json_path = output_dir / "results.json"
            csv_path = output_dir / "results.csv"
            write_results(payload, json_path, csv_path)

            self.assertEqual(json.loads(json_path.read_text())["results"][0]["selected_count"], 2)
            with csv_path.open(newline="", encoding="utf-8") as handle:
                summary_rows = list(csv.DictReader(handle))
            with (output_dir / "results.layers.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                layer_rows = list(csv.DictReader(handle))

        self.assertEqual(summary_rows[0]["eligible_sparsity"], "0.5")
        self.assertEqual(layer_rows[0]["name"], "block.attn.q_proj.weight")


if __name__ == "__main__":
    unittest.main()
