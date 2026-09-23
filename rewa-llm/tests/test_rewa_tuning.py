import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.run_rewa_tuning import (
    DEFAULT_KM,
    DEFAULT_LRS,
    Phase,
    TuneSpec,
    execute_spec,
    read_metrics,
    screen_specs,
    select_screen_rows,
    select_geometry_rows,
    select_finalists,
    variant_specs,
)


def arguments(**overrides):
    values = {
        "seed": 0,
        "screen_configs": list(DEFAULT_KM),
        "learning_rates": list(DEFAULT_LRS),
        "screen_eps": [0.0],
        "screen_weight_decays": [1e-4],
        "variant_eps": [0.0, 1e-6, 1e-3],
        "variant_weight_decays": [1e-4, 0.1, 1.0],
        "variant_decay_reference_lr": 3e-3,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class TuningGridTests(unittest.TestCase):
    def test_failed_run_is_not_retried_without_explicit_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = arguments(
                retry_failed=False,
                output_root=root / "outputs",
                artifact_dir=root / "artifacts",
                seq_len=256,
                batch_size=8,
                gradient_accumulation_steps=16,
            )
            spec = TuneSpec(9, 2, 0.024)
            phase = Phase("screen", 3_000_000, 20, 46, 8, 20, (0.0, 0.7, 0.8))
            rows = {
                f"screen/{spec.name}": {
                    "run_id": f"screen/{spec.name}",
                    "status": "failed",
                    "error": "FloatingPointError",
                }
            }

            with mock.patch("scripts.run_rewa_tuning.run_command") as run_command:
                completed = execute_spec(
                    0, spec, phase, args, root / "data", args.output_root,
                    args.artifact_dir, root / "manifest.csv", rows, "commit",
                )

            self.assertFalse(completed)
            self.assertEqual(rows[f"screen/{spec.name}"]["status"], "failed")
            run_command.assert_not_called()

    def test_metrics_accumulate_skipped_optimizer_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {"iter": 10, "val_loss": 3.0, "skipped_optimizer_steps": 2},
                        {"iter": 20, "val_loss": 2.5, "skipped_optimizer_steps": 1},
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            completed, best, skipped = read_metrics(path)

            self.assertEqual(completed, 20)
            self.assertEqual(best, 2.5)
            self.assertEqual(skipped, 3)

    def test_default_screen_has_28_unique_theory_admissible_specs(self):
        specs = screen_specs(arguments())

        self.assertEqual(len(specs), 28)
        self.assertEqual(len({spec.name for spec in specs}), 28)
        self.assertEqual(tuple(DEFAULT_LRS), (6e-3, 1.2e-2, 2.4e-2, 3.6e-2))
        for spec in specs:
            self.assertGreater(spec.k, 1)
            self.assertGreaterEqual(spec.m, 0)
            self.assertLess(spec.m, spec.k - 1)
            self.assertEqual(spec.eps, 0)
            self.assertEqual(spec.weight_decay, 1e-4)

    def test_screen_can_expand_epsilon_and_decay_cartesian_grid(self):
        specs = screen_specs(
            arguments(
                screen_configs=[(3.0, 0.0)],
                learning_rates=[0.006],
                screen_eps=[0.0, 1e-4],
                screen_weight_decays=[1e-2, 1e-1],
            )
        )

        self.assertEqual(len(specs), 4)
        self.assertEqual(
            {(spec.eps, spec.weight_decay) for spec in specs},
            {(0.0, 1e-2), (0.0, 1e-1), (1e-4, 1e-2), (1e-4, 1e-1)},
        )

    def test_variant_grid_filters_positive_epsilon_outside_configuration_b(self):
        low_m = TuneSpec(3, 0, 6e-4)
        boundary = TuneSpec(9, 2, 6e-4)

        variants = variant_specs([low_m, boundary], arguments())
        low_m_variants = [spec for spec in variants if spec.k == 3]
        boundary_variants = [spec for spec in variants if spec.k == 9]

        self.assertTrue(any(spec.eps > 0 for spec in low_m_variants))
        self.assertFalse(any(spec.eps > 0 for spec in boundary_variants))
        self.assertFalse(any(spec.eps > 0 and spec.weight_decay >= 1 for spec in variants))
        self.assertNotIn(low_m.name, {spec.name for spec in variants})
        self.assertNotIn(boundary.name, {spec.name for spec in variants})

    def test_variant_decay_is_normalized_by_learning_rate(self):
        base = TuneSpec(3, 0, 1.2e-2)

        variants = variant_specs(
            [base],
            arguments(variant_eps=[0.0], variant_weight_decays=[0.1]),
        )

        self.assertEqual(len(variants), 1)
        self.assertAlmostEqual(variants[0].weight_decay, 0.025)
        self.assertAlmostEqual(
            variants[0].learning_rate * variants[0].weight_decay,
            3e-3 * 0.1,
        )

    def test_invalid_k_m_is_rejected(self):
        for k, m in ((1, 0), (3, -1), (3, 2)):
            with self.subTest(k=k, m=m):
                with self.assertRaises(ValueError):
                    TuneSpec(k, m, 6e-4)

    def test_geometry_selection_guarantees_positive_epsilon_branch(self):
        ranked = [
            {"run_id": "a", "rewa_m": "2"},
            {"run_id": "b", "rewa_m": "4"},
            {"run_id": "c", "rewa_m": "1"},
            {"run_id": "d", "rewa_m": "0"},
        ]

        selected = select_geometry_rows(ranked)

        self.assertEqual([row["run_id"] for row in selected], ["a", "b", "c"])

    def test_geometry_selection_does_not_add_redundant_low_m_branch(self):
        ranked = [
            {"run_id": "a", "rewa_m": "1"},
            {"run_id": "b", "rewa_m": "2"},
            {"run_id": "c", "rewa_m": "0"},
        ]

        selected = select_geometry_rows(ranked)

        self.assertEqual([row["run_id"] for row in selected], ["a", "b"])

    def test_geometry_selection_keeps_guarded_high_sparsity_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ranked = []
            for name, unpruned, pruned70, pruned80, k, m in (
                ("accuracy", 2.30, 3.1, 5.0, 3, 1),
                ("second_accuracy", 2.31, 3.2, 4.8, 3, 0),
                ("sparse", 2.32, 2.7, 3.5, 9, 2),
            ):
                path = root / f"{name}.json"
                path.write_text(
                    json.dumps(
                        {
                            "results": [
                                {"target_sparsity": 0.0, "validation_loss": unpruned,
                                 "perplexity": 2.718281828 ** unpruned},
                                {"target_sparsity": 0.7, "validation_loss": pruned70,
                                 "perplexity": 2.718281828 ** pruned70},
                                {"target_sparsity": 0.8, "validation_loss": pruned80,
                                 "perplexity": 2.718281828 ** pruned80},
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                ranked.append(
                    {"run_id": name, "rewa_k": str(k), "rewa_m": str(m),
                     "pruning_json": str(path)}
                )

            selected = select_geometry_rows(ranked)

            self.assertEqual([row["run_id"] for row in selected], ["accuracy", "sparse"])

    def test_screen_selection_retains_canonical_k9_m2_geometry(self):
        ranked = [
            {"run_id": "a", "rewa_k": "3", "rewa_m": "0"},
            {"run_id": "b", "rewa_k": "3", "rewa_m": "1"},
            {"run_id": "c", "rewa_k": "5", "rewa_m": "0"},
            {"run_id": "d", "rewa_k": "5", "rewa_m": "2"},
            {"run_id": "canonical", "rewa_k": "9", "rewa_m": "2"},
        ]

        selected = select_screen_rows(ranked, 4)

        self.assertEqual([row["run_id"] for row in selected], ["a", "canonical", "b", "c"])

    def test_screen_selection_uses_high_sparsity_curve_with_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ranked = []
            for name, unpruned, pruned70, pruned80, k, m in (
                ("accuracy", 2.0, 7.0, 9.0, 3, 0),
                ("sparse", 2.1, 3.0, 4.0, 3, 1),
                ("unguarded", 2.5, 1.0, 1.0, 5, 0),
                ("canonical", 2.2, 5.0, 6.0, 9, 2),
            ):
                path = root / f"{name}.json"
                path.write_text(
                    json.dumps(
                        {
                            "results": [
                                {"target_sparsity": 0.0, "validation_loss": unpruned,
                                 "perplexity": 2.718281828 ** unpruned},
                                {"target_sparsity": 0.7, "validation_loss": pruned70,
                                 "perplexity": 2.718281828 ** pruned70},
                                {"target_sparsity": 0.8, "validation_loss": pruned80,
                                 "perplexity": 2.718281828 ** pruned80},
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                ranked.append(
                    {"run_id": name, "rewa_k": str(k), "rewa_m": str(m),
                     "pruning_json": str(path)}
                )

            selected = select_screen_rows(ranked, 3)

            self.assertEqual(
                [row["run_id"] for row in selected],
                ["accuracy", "sparse", "canonical"],
            )


class FinalistSelectionTests(unittest.TestCase):
    @staticmethod
    def write_pruning(
        path: Path, unpruned: float, pruned70: float, pruned80: float | None = None
    ) -> None:
        pruned80 = pruned70 if pruned80 is None else pruned80
        path.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "target_sparsity": 0.0,
                            "validation_loss": unpruned,
                            "perplexity": 2.718281828 ** unpruned,
                        },
                        {
                            "target_sparsity": 0.7,
                            "validation_loss": pruned70,
                            "perplexity": 2.718281828 ** pruned70,
                        },
                        {
                            "target_sparsity": 0.8,
                            "validation_loss": pruned80,
                            "perplexity": 2.718281828 ** pruned80,
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

    def test_selects_best_unpruned_and_guarded_sparse_representative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = []
            # A is the accuracy representative. B is within the 10% PPL guard
            # and has the best mean 70%/80%-pruned loss. C is sparse but fails the guard.
            for name, unpruned, pruned70, pruned80 in (
                ("a", 2.30, 3.0, 4.0),
                ("b", 2.35, 2.7, 3.0),
                ("c", 2.60, 2.0, 2.0),
            ):
                path = root / f"{name}.json"
                self.write_pruning(path, unpruned, pruned70, pruned80)
                candidates.append(
                    {
                        "run_id": f"confirm/{name}",
                        "pruning_json": str(path),
                    }
                )

            selected = select_finalists(candidates)

            self.assertEqual(selected[0][0]["run_id"], "confirm/a")
            self.assertEqual(selected[1][0]["run_id"], "confirm/b")
            self.assertIn("70%/80%-pruned", selected[1][1])

    def test_falls_back_to_second_best_accuracy_when_guard_has_no_alternate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = []
            for name, unpruned, pruned in (("a", 2.0, 3.0), ("b", 2.5, 2.0)):
                path = root / f"{name}.json"
                self.write_pruning(path, unpruned, pruned)
                candidates.append({"run_id": f"variant/{name}", "pruning_json": str(path)})

            selected = select_finalists(candidates)

            self.assertEqual(selected[0][0]["run_id"], "variant/a")
            self.assertEqual(selected[1][0]["run_id"], "variant/b")
            self.assertIn("no alternate passed", selected[1][1])


if __name__ == "__main__":
    unittest.main()
