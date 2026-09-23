from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_pilot import (
    RunSpec,
    assert_compatible_existing_run,
    execute_run,
    reconcile_active_regularized_runs,
    reconcile_selected_learning_rate,
    rewa_config,
)
from scripts.summarize_stage1 import (
    collect_layer_rows,
    collect_pruning_rows,
    collect_training_rows,
    flatten_candidate_grid_envelope,
    make_envelope_plot,
    select_best_runs,
    select_candidate_grid_envelope,
    selected_curves,
    write_results,
)


class PilotConfigurationTests(unittest.TestCase):
    def test_theory_conforming_rewa_pairs(self):
        self.assertEqual(rewa_config("3:0"), (3.0, 0.0))
        self.assertEqual(rewa_config("9:2"), (9.0, 2.0))

    def test_rejects_boundary_m_equal_k_minus_one(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            rewa_config("3:2")

    def test_rejects_malformed_pair(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            rewa_config("3,0")


class ExistingRunCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace(
            eval_seed=10_000,
            device="cuda",
            dtype="float16",
            batch_size=8,
            eval_batch_size=None,
            gradient_accumulation_steps=16,
            seq_len=256,
            n_layer=6,
            n_head=6,
            n_embd=384,
            intermediate_size=1024,
            train_tokens=20_000_000,
            max_iters=None,
            eval_iters=50,
            eval_interval=100,
            warmup_iters=50,
            weight_decay=0.1,
            pruning_eval_iters=100,
            force_evaluation=False,
        )

    def saved_args(self, spec: RunSpec, data_dir: Path) -> dict[str, object]:
        return {
            "data_dir": str(data_dir),
            "method": spec.method,
            "seed": spec.seed,
            "eval_seed": self.args.eval_seed,
            "device": self.args.device,
            "dtype": self.args.dtype,
            "learning_rate": spec.learning_rate,
            "batch_size": self.args.batch_size,
            "eval_batch_size": self.args.eval_batch_size,
            "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
            "seq_len": self.args.seq_len,
            "n_layer": self.args.n_layer,
            "n_head": self.args.n_head,
            "n_embd": self.args.n_embd,
            "intermediate_size": self.args.intermediate_size,
            "train_tokens": self.args.train_tokens,
            "max_iters": self.args.max_iters,
            "eval_iters": self.args.eval_iters,
            "eval_interval": self.args.eval_interval,
            "warmup_iters": self.args.warmup_iters,
            "weight_decay": self.args.weight_decay,
            # Current train.py defaults. These fields are irrelevant to Dense/L1.
            "l1_alpha": spec.l1_alpha,
            "rewa_k": 9.0,
            "rewa_m": 2.0,
            "rewa_weight_decay": 1e-4,
            "rewa_eps": 0.0,
        }

    def write_config(self, path: Path, saved: dict[str, object]) -> None:
        path.write_text(json.dumps({"train_args": saved}), encoding="utf-8")

    def test_dense_and_l1_ignore_irrelevant_rewa_defaults_on_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            for spec in (
                RunSpec(order=0, method="dense", seed=0, learning_rate=6e-4),
                RunSpec(
                    order=1,
                    method="l1",
                    seed=0,
                    learning_rate=6e-4,
                    l1_alpha=1e-6,
                ),
            ):
                with self.subTest(method=spec.method):
                    path = root / f"{spec.method}.json"
                    self.write_config(path, self.saved_args(spec, data_dir))
                    assert_compatible_existing_run(path, spec, self.args, data_dir)

    def test_completed_dense_and_l1_runs_skip_with_current_train_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            output_root = root / "outputs"
            artifact_dir = root / "artifacts"
            manifest_path = artifact_dir / "run-manifest.csv"
            rows: dict[str, dict[str, object]] = {}
            for spec in (
                RunSpec(order=0, method="dense", seed=0, learning_rate=6e-4),
                RunSpec(
                    order=1,
                    method="l1",
                    seed=0,
                    learning_rate=6e-4,
                    l1_alpha=1e-6,
                ),
            ):
                with self.subTest(method=spec.method):
                    run_dir = output_root / spec.name
                    run_dir.mkdir(parents=True)
                    self.write_config(
                        run_dir / "run_config.json", self.saved_args(spec, data_dir)
                    )
                    (run_dir / "best.pt").touch()
                    (run_dir / "latest.pt").touch()
                    (run_dir / "metrics.jsonl").write_text(
                        json.dumps({"iter": 611, "val_loss": 2.0}) + "\n",
                        encoding="utf-8",
                    )
                    with (
                        patch("scripts.run_pilot.run_command") as run_command,
                        patch(
                            "scripts.run_pilot.evaluation_is_complete", return_value=True
                        ),
                    ):
                        self.assertTrue(
                            execute_run(
                                spec,
                                self.args,
                                data_dir,
                                output_root,
                                artifact_dir,
                                manifest_path,
                                rows,
                                "test-commit",
                            )
                        )
                    run_command.assert_not_called()
                    self.assertEqual(rows[spec.name]["status"], "completed")

    def test_relevant_mismatches_are_still_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            cases = [
                (
                    RunSpec(order=0, method="dense", seed=0, learning_rate=6e-4),
                    "learning_rate",
                    3e-4,
                ),
                (
                    RunSpec(
                        order=1,
                        method="l1",
                        seed=0,
                        learning_rate=6e-4,
                        l1_alpha=1e-6,
                    ),
                    "l1_alpha",
                    1e-5,
                ),
                (
                    RunSpec(
                        order=2,
                        method="rewa",
                        seed=0,
                        learning_rate=6e-4,
                        rewa_k=9.0,
                        rewa_m=2.0,
                        rewa_weight_decay=0.1,
                        rewa_eps=0.0,
                    ),
                    "rewa_weight_decay",
                    1.0,
                ),
            ]
            for index, (spec, field, mismatched_value) in enumerate(cases):
                with self.subTest(method=spec.method, field=field):
                    saved = self.saved_args(spec, data_dir)
                    if spec.method == "rewa":
                        saved.update(
                            rewa_k=spec.rewa_k,
                            rewa_m=spec.rewa_m,
                            rewa_weight_decay=spec.rewa_weight_decay,
                            rewa_eps=spec.rewa_eps,
                        )
                    saved[field] = mismatched_value
                    path = root / f"mismatch-{index}.json"
                    self.write_config(path, saved)
                    with self.assertRaisesRegex(RuntimeError, field):
                        assert_compatible_existing_run(path, spec, self.args, data_dir)


class ManifestReconciliationTests(unittest.TestCase):
    def test_marks_only_non_dense_rows_with_stale_learning_rate(self):
        rows = {
            "dense-old": {
                "run_name": "dense-old",
                "method": "dense",
                "status": "completed",
                "learning_rate": "0.0003",
                "checkpoint": "old-dense.pt",
            },
            "l1-stale": {
                "run_name": "l1-stale",
                "method": "l1",
                "status": "completed",
                "learning_rate": "0.0003",
                "checkpoint": "stale-l1.pt",
            },
            "rewa-active": {
                "run_name": "rewa-active",
                "method": "rewa",
                "status": "completed",
                "learning_rate": "0.0006",
                "checkpoint": "active-rewa.pt",
            },
        }

        changed = reconcile_selected_learning_rate(rows, 6e-4)

        self.assertEqual(changed, ["l1-stale"])
        self.assertEqual(rows["l1-stale"]["status"], "superseded")
        self.assertEqual(rows["l1-stale"]["superseded_from_status"], "completed")
        self.assertIn("status=completed", rows["l1-stale"]["status_detail"])
        self.assertIn("selected dense learning_rate=0.0006", rows["l1-stale"]["status_detail"])
        self.assertEqual(rows["l1-stale"]["checkpoint"], "stale-l1.pt")
        self.assertEqual(rows["dense-old"]["status"], "completed")
        self.assertEqual(rows["rewa-active"]["status"], "completed")

        self.assertEqual(reconcile_selected_learning_rate(rows, 6e-4), [])
        self.assertIn("status=completed", rows["l1-stale"]["status_detail"])

    def test_marks_same_lr_rows_outside_current_grid(self):
        rows = {
            "l1-active": {
                "run_name": "l1-active",
                "method": "l1",
                "status": "completed",
                "learning_rate": "0.0006",
            },
            "rewa-old-decay": {
                "run_name": "rewa-old-decay",
                "method": "rewa",
                "status": "training",
                "learning_rate": "0.0006",
                "rewa_weight_decay": "0.001",
                "checkpoint": "partial.pt",
            },
            "dense-extra": {
                "run_name": "dense-extra",
                "method": "dense",
                "status": "completed",
                "learning_rate": "0.0003",
            },
        }

        changed = reconcile_active_regularized_runs(rows, {"l1-active"})

        self.assertEqual(changed, ["rewa-old-decay"])
        stale = rows["rewa-old-decay"]
        self.assertEqual(stale["status"], "superseded")
        self.assertEqual(stale["superseded_from_status"], "training")
        self.assertIn("current frozen L1/ReWA candidate grid", stale["status_detail"])
        self.assertEqual(stale["checkpoint"], "partial.pt")
        self.assertEqual(rows["l1-active"]["status"], "completed")
        self.assertEqual(rows["dense-extra"]["status"], "completed")
        self.assertEqual(
            reconcile_active_regularized_runs(rows, {"l1-active"}), []
        )

    def test_summary_excludes_and_reports_superseded_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "run-manifest.csv"
            manifest_path.touch()

            def artifact_set(prefix: str) -> tuple[Path, Path, Path]:
                pruning = root / f"{prefix}-pruning.csv"
                with pruning.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=["target_sparsity", "validation_loss", "perplexity"],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "target_sparsity": "0.0",
                            "validation_loss": "1.0",
                            "perplexity": "2.718",
                        }
                    )
                layers = root / f"{prefix}-layers.csv"
                with layers.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=["target_sparsity", "name"]
                    )
                    writer.writeheader()
                    writer.writerow({"target_sparsity": "0.0", "name": "layer"})
                metrics = root / f"{prefix}-metrics.jsonl"
                metrics.write_text(json.dumps({"iter": 1, "train_loss": 1.0}) + "\n")
                return pruning, layers, metrics

            active_files = artifact_set("active")
            stale_files = artifact_set("stale")
            manifest = [
                {
                    "run_name": "dense-active",
                    "method": "dense",
                    "status": "completed",
                    "seed": "0",
                    "learning_rate": "0.0006",
                    "dense_lr_selected": "yes",
                    "pruning_csv": str(active_files[0]),
                    "layer_csv": str(active_files[1]),
                    "metrics_jsonl": str(active_files[2]),
                },
                {
                    "run_name": "l1-stale",
                    "method": "l1",
                    "status": "superseded",
                    "seed": "0",
                    "learning_rate": "0.0003",
                    "status_detail": "superseded for test",
                    "pruning_csv": str(stale_files[0]),
                    "layer_csv": str(stale_files[1]),
                    "metrics_jsonl": str(stale_files[2]),
                },
            ]

            pruning, warnings = collect_pruning_rows(manifest, manifest_path)
            layers = collect_layer_rows(manifest, manifest_path)
            training = collect_training_rows(manifest, manifest_path)
            self.assertFalse(warnings)
            self.assertEqual({row["run_name"] for row in pruning}, {"dense-active"})
            self.assertEqual({row["run_name"] for row in layers}, {"dense-active"})
            self.assertEqual({row["run_name"] for row in training}, {"dense-active"})

            results = root / "RESULTS.md"
            write_results(
                results,
                manifest,
                pruning,
                {},
                {},
                warnings,
                root / "global-pruning.csv",
                root / "ppl-vs-sparsity.png",
            )
            report = results.read_text(encoding="utf-8")
            self.assertIn("## Superseded configurations", report)
            self.assertIn("`l1-stale`: superseded for test", report)
            self.assertIn("excluded from every aggregate table", report)


class CandidateGridEnvelopeTests(unittest.TestCase):
    def test_results_do_not_treat_collapsed_selected_wins_as_scale_up_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def points(
                run_name: str,
                method: str,
                values: list[tuple[float, float, float]],
                **config: str,
            ) -> list[dict[str, str]]:
                return [
                    {
                        "run_name": run_name,
                        "method": method,
                        "seed": "0",
                        "learning_rate": "0.0006",
                        "target_sparsity": str(target),
                        "validation_loss": str(loss),
                        "perplexity": str(perplexity),
                        **config,
                    }
                    for target, loss, perplexity in values
                ]

            rows = [
                *points(
                    "dense",
                    "dense",
                    [(0.0, 2.30, 10.0), (0.9, 7.60, 2000.0), (0.95, 8.29, 4000.0)],
                ),
                *points(
                    "l1-base",
                    "l1",
                    [(0.0, 2.31, 10.1), (0.9, 7.50, 1800.0), (0.95, 8.16, 3500.0)],
                    l1_alpha="1e-7",
                ),
                *points(
                    "l1-sparse",
                    "l1",
                    [(0.0, 2.48, 12.0), (0.9, 4.61, 100.0), (0.95, 7.60, 2000.0)],
                    l1_alpha="1e-5",
                ),
                *points(
                    "rewa",
                    "rewa",
                    [(0.0, 2.71, 15.0), (0.9, 7.31, 1500.0), (0.95, 7.09, 1200.0)],
                    rewa_k="9",
                    rewa_m="2",
                    rewa_weight_decay="0.0001",
                ),
            ]
            manifest = [
                {
                    "run_name": run_name,
                    "method": method,
                    "status": "completed",
                    "seed": "0",
                    "learning_rate": "0.0006",
                }
                for run_name, method in (
                    ("dense", "dense"),
                    ("l1-base", "l1"),
                    ("l1-sparse", "l1"),
                    ("rewa", "rewa"),
                )
            ]
            selected = select_best_runs(rows)
            curves = selected_curves(rows, selected)
            envelope = select_candidate_grid_envelope(rows)
            results = root / "RESULTS.md"
            write_results(
                results,
                manifest,
                rows,
                selected,
                curves,
                [],
                root / "global-pruning.csv",
                root / "ppl-vs-sparsity.png",
                envelope=envelope,
            )

            report = results.read_text(encoding="utf-8")
            self.assertIn("unpruned PPL is 50.00% higher", report)
            self.assertIn("selected dense run at 2 level(s)", report)
            self.assertIn("selected L1 run at 2 level(s)", report)
            self.assertIn("both dense and L1 at 1 of 2", report)
            self.assertIn("joint-win target(s) 90%, 95%", report)
            self.assertIn("all three methods have PPL above 1000", report)
            self.assertIn("not treated as practical pruning advantages", report)
            self.assertIn("frozen stage-one scale-up rule is not met", report)

    def test_envelope_switches_configs_by_sparsity_and_excludes_superseded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "run-manifest.csv"
            manifest_path.touch()

            def pruning_csv(name: str, points: list[tuple[float, float, float]]) -> Path:
                path = root / f"{name}.csv"
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=[
                            "target_sparsity",
                            "validation_loss",
                            "perplexity",
                        ],
                    )
                    writer.writeheader()
                    for sparsity, loss, perplexity in points:
                        writer.writerow(
                            {
                                "target_sparsity": sparsity,
                                "validation_loss": loss,
                                "perplexity": perplexity,
                            }
                        )
                return path

            base_path = pruning_csv(
                "l1-base",
                [(0.0, 2.0, 7.4), (0.7, 4.0, 54.6), (0.8, 5.0, 148.4)],
            )
            sparse_path = pruning_csv(
                "l1-sparse",
                [(0.0, 2.5, 12.2), (0.7, 2.2, 9.0), (0.8, 2.4, 11.0)],
            )
            stale_path = pruning_csv(
                "l1-stale",
                [(0.0, 0.1, 1.1), (0.7, 0.1, 1.1), (0.8, 0.1, 1.1)],
            )
            manifest = [
                {
                    "run_name": "l1-base",
                    "method": "l1",
                    "status": "completed",
                    "seed": "0",
                    "learning_rate": "0.0006",
                    "l1_alpha": "0.000001",
                    "pruning_csv": str(base_path),
                },
                {
                    "run_name": "l1-sparse",
                    "method": "l1",
                    "status": "completed",
                    "seed": "0",
                    "learning_rate": "0.0006",
                    "l1_alpha": "0.00001",
                    "pruning_csv": str(sparse_path),
                },
                {
                    "run_name": "l1-stale",
                    "method": "l1",
                    "status": "superseded",
                    "seed": "0",
                    "learning_rate": "0.0003",
                    "l1_alpha": "0.001",
                    "pruning_csv": str(stale_path),
                },
            ]

            pruning, warnings = collect_pruning_rows(manifest, manifest_path)
            self.assertFalse(warnings)
            self.assertEqual(
                {row["run_name"] for row in pruning}, {"l1-base", "l1-sparse"}
            )

            selected = select_best_runs(pruning)
            self.assertEqual(selected["l1"]["run_name"], "l1-base")

            envelope = select_candidate_grid_envelope(pruning)
            chosen = {
                float(row["target_sparsity"]): row["run_name"]
                for row in envelope["l1"]
            }
            self.assertEqual(
                chosen, {0.0: "l1-base", 0.7: "l1-sparse", 0.8: "l1-sparse"}
            )

            flattened = flatten_candidate_grid_envelope(envelope)
            self.assertEqual(len(flattened), 3)
            self.assertEqual(flattened[1]["selected_run_name"], "l1-sparse")
            self.assertIn("alpha=1e-05", flattened[1]["selected_configuration"])

            envelope_plot = root / "candidate-grid-oracle-envelope.png"
            make_envelope_plot(envelope_plot, envelope)
            self.assertGreater(envelope_plot.stat().st_size, 0)

            results = root / "RESULTS.md"
            write_results(
                results,
                manifest,
                pruning,
                selected,
                selected_curves(pruning, selected),
                warnings,
                root / "global-pruning.csv",
                root / "ppl-vs-sparsity.png",
                envelope=envelope,
                envelope_csv_path=root / "candidate-grid-oracle-envelope.csv",
                envelope_plot_path=envelope_plot,
            )
            report = results.read_text(encoding="utf-8")
            self.assertIn("Candidate-grid oracle/Pareto envelope (optimistic)", report)
            self.assertIn("Selection and reporting use the same validation set", report)
            self.assertIn("may switch runs between sparsity levels", report)
            self.assertIn("`l1-base`", report)
            self.assertIn("`l1-sparse`", report)
            self.assertIn("alpha=1e-05", report)
            self.assertIn("candidate-grid-oracle-envelope.png", report)


if __name__ == "__main__":
    unittest.main()
