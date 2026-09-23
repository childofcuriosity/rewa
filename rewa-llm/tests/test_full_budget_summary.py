import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_full_budget import collect_runs


class FullBudgetSummaryTests(unittest.TestCase):
    def test_only_complete_curves_are_ranked_by_both_sparse_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifacts" / "search"
            runs = artifact / "runs"
            runs.mkdir(parents=True)

            manifest = artifact / "manifest.csv"
            fields = (
                "run_id", "status", "seed", "rewa_k", "rewa_m",
                "learning_rate", "rewa_eps", "rewa_weight_decay",
                "train_tokens", "best_val_ppl", "pruning_json",
            )
            rows = []
            for name, status, ppls in (
                ("winner", "completed", (10.0, 10.5, 12.0, 13.0)),
                ("loser", "completed", (9.0, 11.0, 15.0, 20.0)),
                ("running", "running", (8.0, 8.0, 8.0, 8.0)),
            ):
                path = runs / f"{name}.json"
                path.write_text(
                    json.dumps(
                        {
                            "results": [
                                {"target_sparsity": target, "perplexity": ppl}
                                for target, ppl in zip((0.0, 0.5, 0.7, 0.8), ppls)
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                rows.append(
                    {
                        "run_id": name,
                        "status": status,
                        "seed": 0,
                        "rewa_k": 9,
                        "rewa_m": 2,
                        "learning_rate": 0.006,
                        "rewa_eps": 0,
                        "rewa_weight_decay": 1e-4,
                        "train_tokens": 100_000_000,
                        "best_val_ppl": ppls[0],
                        "pruning_json": str(path.relative_to(root)),
                    }
                )
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

            ranked = collect_runs(manifest, l1_ppl70=13.0, l1_ppl80=14.0)

            self.assertEqual([row["run_id"] for row in ranked], ["winner", "loser"])
            self.assertTrue(ranked[0]["beats_l1_both"])
            self.assertFalse(ranked[1]["beats_l1_both"])
            self.assertAlmostEqual(ranked[0]["worst_l1_ratio"], 13.0 / 14.0)


if __name__ == "__main__":
    unittest.main()
