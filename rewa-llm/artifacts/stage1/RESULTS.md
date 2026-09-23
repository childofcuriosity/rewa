# Stage-one single-seed pilot results

This exploratory TinyStories pilot contains 8 completed configuration(s) at seed(s) 0. The curves compare exact global absolute-magnitude pruning over eligible attention and MLP matrices.

Generated: `2026-09-22T16:06:26+00:00`.

The dense sweep selected `lr=0.0006` from training-time `best_val_loss`; every L1 and ReWA candidate used that learning rate.

## Best unpruned configuration per method

| Method | Selected run | Configuration | Unpruned PPL |
| --- | --- | --- | --- |
| Dense AdamW | `dense-lr6e-4-seed0` | lr=0.0006 | 10.4127 |
| AdamW + L1 | `l1-a1e-7-lr6e-4-seed0` | lr=0.0006, alpha=1e-07 | 10.4060 |
| ReWA-AdamW | `rewa-k9-m2-wd1e-4-lr6e-4-seed0` | lr=0.0006, K=9, M=2, wd=0.0001 | 15.1702 |

Selection uses the common unpruned (`target_sparsity = 0`) validation evaluation. The same validation protocol then supplies every point on each selected curve.

## Selected global-pruning curves

| Eligible sparsity | Dense PPL | L1 PPL | ReWA PPL |
| --- | --- | --- | --- |
| 0% | 10.4127 | 10.4060 | 15.1702 |
| 30% | 10.8273 | 10.7998 | 17.2610 |
| 50% | 14.7890 | 14.5116 | 50.8597 |
| 70% | 229.4289 | 191.1771 | 209390.5013 |
| 80% | 131020.7280 | 103805.6152 | 3750272.3775 |
| 90% | 17672317.5762 | 16830847.9075 | 14534381.6618 |
| 95% | 39723229.0679 | 39692026.4756 | 16264643.2610 |

## Candidate-grid oracle/Pareto envelope (optimistic)

The candidate grid's lower envelope records the lowest observed validation perplexity at each tested sparsity within each method. Every point below names the run and configuration that produced it.

Selection and reporting use the same validation set, and a method's envelope may switch runs between sparsity levels. The envelope is therefore an optimistic pilot diagnostic, not a held-out estimate or a single-run pruning curve.

| Eligible sparsity | Method | Lowest PPL | Chosen run | Chosen configuration |
| --- | --- | --- | --- | --- |
| 0% | Dense AdamW | 10.4127 | `dense-lr6e-4-seed0` | lr=0.0006 |
| 0% | AdamW + L1 | 10.4060 | `l1-a1e-7-lr6e-4-seed0` | lr=0.0006, alpha=1e-07 |
| 0% | ReWA-AdamW | 15.1702 | `rewa-k9-m2-wd1e-4-lr6e-4-seed0` | lr=0.0006, K=9, M=2, wd=0.0001 |
| 30% | Dense AdamW | 10.8273 | `dense-lr6e-4-seed0` | lr=0.0006 |
| 30% | AdamW + L1 | 10.6127 | `l1-a1e-6-lr6e-4-seed0` | lr=0.0006, alpha=1e-06 |
| 30% | ReWA-AdamW | 17.2610 | `rewa-k9-m2-wd1e-4-lr6e-4-seed0` | lr=0.0006, K=9, M=2, wd=0.0001 |
| 50% | Dense AdamW | 14.7890 | `dense-lr6e-4-seed0` | lr=0.0006 |
| 50% | AdamW + L1 | 12.4302 | `l1-a1e-6-lr6e-4-seed0` | lr=0.0006, alpha=1e-06 |
| 50% | ReWA-AdamW | 29.1469 | `rewa-k9-m2-wd1-lr6e-4-seed0` | lr=0.0006, K=9, M=2, wd=1 |
| 70% | Dense AdamW | 229.4289 | `dense-lr6e-4-seed0` | lr=0.0006 |
| 70% | AdamW + L1 | 13.9731 | `l1-a1e-5-lr6e-4-seed0` | lr=0.0006, alpha=1e-05 |
| 70% | ReWA-AdamW | 49749.1150 | `rewa-k9-m2-wd1-lr6e-4-seed0` | lr=0.0006, K=9, M=2, wd=1 |
| 80% | Dense AdamW | 131020.7280 | `dense-lr6e-4-seed0` | lr=0.0006 |
| 80% | AdamW + L1 | 14.6939 | `l1-a1e-5-lr6e-4-seed0` | lr=0.0006, alpha=1e-05 |
| 80% | ReWA-AdamW | 213550.3535 | `rewa-k9-m2-wd1-lr6e-4-seed0` | lr=0.0006, K=9, M=2, wd=1 |
| 90% | Dense AdamW | 1474045.1114 | `dense-lr3e-4-seed0` | lr=0.0003 |
| 90% | AdamW + L1 | 127.7724 | `l1-a1e-5-lr6e-4-seed0` | lr=0.0006, alpha=1e-05 |
| 90% | ReWA-AdamW | 328464.4166 | `rewa-k9-m2-wd1-lr6e-4-seed0` | lr=0.0006, K=9, M=2, wd=1 |
| 95% | Dense AdamW | 1543652.2543 | `dense-lr3e-4-seed0` | lr=0.0003 |
| 95% | AdamW + L1 | 756699.3303 | `l1-a1e-5-lr6e-4-seed0` | lr=0.0006, alpha=1e-05 |
| 95% | ReWA-AdamW | 345515.5277 | `rewa-k9-m2-wd1-lr6e-4-seed0` | lr=0.0006, K=9, M=2, wd=1 |

## Pilot observation

The selected ReWA run's unpruned PPL is 45.69% higher than the selected dense run (15.1702 versus 10.4127). For this deterministic pilot summary, 'close' means no more than 5% above dense.

Across 6 shared nonzero sparsity levels, the selected ReWA run has lower perplexity than the selected dense run at 2 level(s) and lower perplexity than the selected L1 run at 2 level(s).

On the optimistic candidate-grid envelope, ReWA is lower than both dense and L1 at 1 of 6 shared nonzero sparsity level(s).

At selected-curve joint-win target(s) 90%, 95%, all three methods have PPL above 1000. These collapsed-regime orderings are not treated as practical pruning advantages.

The frozen stage-one scale-up rule is not met: ReWA is not close to dense before pruning and the candidate-grid envelope improves on both baselines at no more than one shared nonzero sparsity level. The current evidence therefore does not support scale-up.

This is a single-seed hyperparameter-selection pilot. A follow-up with fresh seeds is the next measurement needed to estimate variance and confirm the observed ordering.

## Artifacts

- Full configuration-by-sparsity table: `global-pruning.csv`
- Selected-curve plot: `ppl-vs-sparsity.png`
- Candidate-grid oracle/Pareto envelope table: `candidate-grid-oracle-envelope.csv`
- Candidate-grid oracle/Pareto envelope plot: `candidate-grid-oracle-envelope.png`
- Per-layer table: `layer-sparsity.csv`
- Aggregated training log: `training-curves.csv`
- Run provenance and failures: `run-manifest.csv`

## Superseded configurations

These preserved runs were excluded from every aggregate table, selection, and curve.

- `l1-a1e-7-lr3e-4-seed0`: Superseded from status=training: run is not part of the current frozen L1/ReWA candidate grid.
- `rewa-k9-m2-wd0.001-lr6e-4-seed0`: Superseded from status=training: run is not part of the current frozen L1/ReWA candidate grid.
