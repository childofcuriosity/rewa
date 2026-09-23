# ReWA 10M-token high-learning-rate confirmation

All ReWA rows below were trained locally from scratch with seed 0, 10,027,008 realized tokens, batch size 8, gradient accumulation 16, maximum LR 0.006, and a cosine schedule ending at zero. Each pruning ratio was applied independently to the same best checkpoint.

## ReWA geometry comparison

| K | M | Unpruned PPL | 50% PPL | 70% PPL | 80% PPL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 1 | 10.7753 | 11.6110 | 24.4703 | 379.5389 |
| 3 | 0 | 10.8740 | 11.8086 | 26.0982 | 330.6920 |
| 9 | 2 | 10.9777 | 11.6535 | 18.2948 | 43.4725 |

## Frozen 20M-token baselines

These baselines use the same model, data hashes, seed, effective batch, eligible tensors, and pruning seed. They have twice the training-token budget, so this comparison does not favor ReWA.

| Method | Unpruned PPL | 50% PPL | 70% PPL | 80% PPL |
| --- | ---: | ---: | ---: | ---: |
| Dense AdamW | 10.4127 | 14.7890 | 229.4289 | 131020.7280 |
| AdamW + L1 | 10.4060 | 14.5116 | 191.1771 | 103805.6152 |

## Evidence-based interpretation

The accuracy representative is `confirm/k3-m1-eps0-wd1e-4-lr0.006-seed0`. The sparsity representative is `confirm/k9-m2-eps0-wd1e-4-lr0.006-seed0` (best mean 70%/80%-pruned loss within 10% unpruned-PPL guard).
Its unpruned PPL is 5.43% above the frozen Dense baseline (10.9777 versus 10.4127).
At 70% sparsity, the ReWA representative has PPL 18.2948, versus 229.4289 for Dense and 191.1771 for the best-unpruned L1 curve.
At 80% sparsity, the ReWA representative has PPL 43.4725, versus 131020.7280 for Dense and 103805.6152 for the best-unpruned L1 curve.
The optimistic L1 candidate-grid comparison selects a different, strongly regularized run (`l1-a1e-5-lr6e-4-seed0`): PPL 13.9520 unpruned, 13.9731 at 70%, and 14.6939 at 80%. This is a different accuracy/sparsity operating point, not the best-unpruned L1 curve.

The result is a single-seed confirmation on the tuning validation split. It establishes a strong local signal and a reproducible configuration; fresh seeds are still required for an uncertainty estimate.

## Artifacts

- `tuning-manifest.csv`
- `screen-ranking.csv`
- `ten-million-ranking.csv`
- `ten-million-global-pruning.csv`
