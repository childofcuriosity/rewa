# ReWA stage-one hyperparameter tuning

Short-budget rankings are exploratory and were used only for successive halving. The two finalists were retrained from scratch at 20M tokens before comparison.

## Screen

| Rank | K | M | LR | eps | wd | Best validation PPL |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5 | 0 | 0.003 | 0 | 0.0001 | 35.3347 |
| 2 | 5 | 2 | 0.003 | 0 | 0.0001 | 35.3957 |
| 3 | 3 | 0 | 0.003 | 0 | 0.0001 | 35.6536 |
| 4 | 3 | 1 | 0.003 | 0 | 0.0001 | 35.8946 |
| 5 | 9 | 2 | 0.003 | 0 | 0.0001 | 36.8553 |
| 6 | 9 | 4 | 0.003 | 0 | 0.0001 | 37.1135 |
| 7 | 7 | 2 | 0.003 | 0 | 0.0001 | 38.7888 |
| 8 | 9 | 4 | 0.002 | 0 | 0.0001 | 38.8660 |
| 9 | 9 | 2 | 0.002 | 0 | 0.0001 | 39.1881 |
| 10 | 7 | 2 | 0.002 | 0 | 0.0001 | 39.5498 |

## Ten-million-token confirmation and local variants

| Rank | Run | Unpruned PPL | 50% PPL | Selected |
| ---: | --- | ---: | ---: | --- |
| 1 | `variant/k3-m0-eps1e-6-wd1e-4-lr0.003-seed0` | 11.6566 | 14.9419 | yes |
| 2 | `confirm/k3-m0-eps0-wd1e-4-lr0.003-seed0` | 11.6603 | 14.8199 | no |
| 3 | `variant/k3-m0-eps0.001-wd1e-4-lr0.003-seed0` | 11.6632 | 14.8229 | no |
| 4 | `confirm/k3-m1-eps0-wd1e-4-lr0.003-seed0` | 11.6975 | 14.8566 | no |
| 5 | `variant/k3-m1-eps1e-6-wd1e-4-lr0.003-seed0` | 11.7052 | 14.8495 | no |
| 6 | `variant/k3-m1-eps0.001-wd1e-4-lr0.003-seed0` | 11.7079 | 14.8163 | no |
| 7 | `variant/k3-m0-eps0-wd0.1-lr0.003-seed0` | 11.7079 | 14.7265 | no |
| 8 | `variant/k3-m0-eps1e-6-wd0.1-lr0.003-seed0` | 11.7080 | 14.7228 | no |
| 9 | `variant/k3-m0-eps0.001-wd0.1-lr0.003-seed0` | 11.7112 | 14.7626 | no |
| 10 | `variant/k3-m1-eps0-wd0.1-lr0.003-seed0` | 11.7622 | 14.6520 | no |
| 11 | `variant/k3-m1-eps1e-6-wd0.1-lr0.003-seed0` | 11.7623 | 14.6287 | yes |
| 12 | `variant/k3-m1-eps0.001-wd0.1-lr0.003-seed0` | 11.7632 | 14.6307 | no |
| 13 | `confirm/k5-m0-eps0-wd1e-4-lr0.003-seed0` | 11.7747 | 15.4028 | no |
| 14 | `confirm/k5-m2-eps0-wd1e-4-lr0.003-seed0` | 11.7857 | 15.3705 | no |
| 15 | `variant/k3-m0-eps0-wd1-lr0.003-seed0` | 15.1924 | 16.2555 | no |
| 16 | `variant/k3-m1-eps0-wd1-lr0.003-seed0` | 15.3553 | 16.2901 | no |

## Final 20M-token comparison

| Sparsity | Dense PPL | L1 PPL | Tuned finalist 1 PPL | Tuned finalist 2 PPL |
| ---: | ---: | ---: | ---: | ---: |
| 0% | 10.4127 | 10.4060 | 8.4599 | 8.4853 |
| 30% | 10.8273 | 10.7998 | 8.6368 | 8.6096 |
| 50% | 14.7890 | 14.5116 | 10.4703 | 9.9694 |
| 70% | 229.4289 | 191.1771 | 275.3718 | 2694.3610 |
| 80% | 131020.7280 | 103805.6152 | 2893689.5528 | 60012649.2001 |
| 90% | 17672317.5762 | 16830847.9075 | 50153750141.0775 | 85864871027.2897 |
| 95% | 39723229.0679 | 39692026.4756 | 388693435215.1984 | 258522250950.9399 |

## Interpretation

At least one tuned finalist is within 5% of Dense before pruning and tuned ReWA beats both frozen baselines at more than one non-collapsed sparsity. This is a positive single-seed tuning signal, not yet a multi-seed estimate.

All selection used the same validation split, so screening ranks and the 10M pruning guard are optimistic tuning diagnostics. Final curves remain single-seed evidence.

## Artifacts

- `tuning-manifest.csv`
- `screen-ranking.csv`
- `ten-million-ranking.csv`
- `final-global-pruning.csv`
- `rewa-tuning-vs-baselines.png`
