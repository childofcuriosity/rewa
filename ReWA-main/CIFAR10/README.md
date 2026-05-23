# ReWA on CIFAR10

## Usage

We just follow Spred implementation, so you can check [here](https://github.com/zihao-wang/spred/tree/master/STR%20sparse) for installation. Thanks for their great work.

We provide some shell scripts (`run_ell_9_cifar10.sh` and `run_ell_cifar10.sh`). You can directly run these scripts after installation.

 
Key arguments:
| Argument | Description |
|---|---|
| `--ell_order` | Reparameterization order K (odd integer; K=9 recommended) |
| `--ell_t` | Adaptive learning rate parameter M (0 ≤ M < K−1) |
| `--ell_eps` | Stabilizer ε (0 for simple datasets, >0 for complex tasks) |
| `--ell_base_optimizer` | Base optimizer: `SGD` (Algorithm 1) or `AdamW` (Algorithm 2) |
| `--weight-decay` | Weight decay λ |

Recommended settings (CIFAR-10 / ResNet-18): `K=9, M=2, ε=0, λ=1e-4`.

Baselines: `run_L1_cifar10.sh`, `run_spred_cifar10.sh`, `run_STR_cifar10.sh`.