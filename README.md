# ReWA: Reparameterization, Weight Decay, and Adaptive Learning Rate

Code and experiment assets for the ICML 2026 paper **“Theoretical Analysis of
Sparse Optimization with Reparameterization, Weight Decay, and Adaptive
Learning Rate.”**

ReWA combines a product reparameterization, weight decay, and a
coordinate-wise adaptive learning rate. This repository contains four
experiment tracks used to study the method: synthetic linear tasks,
CIFAR-10/100 image classification, ImageNet training, and a decoder-only
Transformer study.

## Repository layout

```text
.
├── ReWA-main/
│   ├── linear/                  # synthetic linear experiments
│   └── CIFAR10/                 # CIFAR-10/100 and legacy sparse baselines
├── ffcv-results-backup/         # ImageNet implementation based on FFCV
├── docs/
│   └── source-inventory.md      # provenance and consolidation notes
├── rewa-llm/                    # Transformer and global-pruning experiments
└── README.md
```

The four tracks retain their original experiment-specific environments. Start
with one track rather than installing all dependencies into a single Python
environment.

## Quick start

### Synthetic linear task

```bash
cd ReWA-main/linear
conda env create -f environment.yaml
conda activate ReWA
python main.py --task linear --batch_size 25 --epochs 1000 \
  --optimizer ell --lr_scheduler cosine --dim 10000 --dataset_len 2000
```

See [`ReWA-main/linear/README.md`](ReWA-main/linear/README.md) for the main
hyperparameters.

### CIFAR-10 / CIFAR-100

```bash
cd ReWA-main/CIFAR10
pip install -r requirements.txt
bash run_ell_cifar10.sh       # ReWA with SGD
bash run_ell_cifar10_adam.sh  # ReWA with AdamW
```

The CIFAR implementation builds on the sparse-training code released with
[Spred](https://github.com/zihao-wang/spred/tree/master/STR%20sparse). The
provided scripts cover ReWA and the L1, Spred, and STR baselines. See
[`ReWA-main/CIFAR10/README.md`](ReWA-main/CIFAR10/README.md).

### ImageNet

```bash
cd ffcv-results-backup
pip install -r requirements.txt
```

ImageNet training uses [FFCV](https://github.com/libffcv/ffcv/). Dataset
serialization and training commands are documented in
[`ffcv-results-backup/README.md`](ffcv-results-backup/README.md); evaluation
scripts are described in
[`ffcv-results-backup/by_eval/README.md`](ffcv-results-backup/by_eval/README.md).

## Main ReWA parameters

| Parameter | Meaning |
|---|---|
| `K` / `--ell_order` | Reparameterization order; integer and non-integer values are supported in the CIFAR implementation |
| `M` / `--ell_t` | Exponent used by the adaptive learning-rate rule |
| `epsilon` / `--ell_eps` | Numerical stabilizer |
| `--weight-decay` or `--kappa` | Weight-decay strength |
| `--ell_base_optimizer` | Base optimizer for the CIFAR implementation: `SGD` or `AdamW` |

## Results and generated files

Small CSV files needed to inspect reported runs are kept with the corresponding
experiment. Large datasets, checkpoints, run directories, caches, and local
tracking logs are excluded by `.gitignore`. Historical full snapshots remain
outside this repository and are described in
[`docs/source-inventory.md`](docs/source-inventory.md).

## Reproducibility notes

- The linear environment targets Python 3.7, PyTorch 1.11, and CUDA 11.3.
- The CIFAR requirements record the older Spred-compatible stack (PyTorch 1.3
  and torchvision 0.4.1).
- The ImageNet track has its own FFCV-oriented requirements.
- Dataset paths in YAML configuration files are machine-specific defaults;
  override them for the target system.

## License

The bundled FFCV-derived ImageNet code includes its upstream license in
`ffcv-results-backup/LICENSE`. A repository-wide license has not yet been
declared.

## LLM extension: ReWA on a decoder-only Transformer

The `rewa-llm/` track extends ReWA-AdamW to a 14M-parameter decoder-only
Transformer trained on TinyStories. It provides deterministic data
preparation, Dense/L1/ReWA training, restart-safe hyperparameter screening,
and exact global pruning over attention and MLP weight matrices.

The local high-learning-rate study found that the canonical `K=9, M=2`
geometry needs a substantially larger peak learning rate than ordinary AdamW.
With peak LR `0.006` and 10M training tokens, its validation PPL changes from
`10.98` before pruning to `18.29` at 70% global sparsity and `43.47` at 80%.
The frozen best-unpruned Dense curve, trained for 20M tokens, reaches
`229.43` and `131020.73` at the same sparsities.

![Global-pruning comparison for the ReWA LLM extension](rewa-llm/artifacts/stage1-rewa-high-lr/rewa-llm-pruning-comparison.png)

**Figure: Large-step ReWA with `K=9, M=2` preserves the strongest
accuracy–sparsity trade-off among the tested ReWA geometries.** Panel (a)
compares the 10M-token ReWA checkpoint with frozen 20M-token Dense/L1 curves
and shows the separate strongly regularized L1 operating point. Panel (b)
holds the ReWA training budget, seed, schedule, and peak LR fixed while varying
`(K, M)`. Perplexity uses a logarithmic scale; all measurements are single-seed
validation results.

### Reproduce the LLM result

```bash
cd rewa-llm
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/prepare_tinystories.py \
  --output-dir data/tinystories --train-docs 100000 --validation-docs 5000
.venv/bin/python scripts/run_rewa_tuning.py \
  --data-dir data/tinystories \
  --screen-configs 3:0 3:1 9:2 \
  --learning-rates 0.006 0.012 0.024 0.036 \
  --batch-size 8 --gradient-accumulation-steps 16 \
  --pruning-batch-size 16 --top-k 3 --stop-after confirm \
  --device cuda --dtype float16
.venv/bin/python scripts/plot_high_lr_results.py
```

The Windows equivalents use `.venv\Scripts\python.exe`. The runner skips
compatible completed runs and known failed configurations by default; pass
`--retry-failed` to repeat a failed point.

Detailed protocol and result artifacts:

- [`rewa-llm/README.md`](rewa-llm/README.md)
- [`CONFIRM_RESULTS.md`](rewa-llm/artifacts/stage1-rewa-high-lr/CONFIRM_RESULTS.md)
- [`ten-million-global-pruning.csv`](rewa-llm/artifacts/stage1-rewa-high-lr/ten-million-global-pruning.csv)
- [`tuning-manifest.csv`](rewa-llm/artifacts/stage1-rewa-high-lr/tuning-manifest.csv)
- [`plot_high_lr_results.py`](rewa-llm/scripts/plot_high_lr_results.py)

### Extended repository layout

```text
.
├── README.md               # existing project overview, extended in place
├── ReWA-main/              # core algorithm implementation
│   ├── CIFAR10/            # CIFAR-10/100 image classification
│   └── linear/             # synthetic linear regression
├── ffcv-results-backup/    # ImageNet training with FFCV
└── rewa-llm/               # Transformer training and global-pruning study
```
