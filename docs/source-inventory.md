# Source inventory and consolidation record

This repository is the canonical, version-controlled project assembled from
four working sources. The inventory below records what each source contributes
and prevents historical artifacts from being mistaken for active code.

| Source | Role | Material retained in the Git project |
|---|---|---|
| `codeaftericml_github` | Canonical Git repository | Linear, CIFAR, and ImageNet experiment code plus lightweight result tables |
| `think710/CIFAR10` | Later CIFAR working copy | Non-integer `K`/`M` CLI support, corrected SGD and AdamW launch scripts, and the latest small CSV result tables |
| `codeaftericml快照` | Full historical snapshot | Kept outside Git as an archive containing raw data, checkpoints, caches, figures, logs, and plotting inputs |
| `rewa-llm` | Transformer extension | Decoder-only Transformer code, deterministic TinyStories preparation, global-pruning evaluation, tests, and lightweight result artifacts |

## Verified relationships

The CIFAR working copy matches the latest remote CIFAR tree file-for-file except
for three local changes: `args.py`, `run_ell_cifar10.sh`, and
`run_ell_cifar10_adam.sh`. The result CSV files and the non-integer-power
stability changes were already present in the remote history. This consolidation
therefore imports only those three remaining changes rather than copying the
whole directory again.

The historical `ReWA-main.tar.gz` snapshot contains 561 entries, including 123
cache files, 113 result-like files, raw CIFAR data, run directories, and an
older `RigL` tree. The historical `ffcv-results-backup.tar.gz` contains 62
entries, including plotting outputs, spreadsheets, helper scripts, and notebook
checkpoints. These archives remain useful as research records, while the Git
repository carries the compact runnable source tree.

The LLM extension retains source, tests, configuration snapshots, aggregate
tables, and lightweight figures. Prepared token streams, virtual environments,
training checkpoints, and full output directories remain outside Git.

## Canonical experiment entry points

| Track | Entry point | Environment |
|---|---|---|
| Linear | `ReWA-main/linear/main.py` | `ReWA-main/linear/environment.yaml` |
| CIFAR-10/100 | `ReWA-main/CIFAR10/main.py` | `ReWA-main/CIFAR10/requirements.txt` |
| ImageNet | `ffcv-results-backup/train_imagenet.py` | `ffcv-results-backup/requirements.txt` |
| Decoder-only Transformer | `rewa-llm/train.py` | `rewa-llm/requirements.txt` |

## Follow-up cleanup

The next structural improvement is to place the shared ReWA optimizer in one
importable module. It is currently implemented separately for the linear,
CIFAR, ImageNet, and LLM tracks, and the variants expose different arguments.
That refactor should be validated against representative runs from all four
tracks before the duplicate implementations are removed.
