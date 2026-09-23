#!/usr/bin/env bash
set -euo pipefail

# Full-budget search for a new machine. Every candidate trains for the same
# token budget; no short-budget result is used to eliminate configurations.
PYTHON=${PYTHON:-.venv/bin/python}
DATA_DIR=${DATA_DIR:-data/tinystories}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/rewa-full-budget}
ARTIFACT_DIR=${ARTIFACT_DIR:-artifacts/rewa-full-budget}
L1_OUTPUT=${L1_OUTPUT:-outputs/rewa-full-budget-l1/l1-a1e-5-lr6e-4-seed0}
L1_ARTIFACT=${L1_ARTIFACT:-artifacts/rewa-full-budget-l1}
DEVICE=${DEVICE:-cuda}
DTYPE=${DTYPE:-float16}
BATCH_SIZE=${BATCH_SIZE:-32}
GRAD_ACCUM=${GRAD_ACCUM:-4}
TRAIN_TOKENS=${TRAIN_TOKENS:-100000000}
SEED=${SEED:-0}

if (( BATCH_SIZE * GRAD_ACCUM != 128 )); then
  echo "BATCH_SIZE * GRAD_ACCUM must equal the frozen effective batch 128" >&2
  exit 2
fi

mkdir -p "$L1_OUTPUT" "$L1_ARTIFACT"

if [[ ! -f "$L1_ARTIFACT/global-pruning.json" ]]; then
  resume_args=()
  if [[ -f "$L1_OUTPUT/latest.pt" ]]; then
    resume_args=(--resume "$L1_OUTPUT/latest.pt")
  fi
  "$PYTHON" train.py \
    --data-dir "$DATA_DIR" --out-dir "$L1_OUTPUT" --method l1 \
    --seed "$SEED" --eval-seed 10000 --device "$DEVICE" --dtype "$DTYPE" \
    --seq-len 256 --n-layer 6 --n-head 6 --n-embd 384 \
    --intermediate-size 1024 --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --train-tokens "$TRAIN_TOKENS" --eval-iters 50 --eval-interval 200 \
    --log-interval 10 --save-interval 200 --warmup-iters 100 \
    --learning-rate 0.0006 --min-lr-ratio 0.1 --weight-decay 0.1 \
    --l1-alpha 1e-5 "${resume_args[@]}"

  "$PYTHON" evaluate_global_pruning.py \
    --checkpoint "$L1_OUTPUT/best.pt" --data-dir "$DATA_DIR" \
    --sparsities 0 0.5 0.7 0.8 --batch-size 16 --seq-len 256 \
    --eval-iters 100 --seed 20260922 --device "$DEVICE" --dtype "$DTYPE" \
    --output-json "$L1_ARTIFACT/global-pruning.json" \
    --output-csv "$L1_ARTIFACT/global-pruning.csv" \
    --layer-csv "$L1_ARTIFACT/layer-sparsity.csv"
fi

common_rewa_args=(
  --python "$PYTHON"
  --data-dir "$DATA_DIR"
  --output-root "$OUTPUT_ROOT"
  --artifact-dir "$ARTIFACT_DIR"
  --baseline-artifact-dir artifacts/stage1
  --seed "$SEED"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --batch-size "$BATCH_SIZE"
  --gradient-accumulation-steps "$GRAD_ACCUM"
  --screen-tokens "$TRAIN_TOKENS"
  --screen-eval-iters 50
  --screen-eval-interval 200
  --screen-warmup-iters 100
  --short-pruning-eval-iters 100
  --stop-after screen
)

# Full K9/M2 epsilon x y-space-decay grid at the established stable peak LR.
# Positive epsilon with M=2 is an explicitly empirical boundary/out-of-
# Configuration-B probe; the manifest records that status rather than silently
# presenting it as theorem-conforming.
"$PYTHON" scripts/run_rewa_tuning.py "${common_rewa_args[@]}" \
  --screen-configs 9:2 --learning-rates 0.006 \
  --screen-eps 0 1e-6 1e-4 1e-3 1e-2 \
  --screen-weight-decays 1e-4 1e-2 3e-2 1e-1 3e-1 \
  --top-k 4

# Full-budget geometry controls. The K9/M2 anchor is restart-skipped because it
# is already part of the Cartesian grid above.
"$PYTHON" scripts/run_rewa_tuning.py "${common_rewa_args[@]}" \
  --screen-configs 3:0 3:1 5:0 5:2 7:2 9:2 9:4 \
  --learning-rates 0.006 --screen-eps 0 --screen-weight-decays 1e-4 \
  --top-k 4

"$PYTHON" scripts/summarize_full_budget.py \
  --manifest "$ARTIFACT_DIR/tuning-manifest.csv" \
  --l1-csv "$L1_ARTIFACT/global-pruning.csv" \
  --output-dir "$ARTIFACT_DIR"
