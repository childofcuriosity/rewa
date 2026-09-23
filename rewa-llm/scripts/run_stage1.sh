#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 {dense|l1|rewa} SEED [additional train.py arguments]" >&2
  exit 2
fi

method="$1"
seed="$2"
shift 2

case "$method" in
  dense|l1|rewa) ;;
  *)
    echo "unknown method: $method (expected dense, l1, or rewa)" >&2
    exit 2
    ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$script_dir/.." && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="$PYTHON_BIN"
elif [[ -x "$project_dir/.venv/bin/python" ]]; then
  python_bin="$project_dir/.venv/bin/python"
elif [[ -x /root/miniconda3/bin/python ]]; then
  python_bin=/root/miniconda3/bin/python
else
  python_bin=python3
fi

data_dir="${DATA_DIR:-$project_dir/data/tinystories}"
output_root="${OUTPUT_ROOT:-$project_dir/outputs}"
learning_rate="${LEARNING_RATE:-6e-4}"

case "$method" in
  dense)
    default_name="dense-lr${learning_rate}-seed${seed}"
    method_args=()
    ;;
  l1)
    l1_alpha="${L1_ALPHA:-1e-6}"
    default_name="l1-a${l1_alpha}-lr${learning_rate}-seed${seed}"
    method_args=(--l1-alpha "$l1_alpha")
    ;;
  rewa)
    rewa_k="${REWA_K:-9}"
    rewa_m="${REWA_M:-2}"
    rewa_weight_decay="${REWA_WEIGHT_DECAY:-1e-4}"
    default_name="rewa-k${rewa_k}-m${rewa_m}-wd${rewa_weight_decay}-lr${learning_rate}-seed${seed}"
    method_args=(
      --rewa-k "$rewa_k"
      --rewa-m "$rewa_m"
      --rewa-weight-decay "$rewa_weight_decay"
      --rewa-eps "${REWA_EPS:-0}"
    )
    ;;
esac

run_name="${RUN_NAME:-$default_name}"
out_dir="$output_root/$run_name"

exec "$python_bin" "$project_dir/train.py" \
  --data-dir "$data_dir" \
  --out-dir "$out_dir" \
  --method "$method" \
  --seed "$seed" \
  --device "${DEVICE:-cuda}" \
  --dtype "${DTYPE:-float16}" \
  --learning-rate "$learning_rate" \
  --train-tokens "${TRAIN_TOKENS:-20000000}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --gradient-accumulation-steps "${GRAD_ACCUM_STEPS:-16}" \
  --warmup-iters "${WARMUP_ITERS:-50}" \
  --eval-interval "${EVAL_INTERVAL:-100}" \
  --eval-iters "${EVAL_ITERS:-50}" \
  --log-interval "${LOG_INTERVAL:-10}" \
  --save-interval "${SAVE_INTERVAL:-100}" \
  "${method_args[@]}" \
  "$@"
