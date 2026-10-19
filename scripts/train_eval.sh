#!/usr/bin/env bash
# Manually run on an existing GPU worker. No resource provisioning.
set -euo pipefail
TG_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -eq 1 && ( "$1" == --help || "$1" == -h ) ]]; then
  echo "Usage: bash scripts/train_eval.sh BASE_MODEL DATA_DIRECTORY NEW_RUN_DIRECTORY"
  echo "Train 500 updates, then evaluate the trained checkpoint and initialized Base on full validation/test."
  exit 0
fi
if [[ $# -ne 3 ]]; then
  echo "Usage: bash scripts/train_eval.sh BASE_MODEL DATA_DIRECTORY NEW_RUN_DIRECTORY" >&2
  exit 2
fi
TG_MODEL="$1"
TG_DATA="$2"
TG_RUN="$3"
[[ -f "$TG_MODEL/config.json" && -f "$TG_DATA/manifest.json" ]] || { echo "Missing model or data manifest" >&2; exit 1; }
[[ ! -e "$TG_RUN" ]] || { echo "Output already exists: $TG_RUN" >&2; exit 1; }
mkdir -p -- "$(dirname -- "$TG_RUN")"
mkdir -- "$TG_RUN"
bash "$TG_ROOT/scripts/train.sh" --model "$TG_MODEL" --data-dir "$TG_DATA" \
  --output-dir "$TG_RUN/train" --steps 500 --batch-size 2 --grad-accum 2 \
  --learning-rate 6.25e-5 --seed 1 --val-windows-per-dataset 64 \
  --device cuda --dtype bfloat16 2>&1 | tee "$TG_RUN/train.log"
bash "$TG_ROOT/scripts/eval_bacc.sh" --checkpoint "$TG_RUN/train/model" --data-dir "$TG_DATA" \
  --windows-per-dataset 100000 --batch-size 4 --seed 1 --device cuda --dtype bfloat16 \
  --output "$TG_RUN/500step_full_metrics.json" 2>&1 | tee "$TG_RUN/500step_eval.log"
bash "$TG_ROOT/scripts/eval_bacc.sh" --base-model "$TG_MODEL" --data-dir "$TG_DATA" \
  --windows-per-dataset 100000 --batch-size 4 --seed 1 --init-seed 1 --device cuda --dtype bfloat16 \
  --output "$TG_RUN/base_full_metrics.json" 2>&1 | tee "$TG_RUN/base_eval.log"
