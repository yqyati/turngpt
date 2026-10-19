#!/usr/bin/env bash
# CPU preprocessing with either the Qwen3 or GPT-2 tokenizer.
set -euo pipefail
TG_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -eq 1 && ( "$1" == --help || "$1" == -h ) ]]; then
  echo "Usage: bash scripts/prepare_data.sh TOKENIZER_DIRECTORY NEW_OUTPUT_DIRECTORY"
  echo "Prepare five source groups, then write the eight-corpus cache to NEW_OUTPUT_DIRECTORY/merged."
  exit 0
fi
if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/prepare_data.sh TOKENIZER_DIRECTORY NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi
TG_MODEL="$1"
TG_OUTPUT="$2"
[[ -d "$TG_MODEL" ]] || { echo "Missing tokenizer directory: $TG_MODEL" >&2; exit 1; }
[[ ! -e "$TG_OUTPUT" ]] || { echo "Output already exists: $TG_OUTPUT" >&2; exit 1; }
mkdir -p -- "$(dirname -- "$TG_OUTPUT")"
mkdir -- "$TG_OUTPUT"
TG_OUTPUT="$(cd -- "$TG_OUTPUT" && pwd)"
TG_PREPARERS=(prepare_qwen3_turngpt_text.py prepare_ccpe_turngpt.py prepare_taskmaster1_turngpt.py prepare_taskmaster_turngpt.py prepare_metawoz_turngpt.py)
TG_NAMES=(three_corpus ccpe taskmaster1 taskmaster2_3 metawoz)
for TG_I in "${!TG_PREPARERS[@]}"; do
  "${TG_PYTHON:-python}" "$TG_ROOT/${TG_PREPARERS[$TG_I]}" \
    --model "$TG_MODEL" --output-dir "$TG_OUTPUT/sources/${TG_NAMES[$TG_I]}" \
    --max-length 256 --keep-length 64 --overlap-length 10 \
    2>&1 | tee "$TG_OUTPUT/${TG_NAMES[$TG_I]}.log"
done
"${TG_PYTHON:-python}" - "$TG_OUTPUT" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
mapping = {"daily_dialog": "three_corpus", "personachat": "three_corpus", "multiwoz21": "three_corpus",
           "ccpe": "ccpe", "taskmaster1": "taskmaster1", "taskmaster2": "taskmaster2_3",
           "taskmaster3": "taskmaster2_3", "metawoz": "metawoz"}
(root / "source_map.json").write_text(json.dumps({k: "sources/" + v for k, v in mapping.items()}, indent=2) + "\n")
PY
"${TG_PYTHON:-python}" "$TG_ROOT/merge_turngpt_processed.py" \
  --source-map "$TG_OUTPUT/source_map.json" --output-dir "$TG_OUTPUT/merged" \
  2>&1 | tee "$TG_OUTPUT/merge.log"
"${TG_PYTHON:-python}" "$TG_ROOT/verify_data.py" --data-dir "$TG_OUTPUT/merged"
