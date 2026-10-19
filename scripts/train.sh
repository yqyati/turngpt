#!/usr/bin/env bash
set -euo pipefail
TG_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${TG_PYTHON:-python}" "$TG_ROOT/train_qwen3_turngpt_subset.py" "$@"
