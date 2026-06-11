#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/raw/kitpri-v2}"
METADATA_ROOT="${METADATA_ROOT:-${DATA_ROOT}/metadata}"

cd "${REPO_ROOT}"
python -m src.train \
  --config configs/config.yaml \
  --train-csv "${METADATA_ROOT}/train.csv" \
  --val-csv "${METADATA_ROOT}/val.csv" \
  --data-root "${DATA_ROOT}" \
  --output-dir results
