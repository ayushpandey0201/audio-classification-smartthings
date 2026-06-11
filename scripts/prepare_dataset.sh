#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESTINATION="${REPO_ROOT}/data/raw/kitpri-v2"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Kaggle CLI not found. Install dependencies with: pip install -r requirements.txt" >&2
  exit 1
fi

if [[ ! -f "${HOME}/.kaggle/kaggle.json" ]]; then
  echo "Missing ${HOME}/.kaggle/kaggle.json. Configure your Kaggle API token first." >&2
  exit 1
fi

mkdir -p "${DESTINATION}"
kaggle datasets download \
  --dataset ayushalia/kitpri-v2 \
  --path "${DESTINATION}" \
  --unzip

echo "kitpri_v2 extracted to ${DESTINATION}"
