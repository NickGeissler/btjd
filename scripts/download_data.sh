#!/bin/bash
# Download the datasets for all four experiments into data/.
# Source: https://huggingface.co/datasets/NickGeissler/btjd-data (~86 GB).
# Usage: bash scripts/download_data.sh [duffing|rb|burgers|cfd ...]   (default: all)
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:-data}
HF=${HF:-hf}
DOMAINS=${*:-duffing rb burgers cfd}

INCLUDE=()
for d in $DOMAINS; do INCLUDE+=(--include "$d/*"); done
$HF download NickGeissler/btjd-data --repo-type dataset --local-dir "$DATA" "${INCLUDE[@]}"
