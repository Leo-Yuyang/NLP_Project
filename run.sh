#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
OUTPUT="${1:-$ROOT/outputs}"
PYTHON="${PYTHON:-python3}"
if [ "$#" -ge 2 ]; then
    set -- --input "$2"
else
    set --
fi

mkdir -p "$OUTPUT"
for experiment in main baseline transfer efficiency scale control ablation robustness calibration significance subsample_stability; do
    "$PYTHON" "$ROOT/${experiment}_experiment.py" "$@" --output "$OUTPUT/$experiment.json"
done
