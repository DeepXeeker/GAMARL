#!/usr/bin/env bash
set -euo pipefail
CFG="${1:-configs/env_synthetic.yaml}"
NAME="$(basename "$CFG" .yaml)"
PYTHONPATH=src python -m train --config-name "$NAME"
