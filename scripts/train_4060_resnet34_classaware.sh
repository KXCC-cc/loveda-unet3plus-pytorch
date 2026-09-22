#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/venvs/unet3/bin/activate"

python train.py --config configs/resnet34_512_classaware_only.yaml "$@"
