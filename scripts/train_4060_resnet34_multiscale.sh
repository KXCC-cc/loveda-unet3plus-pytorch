#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/venvs/unet3/bin/activate"

python train.py --config configs/resnet_pretrained_512_multiscale.yaml "$@"
