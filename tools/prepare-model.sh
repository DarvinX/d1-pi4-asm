#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 ]]; then
    printf 'Usage: %s MODEL.gguf PROJECTOR.gguf\n' "$0" >&2
    exit 2
fi
model=$(realpath "$1")
projector=$(realpath "$2")
cd "$(dirname "$0")/.."
vision=research/d1-640x480
mkdir -p build/panels
python3 tools/prepare.py "$model" --d1-backbone --out build/d1-backbone-assets
python3 "$vision/prepare_vision.py" "$projector" build/d1-vision-assets
python3 "$vision/prepare_embedding_crc.py" "$model" build/d1-embed-crc
python3 "$vision/prepare_deferred_model_crc.py" build/d1-backbone-assets/model.inc "$model" \
    build/d1-deferred-crc/segments.json --assembly build/d1-deferred-crc/segments.inc
python3 "$vision/prepare_deferred_model_crc.py" build/d1-vision-assets/vision.inc "$projector" \
    build/d1-deferred-crc/projector.json --assembly build/d1-deferred-crc/projector.inc --projector
python3 "$vision/prepare_q4_i8.py" "$model" build/panels/q4-panel.bin
python3 "$vision/prepare_q6_i16.py" "$model" build/panels/q6-panel.bin
python3 "$vision/prepare_vision_q8_i8.py" "$projector" build/panels/q8-resident.bin --early-resident
