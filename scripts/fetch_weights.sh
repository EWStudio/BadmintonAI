#!/usr/bin/env bash
set -euo pipefail

REL="https://github.com/yo-WASSUP/Good-Badminton/releases/download/v0.1.0"
DEST="${1:-vendor/Good-Badminton/weights}"
mkdir -p "$DEST"

FILES=(
  "yolo11s-ball.pt"
  "yolo11n-pose.pt"
  "yolox_nano_8xb8-300e_humanart-40f6f0d0.onnx"
  "rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.onnx"
  "rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.onnx"
)

for f in "${FILES[@]}"; do
  if [[ -s "$DEST/$f" ]]; then
    echo "  have  $f"
    continue
  fi
  echo "  get   $f"
  curl -fL --progress-bar -o "$DEST/$f" "$REL/$f"
done

echo
echo "Weights in $DEST:"
ls -lh "$DEST" | tail -n +2 | awk '{printf "  %-72s %s\n", $9, $5}'
echo
echo "NOTE: the release ships only ONE checkpoint per pose family."
echo "  rtmpose -> only --pose-mode balanced   (rtmpose-s)"
echo "  rtmo    -> only --pose-mode lightweight (rtmo-s)"
echo "Any other --pose-mode falls through to rtmlib's online downloader,"
echo "which pulls from OpenMMLab and may fail or stall."
