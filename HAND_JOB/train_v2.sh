#!/usr/bin/env bash
# Train seg_v2 (256px, heavy aug) then gesture_v2 (128px, heavy aug) sequentially.
# Logs stream to stdout and tee'd to files — monitor from another terminal:
#     tail -f HAND_JOB/logs/seg_v2.log
#     tail -f HAND_JOB/logs/gesture_v2.log
# Run from HAND_JOB/:
#     ./train_v2.sh
set -euo pipefail

cd "$(dirname "$0")"
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || { echo "ERR: $PY not found — activate your venv"; exit 1; }

mkdir -p logs
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "=== [$(ts)] Stage 1/2: hand_seg v2 (256px, AUG=heavy) ==="
cd hand_seg
IMG_SIZE=256 BATCH=16 EPOCHS=30 LR=3e-4 WORKERS=6 \
    RUN_TAG=v2_heavy256 AUG=heavy \
    "../$PY" -u train.py 2>&1 | tee "../logs/seg_v2.log"
cd ..
echo "=== [$(ts)] Stage 1 DONE ==="

echo "=== [$(ts)] Stage 2/2: gesture v2 (128px, AUG=heavy, Wide) ==="
cd gesture
IMG_SIZE=128 BATCH=96 EPOCHS=40 LR=3e-4 WORKERS=6 \
    RUN_TAG=v2_wide128 AUG=heavy MODEL=wide SCHED=cosine \
    "../$PY" -u train.py 2>&1 | tee "../logs/gesture_v2.log"
cd ..
echo "=== [$(ts)] Stage 2 DONE ==="

echo ""
echo "Checkpoints:"
echo "  hand_seg/checkpoints/hand_seg_v2_heavy256.pt"
echo "  gesture/checkpoints/gesture_v2_wide128.pt"
echo ""
echo "To use them, edit live_app/config.py:"
echo "  SEG_CKPT     = BASE / 'hand_seg/checkpoints/hand_seg_v2_heavy256.pt'"
echo "  GESTURE_CKPT = BASE / 'gesture/checkpoints/gesture_v2_wide128.pt'"
