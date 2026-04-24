#!/usr/bin/env bash
# Train seg_v3 then gesture_v3 sequentially. Stream to stdout + tee to logs/.
# Resumable: if interrupted (Ctrl-C, crash, shutdown), just re-run — each script
# auto-resumes from its last saved epoch (checkpoints/*.last.pt). Stage 1 finishing
# means stage 2 starts (or resumes) next. To force fresh: RESUME=0 prefix.
# Monitor from another terminal:
#     tail -f HAND_JOB/logs/seg_v3.log
#     tail -f HAND_JOB/logs/gesture_v3.log
#     sudo python3 HAND_JOB/monitor.py 5    # thermal/GPU monitor
# Run:
#     cd HAND_JOB && ./train_v3.sh
set -euo pipefail

cd "$(dirname "$0")"
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || { echo "ERR: $PY not found — activate your venv"; exit 1; }

mkdir -p logs
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "=== [$(ts)] Stage 1/2: hand_seg v3 ==="
cd hand_seg
IMG_SIZE=256 BATCH=16 EPOCHS=25 LR=2e-4 WORKERS=6 \
    WARMUP=2 EMA=0.999 AMP=1 RESUME=1 RUN_TAG=v3 \
    "../$PY" -u train_v3.py 2>&1 | tee -a "../logs/seg_v3.log"
cd ..
echo "=== [$(ts)] Stage 1 DONE ==="

echo "=== [$(ts)] Stage 2/2: gesture v3 ==="
cd gesture
IMG_SIZE=128 BATCH=96 EPOCHS=35 LR=4e-4 WORKERS=6 \
    WARMUP=2 EMA=0.999 MIXUP=0.2 LS=0.1 AMP=1 RESUME=1 RUN_TAG=v3 \
    "../$PY" -u train_v3.py 2>&1 | tee -a "../logs/gesture_v3.log"
cd ..
echo "=== [$(ts)] Stage 2 DONE ==="

echo ""
echo "Checkpoints:"
echo "  hand_seg/checkpoints/hand_seg_v3.pt"
echo "  gesture/checkpoints/gesture_v3.pt"
echo ""
echo "To use them in the live app, edit live_app/config.py:"
echo "  SEG_CKPT     = BASE / 'hand_seg/checkpoints/hand_seg_v3.pt'"
echo "  GESTURE_CKPT = BASE / 'gesture/checkpoints/gesture_v3.pt'"
