# FACE_JOB/train_all.py
"""
FACE_JOB sequential training orchestrator with thermal protection.

Ported from HAND_JOB/train_all.py. Runs in order:
  1. Face detector
  2. Face-part U-Net
  3. Emotion classifier

Thermal protection: powermetrics polled every 3s. If Trapping/Sleeping →
SIGSTOP training subprocess; resume with SIGCONT at Nominal. Needs root.

Run from repo root:
    sudo python3 FACE_JOB/train_all.py
    sudo python3 FACE_JOB/train_all.py --skip-det
    sudo python3 FACE_JOB/train_all.py --skip-parts
    sudo python3 FACE_JOB/train_all.py --skip-emotion
    sudo python3 FACE_JOB/train_all.py --only emotion
"""
import argparse, os, re, signal, subprocess, sys, time, threading
from pathlib import Path

DET_TAG     = "v1"
PARTS_TAG   = "v1"
EMOTION_TAG = "v1"

BASE = Path(__file__).parent
DET_CKPT     = BASE / f"face_det/checkpoints/face_det_{DET_TAG}.pt"
PARTS_CKPT   = BASE / f"face_parts/checkpoints/face_parts_{PARTS_TAG}.pt"
EMOTION_CKPT = BASE / f"emotion/checkpoints/emotion_{EMOTION_TAG}.pt"

PYTHON = sys.executable

PAUSE_LEVELS = {"Trapping", "Sleeping"}
RESUME_LEVEL = "Nominal"
POLL_SECS    = 3
RE_PRESSURE  = re.compile(r'Current pressure level:\s*(\w+)')


class ThermalWatchdog(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True, name="thermal-watchdog")
        self._proc = None
        self._paused = False
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def set_proc(self, proc):
        with self._lock:
            self._proc = proc
            self._paused = False

    def stop(self): self._stop.set()

    def _pressure(self):
        try:
            r = subprocess.run(
                ["powermetrics", "--samplers", "thermal", "-i", "1000", "-n", "1", "-f", "text"],
                capture_output=True, text=True, timeout=12,
            )
            m = RE_PRESSURE.search(r.stdout)
            return m.group(1) if m else None
        except Exception:
            return None

    def run(self):
        while not self._stop.is_set():
            level = self._pressure()
            with self._lock:
                proc = self._proc
                if proc is not None:
                    if level in PAUSE_LEVELS and not self._paused:
                        try:
                            os.kill(proc.pid, signal.SIGSTOP); self._paused = True
                            print(f"\n[thermal] {level} — training frozen.", flush=True)
                        except ProcessLookupError: pass
                    elif self._paused and level == RESUME_LEVEL:
                        try:
                            os.kill(proc.pid, signal.SIGCONT); self._paused = False
                            print(f"\n[thermal] Nominal — resumed.", flush=True)
                        except ProcessLookupError: pass
            self._stop.wait(POLL_SECS)


watchdog = ThermalWatchdog()


def run_stage(cmd, label, env_extra=None):
    merged = {**os.environ, **(env_extra or {})}
    print(f"\n{'='*60}\n{label}\n{'='*60}\n", flush=True)
    proc = subprocess.Popen(cmd, env=merged, cwd=str(BASE.parent))
    watchdog.set_proc(proc)
    proc.wait()
    watchdog.set_proc(None)
    if proc.returncode not in (0, -signal.SIGSTOP):
        print(f"[train_all] {label} exited with code {proc.returncode}", flush=True)
        sys.exit(proc.returncode)


def train_det():
    run_stage(
        [PYTHON, "-m", "FACE_JOB.face_det.train"],
        label="STAGE 1: Face detector",
        env_extra={"RUN_TAG": DET_TAG},
    )


def train_parts():
    run_stage(
        [PYTHON, "-m", "FACE_JOB.face_parts.train"],
        label="STAGE 2: Face-part U-Net",
        env_extra={"RUN_TAG": PARTS_TAG},
    )


def train_emotion():
    run_stage(
        [PYTHON, "-m", "FACE_JOB.emotion.train"],
        label="STAGE 3: Emotion classifier",
        env_extra={"RUN_TAG": EMOTION_TAG},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-det", action="store_true")
    ap.add_argument("--skip-parts", action="store_true")
    ap.add_argument("--skip-emotion", action="store_true")
    ap.add_argument("--only", choices=["det", "parts", "emotion"])
    args = ap.parse_args()

    if args.only:
        args.skip_det     = args.only != "det"
        args.skip_parts   = args.only != "parts"
        args.skip_emotion = args.only != "emotion"

    if os.geteuid() == 0:
        watchdog.start()
        print("[thermal] Watchdog active.", flush=True)
    else:
        print("[thermal] WARNING: not root — thermal protection disabled.\n"
              "          Re-run with: sudo python3 FACE_JOB/train_all.py", flush=True)

    t0 = time.time()
    if not args.skip_det:     train_det()
    if not args.skip_parts:   train_parts()
    if not args.skip_emotion: train_emotion()

    for label, path, skipped in [
        ("face_det",    DET_CKPT,     args.skip_det),
        ("face_parts",  PARTS_CKPT,   args.skip_parts),
        ("emotion",     EMOTION_CKPT, args.skip_emotion),
    ]:
        if skipped: continue
        if path.exists():
            print(f"[train_all] {label}: checkpoint saved at {path}", flush=True)
        else:
            print(f"[train_all] WARNING: {label} checkpoint missing at {path}", flush=True)

    watchdog.stop()
    print(f"\n[train_all] All done in {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
