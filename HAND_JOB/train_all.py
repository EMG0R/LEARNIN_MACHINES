"""
Single entry point: trains segmentation model then gesture model, then updates config.

Thermal protection: powermetrics runs silently in the background.
If Trapping or Sleeping holds for >10s → training subprocess is frozen (SIGSTOP).
Resumes automatically with SIGCONT when pressure returns to Nominal.
Heavy is fine — only the level after it triggers a pause.

Run from HAND_JOB/:
    sudo .venv/bin/python3 train_all.py          # sudo needed for powermetrics
    sudo .venv/bin/python3 train_all.py --skip-seg
    sudo .venv/bin/python3 train_all.py --skip-gesture
"""
import argparse, os, re, signal, subprocess, sys, time, threading
from pathlib import Path

SEG_TAG     = "v7"
GESTURE_TAG = "v7"

SEG_CKPT_PATH     = Path("hand_seg/checkpoints") / f"hand_seg_{SEG_TAG}.pt"
GESTURE_CKPT_PATH = Path("gesture/checkpoints")  / f"gesture_{GESTURE_TAG}.pt"
CONFIG_PATH       = Path("live_app/config.py")

PYTHON = sys.executable

PAUSE_LEVELS = {"Trapping", "Sleeping"}   # Heavy is OK
RESUME_LEVEL = "Nominal"
POLL_SECS    = 3

RE_PRESSURE = re.compile(r'Current pressure level:\s*(\w+)')


# ─── THERMAL WATCHDOG ────────────────────────────────────────────────────────

class ThermalWatchdog(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True, name="thermal-watchdog")
        self._proc: "subprocess.Popen | None" = None
        self._paused = False
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def set_proc(self, proc):
        with self._lock:
            self._proc = proc
            self._paused = False

    def stop(self):
        self._stop.set()

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
                            os.kill(proc.pid, signal.SIGSTOP)
                            self._paused = True
                            print(f"\n[thermal] {level} — training frozen. Waiting for Nominal...", flush=True)
                        except ProcessLookupError:
                            pass
                    elif self._paused and level == RESUME_LEVEL:
                        try:
                            os.kill(proc.pid, signal.SIGCONT)
                            self._paused = False
                            print(f"\n[thermal] Nominal — training resumed.", flush=True)
                        except ProcessLookupError:
                            pass

            self._stop.wait(POLL_SECS)


watchdog = ThermalWatchdog()


# ─── STAGE RUNNER ────────────────────────────────────────────────────────────

def run_stage(cmd, label, env_extra=None, cwd=None):
    merged = {**os.environ, **(env_extra or {})}
    print(f"\n{'='*60}\n{label}\n{'='*60}\n", flush=True)
    proc = subprocess.Popen(cmd, env=merged, cwd=str(cwd) if cwd else None)
    watchdog.set_proc(proc)
    proc.wait()
    watchdog.set_proc(None)
    if proc.returncode not in (0, -signal.SIGSTOP):
        print(f"[train_all] {label} exited with code {proc.returncode}", flush=True)
        sys.exit(proc.returncode)


def train_seg():
    run_stage(
        [PYTHON, "train.py"],
        label="STAGE 1: Segmentation model",
        env_extra={
            "IMG_SIZE": "192", "BATCH": "32", "EPOCHS": "35",
            "LR": "3e-4", "WD": "1e-4", "WORKERS": "6",
            "PATIENCE": "6", "RUN_TAG": SEG_TAG, "AUG": "heavy",
        },
        cwd=Path.cwd() / "hand_seg",
    )


def train_gesture():
    run_stage(
        [PYTHON, "train_v7.py"],
        label="STAGE 2: Gesture classifier",
        cwd=Path.cwd() / "gesture",
    )


def update_config():
    print("\n[train_all] Updating live_app/config.py ...", flush=True)
    text = CONFIG_PATH.read_text()
    text = re.sub(
        r'SEG_CKPT\s*=\s*BASE\s*/\s*"[^"]*"',
        f'SEG_CKPT     = BASE / "hand_seg/checkpoints/hand_seg_{SEG_TAG}.pt"',
        text,
    )
    text = re.sub(
        r'GESTURE_CKPT\s*=\s*BASE\s*/\s*"[^"]*"',
        f'GESTURE_CKPT = BASE / "gesture/checkpoints/gesture_{GESTURE_TAG}.pt"',
        text,
    )
    CONFIG_PATH.write_text(text)
    print(f"[train_all]   SEG_CKPT  → hand_seg_{SEG_TAG}.pt", flush=True)
    print(f"[train_all]   GESTURE   → gesture_{GESTURE_TAG}.pt", flush=True)


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-seg",     action="store_true")
    ap.add_argument("--skip-gesture", action="store_true")
    args = ap.parse_args()

    if os.geteuid() == 0:
        watchdog.start()
        print("[thermal] Watchdog active — pauses instantly at Trapping/Sleeping, resumes at Nominal.", flush=True)
    else:
        print(
            "[thermal] WARNING: not root — thermal protection disabled.\n"
            "          Re-run with: sudo .venv/bin/python3 train_all.py",
            flush=True,
        )

    t0 = time.time()

    if not args.skip_seg:
        train_seg()
        print(f"\n[train_all] Seg done in {(time.time()-t0)/60:.1f} min", flush=True)
        if not SEG_CKPT_PATH.exists():
            print(f"[train_all] WARNING: checkpoint not found: {SEG_CKPT_PATH}", flush=True)
    else:
        print("[train_all] Skipping seg.")

    if not args.skip_gesture:
        train_gesture()
        print(f"\n[train_all] Gesture done in {(time.time()-t0)/60:.1f} min", flush=True)
        if not GESTURE_CKPT_PATH.exists():
            print(f"[train_all] WARNING: checkpoint not found: {GESTURE_CKPT_PATH}", flush=True)
    else:
        print("[train_all] Skipping gesture.")

    seg_ok     = SEG_CKPT_PATH.exists()     or args.skip_seg
    gesture_ok = GESTURE_CKPT_PATH.exists() or args.skip_gesture
    if seg_ok and gesture_ok:
        update_config()
    else:
        print("[train_all] Skipping config update — checkpoint(s) missing.")

    watchdog.stop()
    print(f"\n[train_all] All done in {(time.time()-t0)/60:.1f} min.")
    print("[train_all] Run: python3 -m live_app.app")


if __name__ == "__main__":
    main()
