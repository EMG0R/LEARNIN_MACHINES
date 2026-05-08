"""Sequential orchestrator for OBJECTIFICATION training.

Currently a single stage (seg/train.py) but uses the same thermal-watchdog
+ subprocess pattern as HAND_JOB/train_all.py so a second stage (e.g. a
fine-tune pass) can be appended later without restructuring.

Run from OBJECTIFICATION/:
    python train_all.py
Or from repo root:
    python -m OBJECTIFICATION.train_all
Env vars passed through to seg/train.py: IMG_SIZE, BATCH, EPOCHS, LR, WORKERS, RUN_TAG, SMOKE.

Thermal protection: powermetrics polls silently every 3 s (requires sudo/root).
If Trapping or Sleeping → training subprocess is frozen (SIGSTOP).
Resumes automatically with SIGCONT when pressure returns to Nominal.
Heavy is fine — only Trapping/Sleeping triggers a pause.

sudo .venv/bin/python3 OBJECTIFICATION/train_all.py   # sudo needed for powermetrics
"""
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "seg" / "checkpoints"
LOG_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = LOG_DIR / "train_all_summary.json"

DL_TRAIN_PER_CLASS = int(os.environ.get("DL_TRAIN_PER_CLASS", 3000))
DL_VAL_PER_CLASS   = int(os.environ.get("DL_VAL_PER_CLASS",   500))
SKIP_DOWNLOAD      = bool(int(os.environ.get("SKIP_DOWNLOAD", 0)))

DATA_ROOT = ROOT / "shared" / "datasets" / "openimages_v7"

PAUSE_LEVELS = {"Trapping", "Sleeping"}   # Heavy is OK
RESUME_LEVEL = "Nominal"
POLL_SECS    = 3

RE_PRESSURE = re.compile(r'Current pressure level:\s*(\w+)')


# ── THERMAL WATCHDOG ──────────────────────────────────────────────────────────

class ThermalWatchdog(threading.Thread):
    """Polls `powermetrics` for thermal pressure every POLL_SECS seconds.
    SIGSTOP child on Trapping/Sleeping; SIGCONT on Nominal.
    Mirrors HAND_JOB/train_all.py (polling pattern, not streaming).
    """
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


# ── STAGE RUNNER ──────────────────────────────────────────────────────────────

def run_stage(name, cmd):
    """Run one training stage under the thermal watchdog.

    Returns a result dict with name, exit code, and wall-clock duration.
    Mirrors HAND_JOB/train_all.py run_stage() pattern.
    """
    print(f"\n{'='*60}\n{name}\n{'='*60}\n", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=os.environ.copy())
    watchdog.set_proc(proc)
    rc = proc.wait()
    watchdog.set_proc(None)
    dur = time.time() - t0
    print(f"\n=== [{name}] exit={rc} duration={dur:.0f}s ===", flush=True)
    return {"name": name, "exit": rc, "duration_s": round(dur, 1)}


# ── HELPERS ───────────────────────────────────────────────────────────────────

def split_ready(split: str) -> bool:
    """True if {split}/masks/ has at least one PNG (i.e. prepare_masks ran)."""
    masks_dir = DATA_ROOT / split / "masks"
    if not masks_dir.exists():
        return False
    return any(masks_dir.glob("*.png"))


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    if os.geteuid() == 0:
        watchdog.start()
        print("[thermal] Watchdog active — pauses instantly at Trapping/Sleeping, resumes at Nominal.", flush=True)
    else:
        print(
            "[thermal] WARNING: not root — thermal protection disabled.\n"
            "          Re-run with: sudo .venv/bin/python3 OBJECTIFICATION/train_all.py",
            flush=True,
        )

    stages = []
    if not SKIP_DOWNLOAD:
        for split, n in (("train", DL_TRAIN_PER_CLASS), ("val", DL_VAL_PER_CLASS)):
            if not split_ready(split):
                stages.append((f"download-{split}",
                               [sys.executable, "-m", "OBJECTIFICATION.data_pipeline.download",
                                "--split", split, "--max-per-class", str(n)]))
                stages.append((f"prepare-{split}",
                               [sys.executable, "-m", "OBJECTIFICATION.data_pipeline.prepare_masks",
                                "--split", split]))
    stages.append(("seg", [sys.executable, "-m", "OBJECTIFICATION.seg.train"]))

    summary = []
    for name, cmd in stages:
        print(f"\n=== [{name}] starting: {' '.join(cmd)} ===", flush=True)
        r = run_stage(name, cmd)
        summary.append(r)
        if r["exit"] != 0:
            print(f"[{name}] failed (exit={r['exit']}) -> aborting", flush=True)
            break

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary written to {SUMMARY_PATH}", flush=True)

    watchdog.stop()

    overall_ok = all(r["exit"] == 0 for r in summary)
    total_min = sum(r["duration_s"] for r in summary) / 60
    print(f"\n[train_all] All done in {total_min:.1f} min. Status: {'OK' if overall_ok else 'FAILED'}")
    if overall_ok:
        print("[train_all] Run: python -m OBJECTIFICATION.live_app.app")


if __name__ == "__main__":
    # Run from repo root so `-m OBJECTIFICATION.seg.train` import resolves.
    os.chdir(ROOT.parent)
    main()
