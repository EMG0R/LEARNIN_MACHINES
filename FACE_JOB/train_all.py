# FACE_JOB/train_all.py
"""
FACE_JOB sequential training orchestrator with thermal protection.

Runs in order:
  1. Face detector
  2. Face-part U-Net
  3. Emotion classifier

Resume: stages with both checkpoint + log already present are skipped automatically.
Use --force to re-run completed stages anyway.

Progress is written to logs/progress.json and logs/TRAINING_LOG.md after every stage.

Thermal protection: powermetrics polled every 3s. If Trapping/Sleeping →
SIGSTOP training subprocess; resume with SIGCONT at Nominal. Needs root.

Run from repo root:
    sudo python3 FACE_JOB/train_all.py
    sudo python3 FACE_JOB/train_all.py --only emotion
    sudo python3 FACE_JOB/train_all.py --force          # ignore completed stages
"""
import argparse, json, os, re, signal, subprocess, sys, time, threading
from datetime import datetime, timezone
from pathlib import Path

DET_TAG     = "v1"
PARTS_TAG   = "v1"
EMOTION_TAG = "v1"

# epochs each trainer runs (must match trainer defaults / env overrides)
DET_EPOCHS     = int(os.environ.get("EPOCHS", 25))
PARTS_EPOCHS   = int(os.environ.get("EPOCHS", 30))
EMOTION_EPOCHS = int(os.environ.get("EPOCHS", 35))

BASE     = Path(__file__).parent
LOGS_DIR = BASE / "logs"
LOGS_DIR.mkdir(exist_ok=True)

PROGRESS_FILE = LOGS_DIR / "progress.json"
MD_FILE       = LOGS_DIR / "TRAINING_LOG.md"

DET_CKPT     = BASE / f"face_det/checkpoints/face_det_{DET_TAG}.pt"
PARTS_CKPT   = BASE / f"face_parts/checkpoints/face_parts_{PARTS_TAG}.pt"
EMOTION_CKPT = BASE / f"emotion/checkpoints/emotion_{EMOTION_TAG}.pt"

DET_LOG     = BASE / f"face_det/checkpoints/face_det_{DET_TAG}.log.json"
PARTS_LOG   = BASE / f"face_parts/checkpoints/face_parts_{PARTS_TAG}.log.json"
EMOTION_LOG = BASE / f"emotion/checkpoints/emotion_{EMOTION_TAG}.log.json"

PYTHON = sys.executable

PAUSE_LEVELS = {"Trapping", "Sleeping"}
RESUME_LEVEL = "Nominal"
POLL_SECS    = 3
RE_PRESSURE  = re.compile(r'Current pressure level:\s*(\w+)')

# ── stage registry ─────────────────────────────────────────────────────────────

STAGES = [
    {
        "key":    "det",
        "label":  "Face detector",
        "module": "FACE_JOB.face_det.train",
        "tag":    DET_TAG,
        "ckpt":   DET_CKPT,
        "log":    DET_LOG,
        "epochs": DET_EPOCHS,
        "metric": "best_val_f1",
        "env":    {"RUN_TAG": DET_TAG},
    },
    {
        "key":    "parts",
        "label":  "Face-part U-Net",
        "module": "FACE_JOB.face_parts.train",
        "tag":    PARTS_TAG,
        "ckpt":   PARTS_CKPT,
        "log":    PARTS_LOG,
        "epochs": PARTS_EPOCHS,
        "metric": "best_fg_miou",
        "env":    {"RUN_TAG": PARTS_TAG},
    },
    {
        "key":    "emotion",
        "label":  "Emotion classifier",
        "module": "FACE_JOB.emotion.train",
        "tag":    EMOTION_TAG,
        "ckpt":   EMOTION_CKPT,
        "log":    EMOTION_LOG,
        "epochs": EMOTION_EPOCHS,
        "metric": "best_val_f1",
        "env":    {"RUN_TAG": EMOTION_TAG},
    },
]


# ── progress helpers ───────────────────────────────────────────────────────────

def _load_progress() -> dict:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_progress(prog: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(prog, indent=2))


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ── markdown log ──────────────────────────────────────────────────────────────

def _write_md(prog: dict, run_start: float) -> None:
    lines = [
        "# FACE_JOB Training Log",
        "",
        f"_Last updated: {_now()}_",
        "",
        f"Run started: {prog.get('run_started', '—')}",
        "",
        "## Stages",
        "",
        "| # | Stage | Status | Epochs | Metric | Stopped | Duration |",
        "|---|-------|--------|--------|--------|---------|----------|",
    ]

    for i, s in enumerate(STAGES, 1):
        info   = prog.get(s["key"], {})
        status = info.get("status", "pending")

        if status == "pending":
            icon, status_str = "⏳", "pending"
            epochs_str = metric_str = stop_str = dur_str = "—"
        elif status == "running":
            icon, status_str = "🔄", "running"
            elapsed  = time.time() - info.get("started_ts", time.time())
            dur_str  = f"{elapsed/60:.1f} min"
            epochs_str = metric_str = stop_str = "—"
        elif status == "skipped":
            icon, status_str = "⏭", "skipped (already done)"
            epochs_str = str(info.get("epochs_run", "—"))
            mv         = info.get("metric_value")
            metric_str = f"{mv:.4f}" if mv is not None else "—"
            stop_str   = "⚡ early stop" if info.get("early_stop") else "🏁 full run"
            dur_str    = "—"
        elif status == "done":
            icon, status_str = "✅", "done"
            epochs_str = str(info.get("epochs_run", "—"))
            mv         = info.get("metric_value")
            metric_str = f"{mv:.4f}" if mv is not None else "—"
            stop_str   = "⚡ early stop" if info.get("early_stop") else "🏁 full run"
            secs       = info.get("duration_secs", 0)
            dur_str    = f"{secs/60:.1f} min"
        elif status == "failed":
            icon, status_str = "❌", f"failed (exit {info.get('exit_code', '?')})"
            epochs_str = metric_str = stop_str = "—"
            secs       = info.get("duration_secs", 0)
            dur_str    = f"{secs/60:.1f} min"
        else:
            icon, status_str = "?", status
            epochs_str = metric_str = stop_str = dur_str = "—"

        lines.append(
            f"| {i} | {icon} {s['label']} | {status_str} | "
            f"{epochs_str} | {metric_str} | {stop_str} | {dur_str} |"
        )

    total_elapsed = time.time() - run_start
    lines += ["", f"**Total elapsed:** {total_elapsed/60:.1f} min", ""]

    for s in STAGES:
        info = prog.get(s["key"], {})
        if info.get("status") not in ("done", "skipped"):
            continue
        if not s["log"].exists():
            continue
        try:
            log_data = json.loads(s["log"].read_text())
        except Exception:
            continue
        history = log_data.get("history", [])
        if not history:
            continue

        sample         = history[0]
        numeric_keys   = [k for k in sample if k not in ("epoch",) and isinstance(sample[k], (int, float))]
        metric_cols    = [k for k in numeric_keys if any(t in k for t in ("f1", "iou", "miou", "loss", "lr"))]

        lines += [
            "",
            f"### {s['label']} — epoch history",
            "",
            "| ep | " + " | ".join(metric_cols) + " |",
            "|----" + ("|-----" * len(metric_cols)) + "|",
        ]
        for row in history:
            vals = " | ".join(
                f"{row[k]:.4f}" if isinstance(row.get(k), float) else str(row.get(k, ""))
                for k in metric_cols
            )
            lines.append(f"| {row['epoch']:3d} | {vals} |")

    MD_FILE.write_text("\n".join(lines) + "\n")


# ── thermal watchdog ───────────────────────────────────────────────────────────

class ThermalWatchdog(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True, name="thermal-watchdog")
        self._proc   = None
        self._paused = False
        self._lock   = threading.Lock()
        self._stop   = threading.Event()

    def set_proc(self, proc):
        with self._lock:
            self._proc   = proc
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
                            os.kill(proc.pid, signal.SIGSTOP)
                            self._paused = True
                            print(f"\n[thermal] {level} — training frozen.", flush=True)
                        except ProcessLookupError:
                            pass
                    elif self._paused and level == RESUME_LEVEL:
                        try:
                            os.kill(proc.pid, signal.SIGCONT)
                            self._paused = False
                            print(f"\n[thermal] Nominal — resumed.", flush=True)
                        except ProcessLookupError:
                            pass
            self._stop.wait(POLL_SECS)


watchdog = ThermalWatchdog()


# ── stage runner ───────────────────────────────────────────────────────────────

def _parse_stage_result(stage: dict) -> dict:
    if not stage["log"].exists():
        return {}
    try:
        data       = json.loads(stage["log"].read_text())
        history    = data.get("history", [])
        early_stop = len(history) < stage["epochs"]
        return {
            "epochs_run":   len(history),
            "metric_value": data.get(stage["metric"]),
            "early_stop":   early_stop,
        }
    except Exception:
        return {}


def _stage_already_done(stage: dict) -> bool:
    return stage["ckpt"].exists() and stage["log"].exists()


def run_stage(stage: dict, prog: dict, run_start: float) -> None:
    key   = stage["key"]
    label = stage["label"]

    print(f"\n{'='*60}\nSTAGE: {label}\n{'='*60}\n", flush=True)

    prog[key] = {
        "status":     "running",
        "started_at": _now(),
        "started_ts": time.time(),
    }
    _save_progress(prog)
    _write_md(prog, run_start)

    t0   = time.time()
    proc = subprocess.Popen(
        [PYTHON, "-m", stage["module"]],
        env={**os.environ, **stage["env"]},
        cwd=str(BASE.parent),
    )
    watchdog.set_proc(proc)
    proc.wait()
    watchdog.set_proc(None)
    duration = time.time() - t0

    result = _parse_stage_result(stage)

    if proc.returncode not in (0, -signal.SIGSTOP):
        prog[key] = {
            "status":        "failed",
            "exit_code":     proc.returncode,
            "duration_secs": duration,
            "finished_at":   _now(),
            **result,
        }
        _save_progress(prog)
        _write_md(prog, run_start)
        print(f"[train_all] {label} exited with code {proc.returncode}", flush=True)
        sys.exit(proc.returncode)

    stop_str   = "⚡ early stop" if result.get("early_stop") else "🏁 full run"
    metric_str = f"{result['metric_value']:.4f}" if result.get("metric_value") is not None else "n/a"
    print(
        f"\n[train_all] {label} — {stop_str} | "
        f"{result.get('epochs_run', '?')} epochs | "
        f"{stage['metric']}={metric_str} | "
        f"{duration/60:.1f} min",
        flush=True,
    )

    prog[key] = {
        "status":        "done",
        "duration_secs": duration,
        "finished_at":   _now(),
        **result,
    }
    _save_progress(prog)
    _write_md(prog, run_start)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-det",     action="store_true")
    ap.add_argument("--skip-parts",   action="store_true")
    ap.add_argument("--skip-emotion", action="store_true")
    ap.add_argument("--only",  choices=["det", "parts", "emotion"])
    ap.add_argument("--force", action="store_true",
                    help="Re-run stages even if checkpoint + log already exist")
    args = ap.parse_args()

    if args.only:
        args.skip_det     = args.only != "det"
        args.skip_parts   = args.only != "parts"
        args.skip_emotion = args.only != "emotion"

    explicitly_skipped = {
        "det":     args.skip_det,
        "parts":   args.skip_parts,
        "emotion": args.skip_emotion,
    }

    prog      = _load_progress()
    run_start = time.time()

    if "run_started" not in prog:
        prog["run_started"] = _now()

    if os.geteuid() == 0:
        watchdog.start()
        print("[thermal] Watchdog active.", flush=True)
    else:
        print(
            "[thermal] WARNING: not root — thermal protection disabled.\n"
            "          Re-run with: sudo python3 FACE_JOB/train_all.py",
            flush=True,
        )

    for stage in STAGES:
        key = stage["key"]

        if explicitly_skipped[key]:
            continue

        if not args.force and _stage_already_done(stage):
            result   = _parse_stage_result(stage)
            stop_str = "⚡ early stop" if result.get("early_stop") else "🏁 full run"
            print(
                f"[train_all] {stage['label']} already done "
                f"({stop_str}, {result.get('epochs_run', '?')} epochs). "
                f"Skipping. Use --force to re-run.",
                flush=True,
            )
            if prog.get(key, {}).get("status") != "done":
                prog[key] = {"status": "skipped", **result}
                _save_progress(prog)
            _write_md(prog, run_start)
            continue

        run_stage(stage, prog, run_start)

    watchdog.stop()

    total = time.time() - run_start
    print(f"\n[train_all] All done in {total/60:.1f} min.", flush=True)
    print(f"[train_all] Log: {MD_FILE}", flush=True)


if __name__ == "__main__":
    main()
