#!/usr/bin/env python3
"""Apple Silicon (incl. M4) thermal + GPU monitor.

M4 doesn't expose numeric die temps via powermetrics — it reports "thermal pressure
level" instead. That level is the official throttling indicator; translating it:

    Nominal   = cool,   no throttle       (green  — safe)
    Moderate  = warm,   no throttle       (green  — safe under load)
    Heavy     = hot,    mild throttle     (yellow — fine, but slowing)
    Trapping  = too hot, strong throttle  (red    — kill load)
    Sleeping  = emergency cooldown        (red bold — stop now)

Usage:
    sudo python3 monitor.py
    sudo python3 monitor.py 5
"""
import subprocess, sys, os, re, time
import psutil

LEVELS = {
    'Nominal':  ('ok',       '\033[92m'),         # green
    'Moderate': ('warm',     '\033[92m'),         # green (still safe)
    'Heavy':    ('HOT',      '\033[93m'),         # yellow
    'Trapping': ('CRITICAL', '\033[91m'),         # red
    'Sleeping': ('EMERGENCY','\033[91m\033[1m'),  # red bold
}
BOLD, RESET = '\033[1m', '\033[0m'

def poll():
    try:
        r = subprocess.run(
            ['powermetrics', '--samplers', 'thermal,gpu_power,cpu_power',
             '-i', '1000', '-n', '1', '-f', 'text'],
            capture_output=True, text=True, timeout=10)
        return r.stdout, r.stderr, r.returncode
    except Exception as e:
        return '', f'exception: {e}', -1

RE_PRESSURE = re.compile(r'Current pressure level:\s*(\w+)')
RE_GPU_UTIL = re.compile(r'GPU HW active residency:\s*([\d.]+)\s*%')
RE_GPU_PWR  = re.compile(r'GPU Power:\s*(\d+)\s*mW')
RE_CPU_PWR  = [re.compile(r'CPU Power:\s*(\d+)\s*mW'),
               re.compile(r'Combined Power \(CPU \+ GPU \+ ANE\):\s*(\d+)\s*mW')]
RE_ECL_UTIL = re.compile(r'E-Cluster HW active residency:\s*([\d.]+)\s*%')
RE_PCL_UTIL = re.compile(r'P-Cluster HW active residency:\s*([\d.]+)\s*%')
RE_CPU_UTIL = re.compile(r'CPU HW active residency:\s*([\d.]+)\s*%')

def extract_num(out, pat, as_int=False):
    m = pat.search(out)
    if not m: return None
    return int(m.group(1)) if as_int else float(m.group(1))

def main():
    if os.geteuid() != 0:
        print('Run with: sudo python3 monitor.py'); sys.exit(1)

    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    print(f'{BOLD}Apple Silicon thermal + GPU monitor{RESET}  (every {interval}s, Ctrl-C to stop)')
    print('Pressure: Nominal/Moderate = safe | Heavy = throttle | Trapping/Sleeping = DANGER')
    print('-' * 95)
    psutil.cpu_percent(interval=None)  # prime

    while True:
        out, err, rc = poll()
        if rc != 0 or not out:
            print(f'powermetrics failed rc={rc}: {err.strip()[:200]}')
            time.sleep(interval); continue

        m = RE_PRESSURE.search(out)
        level = m.group(1) if m else 'Unknown'
        label, color = LEVELS.get(level, (level, RESET))

        gpu_util = extract_num(out, RE_GPU_UTIL)
        gpu_pwr  = extract_num(out, RE_GPU_PWR, as_int=True)
        cpu_pwr  = None
        for pat in RE_CPU_PWR:
            v = extract_num(out, pat, as_int=True)
            if v is not None: cpu_pwr = v; break

        # CPU util: psutil is reliable across macOS versions; fall back to powermetrics
        try:
            cpu_util = psutil.cpu_percent(interval=None)
        except Exception:
            cpu_util = None
        if cpu_util is None or cpu_util == 0.0:
            v = extract_num(out, RE_CPU_UTIL)
            if v is None:
                e = extract_num(out, RE_ECL_UTIL); p = extract_num(out, RE_PCL_UTIL)
                if e is not None and p is not None: v = (e + p) / 2
                elif p is not None:                 v = p
            if v is not None: cpu_util = v

        ts = time.strftime('%H:%M:%S')
        pres_s = f'{color}{level[:4]:4s}({label}){RESET}'
        gu = f'{gpu_util:4.0f}%' if gpu_util is not None else ' N/A'
        gp = f'{gpu_pwr/1000:5.2f}W' if gpu_pwr is not None else '  N/A'
        cu = f'{cpu_util:4.0f}%' if cpu_util is not None else ' N/A'
        cp = f'{cpu_pwr/1000:5.2f}W' if cpu_pwr is not None else '  N/A'

        print(f'{ts} {pres_s} GPU {gu} {gp} CPU {cu} {cp}')
        time.sleep(interval)

if __name__ == '__main__':
    try: main()
    except KeyboardInterrupt: print('\nstopped.')
