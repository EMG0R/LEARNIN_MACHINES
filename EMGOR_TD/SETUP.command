#!/usr/bin/env bash
# EMGOR_TD — one-click per-machine setup (Mac).
# Double-click this file in Finder. It will:
#   1. cd into its own folder (wherever you put EMGOR_TD/)
#   2. find a Python 3.11 (preferred) or 3.12, fallback to `python3`
#   3. build venv/ next to itself and install all deps
# Re-run any time — it's idempotent.

cd "$(dirname "$0")" || exit 1

echo "================================================================="
echo " EMGOR_TD setup"
echo " folder: $(pwd)"
echo "================================================================="

# Pick the best available Python — TD bundles 3.11.
if [ -n "$PYTHON" ]; then
  PY="$PYTHON"
elif command -v python3.11 >/dev/null 2>&1; then
  PY="python3.11"
elif command -v python3.12 >/dev/null 2>&1; then
  PY="python3.12"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  echo "ERROR: no python3 found. Install Python 3.11 first:"
  echo "  brew install python@3.11"
  echo
  read -n 1 -s -r -p "Press any key to close..."
  exit 1
fi

echo "[setup] Using $($PY --version) at $(command -v $PY)"
echo

if [ ! -d venv ]; then
  echo "[setup] Creating venv/ ..."
  "$PY" -m venv venv || { echo "venv create failed"; read -n 1 -s; exit 1; }
fi

# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision opencv-python numpy scipy Pillow

SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
PYV="$(python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"

echo
echo "================================================================="
echo " DONE."
echo
echo " Python version: $PYV"
echo " venv path:      $(pwd)/venv"
echo
echo " You don't need to set TD's Python Module Path — the callbacks"
echo " auto-bootstrap from this venv on first cook."
echo
echo " (Optional, only if you want it: $SITE )"
echo "================================================================="
echo
read -n 1 -s -r -p "Press any key to close this window..."
echo
