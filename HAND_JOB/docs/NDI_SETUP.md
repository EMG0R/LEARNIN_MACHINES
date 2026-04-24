# NDI Setup on Apple Silicon (macOS)

Steps to get `ndi-python` sending video to TouchDesigner from the `live_app`.

## 1. Install NDI SDK

Download from https://ndi.video/for-developers/ndi-sdk/ and run the installer.

Installs to: `/Library/NDI SDK for Apple/`

Verify:
```bash
ls "/Library/NDI SDK for Apple/lib/macOS/"
# expect: libndi.dylib, libndi_licenses.txt
```

**Restart your computer after installing** — NDI's runtime services register at login.

## 2. Install build tools

```bash
brew install cmake
```

## 3. Build `ndi-python` from source

The PyPI tarball is broken (missing the `pybind11` git submodule), and a prebuilt `.so` in the repo can shadow the fresh build. Build from git:

```bash
rm -rf /tmp/ndi-python
git clone --recursive https://github.com/buresu/ndi-python.git /tmp/ndi-python
cd /tmp/ndi-python
pip3 install --break-system-packages \
  --config-settings=cmake.args="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" .
```

The `CMAKE_POLICY_VERSION_MINIMUM=3.5` flag is required because the bundled pybind11 declares a minimum CMake version that modern CMake rejects.

`FindNDI.cmake` auto-detects the SDK at `/Library/NDI SDK for Apple/` — no env vars needed.

## 4. Verify

```bash
cd /tmp  # avoid directory-shadow
python3 -c "import NDIlib as ndi; print(ndi.initialize())"
```

Expect: `True`.

## Known pitfalls

- **`ModuleNotFoundError: No module named 'NDIlib.NDIlib'`** — a stale `.so` built for a different Python minor version is in the installed package. Check:
  ```bash
  ls $(python3 -c "import site; print(site.getsitepackages()[0])")/NDIlib/
  ```
  The `.so` filename (e.g. `NDIlib.cpython-311-darwin.so`) must match your current Python (`python3 --version`). If not, uninstall and rebuild against the right interpreter:
  ```bash
  pip3 uninstall -y ndi-python
  # then rebuild with the matching python3
  ```

- **`cmake: command not found`** — install via `brew install cmake`.

- **`Could NOT find NDI (missing: NDI_DIR FALSE)`** — the NDI SDK is not installed at `/Library/NDI SDK for Apple/`. Re-run the SDK installer.

- **`add_subdirectory given source "lib/pybind11" which is not an existing directory`** — you installed from PyPI or cloned without `--recursive`. Re-clone with `--recursive`.

## 5. Use in `live_app`

`live_app/ndi_sender.py` auto-detects NDIlib. Once import succeeds, running:

```bash
cd HAND_JOB
python3 -m live_app.app
```

publishes an NDI source named `LEARNIN_MACHINES` (configured in `live_app/config.py`). In TouchDesigner, add a **Video Device In TOP** → source list should show it.
