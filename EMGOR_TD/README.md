# EMGOR_TD

Self-contained TouchDesigner kit for the EMGOR hand + face mesh system.
Two drop-in COMPs:

- `EMGOR_HAND` — webcam IN → mesh **points CHOP** + edges DAT + OSC `/emgor/hand`
- `EMGOR_FACE` — webcam IN → mesh **points CHOP** + edges DAT + OSC `/emgor/face`

Each COMP outputs **mesh coordinates as point data** (not a texture).
Render the wireframe natively in TD using the included `td_render_lines.py`
Script SOP, then composite the rendered mesh over the webcam.

```
EMGOR_TD/
├── README.md                ← you are here
├── SETUP.command            ← double-click on Mac to install deps
├── EXAMPLE.toe              ← your TD project (lives in here)
├── EMGOR_HAND.tox           ← saved drag-and-drop COMP (after first build)
├── EMGOR_FACE.tox
├── venv/                    ← Python deps (created by SETUP.command)
└── RESOURCES/
    ├── td_hand.py           ← Script CHOP — hand inference + points
    ├── td_face.py           ← Script CHOP — face inference + points
    ├── td_render_lines.py   ← Script SOP — points + edges → line geometry
    ├── lib/                 ← bundled Python modules (don't move)
    └── checkpoints/         ← model weights (~33 MB)
```

The whole folder is the kit. Save `.toe`, `.tox`, and venv all inside it
— move the folder, the project moves with everything it needs.

---

## One-time setup per machine — just double-click

In Finder, open `EMGOR_TD/` and **double-click `SETUP.command`**. A
Terminal window opens, finds Python 3.11 (or falls back to 3.12 / system
`python3`), creates `EMGOR_TD/venv/`, installs torch / opencv / numpy /
scipy / Pillow. Press any key to close when it says DONE.

That's it. **No TD preferences change required.** The callback scripts
auto-discover `EMGOR_TD/venv/` and inject it into TD's `sys.path` on
first cook.

> Re-running `SETUP.command` is safe and idempotent — handy if Python
> deps got messed up.

> If TD's bundled Python version doesn't match what setup picked, run
> from terminal with: `PYTHON=python3.X ./SETUP.command` (check TD's
> `Help > About` for the version).

---

## Building the COMP — `EMGOR_HAND` (do once, save as .tox)

Save your `.toe` next to `EMGOR_TD/`. **Drop a Base COMP**, rename
`EMGOR_HAND`, double-click into it, then build:

| Operator name | Type           | Wired/parameters                                                 |
| ------------- | -------------- | ---------------------------------------------------------------- |
| `in1`         | In TOP         | (no params — receives webcam from outside the COMP)              |
| `script1`     | **Script CHOP**| **Setup** page → Callbacks DAT = `callbacks`                      |
| `out1`        | Out CHOP       | wired from `script1`                                             |
| `callbacks`   | Text DAT       | File = `EMGOR_TD/RESOURCES/td_hand.py`, Sync to File = ON         |
| `edges`       | Table DAT      | (empty — script populates it each cook)                          |
| `oscout1`     | OSC Out DAT    | Network Address e.g. `127.0.0.1`, Network Port e.g. `7000`        |

Notes:
- The Script CHOP doesn't take a wired input — it reads the webcam by
  referencing `op('in1')` in code. Just make sure `in1` (In TOP) is in the
  same COMP.
- Watch **Textport** (`Alt+T`). First cook freezes ~1–2s while PyTorch
  loads, then prints `[models] loading hand seg...`.
- `script1` will output a CHOP whose sample count fluctuates with the
  number of mesh points found. Channels: `tx`, `ty`, `region`.

Exit the COMP. Right-click `EMGOR_HAND` → **Save Component .tox...** →
save as `EMGOR_HAND.tox` into `EMGOR_TD/`.

### Build `EMGOR_FACE`

Identical, but:
- COMP named `EMGOR_FACE`
- `callbacks` File = `EMGOR_TD/RESOURCES/td_face.py`
- Save as `EMGOR_FACE.tox`

---

## Rendering the mesh in TD over the webcam

The COMP gives you **points** (Out CHOP) and **edges** (`edges` Table DAT
inside the COMP). To turn those into renderable wireframe geometry that
sits over the webcam:

### 1. Build a renderable mesh SOP (do once per COMP)

Outside the EMGOR_HAND COMP, drop a **Script SOP**, name it `mesh_sop`.
Configure:

- **Inputs**: it accepts up to 9 inputs. Wire input 0 from `EMGOR_HAND`'s
  `out1` (the points CHOP — Script SOPs can accept CHOP and DAT inputs by
  parameter reference). Wire input 1 from the `edges` DAT inside the
  COMP — drag `EMGOR_HAND/edges` onto the second input slot, or set the
  **`Input2`** parameter to `EMGOR_HAND/edges`.
- **Setup** page → Callbacks DAT = a Text DAT loaded from
  `EMGOR_TD/RESOURCES/td_render_lines.py`.

The Script SOP now produces 2D line geometry in normalized [0,1] × [0,1]
space (Y already flipped so image-top stays on top).

### 2. Render the geometry

```
mesh_sop ──→ Geometry COMP (sop1 = mesh_sop)
                     │
                     ▼
            Render TOP   ←── Camera COMP (Orthographic, see params below)
                     │
                     ▼
            Composite TOP (Over)  ◀── Video Device In TOP
                     │
                     ▼
                  output
```

**Camera COMP parameters:**
- **View** page → Projection = **Orthographic**
- **View** page → Ortho Width = `1`
- **Xform** page → Translate = `(0.5, 0.5, 1)` (look at center of the unit
  square from Z=1)
- **Xform** page → Rotate = `(0, 0, 0)`

**Render TOP parameters:**
- **Common** page → Pixel Format = `RGBA 32-bit float (RGBA32F)` (so alpha
  is preserved)
- **Render** page → Background Alpha = `0`

**Material on Geometry COMP:**
- Drop a **Constant MAT**. Set Color to whatever you want (e.g. white
  `(1,1,1)`). Optionally a **Wireframe MAT** if you want only edges.
- Assign it to the Geometry COMP's Material parameter.

### 3. Composite over webcam

Wire your webcam (Video Device In TOP) and the Render TOP into a
**Composite TOP** with Operation = `Over`. The mesh now sits on top of
the webcam, transparent everywhere it's not drawn.

For a glow effect: feed the Render TOP through a **Blur TOP** + **Level
TOP**, then `Add` over the webcam.

### Color the mesh by region (optional)

The points CHOP carries a `region` channel:
- Hand COMP: 0 = hand
- Face COMP: 1 = skin, 2 = eye_l, 3 = eye_r, 4 = mouth

To color by region, use a **GLSL MAT** and pass `region` as a vertex
attribute via the CHOP-to-SOP. Or split the SOP per region with a
**Group SOP** + per-group MAT. (Ask if you want a recipe for this.)

---

## OSC output (unchanged)

`oscout1` DAT inside each COMP sends every cook:

- **Hand** — `/emgor/hand "<gesture>" <conf> <idx>`
- **Face** — `/emgor/face "<emotion>" <conf>`

`<gesture>` / `<emotion>` = `"none"` when no detection or below threshold.
Configure target IP / port on `oscout1`'s parameters.

You can also read the latest classification programmatically:
```python
op('script1').fetch('classification')
```

---

## Caveats

- **First cook is slow** (~1–2s) while PyTorch loads checkpoints. After
  that, inference runs every cook on Apple MPS / CUDA / CPU.
- **Synchronous inference** caps TD's frame rate to model speed (≈10–25
  fps at 1280×720 on MPS).
- The points CHOP has **variable sample count per cook** — downstream
  CHOPs and the Script SOP handle this fine, but if you connect to
  fixed-size operators, use a **Trim CHOP** or pad samples first.

---

## Troubleshooting

- **`ModuleNotFoundError: combined_app`** — `EMGOR_TD/` isn't next to your
  `.toe`. Either move it, or set env var `EMGOR_RESOURCES`.
- **`ModuleNotFoundError: torch`** — TD's Python module path isn't
  pointing at `EMGOR_TD/venv/lib/python3.11/site-packages`. Or your TD
  Python version differs from 3.11 — re-run `setup.sh` with
  `PYTHON=python3.X ./setup.sh` matching TD's version (Help > About).
- **Empty CHOP / no points** — In TOP `in1` not connected, or webcam not
  streaming. Check `op('in1').width` is non-zero in Textport.
- **No lines rendered** — `mesh_sop` script might not be reading the
  edges DAT. Confirm `EMGOR_HAND/edges` is wired into input 1 (or its
  path is set as the Input2 param).
- **No OSC messages** — `oscout1` DAT name is wrong, or its Network
  Address/Port aren't configured.
