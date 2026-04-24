import cv2
import numpy as np
from scipy.spatial import Delaunay

FILL_OPACITY    = 0.55
MESH_OPACITY    = 0.45
GLOW_WIDTH      = 5
CONTOUR_POINTS  = 14
INTERIOR_POINTS = 5
NETWORK_DIST    = 160
NETWORK_MAX     = 20
BLOB_MIN_AREA   = 800

MESH_BASE_W    = 1
NET_BASE_W     = 1
MAX_THICK_MULT = 3.9   # generative variation reaches up to ~1.3x base

FACE_CONTOUR_PTS  = 24
FACE_INTERIOR_PTS = 6


# ── Color ─────────────────────────────────────────────────────────────────────

def valence_to_color(valence: float) -> tuple:
    """[-1,1] → BGR: red at -1, silver at 0, blue at +1."""
    v = max(-1.0, min(1.0, valence))
    neutral = (160, 160, 160)
    if v >= 0:
        r = int(neutral[0] * (1 - v) + 220 * v)
        g = int(neutral[1] * (1 - v) + 80  * v)
        b = int(neutral[2] * (1 - v) + 20  * v)
    else:
        v = -v
        r = int(neutral[0] * (1 - v) + 20  * v)
        g = int(neutral[1] * (1 - v) + 20  * v)
        b = int(neutral[2] * (1 - v) + 220 * v)
    return (r, g, b)   # BGR


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sample_contour(mask, n):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours: return None
    cnt = max(contours, key=cv2.contourArea).squeeze()
    if cnt.ndim < 2 or len(cnt) < 3: return None
    idx = np.linspace(0, len(cnt) - 1, n, dtype=int)
    return cnt[idx]


def _interior_points(mask, n):
    ys, xs = np.where(mask > 0)
    if len(xs) < n: return np.empty((0, 2), dtype=int)
    cx, cy = int(xs.mean()), int(ys.mean())
    idx = np.random.choice(len(xs), size=min(n, len(xs)), replace=False)
    pts = np.stack([xs[idx], ys[idx]], axis=1)
    return np.vstack([pts, [[cx, cy]]])


def _dim(color, f):
    return tuple(min(255, int(c * f)) for c in color)


def _edge_thickness(p1, p2, dist_map, base):
    if dist_map is None: return base
    mx = int((p1[0] + p2[0]) / 2); my = int((p1[1] + p2[1]) / 2)
    h, w = dist_map.shape
    if not (0 <= my < h and 0 <= mx < w): return base
    max_d = dist_map.max()
    if max_d == 0: return base
    t = float(dist_map[my, mx]) / max_d
    return max(1, int(base * (1.0 + (MAX_THICK_MULT - 1.0) * t)))


def _bloom(canvas, color, mask, fade):
    glow = np.zeros_like(canvas)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return canvas
    cv2.drawContours(glow, contours, -1, color, -1)
    glow = cv2.GaussianBlur(glow, (61, 61), 0)
    return cv2.addWeighted(canvas, 1.0, glow, 0.35 * fade, 0)


def label_anchor_raw(mask: np.ndarray, frame_h: int, frame_w: int):
    """
    Raw (x, y) for label: finds the largest connected component in the mask,
    uses its centroid, then offsets 15% of width right and well above it.
    """
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    # pick largest component (skip background label 0)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cx = int(centroids[best][0])
    cy = int(centroids[best][1])
    # walk up from centroid to find top of that component
    col = min(cx, frame_w - 1)
    y = cy
    while y > 0 and labels[y, col] == best:
        y -= 1
    y = max(22, y - int(frame_h * 0.10))
    x = max(4, min(frame_w - 120, cx - 50))
    return (x, y)


# ── Mesh draw ─────────────────────────────────────────────────────────────────

def draw_mesh(frame: np.ndarray, mask: np.ndarray, color: tuple,
              fade: float, feedback_bufs: dict, subject_id: str,
              extra_pts: np.ndarray | None = None,
              lines_only: bool = False,
              n_contour: int = CONTOUR_POINTS,
              n_interior: int = INTERIOR_POINTS) -> np.ndarray:
    """
    Draw a mesh over `mask` in `color`.
    `feedback_bufs` is a shared dict; `subject_id` namespaces the trail buffer.
    `extra_pts` are additional mesh nodes (e.g. eye/mouth centroids for faces).
    """
    out = frame.copy()
    if fade < 0.01 or mask is None or not mask.any():
        feedback_bufs.pop(subject_id, None)
        return out

    h, w = frame.shape[:2]

    if not lines_only:
        # feedback trail
        fb = feedback_bufs.get(subject_id)
        if fb is not None and fb.shape == (h, w, 3):
            M = cv2.getRotationMatrix2D((w / 2, h / 2), 0.0, 0.97)
            warped = cv2.warpAffine(fb, M, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT)
            out = cv2.addWeighted(out, 1.0, warped, 0.35 * fade, 0)

        out = _bloom(out, _dim(color, 0.7), mask, fade)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            fill = out.copy()
            cv2.drawContours(fill, contours, -1, color, -1)
            cv2.addWeighted(fill, FILL_OPACITY * fade, out, 1 - FILL_OPACITY * fade, 0, out)
    else:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    contour_pts = _sample_contour(mask, n_contour)
    if contour_pts is not None and len(contour_pts) >= 3:
        interior_pts = _interior_points(mask, n_interior)
        all_pts = np.vstack([contour_pts, interior_pts]) if len(interior_pts) else contour_pts
        if extra_pts is not None and len(extra_pts):
            all_pts = np.vstack([all_pts, extra_pts.astype(int)])

        try:
            tri = Delaunay(all_pts)
            fill_layer = out.copy(); mesh_layer = out.copy()
            for simplex in tri.simplices:
                p1=tuple(all_pts[simplex[0]]); p2=tuple(all_pts[simplex[1]]); p3=tuple(all_pts[simplex[2]])
                cx_=int((p1[0]+p2[0]+p3[0])//3); cy_=int((p1[1]+p2[1]+p3[1])//3)
                if not (0<=cy_<h and 0<=cx_<w): continue
                mids=[(int((p1[0]+p2[0])//2),int((p1[1]+p2[1])//2)),
                      (int((p2[0]+p3[0])//2),int((p2[1]+p3[1])//2)),
                      (int((p3[0]+p1[0])//2),int((p3[1]+p1[1])//2))]
                if mask[cy_,cx_]>0 and all(0<=my<h and 0<=mx<w and mask[my,mx]>0 for mx,my in mids):
                    cv2.fillPoly(fill_layer,[np.array([p1,p2,p3],dtype=np.int32)],color)
                    cv2.line(mesh_layer,p1,p2,(255,255,255),_edge_thickness(p1,p2,dist_map,MESH_BASE_W))
                    cv2.line(mesh_layer,p2,p3,(255,255,255),_edge_thickness(p2,p3,dist_map,MESH_BASE_W))
                    cv2.line(mesh_layer,p3,p1,(255,255,255),_edge_thickness(p3,p1,dist_map,MESH_BASE_W))
            cv2.addWeighted(fill_layer, 0.22*fade, out, 1-0.22*fade, 0, out)
            cv2.addWeighted(mesh_layer, MESH_OPACITY*fade, out, 1-MESH_OPACITY*fade, 0, out)
        except Exception:
            pass

        pts_f = all_pts.astype(np.float32)
        pairs = [(float(np.linalg.norm(pts_f[i]-pts_f[j])), i, j)
                 for i in range(len(pts_f)) for j in range(i+1,len(pts_f))
                 if np.linalg.norm(pts_f[i]-pts_f[j]) < NETWORK_DIST]
        pairs.sort(key=lambda t: -t[0])
        net_layer = out.copy()
        for d, i, j in pairs[:NETWORK_MAX]:
            brightness = 0.3 + 0.5 * (d / NETWORK_DIST)
            c = tuple(min(255, int(255*brightness)) for _ in range(3))
            p1, p2 = tuple(all_pts[i]), tuple(all_pts[j])
            cv2.line(net_layer, p1, p2, c, _edge_thickness(p1,p2,dist_map,NET_BASE_W))
        cv2.addWeighted(net_layer, 0.30*fade, out, 1-0.30*fade, 0, out)

        if not lines_only:
            node_layer = out.copy()
            for pt in all_pts:
                x_, y_ = int(pt[0]), int(pt[1])
                if 0<=x_<w and 0<=y_<h:
                    cv2.circle(node_layer, (x_,y_), 3, color, -1)
                    cv2.circle(node_layer, (x_,y_), 1, (255,255,255), -1)
            cv2.addWeighted(node_layer, 0.55*fade, out, 1-0.55*fade, 0, out)

    if not lines_only and contours:
        max_d = float(dist_map.max()) if dist_map is not None else 0.0
        depth_t = min(1.0, max_d / (max(frame.shape[:2]) * 0.15))
        gw = max(GLOW_WIDTH, int(GLOW_WIDTH*(1+(MAX_THICK_MULT-1)*depth_t)))
        gl = out.copy()
        cv2.drawContours(gl, contours, -1, color, gw*2)
        cv2.addWeighted(gl, 0.6*fade, out, 1-0.6*fade, 0, out)

    if not lines_only:
        feedback_bufs[subject_id] = cv2.subtract(out, frame)
    return out


def draw_face_aura(frame: np.ndarray, skin_mask: np.ndarray, fade: float) -> np.ndarray:
    """Soft dark-purple bloom around the face region."""
    if not skin_mask.any():
        return frame
    PURPLE = (90, 30, 0)   # BGR dark blue
    glow = np.zeros_like(frame)
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return frame
    cv2.drawContours(glow, contours, -1, PURPLE, -1)
    glow = cv2.GaussianBlur(glow, (101, 101), 0)
    return cv2.addWeighted(frame, 1.0, glow, 0.28 * fade, 0)


def draw_label(frame: np.ndarray, pos: tuple, text: str,
               color: tuple, conf: float) -> np.ndarray:
    """Draw a floating label at smoothed `pos` = (x, y)."""
    if pos is None:
        return frame
    out = frame.copy()
    font  = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.50
    thick = 1
    label = text.upper().replace("_", " ")
    (lw, lh), _ = cv2.getTextSize(label, font, scale, thick)
    x, y = int(pos[0]), int(pos[1])
    pad = 4
    cv2.rectangle(out, (x-pad, y-lh-pad), (x+lw+pad, y+pad), (0, 0, 0), -1)
    cv2.putText(out, label, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def draw_blobs(frame: np.ndarray, fg_mask: np.ndarray,
               exclude_masks: list[np.ndarray]) -> np.ndarray:
    """Light mesh over moving non-subject blobs."""
    if fg_mask is None: return frame
    out = frame.copy()
    h, w = frame.shape[:2]
    other = fg_mask.copy()
    for m in exclude_masks:
        if m is not None: other[m > 0] = 0
    contours, _ = cv2.findContours(other, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < BLOB_MIN_AREA: continue
        sq = cnt.squeeze()
        if sq.ndim < 2 or len(sq) < 3: continue
        n = min(12, len(sq))
        pts = sq[np.linspace(0, len(sq)-1, n, dtype=int)]
        try:
            tri = Delaunay(pts); layer = out.copy()
            for simplex in tri.simplices:
                p1,p2,p3 = tuple(pts[simplex[0]]),tuple(pts[simplex[1]]),tuple(pts[simplex[2]])
                cv2.line(layer,p1,p2,(200,200,200),1); cv2.line(layer,p2,p3,(200,200,200),1)
                cv2.line(layer,p3,p1,(200,200,200),1)
            cv2.addWeighted(layer, 0.18, out, 0.82, 0, out)
        except Exception:
            pass
    return out
