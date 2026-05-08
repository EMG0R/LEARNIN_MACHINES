"""Mesh extraction — same point sampling + Delaunay as renderer.draw_mesh,
but returns (points, edges) instead of drawing pixels.

Used by td_hand.py / td_face.py to emit point/edge data into TD without
ever building a raster image.
"""
import numpy as np
import cv2
from scipy.spatial import Delaunay
from .renderer import _sample_contour, _interior_points, _stable_interior_points


def extract_mesh(mask: np.ndarray, region_id: int,
                 n_contour: int, n_interior: int,
                 stable_interior: bool = False):
    """Returns (points_xy_region, edges_ab) for the largest blob in `mask`.

    points: list of (x_pixel, y_pixel, region_id_float)
    edges:  list of (a_idx, b_idx)
    """
    if mask is None or not mask.any():
        return [], []

    contour_pts = _sample_contour(mask, n_contour)
    if contour_pts is None or len(contour_pts) < 3:
        return [], []

    if stable_interior:
        dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        interior_pts = _stable_interior_points(dist_map, n_interior)
    else:
        interior_pts = _interior_points(mask, n_interior)

    all_pts = np.vstack([contour_pts, interior_pts]) if len(interior_pts) else contour_pts
    if len(all_pts) < 3:
        return [], []

    try:
        tri = Delaunay(all_pts)
    except Exception:
        return [], []

    h, w = mask.shape
    valid_simplices = []
    for s in tri.simplices:
        p1, p2, p3 = all_pts[s[0]], all_pts[s[1]], all_pts[s[2]]
        cx = int((p1[0] + p2[0] + p3[0]) // 3)
        cy = int((p1[1] + p2[1] + p3[1]) // 3)
        if not (0 <= cy < h and 0 <= cx < w):
            continue
        if mask[cy, cx] > 0:
            valid_simplices.append(s)

    edges = set()
    for s in valid_simplices:
        for a, b in ((s[0], s[1]), (s[1], s[2]), (s[2], s[0])):
            edges.add((min(int(a), int(b)), max(int(a), int(b))))

    points = [(float(p[0]), float(p[1]), float(region_id)) for p in all_pts]
    return points, sorted(edges)
