import cv2
import numpy as np
from scipy.spatial import Delaunay
from pythonosc.udp_client import SimpleUDPClient
from live_app.config import OSC_IP, OSC_PORT, CONF_THRESHOLD
from live_app.renderer import CONTOUR_POINTS


class OSCSender:
    def __init__(self, ip: str = OSC_IP, port: int = OSC_PORT):
        self._client = SimpleUDPClient(ip, port)
        self._prev_centroid = None

    def send(self, result: dict):
        present = result.get("present", False)
        self._client.send_message("/hand/present", int(present))
        self._client.send_message("/hand/fps",     float(result.get("fps", 0.0)))

        if not present:
            self._prev_centroid = None
            return

        mask = result.get("mask")
        if mask is not None:
            self._send_mask_data(mask)
        self._send_gesture_data(result)

    def _send_mask_data(self, mask: np.ndarray):
        h, w = mask.shape
        area = float(np.count_nonzero(mask)) / (h * w)
        self._client.send_message("/hand/area", area)

        coords = cv2.findNonZero(mask)
        if coords is None:
            return
        bx, by, bw, bh = cv2.boundingRect(coords)
        self._client.send_message("/hand/bbox", [bx/w, by/h, bw/w, bh/h])
        self._client.send_message("/hand/aspect_ratio",
                                  float(bw / bh) if bh > 0 else 0.0)

        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"] / w
            cy = M["m01"] / M["m00"] / h
            self._client.send_message("/hand/centroid", [cx, cy])
            if self._prev_centroid is not None:
                dx = cx - self._prev_centroid[0]
                dy = cy - self._prev_centroid[1]
                self._client.send_message("/hand/velocity", [float(dx), float(dy)])
                self._client.send_message("/hand/speed",
                                          float((dx**2 + dy**2) ** 0.5))
            else:
                self._client.send_message("/hand/velocity", [0.0, 0.0])
                self._client.send_message("/hand/speed", 0.0)
            self._prev_centroid = (cx, cy)

            denom = M["mu20"] - M["mu02"]
            if abs(denom) > 0 or abs(M["mu11"]) > 0:
                angle = 0.5 * np.degrees(np.arctan2(2 * M["mu11"], denom))
                self._client.send_message("/hand/orientation", float(angle))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        cnt = max(contours, key=cv2.contourArea)
        hull      = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            self._client.send_message("/hand/solidity",
                                      float(cv2.contourArea(cnt) / hull_area))

        full = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)[0]
        if not full:
            return
        full_pts = max(full, key=cv2.contourArea).squeeze()
        if full_pts.ndim < 2 or len(full_pts) < 3:
            return
        indices = np.linspace(0, len(full_pts) - 1, CONTOUR_POINTS, dtype=int)
        sampled = full_pts[indices]

        flat_contour = []
        for pt in sampled:
            flat_contour.extend([float(pt[0] / w), float(pt[1] / h)])
        self._client.send_message("/hand/contour", flat_contour)

        tri      = Delaunay(sampled)
        flat_tri = []
        for simplex in tri.simplices:
            p1, p2, p3 = sampled[simplex[0]], sampled[simplex[1]], sampled[simplex[2]]
            cx_t = int((p1[0] + p2[0] + p3[0]) // 3)
            cy_t = int((p1[1] + p2[1] + p3[1]) // 3)
            if 0 <= cy_t < h and 0 <= cx_t < w and mask[cy_t, cx_t] > 0:
                for pt in (p1, p2, p3):
                    flat_tri.extend([float(pt[0] / w), float(pt[1] / h)])
        self._client.send_message("/hand/triangle_count", len(flat_tri) // 6)
        self._client.send_message("/hand/triangles", flat_tri)

    def _send_gesture_data(self, result: dict):
        gesture = result.get("gesture")
        conf    = result.get("confidence", 0.0)
        if gesture is None or conf < CONF_THRESHOLD:
            return
        self._client.send_message("/hand/gesture",            gesture)
        self._client.send_message("/hand/confidence",         float(conf))
        self._client.send_message("/hand/gesture/confidence", float(conf))
        second = result.get("second")
        if second:
            self._client.send_message("/hand/gesture/second",      second)
            self._client.send_message("/hand/gesture/second_conf",
                                      float(result.get("second_conf", 0.0)))
