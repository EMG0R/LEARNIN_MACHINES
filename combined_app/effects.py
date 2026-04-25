import numpy as np
import cv2


class FlowField:
    """Ambient generative background — slow numpy sine-wave flow field.

    Generates a BGR uint8 buffer each tick via two overlapping sine fields
    driving hue and luminance. No per-pixel Python loops — fully vectorized.
    Target cost: ≤5ms per tick on Pi ARM at 1280×720.
    """

    def __init__(self, height: int, width: int, opacity: float = 0.25, step: float = 0.005):
        self.h = height
        self.w = width
        self.opacity = opacity
        self.step = step
        self.t = 0.0
        self.buffer = np.zeros((height, width, 3), dtype=np.uint8)

        yy, xx = np.mgrid[0:height, 0:width]
        self.xn = (xx / width).astype(np.float32)
        self.yn = (yy / height).astype(np.float32)

    def tick(self) -> None:
        """Advance one frame. Updates self.buffer in place."""
        t = self.t
        f1 = np.sin(self.xn * 4.0 + t) * np.cos(self.yn * 3.0 + t * 0.7)
        f2 = np.sin(self.xn * 2.5 + t * 0.5) * np.cos(self.yn * 5.0 + t * 1.3)
        hue = ((f1 + f2 + 2.0) / 4.0 * 179).astype(np.uint8)
        sat = np.full((self.h, self.w), 200, dtype=np.uint8)
        val = ((np.abs(f1) * 0.6 + 0.2) * 255).clip(0, 255).astype(np.uint8)
        self.buffer = cv2.cvtColor(np.stack([hue, sat, val], axis=2), cv2.COLOR_HSV2BGR)
        self.t += self.step

    def blend_onto(self, frame: np.ndarray) -> np.ndarray:
        """Additively composite effects buffer onto frame.

        Dark areas of frame let effects show through; bright areas dominate.
        Returns a new uint8 BGR array — does not modify frame in place.
        """
        return cv2.addWeighted(frame, 1.0, self.buffer, self.opacity, 0)
