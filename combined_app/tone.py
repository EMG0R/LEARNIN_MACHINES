import numpy as np
import cv2


def build_lut(gamma: float = 1.1) -> np.ndarray:
    """Precompute a uint8 gamma LUT for the mirror tone grade.

    gamma > 1 brightens midtones (standard for mirror-like reflectivity).
    Returns a (256,) uint8 array for use with cv2.LUT.
    """
    table = np.arange(256, dtype=np.float32) / 255.0
    table = np.power(table, 1.0 / gamma) * 255.0
    return np.clip(table, 0, 255).astype(np.uint8)


def apply_grade(frame: np.ndarray, lut: np.ndarray, sat_factor: float = 0.90) -> np.ndarray:
    """Apply mirror color grade: gamma lift + slight desaturation.

    Args:
        frame: BGR uint8 HxWx3
        lut: precomputed gamma LUT from build_lut()
        sat_factor: 1.0 = no change, 0.9 = 10% desaturation

    Returns a new BGR uint8 array.
    """
    graded = cv2.LUT(frame, lut)
    if sat_factor == 1.0:
        return graded
    hsv = cv2.cvtColor(graded, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
