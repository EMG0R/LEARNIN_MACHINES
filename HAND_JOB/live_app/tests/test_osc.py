import sys
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def _result_with_hand():
    mask = np.zeros((720, 1280), dtype=np.uint8)
    cv2.ellipse(mask, (640, 360), (80, 140), 0, 0, 360, 255, -1)
    return {
        "present": True, "fps": 28.5, "mask": mask,
        "gesture": "like", "confidence": 0.84,
        "second": "palm", "second_conf": 0.10, "gesture_idx": 4,
    }

def test_osc_sender_builds():
    from live_app.osc_sender import OSCSender
    sender = OSCSender(ip="127.0.0.1", port=9001)
    assert sender is not None

def test_osc_sends_all_key_addresses(monkeypatch):
    from live_app.osc_sender import OSCSender
    sent = []
    sender = OSCSender(ip="127.0.0.1", port=9001)
    monkeypatch.setattr(sender._client, "send_message",
                        lambda addr, val: sent.append((addr, val)))
    sender.send(_result_with_hand())
    addresses = [a for a, _ in sent]
    for expected in ["/hand/present", "/hand/gesture", "/hand/confidence",
                     "/hand/contour", "/hand/triangles", "/hand/area",
                     "/hand/centroid", "/hand/solidity"]:
        assert expected in addresses, f"Missing: {expected}"

def test_osc_no_hand_sends_only_present_and_fps(monkeypatch):
    from live_app.osc_sender import OSCSender
    sent = []
    sender = OSCSender(ip="127.0.0.1", port=9001)
    monkeypatch.setattr(sender._client, "send_message",
                        lambda addr, val: sent.append((addr, val)))
    sender.send({"present": False, "fps": 28.0, "mask": None,
                 "gesture": None, "confidence": 0.0,
                 "second": None, "second_conf": 0.0, "gesture_idx": 0})
    addresses = [a for a, _ in sent]
    assert "/hand/present" in addresses
    assert "/hand/gesture" not in addresses
    assert "/hand/contour" not in addresses
