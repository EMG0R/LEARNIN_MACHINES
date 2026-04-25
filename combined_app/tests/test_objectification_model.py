import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from combined_app.objectification_model import load_objectification, run_objectification


def test_load_returns_model_or_none():
    model = load_objectification()
    # either loads successfully or returns None — never raises
    assert model is None or hasattr(model, "forward")


def test_run_with_none_model():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = run_objectification(frame, None)
    assert result["enabled"] is False
    assert result["class_map"] == {}
    assert result["person_present"] is False


def test_run_shape_with_model():
    model = load_objectification()
    if model is None:
        pytest.skip("No OBJECTIFICATION checkpoint available")
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    result = run_objectification(frame, model)
    assert result["enabled"] is True
    assert "class_map" in result
    assert isinstance(result["person_present"], bool)


def test_run_class_map_keys_are_strings():
    model = load_objectification()
    if model is None:
        pytest.skip("No OBJECTIFICATION checkpoint available")
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = run_objectification(frame, model)
    for k in result["class_map"]:
        assert isinstance(k, str)
