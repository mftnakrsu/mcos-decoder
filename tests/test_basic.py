"""Smoke tests for mcos_decoder.

These tests run only if the Halmstad Drone Detection Dataset is available
locally (set MCOS_TEST_DATA env var to point at the Video_IR folder).
Otherwise tests skip so the package can still pass CI without the dataset.
"""
import os
from pathlib import Path
import pytest

from mcos_decoder import load_groundtruth, MatStream
from mcos_decoder.stream import MatElement

DATA = os.environ.get(
    "MCOS_TEST_DATA",
    "/home/burak/ir-tracker/publication/drone_detection_dataset/"
    "Drone-detection-dataset-master/Data/Video_IR",
)
HAS_DATA = Path(DATA).exists()


@pytest.mark.skipif(not HAS_DATA, reason="MCOS_TEST_DATA not available")
def test_drone_001_full_visibility():
    bb = load_groundtruth(Path(DATA) / "IR_DRONE_001_LABELS.mat")
    assert len(bb) == 301              # frame count
    n_visible = sum(1 for x in bb if x is not None)
    assert n_visible == 301             # target visible in every frame
    # Each bbox is a length-4 tuple of floats
    for box in bb:
        assert isinstance(box, tuple) and len(box) == 4
        x, y, w, h = box
        assert 0 < x < 320 and 0 < y < 256
        assert 0 < w < 320 and 0 < h < 256


@pytest.mark.skipif(not HAS_DATA, reason="MCOS_TEST_DATA not available")
def test_bird_001_partial_visibility():
    bb = load_groundtruth(Path(DATA) / "IR_BIRD_001_LABELS.mat")
    # 310 video frames; decoder may return one extra metadata slot
    assert 309 <= len(bb) <= 311
    n_visible = sum(1 for x in bb if x is not None)
    assert 200 < n_visible < 280        # ~235 visible (~76%)
    n_absent = sum(1 for x in bb if x is None)
    assert n_absent + n_visible == len(bb)


@pytest.mark.skipif(not HAS_DATA, reason="MCOS_TEST_DATA not available")
def test_all_classes_parse():
    """All four target classes should parse without error."""
    for cls in ("AIRPLANE", "BIRD", "DRONE", "HELICOPTER"):
        path = Path(DATA) / f"IR_{cls}_001_LABELS.mat"
        bb = load_groundtruth(path)
        assert len(bb) > 0


@pytest.mark.skipif(not HAS_DATA, reason="MCOS_TEST_DATA not available")
def test_stream_walks_without_error():
    """Generic MatStream walker should not crash on any sample file."""
    import scipy.io
    path = Path(DATA) / "IR_DRONE_001_LABELS.mat"
    m = scipy.io.loadmat(str(path))
    fw = m["__function_workspace__"].tobytes()
    stream = MatStream(fw, skip_header=8)
    elems = list(stream.walk())
    assert len(elems) > 0
    # Should find at least one miMATRIX at the top
    assert any(e.type_name == "miMATRIX" for e in elems)


def test_mat_element_dataclass():
    """MatElement basics work without data."""
    el = MatElement(type_id=9, type_name="miDOUBLE", offset=0,
                    payload_size=32, payload=b"\x00" * 32)
    arr = el.as_array()
    assert arr is not None
    assert len(arr) == 4


def test_string_decode():
    el = MatElement(type_id=1, type_name="miINT8", offset=0,
                    payload_size=5, payload=b"hello")
    assert el.as_string() == "hello"
