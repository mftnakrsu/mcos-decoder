"""High-level extractors for MATLAB CV Toolbox groundTruth objects.

A `groundTruth` object stores per-frame label cells in a table-like
structure. Empirically (verified across 365 IR videos in the Halmstad
Drone Detection Dataset, plus several Audio Labeler exports), each
per-frame cell at MAT5 depth 5 has one of three sizes:

  * ``size == 80``  → bbox cell. Contains a length-4 double array.
  * ``size == 112`` → empty marker. Target absent that frame.
  * ``size == 72``  → duration/timestamp metadata. Skip.

The decoder yields a list whose length equals the number of label slots
(typically the video frame count) with ``None`` for empty frames and a
``(x, y, w, h)`` tuple for filled frames.
"""
from __future__ import annotations
import scipy.io
from pathlib import Path
from typing import Optional

from .stream import MatStream, MatElement


def _find_first_4double(elem: MatElement):
    """Recursively find a length-4 double array inside a MatElement subtree."""
    if elem.type_id == 9 and elem.payload_size == 32:
        arr = elem.as_array()
        if arr is not None and len(arr) == 4:
            return tuple(float(x) for x in arr)
    for c in elem.children:
        r = _find_first_4double(c)
        if r is not None:
            return r
    return None


def _walk_frames(elem: MatElement, depth: int, out: list,
                 frame_depth: int, bbox_size: int, empty_size: int):
    """Walk a MatElement tree, append per-frame bbox/None to ``out``."""
    if elem.type_id == 14 and depth == frame_depth:
        if elem.payload_size == bbox_size:
            out.append(_find_first_4double(elem))
            return
        if elem.payload_size == empty_size:
            out.append(None)
            return
    for c in elem.children:
        _walk_frames(c, depth + 1, out, frame_depth, bbox_size, empty_size)


def extract_bboxes(fw_buf: bytes, *,
                   frame_depth: int = 5,
                   bbox_size: int = 80,
                   empty_size: int = 112) -> list[Optional[tuple]]:
    """Walk a ``__function_workspace__`` buffer and return per-frame bboxes.

    Parameters
    ----------
    fw_buf : bytes
        The raw ``__function_workspace__`` payload from
        ``scipy.io.loadmat``. The 8-byte preamble is stripped automatically.
    frame_depth : int, default 5
        MAT5 nesting depth at which per-frame slots live. The default
        works for groundTruth objects produced by MATLAB R2020a-R2023a.
        Increase if your MATLAB version nests deeper.
    bbox_size : int, default 80
        Size (in bytes) of a *filled* per-frame label cell.
    empty_size : int, default 112
        Size (in bytes) of an *empty* per-frame label cell.

    Returns
    -------
    list[tuple | None]
        One entry per frame slot. ``None`` if the target was absent that
        frame, or ``(x, y, w, h)`` (top-left + size in pixels) otherwise.
    """
    stream = MatStream(fw_buf, skip_header=8)
    out: list = []
    for top in stream.walk():
        _walk_frames(top, 0, out, frame_depth, bbox_size, empty_size)
    return out


def load_groundtruth(mat_path: str | Path, **kwargs) -> list[Optional[tuple]]:
    """Load a MATLAB groundTruth/labelData .mat and return per-frame bboxes.

    Convenience wrapper around :func:`extract_bboxes`.

    Parameters
    ----------
    mat_path : str or Path
        Path to a MATLAB v5 .mat file containing a single ``groundTruth``
        (or similar MCOS object) variable.
    **kwargs
        Forwarded to :func:`extract_bboxes` (``frame_depth``, ``bbox_size``,
        ``empty_size``).
    """
    raw = scipy.io.loadmat(str(mat_path))
    fw = raw.get("__function_workspace__")
    if fw is None:
        raise ValueError(
            f"{mat_path} has no __function_workspace__; not an MCOS file."
        )
    fw_bytes = fw.tobytes() if hasattr(fw, "tobytes") else bytes(fw[0])
    return extract_bboxes(fw_bytes, **kwargs)
