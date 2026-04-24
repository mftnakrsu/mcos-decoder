# mcos-decoder

[![PyPI](https://img.shields.io/pypi/v/mcos-decoder.svg)](https://pypi.org/project/mcos-decoder/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Decode MATLAB MCOS objects (Computer Vision Toolbox `groundTruth`,
Audio/Signal Labeler exports, etc.) from `.mat` files in pure Python — no
MATLAB license required.**

## Why this exists

MATLAB's Computer Vision Toolbox saves video annotations as a
`groundTruth` MCOS object. When loaded with `scipy.io.loadmat` (or
`pymatreader`, `mat73`), the actual labels are hidden inside an opaque
`__function_workspace__` blob:

```python
>>> import scipy.io
>>> m = scipy.io.loadmat("IR_DRONE_001_LABELS.mat")
>>> m["None"]
array([(b'gTruth', b'MCOS', b'groundTruth', array([[3707764736], ...])), ...])
# ↑ no label data accessible
```

This problem has been [open in scipy since 2024](https://github.com/scipy/scipy/issues/22736)
and is the subject of multiple unanswered help threads
(e.g. [DroneDetectionThesis Issue #3](https://github.com/DroneDetectionThesis/Drone-detection-dataset/issues/3)).
Until now the only third-party reader was the C# library
[MatFileHandler](https://github.com/mahalex/MatFileHandler).

`mcos-decoder` fills the gap for Python users by walking the
`__function_workspace__` MAT5 element tree and extracting per-frame label
cells.

## Install

```bash
pip install mcos-decoder
```

## Quick start

```python
from mcos_decoder import load_groundtruth

bboxes = load_groundtruth("IR_DRONE_001_LABELS.mat")
# → list[tuple | None] of length n_frames
# Each entry is (x, y, w, h) for filled frames, or None when target absent.

for frame_idx, bbox in enumerate(bboxes, start=1):
    if bbox is None:
        print(f"Frame {frame_idx}: target absent")
    else:
        x, y, w, h = bbox
        print(f"Frame {frame_idx}: bbox=({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f})")
```

## Lower-level API

If your `.mat` uses a non-default MCOS layout (e.g. older MATLAB
versions), you can override the depth and slot sizes:

```python
from mcos_decoder import load_groundtruth

# Audio Labeler often nests one level deeper
bboxes = load_groundtruth(
    "audio_labels.mat",
    frame_depth=6,      # default 5
    bbox_size=80,       # default 80 (filled cell size in bytes)
    empty_size=112,     # default 112 (empty marker size)
)
```

For raw element-tree access:

```python
import scipy.io
from mcos_decoder import MatStream

m = scipy.io.loadmat("labels.mat")
fw = m["__function_workspace__"].tobytes()
stream = MatStream(fw, skip_header=8)

for elem in stream.walk():
    if elem.type_name == "miMATRIX":
        print(f"depth-1 matrix at offset {elem.offset:#x}, "
              f"size {elem.payload_size}, "
              f"children: {len(elem.children)}")
```

## Verified compatibility

| Source | MATLAB version | Status |
|---|---|---|
| Halmstad Drone Detection Dataset (Svanström 2021) | R2020a | ✓ 365 IR videos, 4 classes |
| Anti-UAV410 retrained tracker results | — | N/A (uses txt) |

If you successfully use the package on another dataset, please open a
PR adding it to this table.

## Limitations

- **Single-target per cell.** The current decoder returns one bbox per
  frame; if your `.mat` has multiple bboxes per frame (e.g. MOT-style
  multi-object labels), the first one is returned. Multi-target support
  is on the roadmap.
- **Numeric bbox only.** Categorical labels are not yet extracted.
- **`groundTruth` only.** Other MCOS types (e.g. `videoLabeler`) may
  require different `frame_depth` / size parameters.

## How it works

The MCOS object data is stored in `__function_workspace__` as a recursive
MAT5 element stream. Each element has an 8-byte tag (type, size); type 14
(`miMATRIX`) elements contain nested streams. Per-frame label cells are
serialized at a fixed nesting depth (5 in current MATLAB versions) with
size 80 (filled) or 112 (empty marker). The decoder walks the tree
breadth-first and extracts the bbox doubles from the filled cells.

See [`mcos_decoder/groundtruth.py`](mcos_decoder/groundtruth.py) for the
full implementation (~70 lines).

## Citation

If this package supports your research, please cite:

```bibtex
@software{ozdemir2026mcos,
  title  = {mcos-decoder: A Python decoder for MATLAB MCOS objects},
  author = {Ozdemir, Burak},
  year   = {2026},
  url    = {https://github.com/bozdemir/mcos-decoder}
}
```

A short companion arXiv preprint is forthcoming.

## License

MIT — see [LICENSE](LICENSE).
