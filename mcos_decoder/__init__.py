"""mcos_decoder — Decode MATLAB MCOS objects from .mat ``__function_workspace__``.

Most MATLAB Computer Vision Toolbox / Audio Labeler / Signal Labeler outputs
are stored as MCOS (MATLAB Class Object Storage) objects which standard
Python loaders (``scipy.io.loadmat``, ``mat73``, ``pymatreader``) expose only
as opaque blobs. This package walks the hidden ``__function_workspace__``
MAT5 stream and extracts the per-frame label cells without requiring MATLAB.

Public API
----------
- :func:`load_groundtruth` — high-level helper for groundTruth / labelData
- :class:`MatStream` — generic walker over MAT5 element trees
- :func:`extract_bboxes` — collect all length-4 double arrays (bbox shaped)

See :mod:`mcos_decoder.groundtruth` for label-aware extractors.
"""
from .stream import MatStream, MatElement
from .groundtruth import load_groundtruth, extract_bboxes

__version__ = "0.1.0"
__all__ = [
    "MatStream",
    "MatElement",
    "load_groundtruth",
    "extract_bboxes",
    "__version__",
]
