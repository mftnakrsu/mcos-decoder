"""Low-level MAT5 element-stream walker.

Reads the binary element tree inside ``__function_workspace__`` of a MATLAB
v5 .mat file. Each element has an 8-byte tag (type, size) and a payload;
miMATRIX (type 14) elements contain nested streams. We yield typed
:class:`MatElement` records so callers can inspect the tree without writing
the byte-level loop themselves.
"""
from __future__ import annotations
import struct
from dataclasses import dataclass
from typing import Iterator

import numpy as np

# MAT5 element type IDs
TYPE_NAMES = {
    1: "miINT8", 2: "miUINT8", 3: "miINT16", 4: "miUINT16",
    5: "miINT32", 6: "miUINT32", 7: "miSINGLE", 9: "miDOUBLE",
    12: "miINT64", 13: "miUINT64", 14: "miMATRIX", 15: "miCOMPRESSED",
    16: "miUTF8", 17: "miUTF16", 18: "miUTF32",
}
NUMERIC_DTYPES = {
    1: "<i1", 2: "<u1", 3: "<i2", 4: "<u2",
    5: "<i4", 6: "<u4", 7: "<f4", 9: "<f8",
    12: "<i8", 13: "<u8",
}


@dataclass
class MatElement:
    """A single decoded MAT5 element from the walked stream."""
    type_id: int
    type_name: str
    offset: int        # offset of element header inside its parent buffer
    payload_size: int  # payload bytes (excludes header + padding)
    payload: bytes     # raw payload (or empty for miMATRIX, see ``children``)
    children: tuple = ()

    def as_array(self):
        """Return the payload as a NumPy array if numeric, else None."""
        if self.type_id in NUMERIC_DTYPES:
            return np.frombuffer(self.payload, dtype=NUMERIC_DTYPES[self.type_id])
        return None

    def as_string(self) -> str | None:
        """Return the payload decoded as UTF-8 (None if not a string type)."""
        if self.type_id in (1, 2, 16):
            try:
                return self.payload.decode("utf-8", errors="replace")
            except UnicodeDecodeError:
                return None
        return None


class MatStream:
    """Walker over the MAT5 element tree.

    Usage::

        s = MatStream(buf)               # buf = bytes from __function_workspace__
        for elem in s.walk():
            ...                          # depth-first traversal

        # Or filter while walking
        for elem in s.walk():
            if elem.type_id == 9 and len(elem.payload) == 32:
                bbox = elem.as_array()   # length-4 double — likely bbox
    """

    def __init__(self, buf: bytes, *, skip_header: int = 8, max_depth: int = 12):
        """Initialise.

        Parameters
        ----------
        buf : bytes
            Raw ``__function_workspace__`` payload from ``scipy.io.loadmat``.
        skip_header : int, default 8
            Bytes to skip at the start of ``buf``. The standard
            ``__function_workspace__`` payload has an 8-byte preamble before
            the first element.
        max_depth : int, default 12
            Maximum recursion depth.
        """
        self.buf = buf[skip_header:]
        self.max_depth = max_depth

    def walk(self) -> Iterator[MatElement]:
        """Yield every element in the tree (depth-first, parent before children)."""
        yield from self._walk(self.buf, depth=0)

    def _walk(self, buf: bytes, depth: int) -> Iterator[MatElement]:
        if depth > self.max_depth:
            return
        off = 0
        n = len(buf)
        while off + 8 <= n:
            small_type = struct.unpack_from("<H", buf, off)[0]
            small_size = struct.unpack_from("<H", buf, off + 2)[0]
            if small_size != 0 and small_type < 20:
                t, sz = small_type, small_size
                payload_off = off + 4
            else:
                try:
                    t, sz = struct.unpack_from("<II", buf, off)
                except struct.error:
                    return
                payload_off = off + 8
            if t == 0 or t > 20:
                return
            if payload_off + sz > n:
                return
            payload = buf[payload_off:payload_off + sz]
            children = ()
            if t == 14:  # miMATRIX — recurse
                children = tuple(self._walk(payload, depth + 1))
                # A miMATRIX element's payload is itself a stream of elements;
                # don't expose the raw bytes (children carries the decoded form)
                payload = b""
            yield MatElement(
                type_id=t,
                type_name=TYPE_NAMES.get(t, f"miUNKNOWN({t})"),
                offset=off,
                payload_size=sz,
                payload=payload,
                children=children,
            )
            # Pad to 8-byte alignment after data
            next_off = payload_off + sz
            next_off += (8 - (next_off % 8)) % 8
            if next_off <= off:
                return
            off = next_off
