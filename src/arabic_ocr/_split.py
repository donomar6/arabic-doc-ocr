"""Split very wide text-line crops at whitespace gaps.

The recognisers are trained on crops around 320x48 (aspect ratio ~7). Lines
with a ratio above ~12 lose characters, typically digits and Latin runs. A
wide crop is cut into pieces at the widest ink-free column gaps so every piece
stays below ``max_ratio``; the pieces are recognised separately and joined.
"""
from __future__ import annotations

from typing import List

import numpy as np
from PIL import Image


def _gap_runs(ink: np.ndarray, min_gap: int):
    """Runs of consecutive ink-free columns of at least ``min_gap`` width: list of (start, end)."""
    empty = ~ink
    padded = np.concatenate([[False], empty, [False]])
    d = np.diff(padded.astype(np.int8))
    starts = np.nonzero(d == 1)[0]
    ends = np.nonzero(d == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_gap]


def split_wide_crop(crop: Image.Image, max_ratio: float = 8.0, min_ratio: float = 3.0) -> List[Image.Image]:
    """Return crop pieces (left to right), each with width/height <= max_ratio when possible."""
    w, h = crop.size
    if h == 0 or w / h <= max_ratio:
        return [crop]
    gray = np.asarray(crop.convert("L"), dtype=np.float32)
    # text is darker than paper after auto-invert; threshold halfway between the extremes
    lo, hi = np.percentile(gray, 2), np.percentile(gray, 98)
    if hi - lo < 20:
        return [crop]
    ink_cols = (gray < (lo + hi) / 2).sum(axis=0) > max(1, h * 0.04)
    gaps = _gap_runs(ink_cols, min_gap=max(2, int(h * 0.2)))
    if not gaps:
        return [crop]
    pieces: List[Image.Image] = []
    target = max_ratio * h * 0.85  # aim a little below the limit
    min_width = min_ratio * h
    start = 0
    while w - start > max_ratio * h:
        limit = start + target
        # widest gap whose centre lies in (start + min_width, limit]
        cands = [(e - s, (s + e) // 2) for s, e in gaps if start + min_width < (s + e) // 2 <= limit]
        if not cands:
            cands = [(e - s, (s + e) // 2) for s, e in gaps if (s + e) // 2 > start + min_width]
            if not cands:
                break
            cands = [min(cands, key=lambda c: c[1])]
        cut = max(cands)[1]
        pieces.append(crop.crop((start, 0, cut, h)))
        start = cut
    pieces.append(crop.crop((start, 0, w, h)))
    return pieces
