"""Group detected text boxes into lines and order them for reading."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from . import _geometry as geo


def sort_boxes(boxes: np.ndarray) -> List[int]:
    """RapidOCR ``sorted_boxes`` order (top to bottom, then left to right). Returns indices."""
    n = len(boxes)
    idx = sorted(range(n), key=lambda i: (boxes[i][0][1], boxes[i][0][0]))
    for i in range(n - 1):
        for j in range(i, -1, -1):
            a, b = boxes[idx[j + 1]], boxes[idx[j]]
            if abs(a[0][1] - b[0][1]) < 10 and a[0][0] < b[0][0]:
                idx[j], idx[j + 1] = idx[j + 1], idx[j]
            else:
                break
    return idx


def group_lines(boxes: Sequence[np.ndarray], rtl: bool = True, overlap: float = 0.5) -> List[List[int]]:
    """Cluster boxes into text lines.

    A box joins the current line when its vertical centre lies inside the
    line's vertical band and the vertical overlap exceeds ``overlap`` of the
    smaller height. Lines are returned top to bottom; inside a line boxes are
    ordered right to left when ``rtl`` is true.
    """
    if len(boxes) == 0:
        return []
    arr = np.asarray([np.asarray(b, dtype=np.float64) for b in boxes])
    top = arr[:, :, 1].min(1)
    bottom = arr[:, :, 1].max(1)
    left = arr[:, :, 0].min(1)
    right = arr[:, :, 0].max(1)
    cy = (top + bottom) / 2
    order = np.argsort(cy, kind="stable")
    lines: List[List[int]] = []
    bands: List[List[float]] = []  # [top, bottom] running band per line
    for i in order:
        placed = False
        for line, band in zip(lines, bands):
            inter = min(bottom[i], band[1]) - max(top[i], band[0])
            h = min(bottom[i] - top[i], band[1] - band[0])
            if h <= 0:
                h = 1.0
            if band[0] <= cy[i] <= band[1] and inter / h >= overlap:
                line.append(int(i))
                band[0] = (band[0] * (len(line) - 1) + top[i]) / len(line)
                band[1] = (band[1] * (len(line) - 1) + bottom[i]) / len(line)
                placed = True
                break
        if not placed:
            lines.append([int(i)])
            bands.append([float(top[i]), float(bottom[i])])
    lines = [l for _, l in sorted(zip([b[0] for b in bands], lines), key=lambda t: t[0])]
    for line in lines:
        if rtl:
            line.sort(key=lambda i: -right[i])
        else:
            line.sort(key=lambda i: left[i])
    return lines


def merge_line_boxes(
    boxes: Sequence[np.ndarray], lines: List[List[int]], gap_ratio: float = 1.0
) -> Tuple[List[np.ndarray], List[List[int]]]:
    """Merge neighbouring boxes of a line when the gap between them is small.

    DB detection often cuts a line into words (and drops lone punctuation such
    as a colon). Recognising the whole line keeps punctuation and gives the
    model more context. Two boxes are merged when the horizontal gap is less
    than ``gap_ratio`` times the taller of the two. Returns the new boxes and
    the new line structure (indices into the new boxes).
    """
    arr = [np.asarray(b, dtype=np.float64) for b in boxes]
    merged: List[np.ndarray] = []
    new_lines: List[List[int]] = []
    for line in lines:
        groups: List[List[int]] = []
        for i in line:  # already ordered along the reading direction
            if groups:
                prev = groups[-1]
                cur_pts = np.concatenate([arr[j] for j in prev])
                h_prev = cur_pts[:, 1].max() - cur_pts[:, 1].min()
                h_i = arr[i][:, 1].max() - arr[i][:, 1].min()
                gap = max(cur_pts[:, 0].min() - arr[i][:, 0].max(), arr[i][:, 0].min() - cur_pts[:, 0].max())
                inter = min(cur_pts[:, 1].max(), arr[i][:, 1].max()) - max(cur_pts[:, 1].min(), arr[i][:, 1].min())
                if gap < gap_ratio * max(h_prev, h_i) and inter > 0.5 * min(h_prev, h_i):
                    prev.append(i)
                    continue
            groups.append([i])
        new_line = []
        for g in groups:
            if len(g) == 1:
                box = arr[g[0]]
            else:
                pts = np.concatenate([arr[j] for j in g])
                box = geo.order_box(geo.min_area_rect(pts).corners())
            new_line.append(len(merged))
            merged.append(box)
        new_lines.append(new_line)
    return merged, new_lines
