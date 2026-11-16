"""Pure numpy geometry helpers.

These replace the OpenCV / Shapely / pyclipper calls used by RapidOCR's DB
post-processing: connected components, convex hull, minimum-area rectangle,
convex polygon rasterisation and perspective cropping (via Pillow).
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# Connected components (8-connectivity), run-length based
# --------------------------------------------------------------------------- #
class Component:
    """One connected blob of a binary mask.

    ``rows`` are the y coordinates (ascending) and ``xmin``/``xmax`` the
    inclusive horizontal extent of the blob on each row. ``area`` is the pixel
    count. Hull points are the per-row extremes, which is all the convex hull
    needs.
    """

    __slots__ = ("rows", "xmin", "xmax", "area")

    def __init__(self, rows, xmin, xmax, area):
        self.rows = rows
        self.xmin = xmin
        self.xmax = xmax
        self.area = area

    def hull_points(self) -> np.ndarray:
        rows = np.asarray(self.rows, dtype=np.float64)
        xmin = np.asarray(self.xmin, dtype=np.float64)
        xmax = np.asarray(self.xmax, dtype=np.float64)
        pts = np.concatenate([np.stack([xmin, rows], 1), np.stack([xmax, rows], 1)], 0)
        return np.unique(pts, axis=0)


def connected_components(mask: np.ndarray) -> List[Component]:
    """Label an 8-connected binary mask. Returns components in scan order."""
    mask = np.asarray(mask)
    h, w = mask.shape
    padded = np.zeros((h, w + 2), dtype=np.int8)
    padded[:, 1:-1] = mask != 0
    d = np.diff(padded, axis=1)
    row_s, starts = np.nonzero(d == 1)  # run start column (inclusive)
    row_e, ends = np.nonzero(d == -1)  # run end column (exclusive)
    n = len(starts)
    if n == 0:
        return []
    assert len(ends) == n and np.array_equal(row_s, row_e)
    rows = row_s.tolist()
    starts_l = starts.tolist()
    ends_l = ends.tolist()
    first = np.searchsorted(row_s, np.arange(h + 1)).tolist()

    parent = list(range(n))

    def find(i):
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    for r in range(1, h):
        i0, i1 = first[r], first[r + 1]
        j0, j1 = first[r - 1], first[r]
        if i0 == i1 or j0 == j1:
            continue
        j = j0
        for i in range(i0, i1):
            s = starts_l[i]
            e = ends_l[i]
            while j < j1 and ends_l[j] < s:  # previous run ends strictly before (8-conn touch needs end >= s)
                j += 1
            k = j
            while k < j1 and starts_l[k] <= e:
                ri, rk = find(i), find(k)
                if ri != rk:
                    if ri < rk:
                        parent[rk] = ri
                    else:
                        parent[ri] = rk
                k += 1

    comps = {}
    order = []
    for i in range(n):
        root = find(i)
        c = comps.get(root)
        r = rows[i]
        s = starts_l[i]
        e = ends_l[i] - 1
        if c is None:
            comps[root] = c = [[r], [s], [e], e - s + 1]
            order.append(root)
        else:
            if c[0][-1] == r:
                if s < c[1][-1]:
                    c[1][-1] = s
                if e > c[2][-1]:
                    c[2][-1] = e
            else:
                c[0].append(r)
                c[1].append(s)
                c[2].append(e)
            c[3] += e - s + 1
    return [Component(*comps[r]) for r in order]


def dilate2x2(mask: np.ndarray) -> np.ndarray:
    """Equivalent of ``cv2.dilate(mask, np.ones((2, 2)))`` (anchor at (1, 1))."""
    m = mask.astype(bool)
    out = m.copy()
    out[:, 1:] |= m[:, :-1]
    out[1:, :] |= m[:-1, :]
    out[1:, 1:] |= m[:-1, :-1]
    return out


# --------------------------------------------------------------------------- #
# Convex hull and minimum-area rectangle
# --------------------------------------------------------------------------- #
def convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain. Returns hull vertices counter-clockwise (y down => visually clockwise)."""
    pts = np.unique(np.asarray(points, dtype=np.float64), axis=0)  # sorted by x then y
    if len(pts) <= 2:
        return pts
    pts_l = pts.tolist()

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts_l:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts_l):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


class Rect:
    """Oriented rectangle: ``center`` plus unit axes ``u`` (length ``w``) and ``v`` (length ``h``)."""

    __slots__ = ("center", "u", "v", "w", "h")

    def __init__(self, center, u, v, w, h):
        self.center = np.asarray(center, dtype=np.float64)
        self.u = np.asarray(u, dtype=np.float64)
        self.v = np.asarray(v, dtype=np.float64)
        self.w = float(w)
        self.h = float(h)

    @property
    def short_side(self) -> float:
        return min(self.w, self.h)

    def corners(self) -> np.ndarray:
        hw, hh = self.w / 2.0, self.h / 2.0
        c, u, v = self.center, self.u, self.v
        return np.stack([c - u * hw - v * hh, c + u * hw - v * hh, c + u * hw + v * hh, c - u * hw + v * hh])

    def expanded(self, d: float) -> "Rect":
        return Rect(self.center, self.u, self.v, self.w + 2 * d, self.h + 2 * d)


def min_area_rect(points: np.ndarray) -> Rect:
    """Minimum-area bounding rectangle of a point set (rotating calipers on the hull)."""
    hull = convex_hull(points)
    if len(hull) == 1:
        return Rect(hull[0], (1.0, 0.0), (0.0, 1.0), 0.0, 0.0)
    if len(hull) == 2:
        d = hull[1] - hull[0]
        length = float(np.hypot(*d))
        u = d / length
        return Rect((hull[0] + hull[1]) / 2, u, (-u[1], u[0]), length, 0.0)
    edges = np.roll(hull, -1, axis=0) - hull
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    keep = lengths > 1e-12
    edges, lengths = edges[keep], lengths[keep]
    us = edges / lengths[:, None]  # (m, 2)
    vs = np.stack([-us[:, 1], us[:, 0]], 1)
    pu = us @ hull.T  # (m, n) projections on each edge direction
    pv = vs @ hull.T
    umin, umax = pu.min(1), pu.max(1)
    vmin, vmax = pv.min(1), pv.max(1)
    areas = (umax - umin) * (vmax - vmin)
    i = int(np.argmin(areas))
    u, v = us[i], vs[i]
    w, h = umax[i] - umin[i], vmax[i] - vmin[i]
    center = u * (umin[i] + umax[i]) / 2 + v * (vmin[i] + vmax[i]) / 2
    return Rect(center, u, v, w, h)


def order_box(points: np.ndarray) -> np.ndarray:
    """Order 4 corners as top-left, top-right, bottom-right, bottom-left.

    Same rule as RapidOCR's ``get_mini_boxes``: sort by x, then split left/right
    pairs by y.
    """
    pts = sorted(np.asarray(points, dtype=np.float64).tolist(), key=lambda p: (p[0], p[1]))
    if pts[1][1] > pts[0][1]:
        i1, i4 = 0, 1
    else:
        i1, i4 = 1, 0
    if pts[3][1] > pts[2][1]:
        i2, i3 = 2, 3
    else:
        i2, i3 = 3, 2
    return np.asarray([pts[i1], pts[i2], pts[i3], pts[i4]], dtype=np.float64)


def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    """RapidOCR's ``order_points_clockwise``: tl, tr, br, bl by coordinate sums/diffs."""
    pts = np.asarray(pts, dtype=np.float64)
    rect = np.zeros((4, 2), dtype=np.float64)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
    diff = np.diff(tmp, axis=1)
    rect[1] = tmp[np.argmin(diff)]
    rect[3] = tmp[np.argmax(diff)]
    return rect


def polygon_area_perimeter(poly: np.ndarray) -> Tuple[float, float]:
    p = np.asarray(poly, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    perim = float(np.hypot(*(np.roll(p, -1, axis=0) - p).T).sum())
    return area, perim


def convex_polygon_mask(poly: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Boolean mask of pixel centres inside (or on) a convex polygon."""
    poly = np.asarray(poly, dtype=np.float64)
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w]
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)
    inside = np.ones((h, w), dtype=bool)
    x, y = poly[:, 0], poly[:, 1]
    signed = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    sign = 1.0 if signed >= 0 else -1.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        cross = (x1 - x0) * (ys - y0) - (y1 - y0) * (xs - x0)
        inside &= sign * cross >= -1e-9
    return inside


def mean_in_polygon(prob: np.ndarray, poly: np.ndarray) -> float:
    """Mean of ``prob`` over the (convex) polygon, like RapidOCR's ``box_score_fast``."""
    h, w = prob.shape
    poly = np.asarray(poly, dtype=np.float64)
    xmin = int(np.clip(np.floor(poly[:, 0].min()), 0, w - 1))
    xmax = int(np.clip(np.ceil(poly[:, 0].max()), 0, w - 1))
    ymin = int(np.clip(np.floor(poly[:, 1].min()), 0, h - 1))
    ymax = int(np.clip(np.ceil(poly[:, 1].max()), 0, h - 1))
    local = poly.copy()
    local[:, 0] -= xmin
    local[:, 1] -= ymin
    region = prob[ymin : ymax + 1, xmin : xmax + 1]
    mask = convex_polygon_mask(np.floor(local), region.shape)
    if not mask.any():
        return 0.0
    return float(region[mask].mean())


# --------------------------------------------------------------------------- #
# Perspective crop
# --------------------------------------------------------------------------- #
def _perspective_coeffs(dst_quad, src_quad) -> List[float]:
    """Coefficients for ``Image.transform(..., Image.PERSPECTIVE)`` mapping dst -> src."""
    rows = []
    rhs = []
    for (x, y), (u, v) in zip(dst_quad, src_quad):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs.extend([u, v])
    a = np.asarray(rows, dtype=np.float64)
    b = np.asarray(rhs, dtype=np.float64)
    coeffs, *_ = np.linalg.lstsq(a, b, rcond=None)
    return coeffs.tolist()


def crop_quad(image: Image.Image, box: np.ndarray) -> Image.Image:
    """Rectify the quadrilateral ``box`` (tl, tr, br, bl) into an upright crop.

    Mirrors RapidOCR's ``get_rotate_crop_image``: output size from the longer
    edge pair, bicubic resampling, and a 90 degree rotation when the crop is
    much taller than wide (vertical text).
    """
    box = np.asarray(box, dtype=np.float64)
    tl, tr, br, bl = box
    width = int(max(np.linalg.norm(tl - tr), np.linalg.norm(bl - br)))
    height = int(max(np.linalg.norm(tl - bl), np.linalg.norm(tr - br)))
    width = max(width, 1)
    height = max(height, 1)
    dst = [(0, 0), (width, 0), (width, height), (0, height)]
    coeffs = _perspective_coeffs(dst, [tuple(p) for p in box])
    crop = image.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    if height * 1.0 / width >= 1.5:
        crop = crop.transpose(Image.ROTATE_90)
    return crop


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (used by tests and the evaluation tool)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
