"""DBNet text detection (PP-OCRv4 det) with a numpy-only post-processor."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from PIL import Image

from . import _geometry as geo
from ._runtime import create_session, model_path

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class TextDetector:
    def __init__(
        self,
        thresh: float = 0.3,
        box_thresh: float = 0.5,
        unclip_ratio: float = 1.6,
        limit_side_len: int = 736,
        limit_type: str = "min",
        max_side_len: int = 2560,
        max_candidates: int = 1000,
        use_dilation: bool = True,
        num_threads=None,
    ):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio
        self.limit_side_len = limit_side_len
        self.limit_type = limit_type
        self.max_side_len = max_side_len
        self.max_candidates = max_candidates
        self.use_dilation = use_dilation
        self.session = create_session(model_path("ch_PP-OCRv4_det_infer.onnx"), num_threads)
        self.input_name = self.session.get_inputs()[0].name

    # ------------------------------------------------------------------ #
    def _resize(self, img: Image.Image) -> Tuple[Image.Image, float, float]:
        w, h = img.size
        if self.limit_type == "max":
            ratio = self.limit_side_len / max(h, w) if max(h, w) > self.limit_side_len else 1.0
        else:
            ratio = self.limit_side_len / min(h, w) if min(h, w) < self.limit_side_len else 1.0
        if self.max_side_len and max(h, w) * ratio > self.max_side_len:
            ratio = self.max_side_len / max(h, w)
        rh, rw = int(h * ratio), int(w * ratio)
        rh = max(int(round(rh / 32) * 32), 32)
        rw = max(int(round(rw / 32) * 32), 32)
        resized = img.resize((rw, rh), Image.BILINEAR)
        return resized, rw / w, rh / h

    def _to_tensor(self, img: Image.Image) -> np.ndarray:
        arr = np.asarray(img, dtype=np.float32)[:, :, ::-1]  # RGB -> BGR like cv2.imread
        arr = (arr / 255.0 - _MEAN) / _STD
        return np.ascontiguousarray(arr.transpose(2, 0, 1)[None])

    # ------------------------------------------------------------------ #
    def __call__(self, img: Image.Image) -> Tuple[np.ndarray, List[float]]:
        """Return (boxes (N,4,2) int in image coordinates, scores)."""
        src_w, src_h = img.size
        resized, _, _ = self._resize(img)
        prob = self.session.run(None, {self.input_name: self._to_tensor(resized)})[0][0, 0]
        boxes, scores = self._boxes_from_bitmap(prob, src_w, src_h)
        return self._filter(boxes, src_w, src_h, scores)

    def _boxes_from_bitmap(self, prob: np.ndarray, dest_w: int, dest_h: int):
        bitmap = prob > self.thresh
        if self.use_dilation:
            bitmap = geo.dilate2x2(bitmap)
        h, w = bitmap.shape
        comps = geo.connected_components(bitmap)[: self.max_candidates]
        boxes, scores = [], []
        for comp in comps:
            rect = geo.min_area_rect(comp.hull_points())
            if rect.short_side < 3:
                continue
            quad = geo.order_box(rect.corners())
            score = geo.mean_in_polygon(prob, quad)
            if score < self.box_thresh:
                continue
            area, perim = geo.polygon_area_perimeter(quad)
            if perim <= 0:
                continue
            distance = area * self.unclip_ratio / perim
            big = rect.expanded(distance)
            if big.short_side < 5:
                continue
            box = geo.order_box(big.corners())
            box[:, 0] = np.clip(np.round(box[:, 0] / w * dest_w), 0, dest_w)
            box[:, 1] = np.clip(np.round(box[:, 1] / h * dest_h), 0, dest_h)
            boxes.append(box.astype(np.int32))
            scores.append(score)
        if not boxes:
            return np.zeros((0, 4, 2), dtype=np.int32), []
        return np.stack(boxes), scores

    def _filter(self, boxes: np.ndarray, w: int, h: int, scores):
        keep, keep_scores = [], []
        for box, score in zip(boxes, scores):
            box = geo.order_points_clockwise(box)
            box[:, 0] = np.clip(box[:, 0], 0, w - 1)
            box[:, 1] = np.clip(box[:, 1], 0, h - 1)
            rw = int(np.linalg.norm(box[0] - box[1]))
            rh = int(np.linalg.norm(box[0] - box[3]))
            if rw <= 3 or rh <= 3:
                continue
            keep.append(box.astype(np.int32))
            keep_scores.append(float(score))
        if not keep:
            return np.zeros((0, 4, 2), dtype=np.int32), []
        return np.stack(keep), keep_scores
