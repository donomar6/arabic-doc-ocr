"""Public OCR pipeline."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ._bidi import fix_trailing_period, has_arabic, normalize_chars, normalize_digits, visual_to_logical
from ._detector import TextDetector
from ._layout import group_lines, merge_line_boxes
from PIL import ImageOps

from ._preprocess import ImageLike, is_light_on_dark, preprocess, to_pil
from ._recognizer import TextRecognizer
from ._split import split_wide_crop
from . import _geometry as geo


class ArabicOCR:
    """Offline Arabic OCR for scanned documents.

    Parameters
    ----------
    model:
        Recogniser: ``"v5"`` (PP-OCRv5 Arabic, default, reads Arabic-Indic
        digits including zero) or ``"v4"`` (PP-OCRv4 Arabic).
    upscale:
        Resize factor applied before detection. Use 1.5 to 2.0 for small text
        or low-resolution scans (less than about 150 dpi).
    auto_invert:
        Invert text lines that have light ink on a dark background. Decided per
        detected line, so a white page photographed on a dark desk or screen,
        or a page with a white-on-black header, both work.
    grayscale:
        Convert to grayscale first. Removes colour noise from scans.
    denoise:
        Median filter size (odd, 0 to disable). 3 is a good value for noisy scans.
    min_confidence:
        Drop recognised lines whose mean character confidence is lower.
    max_crop_ratio:
        Line crops wider than this many times their height are split at
        whitespace gaps and recognised in pieces (the models degrade on very
        wide inputs). 0 disables splitting.
    merge_words:
        Merge neighbouring detection boxes of one line (gap smaller than
        ``merge_gap`` times the text height) before recognition. Keeps
        punctuation between words and improves accuracy; set to False to get
        one result per detected box.
    logical_order:
        Return text in logical (storage) order. ``None`` (default) does the
        right thing for the chosen model: the v5 model emits visual
        left-to-right order and is reordered, the v4 model already emits
        logical order. ``False`` returns the recogniser output untouched.
    refine_digits:
        Re-read every digit run with a digits-only alphabet located via the
        CTC alignment. Meant for letter/digit confusions in codes and dates;
        no gain on the synthetic test set, so off by default. Try it on your
        own documents.
    normalize:
        Map Persian letter forms / Extended Arabic-Indic digits emitted by the
        model to Arabic ones, and make mixed digit runs use one digit system
        (the v4 model has no Arabic-Indic zero and reads it as ``0``).
    det_* / rec_batch_size / num_threads:
        Detector thresholds and runtime settings (RapidOCR defaults).
    """

    def __init__(
        self,
        model: str = "v5",
        upscale: float = 1.0,
        auto_invert: bool = True,
        grayscale: bool = False,
        denoise: int = 0,
        min_confidence: float = 0.5,
        merge_words: bool = True,
        merge_gap: float = 1.0,
        max_crop_ratio: float = 8.0,
        refine_digits: bool = False,
        logical_order: Optional[bool] = None,
        normalize: bool = True,
        det_thresh: float = 0.3,
        det_box_thresh: float = 0.5,
        det_unclip_ratio: float = 1.6,
        det_limit_side_len: int = 736,
        det_limit_type: str = "min",
        det_max_side_len: int = 2560,
        rec_batch_size: int = 6,
        num_threads: Optional[int] = None,
    ):
        self.upscale = float(upscale)
        self.auto_invert = auto_invert
        self.grayscale = grayscale
        self.denoise = int(denoise)
        self.min_confidence = float(min_confidence)
        self.merge_words = merge_words
        self.merge_gap = float(merge_gap)
        self.max_crop_ratio = float(max_crop_ratio)
        self.refine_digits = refine_digits
        self.normalize = normalize
        self.detector = TextDetector(
            thresh=det_thresh,
            box_thresh=det_box_thresh,
            unclip_ratio=det_unclip_ratio,
            limit_side_len=det_limit_side_len,
            limit_type=det_limit_type,
            max_side_len=det_max_side_len,
            num_threads=num_threads,
        )
        self.recognizer = TextRecognizer(model=model, batch_size=rec_batch_size, num_threads=num_threads)
        self.model = model
        # Reorder only when the model's raw output is in visual order.
        self.reorder = self.recognizer.visual_order if logical_order is None else bool(logical_order)

    # ------------------------------------------------------------------ #
    def read(self, image: ImageLike) -> List[Dict[str, Any]]:
        """Recognise text. Returns one dict per text box in reading order.

        Keys: ``text`` (str, logical order), ``confidence`` (mean character
        probability, 0-1), ``min_confidence`` (weakest character), ``box`` (4 ``[x, y]`` corners, top-left first, clockwise, in the
        coordinates of the input image) and ``line`` (0-based line index).
        """
        img = preprocess(
            to_pil(image),
            upscale=self.upscale,
            auto_invert=self.auto_invert,
            grayscale=self.grayscale,
            denoise=self.denoise,
        )
        boxes, _det_scores = self.detector(img)
        if len(boxes) == 0:
            return []
        lines = group_lines(list(boxes), rtl=True)
        if self.merge_words:
            boxes, lines = merge_line_boxes(list(boxes), lines, self.merge_gap)
        else:
            boxes = [np.asarray(b, dtype=np.float64) for b in boxes]
        crops = [geo.crop_quad(img, box) for box in boxes]
        if self.auto_invert:
            crops = [ImageOps.invert(c) if is_light_on_dark(c) else c for c in crops]
        rec = self._recognize(crops)
        scale = 1.0 / self.upscale if self.upscale else 1.0
        out: List[Dict[str, Any]] = []
        line_no = 0
        for line in lines:
            added = False
            for i in line:
                text, conf, min_conf = rec[i]
                if conf < self.min_confidence or not text.strip():
                    continue
                if self.reorder:
                    text = visual_to_logical(text)
                if self.normalize:
                    text = fix_trailing_period(normalize_digits(normalize_chars(text)))
                pts = np.round(np.asarray(boxes[i], dtype=np.float64) * scale).astype(int).tolist()
                out.append({"text": text, "confidence": round(float(conf), 4), "min_confidence": round(float(min_conf), 4), "box": pts, "line": line_no})
                added = True
            if added:
                line_no += 1
        return out

    def _recognize(self, crops):
        """Recognise crops, splitting over-wide ones into pieces first.

        Returns one ``(text, conf, min_conf)`` per crop in the model's raw order.
        """
        pieces, owner = [], []
        for i, crop in enumerate(crops):
            parts = split_wide_crop(crop, self.max_crop_ratio) if self.max_crop_ratio else [crop]
            pieces.extend(parts)
            owner.extend([i] * len(parts))
        raw = self.recognizer.recognize(pieces)
        if self.refine_digits:
            raw = [self.recognizer.refine_digits(piece, r) for piece, r in zip(pieces, raw)]
        results = []
        for i in range(len(crops)):
            parts = [raw[k] for k in range(len(raw)) if owner[k] == i and raw[k].text.strip()]
            if not parts:
                results.append(("", 0.0, 0.0))
                continue
            if not self.recognizer.visual_order:
                parts = parts[::-1]  # logical-order output: rightmost piece comes first
            text = " ".join(r.text for r in parts)
            weights = [max(len(r.text), 1) for r in parts]
            conf = sum(r.conf * w for r, w in zip(parts, weights)) / sum(weights)
            results.append((text, conf, min(r.min_conf for r in parts)))
        return results

    def read_lines(self, image: ImageLike) -> List[str]:
        """Text lines top to bottom. Boxes on one line are joined right to left with spaces."""
        details = self.read(image)
        lines: Dict[int, List[str]] = {}
        for d in details:
            lines.setdefault(d["line"], []).append(d["text"])
        return [" ".join(lines[k]) for k in sorted(lines)]

    __call__ = read_lines

    def read_text(self, image: ImageLike) -> str:
        """All text as one string, lines joined with newlines."""
        return "\n".join(self.read_lines(image))

    @staticmethod
    def is_arabic(text: str) -> bool:
        return has_arabic(text)
