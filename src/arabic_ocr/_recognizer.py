"""CTC text recognition (PP-OCRv5 / PP-OCRv4 Arabic rec)."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

from ._runtime import create_session, model_path

# Bundled recognisers. ``visual_order`` says whether the model emits the
# characters in visual (left-to-right) order, which then needs reordering.
MODELS = {
    "v5": {"model": "arabic_PP-OCRv5_rec_infer.onnx", "dict": "arabic_dict_v5.txt", "visual_order": True},
    "v4": {"model": "arabic_PP-OCRv4_rec_infer.onnx", "dict": "arabic_dict.txt", "visual_order": False},
}

DIGITS = "0123456789٠١٢٣٤٥٦٧٨٩"
NUMERIC_CHARSET = DIGITS + "/.,:-%٫٬ "
_DIGIT_RUN = re.compile(rf"[{DIGITS}](?:[{DIGITS}/.,:\-٫٬]*[{DIGITS}])?")


@dataclass
class RecResult:
    """Recogniser output for one crop, in the model's native (raw) order."""

    text: str = ""
    conf: float = 0.0  # mean character probability
    min_conf: float = 0.0  # weakest character probability
    probs: List[float] = field(default_factory=list)  # per emitted character
    steps: List[int] = field(default_factory=list)  # CTC time step of each character
    n_steps: int = 0  # total time steps (maps steps to horizontal position)

    def as_tuple(self):
        return self.text, self.conf


class TextRecognizer:
    def __init__(self, model: str = "v5", batch_size: int = 6, img_shape=(3, 48, 320), num_threads=None, model_file=None, dict_file=None):
        if model not in MODELS:
            raise ValueError(f"unknown recogniser {model!r}; choose one of {sorted(MODELS)}")
        self.model = model
        self.visual_order = MODELS[model]["visual_order"]
        self.batch_size = batch_size
        self.img_shape = tuple(img_shape)
        self.session = create_session(model_file or model_path(MODELS[model]["model"]), num_threads)
        self.input_name = self.session.get_inputs()[0].name
        with open(dict_file or model_path(MODELS[model]["dict"]), encoding="utf-8") as f:
            chars = [line.rstrip("\n").rstrip("\r") for line in f]
        if chars and chars[-1] == "":
            chars = chars[:-1]
        self.characters: List[str] = ["blank"] + chars + [" "]
        out_dim = self.session.get_outputs()[0].shape[-1]
        if isinstance(out_dim, int) and out_dim != len(self.characters):
            raise RuntimeError(f"dictionary size {len(self.characters)} does not match model output {out_dim}")
        self._masks = {}

    # ------------------------------------------------------------------ #
    def _resize_norm(self, img: Image.Image, max_wh_ratio: float) -> np.ndarray:
        _, img_h, img_w = self.img_shape
        img_w = int(img_h * max_wh_ratio)
        w, h = img.size
        ratio = w / float(h)
        resized_w = img_w if math.ceil(img_h * ratio) > img_w else int(math.ceil(img_h * ratio))
        resized_w = max(resized_w, 1)
        arr = np.asarray(img.resize((resized_w, img_h), Image.BILINEAR), dtype=np.float32)[:, :, ::-1]  # BGR
        arr = (arr.transpose(2, 0, 1) / 255.0 - 0.5) / 0.5
        out = np.zeros((3, img_h, img_w), dtype=np.float32)
        out[:, :, :resized_w] = arr
        return out

    def _charset_mask(self, charset: Optional[str]) -> Optional[np.ndarray]:
        """Boolean mask over classes: True = allowed. Blank and space always allowed."""
        if charset is None:
            return None
        if charset not in self._masks:
            allowed = set(charset) | {" "}
            mask = np.array([i == 0 or c in allowed for i, c in enumerate(self.characters)])
            self._masks[charset] = mask
        return self._masks[charset]

    def _decode(self, preds: np.ndarray, mask: Optional[np.ndarray] = None) -> List[RecResult]:
        if mask is not None:
            preds = np.where(mask[None, None, :], preds, 0.0)
        idx = preds.argmax(axis=2)
        prob = preds.max(axis=2)
        results = []
        for seq, p in zip(idx, prob):
            keep = np.ones(len(seq), dtype=bool)
            keep[1:] = seq[1:] != seq[:-1]
            keep &= seq != 0
            chars = [self.characters[i] for i in seq[keep]]
            probs = p[keep].astype(float).tolist()
            steps = np.nonzero(keep)[0].tolist()
            results.append(
                RecResult(
                    text="".join(chars),
                    conf=float(np.mean(probs)) if probs else 0.0,
                    min_conf=float(np.min(probs)) if probs else 0.0,
                    probs=probs,
                    steps=steps,
                    n_steps=int(len(seq)),
                )
            )
        return results

    # ------------------------------------------------------------------ #
    def recognize(self, crops: Sequence[Image.Image], charset: Optional[str] = None) -> List[RecResult]:
        """Recognise crops. ``charset`` restricts the output alphabet (e.g. digits only)."""
        n = len(crops)
        results: List[RecResult] = [RecResult() for _ in range(n)]
        if n == 0:
            return results
        mask = self._charset_mask(charset)
        ratios = np.array([c.width / float(c.height) for c in crops])
        order = np.argsort(ratios)
        _, img_h, img_w = self.img_shape
        for beg in range(0, n, self.batch_size):
            end = min(n, beg + self.batch_size)
            max_wh_ratio = img_w / float(img_h)
            for k in range(beg, end):
                max_wh_ratio = max(max_wh_ratio, float(ratios[order[k]]))
            batch = np.stack([self._resize_norm(crops[order[k]], max_wh_ratio) for k in range(beg, end)])
            preds = self.session.run(None, {self.input_name: batch})[0]
            for k, res in zip(range(beg, end), self._decode(preds, mask)):
                results[order[k]] = res
        return results

    def __call__(self, crops: Sequence[Image.Image]):
        """Backwards compatible: list of (text, confidence)."""
        return [r.as_tuple() for r in self.recognize(crops)]

    # ------------------------------------------------------------------ #
    def refine_digits(self, crop: Image.Image, res: RecResult, margin: float = 0.6) -> RecResult:
        """Re-read every digit run of ``res`` with a digits-only alphabet.

        The CTC time steps locate each run horizontally; that strip (plus a
        margin of ``margin`` x crop height on both sides) is recognised again
        with all non-numeric classes masked, which removes letter/digit
        confusions such as ``٥``/``ه`` or ``١``/``ا``. The new reading replaces
        the old one when it keeps the digit count and is at least as confident.
        """
        if not res.text or res.n_steps == 0 or len(res.steps) != len(res.text):
            return res
        w, h = crop.size
        text, probs = res.text, list(res.probs)
        changed = False
        for m in reversed(list(_DIGIT_RUN.finditer(res.text))):  # from the end so indices stay valid
            a, b = m.start(), m.end()
            run = res.text[a:b]
            if sum(c in DIGITS for c in run) < 2:
                continue
            x0 = int(max(0, res.steps[a] / res.n_steps * w - margin * h))
            x1 = int(min(w, (res.steps[b - 1] + 1) / res.n_steps * w + margin * h))
            if x1 - x0 < 4:
                continue
            sub = self.recognize([crop.crop((x0, 0, x1, h))], charset=NUMERIC_CHARSET)[0]
            cand = sub.text.strip().strip("/.,:-")
            if not cand or sum(c in DIGITS for c in cand) != sum(c in DIGITS for c in run):
                continue
            if sub.conf + 0.02 < res.conf and cand != run:
                continue
            sub_probs = sub.probs if len(sub.probs) == len(cand) else [sub.conf] * len(cand)
            if cand == run:
                sub_probs = [max(o, n) for o, n in zip(probs[a:b], sub_probs)]
            text = text[:a] + cand + text[b:]
            probs = probs[:a] + sub_probs + probs[b:]
            changed = True
        if not changed:
            return res
        return RecResult(text=text, conf=float(np.mean(probs)) if probs else 0.0, min_conf=float(np.min(probs)) if probs else 0.0, probs=probs, steps=[], n_steps=res.n_steps)
