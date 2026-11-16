"""End-to-end tests on the rendered images in tests/data (no network, no extra deps)."""
import json
import re
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from arabic_ocr import ArabicOCR
from arabic_ocr._geometry import edit_distance

DATA = Path(__file__).parent / "data"
TRUTH = json.loads((DATA / "ground_truth.json").read_text(encoding="utf-8")) if (DATA / "ground_truth.json").exists() else {}


def norm(s):
    return re.sub(r"[ \t]+", " ", s).strip()


def cer(pred_lines, gold_lines):
    gold = norm("\n".join(gold_lines))
    return edit_distance(norm("\n".join(pred_lines)), gold) / max(len(gold), 1)


@unittest.skipUnless(TRUTH, "tests/data not present")
class EndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ocr = ArabicOCR()

    def test_vendored_runtime_is_private(self):
        self.assertIn("arabic_ocr._vendor.onnxruntime", sys.modules)
        self.assertNotIn("onnxruntime", sys.modules)

    def test_input_types(self):
        path = DATA / "mixed_code__tahoma.png"
        from_path = self.ocr(str(path))
        from_pil = self.ocr(Image.open(path))
        from_array = self.ocr(np.asarray(Image.open(path).convert("RGB")))
        from_gray = self.ocr(np.asarray(Image.open(path).convert("L")))
        self.assertEqual(from_path, from_pil)
        self.assertEqual(from_path, from_array)
        self.assertTrue(from_gray)

    def test_mixed_code_logical_order(self):
        lines = self.ocr(DATA / "mixed_code__tahoma.png")
        self.assertEqual(len(lines), 2)
        # label first, then separator, then the number: logical order, not visually reversed
        self.assertTrue(lines[0].startswith("الكود"), lines[0])
        self.assertTrue(lines[0].endswith("٤٦١٠٠٢"), lines[0])
        self.assertTrue(lines[1].startswith("تاريخ بدء النشاط"), lines[1])
        self.assertTrue(lines[1].endswith("٢٠١٢/٠٦/٢٥"), lines[1])

    def test_read_details(self):
        details = self.ocr.read(DATA / "mixed_code__tahoma.png")
        self.assertTrue(details)
        for d in details:
            self.assertEqual(set(d), {"text", "confidence", "min_confidence", "box", "line"})
            self.assertLessEqual(d["min_confidence"], d["confidence"] + 1e-6)
            self.assertEqual(len(d["box"]), 4)
            self.assertGreaterEqual(d["confidence"], 0.5)
        self.assertEqual(self.ocr.read_text(DATA / "mixed_code__tahoma.png"), "\n".join(self.ocr(DATA / "mixed_code__tahoma.png")))

    def test_clean_cer(self):
        errs = chars = 0
        for name, meta in TRUTH.items():
            if meta["variant"] != "clean":
                continue
            gold = norm("\n".join(meta["lines"]))
            errs += edit_distance(norm("\n".join(self.ocr(DATA / name))), gold)
            chars += len(gold)
        self.assertLess(errs / chars, 0.15, f"clean CER too high: {errs / chars:.3f}")

    def test_scan_cer(self):
        errs = chars = 0
        for name, meta in TRUTH.items():
            if meta["variant"] not in ("scan", "scan_heavy"):
                continue
            gold = norm("\n".join(meta["lines"]))
            errs += edit_distance(norm("\n".join(self.ocr(DATA / name))), gold)
            chars += len(gold)
        self.assertLess(errs / chars, 0.35, f"scan CER too high: {errs / chars:.3f}")

    def test_inverted_page_auto_invert(self):
        lines = self.ocr(DATA / "mixed_code__inverted.png")
        self.assertTrue(lines and lines[0].startswith("الكود"), lines)

    def test_small_text_with_upscale(self):
        ocr = ArabicOCR(upscale=2.0)
        self.assertLess(cer(ocr(DATA / "form__small.png"), TRUTH["form__small.png"]["lines"]), 0.2)

    def test_v4_model_loads(self):
        ocr = ArabicOCR(model="v4")
        lines = ocr(DATA / "paragraph__tahoma.png")
        self.assertEqual(len(lines), 4)
        self.assertTrue(lines[0].startswith("بسم"), lines[0])

    def test_blank_image(self):
        self.assertEqual(self.ocr(np.full((100, 300, 3), 255, np.uint8)), [])


if __name__ == "__main__":
    unittest.main()
