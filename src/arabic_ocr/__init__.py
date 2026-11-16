"""Offline Arabic OCR: PP-OCRv4 detection + Arabic recognition on a bundled onnxruntime."""
from ._api import ArabicOCR
from ._bidi import normalize_chars, normalize_digits, visual_to_logical

__version__ = "0.2.0"
__all__ = ["ArabicOCR", "visual_to_logical", "normalize_digits", "normalize_chars", "__version__"]
