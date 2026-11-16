"""Image loading and optional clean-up before OCR."""
from __future__ import annotations

import os
from typing import Union

import numpy as np
from PIL import Image, ImageFilter, ImageOps

ImageLike = Union[str, "os.PathLike[str]", Image.Image, np.ndarray]


def to_pil(image: ImageLike) -> Image.Image:
    """Accept a path, a PIL image or a numpy array and return an RGB PIL image.

    Numpy arrays are interpreted as HxW (grayscale), HxWx3 (RGB) or HxWx4
    (RGBA), ``uint8``; float arrays in [0, 1] are scaled.
    """
    if isinstance(image, Image.Image):
        img = image
    elif isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = np.asarray(arr, dtype=np.float64)
            if arr.size and arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            img = Image.fromarray(arr, "L")
        elif arr.ndim == 3 and arr.shape[2] == 3:
            img = Image.fromarray(arr, "RGB")
        elif arr.ndim == 3 and arr.shape[2] == 4:
            img = Image.fromarray(arr, "RGBA")
        elif arr.ndim == 3 and arr.shape[2] == 1:
            img = Image.fromarray(arr[:, :, 0], "L")
        else:
            raise ValueError(f"unsupported array shape {arr.shape}")
    elif isinstance(image, (str, bytes, os.PathLike)):
        img = Image.open(image)
        img.load()
    else:
        raise TypeError(f"unsupported image type {type(image)!r}")
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(img, dtype=np.float64)
        arr = (arr - arr.min()) / max(float(arr.max() - arr.min()), 1.0) * 255.0
        img = Image.fromarray(arr.astype(np.uint8), "L")
    if img.mode == "RGBA" or img.mode == "LA" or (img.mode == "P" and "transparency" in img.info):
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.alpha_composite(img.convert("RGBA"))
        img = bg
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img.convert("RGB")


def is_light_on_dark(crop: Image.Image, margin: float = 6.0) -> bool:
    """True when a text crop has light ink on a dark background.

    Inside a text-line crop the background is the majority, so the median is
    the background level; ink pulls the mean towards its own brightness.
    """
    arr = np.asarray(crop.convert("L"), dtype=np.float32)
    if arr.size == 0:
        return False
    return float(arr.mean()) > float(np.median(arr)) + margin


def is_dark_background(img: Image.Image) -> bool:
    """Whole-page polarity guess (kept for callers that want a global decision)."""
    g = img.convert("L")
    if g.width > 768:
        g = g.resize((768, max(1, int(768 * g.height / g.width))))
    return is_light_on_dark(g)


def preprocess(
    img: Image.Image,
    upscale: float = 1.0,
    auto_invert: bool = True,  # noqa: ARG001 - kept for signature compatibility, handled per crop
    grayscale: bool = False,
    denoise: int = 0,
) -> Image.Image:
    """Apply the configured clean-up steps and return an RGB image."""
    if denoise:
        size = int(denoise)
        if size % 2 == 0:
            size += 1
        img = img.filter(ImageFilter.MedianFilter(size))
    if grayscale:
        img = img.convert("L")
    # auto_invert is applied per detected text line (see ArabicOCR.read); a
    # global inversion would be wrong for a white page photographed on a dark
    # table or screen.
    if upscale and upscale != 1.0:
        w, h = img.size
        img = img.resize((max(1, int(round(w * upscale))), max(1, int(round(h * upscale)))), Image.BICUBIC)
    return img.convert("RGB")
