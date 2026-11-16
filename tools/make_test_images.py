#!/usr/bin/env python
"""Render Arabic test images (clean and "scanned" variants) with ground truth.

Uses Pillow + arabic_reshaper + python-bidi (basic layout engine, no raqm) so
the rendering path is the same one most Windows users have. Fonts: Windows
fonts when present (Arial, Tahoma, Simplified Arabic), otherwise the closest
fonts found on this machine. Output: PNG files plus ground_truth.json in
``tests/data``.

    python tools/make_test_images.py [--out tests/data]
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import arabic_reshaper
import numpy as np
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SAMPLES = {
    "mixed_code": ["الكود : ٤٦١٠٠٢", "تاريخ بدء النشاط : ٢٠١٢/٠٦/٢٥"],
    "paragraph": [
        "بسم الله الرحمن الرحيم",
        "تعتبر اللغة العربية من أكثر اللغات انتشاراً في العالم",
        "وهي اللغة الرسمية في اثنتين وعشرين دولة",
        "يتحدث بها أكثر من أربعمائة مليون نسمة",
    ],
    "invoice": [
        "فاتورة ضريبية رقم 2024/0317",
        "اسم العميل : شركة النور للتجارة",
        "الإجمالي : 1250.75 ريال",
        "الرقم الضريبي : 300012345600003",
        "التاريخ : ١٥/٠٨/٢٠٢٤",
    ],
    "form": [
        "الاسم : محمد عبد الرحمن",
        "رقم الهوية : ١٠٨٧٦٥٤٣٢١",
        "الجنسية : مصري",
        "المهنة : مهندس برمجيات",
    ],
}

WINDOWS_FONTS = {
    "arial": ["C:/Windows/Fonts/arial.ttf"],
    "tahoma": ["C:/Windows/Fonts/tahoma.ttf"],
    "simplified_arabic": ["C:/Windows/Fonts/simpo.ttf", "C:/Windows/Fonts/simpfxo.ttf"],
}
FALLBACK_FONTS = {
    # closest substitutes available on a Linux box
    "arial": ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"],
    "tahoma": [
        os.path.expanduser("~/snap/steam/common/.local/share/Steam/steamapps/common/Proton - Experimental/files/share/wine/fonts/tahoma.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
    "simplified_arabic": [
        "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ],
}


def resolve_fonts(extra: dict | None = None) -> dict:
    fonts = {}
    for name in WINDOWS_FONTS:
        for cand in WINDOWS_FONTS[name] + FALLBACK_FONTS[name]:
            if cand and Path(cand).is_file():
                fonts[name] = cand
                break
    if extra:
        fonts.update({k: v for k, v in extra.items() if Path(v).is_file()})
    return fonts


def font_covers(font_path: str, text: str) -> bool:
    try:
        from fontTools.ttLib import TTFont  # noqa: PLC0415
    except ImportError:
        return True
    cmap = TTFont(font_path).getBestCmap()
    return all(ord(c) in cmap for c in text if not c.isspace())


def shape(line: str) -> str:
    return get_display(arabic_reshaper.reshape(line))


def render(lines, font_path, size=40, margin=40, spacing=1.6, width=None):
    font = ImageFont.truetype(font_path, size, layout_engine=ImageFont.Layout.BASIC)
    shaped = [shape(l) for l in lines]
    dummy = ImageDraw.Draw(Image.new("L", (1, 1)))
    widths = [dummy.textbbox((0, 0), s, font=font)[2] for s in shaped]
    line_h = int(size * spacing)
    w = width or max(widths) + 2 * margin
    h = line_h * len(lines) + 2 * margin
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    for i, s in enumerate(shaped):
        tw = widths[i]
        draw.text((w - margin - tw, margin + i * line_h), s, font=font, fill="black")  # right aligned
    return img


def scan_variant(img: Image.Image, rng: random.Random, blur=1.2, noise=18, rotate=0.6, jpeg=60) -> Image.Image:
    out = img.convert("L")
    if rotate:
        out = out.rotate(rng.uniform(-rotate, rotate), resample=Image.BICUBIC, fillcolor=255, expand=False)
    if blur:
        out = out.filter(ImageFilter.GaussianBlur(blur))
    arr = np.asarray(out, dtype=np.float32)
    arr = arr * rng.uniform(0.85, 0.95) + rng.uniform(5, 20)  # grey paper, lower contrast
    arr += np.random.default_rng(rng.randrange(1 << 30)).normal(0, noise, arr.shape)
    # salt and pepper specks
    r = np.random.default_rng(rng.randrange(1 << 30))
    mask = r.random(arr.shape) < 0.002
    arr[mask] = r.choice([0, 255], size=int(mask.sum()))
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "L")
    if jpeg:
        from io import BytesIO  # noqa: PLC0415

        buf = BytesIO()
        out.save(buf, "JPEG", quality=jpeg)
        buf.seek(0)
        out = Image.open(buf).convert("L")
    return out.convert("RGB")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("tests/data"))
    ap.add_argument("--font", action="append", default=[], metavar="NAME=PATH", help="extra font")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    extra = dict(f.split("=", 1) for f in args.font)
    fonts = resolve_fonts(extra)
    print("fonts:", json.dumps(fonts, indent=1))
    rng = random.Random(7)
    truth = {}
    for fname, fpath in fonts.items():
        for sname, lines in SAMPLES.items():
            if not font_covers(fpath, "".join(lines)):
                print(f"skip {sname} with {fname}: missing glyphs")
                continue
            size = 44 if fname == "simplified_arabic" else 38
            clean = render(lines, fpath, size=size)
            base = f"{sname}__{fname}"
            clean.save(args.out / f"{base}.png")
            truth[f"{base}.png"] = {"lines": lines, "font": fname, "variant": "clean"}
            scan = scan_variant(clean, rng)
            scan.save(args.out / f"{base}__scan.png")
            truth[f"{base}__scan.png"] = {"lines": lines, "font": fname, "variant": "scan"}
            heavy = scan_variant(clean, rng, blur=1.8, noise=28, rotate=1.5, jpeg=35)
            heavy.save(args.out / f"{base}__scan_heavy.jpg", quality=80)
            truth[f"{base}__scan_heavy.jpg"] = {"lines": lines, "font": fname, "variant": "scan_heavy"}
    # dark background sample (light text on dark) to exercise auto-invert
    first_font = next(iter(fonts.values()))
    img = render(SAMPLES["mixed_code"], first_font, size=38)
    inv = Image.eval(img, lambda v: 255 - v)
    inv.save(args.out / "mixed_code__inverted.png")
    truth["mixed_code__inverted.png"] = {"lines": SAMPLES["mixed_code"], "font": "first", "variant": "inverted"}
    # small / low resolution sample to exercise upscaling
    small = render(SAMPLES["form"], first_font, size=16, margin=12)
    small.save(args.out / "form__small.png")
    truth["form__small.png"] = {"lines": SAMPLES["form"], "font": "first", "variant": "small"}
    (args.out / "ground_truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(truth)} images to {args.out}")


if __name__ == "__main__":
    main()
