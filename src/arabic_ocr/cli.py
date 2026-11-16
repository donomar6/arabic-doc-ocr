"""Command line interface: ``arabic-ocr image.jpg -o out.txt``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arabic-ocr", description="Offline Arabic OCR for scanned document images.")
    p.add_argument("images", nargs="+", type=Path, help="input image(s): JPG, PNG, TIFF, BMP, ...")
    p.add_argument("-o", "--output", type=Path, help="write text here (UTF-8). Default: stdout")
    p.add_argument("--json", action="store_true", help="emit per-box JSON (text, confidence, box, line)")
    p.add_argument("--fields", action="store_true", help="extract and validate labelled fields (commercial-registry schema), JSON output")
    p.add_argument("--code-table", type=Path, default=None, help="CSV of valid activity codes (first column) for --fields")
    p.add_argument("--passes", default="1", help="comma-separated upscale factors for --fields voting, e.g. 1,2")
    p.add_argument("--review-dir", type=Path, default=None, help="save crops of fields that need review here")
    p.add_argument("--refine-digits", action="store_true", help="re-read numbers with a digits-only alphabet (experimental)")
    p.add_argument("--model", choices=["v5", "v4"], default="v5", help="recogniser: PP-OCRv5 (default) or PP-OCRv4 Arabic")
    p.add_argument("--upscale", type=float, default=1.0, help="resize factor before detection (e.g. 2 for small text)")
    p.add_argument("--no-auto-invert", action="store_true", help="do not invert dark pages")
    p.add_argument("--grayscale", action="store_true", help="convert to grayscale first")
    p.add_argument("--denoise", type=int, default=0, metavar="N", help="median filter size (odd), 0 = off")
    p.add_argument("--min-confidence", type=float, default=0.5, help="drop lines below this confidence")
    p.add_argument("--raw-order", action="store_true", help="return recogniser output without reordering/normalisation")
    p.add_argument("--threads", type=int, default=None, help="onnxruntime intra-op threads")
    p.add_argument("--version", action="version", version=_version())
    return p


def _version() -> str:
    from . import __version__  # noqa: PLC0415

    return f"arabic-ocr {__version__}"


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from ._api import ArabicOCR  # noqa: PLC0415  (import after arg parsing so --help is instant)

    ocr = ArabicOCR(
        model=args.model,
        upscale=args.upscale,
        auto_invert=not args.no_auto_invert,
        grayscale=args.grayscale,
        denoise=args.denoise,
        min_confidence=args.min_confidence,
        logical_order=False if args.raw_order else None,
        normalize=not args.raw_order,
        refine_digits=args.refine_digits,
        num_threads=args.threads,
    )
    extractor = None
    if args.fields:
        from .fields import FieldExtractor  # noqa: PLC0415

        extractor = FieldExtractor(ocr=ocr, code_table=args.code_table, passes=[float(x) for x in args.passes.split(",")])
    chunks = []
    for path in args.images:
        if extractor is not None:
            res = extractor.extract(path, review_dir=(args.review_dir / path.stem) if args.review_dir else None)
            chunks.append(json.dumps({"image": str(path), **res}, ensure_ascii=False, indent=2))
        elif args.json:
            chunks.append(json.dumps({"image": str(path), "results": ocr.read(path)}, ensure_ascii=False, indent=2))
        else:
            text = ocr.read_text(path)
            if len(args.images) > 1:
                chunks.append(f"===== {path}\n{text}")
            else:
                chunks.append(text)
    out = "\n".join(chunks)
    if not out.endswith("\n"):
        out += "\n"
    if args.output:
        args.output.write_text(out, encoding="utf-8")
    else:
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
        sys.stdout.write(out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
