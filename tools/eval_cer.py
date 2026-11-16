#!/usr/bin/env python
"""Run ArabicOCR over tests/data and report character error rate per image.

    python tools/eval_cer.py [--data tests/data] [--upscale 1.0] [--show-mismatch]

CER = Levenshtein(pred, truth) / len(truth) over the page text joined with
newlines (line order matters). Spaces are collapsed before comparison.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_ocr import ArabicOCR  # noqa: E402
from arabic_ocr._geometry import edit_distance  # noqa: E402


def norm(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s).strip()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("tests/data"))
    ap.add_argument("--upscale", type=float, default=1.0)
    ap.add_argument("--denoise", type=int, default=0)
    ap.add_argument("--grayscale", action="store_true")
    ap.add_argument("--model", default="v5", choices=["v5", "v4"])
    ap.add_argument("--raw-order", action="store_true")
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--merge-gap", type=float, default=1.0)
    ap.add_argument("--unclip", type=float, default=1.6)
    ap.add_argument("--max-crop-ratio", type=float, default=8.0)
    ap.add_argument("--refine-digits", action="store_true")
    ap.add_argument("--show-mismatch", action="store_true")
    ap.add_argument("--only", default=None, help="substring filter on file name")
    args = ap.parse_args(argv)
    truth = json.loads((args.data / "ground_truth.json").read_text(encoding="utf-8"))
    ocr = ArabicOCR(
        model=args.model,
        upscale=args.upscale,
        denoise=args.denoise,
        grayscale=args.grayscale,
        logical_order=False if args.raw_order else None,
        normalize=not args.raw_order,
        merge_words=not args.no_merge,
        merge_gap=args.merge_gap,
        det_unclip_ratio=args.unclip,
        max_crop_ratio=args.max_crop_ratio,
        refine_digits=args.refine_digits,
    )
    total_err = total_len = 0
    rows = []
    by_variant = {}
    for name, meta in sorted(truth.items()):
        if args.only and args.only not in name:
            continue
        path = args.data / name
        t = time.time()
        lines = ocr(path)
        dt = time.time() - t
        pred = norm("\n".join(lines))
        gold = norm("\n".join(meta["lines"]))
        err = edit_distance(pred, gold)
        total_err += err
        total_len += len(gold)
        v = by_variant.setdefault(meta["variant"], [0, 0])
        v[0] += err
        v[1] += len(gold)
        cer = err / max(len(gold), 1)
        rows.append((name, cer, len(lines), len(meta["lines"]), dt))
        if args.show_mismatch and err:
            print(f"--- {name}  CER={cer:.3f}")
            for a, b in zip(lines + [""] * (len(meta["lines"]) - len(lines)), meta["lines"]):
                flag = " " if norm(a) == norm(b) else "X"
                print(f" {flag} pred: {a}\n   gold: {b}")
    w = max(len(r[0]) for r in rows)
    print(f"{'image':<{w}}  CER     lines  time")
    for name, cer, nl, gl, dt in rows:
        print(f"{name:<{w}}  {cer:5.3f}  {nl:>2}/{gl:<2}  {dt:4.2f}s")
    print()
    for v, (e, n) in sorted(by_variant.items()):
        print(f"variant {v:<11} CER {e / max(n, 1):.4f}  ({e} errors / {n} chars)")
    print(f"TOTAL CER {total_err / max(total_len, 1):.4f}  ({total_err} errors / {total_len} chars)")


if __name__ == "__main__":
    main()
