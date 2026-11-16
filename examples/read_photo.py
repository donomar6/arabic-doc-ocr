"""Read Arabic text from one image with arabic-doc-ocr and print it.

    pip install arabic-doc-ocr          # Windows x64, Python 3.10
    python read_photo.py photo.jpg
    python read_photo.py photo.jpg --details      # also confidence and position per line
    python read_photo.py photo.jpg -o out.txt     # save as UTF-8 file
"""
import argparse
import sys
from pathlib import Path

from arabic_ocr import ArabicOCR


def main() -> int:
    parser = argparse.ArgumentParser(description="Arabic OCR for one photo or scan.")
    parser.add_argument("image", type=Path, help="JPG, PNG, TIFF or BMP file")
    parser.add_argument("-o", "--output", type=Path, help="write the text here (UTF-8)")
    parser.add_argument("--details", action="store_true", help="print confidence and box per line")
    parser.add_argument("--upscale", type=float, default=1.0, help="use 2 for small text or phone photos")
    args = parser.parse_args()

    if not args.image.is_file():
        print(f"file not found: {args.image}", file=sys.stderr)
        return 1

    ocr = ArabicOCR(upscale=args.upscale)  # loads the models once; reuse for many images

    if args.details:
        for item in ocr.read(args.image):
            print(f"line {item['line']:>2}  conf {item['confidence']:.2f}  min {item['min_confidence']:.2f}  {item['text']}")
    else:
        text = ocr.read_text(args.image)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
            print(f"wrote {args.output}")
        else:
            try:
                sys.stdout.reconfigure(encoding="utf-8")  # Arabic on a Windows console
            except Exception:
                pass
            print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
