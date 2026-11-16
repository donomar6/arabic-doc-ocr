# arabic-doc-ocr

Offline Arabic OCR for scanned documents, in one self-contained wheel.

* Reads Arabic text plus Arabic-Indic (٠١٢٣٤٥٦٧٨٩) and Western digits from
  JPG, PNG, TIFF, BMP (anything Pillow opens), PIL images or numpy arrays.
* **Fully offline.** The detection and recognition models and a private copy
  of ONNX Runtime (CPU) are inside the wheel. Nothing is downloaded at install
  or run time, ever.
* Only two declared dependencies: `numpy>=1.21` and `Pillow>=9.0`. Works with
  numpy 1.23.5; numpy 2 is not required. No OpenCV, no Shapely, no pyclipper.
* Lines come back top to bottom in **logical order** (what you would type), not
  the visually reversed string that raw CTC recognisers emit for RTL text.

Models: PP-OCRv4 text detection (DBNet) and the PP-OCRv5 Arabic recogniser
(default) or the PP-OCRv4 Arabic recogniser, converted to ONNX by the
[RapidOCR](https://github.com/RapidAI/RapidOCR) project from
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (Apache-2.0). The
pre- and post-processing (DB box extraction, crop rectification, CTC decoding,
line ordering, bidi reordering) is re-implemented with numpy and Pillow.

## Install

Wheels are built per platform and interpreter. Currently published:
`cp310-cp310-win_amd64` (Windows x64, CPython 3.10).

```bash
pip install arabic-doc-ocr
```

### Offline install

On a machine with internet access, download the wheel (and, if the target
machine does not have them yet, numpy and Pillow):

```bash
pip download arabic-doc-ocr --only-binary=:all: --python-version 3.10 --platform win_amd64 -d wheels
```

Copy the `wheels` folder to the offline machine and install with no index:

```bash
pip install --no-index --find-links wheels arabic-doc-ocr
# or, if numpy and Pillow are already installed:
pip install --no-index arabic_doc_ocr-0.1.0-cp310-cp310-win_amd64.whl
```

Windows needs the Microsoft Visual C++ 2015-2022 redistributable (x64), which
is already present on almost every machine (it ships with Python itself).

## Usage

```python
from arabic_ocr import ArabicOCR

ocr = ArabicOCR()

lines = ocr("scan.jpg")            # list of str, top to bottom, logical order
text = ocr.read_text("scan.jpg")   # one string, lines joined with "\n"
details = ocr.read("scan.jpg")     # list of dicts: text, confidence, box, line
```

`details` entries look like:

```python
{"text": "الكود : ٤٦١٠٠٢", "confidence": 0.97,
 "box": [[295, 48], [544, 44], [545, 92], [295, 96]],   # tl, tr, br, bl (x, y)
 "line": 0}
```

Input can be a path, a `PIL.Image.Image`, or a numpy array (`HxW` gray,
`HxWx3` RGB or `HxWx4` RGBA, uint8).

### Options

```python
ocr = ArabicOCR(
    model="v5",          # "v5" (PP-OCRv5 Arabic, default) or "v4" (PP-OCRv4 Arabic)
    upscale=1.0,         # 1.5-2.0 helps small text / low-dpi scans
    auto_invert=True,    # invert light-on-dark text, decided per detected line
    grayscale=False,     # drop colour noise
    denoise=0,           # median filter size (odd), e.g. 3 for speckled scans
    min_confidence=0.5,  # drop lines below this mean character confidence
    merge_words=True,    # merge word boxes of a line before recognition
    max_crop_ratio=8.0,  # split line crops wider than 8x their height at whitespace gaps
    refine_digits=False, # experimental digits-only re-read of numbers
    logical_order=None,  # None = right choice per model; False = raw recogniser order
    normalize=True,      # Persian forms -> Arabic, ligatures decomposed, digit runs unified
    num_threads=None,    # onnxruntime intra-op threads
)
```

Detector thresholds (`det_thresh`, `det_box_thresh`, `det_unclip_ratio`,
`det_limit_side_len`, `det_limit_type`, `det_max_side_len`) default to the
RapidOCR values.

### Field extraction with validation

For fixed-layout documents (commercial-registry extracts, licences, ID
cards) use the field layer instead of raw text. It anchors on printed labels,
validates each value by type, checks codes against your table, reads the page
at two scales and flags anything that does not agree.

```python
from arabic_ocr.fields import FieldExtractor, REGISTRY_SCHEMA, Field

ex = FieldExtractor(schema=REGISTRY_SCHEMA, code_table="activity_codes.csv", passes=(1.0, 2.0))
res = ex.extract("extract.jpg", review_dir="review/extract")

res["fields"]["code"][0]
# {"value": "461002", "raw": "٤٦١٠٠٢.", "valid": True, "confidence": 0.93,
#  "reasons": [], "passes_agree": True, "needs_review": False,
#  "lookup": {"activity": "مكاتب التصدير والاستيراد", "risk": "high"}, "boxes": [...]}
res["needs_review"]      # names of fields an analyst must look at
```

`activity_codes.csv` has a header; the first column is the code, the other
columns come back under `lookup`. Unknown codes are repaired by a single
confusable-digit substitution only when exactly one known code results, and
the repair is recorded in `reasons`. `review_dir` receives a crop image per
flagged field plus `review.txt`.

`REGISTRY_SCHEMA` covers Egyptian commercial-registry extracts (main
activity, activity code, start date, extra activities, tax id, register
number, name, national id, issue date). Build your own schema from `Field`:

```python
schema = [
    Field("licence_no", ["رقم الرخصة"], kind="digits", min_length=5, max_length=10, required=True),
    Field("expiry", ["تاريخ الانتهاء"], kind="date"),
    Field("profession", ["المهنة"], kind="text", vocabulary=["مهندس", "طبيب", "محاسب"]),
]
```

Validators: `code` (exact digit count + table), `digits` (length range),
`date` (d/m/y or y/m/d, Arabic or Western digits, not in the future), `name`
(no digits), `text` (optional closed vocabulary, fuzzy matched on normalised
Arabic). `confidence` is the weakest character in the value, not the line
average. `needs_review` is true when the value is invalid, the confidence is
under `review_threshold` (0.8) or the passes disagree.

Command line equivalent:

```bash
arabic-ocr extract.jpg --fields --code-table activity_codes.csv --passes 1,2 --review-dir review
```

### Command line

```bash
arabic-ocr scan.jpg                      # prints UTF-8 text
arabic-ocr scan.jpg -o out.txt           # writes UTF-8 file
arabic-ocr scan.jpg --json               # per-box JSON with boxes and confidences
arabic-ocr scan.jpg --upscale 2 --denoise 3 --grayscale
arabic-ocr scan.jpg --model v4
```

## Accuracy

Measured on synthetic pages rendered with Pillow + arabic_reshaper +
python-bidi (Tahoma, DejaVu Sans as an Arial stand-in, Noto Naskh Arabic as a
Simplified Arabic stand-in, and Amiri) at 38-44 px, including blurred,
noisy, rotated, JPEG-compressed "scan" variants. Character error rate
(Levenshtein distance over the whole page, lines in reading order):

| Variant | PP-OCRv5 (default) | PP-OCRv4 |
|---|---|---|
| clean render | 0.066 | 0.153 |
| scan (blur 1.2, noise, 0.6° skew, JPEG 60) | 0.090 | 0.101 |
| heavy scan (blur 1.8, more noise, 1.5° skew, JPEG 35) | 0.205 | 0.217 |
| inverted (light on dark) | 0.000 | 0.341 |
| small text (16 px) with `upscale=2` | 0.035 | 0.129 |
| **all** | **0.117** | **0.158** |

Logical order is verified in the tests: `"الكود : ٤٦١٠٠٢"` and
`"تاريخ بدء النشاط : ٢٠١٢/٠٦/٢٥"` come back exactly as written. The
re-implemented pipeline matches the official RapidOCR implementation on the
same images (0.328 vs 0.341 CER with the v4 recogniser, before the wide-line
splitting that brings v4 down to 0.158). Splitting line crops wider than 8x
their height at whitespace gaps (`max_crop_ratio`) is the single largest gain:
the recognisers are trained on ~7:1 crops and silently drop digits and words on
30:1 lines.

Regenerate the images with `python tools/make_test_images.py` (uses the real
Windows fonts when run on Windows) and the numbers with
`python tools/eval_cer.py [--model v4]`.

## Limitations

* **CPU only**, mobile-sized models. Expect roughly 0.5-2 s per A4 page at
  300 dpi on a modern laptop.
* The PP-OCRv4 Arabic recogniser's dictionary has **no Arabic-Indic zero**
  (`٠`); it reads it as `0` (unified by `normalize=True`). The v5 model does
  not have this problem, which is why it is the default.
* The default settings are tuned for printed documents. Handwriting, strongly
  rotated pages (no orientation classifier is bundled) and vertical text are
  not supported. Light skew (a few degrees) is fine.
* Diacritics (tashkeel) are often dropped; punctuation standing alone between
  words (e.g. ` : `) is sometimes missed by the detector.
* The v5 recogniser tends to **skip Latin words embedded in an Arabic line**
  (e.g. `(OCR)` inside an Arabic sentence) while reading them fine when they
  stand alone. Digits in Arabic lines are read correctly.
* OCR output is evidence, not a decision. For regulated uses (KYC, risk
  classification) route every invalid, low-confidence or disagreeing field to
  a person, log image, OCR output and rule hits per case, and fail towards the
  safer class.
* The visual-to-logical reordering uses a simplified bidi algorithm: numbers
  and Latin runs keep their direction, mirrored brackets are swapped back.
  Nested bidi (Arabic inside Latin inside Arabic) is not handled.
* Only one wheel per Python version / platform. To add another, run
  `python tools/build_wheel.py --target <plat> --python <ver>` on any host.

## Development

```bash
pip install build setuptools wheel twine pillow numpy arabic_reshaper python-bidi fonttools
python tools/build_wheel.py --target win_amd64 --python 3.10 --keep-vendor   # also vendors ORT into src/ for local runs
python tools/make_test_images.py        # renders tests/data
python tools/eval_cer.py                # CER report
python -m unittest discover -s tests    # PYTHONPATH=src
```

`--keep-vendor` leaves the vendored onnxruntime for the *build host's* platform
in `src/arabic_ocr/_vendor` so the package runs from source. Build for the
host platform first when developing, then for the release target.

## How the wheel is built

```
tools/vendor_onnxruntime.py  copies the onnxruntime wheel for the *target* platform into
                             arabic_ocr/_vendor/onnxruntime, drops everything not needed for
                             CPU inference, rewrites imports to the private namespace and gives
                             the extension module its own pybind11 internals id so it can live
                             next to any onnxruntime the user has installed.
tools/build_wheel.py         runs the above and builds with the target's wheel tag.
```

Nothing is added to `sys.path` and no module is registered under the name
`onnxruntime`, so an installed onnxruntime, OpenCV or anything else is never
shadowed.

## License

The package code is Apache-2.0. Bundled third-party components and their
licenses are listed in `THIRD_PARTY_NOTICES.md`:

* ONNX Runtime (MIT), Microsoft Corporation.
* PP-OCR models (Apache-2.0), PaddlePaddle / RapidAI.
