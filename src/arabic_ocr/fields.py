"""Label-anchored field extraction with validation on top of :class:`ArabicOCR`.

Turns the raw page text into checked values for fixed-layout documents such
as commercial-registry extracts::

    from arabic_ocr.fields import FieldExtractor, REGISTRY_SCHEMA

    ex = FieldExtractor(schema=REGISTRY_SCHEMA, code_table="codes.csv")
    result = ex.extract("extract.jpg")
    result["fields"]["code"][0]["value"]        # "461002"
    result["fields"]["code"][0]["needs_review"] # False

Each field value carries ``valid``, ``confidence`` (weakest character in the
value), ``needs_review`` and ``reasons``. With ``passes=(1.0, 2.0)`` the page
is read twice at different scales and values that disagree are flagged.
"""
from __future__ import annotations

import csv
import datetime as _dt
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from PIL import Image

from ._api import ArabicOCR
from ._bidi import has_arabic
from ._geometry import crop_quad, edit_distance
from ._preprocess import to_pil

_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_DIACRITICS = re.compile(r"[ً-ْٰـ]")
_SEP = re.compile(r"^[\s:：.\-–—]+")

# Digits the recogniser confuses with each other (from measured errors).
CONFUSABLE_DIGITS = {"0": "5", "5": "0", "2": "3", "3": "2", "1": "7", "7": "1", "6": "9", "9": "6"}


def normalize_arabic(text: str) -> str:
    """Fold forms that OCR and typists mix: hamza/alef variants, taa marbuta, yeh, diacritics, digits."""
    text = _DIACRITICS.sub("", text)
    text = text.translate(_AR_DIGITS)
    text = re.sub("[آأإٱ]", "ا", text)  # آ أ إ ٱ -> ا
    text = text.replace("ة", "ه")  # ة -> ه
    text = text.replace("ى", "ي")  # ى -> ي
    text = text.replace("ک", "ك").replace("ی", "ي")
    return re.sub(r"\s+", " ", text).strip()


def western_digits(text: str) -> str:
    return text.translate(_AR_DIGITS)


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
@dataclass
class Field:
    """One field to extract.

    ``labels`` are the printed labels (any variant); matching is fuzzy on the
    normalised text. ``kind`` selects the validator: ``code`` (fixed-length
    digits, optional table), ``date``, ``digits`` (length range), ``text``,
    ``name`` (letters only). ``multiline`` lets the value continue on the
    following lines until the next label. ``repeat`` keeps every occurrence.
    """

    name: str
    labels: Sequence[str]
    kind: str = "text"
    length: Optional[int] = None  # exact digit count for code/digits
    min_length: int = 1
    max_length: int = 64
    multiline: bool = False
    repeat: bool = False
    vocabulary: Optional[Sequence[str]] = None  # closed set of allowed values (fuzzy matched)
    required: bool = False


REGISTRY_SCHEMA: List[Field] = [
    Field("main_activity", ["النشاط الأساسي", "النشاط الاساسي", "النشاط الرئيسي"], kind="text", multiline=True),
    Field("code", ["الكود", "كود النشاط", "رقم النشاط"], kind="code", length=6, repeat=True),
    Field("start_date", ["تاريخ بدء النشاط", "تاريخ البدء", "تاريخ بداية النشاط"], kind="date", repeat=True),
    Field("extra_activities", ["الأنشطة الإضافية", "الانشطة الاضافية", "أنشطة إضافية"], kind="text", multiline=True),
    Field("tax_id", ["الرقم الضريبي", "رقم التسجيل الضريبي"], kind="digits", length=9),
    Field("commercial_register", ["رقم السجل التجاري", "السجل التجاري", "رقم السجل"], kind="digits", min_length=4, max_length=12),
    Field("name", ["الاسم", "اسم المنشأة", "اسم العميل", "الاسم التجاري"], kind="name"),
    Field("national_id", ["الرقم القومي", "رقم الهوية", "رقم البطاقة"], kind="digits", length=14),
    Field("issue_date", ["تاريخ الإصدار", "تاريخ الاصدار"], kind="date"),
]


# --------------------------------------------------------------------------- #
# Validators
# --------------------------------------------------------------------------- #
def parse_date(text: str) -> Optional[_dt.date]:
    """Accept d/m/y and y/m/d with / - . separators, Arabic or Western digits."""
    t = western_digits(text)
    m = re.search(r"(\d{1,4})\s*[/.\-]\s*(\d{1,2})\s*[/.\-]\s*(\d{1,4})", t)
    if not m:
        return None
    a, b, c = (int(x) for x in m.groups())
    try:
        if a > 31:
            return _dt.date(a, b, c)  # y/m/d
        if c < 100:
            c += 2000
        return _dt.date(c, b, a)  # d/m/y
    except ValueError:
        return None


def validate(fld: Field, raw: str, code_table: Optional[Dict[str, Dict[str, str]]] = None) -> Tuple[Any, bool, List[str]]:
    """Return (value, valid, reasons)."""
    reasons: List[str] = []
    text = raw.strip()
    if fld.kind in ("code", "digits"):
        digits = re.sub(r"\D", "", western_digits(text))
        if fld.length and len(digits) != fld.length:
            reasons.append(f"expected {fld.length} digits, got {len(digits)}")
        elif not fld.length and not (fld.min_length <= len(digits) <= fld.max_length):
            reasons.append(f"expected {fld.min_length}-{fld.max_length} digits, got {len(digits)}")
        if fld.kind == "code" and code_table is not None and digits and not reasons:
            if digits not in code_table:
                fixed = repair_code(digits, code_table)
                if fixed:
                    reasons.append(f"code {digits} not in table, repaired to {fixed}")
                    digits = fixed
                else:
                    reasons.append(f"code {digits} not in table")
                    return digits, False, reasons
        return digits, not [r for r in reasons if "repaired" not in r], reasons
    if fld.kind == "date":
        d = parse_date(text)
        if d is None:
            return western_digits(text), False, ["not a date"]
        if d > _dt.date.today():
            reasons.append("date in the future")
        if d.year < 1900:
            reasons.append("year before 1900")
        return d.isoformat(), not reasons, reasons
    if fld.kind == "name":
        if re.search(r"\d", western_digits(text)):
            reasons.append("digits inside a name")
        if len(text) < 3:
            reasons.append("too short")
        return text, not reasons, reasons
    # text
    if fld.vocabulary:
        match, dist = best_match(text, fld.vocabulary)
        if match is not None:
            if dist > 0:
                reasons.append(f"matched vocabulary entry at distance {dist}")
            return match, True, reasons
        return text, False, ["no vocabulary match"]
    if len(text) < fld.min_length:
        return text, False, ["empty"]
    return text, True, reasons


def repair_code(code: str, table: Dict[str, Any]) -> Optional[str]:
    """Single confusable-digit substitution that yields exactly one known code."""
    cands = set()
    for i, c in enumerate(code):
        alt = CONFUSABLE_DIGITS.get(c)
        if alt:
            cand = code[:i] + alt + code[i + 1 :]
            if cand in table:
                cands.add(cand)
    return cands.pop() if len(cands) == 1 else None


def best_match(text: str, vocabulary: Iterable[str], max_ratio: float = 0.3, margin: int = 2) -> Tuple[Optional[str], int]:
    """Closest vocabulary entry by normalised edit distance; None when ambiguous or too far."""
    norm = normalize_arabic(text)
    scored = sorted(((edit_distance(norm, normalize_arabic(v)), v) for v in vocabulary), key=lambda t: t[0])
    if not scored:
        return None, 0
    dist, best = scored[0]
    if dist > max_ratio * max(len(norm), 1):
        return None, dist
    if len(scored) > 1 and scored[1][0] - dist < margin and scored[1][1] != best and dist > 0:
        return None, dist  # runner-up too close to call
    return best, dist


def load_code_table(path: Union[str, Path]) -> Dict[str, Dict[str, str]]:
    """CSV with a header; first column is the code, other columns are kept as lookup data."""
    table: Dict[str, Dict[str, str]] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        key = reader.fieldnames[0]
        for row in reader:
            code = re.sub(r"\D", "", western_digits(row[key]))
            if code:
                table[code] = {k: v for k, v in row.items() if k != key}
    return table


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
@dataclass
class _Line:
    text: str
    conf: float
    min_conf: float
    box: List[List[int]]
    index: int


def _match_label(line_text: str, labels: Sequence[str]) -> Optional[Tuple[str, str]]:
    """If the line starts with one of ``labels`` (fuzzy), return (label, remainder)."""
    norm = normalize_arabic(line_text)
    for label in labels:
        nl = normalize_arabic(label)
        head = norm[: len(nl)]
        tol = 1 if len(nl) <= 6 else 2
        if edit_distance(head, nl) <= tol:
            # remainder from the original (un-normalised) text; labels are at the start
            rest = line_text.strip()[len(label) :] if normalize_arabic(line_text.strip()[: len(label)]) == nl else line_text.strip()[len(head) :]
            return label, _SEP.sub("", rest).strip()
    return None


class FieldExtractor:
    def __init__(
        self,
        schema: Sequence[Field] = REGISTRY_SCHEMA,
        ocr: Optional[ArabicOCR] = None,
        code_table: Union[None, str, Path, Dict[str, Dict[str, str]]] = None,
        passes: Sequence[float] = (1.0,),
        review_threshold: float = 0.8,
        **ocr_kwargs: Any,
    ):
        self.schema = list(schema)
        self.ocr = ocr or ArabicOCR(**ocr_kwargs)
        self.code_table = load_code_table(code_table) if isinstance(code_table, (str, Path)) else code_table
        self.passes = tuple(passes)
        self.review_threshold = review_threshold

    # ------------------------------------------------------------------ #
    def _lines(self, image) -> List[_Line]:
        details = self.ocr.read(image)
        grouped: Dict[int, List[dict]] = {}
        for d in details:
            grouped.setdefault(d["line"], []).append(d)
        lines = []
        for k in sorted(grouped):
            ds = grouped[k]
            xs = [p[0] for d in ds for p in d["box"]]
            ys = [p[1] for d in ds for p in d["box"]]
            lines.append(_Line(" ".join(d["text"] for d in ds), min(d["confidence"] for d in ds), min(d["min_confidence"] for d in ds), [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]], k))
        return lines

    def _extract_once(self, lines: List[_Line]) -> Dict[str, List[dict]]:
        found: Dict[str, List[dict]] = {}
        i = 0
        while i < len(lines):
            line = lines[i]
            hit = None
            for fld in self.schema:
                m = _match_label(line.text, fld.labels)
                if m:
                    hit = (fld, m[1])
                    break
            if hit is None:
                i += 1
                continue
            fld, value = hit
            boxes = [line.box]
            confs = [line.min_conf]
            j = i + 1
            # value on the next line when the label line holds only the label
            if not value and j < len(lines) and not any(_match_label(lines[j].text, f.labels) for f in self.schema):
                value = lines[j].text.strip()
                boxes.append(lines[j].box)
                confs.append(lines[j].min_conf)
                j += 1
            if fld.multiline:
                while j < len(lines) and not any(_match_label(lines[j].text, f.labels) for f in self.schema):
                    value = (value + " " + lines[j].text.strip()).strip()
                    boxes.append(lines[j].box)
                    confs.append(lines[j].min_conf)
                    j += 1
            parsed, valid, reasons = validate(fld, value, self.code_table)
            entry = {
                "value": parsed,
                "raw": value,
                "valid": valid,
                "confidence": round(min(confs), 4),
                "reasons": reasons,
                "boxes": boxes,
            }
            if fld.kind == "code" and self.code_table and parsed in self.code_table:
                entry["lookup"] = self.code_table[parsed]
            found.setdefault(fld.name, []).append(entry)
            i = j if j > i + 1 else i + 1
        return found

    def extract(self, image, review_dir: Union[None, str, Path] = None) -> Dict[str, Any]:
        """Extract all schema fields. Returns ``{"fields": {...}, "lines": [...], "needs_review": [...]}``."""
        img = to_pil(image)
        base_upscale = self.ocr.upscale
        runs = []
        try:
            for scale in self.passes:
                self.ocr.upscale = float(scale)
                lines = self._lines(img)
                runs.append((scale, lines, self._extract_once(lines)))
        finally:
            self.ocr.upscale = base_upscale
        _, lines, fields = runs[0]
        # vote across passes: disagreement always means review, but the
        # reported value comes from a pass whose value validated, if any.
        for name, entries in fields.items():
            for k, entry in enumerate(entries):
                agree = True
                for scale, _, other in runs[1:]:
                    o = other.get(name, [])
                    if k >= len(o):
                        agree = False
                        entry["reasons"].append(f"missing at upscale {scale:g}")
                        continue
                    if o[k]["value"] != entry["value"]:
                        agree = False
                        entry["reasons"].append(f"upscale {scale:g} read {o[k]['value']!r}")
                        if o[k]["valid"] and not entry["valid"]:
                            entry["reasons"].append(f"value taken from upscale {scale:g}")
                            for key in ("value", "raw", "valid", "confidence", "boxes", "lookup"):
                                if key in o[k]:
                                    entry[key] = o[k][key]
                entry["passes_agree"] = agree if len(runs) > 1 else None
                entry["needs_review"] = (not entry["valid"]) or entry["confidence"] < self.review_threshold or agree is False
        for fld in self.schema:
            if fld.required and fld.name not in fields:
                fields[fld.name] = [{"value": None, "raw": "", "valid": False, "confidence": 0.0, "reasons": ["not found"], "boxes": [], "needs_review": True, "passes_agree": None}]
        review = [n for n, es in fields.items() if any(e["needs_review"] for e in es)]
        if review_dir:
            self.save_review(img, fields, review_dir)
        return {"fields": fields, "lines": [l.text for l in lines], "needs_review": review}

    @staticmethod
    def save_review(img: Image.Image, fields: Dict[str, List[dict]], out_dir: Union[str, Path]) -> List[Path]:
        """Save a crop of every field that needs review, named ``<field>_<n>.png``, plus ``review.txt``."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        notes = []
        for name, entries in fields.items():
            for k, e in enumerate(entries):
                if not e.get("needs_review") or not e["boxes"]:
                    continue
                xs = [p[0] for b in e["boxes"] for p in b]
                ys = [p[1] for b in e["boxes"] for p in b]
                pad = 6
                crop = img.crop((max(0, min(xs) - pad), max(0, min(ys) - pad), min(img.width, max(xs) + pad), min(img.height, max(ys) + pad)))
                path = out / f"{name}_{k}.png"
                crop.save(path)
                paths.append(path)
                notes.append(f"{path.name}\tguess={e['raw']}\tvalue={e['value']}\tconf={e['confidence']}\t{'; '.join(e['reasons'])}")
        (out / "review.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
        return paths
