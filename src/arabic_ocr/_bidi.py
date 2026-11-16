"""Visual-order to logical-order conversion for recogniser output.

The CTC recogniser reads a text line left to right, so for Arabic it emits the
characters in *visual* order (the string you would get by displaying the
logical text with the Unicode bidi algorithm and reading the glyphs from the
left). This module inverts that: numbers and Latin runs stay left to right,
everything else is reversed, and mirrored brackets are swapped back.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

_LATIN = "A-Za-zÀ-ɏ"
_EN = "0-9"
_AN = "٠-٩۰-۹"
_DIGITS = _EN + _AN
_L_RUN = re.compile(f"[{_LATIN}{_DIGITS}]+")

# W4/W5 of UAX#9 (simplified): a single separator between two digits, or a
# percent/currency sign next to a number, belongs to the number.
_NUMBER_RUN = re.compile(
    rf"(?:[{_DIGITS}]+(?:[.,:/٫٬،+\-][{_DIGITS}]+)*%?)"
)
_WORD_RUN = re.compile(rf"[{_LATIN}][{_LATIN}{_DIGITS}'\-.]*")

_MIRROR = {
    "(": ")", ")": "(", "[": "]", "]": "[", "{": "}", "}": "{", "<": ">", ">": "<",
    "«": "»", "»": "«", "‹": "›", "›": "‹",
}

_ARABIC_INDIC = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
_WESTERN = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _is_rtl_char(c: str) -> bool:
    return unicodedata.bidirectional(c) in ("R", "AL")


def _segments(visual: str) -> List[Tuple[str, bool]]:
    """Split a visual string into (text, keep_ltr) segments.

    Segments flagged ``keep_ltr`` are numbers or Latin words (plus the
    neutrals between two Latin words) and are kept as they are; the others are
    reversed when converting to logical order.
    """
    tokens: List[Tuple[str, str]] = []  # (text, class) class in {"L", "N", "X"} (X = neutral/RTL)
    i = 0
    n = len(visual)
    while i < n:
        m = _WORD_RUN.match(visual, i) or _NUMBER_RUN.match(visual, i)
        if m and m.end() > i:
            kind = "L" if _WORD_RUN.match(visual, i) else "N"
            tokens.append((m.group(0), kind))
            i = m.end()
            continue
        c = visual[i]
        if _is_rtl_char(c):
            tokens.append((c, "R"))
        else:
            tokens.append((c, "X"))
        i += 1

    # Neutrals between two Latin words take LTR (UAX#9 N1).
    resolved = []
    for idx, (text, kind) in enumerate(tokens):
        if kind == "X":
            prev_kind = next((k for _, k in reversed(tokens[:idx]) if k != "X"), None)
            next_kind = next((k for _, k in tokens[idx + 1 :] if k != "X"), None)
            if prev_kind == "L" and next_kind == "L":
                kind = "L"
        resolved.append((text, kind))

    segments: List[Tuple[str, bool]] = []
    for text, kind in resolved:
        keep = kind in ("L", "N")
        if segments and segments[-1][1] == keep:
            # Adjacent kept tokens (e.g. "12abc") form one LTR run; adjacent
            # reversed tokens form one RTL run. A neutral between two numbers
            # is RTL, so two numbers never merge.
            segments[-1] = (segments[-1][0] + text, keep)
        else:
            segments.append((text, keep))
    return segments


def visual_to_logical(visual: str) -> str:
    """Convert a left-to-right visual string into logical (storage) order."""
    if not visual:
        return visual
    out = []
    for text, keep in reversed(_segments(visual)):
        if keep:
            out.append(text)
        else:
            out.append("".join(_MIRROR.get(c, c) for c in reversed(text)))
    return "".join(out)


def normalize_digits(text: str) -> str:
    """Make each digit run use one digit system.

    The recogniser's dictionary has no Arabic-Indic zero (U+0660); it reads it
    as a Western ``0``. When a run mixes both systems, convert it to the system
    of the majority of its digits.
    """

    def fix(m: re.Match) -> str:
        run = m.group(0)
        an = sum(1 for c in run if "٠" <= c <= "٩")
        en = sum(1 for c in run if "0" <= c <= "9")
        if an and en:
            return run.translate(_ARABIC_INDIC) if an >= en else run.translate(_WESTERN)
        return run

    return _NUMBER_RUN.sub(fix, text)


# Persian/Urdu letter forms and Extended Arabic-Indic digits the models
# sometimes emit for Arabic text, mapped to their Arabic equivalents.
_CHAR_MAP = str.maketrans({
    "\u06a9": "\u0643",  # keheh -> kaf
    "\u06cc": "\u064a",  # farsi yeh -> yeh
    "\u06d5": "\u0647",  # ae -> heh
    "\u06c1": "\u0647",  # heh goal -> heh
    "\u06f0": "\u0660", "\u06f1": "\u0661", "\u06f2": "\u0662", "\u06f3": "\u0663", "\u06f4": "\u0664",
    "\u06f5": "\u0665", "\u06f6": "\u0666", "\u06f7": "\u0667", "\u06f8": "\u0668", "\u06f9": "\u0669",
})


def _is_presentation_form(c: str) -> bool:
    o = ord(c)
    return 0xFB50 <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF


def normalize_chars(text: str) -> str:
    """Map Persian-only letter forms and Extended Arabic-Indic digits to Arabic
    ones, and decompose Arabic presentation forms (e.g. the Allah ligature
    U+FDF2, lam-alef ligatures) into plain letters."""
    if any(_is_presentation_form(c) for c in text):
        text = "".join(unicodedata.normalize("NFKC", c) if _is_presentation_form(c) else c for c in text)
    return text.translate(_CHAR_MAP)


# A full stop after a number at the end of a sentence is drawn exactly like
# the Arabic-Indic zero. In visual order it sits at the far left, so after
# reordering it shows up as a leading zero of the line's last number.
_LEADING_ZERO_PERIOD = re.compile(r"(?<![0-9\u0660-\u0669])\u0660([\u0660-\u0669]{3,8}(?:[/.,:\-][\u0660-\u0669]+)*)\s*$")


_LEADING_ZERO_PERIOD_DATE = re.compile(r"^\u0660([\u0660-\u0669]{1,4}(?:[/.\-][\u0660-\u0669]{1,4}){1,2})\s*$")


def fix_trailing_period(text: str) -> str:
    """``الكود : ٠٤٦١٠٠٢`` -> ``الكود : ٤٦١٠٠٢.``

    Applied to lines with Arabic letters, and to lines that are only a
    date-like number (``٠٢٠١٢/٠٦/٢٥`` -> ``٢٠١٢/٠٦/٢٥.``). Plain numbers that
    start with a zero (codes, phone numbers) are left alone.
    """
    if has_arabic(text):
        return _LEADING_ZERO_PERIOD.sub(lambda m: m.group(1) + ".", text)
    return _LEADING_ZERO_PERIOD_DATE.sub(lambda m: m.group(1) + ".", text)


def has_arabic(text: str) -> bool:
    return any(_is_rtl_char(c) for c in text)
