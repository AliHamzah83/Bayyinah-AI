"""Conservative Arabic and mixed-script text normalization helpers."""

from __future__ import annotations

import re
import unicodedata

_ARABIC_DIACRITICS = re.compile("[\\u0610-\\u061A\\u064B-\\u065F\\u0670\\u06D6-\\u06ED]")
_BIDI_CONTROLS = re.compile("[\\u061C\\u200E\\u200F\\u202A-\\u202E\\u2066-\\u2069]")
_UNWANTED_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\uFFFD]")
_WHITESPACE = re.compile(r"\s+")
_ARABIC_LETTERS = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
_LATIN_LETTERS = re.compile(r"[A-Za-z]")
_ARABIC_CHAR_MAP = str.maketrans(
    {
        "آ": "ا",
        "أ": "ا",
        "إ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ک": "ك",
        "ی": "ي",
        "ـ": "",
    }
)


def normalize_arabic_text(text: str, *, normalize_teh_marbuta: bool = False) -> str:
    """Normalize Unicode and common Arabic orthographic variants for retrieval.

    The caller should retain the original text separately for display and citations.
    Punctuation and ta marbuta are preserved unless its optional folding is requested.
    """
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _BIDI_CONTROLS.sub("", normalized)
    normalized = _UNWANTED_CONTROL_CHARACTERS.sub(" ", normalized)
    normalized = _ARABIC_DIACRITICS.sub("", normalized)
    normalized = normalized.translate(_ARABIC_CHAR_MAP)
    if normalize_teh_marbuta:
        normalized = normalized.replace("ة", "ه")
    return _WHITESPACE.sub(" ", normalized).strip()


def detect_language(text: str) -> str:
    """Classify a text sample by Arabic and Latin letter presence."""
    has_arabic = bool(_ARABIC_LETTERS.search(text))
    has_latin = bool(_LATIN_LETTERS.search(text))
    if has_arabic and has_latin:
        return "mixed"
    if has_arabic:
        return "arabic"
    if has_latin:
        return "english"
    return "unknown"