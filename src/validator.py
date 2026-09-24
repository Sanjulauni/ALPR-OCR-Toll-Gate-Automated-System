"""Validation and correction of OCR output against Sri Lankan license-plate formats.

Sri Lankan plates in circulation follow two patterns once punctuation/whitespace
is stripped:
  - Pre-2013 format: 1-2 letters followed by 4 digits   e.g. "KA1234", "N1234"
  - Post-2013 format: 2-3 letters followed by 4 digits  e.g. "CAB1234", "KY1234"

Older plates also carry a small province code before the registration (e.g.
"WP KM-5141"). OCR often reads it as part of the plate, so it is stripped when
the full text does not fit the plate pattern but the remainder does.

This does not attempt to validate against the maintained list of issued
province/series letter codes -- only the general shape of the plate.
"""
import re
from dataclasses import dataclass
from typing import Optional

_PLATE_PATTERN = re.compile(r"^[A-Z]{1,3}\d{4}$")

# Characters PaddleOCR commonly confuses on the low-resolution plate crops
# produced by toll-gate cameras.
_DIGIT_TO_LETTER = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z"}
_LETTER_TO_DIGIT = {letter: digit for digit, letter in _DIGIT_TO_LETTER.items()}

# Province prefixes: Western, Central, Southern, Northern, Eastern, North
# Western, North Central, Uva, Sabaragamuwa.
PROVINCE_CODES = ("WP", "CP", "SP", "NP", "EP", "NW", "NC", "UP", "SG")

_SPLIT_PATTERN = re.compile(r"^([A-Z0-9]+?)([A-Z0-9]{4})$")


@dataclass
class ValidationResult:
    is_valid: bool
    normalized_text: str
    original_text: str
    province: Optional[str] = None


def correct_common_errors(text: str) -> str:
    """Swap letter/digit look-alikes so the string fits <letters><4 digits>."""
    text = text.upper()
    if _PLATE_PATTERN.match(text):
        return text

    match = _SPLIT_PATTERN.match(text)
    if not match:
        return text

    letters_part, digits_part = match.groups()
    fixed_letters = "".join(_DIGIT_TO_LETTER.get(ch, ch) for ch in letters_part)
    fixed_digits = "".join(_LETTER_TO_DIGIT.get(ch, ch) for ch in digits_part)
    return fixed_letters + fixed_digits


def strip_province_prefix(text: str):
    """Return (text_without_prefix, province) if a province code was removed."""
    text = text.upper()
    if _PLATE_PATTERN.match(text) or not text.startswith(PROVINCE_CODES):
        return text, None
    remainder = correct_common_errors(text[2:])
    if _PLATE_PATTERN.match(remainder):
        return remainder, text[:2]
    return text, None


def validate_plate(text: str) -> ValidationResult:
    normalized = correct_common_errors(text)
    normalized, province = strip_province_prefix(normalized)
    return ValidationResult(
        is_valid=bool(_PLATE_PATTERN.match(normalized)),
        normalized_text=normalized,
        original_text=text,
        province=province,
    )
