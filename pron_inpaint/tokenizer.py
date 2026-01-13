"""Jyutping tokenizer for pronunciation inpainting

Provides:
- ONSETS, NUCLEUSES, CODAS, TONES (lists of strings)
- maps (string->id) with index 0 reserved for padding / no-phoneme
- parse_phone_str(phone_str: str, L: int) -> (onset_list, nucleus_list, coda_list, tone_list)
- convert_phone_str_to_flat_ids(phone_str: str, L: int) -> List[int] (flattened, length 4*L)

This module uses `pycantonese` when available for robust parsing and falls back to a
conservative regex if not.
"""

from typing import List, Tuple
import re

# Fixed component vocabularies (index 0 reserved for padding / no-phoneme)
ONSETS = "b d g gw z p t k kw c m n ng f h s l w j".split()
NUCLEUSES = "aa a i yu u oe e eo o m n ng".split()
CODAS = "p t k m n ng i u".split()
TONES = "1 2 3 4 5 6".split()

# maps: string -> id (1..N); 0 is pad/no-phoneme
ONSET_MAP = {v: i + 1 for i, v in enumerate(ONSETS)}
NUCLEUS_MAP = {v: i + 1 for i, v in enumerate(NUCLEUSES)}
CODA_MAP = {v: i + 1 for i, v in enumerate(CODAS)}
TONE_MAP = {v: i + 1 for i, v in enumerate(TONES)}

# regex to detect a simple jyutping token like 'aa3' or 'gwong2'
_JYUTPING_RE = re.compile(r"^[a-zA-Z]+[1-9]$")

try:
    import pycantonese  # type: ignore

    _HAS_PYCANTONESE = True
except Exception:
    _HAS_PYCANTONESE = False


def parse_jyutping_token(tok: str) -> Tuple[str, str, str, str]:
    """Parse a single jyutping-like token into (onset, nucleus, coda, tone) strings.

    If parsing fails or token is not jyutping, returns empty strings.
    """
    if tok is None:
        return "", "", "", ""
    if not _JYUTPING_RE.match(tok):
        return "", "", "", ""
    if _HAS_PYCANTONESE:
        try:
            parsed = pycantonese.parse_jyutping(tok)
            syll = parsed[0]
            try:
                return syll.onset, syll.nucleus, syll.coda, str(syll.tone)
            except Exception:
                return syll[0], syll[1], syll[2], str(syll[3])
        except Exception:
            pass
    # fallback: split letters and trailing digit
    m = re.match(r"^([a-zA-Z]+?)([1-9])$", tok)
    if m:
        onset = ""
        nucleus = m.group(1)
        coda = ""
        tone = m.group(2)
        return onset, nucleus, coda, tone
    return "", "", "", ""


def parse_phone_str(
    phone_str: str, L: int
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Parse a space-separated phone_str into component lists.

    Args:
        phone_str: e.g. "hou2 aa3 !"
        L: expected number of tokens (should equal tokenized text length)

    Returns:
        onset_list, nucleus_list, coda_list, tone_list

    Raises:
        ValueError: if phone_str is None or split length != L
    """
    if phone_str is None:
        raise ValueError("phone_str is None")
    parts = [p for p in phone_str.strip().split()]
    if len(parts) != L:
        raise ValueError(f"phone length {len(parts)} != expected text length {L}")
    onset_l, nucleus_l, coda_l, tone_l = [], [], [], []
    for tok in parts:
        o, n, c, t = parse_jyutping_token(tok)
        onset_l.append(o)
        nucleus_l.append(n)
        coda_l.append(c)
        tone_l.append(t)
    return onset_l, nucleus_l, coda_l, tone_l


def components_to_ids(
    onset_l: List[str], nucleus_l: List[str], coda_l: List[str], tone_l: List[str]
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Map string component lists to id lists (0 for pad/not found)."""
    onset_ids = [ONSET_MAP.get(s, 0) for s in onset_l]
    nucleus_ids = [NUCLEUS_MAP.get(s, 0) for s in nucleus_l]
    coda_ids = [CODA_MAP.get(s, 0) for s in coda_l]
    tone_ids = [TONE_MAP.get(s, 0) for s in tone_l]
    return onset_ids, nucleus_ids, coda_ids, tone_ids


def convert_phone_str_to_flat_ids(phone_str: str, L: int) -> List[int]:
    """Parse phone_str and return flattened 4*L ids: [onset..., nucleus..., coda..., tone...].

    Raises ValueError on length mismatch or None phone_str.
    """
    onset_l, nucleus_l, coda_l, tone_l = parse_phone_str(phone_str, L)
    onset_ids, nucleus_ids, coda_ids, tone_ids = components_to_ids(
        onset_l, nucleus_l, coda_l, tone_l
    )
    return onset_ids + nucleus_ids + coda_ids + tone_ids


# Simple self-test when run as a script
if __name__ == "__main__":
    phones = "hou2 aa3 !"
    try:
        o, n, c, t = parse_phone_str(phones, 3)
        print("parsed:", o, n, c, t)
        flat = convert_phone_str_to_flat_ids(phones, 3)
        print("flat ids:", flat)
    except Exception as e:
        print("parse error:", e)
