"""Jyutping tokenizer for pronunciation inpainting

Provides:
- ONSETS, NUCLEUSES, CODAS, TONES (lists of strings)
- maps (string->id) with index 0 reserved for padding / no-phoneme
- parse_phone_str(phone_str: str, L: int) -> (onset_list, nucleus_list, coda_list, tone_list)
- convert_phone_str_to_flat_ids(phone_str: str, L: int) -> List[int] (flattened, length 4*L)

This module uses `pycantonese` when available for robust parsing and falls back to a
conservative regex if not.
"""

from typing import List, Tuple, Optional
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


class JyutpingTokenizer:
    """Tokenizer for Jyutping phoneme strings.

    Interface inspired by HuggingFace tokenizers (lightweight):
    - tokenize(list[str]) -> list[list[str]]
    - encode(list[str]) -> list[list[int]]  # flattened 4*L ids
    - decode(flat_ids: List[int], original_tokens: Optional[List[str]] = None) -> str
    - vocab_size() -> [onset_size, nucleus_size, coda_size, tone_size]
    - __len__ returns total vocab size (sum)

    Notes:
    - Unknown tokens or punctuation map to id 0 (pad/UNK)
    - `encode` works on a list of input strings and returns a list of flattened id lists
    - `decode` reconstructs tokens from ids. If `original_tokens` is provided it will be
      used to recover non-phoneme tokens (punctuation) where all component ids == 0.
    """

    def __init__(self):
        self.onset_map = ONSET_MAP
        self.nucleus_map = NUCLEUS_MAP
        self.coda_map = CODA_MAP
        self.tone_map = TONE_MAP

        # reverse maps
        self.onset_inv = {v: k for k, v in self.onset_map.items()}
        self.nucleus_inv = {v: k for k, v in self.nucleus_map.items()}
        self.coda_inv = {v: k for k, v in self.coda_map.items()}
        self.tone_inv = {v: k for k, v in self.tone_map.items()}

    def tokenize(self, texts: List[str]) -> List[List[str]]:
        return [[t for t in s.strip().split()] for s in texts]

    def encode(self, texts: List[str]) -> List[List[int]]:
        outs = []
        for s in texts:
            toks = s.strip().split()
            L = len(toks)
            onset_ids, nucleus_ids, coda_ids, tone_ids = [], [], [], []
            for tok in toks:
                o, n, c, t = parse_jyutping_token(tok)
                onset_ids.append(self.onset_map.get(o, 0))
                nucleus_ids.append(self.nucleus_map.get(n, 0))
                coda_ids.append(self.coda_map.get(c, 0))
                tone_ids.append(self.tone_map.get(t, 0))
            flat = onset_ids + nucleus_ids + coda_ids + tone_ids
            outs.append(flat)
        return outs

    def decode(
        self, flat_ids: List[int], original_tokens: Optional[List[str]] = None
    ) -> str:
        if isinstance(flat_ids[0], list):
            # batch
            return [
                self.decode(x, orig)
                for x, orig in zip(flat_ids, original_tokens or [None] * len(flat_ids))
            ]
        L4 = len(flat_ids)
        if L4 % 4 != 0:
            raise ValueError("flat_ids length must be divisible by 4")
        L = L4 // 4
        o_ids = flat_ids[0:L]
        n_ids = flat_ids[L : 2 * L]
        c_ids = flat_ids[2 * L : 3 * L]
        t_ids = flat_ids[3 * L : 4 * L]
        toks = []
        for i in range(L):
            o = self.onset_inv.get(o_ids[i], "")
            n = self.nucleus_inv.get(n_ids[i], "")
            c = self.coda_inv.get(c_ids[i], "")
            t = self.tone_inv.get(t_ids[i], "")
            if o == "" and n == "" and c == "" and t == "":
                if original_tokens is not None:
                    toks.append(original_tokens[i])
                else:
                    toks.append("[UNK]")
            else:
                # prioritize nucleus+tone if available, otherwise concatenate present parts
                if n != "":
                    tok = n + (t if t != "" else "")
                else:
                    parts = [p for p in (o, n, c, t) if p != ""]
                    tok = "".join(parts) if parts else "[UNK]"
                toks.append(tok)
        return " ".join(toks)

    def vocab_size(self) -> List[int]:
        return [
            len(self.onset_map) + 1,
            len(self.nucleus_map) + 1,
            len(self.coda_map) + 1,
            len(self.tone_map) + 1,
        ]

    def __len__(self) -> int:
        return sum(self.vocab_size())


# small self-test
if __name__ == "__main__":
    tok = JyutpingTokenizer()
    s = "hou2 aa3 !"
    print("tokenize:", tok.tokenize([s]))
    enc = tok.encode([s])[0]
    print("encoded len", len(enc), enc)
    print("decoded:", tok.decode(enc, original_tokens=s.split()))
