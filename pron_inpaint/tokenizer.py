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

# Offsets for composing a single global phoneme id space (pad=0 reserved)
ONSET_OFFSET = 0
NUCLEUS_OFFSET = len(ONSETS)
CODA_OFFSET = NUCLEUS_OFFSET + len(NUCLEUSES)
TONE_OFFSET = CODA_OFFSET + len(CODAS)
# total vocab size (including pad=0)
TOTAL_PHONEME_VOCAB = TONE_OFFSET + len(TONES) + 1  # +1 for pad index 0

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
    """Parse phone_str and return flattened 4*L ids in a single global id space:
    [onset1..onsetL, nucleus1..nucleusL (offsetted), coda1..codaL (offsetted), tone1..toneL (offsetted)].

    This ensures phoneme ids are unique across components. Pad/index 0 remains reserved.
    Raises ValueError on length mismatch or None phone_str.
    """
    onset_l, nucleus_l, coda_l, tone_l = parse_phone_str(phone_str, L)
    onset_ids, nucleus_ids, coda_ids, tone_ids = components_to_ids(
        onset_l, nucleus_l, coda_l, tone_l
    )

    # apply offsets to make unique ids in single space
    def _offset_list(lst: List[int], offset: int) -> List[int]:
        return [i + offset if i != 0 else 0 for i in lst]

    onset_global = _offset_list(onset_ids, ONSET_OFFSET)
    nucleus_global = _offset_list(nucleus_ids, NUCLEUS_OFFSET)
    coda_global = _offset_list(coda_ids, CODA_OFFSET)
    tone_global = _offset_list(tone_ids, TONE_OFFSET)
    return onset_global + nucleus_global + coda_global + tone_global


class JyutpingTokenizer:
    """Tokenizer for Jyutping phoneme strings.

    Interface:
    - tokenize(list[str]) -> list[list[str]]
    - encode(list[str]) -> list[list[int]]  # flattened 4*L global ids (offsetted across components)
    - decode(flat_ids: List[int], original_tokens: Optional[List[str]] = None) -> str
    - vocab_size() -> int  # total phoneme vocab size including pad
    - component_vocab_sizes() -> Tuple[int,int,int,int]  # per-component vocab sizes including pad

    Notes:
    - Unknown tokens or punctuation map to id 0 (pad/UNK)
    - `encode` now returns globalized ids where each component uses a distinct id range
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
        """Encode a list of jyutping strings into flattened global ids (4*L per example).

        Each example maintains the component grouping but component ids are offset so
        they are unique across component types.
        """
        outs = []
        for s in texts:
            toks = s.strip().split()
            L = len(toks)
            flat = convert_phone_str_to_flat_ids(s, L)
            outs.append(flat)
        return outs

    def decode(self, flat_ids: List[int]) -> str:
        """Decode flattened ids into a reconstructed Jyutping string.

        - If multiple items (batch), returns a list of decoded strings.
        - For tokens with all-zero components (no phoneme), returns "[UNK]" for that slot.
        """
        if isinstance(flat_ids[0], list):
            # batch
            return [self.decode(x) for x in flat_ids]
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
            # global ids: convert back to per-component local ids before lookup
            def _local(gid, offset):
                return gid - offset if gid != 0 else 0

            o = self.onset_inv.get(_local(o_ids[i], ONSET_OFFSET), "")
            n = self.nucleus_inv.get(_local(n_ids[i], NUCLEUS_OFFSET), "")
            c = self.coda_inv.get(_local(c_ids[i], CODA_OFFSET), "")
            t = self.tone_inv.get(_local(t_ids[i], TONE_OFFSET), "")
            if o == "" and n == "" and c == "" and t == "":
                toks.append("[UNK]")
            else:
                # Reconstruct Jyutping token as onset + nucleus + coda + tone when available
                tok = "".join([p for p in (o, n, c) if p != ""]) + (
                    t if t != "" else ""
                )
                toks.append(tok if tok != "" else "[UNK]")
        return " ".join(toks)

    def vocab_size(self) -> int:
        """Total phoneme vocab size (sum of component sizes, including pad idx 0)."""
        return TOTAL_PHONEME_VOCAB

    def __len__(self) -> int:
        return self.vocab_size()


# small self-test
if __name__ == "__main__":
    tok = JyutpingTokenizer()
    s = "hou2 aa3 !"
    print("tokenize:", tok.tokenize([s]))
    enc = tok.encode([s])[0]
    print("encoded len", len(enc), enc)
    print("decoded:", tok.decode(enc, original_tokens=s.split()))
