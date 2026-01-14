"""Processor utilities for pronunciation inpainting.

Provides helpers to align original text characters to tokenizer subword tokens.

Main function:
- align_text_and_tokens(text, tokenizer, tokens=None):
    Returns (token_spans, char_to_token_indices)
    - token_spans: list of (start_char, end_char) per token (end_char exclusive)
    - char_to_token_indices: list (len == len(text)) of lists of token indices that cover the character

The implementation first tries to use the tokenizer's fast offset mapping (recommended).
If that's not available, it falls back to a best-effort greedy reconstruction using
`tokenizer.convert_tokens_to_string` per token.
"""

from typing import List, Tuple, Optional


def align_text_and_tokens(
    text: str, tokenizer, tokens: Optional[List[str]] = None
) -> Tuple[List[Tuple[int, int]], List[List[int]]]:
    """Align characters of `text` with subword `tokens` produced by `tokenizer`.

    Args:
        text: original string
        tokenizer: a HuggingFace tokenizer instance (preferably a fast tokenizer)
        tokens: optionally precomputed list of token strings (tokenizer.tokenize(text))

    Returns:
        token_spans: list of (start, end) character offsets (end exclusive) for each token
            If a token does not correspond to any character, its span will be (-1, -1)
        char_to_token_indices: list of lists, one per character in `text`, listing
            token indices that cover that character (can be empty if no token covers it)

    Notes:
        - Uses tokenizer(..., return_offsets_mapping=True, add_special_tokens=False)
          if available for exact offsets.
        - Falls back to greedy per-token reconstruction when offset mapping is unavailable.
    """
    if tokens is None:
        tokens = tokenizer.tokenize(text)

    # Try fast tokenizer offsets if available
    try:
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc.get("offset_mapping")
        if offsets is not None:
            # offsets may be tuples (start,end)
            token_spans: List[Tuple[int, int]] = []
            for s, e in offsets:
                if s is None or e is None:
                    token_spans.append((-1, -1))
                else:
                    token_spans.append((int(s), int(e)))
            # build char -> token indices
            char_to_tokens: List[List[int]] = [[] for _ in range(len(text))]
            for tidx, (s, e) in enumerate(token_spans):
                if s < 0 or e < 0:
                    continue
                for c in range(max(0, s), min(len(text), e)):
                    char_to_tokens[c].append(tidx)
            return token_spans, char_to_tokens
    except Exception:
        # tokenizer may not accept those args or may be a plain function; fallback below
        pass

    # Fallback greedy per-token reconstruction using convert_tokens_to_string
    token_spans = [(-1, -1) for _ in tokens]
    char_to_tokens: List[List[int]] = [[] for _ in range(len(text))]

    # Build normalized text for matching: we won't change the original text, but matching
    # will advance a cursor through it.
    cursor = 0
    for tidx, tok in enumerate(tokens):
        try:
            tok_str = tokenizer.convert_tokens_to_string([tok])
        except Exception:
            tok_str = tok
        # strip leading/trailing whitespace for matching purposes but remember if it had internal spaces
        tok_str_stripped = tok_str.strip()
        if tok_str_stripped == "":
            # Can't match empty reconstruction; leave (-1, -1)
            continue
        # Find the next occurrence of tok_str_stripped in text starting from cursor
        found = text.find(tok_str_stripped, cursor)
        if found == -1:
            # As a more permissive fallback, try to find anywhere in text
            found_any = text.find(tok_str_stripped)
            if found_any != -1:
                found = found_any
            else:
                # Last resort: try matching a single character (use first char of stripped tok)
                ch = tok_str_stripped[0]
                found_ch = text.find(ch, cursor)
                if found_ch != -1:
                    found = found_ch
                else:
                    # give up on matching this token
                    continue
        start = found
        end = found + len(tok_str_stripped)
        token_spans[tidx] = (start, end)
        for c in range(start, min(end, len(text))):
            char_to_tokens[c].append(tidx)
        cursor = end
    return token_spans, char_to_tokens


def tokenize_add_label(
    sample: dict,
    insert_prob: float = 0.25,
    seed: int = 42,
) -> dict:
    """Prepare sample by randomly inserting bracketed Jyutping annotations into text.

    New behavior (used prior to frontend processing):
      - If `sample["phone"]` is present and parsable, deterministically sample which
        syllables to reveal (insert) according to `insert_prob` and `seed`.
      - Construct `text_with_jyutping` which contains inline bracketed annotations
        (e.g., "你好[aa3]！"). This string will be consumed by the `InpaintFrontendWrapper`
        later to produce `text_token` and `phoneme_token` buffers.

    Returns a dict (kept simple so it is safe to run in dataset.map workers):
        - `text`: original text (unchanged)
        - `text_with_jyutping`: text with inserted bracketed tokens
        - `speech_token`: original speech tokens (if present)
        - `phone`: original phone string
        - `valid_phon`: True if `phone` was present and parsed into syllables, False otherwise

    Notes:
      - This function intentionally does not perform tokenization; the frontend wrapper
        will handle tokenization and alignment in a second mapping step (single-process)
        to avoid pickling heavy tokenizer/frontend objects across workers.
    """
    text = sample.get("text", "")
    speech_token = sample.get("speech_tokens", sample.get("speech_token", []))
    if isinstance(speech_token, str):
        speech_token = [int(i) for i in speech_token.split() if i != ""]

    phone_str = (sample.get("phone", "") or "").strip()

    # Quick reject if no phone annotation is present
    if phone_str == "":
        return {
            "text": text,
            "text_with_jyutping": text,
            "speech_token": speech_token,
            "phone": phone_str,
            "valid_phon": False,
        }

    # Try to obtain list of jyutping syllables (simple whitespace split)
    sylls = [p for p in phone_str.split() if p != ""]
    if len(sylls) == 0:
        return {
            "text": text,
            "text_with_jyutping": text,
            "speech_token": speech_token,
            "phone": phone_str,
            "valid_phon": False,
        }

    # Deterministic RNG seeded with sample text and provided seed.
    # Use a stable SHA-256-based integer derived from text+phone to avoid Python's
    # randomized `hash()` which varies per process and can break determinism.
    import random as _rand
    import hashlib

    key_str = (text or "") + "|" + (phone_str or "")
    stable_hash = int.from_bytes(
        hashlib.sha256(key_str.encode("utf-8")).digest()[:8], "little"
    )
    rng = _rand.Random(seed ^ (stable_hash & 0xFFFFFFFF))

    # Attempt word-wise insertion if counts match; otherwise fallback to best-effort
    words = [w for w in list(text) if w != ""]

    if len(words) == len(sylls) and len(words) > 0:
        # Insert syllables after corresponding words according to insert_prob
        new_words = []
        for w, s in zip(words, sylls):
            if rng.random() < insert_prob and s not in [".", ",", "!", "?"]:
                new_words.append(f"{w}[{s}]")
            else:
                new_words.append(w)

        # Rebuild text preserving original surrounding whitespace as best-effort
        # We replace the first occurrence of each word in sequence to avoid altering later duplicates.
        new_text = text
        cursor = 0
        for orig_w, new_w, s in zip(words, new_words, sylls):
            found = new_text.find(orig_w, cursor)
            if found == -1:
                found = new_text.find(orig_w)
            if found != -1:
                new_text = (
                    new_text[: found + len(orig_w)]
                    + (f"[{s}]" if new_w != orig_w else "")
                    + new_text[found + len(orig_w) :]
                )
                cursor = found + len(new_w)
    else:
        # Fallback: map syllables to first N non-space characters or first N words
        nonspace_chars = [i for i, c in enumerate(text) if not c.isspace()]
        if len(nonspace_chars) == len(sylls) and len(sylls) > 0:
            # Insert after characters
            chars = list(text)
            offset = 0
            for j, s in enumerate(sylls):
                if rng.random() < insert_prob:
                    idx = nonspace_chars[j] + 1 + offset
                    chars.insert(idx, f"[{s}]")
                    offset += 1
            new_text = "".join(chars)
        else:
            # Best-effort: annotate up to min(len(words), len(sylls)) words
            new_text = text
            K = min(len(words), len(sylls))
            cursor = 0
            for j in range(K):
                w = words[j]
                s = sylls[j]
                if rng.random() < insert_prob:
                    found = new_text.find(w, cursor)
                    if found == -1:
                        found = new_text.find(w)
                    if found != -1:
                        insert_pos = found + len(w)
                        new_text = (
                            new_text[:insert_pos] + f"[{s}]" + new_text[insert_pos:]
                        )
                        cursor = insert_pos + len(s) + 2

    return {
        "text": text,
        "text_with_jyutping": new_text,
        "speech_token": speech_token,
        "phone": phone_str,
        "valid_phon": True,
    }
