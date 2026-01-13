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
