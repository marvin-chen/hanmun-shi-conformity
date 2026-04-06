"""
feature_extractor.py
Extracts structural features from Classical Chinese / Hanmun poetry texts.
Designed for the KLC corpus (unpunctuated body field).
"""

import re
import unicodedata
from typing import Optional

# ── Punctuation to strip before analysis ─────────────────────────────────────
# KLC body field is mostly unpunctuated, but some records include these
PUNCT = set("。，、；：？！「」『』【】《》〈〉…—·\u3000\n\r\t "
            ",.;:?!()[]{}\"'\\/ ")

# Characters that mark line boundaries in unpunctuated classical text
LINE_BOUNDARY_PUNCT = {"。", "，", "\n"}


def strip_punct(text: str) -> str:
    """Remove punctuation and whitespace from a character."""
    return "".join(c for c in text if c not in PUNCT and not unicodedata.category(c).startswith("P"))


SERIES_MARKERS = {"又", "其", "○", "△"}  # characters that mark "next poem in series"

def split_lines(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []

    raw_lines = text.strip().split("\n")
    lines = [strip_punct(l) for l in raw_lines if strip_punct(l)]

    if len(lines) <= 1:
        raw_lines = text.replace("，", "").split("。")
        lines = [strip_punct(l) for l in raw_lines if strip_punct(l)]

    # Strip leading series markers from each line
    cleaned = []
    for l in lines:
        # Remove leading series marker characters
        while l and l[0] in SERIES_MARKERS:
            l = l[1:]
        if l:  # only keep non-empty lines after stripping
            cleaned.append(l)

    return cleaned


def extract_features(text: str) -> dict:
    """
    Extract structural features from a single poem.

    Returns a dict with:
        num_lines          : int   — total number of lines
        avg_line_length    : float — mean characters per line
        std_line_length    : float — std dev of line lengths
        uniformity         : float — fraction of lines matching the modal length (0.0–1.0)
        is_uniform         : bool  — True if ALL lines are the same length
        is_irregular       : bool  — True if lines vary in length (not uniform)
        modal_line_length  : int   — most common line length
        end_chars          : list  — last character of each line (for rhyme analysis)
        line_lengths       : list  — raw list of per-line character counts
    """
    lines = split_lines(text)

    if not lines:
        return _empty_features()

    lengths = [len(l) for l in lines]
    n = len(lengths)
    avg = sum(lengths) / n
    variance = sum((x - avg) ** 2 for x in lengths) / n
    std = variance ** 0.5

    # Modal line length (most common)
    from collections import Counter
    modal = Counter(lengths).most_common(1)[0][0]
    modal_count = lengths.count(modal)
    uniformity = modal_count / n

    is_uniform = (len(set(lengths)) == 1)
    is_irregular = not is_uniform

    end_chars = [l[-1] for l in lines if len(l) > 0]

    return {
        "num_lines":        n,
        "avg_line_length":  round(avg, 3),
        "std_line_length":  round(std, 3),
        "uniformity":       round(uniformity, 3),
        "is_uniform":       is_uniform,
        "is_irregular":     is_irregular,
        "modal_line_length": modal,
        "end_chars":        end_chars,
        "line_lengths":     lengths,
    }


def _empty_features() -> dict:
    return {
        "num_lines":         0,
        "avg_line_length":   0.0,
        "std_line_length":   0.0,
        "uniformity":        0.0,
        "is_uniform":        False,
        "is_irregular":      True,
        "modal_line_length": 0,
        "end_chars":         [],
        "line_lengths":      [],
    }


# ── Batch extraction ──────────────────────────────────────────────────────────

def extract_features_batch(texts: list[str], verbose: bool = True) -> list[dict]:
    """Run extract_features over a list of texts with optional tqdm progress."""
    try:
        from tqdm import tqdm
        iterator = tqdm(texts, desc="Extracting features") if verbose else texts
    except ImportError:
        iterator = texts

    return [extract_features(t) for t in iterator]


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Perfect qijue (7-char quatrain) — should be uniform, 4 lines, avg 7.0
    test_qijue = "床前明月光\n疑是地上霜\n舉頭望明月\n低頭思故鄕"
    # Perfect wulu (5-char regulated verse) — 8 lines, avg 5.0
    test_wulu = "國破山河在\n城春草木深\n感時花濺淚\n恨別鳥驚心\n烽火連三月\n家書抵萬金\n白頭搔更短\n渾欲不勝簪"

    for name, poem in [("qijue", test_qijue), ("wulu", test_wulu)]:
        f = extract_features(poem)
        print(f"\n[{name}]")
        print(f"  lines={f['num_lines']}, avg_len={f['avg_line_length']}, "
              f"uniform={f['is_uniform']}, end_chars={f['end_chars']}")