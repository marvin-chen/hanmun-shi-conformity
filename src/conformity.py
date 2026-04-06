"""
conformity.py
Computes lüshi conformity scores for Korean hanmun shi poems.
Compares KLC poems against Tang dynasty structural norms.
"""

import pandas as pd
import numpy as np
from typing import Optional

# ── Tang baseline from EAS407 (Chen 2025) ────────────────────────────────────
TANG_BASELINE = {
    "pct_uniform":            85.1,   # % of Tang poems with uniform line lengths
    "avg_lines":               4.66,  # avg lines per poem
    "avg_line_length":        11.74,  # avg chars per line (incl. all chars)
    "pct_irregular":          14.9,   # % of Tang poems with irregular line lengths
    "unique_endchar_per_100": 10.5,   # unique end-chars per 100 poems
}

# ── Expected structural templates per lüshi form ──────────────────────────────
# Source: Classical Chinese prosodic taxonomy
FORM_TEMPLATES = {
    "qijue":       {"num_lines": 4, "line_length": 7},  # 七言絶句
    "wujue":       {"num_lines": 4, "line_length": 5},  # 五言絶句
    "qilu":        {"num_lines": 8, "line_length": 7},  # 七言律詩
    "wulu":        {"num_lines": 8, "line_length": 5},  # 五言律詩
    "lushi_generic": {"num_lines": 8, "line_length": None},  # either 5 or 7
    "wupailü":     {"num_lines": None, "line_length": 5},    # extended, variable length
    "wugushi":     {"num_lines": None, "line_length": 5},    # old style, variable count
    "qigushi":     {"num_lines": None, "line_length": 7},
    "gulushi":     {"num_lines": None, "line_length": None},
}

# Tolerance thresholds
LINE_LENGTH_TOLERANCE = 0.5   # avg_line_length must be within ±0.5 of expected
LINE_COUNT_TOLERANCE  = 0     # num_lines must match exactly for jueju/lüshi


def conformity_score(row: pd.Series) -> Optional[float]:
    """
    Compute a 0.0–1.0 conformity score for a single poem row.

    Three components (equal weight, 1/3 each):
        1. Line count matches template
        2. Avg line length matches template (within tolerance)
        3. All lines are uniform length (is_uniform)

    Returns None for forms without a strict template (general, other).
    """
    form = row.get("form_label")
    template = FORM_TEMPLATES.get(form)

    if template is None or form in ("general", "other", "gulushi"):
        return None

    score = 0.0
    components = 0

    # Component 1: line count — accept exact OR integer multiples
    # (handles series poems: 2 quatrains = 8 lines, 3 = 12, etc.)
    expected_lines = template.get("num_lines")
    if expected_lines is not None:
        components += 1
        actual = row.get("num_lines", 0)
        if actual > 0 and actual % expected_lines == 0:
            score += 1.0

    # Component 2: line length 
    expected_length = template.get("line_length")
    if expected_length is not None:
        components += 1
        actual_length = row.get("avg_line_length", 0)
        if abs(actual_length - expected_length) <= 0.5:
            score += 1.0

    # Component 3: uniformity 
    components += 1
    if row.get("is_uniform", False):
        score += 1.0

    return round(score / components, 4) if components > 0 else None


def add_conformity_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add conformity_score column to a features dataframe."""
    df = df.copy()
    df["conformity_score"] = df.apply(conformity_score, axis=1)
    return df


def period_conformity_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate conformity scores by period and form_label.
    Returns a summary table with mean, std, count per group.
    Only includes rows with a valid conformity_score.
    """
    scored = df[df["conformity_score"].notna()].copy()

    summary = (
        scored.groupby(["period", "form_label"])["conformity_score"]
        .agg(mean_conformity="mean", std_conformity="std", count="count")
        .reset_index()
    )

    # Order periods chronologically
    period_order = ["pre_joseon", "early_joseon", "mid_joseon",
                    "late_joseon", "final_joseon", "unknown"]
    summary["period"] = pd.Categorical(
        summary["period"], categories=period_order, ordered=True
    )
    summary = summary.sort_values(["form_label", "period"])
    return summary


def overall_period_conformity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate conformity scores by period only (collapsed across forms).
    Useful for the main diachronic trend chart.
    """
    scored = df[df["conformity_score"].notna()].copy()

    summary = (
        scored.groupby("period")["conformity_score"]
        .agg(mean_conformity="mean", std_conformity="std", count="count")
        .reset_index()
    )

    period_order = ["pre_joseon", "early_joseon", "mid_joseon",
                    "late_joseon", "final_joseon"]
    summary["period"] = pd.Categorical(
        summary["period"], categories=period_order, ordered=True
    )
    return summary.sort_values("period")


def pct_conformant(df: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    """
    Compute % of poems scoring at or above a conformity threshold per period.
    Default threshold=1.0 means fully conformant (all 3 components pass).
    """
    scored = df[df["conformity_score"].notna()].copy()
    scored["is_conformant"] = scored["conformity_score"] >= threshold

    result = (
        scored.groupby("period")["is_conformant"]
        .agg(pct_conformant=lambda x: round(x.mean() * 100, 2), count="count")
        .reset_index()
    )
    return result


if __name__ == "__main__":
    # Minimal smoke test
    test_row = pd.Series({
        "form_label":      "qijue",
        "num_lines":       4,
        "avg_line_length": 7.0,
        "is_uniform":      True,
    })
    print("qijue perfect score:", conformity_score(test_row))  # expected: 1.0

    test_row2 = pd.Series({
        "form_label":      "wulu",
        "num_lines":       6,       # wrong
        "avg_line_length": 5.1,     # close
        "is_uniform":      True,
    })
    print("wulu partial score:", conformity_score(test_row2))  # expected: ~0.667