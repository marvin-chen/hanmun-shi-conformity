"""
analyzer.py
Main analysis pipeline for Korean hanmun shi conformity study.
Run this after data_loader.py to produce all results CSVs and charts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from feature_extractor import extract_features_batch
from conformity import (
    add_conformity_scores,
    period_conformity_summary,
    overall_period_conformity,
    pct_conformant,
    TANG_BASELINE,
    FORM_TEMPLATES,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH    = "data/processed/klc_poetry.csv"
FEATURES_PATH = "data/processed/klc_features.csv"
RESULTS_DIR  = "results"
FIGURES_DIR  = "results/figures"

PERIOD_ORDER  = ["early_joseon", "mid_joseon",
                 "late_joseon", "final_joseon"]
PERIOD_LABELS = {
    "pre_joseon":    "Pre-Joseon\n(<1392)",
    "early_joseon":  "Early Joseon\n(1392–1550)",
    "mid_joseon":    "Mid Joseon\n(1550–1750)",
    "late_joseon":   "Late Joseon\n(1750–1894)",
    "final_joseon":  "Final Joseon\n(1894–1910)",
}
FORM_COLORS = {
    "qijue": "#c0392b",
    "wujue": "#e67e22",
    "qilu":  "#2980b9",
    "wulu":  "#27ae60",
}


# ── Step 1: Feature extraction ────────────────────────────────────────────────

def run_feature_extraction(df: pd.DataFrame) -> pd.DataFrame:
    """Extract structural features for all poems and merge into df."""
    print(f"Extracting features for {len(df):,} poems...")
    feat_list = extract_features_batch(df["text"].tolist())

    # Flatten — drop list columns before saving to CSV
    flat = []
    for f in feat_list:
        flat.append({
            "num_lines":         f["num_lines"],
            "avg_line_length":   f["avg_line_length"],
            "std_line_length":   f["std_line_length"],
            "uniformity":        f["uniformity"],
            "is_uniform":        f["is_uniform"],
            "is_irregular":      f["is_irregular"],
            "modal_line_length": f["modal_line_length"],
            "end_chars":         "|".join(f["end_chars"]),  # join for CSV storage
        })

    feat_df = pd.DataFrame(flat)
    result = pd.concat([df.reset_index(drop=True), feat_df], axis=1)
    return result


# ── Step 2: Descriptive statistics ───────────────────────────────────────────

def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean structural metrics per period × form_label.
    Comparable to Tang baseline values from EAS407.
    """
    labeled = df[df["form_label"].isin(FORM_TEMPLATES.keys())].copy()
    labeled = labeled[labeled["period"].isin(PERIOD_ORDER)]

    stats = (
        labeled.groupby(["period", "form_label"])
        .agg(
            count=("num_lines", "count"),
            avg_lines=("num_lines", "mean"),
            avg_line_length=("avg_line_length", "mean"),
            pct_uniform=("is_uniform", lambda x: round(x.mean() * 100, 2)),
            pct_irregular=("is_irregular", lambda x: round(x.mean() * 100, 2)),
        )
        .reset_index()
    )
    return stats


# ── Step 3: End-character analysis ───────────────────────────────────────────

def endchar_analysis(df: pd.DataFrame, top_n: int = 20) -> dict:
    """
    Compute top-N end-characters per period.
    Returns dict of {period: Counter}.
    """
    results = {}
    for period in PERIOD_ORDER:
        subset = df[df["period"] == period]
        all_chars = []
        for chars_str in subset["end_chars"].dropna():
            all_chars.extend([c for c in chars_str.split("|") if c])
        results[period] = Counter(all_chars).most_common(top_n)
    return results


def endchar_diversity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute unique end-chars per 100 poems per period.
    Higher = more diverse rhyme vocabulary.
    Comparable to Tang (10.5) and Song (19.1) from EAS407.
    """
    rows = []
    for period in PERIOD_ORDER:
        subset = df[df["period"] == period]
        all_chars = []
        for chars_str in subset["end_chars"].dropna():
            all_chars.extend([c for c in chars_str.split("|") if c])
        n_poems = len(subset)
        n_unique = len(set(all_chars))
        per_100 = round((n_unique / n_poems) * 100, 2) if n_poems > 0 else 0
        rows.append({
            "period": period,
            "n_poems": n_poems,
            "unique_endchars": n_unique,
            "unique_per_100": per_100,
        })
    return pd.DataFrame(rows)


# ── Step 4: Author-level analysis ────────────────────────────────────────────

def author_conformity(df: pd.DataFrame, min_poems: int = 10) -> pd.DataFrame:
    """
    Compute mean conformity per author (only for authors with >= min_poems).
    Useful for identifying whether conformity is elite vs. widespread.
    """
    scored = df[df["conformity_score"].notna()].copy()
    author_stats = (
        scored.groupby("author")
        .agg(
            mean_conformity=("conformity_score", "mean"),
            n_poems=("conformity_score", "count"),
            period=("period", lambda x: x.mode()[0] if len(x) > 0 else "unknown"),
        )
        .reset_index()
    )
    return author_stats[author_stats["n_poems"] >= min_poems].sort_values(
        "mean_conformity", ascending=False
    )


# ── Step 5: Visualization ─────────────────────────────────────────────────────

def plot_conformity_over_time(summary_df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    for form, color in FORM_COLORS.items():
        data = summary_df[summary_df["form_label"] == form].copy()
        data["period"] = pd.Categorical(
            data["period"], categories=PERIOD_ORDER, ordered=True
        )
        data = data.sort_values("period")
        # Drop any periods with no data (e.g. Pre-Joseon for these forms)
        data = data[data["mean_conformity"].notna()]
        data = data[data["period"].isin(PERIOD_ORDER)]

        if len(data) == 0:
            continue

        # Use integer positions — prevents string-axis/set_xticks misalignment
        x_pos = [PERIOD_ORDER.index(str(p)) for p in data["period"]]

        ax.plot(
            x_pos,
            data["mean_conformity"],
            marker="o", linewidth=2.5, markersize=7,
            label=form, color=color,
        )
        # NO std shading — all four forms cluster 0.90–0.96, bands add clutter

    ax.set_xticks(range(len(PERIOD_ORDER)))
    ax.set_xticklabels([PERIOD_LABELS[p] for p in PERIOD_ORDER], fontsize=9)
    ax.set_ylabel("Mean Conformity Score (0–1)", fontsize=11)
    ax.set_title(
        "Structural Conformity to Tang Lüshi Norms\nby Joseon Period and Form",
        fontsize=13,
    )
    # Zoom in to where the data actually lives — 0–1 compresses all variation
    ax.set_ylim(0.85, 1.0)
    ax.legend(title="Form", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(out_dir, "conformity_over_time.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")


def plot_pct_uniform_by_period(df: pd.DataFrame, out_dir: str):
    """
    Bar chart: % of poems with fully uniform line lengths per period.
    Directly comparable to Tang (85.1%) baseline.
    """
    os.makedirs(out_dir, exist_ok=True)

    labeled = df[df["form_label"].isin(["qijue","wujue","qilu","wulu"])]
    labeled = labeled[labeled["period"].isin(PERIOD_ORDER)]

    pct = (
        labeled.groupby("period")["is_uniform"]
        .agg(lambda x: round(x.mean() * 100, 2))
        .reindex(PERIOD_ORDER)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        [PERIOD_LABELS[p] for p in PERIOD_ORDER],
        pct.values,
        color="#2980b9", edgecolor="white", linewidth=0.5,
    )
    # Tang baseline reference line
    ax.axhline(y=TANG_BASELINE["pct_uniform"], color="#c0392b",
               linestyle="--", linewidth=1.8, label=f"Tang baseline ({TANG_BASELINE['pct_uniform']}%)")

    ax.set_ylabel("% Poems with Uniform Line Lengths", fontsize=11)
    ax.set_title("Line-Length Uniformity in Korean Hanmun Shi\nvs. Tang Dynasty Baseline", fontsize=13)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(out_dir, "pct_uniform_by_period.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")


def plot_form_distribution(df: pd.DataFrame, out_dir: str):
    """
    Grouped bar chart: distribution of form_labels per period.
    Shows which forms Korean poets preferred in each era.
    """
    os.makedirs(out_dir, exist_ok=True)

    forms_of_interest = ["qijue", "wujue", "qilu", "wulu", "wugushi", "qigushi"]
    periods_for_forms = ["early_joseon", "mid_joseon", "late_joseon", "final_joseon"]

    labeled = df[df["form_label"].isin(forms_of_interest)]
    labeled = labeled[labeled["period"].isin(periods_for_forms)]

    pivot = (
        labeled.groupby(["period", "form_label"])
        .size().unstack(fill_value=0)
        .reindex(periods_for_forms)
        .reindex(columns=forms_of_interest, fill_value=0)
    )
    # Normalize to % within each period
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(11, 6))
    pivot_pct.plot(kind="bar", ax=ax, colormap="tab10", edgecolor="white")
    ax.set_xticklabels([PERIOD_LABELS[p] for p in periods_for_forms], rotation=0, fontsize=9)
    ax.set_ylabel("% of Labeled Poems", fontsize=11)
    ax.set_title("Distribution of Poetic Forms in Korean Hanmun Shi\nby Joseon Period", fontsize=13)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(title="Form", bbox_to_anchor=(1.01, 1), fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(out_dir, "form_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading KLC poetry data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df):,} poems\n")

    # ── Feature extraction ────────────────────────────────────────────────
    if os.path.exists(FEATURES_PATH):
        print(f"Features file found, loading from {FEATURES_PATH}")
        df = pd.read_csv(FEATURES_PATH)
    else:
        df = run_feature_extraction(df)
        df.to_csv(FEATURES_PATH, index=False, encoding="utf-8-sig")
        print(f"  Saved features → {FEATURES_PATH}\n")

    # ── Conformity scoring ────────────────────────────────────────────────
    print("Computing conformity scores...")
    df = add_conformity_scores(df)
    scored_count = df["conformity_score"].notna().sum()
    print(f"  Scored {scored_count:,} poems (forms with known templates)\n")

    # ── Descriptive stats ─────────────────────────────────────────────────
    print("Computing descriptive statistics...")
    desc = descriptive_stats(df)
    desc.to_csv(f"{RESULTS_DIR}/descriptive_stats.csv", index=False)
    print(desc.to_string(index=False))

    # ── Conformity summary ────────────────────────────────────────────────
    print("\nConformity by period × form:")
    conf_summary = period_conformity_summary(df)
    conf_summary.to_csv(f"{RESULTS_DIR}/conformity_by_period_form.csv", index=False)
    print(conf_summary.to_string(index=False))

    print("\nOverall conformity by period:")
    overall = overall_period_conformity(df)
    overall.to_csv(f"{RESULTS_DIR}/conformity_overall.csv", index=False)
    print(overall.to_string(index=False))

    # ── End-character diversity ───────────────────────────────────────────
    print("\nEnd-character diversity by period:")
    diversity = endchar_diversity(df)
    diversity.to_csv(f"{RESULTS_DIR}/endchar_diversity.csv", index=False)
    print(diversity.to_string(index=False))
    print(f"\n  Tang baseline: {TANG_BASELINE['unique_endchar_per_100']} unique end-chars per 100 poems")

    # ── Author analysis ───────────────────────────────────────────────────
    print("\nTop 20 authors by mean conformity (min 10 poems):")
    auth = author_conformity(df)
    auth.to_csv(f"{RESULTS_DIR}/author_conformity.csv", index=False)
    print(auth.head(20).to_string(index=False))

    # ── Figures ───────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_conformity_over_time(conf_summary, FIGURES_DIR)
    plot_pct_uniform_by_period(df, FIGURES_DIR)
    plot_form_distribution(df, FIGURES_DIR)

    print("\n✓ Pipeline complete. Results in results/")


if __name__ == "__main__":
    run_pipeline()