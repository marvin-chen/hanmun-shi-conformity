"""
topic_model.py
LDA topic modeling pipeline for Korean hanmun 詩 corpus.
Character-level tokenization (Classical Chinese has no word boundaries).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import json
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import normalize
import matplotlib.font_manager as fm

# Use a CJK-capable system font on macOS
for fname in ["Arial Unicode MS", "PingFang SC", "STHeiti", "Heiti TC"]:
    if any(fname in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = fname
        break

# ── Config ────────────────────────────────────────────────────────────────────

FEATURES_PATH = "data/processed/klc_features.csv"
RESULTS_DIR   = "results/topics"
FIGURES_DIR   = "results/topics/figures"

N_TOPICS      = 15      # tune after coherence check
N_TOP_CHARS   = 20      # top characters per topic to display
MAX_FEATURES  = 3000    # vocabulary size
MIN_DF        = 5       # ignore chars appearing in fewer than 5 poems
MAX_DF        = 0.95    # ignore chars in >95% of poems (too generic)
RANDOM_STATE  = 42

# Classical Chinese high-frequency function characters to exclude
# These carry no thematic information — equivalent to English stopwords
STOPCHARS = set(
    "之也者而已矣乎哉於以為其所有無不亦則然若此彼何如是故且"
    "乃及與或若夫今既雖雖然皆各自當於焉云也耳爾兮兮歟邪耶夫"
    "一二三四五六七八九十百千萬上下左右中內外前後"
    "曰日月年時春夏秋冬"   # calendrical — strip for thematic purity
)

PERIOD_ORDER = ["pre_joseon", "early_joseon", "mid_joseon",
                "late_joseon", "final_joseon"]
PERIOD_LABELS = {
    "pre_joseon":    "Pre-Joseon\n(<1392)",
    "early_joseon":  "Early Joseon\n(1392–1550)",
    "mid_joseon":    "Mid Joseon\n(1550–1750)",
    "late_joseon":   "Late Joseon\n(1750–1894)",
    "final_joseon":  "Final Joseon\n(1894–1910)",
}


# ── Step 1: Load and filter ───────────────────────────────────────────────────

def load_general_poetry(path: str = FEATURES_PATH) -> pd.DataFrame:
    """Load only the unlabeled 詩 poems for topic modeling."""
    df = pd.read_csv(path)
    general = df[df["form_label"] == "general"].copy()
    general = general[general["text"].notna() & (general["body_len"] > 10)]
    print(f"Loaded {len(general):,} unlabeled 詩 poems for topic modeling")
    return general


# ── Step 2: Tokenize ──────────────────────────────────────────────────────────

def char_tokenize(text: str) -> str:
    """
    Classical Chinese tokenization: each character IS a token.
    Returns space-separated characters, excluding stopchars and non-CJK.
    """
    tokens = []
    for ch in str(text):
        code = ord(ch)
        # Keep only CJK Unified Ideographs block
        if 0x4E00 <= code <= 0x9FFF and ch not in STOPCHARS:
            tokens.append(ch)
    return " ".join(tokens)


def build_corpus(df: pd.DataFrame) -> tuple:
    """Tokenize all texts and build document-term matrix."""
    print("Tokenizing texts...")
    tokenized = df["text"].apply(char_tokenize).tolist()

    vectorizer = CountVectorizer(
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        max_df=MAX_DF,
        token_pattern=r"(?u)\b\w+\b",
    )
    dtm = vectorizer.fit_transform(tokenized)
    vocab = vectorizer.get_feature_names_out()
    print(f"Vocabulary size: {len(vocab):,} characters")
    print(f"DTM shape: {dtm.shape}")
    return dtm, vocab, vectorizer


# ── Step 3: Coherence proxy — perplexity curve ───────────────────────────────

def find_optimal_k(dtm, k_range=range(5, 31, 5), out_dir: str = FIGURES_DIR):
    """
    Fit LDA for multiple K values and plot perplexity.
    Lower perplexity = better fit (but watch for overfitting).
    Use this to pick N_TOPICS before the full run.
    """
    os.makedirs(out_dir, exist_ok=True)
    perplexities = []
    print("Scanning topic counts for optimal K...")

    for k in k_range:
        lda = LatentDirichletAllocation(
            n_components=k,
            max_iter=10,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        lda.fit(dtm)
        perp = lda.perplexity(dtm)
        perplexities.append(perp)
        print(f"  K={k:2d}  perplexity={perp:.1f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(k_range), perplexities, marker="o", color="#2980b9", linewidth=2)
    ax.set_xlabel("Number of Topics (K)", fontsize=11)
    ax.set_ylabel("Perplexity (lower = better)", fontsize=11)
    ax.set_title("LDA Perplexity vs. Number of Topics\n(Korean Hanmun 詩 Corpus)", fontsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(out_dir, "lda_perplexity_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nPerplexity curve saved → {path}")
    return dict(zip(k_range, perplexities))


# ── Step 4: Fit final LDA ─────────────────────────────────────────────────────

def fit_lda(dtm, n_topics: int = N_TOPICS):
    """Fit final LDA model with full iterations."""
    print(f"\nFitting LDA with K={n_topics} topics (full run)...")
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=50,
        learning_method="batch",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    lda.fit(dtm)
    print(f"  Final perplexity: {lda.perplexity(dtm):.1f}")
    return lda


# ── Step 5: Extract and display topics ───────────────────────────────────────

def get_top_chars(lda, vocab, n: int = N_TOP_CHARS) -> list[dict]:
    """Return top-N characters per topic with weights."""
    topics = []
    for i, comp in enumerate(lda.components_):
        top_idx = comp.argsort()[-n:][::-1]
        top_chars = [(vocab[j], round(comp[j], 2)) for j in top_idx]
        topics.append({
            "topic_id": i,
            "top_chars": top_chars,
            "label": "UNLABELED",   # fill in manually after inspection
        })
    return topics


def print_topics(topics: list[dict]):
    """Print topics in a readable format for manual labeling."""
    print("\n" + "="*60)
    print("TOPIC INSPECTION — label these manually")
    print("="*60)
    for t in topics:
        chars = "  ".join([f"{c}({w:.0f})" for c, w in t["top_chars"][:12]])
        print(f"\nTopic {t['topic_id']:2d}: {chars}")


def save_topics(topics: list[dict], out_dir: str):
    """Save topics to JSON for manual labeling and CSV for paper."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "topics.json"), "w", encoding="utf-8") as f:
        json.dump(topics, f, ensure_ascii=False, indent=2)

    rows = []
    for t in topics:
        rows.append({
            "topic_id": t["topic_id"],
            "label": t["label"],
            "top_chars": "  ".join([c for c, _ in t["top_chars"][:15]]),
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "topics_summary.csv"), index=False, encoding="utf-8-sig"
    )
    print(f"Topics saved → {out_dir}/topics.json")


# ── Step 6: Assign dominant topic per poem ───────────────────────────────────

def assign_topics(lda, dtm) -> np.ndarray:
    """Return dominant topic index per document."""
    doc_topic = lda.transform(dtm)           # shape: (n_docs, n_topics)
    return doc_topic.argmax(axis=1), doc_topic


# ── Step 7: Temporal topic analysis ──────────────────────────────────────────

def topic_by_period(df: pd.DataFrame, doc_topics: np.ndarray,
                    doc_topic_matrix: np.ndarray,
                    topics: list[dict]) -> pd.DataFrame:
    """
    Compute mean topic weight per period.
    Shows which topics dominate each Joseon era.
    """
    df = df.copy().reset_index(drop=True)
    df["dominant_topic"] = doc_topics

    # Add per-topic probability columns
    for i in range(doc_topic_matrix.shape[1]):
        df[f"topic_{i}_weight"] = doc_topic_matrix[:, i]

    period_df = df[df["period"].isin(PERIOD_ORDER)].copy()
    weight_cols = [f"topic_{i}_weight" for i in range(doc_topic_matrix.shape[1])]
    summary = period_df.groupby("period")[weight_cols].mean()
    summary = summary.reindex(PERIOD_ORDER)

    return summary, df


def plot_topic_trends(summary: pd.DataFrame, topics: list[dict],
                      top_n_topics: int = 8, out_dir: str = FIGURES_DIR):
    """
    Line chart: top-N topic weights over Joseon periods.
    Shows thematic shifts across time.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Pick top N topics by overall mean weight
    means = summary.mean(axis=0)
    top_cols = means.nlargest(top_n_topics).index.tolist()

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("tab10")

    for i, col in enumerate(top_cols):
        topic_id = int(col.split("_")[1])
        label = topics[topic_id]["label"]
        top_chars = "・".join([c for c, _ in topics[topic_id]["top_chars"][:5]])
        display_label = f"T{topic_id}: {label if label != 'UNLABELED' else top_chars}"

        ax.plot(
            range(len(PERIOD_ORDER)),
            summary.loc[PERIOD_ORDER, col].values,
            marker="o", linewidth=2.2, markersize=6,
            label=display_label, color=cmap(i),
        )

    ax.set_xticks(range(len(PERIOD_ORDER)))
    ax.set_xticklabels([PERIOD_LABELS[p] for p in PERIOD_ORDER], fontsize=9)
    ax.set_ylabel("Mean Topic Weight", fontsize=11)
    ax.set_title("Topic Distribution Across Joseon Periods\n(Korean Hanmun 詩 — 210k unlabeled poems)", fontsize=13)
    ax.legend(bbox_to_anchor=(1.01, 1), fontsize=8, title="Topic")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(out_dir, "topic_trends_over_time.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")


def plot_dominant_topic_stacked(df_with_topics: pd.DataFrame, topics: list[dict],
                                 out_dir: str = FIGURES_DIR):
    """
    Stacked bar chart: % of poems with each dominant topic, by period.
    Cleaner visual for the paper.
    """
    os.makedirs(out_dir, exist_ok=True)
    subset = df_with_topics[df_with_topics["period"].isin(PERIOD_ORDER)].copy()

    pivot = (
        subset.groupby(["period", "dominant_topic"])
        .size().unstack(fill_value=0)
        .reindex(PERIOD_ORDER)
    )
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    # Label columns with top characters
    col_labels = {}
    for col in pivot_pct.columns:
        t = topics[col]
        top = "".join([c for c, _ in t["top_chars"][:4]])
        lbl = t["label"] if t["label"] != "UNLABELED" else top
        col_labels[col] = f"T{col}:{lbl}"
    pivot_pct = pivot_pct.rename(columns=col_labels)

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_pct.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", edgecolor="none")
    ax.set_xticklabels([PERIOD_LABELS[p] for p in PERIOD_ORDER], rotation=0, fontsize=9)
    ax.set_ylabel("% of Poems", fontsize=11)
    ax.set_title("Dominant Topic Distribution in Korean Hanmun 詩\nby Joseon Period", fontsize=13)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(bbox_to_anchor=(1.01, 1), fontsize=7, title="Topic")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "dominant_topic_stacked.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_topic_pipeline(skip_k_scan: bool = False):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1. Load
    df = load_general_poetry()

    # 2. Build DTM
    dtm, vocab, vectorizer = build_corpus(df)

    # 3. Find optimal K (run once, then set N_TOPICS and skip)
    if not skip_k_scan:
        find_optimal_k(dtm, k_range=range(5, 31, 5))
        print("\n→ Inspect the perplexity curve, pick K, set N_TOPICS at the top of this file.")
        print("→ Then re-run with skip_k_scan=True\n")
        return

    # 4. Fit final model
    lda = fit_lda(dtm, n_topics=N_TOPICS)

    # 5. Topics
    topics = get_top_chars(lda, vocab)
    print_topics(topics)
    save_topics(topics, RESULTS_DIR)

    # 6. Assign
    doc_topics, doc_topic_matrix = assign_topics(lda, dtm)

    # 7. Temporal analysis
    summary, df_labeled = topic_by_period(df, doc_topics, doc_topic_matrix, topics)
    summary.to_csv(f"{RESULTS_DIR}/topic_weights_by_period.csv", encoding="utf-8-sig")
    df_labeled.to_csv(f"{RESULTS_DIR}/poems_with_topics.csv",
                      index=False, encoding="utf-8-sig")

    # 8. Figures
    plot_topic_trends(summary, topics, out_dir=FIGURES_DIR)
    plot_dominant_topic_stacked(df_labeled, topics, out_dir=FIGURES_DIR)

    print("\n✓ Topic modeling complete. Results in results/topics/")
    print("\nNEXT STEP: Open results/topics/topics.json and manually label each topic.")
    print("Then re-run plot functions with labeled topics for paper-ready figures.\n")


if __name__ == "__main__":
    # FIRST RUN: scan for optimal K
    # run_topic_pipeline(skip_k_scan=False)

    # SECOND RUN: after picking K from perplexity curve
    run_topic_pipeline(skip_k_scan=True)
