import jsonlines
import glob
import pandas as pd
from tqdm import tqdm
import os

# ── Poetry genre sets (from corpus exploration) ──────────────────────────────

LUSHI_GENRES = {
    "七言絶句", "詩○七言絶句",
    "五言絶句", "詩○五言絶句",
    "七言律詩", "詩○七言律詩",
    "五言律詩", "詩○五言律詩",
    "七言律",   "詩○七言律",
    "五言律",   "詩○五言律",
    "律詩",
    "五言排律",
    "七言四韻",
    "五言古詩", "詩○五言古詩",
    "七言古詩",
    "古律詩",
}

GENERAL_POETRY_GENRES = {"詩", "[詩]", "詩類"}
ALL_POETRY_GENRES = LUSHI_GENRES | GENERAL_POETRY_GENRES


# ── Normalize variant spellings → canonical form label ───────────────────────

def normalize_form(genre: str) -> str:
    g = genre.replace("詩○", "").strip()
    if g in ("七言絶句",):                       return "qijue"
    if g in ("五言絶句",):                       return "wujue"
    if g in ("七言律詩", "七言律", "七言四韻"):   return "qilu"
    if g in ("五言律詩", "五言律"):               return "wulu"
    if g in ("律詩",):                           return "lushi_generic"
    if g in ("五言排律",):                       return "wupailü"
    if g in ("五言古詩",):                       return "wugushi"
    if g in ("七言古詩",):                       return "qigushi"
    if g in ("古律詩",):                         return "gulushi"
    if g in ("詩", "[詩]", "詩類"):              return "general"
    return "other"


# ── Extract year: direct field → book_pub_year fallback ──────────────────────

def get_year(doc: dict):
    y = doc.get("year")
    if y and str(y).strip().isdigit():
        return int(str(y).strip())
    book_year = doc.get("metadata", {}).get("book_pub_year", "") or ""
    if str(book_year).strip().isdigit():
        return int(str(book_year).strip())
    return None


# ── Assign Joseon period from year ───────────────────────────────────────────

def assign_period(year) -> str:
    if year is None:          return "unknown"
    if year < 1392:           return "pre_joseon"
    elif year <= 1550:        return "early_joseon"
    elif year <= 1750:        return "mid_joseon"
    elif year <= 1894:        return "late_joseon"
    else:                     return "final_joseon"


# ── Main loader ───────────────────────────────────────────────────────────────

def load_klc_poetry(
    raw_dir: str = "data/raw",
    out_path: str = "data/processed/klc_poetry.csv"
) -> pd.DataFrame:

    os.makedirs("data/processed", exist_ok=True)
    records = []

    part_files = sorted(glob.glob(f"{raw_dir}/klc_part_*.jsonl"))
    print(f"Found {len(part_files)} part files\n")

    for fpath in part_files:
        with jsonlines.open(fpath) as reader:
            for doc in tqdm(reader, desc=os.path.basename(fpath)):

                content   = doc.get("content",        {}) or {}
                metadata  = doc.get("metadata",        {}) or {}
                analytics = doc.get("text_analytics",  {}) or {}

                # ── Genre extraction via title slash ──
                title_full = content.get("title", "") or ""
                genre = title_full.split("/")[-1].strip() if "/" in title_full else title_full.strip()

                if genre not in ALL_POETRY_GENRES:
                    continue

                year   = get_year(doc)
                period = assign_period(year)

                records.append({
                    # Identity
                    "id":           doc.get("id"),
                    "url":          metadata.get("elem_url", ""),

                    # Genre / form
                    "genre_raw":    genre,
                    "form_label":   normalize_form(genre),
                    "title_full":   title_full,

                    # Author / book
                    "author":       metadata.get("book_author", ""),
                    "book_name":    metadata.get("book_name", ""),
                    "book_id":      metadata.get("book_id", ""),

                    # Time
                    "year":         year,
                    "period":       period,

                    # Text — use unpunctuated body for analysis
                    "text":         content.get("body", ""),

                    # Lengths (for quick filtering without re-parsing text)
                    "body_len":     analytics.get("content_body_length", 0),
                })

    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Total poetry records:  {len(df):,}")
    print(f"\nBy form_label:")
    print(df["form_label"].value_counts().to_string())
    print(f"\nBy period:")
    print(df["period"].value_counts().to_string())
    print(f"\nYear coverage: {df['year'].notna().sum():,} / {len(df):,} records have a year")
    print(f"Year range:    {df['year'].min()} – {df['year'].max()}")
    print(f"\nSaved → {out_path}")

    return df


if __name__ == "__main__":
    df = load_klc_poetry()