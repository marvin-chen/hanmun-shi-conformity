
"""
run_features.py
Standalone script — run this FIRST after data_loader.py.
Extracts structural features and saves klc_features.csv.
Separated from the full analyzer so you can inspect features before analysis.
"""

import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from feature_extractor import extract_features_batch

DATA_PATH    = "data/processed/klc_poetry.csv"
FEATURES_PATH = "data/processed/klc_features.csv"

print("Loading poems...")
df = pd.read_csv(DATA_PATH)
print(f"  {len(df):,} poems loaded")

print("\nExtracting features...")
feat_list = extract_features_batch(df["text"].fillna("").tolist())

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
        "end_chars":         "|".join(f["end_chars"]),
    })

feat_df = pd.DataFrame(flat)
result = pd.concat([df.reset_index(drop=True), feat_df], axis=1)
result.to_csv(FEATURES_PATH, index=False, encoding="utf-8-sig")

print(f"\nSaved {len(result):,} rows → {FEATURES_PATH}")
print("\n── Feature summary (labeled forms only) ──")
labeled = result[result["form_label"] != "general"]
print(labeled[["form_label", "num_lines", "avg_line_length", "uniformity"]].groupby("form_label").mean().round(3))

print("\n── Uniformity rate by form ──")
print(labeled.groupby("form_label")["is_uniform"].mean().mul(100).round(1).sort_values(ascending=False))