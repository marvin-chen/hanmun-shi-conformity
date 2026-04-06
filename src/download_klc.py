# src/download_klc.py
from huggingface_hub import hf_hub_download
import os

os.makedirs("data/raw", exist_ok=True)

for i in range(1, 9):
    part = f"klc_part_{i:03d}_of_008.jsonl"
    print(f"Downloading {part}...")
    hf_hub_download(
        repo_id="seyoungsong/Open-Korean-Historical-Corpus",
        filename=part,
        repo_type="dataset",
        local_dir="data/raw"
    )
    print(f"  ✓ {part}")

print("Done. All 8 KLC parts downloaded.")