"""
Script 2: DeepJoin Prediction (Standalone)
- Input: All NYC tables (data/benchmarks/NYC/)
- Model: outputs/stage2_deepjoin_model/sentence_transformer/
- Output: outputs/pipeline_results/step4_deepjoin_top6.json (compact)
- Each table pair keeps only top-6 column pairs with probabilities
"""

import os
import sys
import json
import time
import torch
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph_validation.DeepJoin import DeepJoinClassify

# ========== Configuration ==========
TABLES_DIR = str(PROJECT_ROOT / "data/benchmarks/NYC")
MODEL_DIR = str(PROJECT_ROOT / "outputs/stage2_deepjoin_model")
OUTPUT_DIR = str(PROJECT_ROOT / "outputs/pipeline_results")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "step4_deepjoin_top6.json")
TOP_K = 6


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start_time = time.time()

    print(f"[DeepJoin Prediction]")
    print(f"  Tables dir: {TABLES_DIR}")
    print(f"  Model dir: {MODEL_DIR}")
    print(f"  Top-K per table pair: {TOP_K}")
    print(f"  Output: {OUTPUT_FILE}")

    # Run DeepJoin prediction
    # DeepJoinClassify saves results to output file directly
    DeepJoinClassify(
        mode='predict',
        tables=TABLES_DIR,
        model_dir=MODEL_DIR,
        links=None,
        output=OUTPUT_FILE,
        top_k=TOP_K,
        isfolder=True
    )

    # Verify output
    if os.path.exists(OUTPUT_FILE):
        file_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
        with open(OUTPUT_FILE, 'r') as f:
            data = json.load(f)
        
        # Count unique table pairs
        table_pairs = set()
        for item in data:
            pair = tuple(sorted([item['table1'], item['table2']]))
            table_pairs.add(pair)

        elapsed = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"DeepJoin Prediction Complete!")
        print(f"  Total join records: {len(data)}")
        print(f"  Unique table pairs: {len(table_pairs)}")
        print(f"  Output: {OUTPUT_FILE} ({file_size:.2f} MB)")
        print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
        print(f"{'='*60}")
    else:
        print(f"ERROR: Output file not created!")


if __name__ == "__main__":
    main()
