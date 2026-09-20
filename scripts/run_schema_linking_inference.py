"""
Script 1: Schema Linking Inference (Standalone)
- Input: schema_linking_gt_nyc.json (800 queries)
- Model: outputs/stage1_model/best_model.pt
- Output: outputs/pipeline_results/step1_schema_linking_top30.json (compact, ~1-2MB)
- Each query keeps only top-30 (table_name, score) pairs
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

from src.table_identification.schema_linking_DL_crossencoder import (
    CrossEncoderSchemaLinker,
    CrossEncoderTrainer,
    GlobalTablePool
)

# ========== Configuration ==========
GT_FILE = str(PROJECT_ROOT / "data/benchmarks/NYC/schema_linking_gt_nyc.json")
MODEL_PATH = str(PROJECT_ROOT / "outputs/stage1_model/best_model.pt")
CSV_DATASET_ROOT = str(PROJECT_ROOT / "data/benchmarks/NYC")
OUTPUT_DIR = str(PROJECT_ROOT / "outputs/pipeline_results")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "step1_schema_linking_top30.json")
TOP_K = 30
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start_time = time.time()

    # 1. Load queries from GT file
    print(f"[1/4] Loading queries from {GT_FILE}...")
    with open(GT_FILE, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    # gt_data is a list of dicts, each with 'query', 'project_name', 'query_related_table', etc.
    queries = gt_data
    print(f"  Loaded {len(queries)} queries")

    # 2. Build table pool from CSV folders
    print(f"[2/4] Building table pool from {CSV_DATASET_ROOT}...")
    table_pool = GlobalTablePool()
    table_pool.build_from_csv_folders(CSV_DATASET_ROOT)
    all_tables = table_pool.get_all_tables()
    print(f"  Table pool: {len(all_tables)} tables")

    # 3. Load model and run inference
    print(f"[3/4] Loading model from {MODEL_PATH}...")
    model = CrossEncoderSchemaLinker(model_name='microsoft/deberta-v3-large', dropout=0.1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # Create trainer for inference
    trainer = CrossEncoderTrainer(
        model=model,
        train_dataloader=None,
        val_dataloader=None,
        val_data=None,
        train_table_pool=table_pool,
        test_data=queries,
        test_table_pool=table_pool,
        device=DEVICE
    )

    print(f"  Running inference on {len(queries)} queries (top_k={TOP_K})...")
    results, metrics = trainer.evaluate_on_full_table_pool(
        queries,
        top_k=TOP_K,
        max_samples=None,
        use_global_pool=True,
        save_detailed=True
    )

    print(f"  Metrics: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

    # 4. Save compact results (only top-30 per query)
    print(f"[4/4] Saving compact results to {OUTPUT_FILE}...")
    compact_results = []
    for i, r in enumerate(results):
        compact_results.append({
            "sample_id": r.get("sample_id", i),
            "query": r["query"],
            "project_name": queries[i].get("project_name", "unknown"),
            "table_scores": r["table_scores"][:TOP_K],  # top-30 only: [[table_name, score], ...]
            "predicted_tables": [ts[0] for ts in r["table_scores"][:TOP_K]],
            "query_related_table": queries[i].get("query_related_table", []),
            "precision": r.get("precision", 0),
            "recall": r.get("recall", 0),
            "f1": r.get("f1", 0)
        })

    # Save with metrics header
    output_data = {
        "metadata": {
            "total_queries": len(compact_results),
            "top_k": TOP_K,
            "model_path": MODEL_PATH,
            "metrics": metrics,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "results": compact_results
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=1)

    file_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Schema Linking Inference Complete!")
    print(f"  Queries: {len(compact_results)}")
    print(f"  Top-K: {TOP_K}")
    print(f"  Output: {OUTPUT_FILE} ({file_size:.2f} MB)")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  Avg Precision: {metrics['precision']:.4f}")
    print(f"  Avg Recall: {metrics['recall']:.4f}")
    print(f"  Avg F1: {metrics['f1']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
