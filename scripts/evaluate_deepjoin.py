#!/usr/bin/env python3
"""
Evaluate DeepJoin model on NYC benchmark using split_statistics.json as ground truth.

For each project in split_statistics.json:
  - Extract ground truth join columns from edge_join_columns + node_mapping
  - Load sub-tables from NYC benchmark directory
  - Use trained DeepJoin model to predict join relationships
  - Compare predictions vs ground truth

Metrics:
  - Edge-level: whether the correct table pair is predicted (regardless of column)
  - Column-level: whether the correct (table1, table2, col1, col2) is predicted
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from src.graph_validation.DeepJoin import (
    DeepJoinClassifier, GlobalOptimizer, Table,
    load_tables_from_folder_structure, normalize_table_name
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DeepJoin-Eval")


def extract_ground_truth(split_stats: dict, nyc_dir: str):
    """
    Extract ground truth join relationships from split_statistics.json.

    Returns:
        list of dicts, each containing:
            - project_name: str
            - category: str
            - folder_name: str (project_name + "_" + category)
            - gt_joins: list of (subtable1_file, subtable2_file, join_columns)
              where join_columns is a list of column names used for the join
    """
    projects = []

    for category, items in split_stats.items():
        if not isinstance(items, list):
            continue  # Skip non-list entries like 'summary'
        for item in items:
            table_name = item["table_name"]
            cat = item["category"]
            folder_name = f"{table_name}_{cat}"
            edge_join_cols = item.get("edge_join_columns", {})
            node_mapping = item.get("node_mapping", {})

            gt_joins = []
            for edge_key, join_cols in edge_join_cols.items():
                # edge_key is like "1-2", "2-3", etc.
                parts = edge_key.split("-")
                if len(parts) != 2:
                    continue
                node1, node2 = parts[0], parts[1]

                # Get subtable file names from node_mapping
                if node1 in node_mapping and node2 in node_mapping:
                    st1 = node_mapping[node1].get("subtable_file", "")
                    st2 = node_mapping[node2].get("subtable_file", "")
                    if st1 and st2:
                        gt_joins.append({
                            "table1": st1,
                            "table2": st2,
                            "join_columns": join_cols,
                            "edge": edge_key
                        })

            if gt_joins:
                projects.append({
                    "project_name": table_name,
                    "category": cat,
                    "folder_name": folder_name,
                    "gt_joins": gt_joins,
                    "node_mapping": node_mapping
                })

    return projects


def load_project_tables(nyc_dir: str, folder_name: str):
    """Load all sub-tables for a project from its folder."""
    folder_path = Path(nyc_dir) / folder_name
    if not folder_path.exists():
        return {}

    tables = {}
    import pandas as pd
    for csv_file in sorted(folder_path.glob("*.csv")):
        try:
            df = pd.read_csv(csv_file, nrows=200, low_memory=False)
            columns = {}
            for col in df.columns:
                columns[col] = df[col].tolist()
            # unique_id = folder_name/stem (without .csv)
            unique_id = f"{folder_name}/{csv_file.stem}"
            t = Table(
                table_name=csv_file.stem,
                columns=columns,
                unique_id=unique_id
            )
            tables[unique_id] = t
        except Exception as e:
            logger.warning(f"Failed to load {csv_file}: {e}")
    return tables


def evaluate_project(clf, project, nyc_dir, top_k=6):
    """
    Evaluate DeepJoin predictions for a single project.

    Returns:
        dict with edge-level and column-level metrics
    """
    folder_name = project["folder_name"]
    gt_joins = project["gt_joins"]

    # Load tables
    tables = load_project_tables(nyc_dir, folder_name)
    if not tables:
        return None

    table_list = list(tables.values())
    if len(table_list) < 2:
        return None

    # Build embeddings
    clf.build_embeddings(table_list)

    # Predict
    opt = GlobalOptimizer(clf, top_k)
    predictions = opt.predict_bi_model(table_list)

    # Build ground truth sets
    # Edge-level: set of (table1_stem, table2_stem) pairs (unordered)
    gt_edges = set()
    # Column-level: set of (table1_stem, table2_stem, col) tuples
    gt_col_joins = set()

    for gj in gt_joins:
        t1 = gj["table1"]  # e.g. "Property Valuation..._1"
        t2 = gj["table2"]  # e.g. "Property Valuation..._2"
        edge_pair = tuple(sorted([t1, t2]))
        gt_edges.add(edge_pair)
        for col in gj["join_columns"]:
            gt_col_joins.add((t1, t2, col))
            gt_col_joins.add((t2, t1, col))  # bidirectional

    # Build prediction sets
    pred_edges = set()
    pred_col_joins = set()

    for t1, t2, c1, c2, prob in predictions:
        t1_stem = t1.table_name
        t2_stem = t2.table_name
        edge_pair = tuple(sorted([t1_stem, t2_stem]))
        pred_edges.add(edge_pair)
        pred_col_joins.add((t1_stem, t2_stem, c1))
        pred_col_joins.add((t1_stem, t2_stem, c2))
        pred_col_joins.add((t2_stem, t1_stem, c1))
        pred_col_joins.add((t2_stem, t1_stem, c2))

    # Compute edge-level metrics
    edge_tp = len(gt_edges & pred_edges)
    edge_precision = edge_tp / len(pred_edges) if pred_edges else 0
    edge_recall = edge_tp / len(gt_edges) if gt_edges else 0
    edge_f1 = 2 * edge_precision * edge_recall / (edge_precision + edge_recall) if (edge_precision + edge_recall) > 0 else 0

    # Compute column-level metrics (check if GT join columns appear in predictions)
    col_tp = 0
    col_total_gt = 0
    for gj in gt_joins:
        t1 = gj["table1"]
        t2 = gj["table2"]
        for col in gj["join_columns"]:
            col_total_gt += 1
            # Check if this column appears in any prediction for this table pair
            if (t1, t2, col) in pred_col_joins or (t2, t1, col) in pred_col_joins:
                col_tp += 1

    col_recall = col_tp / col_total_gt if col_total_gt > 0 else 0

    return {
        "project": project["project_name"],
        "category": project["category"],
        "num_tables": len(table_list),
        "num_gt_edges": len(gt_edges),
        "num_pred_edges": len(pred_edges),
        "edge_tp": edge_tp,
        "edge_precision": edge_precision,
        "edge_recall": edge_recall,
        "edge_f1": edge_f1,
        "col_total_gt": col_total_gt,
        "col_tp": col_tp,
        "col_recall": col_recall,
        "gt_joins": gt_joins,
        "predictions": [(t1.table_name, t2.table_name, c1, c2, float(prob))
                        for t1, t2, c1, c2, prob in predictions]
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepJoin on NYC benchmark")
    parser.add_argument("--model_dir", type=str,
                        default="/data/workspace/EcoTableJ/outputs/stage2_deepjoin_model",
                        help="Path to trained DeepJoin model")
    parser.add_argument("--nyc_dir", type=str,
                        default="/data/workspace/EcoTableJ/data/benchmarks/NYC",
                        help="Path to NYC benchmark directory")
    parser.add_argument("--split_stats", type=str,
                        default="/data/workspace/EcoTableJ/data/benchmarks/NYC/split_statistics.json",
                        help="Path to split_statistics.json")
    parser.add_argument("--output", type=str,
                        default="/data/workspace/EcoTableJ/outputs/stage2_deepjoin_model/eval_results.json",
                        help="Output file for evaluation results")
    parser.add_argument("--top_k", type=int, default=6,
                        help="Top-K edges per table pair")
    parser.add_argument("--max_projects", type=int, default=0,
                        help="Max projects to evaluate (0 = all)")
    args = parser.parse_args()

    # Load split_statistics.json
    logger.info(f"Loading split_statistics from {args.split_stats}")
    with open(args.split_stats) as f:
        split_stats = json.load(f)

    # Extract ground truth
    projects = extract_ground_truth(split_stats, args.nyc_dir)
    logger.info(f"Found {len(projects)} projects with join relationships")

    # Filter to projects that exist in NYC dir
    existing_projects = []
    for p in projects:
        folder_path = Path(args.nyc_dir) / p["folder_name"]
        if folder_path.exists():
            existing_projects.append(p)
        else:
            logger.debug(f"Skipping {p['folder_name']}: folder not found")
    logger.info(f"{len(existing_projects)} projects have matching folders in NYC benchmark")

    if args.max_projects > 0:
        existing_projects = existing_projects[:args.max_projects]
        logger.info(f"Limiting to {args.max_projects} projects")

    # Load model
    logger.info("Loading DeepJoin model...")
    clf = DeepJoinClassifier()
    clf.load_model(args.model_dir)
    logger.info("Model loaded successfully")

    # Evaluate each project
    results = []
    total_edge_tp = 0
    total_edge_gt = 0
    total_edge_pred = 0
    total_col_tp = 0
    total_col_gt = 0

    for i, project in enumerate(existing_projects):
        logger.info(f"[{i+1}/{len(existing_projects)}] Evaluating: {project['project_name']} ({project['category']})")
        try:
            result = evaluate_project(clf, project, args.nyc_dir, args.top_k)
            if result:
                results.append(result)
                total_edge_tp += result["edge_tp"]
                total_edge_gt += result["num_gt_edges"]
                total_edge_pred += result["num_pred_edges"]
                total_col_tp += result["col_tp"]
                total_col_gt += result["col_total_gt"]

                logger.info(f"  Edge: P={result['edge_precision']:.3f} R={result['edge_recall']:.3f} F1={result['edge_f1']:.3f} | "
                           f"Col Recall={result['col_recall']:.3f} ({result['col_tp']}/{result['col_total_gt']})")
        except Exception as e:
            logger.error(f"  Error: {e}")
            continue

    # Compute overall metrics
    overall_edge_precision = total_edge_tp / total_edge_pred if total_edge_pred > 0 else 0
    overall_edge_recall = total_edge_tp / total_edge_gt if total_edge_gt > 0 else 0
    overall_edge_f1 = (2 * overall_edge_precision * overall_edge_recall /
                       (overall_edge_precision + overall_edge_recall)
                       if (overall_edge_precision + overall_edge_recall) > 0 else 0)
    overall_col_recall = total_col_tp / total_col_gt if total_col_gt > 0 else 0

    # Per-category metrics
    category_metrics = defaultdict(lambda: {"edge_tp": 0, "edge_gt": 0, "edge_pred": 0,
                                             "col_tp": 0, "col_gt": 0, "count": 0})
    for r in results:
        cat = r["category"]
        category_metrics[cat]["edge_tp"] += r["edge_tp"]
        category_metrics[cat]["edge_gt"] += r["num_gt_edges"]
        category_metrics[cat]["edge_pred"] += r["num_pred_edges"]
        category_metrics[cat]["col_tp"] += r["col_tp"]
        category_metrics[cat]["col_gt"] += r["col_total_gt"]
        category_metrics[cat]["count"] += 1

    cat_summary = {}
    for cat, m in category_metrics.items():
        p = m["edge_tp"] / m["edge_pred"] if m["edge_pred"] > 0 else 0
        r = m["edge_tp"] / m["edge_gt"] if m["edge_gt"] > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        cr = m["col_tp"] / m["col_gt"] if m["col_gt"] > 0 else 0
        cat_summary[cat] = {
            "count": m["count"],
            "edge_precision": round(p, 4),
            "edge_recall": round(r, 4),
            "edge_f1": round(f1, 4),
            "col_recall": round(cr, 4)
        }

    summary = {
        "total_projects": len(results),
        "overall": {
            "edge_precision": round(overall_edge_precision, 4),
            "edge_recall": round(overall_edge_recall, 4),
            "edge_f1": round(overall_edge_f1, 4),
            "col_recall": round(overall_col_recall, 4),
            "edge_tp": total_edge_tp,
            "edge_gt": total_edge_gt,
            "edge_pred": total_edge_pred,
            "col_tp": total_col_tp,
            "col_gt": total_col_gt
        },
        "per_category": cat_summary,
        "per_project": results
    }

    # Print summary
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total projects evaluated: {len(results)}")
    logger.info(f"Overall Edge:  P={overall_edge_precision:.4f}  R={overall_edge_recall:.4f}  F1={overall_edge_f1:.4f}")
    logger.info(f"Overall Column Recall: {overall_col_recall:.4f} ({total_col_tp}/{total_col_gt})")
    logger.info("-" * 60)
    for cat, cs in cat_summary.items():
        logger.info(f"  [{cat}] ({cs['count']} projects): "
                    f"Edge P={cs['edge_precision']:.4f} R={cs['edge_recall']:.4f} F1={cs['edge_f1']:.4f} | "
                    f"Col R={cs['col_recall']:.4f}")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
