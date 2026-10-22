"""
Main evaluator: orchestrates GT building, pred building, and metric computation.

Provides:
  - run_evaluation(): Full evaluation pipeline that loads data, builds GT/pred
    paths, computes table-level and column-level F1, prints report, and saves
    results to evaluation_result.json.
"""

import json
import os
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from ..common.load_file import extract_table_name
from ..join_evaluation import build_steiner_for_query
from .gt_builder import build_base_table_graphs, build_gt_path_for_query
from .metrics import compute_edge_f1, compute_edge_col_f1
from .pred_builder import load_pruned_graph, build_verified_graph, get_pred_path_for_query

Edge = Tuple[str, str]


def _canon(u: str, v: str) -> Edge:
    return tuple(sorted([u, v]))


def run_evaluation(
    gt_query_file: str,
    split_stats_file: str,
    pruned_graph_file: str,
    final_summary_file: str,
    mapper,
    output_dir: str,
    r: int = 3,
) -> dict:
    """Run full join path evaluation.

    Args:
        gt_query_file: Path to schema_linking_gt_nyc.json.
        split_stats_file: Path to split_statistics.json.
        pruned_graph_file: Path to pruned_graph_from_llm.json.
        final_summary_file: Path to final_summary.json from iterative validation.
        mapper: GTNameMapper instance.
        output_dir: Directory to save evaluation_result.json.
        r: Cluster size parameter (for display only).

    Returns:
        Evaluation output dict (also saved to output_dir/evaluation_result.json).
    """
    print(f"[1] Loading data...")

    # Load GT queries
    with open(gt_query_file, 'r', encoding='utf-8') as f:
        gt_queries = json.load(f)
    print(f"    GT queries: {len(gt_queries)}")

    # Load split statistics and build base table graphs
    with open(split_stats_file, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    base_graphs = build_base_table_graphs(stats)
    print(f"    Base tables: {len(base_graphs)}")

    # Load pruned graph
    G_original = load_pruned_graph(pruned_graph_file)
    print(f"    Pruned graph: {G_original.number_of_nodes()}N {G_original.number_of_edges()}E")

    # Load final summary
    with open(final_summary_file, 'r', encoding='utf-8') as f:
        final = json.load(f)
    confirmed = set(_canon(e[0], e[1]) for e in final['confirmed_edges'])
    removed = set(_canon(e[0], e[1]) for e in final['removed_edges'])
    qjps = final['query_join_paths']
    print(f"    Confirmed: {len(confirmed)}, Removed: {len(removed)}")
    print(f"    Query join paths: {len(qjps)}")

    # Build verified graph for fallback Steiner tree
    G_verified = build_verified_graph(G_original, confirmed, removed)
    print(f"    Verified graph: {G_verified.number_of_nodes()}N {G_verified.number_of_edges()}E")

    graph_nodes = set(G_original.nodes())

    print(f"\n[2] Building GT and Predicted join paths for {len(gt_queries)} queries...")

    results = []
    for qi, gt_q in enumerate(gt_queries):
        short_names = [extract_table_name(t)
                       for t in (gt_q.get('query_related_table') or []) if t]
        mapped_full, unresolved = mapper.map_query_tables(short_names)
        required_in_graph = [t for t in mapped_full if t in graph_nodes]

        # GT path
        gt_info = build_gt_path_for_query(short_names, mapper, base_graphs)
        gt_edges = gt_info['gt_edges'] if gt_info else []

        # Predicted path
        qjp = qjps[qi] if qi < len(qjps) else {'n_edges': 0, 'path_edges': []}
        pred_edges = get_pred_path_for_query(qjp, G_verified, required_in_graph)

        # Build pred edges with columns for column-level evaluation
        pred_edges_with_cols = []
        pred_edge_col_map = {}
        for pe in qjp.get('path_edges', []):
            edge = _canon(pe['edge'][0], pe['edge'][1])
            jc = pe.get('join_columns', [])
            pred_edge_col_map[edge] = jc
        for e in pred_edges:
            pred_edges_with_cols.append({
                'edge': e,
                'join_columns': pred_edge_col_map.get(e, []),
            })

        gt_edges_with_cols = gt_info.get('gt_edges_with_cols', []) if gt_info else []

        # Compute table-level F1
        metrics_table = compute_edge_f1(gt_edges, pred_edges)
        # Compute column-level F1
        metrics_col = compute_edge_col_f1(gt_edges_with_cols, pred_edges_with_cols)

        results.append({
            'query_idx': qi,
            'query': gt_q.get('query', '')[:100],
            'n_gt_tables': len(short_names),
            'n_mapped': len(mapped_full),
            'n_in_graph': len(required_in_graph),
            'gt_edges': [list(e) for e in gt_edges],
            'pred_edges': [list(e) for e in pred_edges],
            'n_gt_edges': len(gt_edges),
            'n_pred_edges': len(pred_edges),
            'fully_verified': qjp.get('fully_verified', False),
            'gt_status': gt_info['status'] if gt_info else 'error',
            # Table-level metrics
            'precision': metrics_table['precision'],
            'recall': metrics_table['recall'],
            'f1': metrics_table['f1'],
            'tp': metrics_table['tp'],
            'fp': metrics_table['fp'],
            'fn': metrics_table['fn'],
            # Column-level metrics
            'col_precision': metrics_col['precision'],
            'col_recall': metrics_col['recall'],
            'col_f1': metrics_col['f1'],
            'col_tp': metrics_col['tp'],
            'col_fp': metrics_col['fp'],
            'col_fn': metrics_col['fn'],
        })

    # ── Aggregate metrics ──
    print(f"\n[3] Evaluation Results (r={r})")
    print(f"{'='*70}")

    all_f1 = [r_['f1'] for r_ in results]
    all_prec = [r_['precision'] for r_ in results]
    all_rec = [r_['recall'] for r_ in results]
    avg_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0
    avg_prec = sum(all_prec) / len(all_prec) if all_prec else 0
    avg_rec = sum(all_rec) / len(all_rec) if all_rec else 0

    all_col_f1 = [r_['col_f1'] for r_ in results]
    all_col_prec = [r_['col_precision'] for r_ in results]
    all_col_rec = [r_['col_recall'] for r_ in results]
    avg_col_f1 = sum(all_col_f1) / len(all_col_f1) if all_col_f1 else 0
    avg_col_prec = sum(all_col_prec) / len(all_col_prec) if all_col_prec else 0
    avg_col_rec = sum(all_col_rec) / len(all_col_rec) if all_col_rec else 0

    print(f"  Overall (all {len(results)} queries):")
    print(f"    [Table-level]")
    print(f"      Macro-Avg Precision : {avg_prec:.4f}")
    print(f"      Macro-Avg Recall    : {avg_rec:.4f}")
    print(f"      Macro-Avg F1        : {avg_f1:.4f}")
    print(f"    [Column-level]")
    print(f"      Macro-Avg Precision : {avg_col_prec:.4f}")
    print(f"      Macro-Avg Recall    : {avg_col_rec:.4f}")
    print(f"      Macro-Avg F1        : {avg_col_f1:.4f}")

    status_cnt = Counter(r_['gt_status'] for r_ in results)
    print(f"\n  GT status distribution: {dict(status_cnt)}")

    has_gt = [r_ for r_ in results if r_['n_gt_edges'] > 0]
    if has_gt:
        avg_f1_gt = sum(r_['f1'] for r_ in has_gt) / len(has_gt)
        avg_prec_gt = sum(r_['precision'] for r_ in has_gt) / len(has_gt)
        avg_rec_gt = sum(r_['recall'] for r_ in has_gt) / len(has_gt)
        avg_col_f1_gt = sum(r_['col_f1'] for r_ in has_gt) / len(has_gt)
        avg_col_prec_gt = sum(r_['col_precision'] for r_ in has_gt) / len(has_gt)
        avg_col_rec_gt = sum(r_['col_recall'] for r_ in has_gt) / len(has_gt)
        print(f"\n  Queries with GT edges ({len(has_gt)}):")
        print(f"    [Table-level]  P={avg_prec_gt:.4f}  R={avg_rec_gt:.4f}  F1={avg_f1_gt:.4f}")
        print(f"    [Column-level] P={avg_col_prec_gt:.4f}  R={avg_col_rec_gt:.4f}  F1={avg_col_f1_gt:.4f}")

    fully = [r_ for r_ in results if r_['fully_verified']]
    if fully:
        avg_f1_fv = sum(r_['f1'] for r_ in fully) / len(fully)
        avg_col_f1_fv = sum(r_['col_f1'] for r_ in fully) / len(fully)
        print(f"\n  Fully verified queries ({len(fully)}):")
        print(f"    [Table-level]  Macro-Avg F1 : {avg_f1_fv:.4f}")
        print(f"    [Column-level] Macro-Avg F1 : {avg_col_f1_fv:.4f}")

    f1_bins = {'0.0': 0, '(0,0.5)': 0, '[0.5,1.0)': 0, '1.0': 0}
    col_f1_bins = {'0.0': 0, '(0,0.5)': 0, '[0.5,1.0)': 0, '1.0': 0}
    for r_ in results:
        if r_['f1'] == 0: f1_bins['0.0'] += 1
        elif r_['f1'] < 0.5: f1_bins['(0,0.5)'] += 1
        elif r_['f1'] < 1.0: f1_bins['[0.5,1.0)'] += 1
        else: f1_bins['1.0'] += 1
        if r_['col_f1'] == 0: col_f1_bins['0.0'] += 1
        elif r_['col_f1'] < 0.5: col_f1_bins['(0,0.5)'] += 1
        elif r_['col_f1'] < 1.0: col_f1_bins['[0.5,1.0)'] += 1
        else: col_f1_bins['1.0'] += 1
    print(f"\n  F1 distribution (table-level):  {f1_bins}")
    print(f"  F1 distribution (column-level): {col_f1_bins}")

    low_f1 = sorted([r_ for r_ in results if r_['n_gt_edges'] > 0],
                     key=lambda r_: r_['f1'])[:5]
    if low_f1:
        print(f"\n  Lowest F1 queries (with GT edges):")
        for r_ in low_f1:
            print(f"    Q{r_['query_idx']}: F1={r_['f1']:.3f}(tbl) {r_['col_f1']:.3f}(col) "
                  f"P={r_['precision']:.3f} R={r_['recall']:.3f} "
                  f"GT={r_['n_gt_edges']}e Pred={r_['n_pred_edges']}e "
                  f"verified={r_['fully_verified']} gt_status={r_['gt_status']}")

    col_miss = [r_ for r_ in results if r_['tp'] > 0 and r_['col_tp'] < r_['tp']]
    if col_miss:
        print(f"\n  Queries with table-match but column-mismatch: {len(col_miss)}")
        for r_ in col_miss[:5]:
            print(f"    Q{r_['query_idx']}: tbl_tp={r_['tp']} col_tp={r_['col_tp']} "
                  f"tbl_F1={r_['f1']:.3f} col_F1={r_['col_f1']:.3f}")

    # ── Save results ──
    eval_output = {
        'config': {'r': r, 'pruned_graph': pruned_graph_file},
        'overall': {
            'n_queries': len(results),
            'table_level': {
                'macro_precision': round(avg_prec, 4),
                'macro_recall': round(avg_rec, 4),
                'macro_f1': round(avg_f1, 4),
            },
            'column_level': {
                'macro_precision': round(avg_col_prec, 4),
                'macro_recall': round(avg_col_rec, 4),
                'macro_f1': round(avg_col_f1, 4),
            },
        },
        'per_query': results,
    }
    os.makedirs(output_dir, exist_ok=True)
    eval_path = os.path.join(output_dir, 'evaluation_result.json')
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[Save] {eval_path}")

    return eval_output
