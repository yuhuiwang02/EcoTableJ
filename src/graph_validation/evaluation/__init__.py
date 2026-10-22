"""
Evaluation utilities for join path accuracy.

Modules:
    gt_builder    - Build GT join paths from split_statistics.json
    pred_builder  - Build predicted join paths from final_summary.json
    metrics       - Table-level and column-level F1 computation
    evaluator     - Main evaluation orchestrator (run_evaluation)

Usage:
    from src.graph_validation.evaluation import run_evaluation
    from src.graph_validation.evaluation.metrics import compute_edge_f1, compute_edge_col_f1
    from src.graph_validation.evaluation.gt_builder import build_base_table_graphs
"""

from .evaluator import run_evaluation
from .metrics import compute_edge_f1, compute_edge_col_f1
from .gt_builder import build_base_table_graphs, build_gt_path_for_query
from .pred_builder import build_verified_graph, get_pred_path_for_query, load_pruned_graph
