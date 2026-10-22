"""
F1 evaluation metrics for join path accuracy.

Provides:
  - compute_edge_f1(): Table-level F1 (edge set comparison)
  - compute_edge_col_f1(): Column-level F1 (edge + join column matching)
"""

from typing import Dict, List, Set, Tuple

Edge = Tuple[str, str]


def compute_edge_f1(gt_edges: List[Edge], pred_edges: List[Edge]) -> dict:
    """Compute precision, recall, F1 for edge sets (table-level only).

    Args:
        gt_edges: List of canonical GT edge tuples.
        pred_edges: List of canonical predicted edge tuples.

    Returns:
        {'precision', 'recall', 'f1', 'tp', 'fp', 'fn'}
    """
    gt_set = set(gt_edges)
    pred_set = set(pred_edges)

    if len(pred_set) == 0 and len(gt_set) == 0:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'tp': 0, 'fp': 0, 'fn': 0}
    if len(pred_set) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': len(gt_set)}
    if len(gt_set) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': len(pred_set), 'fn': 0}

    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    return {'precision': prec, 'recall': rec, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}


def _normalize_col(col: str) -> str:
    """Normalize a column name for comparison (lowercase, strip trailing underscores/spaces)."""
    return col.lower().strip().rstrip('_').strip()


def compute_edge_col_f1(
    gt_edges_with_cols: List[dict],
    pred_edges_with_cols: List[dict],
) -> dict:
    """Compute precision, recall, F1 with column-level matching.

    An edge is a TP only if:
      1. The table pair (subtable_A, subtable_B) matches, AND
      2. At least one predicted join column matches a GT join column.
         Column matching is case-insensitive with trailing underscore normalization.
         If GT has no columns (no_join_cols category), table-level match suffices.

    Args:
        gt_edges_with_cols: [{'edge': (A, B), 'join_columns': ['col1', ...]}, ...]
        pred_edges_with_cols: [{'edge': (A, B), 'join_columns': [{'col1': x, 'col2': y}, ...]}, ...]

    Returns:
        {'precision', 'recall', 'f1', 'tp', 'fp', 'fn'}
    """
    if not gt_edges_with_cols and not pred_edges_with_cols:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'tp': 0, 'fp': 0, 'fn': 0}
    if not pred_edges_with_cols:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'tp': 0, 'fp': 0, 'fn': len(gt_edges_with_cols)}
    if not gt_edges_with_cols:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'tp': 0, 'fp': len(pred_edges_with_cols), 'fn': 0}

    # Build GT lookup: edge -> set of normalized column names
    gt_map: Dict[Edge, Set[str]] = {}
    for item in gt_edges_with_cols:
        edge = item['edge']
        cols = {_normalize_col(c) for c in item.get('join_columns', []) if c}
        if edge in gt_map:
            gt_map[edge] |= cols
        else:
            gt_map[edge] = cols

    # Build Pred lookup: edge -> set of normalized column names (union of col1 and col2)
    pred_map: Dict[Edge, Set[str]] = {}
    for item in pred_edges_with_cols:
        edge = item['edge']
        cols = set()
        for pair in item.get('join_columns', []):
            if isinstance(pair, dict):
                c1 = pair.get('col1', '')
                c2 = pair.get('col2', '')
                if c1:
                    cols.add(_normalize_col(c1))
                if c2:
                    cols.add(_normalize_col(c2))
        if edge in pred_map:
            pred_map[edge] |= cols
        else:
            pred_map[edge] = cols

    tp = 0
    matched_gt = set()
    matched_pred = set()

    for edge in pred_map:
        if edge in gt_map:
            gt_cols = gt_map[edge]
            pred_cols = pred_map[edge]
            # If GT has no columns (no_join_cols category), table-level match is enough
            if not gt_cols or (gt_cols & pred_cols):
                tp += 1
                matched_gt.add(edge)
                matched_pred.add(edge)

    fp = len(pred_map) - len(matched_pred)
    fn = len(gt_map) - len(matched_gt)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    return {'precision': prec, 'recall': rec, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}
