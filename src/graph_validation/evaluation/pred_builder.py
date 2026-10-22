"""
Predicted join path builder.

Provides:
  - build_verified_graph(): Build graph with removed edges deleted.
  - get_pred_path_for_query(): Get predicted join path for a single query,
    with fallback to Steiner tree on verified graph.
"""

import json
from typing import List, Set, Tuple

import networkx as nx
from networkx.readwrite import json_graph

from ..join_evaluation import build_steiner_for_query

Edge = Tuple[str, str]


def _canon(u: str, v: str) -> Edge:
    """Canonical (sorted) edge representation."""
    return tuple(sorted([u, v]))


def load_pruned_graph(path: str) -> nx.Graph:
    """Load a pruned graph from JSON (node-link format)."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    gdata = data.get('graph', data)
    try:
        return json_graph.node_link_graph(gdata, edges="edges")
    except TypeError:
        return json_graph.node_link_graph(gdata)


def build_verified_graph(G_original: nx.Graph,
                         confirmed: Set[Edge],
                         removed: Set[Edge]) -> nx.Graph:
    """Build graph with removed edges deleted, confirmed edges reweighted.

    Args:
        G_original: Original pruned graph.
        confirmed: Set of confirmed (joinable) edges.
        removed: Set of removed (unjoinable) edges.

    Returns:
        Modified copy of the graph.
    """
    G = G_original.copy()
    for e in removed:
        if G.has_edge(e[0], e[1]):
            G.remove_edge(e[0], e[1])
    for e in confirmed:
        if G.has_edge(e[0], e[1]):
            G[e[0]][e[1]]['weight'] = 0.001
    return G


def get_pred_path_for_query(qjp: dict, G_verified: nx.Graph,
                            required_tables: List[str]) -> List[Edge]:
    """Get predicted join path edges for a query.

    If the query has edges from iterative validation, use them directly.
    Otherwise, find the Steiner tree on the verified graph that connects
    the most tables.

    Args:
        qjp: Query join path dict from final_summary.json.
        G_verified: Verified graph (removed edges deleted).
        required_tables: List of required table names for the query.

    Returns:
        List of canonical edge tuples.
    """
    # If has edges (verified or not), use them
    if qjp.get('n_edges', 0) > 0:
        return [_canon(e['edge'][0], e['edge'][1]) for e in qjp['path_edges']]

    # No path from iterative validation — find best Steiner tree on verified graph
    if len(required_tables) < 2:
        return []

    edges, covered, status = build_steiner_for_query(required_tables, G_verified)
    if edges:
        return [_canon(u, v) for u, v in edges]

    return []
