"""
I/O utilities for cluster-based edge validation.

Provides canonical edge representation, filesystem-safe naming,
graph loading, and table CSV loading.
"""

import hashlib
import json
import os
from typing import Dict, Optional, Set, Tuple

import pandas as pd
import networkx as nx
from networkx.readwrite import json_graph

Edge = Tuple[str, str]


def canon_edge(u: str, v: str) -> Edge:
    """Canonical (sorted) edge representation."""
    return tuple(sorted([u, v]))


def safe_name(node: str) -> str:
    """Make a node name filesystem-safe."""
    return node.replace('/', '__').replace('\\', '__').replace(':', '_')


def edge_dir_name(edge: Edge) -> str:
    """Filesystem-safe directory name for an edge. Truncated + hashed if too long."""
    full = safe_name(edge[0]) + '___' + safe_name(edge[1])
    if len(full) > 200:
        h = hashlib.md5(full.encode()).hexdigest()[:12]
        a = safe_name(edge[0])[:90]
        b = safe_name(edge[1])[:90]
        full = f"{a}____{b}____{h}"
    return full


def load_pruned_graph(path: str) -> nx.Graph:
    """Load a pruned graph from JSON (node-link format).

    For large graphs (>10k links), uses manual construction instead of
    json_graph.node_link_graph which is extremely slow on large inputs.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    gdata = data.get('graph', data)
    links = gdata.get('links', gdata.get('edges', []))
    # Fast path for large graphs
    if len(links) > 10000:
        G = nx.Graph()
        for node in gdata.get('nodes', []):
            nid = node.get('id', node.get('label', ''))
            G.add_node(nid, **{k: v for k, v in node.items() if k != 'id'})
        for link in links:
            src = link.get('source', '')
            tgt = link.get('target', '')
            attrs = {k: v for k, v in link.items() if k not in ('source', 'target')}
            G.add_edge(src, tgt, **attrs)
        return G
    try:
        return json_graph.node_link_graph(gdata, edges="edges")
    except TypeError:
        return json_graph.node_link_graph(gdata)


def find_table_csv(node_name: str, tables_root: str) -> Optional[str]:
    """Find the CSV file for a graph node name under tables_root."""
    from pathlib import Path
    p = Path(tables_root) / (node_name + '.csv')
    if p.exists():
        return str(p)
    sub = node_name.split('/')[-1] if '/' in node_name else node_name
    for root, _, files in os.walk(tables_root):
        for f in files:
            if f == sub + '.csv':
                return os.path.join(root, f)
    return None


def load_table_dfs(names: Set[str], tables_root: str) -> Dict[str, pd.DataFrame]:
    """Load CSV tables into DataFrames for a set of node names."""
    dfs = {}
    for n in sorted(names):
        p = find_table_csv(n, tables_root)
        if p:
            try:
                dfs[n] = pd.read_csv(p)
            except Exception:
                pass
    return dfs
