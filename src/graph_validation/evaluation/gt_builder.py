"""
Ground-truth join path builder from split_statistics.json.

Builds internal connection graphs for each base table, then for a given query
finds the GT Steiner tree within the base table graph and maps node_ids back
to full graph node names.
"""

import json
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

Edge = Tuple[str, str]


def _canon(u: str, v: str) -> Edge:
    """Canonical (sorted) edge representation."""
    return tuple(sorted([u, v]))


def build_base_table_graphs(stats: dict) -> Dict[str, dict]:
    """Build internal connection graph for each base table from split_statistics.

    Args:
        stats: Parsed split_statistics.json dict with 'normal', 'no_overlap',
               'no_join_cols' categories.

    Returns:
        {base_table_name: {
            'category': str,
            'node_mapping': {node_id: subtable_file},
            'subtable_to_node': {subtable_file: node_id},
            'graph': nx.Graph (nodes=node_ids, edges with join_columns attr),
            'edge_join_columns': {(nid1, nid2): [col1, col2, ...]},
        }}
    """
    base_graphs = {}
    for cat in ['normal', 'no_overlap', 'no_join_cols']:
        for entry in stats.get(cat, []):
            bt = entry['table_name']
            category = entry['category']
            nm = entry['node_mapping']
            ejc = entry.get('edge_join_columns', {})

            node_map = {}
            sub_to_node = {}
            for nid, info in nm.items():
                sf = info['subtable_file']
                node_map[nid] = sf
                sub_to_node[sf] = nid

            G = nx.Graph()
            for nid in nm:
                G.add_node(nid)

            edge_jc = {}
            for edge_key, cols in ejc.items():
                parts = edge_key.split('-')
                if len(parts) == 2:
                    n1, n2 = parts[0], parts[1]
                    G.add_edge(n1, n2, join_columns=cols)
                    edge_jc[_canon(n1, n2)] = cols

            base_graphs[bt] = {
                'category': category,
                'node_mapping': node_map,
                'subtable_to_node': sub_to_node,
                'graph': G,
                'edge_join_columns': edge_jc,
            }
    return base_graphs


def build_gt_path_for_query(query_tables_short: List[str],
                            mapper,
                            base_graphs: Dict[str, dict]) -> Optional[dict]:
    """Build GT join path for a single query.

    Args:
        query_tables_short: Short table names from the query.
        mapper: GTNameMapper instance for resolving short names to full graph names.
        base_graphs: Output of build_base_table_graphs().

    Returns:
        {
            'gt_edges': [(subtable_A, subtable_B), ...],
            'gt_edges_with_cols': [{'edge': (A, B), 'join_columns': [...]}, ...],
            'base_table': str,
            'status': 'success' | 'partial' | 'no_base' | 'single_table',
        }
    """
    if len(query_tables_short) < 2:
        return {'gt_edges': [], 'gt_edges_with_cols': [], 'base_table': None,
                'status': 'single_table'}

    mapped_full, unresolved = mapper.map_query_tables(query_tables_short)
    if len(mapped_full) < 2:
        return {'gt_edges': [], 'gt_edges_with_cols': [], 'base_table': None,
                'status': 'no_base'}

    base_tables = set()
    node_ids_per_base = defaultdict(list)
    for full_name in mapped_full:
        if '/' not in full_name:
            continue
        prefix, subtable = full_name.split('/', 1)
        for bt, bg in base_graphs.items():
            cat = bg['category']
            expected_prefix = f"{bt}_{cat}"
            if prefix == expected_prefix:
                base_tables.add(bt)
                nid = bg['subtable_to_node'].get(subtable)
                if nid is None:
                    nid = bg['subtable_to_node'].get(full_name.split('/')[-1])
                if nid is not None:
                    node_ids_per_base[bt].append(nid)
                break

    if not base_tables:
        return {'gt_edges': [], 'gt_edges_with_cols': [], 'base_table': None,
                'status': 'no_base'}

    all_gt_edges = []
    all_gt_edges_with_cols = []
    primary_base = None

    for bt in base_tables:
        bg = base_graphs[bt]
        nids = node_ids_per_base.get(bt, [])
        if len(nids) < 2:
            continue
        primary_base = bt

        G_bt = bg['graph']
        nids_in_graph = [n for n in nids if n in G_bt]
        if len(nids_in_graph) < 2:
            continue

        try:
            steiner_edges = set()
            for i in range(len(nids_in_graph)):
                for j in range(i + 1, len(nids_in_graph)):
                    try:
                        path = nx.shortest_path(G_bt, nids_in_graph[i],
                                                nids_in_graph[j])
                        for k in range(len(path) - 1):
                            steiner_edges.add(_canon(path[k], path[k + 1]))
                    except nx.NetworkXNoPath:
                        pass

            for e in steiner_edges:
                cat = bg['category']
                sub_a = bg['node_mapping'].get(e[0], e[0])
                sub_b = bg['node_mapping'].get(e[1], e[1])
                full_a = f"{bt}_{cat}/{sub_a}"
                full_b = f"{bt}_{cat}/{sub_b}"

                canon_edge = _canon(full_a, full_b)
                jc = bg['edge_join_columns'].get(e, [])
                all_gt_edges.append(canon_edge)
                all_gt_edges_with_cols.append({
                    'edge': canon_edge, 'join_columns': jc,
                })
        except Exception:
            pass

    return {
        'gt_edges': all_gt_edges,
        'gt_edges_with_cols': all_gt_edges_with_cols,
        'base_table': primary_base,
        'status': 'success' if all_gt_edges else 'no_path',
    }
