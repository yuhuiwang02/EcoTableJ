"""
Single-cluster verification and cross-cluster voting.

Provides:
  - verify_cluster(): Verify all edges in one cluster (Step1 + Phase2),
    immediately write cluster_N.json and per_edge/validation_K.json.
  - cross_cluster_vote(): Aggregate votes across clusters to decide each edge.
"""

import json
import os
import threading
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import pandas as pd
from openai import OpenAI

from .io_utils import Edge, canon_edge, edge_dir_name
from .llm_helpers import (
    call_llm, parse_json_response, parse_selected_pairs,
    serialize_table, build_candidate_pairs,
)
from .llm_helpers_relaxed import run_phase2_single


def verify_cluster(cluster_idx: int, cluster: dict,
                   client: OpenAI, model: str,
                   table_dfs: Dict[str, pd.DataFrame],
                   cluster_prompt_tpl: str, edge_prompt_tpl: str, r: int,
                   cluster_dir: str, per_edge_dir: str,
                   edge_demand: Dict[Edge, int],
                   edge_val_counter: Dict[Edge, int],
                   counter_lock: threading.Lock) -> dict:
    """Verify one cluster and IMMEDIATELY write results to disk.

    Args:
        cluster_idx: 0-based cluster index.
        cluster: Dict with 'tables' and 'edges' keys.
        client: OpenAI client.
        model: Model name.
        table_dfs: Pre-loaded DataFrames keyed by node name.
        cluster_prompt_tpl: Cluster prompt template string.
        edge_prompt_tpl: Edge prompt template string (unused, kept for API compat).
        r: Cluster size bound.
        cluster_dir: Directory to write cluster_N.json.
        per_edge_dir: Directory to write per_edge/validation_K.json.
        edge_demand: {canonical_edge: demand} mapping.
        edge_val_counter: Shared counter for validation indices (thread-safe with lock).
        counter_lock: Threading lock for edge_val_counter.

    Returns:
        Cluster result dict (same as what's written to cluster_N.json).
    """
    tables = cluster['tables']
    edges = cluster['edges']

    # Build cluster prompt
    tables_str = ""
    for t in tables:
        if t in table_dfs:
            tables_str += serialize_table(table_dfs[t], t) + "\n---\n\n"
    edges_str = ""
    for ei, (t1, t2) in enumerate(edges, 1):
        cands = (build_candidate_pairs(table_dfs[t1], table_dfs[t2])
                 if t1 in table_dfs and t2 in table_dfs else [])
        edges_str += f"**Edge {ei}: {t1} <-> {t2}**\n"
        for j, (c1, c2) in enumerate(cands, 1):
            edges_str += f"  {j}. `({t1}.{c1}, {t2}.{c2})`\n"
        edges_str += "\n"

    prompt = (cluster_prompt_tpl
              .replace("{R}", str(r))
              .replace("{NUM_TABLES}", str(len(tables)))
              .replace("{NUM_EDGES}", str(len(edges)))
              .replace("{CLUSTER_TABLES}", tables_str)
              .replace("{CLUSTER_EDGES}", edges_str))
    msgs = [
        {"role": "system", "content": "You are a rigorous data architect."},
        {"role": "user", "content": prompt},
    ]

    # Step 1: cluster-level LLM call
    step1_raw = ""
    step1_parsed = {}
    step1_tokens = 0
    step1_latency = 0
    try:
        res = call_llm(client, model, msgs)
        step1_raw = res['response']
        step1_tokens = res['tokens']
        step1_latency = res['latency']
        step1_parsed = parse_json_response(res['response'])
    except Exception as e:
        step1_raw = f"ERROR: {e}"

    edge_results_raw = step1_parsed.get('edge_results', {})

    # Per-edge Phase 2
    edge_details: Dict[str, dict] = {}
    for (t1, t2) in edges:
        edge = canon_edge(t1, t2)
        ek = f"{t1} <-> {t2}"
        ek_rev = f"{t2} <-> {t1}"
        edata = edge_results_raw.get(ek) or edge_results_raw.get(ek_rev) or {}
        same_scope = edata.get('same_scope', True)
        selected_pairs = edata.get('selected_pairs', [])
        step1_ok = same_scope and len(selected_pairs) > 0

        if not step1_ok:
            edge_details[f"{edge[0]} <-> {edge[1]}"] = {
                'step1': {'same_scope': same_scope, 'selected_pairs': selected_pairs,
                          'reason': edata.get('reason', ''), 'joinable': False},
                'phase2': None, 'vote': 0,
            }
            continue

        # Phase 2
        col_pairs = parse_selected_pairs(selected_pairs)
        if not col_pairs or t1 not in table_dfs or t2 not in table_dfs:
            edge_details[f"{edge[0]} <-> {edge[1]}"] = {
                'step1': {'same_scope': True, 'selected_pairs': selected_pairs,
                          'reason': edata.get('reason', ''), 'joinable': True},
                'phase2': {'joinable': False, 'reason': 'no_col_pairs_or_table'},
                'vote': 0,
            }
            continue

        p2_ok, p2_details, p2_tok, p2_lat = run_phase2_single(
            client, model, table_dfs[t1], table_dfs[t2], col_pairs)
        edge_details[f"{edge[0]} <-> {edge[1]}"] = {
            'step1': {'same_scope': True, 'selected_pairs': selected_pairs,
                      'reason': edata.get('reason', ''), 'joinable': True},
            'phase2': {'joinable': p2_ok, 'details': p2_details,
                       'tokens': p2_tok, 'latency': p2_lat},
            'vote': 1 if p2_ok else 0,
        }

    n_yes = sum(1 for v in edge_details.values() if v['vote'] == 1)
    print(f"    [C{cluster_idx+1}] {len(tables)}T {len(edges)}E → "
          f"{n_yes}/{len(edges)} joinable", flush=True)

    cluster_data = {
        'cluster_id': cluster_idx + 1,
        'tables': tables,
        'edges': [list(e) for e in edges],
        'step1': {'raw_response': step1_raw, 'parsed': step1_parsed,
                  'tokens': step1_tokens, 'latency': step1_latency},
        'edge_results': edge_details,
        'summary': {'n_edges': len(edges), 'n_joinable': n_yes,
                     'n_unjoinable': len(edges) - n_yes},
    }

    # ── Immediately write cluster_N.json ──
    cpath = os.path.join(cluster_dir, f"cluster_{cluster_idx+1}.json")
    with open(cpath, 'w', encoding='utf-8') as f:
        json.dump(cluster_data, f, indent=2, ensure_ascii=False)

    # ── Immediately write per_edge/validation_K.json ──
    for ek, ed in edge_details.items():
        parts = ek.split(' <-> ')
        if len(parts) != 2:
            continue
        edge = canon_edge(parts[0], parts[1])
        with counter_lock:
            edge_val_counter[edge] += 1
            idx = edge_val_counter[edge]
        edir = os.path.join(per_edge_dir, edge_dir_name(edge))
        os.makedirs(edir, exist_ok=True)
        val_data = {
            'edge': list(edge), 'validation_index': idx,
            'from_cluster_id': cluster_idx + 1,
            'demand': edge_demand.get(edge, 1),
            'vote': ed['vote'],
            'step1': ed.get('step1'), 'phase2': ed.get('phase2'),
        }
        vpath = os.path.join(edir, f'validation_{idx}.json')
        with open(vpath, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)

    return cluster_data


def cross_cluster_vote(all_cluster_results: List[dict],
                       edge_demand: Dict[Edge, int]) -> Dict[Edge, dict]:
    """Aggregate votes across clusters to decide each edge.

    Args:
        all_cluster_results: List of cluster result dicts.
        edge_demand: {canonical_edge: demand} mapping.

    Returns:
        {canonical_edge: {is_joinable, votes, yes, total, demand, threshold}}
    """
    edge_votes: Dict[Edge, List[dict]] = defaultdict(list)
    for ci, cr in enumerate(all_cluster_results):
        for ek, ed in cr['edge_results'].items():
            parts = ek.split(' <-> ')
            if len(parts) == 2:
                edge = canon_edge(parts[0], parts[1])
                edge_votes[edge].append({
                    'cluster_id': ci + 1, 'vote': ed['vote'],
                    'step1': ed.get('step1'), 'phase2': ed.get('phase2'),
                })

    decisions: Dict[Edge, dict] = {}
    for edge, votes in edge_votes.items():
        d = edge_demand.get(edge, 1)
        yes = sum(v['vote'] for v in votes)
        threshold = 2 if d == 3 else 1
        decisions[edge] = {
            'is_joinable': yes >= threshold,
            'votes': votes,
            'yes': yes, 'total': len(votes),
            'demand': d, 'threshold': threshold,
        }
    return decisions
