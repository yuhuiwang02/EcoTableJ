"""
Iterative cluster-based edge validation runner.

Orchestrates the full iterative pipeline:
  1. Build Steiner trees for queries → collect unverified edges (E_new)
  2. Cluster E_new using Algorithm 1
  3. Verify clusters concurrently (Step1 + Phase2)
  4. Cross-cluster voting → update graph
  5. Repeat until all query paths are fully verified or max_iter reached
  6. Build final per-query join paths with join columns

Supports resuming from existing iterations (reads per_edge/edge_NNN/edge_info.json).
"""

import json
import multiprocessing as mp
import os
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Set, Tuple

import networkx as nx
from openai import OpenAI

from ..join_evaluation import build_steiner_for_query, convert_networkx_to_adjacency_list
from .io_utils import Edge, canon_edge, load_table_dfs
from .clustering import paper_clustering
from .cluster_verifier import verify_cluster, cross_cluster_vote
from .llm_helpers import parse_selected_pairs


# ═══════════════════════════════════════════════════════════════════════════════
#  Query path building + E_new collection
# ═══════════════════════════════════════════════════════════════════════════════

# ── Multiprocessing helpers for parallel Steiner tree computation ──
_MP_ADJ_LIST = None
_MP_COMPONENTS = None
_MP_GRAPH = None


def _init_steiner_worker(graph, adj_list, components):
    """Initializer for multiprocessing pool workers."""
    global _MP_GRAPH, _MP_ADJ_LIST, _MP_COMPONENTS
    _MP_GRAPH = graph
    _MP_ADJ_LIST = adj_list
    _MP_COMPONENTS = components


def _steiner_worker(args):
    """Worker function: compute Steiner tree for a single query."""
    qi, tables = args
    if len(tables) < 2:
        return qi, None, tables, 'single_table'
    edges, covered, status = build_steiner_for_query(
        tables, _MP_GRAPH, adj_list=_MP_ADJ_LIST, components=_MP_COMPONENTS)
    return qi, edges, covered, status


def build_query_paths(graph: nx.Graph, queries: List[dict],
                      confirmed_edges: Set[Edge],
                      n_workers: int = 12) -> Tuple[List[dict], Counter]:
    """Build Steiner tree for each query using multiprocessing.

    Return per-query path info + edge freq.
    A query is 'fully_verified' when ALL edges in its path are in confirmed_edges.
    Uses multiprocessing.Pool for parallel Steiner tree computation on large graphs.
    """
    # Pre-compute adj_list and components once (critical for large graphs)
    adj_list = convert_networkx_to_adjacency_list(graph)
    components = list(nx.connected_components(graph))

    # Build tasks
    tasks = [(qi, q['required_tables']) for qi, q in enumerate(queries)]

    # Use multiprocessing for large graphs, serial for small ones
    n_edges = graph.number_of_edges()
    if n_edges > 10000 and n_workers > 1:
        with mp.Pool(processes=n_workers,
                     initializer=_init_steiner_worker,
                     initargs=(graph, adj_list, components)) as pool:
            results = pool.map(_steiner_worker, tasks, chunksize=10)
    else:
        # Serial fallback for small graphs (fast enough)
        _init_steiner_worker(graph, adj_list, components)
        results = [_steiner_worker(t) for t in tasks]

    # Aggregate results
    edge_freq: Counter = Counter()
    query_paths = []
    for qi, edges, covered, status in results:
        tables = queries[qi]['required_tables']
        if status == 'single_table' or len(tables) < 2:
            query_paths.append({'query_idx': qi, 'tables': tables, 'edges': [],
                                'status': 'single_table', 'fully_verified': True})
            continue
        if not edges:
            query_paths.append({'query_idx': qi, 'tables': tables, 'edges': [],
                                'status': status, 'fully_verified': False, 'covered': covered})
            continue
        canon_edges = [canon_edge(u, v) for u, v in edges]
        for e in canon_edges:
            edge_freq[e] += 1
        unverified = [e for e in canon_edges if e not in confirmed_edges]
        query_paths.append({
            'query_idx': qi, 'tables': covered, 'edges': canon_edges,
            'status': status, 'fully_verified': len(unverified) == 0,
            'n_unverified': len(unverified),
        })
    return query_paths, edge_freq


def collect_e_new(query_paths: List[dict], edge_freq: Counter,
                  verified_edges: Set[Edge]) -> List[Tuple[str, str, int]]:
    """Collect unverified edges from query paths as E_new."""
    e_new_set: Set[Edge] = set()
    for qp in query_paths:
        for e in qp['edges']:
            if e not in verified_edges:
                e_new_set.add(e)
    e_new = []
    for e in e_new_set:
        freq = edge_freq.get(e, 1)
        e_new.append((e[0], e[1], 3 if freq > 2 else 1))
    e_new.sort()
    return e_new


# ═══════════════════════════════════════════════════════════════════════════════
#  Iteration summary I/O
# ═══════════════════════════════════════════════════════════════════════════════

def save_iteration_summary(base_dir: str, iteration: int,
                           decisions: dict, iter_stats: dict, n_clusters: int):
    """Save iteration_summary.json (clusters + per_edge already written live)."""
    iter_dir = os.path.join(base_dir, f'iteration_{iteration}')
    summary = {
        'iteration': iteration,
        'n_clusters': n_clusters,
        'n_edges_verified': len(decisions),
        'n_joinable': sum(1 for d in decisions.values() if d['is_joinable']),
        'n_unjoinable': sum(1 for d in decisions.values() if not d['is_joinable']),
        'edge_decisions': {
            f"{e[0]} <-> {e[1]}": {
                'is_joinable': d['is_joinable'], 'votes_yes': d['yes'],
                'votes_total': d['total'], 'demand': d['demand'],
                'threshold': d['threshold'],
            }
            for e, d in decisions.items()
        },
        **iter_stats,
    }
    with open(os.path.join(iter_dir, 'iteration_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  Resume from existing iterations
# ═══════════════════════════════════════════════════════════════════════════════

def resume_from_existing(base_dir: str, G: nx.Graph):
    """Scan existing iteration dirs and restore verified/confirmed/removed edges.

    Returns:
        (verified_edges, confirmed_edges, removed_edges, iteration_logs, start_iteration)
    """
    verified_edges: Dict[Edge, bool] = {}
    confirmed_edges: Set[Edge] = set()
    removed_edges: Set[Edge] = set()
    iteration_logs: List[dict] = []
    start_iteration = 1

    existing_iters = sorted(
        [d for d in os.listdir(base_dir)
         if d.startswith('iteration_') and os.path.isdir(os.path.join(base_dir, d))],
        key=lambda d: int(d.split('_')[1])
    )

    if not existing_iters:
        return verified_edges, confirmed_edges, removed_edges, iteration_logs, start_iteration

    print(f"\n[Resume] Found {len(existing_iters)} existing iteration(s): {existing_iters}")
    for iter_name in existing_iters:
        iter_num = int(iter_name.split('_')[1])
        pe_dir = os.path.join(base_dir, iter_name, 'per_edge')
        if not os.path.isdir(pe_dir):
            print(f"  [Skip] {iter_name}: no per_edge directory")
            continue
        n_conf = n_rem = 0
        for edge_dir in os.listdir(pe_dir):
            info_path = os.path.join(pe_dir, edge_dir, 'edge_info.json')
            if not os.path.exists(info_path):
                continue
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            edge = canon_edge(info['edge'][0], info['edge'][1])
            is_joinable = info['is_joinable']
            verified_edges[edge] = is_joinable
            if is_joinable:
                confirmed_edges.add(edge)
                if G.has_edge(edge[0], edge[1]):
                    G[edge[0]][edge[1]]['weight'] = 0.001
                n_conf += 1
            else:
                removed_edges.add(edge)
                if G.has_edge(edge[0], edge[1]):
                    G.remove_edge(edge[0], edge[1])
                n_rem += 1
        print(f"  [{iter_name}] Restored {n_conf} confirmed + {n_rem} removed = {n_conf+n_rem} edges")

        sum_path = os.path.join(base_dir, iter_name, 'iteration_summary.json')
        if os.path.exists(sum_path):
            with open(sum_path, 'r', encoding='utf-8') as f:
                isum = json.load(f)
            iteration_logs.append({
                'iteration': iter_num,
                'e_new': isum.get('e_new', isum.get('n_edges_verified', 0)),
                'n_critical': isum.get('n_critical', 0),
                'n_ordinary': isum.get('n_ordinary', 0),
                'n_clusters': isum.get('n_clusters', 0),
                'n_confirmed': n_conf, 'n_removed': n_rem,
                'graph_edges': G.number_of_edges(),
                'n_fully_verified': isum.get('n_fully_verified', 0),
                'n_has_path': isum.get('n_has_path', 0),
                'time_sec': isum.get('time_sec', 0),
            })
        start_iteration = iter_num + 1

    print(f"  Graph after restore: {G.number_of_nodes()}N {G.number_of_edges()}E")
    print(f"  Verified={len(verified_edges)} Confirmed={len(confirmed_edges)} Removed={len(removed_edges)}")
    print(f"  Resuming from iteration {start_iteration}")

    return verified_edges, confirmed_edges, removed_edges, iteration_logs, start_iteration


# ═══════════════════════════════════════════════════════════════════════════════
#  Build final query join paths
# ═══════════════════════════════════════════════════════════════════════════════

def build_final_query_paths(base_dir: str, queries: List[dict],
                            final_query_paths: List[dict],
                            confirmed_edges: Set[Edge],
                            iteration_logs: List[dict],
                            G_verified: nx.Graph = None) -> Tuple[List[dict], int]:
    """Build final per-query output with join columns from cluster results.

    For queries without a path (e.g. insufficient_tables after edge removal),
    falls back to finding the Steiner tree on G_verified (the graph after
    removing rejected edges) that connects the most required tables.

    Returns:
        (query_results, n_fully_verified)
    """
    # Collect join columns from cluster results for confirmed edges
    edge_join_cols: Dict[Edge, List[Tuple[str, str]]] = {}
    for iter_i in range(1, len(iteration_logs) + 1):
        iter_cluster_dir = os.path.join(base_dir, f'iteration_{iter_i}', 'clusters')
        if not os.path.isdir(iter_cluster_dir):
            continue
        for cf in os.listdir(iter_cluster_dir):
            if not cf.endswith('.json'):
                continue
            with open(os.path.join(iter_cluster_dir, cf), 'r', encoding='utf-8') as f:
                cdata = json.load(f)
            for ek, ed in cdata.get('edge_results', {}).items():
                parts = ek.split(' <-> ')
                if len(parts) != 2:
                    continue
                edge = canon_edge(parts[0], parts[1])
                if edge not in confirmed_edges:
                    continue
                if ed.get('vote', 0) == 1:
                    sp = ed.get('step1', {}).get('selected_pairs', [])
                    cols = parse_selected_pairs(sp)
                    if cols and edge not in edge_join_cols:
                        edge_join_cols[edge] = cols

    # Build per-query output
    query_results = []
    n_fully = 0
    for qi, qp in enumerate(final_query_paths):
        if qp is None:
            qp = {'query_idx': qi, 'tables': queries[qi]['required_tables'],
                  'edges': [], 'status': 'no_path', 'fully_verified': False}
        q_edges = []
        all_verified = True
        for e in qp['edges']:
            is_confirmed = e in confirmed_edges
            join_cols = edge_join_cols.get(e, [])
            q_edges.append({
                'edge': list(e),
                'is_verified': is_confirmed,
                'join_columns': [{'col1': c1, 'col2': c2} for c1, c2 in join_cols],
            })
            if not is_confirmed:
                all_verified = False
        # Fallback: if no edges and >=2 required tables, find best Steiner tree on verified graph
        status = qp['status']
        path_tables = qp.get('tables', [])
        if len(q_edges) == 0 and len(queries[qi]['required_tables']) >= 2:
            fb_edges, fb_covered, fb_status = None, [], 'no_graph'
            if G_verified is not None:
                fb_edges, fb_covered, fb_status = build_steiner_for_query(
                    queries[qi]['required_tables'], G_verified)
            if fb_edges:
                for u, v in fb_edges:
                    e = canon_edge(u, v)
                    is_confirmed = e in confirmed_edges
                    join_cols = edge_join_cols.get(e, [])
                    q_edges.append({
                        'edge': list(e),
                        'is_verified': is_confirmed,
                        'join_columns': [{'col1': c1, 'col2': c2} for c1, c2 in join_cols],
                    })
                path_tables = fb_covered
                status = f'fallback_{fb_status}'
                all_verified = False  # fallback paths are not fully verified

        if all_verified and len(q_edges) > 0:
            n_fully += 1
        query_results.append({
            'query_idx': qi,
            'query': queries[qi]['query'],
            'required_tables': queries[qi]['required_tables'],
            'path_tables': path_tables,
            'path_edges': q_edges,
            'status': status,
            'fully_verified': all_verified and len(q_edges) > 0,
            'n_edges': len(q_edges),
        })

    return query_results, n_fully


# ═══════════════════════════════════════════════════════════════════════════════
#  Main iterative loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_iterative_validation(
    G: nx.Graph,
    queries: List[dict],
    client: OpenAI,
    model: str,
    cluster_prompt_tpl: str,
    edge_prompt_tpl: str,
    r: int,
    base_dir: str,
    tables_root: str,
    max_iter: int = 10,
    max_workers: int = 5,
    reorganize_fn=None,
) -> dict:
    """Run the full iterative cluster-based edge validation pipeline.

    Args:
        G: Pruned graph (will be modified in-place: edges removed/reweighted).
        queries: List of query dicts with 'query' and 'required_tables'.
        client: OpenAI client.
        model: Model name.
        cluster_prompt_tpl: Cluster prompt template string.
        edge_prompt_tpl: Edge prompt template string.
        r: Cluster size bound.
        base_dir: Output directory for this r value.
        tables_root: Root directory containing table CSV files.
        max_iter: Maximum number of iterations.
        max_workers: Number of concurrent workers for cluster verification.
        reorganize_fn: Optional callable(iter_dir) to reorganize per_edge after each iteration.

    Returns:
        Final summary dict (also saved to base_dir/final_summary.json).
    """
    os.makedirs(base_dir, exist_ok=True)

    # Resume from existing iterations
    (verified_edges, confirmed_edges, removed_edges,
     iteration_logs, start_iteration) = resume_from_existing(base_dir, G)

    print(f"\n{'='*80}\n  Iterative Validation  r={r}  workers={max_workers}\n{'='*80}")

    final_query_paths = [None] * len(queries)

    for iteration in range(start_iteration, max_iter + 1):
        t0 = time.time()
        print(f"\n{'─'*80}\n  ITERATION {iteration}\n{'─'*80}")

        # A: Build Steiner trees, find unverified edges
        print(f"  [A] Steiner trees (G: {G.number_of_nodes()}N, {G.number_of_edges()}E)...")
        query_paths, edge_freq = build_query_paths(G, queries, confirmed_edges,
                                                    n_workers=max_workers)
        n_fully = sum(1 for qp in query_paths if qp['fully_verified'])
        n_partial = len(queries) - n_fully
        print(f"      Fully verified paths: {n_fully}/{len(queries)}, Need verification: {n_partial}")

        for qp in query_paths:
            final_query_paths[qp['query_idx']] = qp

        e_new = collect_e_new(query_paths, edge_freq, set(verified_edges.keys()))
        n_crit = sum(1 for _, _, d in e_new if d == 3)
        n_ord = sum(1 for _, _, d in e_new if d == 1)
        print(f"      E_new={len(e_new)} (critical={n_crit}, ordinary={n_ord}, "
              f"demand={sum(d for _, _, d in e_new)})")
        if not e_new:
            print(f"  [STOP] All query paths fully verified!")
            iteration_logs.append({'iteration': iteration, 'e_new': 0, 'reason': 'all_verified'})
            break

        # B: Clustering
        print(f"  [B] Clustering (Algorithm 1, r={r})...")
        clusters = paper_clustering(e_new, r)
        avg_sz = sum(len(c['tables']) for c in clusters) / max(1, len(clusters))
        print(f"      {len(clusters)} clusters (avg_size={avg_sz:.1f})")
        edge_demand = {canon_edge(u, v): d for u, v, d in e_new}

        # C: Load tables
        needed = set()
        for c in clusters:
            needed.update(c['tables'])
        print(f"  [C] Loading {len(needed)} tables...")
        table_dfs = load_table_dfs(needed, tables_root)
        print(f"      Loaded {len(table_dfs)}")

        # D: Verify clusters concurrently
        iter_dir = os.path.join(base_dir, f'iteration_{iteration}')
        cluster_dir = os.path.join(iter_dir, 'clusters')
        per_edge_dir = os.path.join(iter_dir, 'per_edge')
        os.makedirs(cluster_dir, exist_ok=True)
        os.makedirs(per_edge_dir, exist_ok=True)

        edge_val_counter: Dict[Edge, int] = defaultdict(int)
        counter_lock = threading.Lock()

        print(f"  [D] Verifying {len(clusters)} clusters (workers={max_workers})...")
        all_cluster_results = [None] * len(clusters)

        def _verify(ci, cl):
            return ci, verify_cluster(
                ci, cl, client, model, table_dfs,
                cluster_prompt_tpl, edge_prompt_tpl, r,
                cluster_dir, per_edge_dir, edge_demand,
                edge_val_counter, counter_lock)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {pool.submit(_verify, ci, cl): ci for ci, cl in enumerate(clusters)}
            for fut in as_completed(futs):
                ci, result = fut.result()
                all_cluster_results[ci] = result

        # E: Cross-cluster voting
        print(f"  [E] Cross-cluster voting...")
        decisions = cross_cluster_vote(all_cluster_results, edge_demand)
        n_join = sum(1 for d in decisions.values() if d['is_joinable'])
        print(f"      Joinable={n_join}, Unjoinable={len(decisions)-n_join}")
        crit_dec = {e: d for e, d in decisions.items() if d['demand'] == 3}
        if crit_dec:
            n_crit_join = sum(1 for d in crit_dec.values() if d['is_joinable'])
            print(f"      Critical: {len(crit_dec)} total, {n_crit_join} joinable (≥2/3)")

        # F: Update graph
        print(f"  [F] Updating graph...")
        n_conf = n_rem = 0
        for edge, dec in decisions.items():
            verified_edges[edge] = dec['is_joinable']
            if dec['is_joinable']:
                confirmed_edges.add(edge)
                if G.has_edge(edge[0], edge[1]):
                    G[edge[0]][edge[1]]['weight'] = 0.001
                n_conf += 1
            else:
                removed_edges.add(edge)
                if G.has_edge(edge[0], edge[1]):
                    G.remove_edge(edge[0], edge[1])
                n_rem += 1
        print(f"      Confirmed={n_conf}, Removed={n_rem}, "
              f"Graph: {G.number_of_nodes()}N {G.number_of_edges()}E")

        # G: Re-check query paths (lightweight: reuse step A paths, just update verification status)
        #    No need to rebuild Steiner trees — just re-check which edges are now confirmed/removed
        n_fully_post = 0
        n_has_path = 0
        for qp in query_paths:
            qi = qp['query_idx']
            if qp['status'] == 'single_table':
                n_fully_post += 1
                n_has_path += 1
                continue
            if not qp['edges']:
                continue
            # Check if any edge was removed in this iteration
            path_broken = any(e in removed_edges for e in qp['edges'])
            if path_broken:
                # Path is broken — will be rebuilt in next iteration's step A
                qp_updated = dict(qp)
                qp_updated['fully_verified'] = False
                qp_updated['status'] = 'path_broken'
                final_query_paths[qi] = qp_updated
                continue
            # Path still valid — check if all edges are confirmed
            unverified = [e for e in qp['edges'] if e not in confirmed_edges]
            is_fully = len(unverified) == 0
            qp_updated = dict(qp)
            qp_updated['fully_verified'] = is_fully
            qp_updated['n_unverified'] = len(unverified)
            final_query_paths[qi] = qp_updated
            if is_fully:
                n_fully_post += 1
            n_has_path += 1
        print(f"  [G] Fully verified paths: {n_fully_post}/{len(queries)}, "
              f"Has path: {n_has_path}/{len(queries)}")

        elapsed = time.time() - t0
        iter_stats = {
            'e_new': len(e_new), 'n_critical': n_crit, 'n_ordinary': n_ord,
            'n_clusters': len(clusters), 'n_confirmed': n_conf, 'n_removed': n_rem,
            'graph_edges': G.number_of_edges(),
            'n_fully_verified': n_fully_post, 'n_has_path': n_has_path,
            'time_sec': round(elapsed, 1),
        }
        iteration_logs.append({'iteration': iteration, **iter_stats})

        # H: Save iteration summary
        print(f"  [H] Saving iteration summary...")
        save_iteration_summary(base_dir, iteration, decisions, iter_stats, len(clusters))
        print(f"      Saved to {base_dir}/iteration_{iteration}/")

        # I: Reorganize per_edge
        if reorganize_fn:
            print(f"  [I] Reorganizing per_edge to edge_NNN format...")
            reorganize_fn(os.path.join(base_dir, f'iteration_{iteration}'))

        if n_fully_post == len(queries):
            print(f"\n  ✓ All {len(queries)} queries have fully verified paths!")
            break

        # Rebuild Steiner trees on the UPDATED graph (edges removed in F, confirmed
        # edges reweighted) so that any queries whose shortest Steiner path shifted
        # to a new, still-unverified edge are picked up as E_new for the next iter.
        # This is the critical fix: without rebuilding, collect_e_new only looks at
        # the *pre-update* Steiner trees, which have all their edges verified in
        # the current iteration -> next_e_new is always empty -> premature STOP.
        print(f"  [G'] Re-building Steiner trees on updated graph for next iter...")
        rebuilt_paths, rebuilt_freq = build_query_paths(
            G, queries, confirmed_edges, n_workers=max_workers)
        next_e_new = collect_e_new(
            rebuilt_paths, rebuilt_freq, set(verified_edges.keys()))
        print(f"      next E_new={len(next_e_new)} (after rebuild on updated G)")
        if not next_e_new:
            print(f"  [STOP] No more unverified edges to check.")
            break
        # Carry the rebuilt paths into the next iteration so step A can reuse them.
        query_paths, edge_freq = rebuilt_paths, rebuilt_freq
        for qp in query_paths:
            final_query_paths[qp['query_idx']] = qp

        # Convergence detection: if E_new is not decreasing, stop
        if iteration >= 2 and len(iteration_logs) >= 2:
            prev_e_new = iteration_logs[-2].get('e_new', float('inf'))
            curr_e_new = len(e_new)
            if len(next_e_new) >= curr_e_new and curr_e_new <= prev_e_new:
                print(f"  [STOP] E_new not converging ({prev_e_new} → {curr_e_new} → {len(next_e_new)}). "
                      f"Stopping iteration.")
                break

        print(f"      {len(queries) - n_fully_post} queries still need verification → next iteration...")

    # ── For queries with unverified edges, rebuild path on verified graph ──
    # The verified graph G only contains confirmed + untouched edges (removed edges deleted).
    # Re-run Steiner tree so that the path only uses edges still in G.
    # Pre-compute adj_list and components once for efficiency on large graphs.
    adj_list = convert_networkx_to_adjacency_list(G)
    components = list(nx.connected_components(G))
    n_rebuilt = 0
    for qi, qp in enumerate(final_query_paths):
        if qp is None or qp.get('fully_verified', False):
            continue
        tables = queries[qi]['required_tables']
        if len(tables) < 2:
            continue
        # Rebuild on verified graph — path will only use surviving edges
        edges, covered, status = build_steiner_for_query(
            tables, G, adj_list=adj_list, components=components)
        if edges:
            canon_edges = [canon_edge(u, v) for u, v in edges]
            # Check if all edges are confirmed
            all_conf = all(e in confirmed_edges for e in canon_edges)
            final_query_paths[qi] = {
                'query_idx': qi, 'tables': covered, 'edges': canon_edges,
                'status': 'rebuilt_' + status,
                'fully_verified': all_conf,
            }
            n_rebuilt += 1
    if n_rebuilt:
        print(f"\n  [Rebuild] Rebuilt {n_rebuilt} query paths on verified graph")

    # ── Build final query join paths ──
    print(f"\n{'='*80}\n  BUILDING FINAL QUERY JOIN PATHS  (r={r})\n{'='*80}")
    query_results, n_fully = build_final_query_paths(
        base_dir, queries, final_query_paths, confirmed_edges, iteration_logs,
        G_verified=G)

    n_single = sum(1 for qr in query_results
                   if qr['n_edges'] == 0 and len(qr['required_tables']) < 2)
    print(f"  Edges with join columns: counted from clusters")
    print(f"  Fully verified paths: {n_fully}/{len(queries)}")
    print(f"  Single-table queries: {n_single}")
    print(f"  Queries without full verified path: {len(queries) - n_fully - n_single}")

    # ── Final summary ──
    print(f"\n{'='*80}\n  FINAL RESULTS  (r={r})\n{'='*80}")
    print(f"  Verified={len(verified_edges)}  Confirmed={len(confirmed_edges)}  "
          f"Removed={len(removed_edges)}")
    print(f"  Fully verified query paths: {n_fully}/{len(queries)}")
    print(f"\n  {'Iter':>4} {'E_new':>6} {'Crit':>5} {'Clust':>6} {'Conf':>5} {'Rem':>5} "
          f"{'G_E':>6} {'Verified':>8} {'Time':>7}")
    for lg in iteration_logs:
        print(f"  {lg['iteration']:>4} {lg.get('e_new',0):>6} {lg.get('n_critical',0):>5} "
              f"{lg.get('n_clusters',0):>6} {lg.get('n_confirmed',0):>5} "
              f"{lg.get('n_removed',0):>5} {lg.get('graph_edges',0):>6} "
              f"{lg.get('n_fully_verified',0):>8} {lg.get('time_sec',0):>6.1f}s")

    final = {
        'config': {'model': model, 'r': r, 'max_iter': max_iter,
                   'max_workers': max_workers},
        'summary': {'iterations': len(iteration_logs),
                    'verified': len(verified_edges),
                    'confirmed': len(confirmed_edges),
                    'removed': len(removed_edges),
                    'n_fully_verified_paths': n_fully},
        'iteration_logs': iteration_logs,
        'confirmed_edges': [list(e) for e in sorted(confirmed_edges)],
        'removed_edges': [list(e) for e in sorted(removed_edges)],
        'query_join_paths': query_results,
    }
    fpath = os.path.join(base_dir, 'final_summary.json')
    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(final, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[Save] {fpath}")

    return final
