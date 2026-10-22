"""
EcoTable+ Pipeline — Unified Entry Point
==========================================

Supports the full EcoTable+ pipeline with configurable hyperparameters:

  Layer 1: Table Identification
    - Step 1: Schema Linking (PLM CrossEncoder, cached)
    - Step 2: LLM Filter (GPT-4o, concurrent)
    - Step 3: Table Name → Path Conversion

  Layer 2: Graph Validation
    - Step 4: DeepJoin join prediction (cached)
    - Step 5a: Build pruned graph (Steiner tree + t-spanner)
    - Step 5b: Edge verification (per-edge OR cluster-based, iterative)

  Layer 3: Table Transformation (optional)
    - Step 6: ReAct-style Transform Agent

Key hyperparameters:
  --verification_method  per_edge | cluster   (default: cluster)
  --cluster_size         r = 3,4,5,...        (default: 5)
  --use_pruning          enable graph pruning  (default: True)
  --max_iter             max validation iters  (default: 10)

Usage:
  # Full EcoTable+ (pruned graph + cluster r=5)
  python -m src.pipeline_main --config configs/default.json

  # EcoTable original (no pruning + per-edge)
  python -m src.pipeline_main --config configs/default.json \
      --verification_method per_edge --no_pruning

  # EcoTable+ w/o pruning (complete graph + cluster r=5)
  python -m src.pipeline_main --config configs/default.json --no_pruning

  # EcoTable+ with different cluster size
  python -m src.pipeline_main --config configs/default.json --cluster_size 3
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
from openai import OpenAI

# ── Project root ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Lazy imports to avoid circular deps and slow startup ──
# Table Identification
from .table_identification.LLM_Filter_nyc import LLMTableFilter, load_all_table_columns
from .table_identification.table_converter_interface import TableNameConverter

# Graph Validation — shared
from .graph_validation.common.load_file import (
    load_json_data, build_global_graph, load_ground_truth, extract_table_name,
    find_k_shortest_paths, calculate_path_weight, prune_paths_global,
)
from .graph_validation.join_evaluation import (
    build_sparse_graph_from_queries,
    build_steiner_for_query,
    convert_networkx_to_adjacency_list,
    load_queries_from_llm_filter,
    extract_candidate_pairs_from_json,
    save_pruned_graph,
    save_pruned_graph_with_join_cols,
)

# Graph Validation — per-edge method
from .graph_validation.per_edge.llm_verifier import LLMVerifier
from .graph_validation.per_edge.edge_validator import EdgeValidator

# Graph Validation — cluster method
from .graph_validation.cluster import (
    load_pruned_graph as cluster_load_pruned_graph,
    load_table_dfs,
    paper_clustering,
    run_iterative_validation,
    build_query_paths,
    collect_e_new,
)

# Table Transformation
from .table_transformation.transform_agent_interface import (
    run_transform_agent,
    prepare_duckdb_from_tables,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Layer 1: Table Identification (Steps 1-3)
# ═══════════════════════════════════════════════════════════════════════════════

def load_schema_linking_cache(cache_file: str) -> List[Dict]:
    """Load cached Schema Linking results (Step 1).

    Expected format:
      {"metadata": {...}, "results": [{sample_id, query, project_name,
                                        table_scores: [[table, score], ...]}, ...]}
    """
    with open(cache_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = data.get('results', data)
    print(f"[Step 1/Cache] Loaded {len(results)} schema-linking results from {cache_file}")
    return results


async def run_llm_filter(
    schema_results: List[Dict],
    column_info_dict: Dict[str, List[str]],
    api_key: str,
    model: str = "gpt-4o",
    base_url: Optional[str] = "https://api.openai.com/v1",
    top_n_candidates: int = 30,
    concurrency: int = 10,
    resume_file: Optional[str] = None,
) -> List[Dict]:
    """Step 2: Concurrent LLM filter with resume support."""
    filter_obj = LLMTableFilter(api_key=api_key, model=model, base_url=base_url)

    # Resume from checkpoint
    done_map: Dict[int, Dict] = {}
    if resume_file and os.path.exists(resume_file):
        try:
            with open(resume_file, encoding='utf-8') as f:
                prev = json.load(f)
            for r in prev.get('detailed_results', []):
                done_map[r['sample_id']] = r
            print(f"[Step 2] Resume: {len(done_map)} already-done queries")
        except Exception as e:
            print(f"[Step 2] Resume file load failed ({e}), starting fresh")

    total = len(schema_results)
    todo = [item for item in schema_results
            if item.get('sample_id', schema_results.index(item)) not in done_map]
    print(f"[Step 2] {total} queries total, {len(done_map)} cached, "
          f"{len(todo)} to run (concurrency={concurrency})")

    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    done_cnt = [len(done_map)]

    async def process_one(item: Dict) -> Dict:
        sid = item.get('sample_id', 0)
        query = item['query']
        candidates = [(ts[0], ts[1]) for ts in item.get('table_scores', [])[:top_n_candidates]]
        column_info = {t: {col: '' for col in column_info_dict[t]}
                       for t, _ in candidates if t in column_info_dict}
        async with sem:
            try:
                fr = await filter_obj.filter_candidates(
                    query=query, candidates=candidates, column_info=column_info)
            except Exception as e:
                print(f"  [WARN] query {sid} failed: {e}")
                fr = {'filtered_tables': [], 'temporal_scope': None, 'llm_interaction': None}
        result = {
            'sample_id': sid, 'query': query, 'candidates': candidates,
            'selected_tables': fr['filtered_tables'],
            'temporal_scope': fr.get('temporal_scope'),
            'llm_interaction': fr.get('llm_interaction'),
        }
        async with lock:
            done_map[sid] = result
            done_cnt[0] += 1
            if done_cnt[0] % 20 == 0 or done_cnt[0] == total:
                print(f"  [Step 2] {done_cnt[0]}/{total} done")
            if resume_file and done_cnt[0] % 50 == 0:
                _save_checkpoint(resume_file, done_map, model)
        return result

    tasks = [asyncio.create_task(process_one(item)) for item in todo]
    await asyncio.gather(*tasks)

    if resume_file:
        _save_checkpoint(resume_file, done_map, model)

    filtered_results = [done_map[item.get('sample_id', i)]
                        for i, item in enumerate(schema_results)
                        if item.get('sample_id', i) in done_map]
    print(f"[Step 2] Completed. {len(filtered_results)} results.")
    return filtered_results


def _save_checkpoint(path: str, done_map: Dict, model: str):
    tmp = path + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({'metadata': {'model': model},
                   'detailed_results': list(done_map.values())},
                  f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def convert_table_names(llm_filter_results: List[Dict], tables_dir: str) -> List[Dict]:
    """Step 3: Convert short table names to full CSV paths."""
    converter = TableNameConverter(tables_dir)
    return converter.convert_llm_filter_results(llm_filter_results, inplace=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Layer 2: Graph Validation (Steps 4-5)
# ═══════════════════════════════════════════════════════════════════════════════

def build_global_graph_from_deepjoin(deepjoin_file: str, verbose: bool = False):
    """Step 4: Load DeepJoin predictions and build global graph."""
    print(f"\n[Step 4] Loading DeepJoin predictions from {deepjoin_file}...")
    json_data = load_json_data(deepjoin_file)
    print(f"  Loaded {len(json_data)} predictions")
    global_graph, edge_details = build_global_graph(json_data, verbose=verbose)
    return json_data, global_graph, edge_details


def build_pruned_graph(
    queries: List[List[str]],
    global_graph: nx.Graph,
    top_k_paths: int = 10,
    max_path_length: int = 100,
    t_spanner: float = 1.1,
    cache_file: Optional[str] = None,
    overwrite: bool = False,
    verbose: bool = False,
) -> nx.Graph:
    """Step 5a: Build pruned sparse graph from queries via Steiner tree + t-spanner."""
    print(f"\n[Step 5a] Building pruned sparse graph...")
    print(f"  top_k_paths={top_k_paths}, max_path_length={max_path_length}, t_spanner={t_spanner}")
    sparse_graph = build_sparse_graph_from_queries(
        queries, global_graph,
        k=top_k_paths, max_path_length=max_path_length, t_spanner=t_spanner,
        cache_file=cache_file, overwrite=overwrite, verbose=verbose,
    )
    print(f"  Pruned graph: {sparse_graph.number_of_nodes()} nodes, "
          f"{sparse_graph.number_of_edges()} edges")
    return sparse_graph


def run_per_edge_validation(
    queries: List[List[str]],
    sparse_graph: nx.Graph,
    json_data: List[Dict],
    edge_details: Dict,
    gt_csv: str,
    tables_dir: str,
    output_dir: str,
    api_key: str,
    model: str = "gpt-4o",
    sample_rows: int = 5,
    overwrite: bool = False,
    verbose: bool = False,
) -> Tuple[List[Dict], Dict]:
    """Step 5b (per-edge): EcoTable original per-edge verification."""
    print(f"\n[Step 5b] Per-edge verification (EcoTable original method)...")

    candidate_pairs_dict = extract_candidate_pairs_from_json(json_data, edge_details)
    _, gt_edges, gt_edges_by_basetable = load_ground_truth(gt_csv, verbose=verbose)

    prompt_path = str(PROJECT_ROOT / 'src' / 'graph_validation' / 'per_edge' / 'prompt.txt')
    llm_verifier = LLMVerifier(
        api_key=api_key, model=model, sample_rows=sample_rows,
        prompt_template_path=prompt_path, test_mode=False, mock_mode=False,
    )
    edge_verification_cache = os.path.join(output_dir, 'edge_verification_cache')
    edge_validator = EdgeValidator(
        llm_verifier=llm_verifier, tables_dir=tables_dir,
        verification_base_dir=edge_verification_cache, overwrite=overwrite,
    )

    from .graph_validation.join_evaluation import evaluate_all_queries
    results, verified_edges = evaluate_all_queries(
        queries, sparse_graph, edge_validator, candidate_pairs_dict,
        gt_edges, verbose,
    )
    return results, verified_edges


def run_cluster_validation(
    sparse_graph: nx.Graph,
    queries_with_tables: List[Dict],
    tables_dir: str,
    output_dir: str,
    api_key: str,
    model: str = "gpt-4o",
    base_url: Optional[str] = "https://api.openai.com/v1",
    cluster_size: int = 5,
    max_iter: int = 10,
    max_workers: int = 5,
) -> Dict:
    """Step 5b (cluster): EcoTable+ cluster-based iterative verification.

    Args:
        sparse_graph: Pruned (or complete) graph.
        queries_with_tables: List of dicts with 'query' and 'required_tables'.
        tables_dir: Root directory containing table CSV files.
        output_dir: Output directory for this validation run.
        api_key: LLM API key.
        model: LLM model name.
        base_url: LLM API base URL.
        cluster_size: Max cluster size r.
        max_iter: Max number of iterations.
        max_workers: Concurrent workers for cluster verification.

    Returns:
        Final summary dict from run_iterative_validation.
    """
    print(f"\n[Step 5b] Cluster-based iterative verification (r={cluster_size})...")
    print(f"  max_iter={max_iter}, max_workers={max_workers}, model={model}")

    client = OpenAI(api_key=api_key, base_url=base_url)

    # Load prompt templates
    cluster_prompt_path = PROJECT_ROOT / 'src' / 'graph_validation' / 'cluster' / 'prompt_cluster.txt'
    edge_prompt_path = PROJECT_ROOT / 'src' / 'graph_validation' / 'per_edge' / 'prompt.txt'
    cluster_prompt_tpl = cluster_prompt_path.read_text(encoding='utf-8')
    edge_prompt_tpl = edge_prompt_path.read_text(encoding='utf-8')

    os.makedirs(output_dir, exist_ok=True)

    final_summary = run_iterative_validation(
        G=sparse_graph,
        queries=queries_with_tables,
        client=client,
        model=model,
        cluster_prompt_tpl=cluster_prompt_tpl,
        edge_prompt_tpl=edge_prompt_tpl,
        r=cluster_size,
        base_dir=output_dir,
        tables_root=tables_dir,
        max_iter=max_iter,
        max_workers=max_workers,
    )

    return final_summary


# ═══════════════════════════════════════════════════════════════════════════════
#  Layer 3: Table Transformation (Step 6, optional)
# ═══════════════════════════════════════════════════════════════════════════════

def run_layer3_transform(
    pruned_graph_path: str,
    llm_filter_results: List[Dict],
    tables_dir: str,
    output_dir: str,
    model: str = "gpt-4o",
    max_iterations: int = 10,
    max_workers: int = 8,
    eval_file: Optional[str] = None,
) -> Optional[Dict]:
    """Step 6: Transform Agent (optional)."""
    if not os.path.exists(pruned_graph_path):
        print(f"[Step 6] WARNING: pruned graph not found at {pruned_graph_path}, skipping")
        return None

    print(f"\n[Step 6] Transform Agent...")
    all_selected = set()
    for item in llm_filter_results:
        all_selected.update(item.get('selected_tables', []))

    duckdb_path = os.path.join(output_dir, 'tables.duckdb')
    prepare_duckdb_from_tables(
        tables_dir=tables_dir, selected_tables=list(all_selected),
        output_duckdb=duckdb_path, verbose=True,
    )

    transform_output_dir = os.path.join(output_dir, 'transform_agent_output')
    summary = run_transform_agent(
        graph_json=pruned_graph_path, duckdb_path=duckdb_path,
        output_dir=transform_output_dir, model=model,
        max_iterations=max_iterations, max_workers=max_workers,
        test_mode=False, eval_file=eval_file, verbose=True,
    )
    print(f"[Step 6] Transform agent output -> {transform_output_dir}")
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Pipeline Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

async def run_pipeline(cfg: Dict) -> Dict:
    """Run the full EcoTable/EcoTable+ pipeline.

    Args:
        cfg: Configuration dict with all parameters. See build_config_from_args().

    Returns:
        Pipeline result summary.
    """
    output_dir = cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    t_start = time.time()

    # ── Layer 1: Table Identification ──
    print("\n" + "=" * 80)
    print("  LAYER 1: Table Identification")
    print("=" * 80)

    # Step 1: Schema Linking (cached)
    schema_results = load_schema_linking_cache(cfg['schema_cache'])

    # Step 2: LLM Filter
    column_info_dict = {}
    if cfg.get('column_stats_file') and os.path.exists(cfg['column_stats_file']):
        column_info_dict = load_all_table_columns(cfg['column_stats_file'])
        print(f"[Step 2] Loaded column info for {len(column_info_dict)} tables")

    resume_file = os.path.join(output_dir, "step2_llm_filter_results.json")
    llm_filter_results = await run_llm_filter(
        schema_results, column_info_dict, cfg['llm_api_key'],
        model=cfg['llm_model'], base_url=cfg.get('llm_base_url'),
        top_n_candidates=cfg.get('top_n_candidates', 30),
        concurrency=cfg.get('concurrency', 10),
        resume_file=resume_file,
    )

    # Save LLM filter results
    llm_filter_output = os.path.join(output_dir, "step2_llm_filter_results.json")
    with open(llm_filter_output, 'w', encoding='utf-8') as f:
        json.dump({'metadata': {'model': cfg['llm_model']},
                   'detailed_results': llm_filter_results},
                  f, ensure_ascii=False, indent=2)

    # Step 3: Table Name Conversion
    llm_filter_results = convert_table_names(llm_filter_results, cfg['tables_dir'])
    with open(llm_filter_output, 'w', encoding='utf-8') as f:
        json.dump({'metadata': {'model': cfg['llm_model']},
                   'detailed_results': llm_filter_results},
                  f, ensure_ascii=False, indent=2)
    print(f"[Step 3] Table names converted -> {llm_filter_output}")

    # ── Layer 2: Graph Validation ──
    print("\n" + "=" * 80)
    print("  LAYER 2: Graph Validation")
    print("=" * 80)

    # Step 4: DeepJoin (cached)
    deepjoin_input = os.path.join(output_dir, "step4_deepjoin_top6.json")
    if os.path.abspath(cfg['deepjoin_cache']) != os.path.abspath(deepjoin_input):
        shutil.copyfile(cfg['deepjoin_cache'], deepjoin_input)
    print(f"[Step 4/Cache] Using DeepJoin result -> {deepjoin_input}")

    json_data, global_graph, edge_details = build_global_graph_from_deepjoin(
        deepjoin_input, verbose=cfg.get('verbose', False))

    # Load queries from LLM filter output
    queries_raw = load_queries_from_llm_filter(llm_filter_output, verbose=True)

    # Step 5a: Build graph (pruned or complete)
    use_pruning = cfg.get('use_pruning', True)
    graph_cache = os.path.join(output_dir, 'sparse_graph_cache.json')

    if use_pruning:
        sparse_graph = build_pruned_graph(
            queries_raw, global_graph,
            top_k_paths=cfg.get('top_k_paths', 10),
            max_path_length=cfg.get('max_path_length', 100),
            t_spanner=cfg.get('t_spanner', 1.1),
            cache_file=graph_cache,
            overwrite=cfg.get('overwrite', False),
            verbose=cfg.get('verbose', False),
        )
    else:
        # Use the complete global graph (no pruning)
        print(f"\n[Step 5a] Using complete global graph (no pruning)")
        print(f"  Global graph: {global_graph.number_of_nodes()} nodes, "
              f"{global_graph.number_of_edges()} edges")
        sparse_graph = global_graph

    # Save pruned graph for reference
    pruned_graph_path = os.path.join(output_dir, 'pruned_graph.json')
    save_pruned_graph(sparse_graph, pruned_graph_path, metadata={
        'use_pruning': use_pruning,
        'k': cfg.get('top_k_paths', 10),
        't_spanner': cfg.get('t_spanner', 1.1),
        'max_path_length': cfg.get('max_path_length', 100),
    })

    # Step 5b: Edge Verification
    verification_method = cfg.get('verification_method', 'cluster')
    gt_csv = cfg.get('ground_truth_csv', '')

    if verification_method == 'per_edge':
        # EcoTable original: per-edge verification
        results, verified_edges = run_per_edge_validation(
            queries_raw, sparse_graph, json_data, edge_details,
            gt_csv, cfg['tables_dir'], output_dir,
            api_key=cfg['llm_api_key'], model=cfg['llm_model'],
            sample_rows=cfg.get('sample_rows', 5),
            overwrite=cfg.get('overwrite', False),
            verbose=cfg.get('verbose', False),
        )
        eval_output = os.path.join(output_dir, 'evaluation_result.json')
        with open(eval_output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[Step 5b] Per-edge results -> {eval_output}")
        final_summary = {'method': 'per_edge', 'results': results}

    elif verification_method == 'cluster':
        # EcoTable+: cluster-based iterative verification
        cluster_size = cfg.get('cluster_size', 5)
        validation_dir = os.path.join(output_dir, f'edge_cluster_validation')
        run_name = f"r{cluster_size}_relax"
        if not use_pruning:
            run_name = f"r{cluster_size}_no_pruning"
        run_dir = os.path.join(validation_dir, run_name)

        # Build query dicts for iterative runner
        # Load GT for name mapping
        from .graph_validation.gt_name_mapper import GTNameMapper
        mapper = GTNameMapper(cfg['tables_dir'])
        gt_file = cfg.get('gt_file', '')

        queries_with_tables = []
        detailed_results = json.load(
            open(llm_filter_output, 'r', encoding='utf-8')
        ).get('detailed_results', [])

        graph_nodes = set(sparse_graph.nodes())
        for item in detailed_results:
            query_text = item.get('query', '')
            tables_raw = [extract_table_name(t) for t in item.get('selected_tables', []) if t]
            mapped, _ = mapper.map_query_tables(tables_raw)
            final_tables = [t for t in mapped if t in graph_nodes]
            queries_with_tables.append({
                'query': query_text,
                'required_tables': final_tables,
            })

        final_summary = run_cluster_validation(
            sparse_graph=sparse_graph,
            queries_with_tables=queries_with_tables,
            tables_dir=cfg['tables_dir'],
            output_dir=run_dir,
            api_key=cfg['llm_api_key'],
            model=cfg['llm_model'],
            base_url=cfg.get('llm_base_url'),
            cluster_size=cluster_size,
            max_iter=cfg.get('max_iter', 10),
            max_workers=cfg.get('max_workers', 5),
        )
    else:
        raise ValueError(f"Unknown verification_method: {verification_method}")

    # ── Layer 3: Table Transformation (optional) ──
    transform_summary = None
    if cfg.get('enable_transform_agent', False):
        print("\n" + "=" * 80)
        print("  LAYER 3: Table Transformation")
        print("=" * 80)

        # Find the pruned graph with join columns
        if verification_method == 'cluster':
            # Cluster method saves final_summary.json with confirmed edges
            pg_path = pruned_graph_path
        else:
            pg_path = os.path.join(output_dir, 'edge_verification_cache',
                                   'pruned_graph_join_cols_version.json')

        transform_summary = run_layer3_transform(
            pruned_graph_path=pg_path,
            llm_filter_results=llm_filter_results,
            tables_dir=cfg['tables_dir'],
            output_dir=output_dir,
            model=cfg['llm_model'],
            max_iterations=cfg.get('transform_max_iterations', 10),
            max_workers=cfg.get('transform_max_workers', 8),
        )

    # ── Pipeline Complete ──
    elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"  Pipeline Complete!  ({elapsed:.1f}s)")
    print("=" * 80)
    print(f"  Verification method : {verification_method}")
    print(f"  Use pruning         : {use_pruning}")
    if verification_method == 'cluster':
        print(f"  Cluster size (r)    : {cfg.get('cluster_size', 5)}")
    print(f"  Output directory    : {output_dir}")

    pipeline_result = {
        'config': {
            'verification_method': verification_method,
            'use_pruning': use_pruning,
            'cluster_size': cfg.get('cluster_size', 5),
            'model': cfg['llm_model'],
        },
        'timing': {'total_seconds': round(elapsed, 1)},
        'output_dir': output_dir,
        'final_summary': final_summary,
    }
    if transform_summary:
        pipeline_result['transform_summary'] = transform_summary

    # Save pipeline summary
    summary_path = os.path.join(output_dir, 'pipeline_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(pipeline_result, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Pipeline summary -> {summary_path}")

    return pipeline_result


# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════════

def build_config_from_args(args) -> Dict:
    """Build pipeline config dict from argparse namespace + config file."""
    cfg = {}

    # Load config file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            file_cfg = json.load(f)
        # Flatten relevant sections
        cfg['llm_model'] = file_cfg.get('table_identification', {}).get(
            'llm_filter', {}).get('model', 'gpt-4o')
        cfg['llm_base_url'] = file_cfg.get('table_identification', {}).get(
            'llm_filter', {}).get('base_url', 'https://api.openai.com/v1')
        cfg['top_k_paths'] = file_cfg.get('graph_validation', {}).get(
            'join_evaluation', {}).get('top_k_paths', 10)
        cfg['max_path_length'] = file_cfg.get('graph_validation', {}).get(
            'join_evaluation', {}).get('max_path_length', 100)
        cfg['t_spanner'] = file_cfg.get('graph_validation', {}).get(
            'join_evaluation', {}).get('t_spanner', 1.1)
        cfg['sample_rows'] = file_cfg.get('graph_validation', {}).get(
            'join_evaluation', {}).get('sample_rows', 5)
        cfg['transform_max_iterations'] = file_cfg.get(
            'table_transformation', {}).get('max_iterations', 10)
        cfg['transform_max_workers'] = file_cfg.get(
            'table_transformation', {}).get('max_workers', 8)

        # Verification defaults from config file
        verif_cfg = file_cfg.get('graph_validation', {}).get('verification', {})
        cfg.setdefault('verification_method', verif_cfg.get('method', 'cluster'))
        cfg.setdefault('cluster_size', verif_cfg.get('cluster_size', 5))
        cfg.setdefault('use_pruning', verif_cfg.get('use_pruning', True))
        cfg.setdefault('max_iter', verif_cfg.get('max_iter', 10))
        cfg.setdefault('max_workers', verif_cfg.get('max_workers', 5))

    # CLI overrides
    cfg['schema_cache'] = args.schema_cache
    cfg['deepjoin_cache'] = args.deepjoin_cache
    cfg['tables_dir'] = args.tables_dir
    cfg['ground_truth_csv'] = args.ground_truth_csv
    cfg['gt_file'] = args.gt_file
    cfg['output_dir'] = args.output_dir
    cfg['llm_api_key'] = args.llm_api_key or os.environ.get('OPENAI_API_KEY', '')
    cfg['column_stats_file'] = args.column_stats_file
    cfg['concurrency'] = args.concurrency
    cfg['verbose'] = args.verbose
    cfg['overwrite'] = args.overwrite

    # Verification hyperparameters (CLI always overrides config file defaults)
    cfg['verification_method'] = args.verification_method
    cfg['cluster_size'] = args.cluster_size
    cfg['use_pruning'] = not args.no_pruning
    cfg['max_iter'] = args.max_iter
    cfg['max_workers'] = args.max_workers

    # Transform agent
    cfg['enable_transform_agent'] = args.enable_transform_agent

    # Override from CLI if explicitly set
    if args.llm_model:
        cfg['llm_model'] = args.llm_model
    if args.llm_base_url:
        cfg['llm_base_url'] = args.llm_base_url

    return cfg


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="EcoTable+ Pipeline — Unified Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # EcoTable+ (pruned graph + cluster r=5, default)
  python -m src.pipeline_main \\
      --schema_cache outputs/pipeline_results/step1_schema_linking_top30.json \\
      --deepjoin_cache outputs/pipeline_results/step4_deepjoin_top6.json \\
      --tables_dir data/benchmarks/NYC \\
      --ground_truth_csv data/training/stage2/trainDataForLayer2.csv \\
      --gt_file data/benchmarks/NYC/schema_linking_gt_nyc.json \\
      --llm_api_key "sk-xxx"

  # EcoTable original (per-edge, no pruning)
  python -m src.pipeline_main \\
      --verification_method per_edge --no_pruning \\
      ...

  # EcoTable+ w/o pruning (cluster on complete graph)
  python -m src.pipeline_main --no_pruning ...

  # EcoTable+ with cluster size r=3
  python -m src.pipeline_main --cluster_size 3 ...
""")

    # ── Input data ──
    parser.add_argument('--config', default='configs/default.json',
                        help='Config file (default: configs/default.json)')
    parser.add_argument('--schema_cache',
                        default='outputs/pipeline_results/step1_schema_linking_top30.json',
                        help='Cached schema linking results')
    parser.add_argument('--deepjoin_cache',
                        default='outputs/pipeline_results/step4_deepjoin_top6.json',
                        help='Cached DeepJoin predictions')
    parser.add_argument('--tables_dir', default='data/benchmarks/NYC',
                        help='Table data directory')
    parser.add_argument('--ground_truth_csv',
                        default='data/training/stage2/trainDataForLayer2.csv',
                        help='Ground truth CSV file')
    parser.add_argument('--gt_file',
                        default='data/benchmarks/NYC/schema_linking_gt_nyc.json',
                        help='Schema linking GT file (for name mapping)')
    parser.add_argument('--output_dir', default='outputs/pipeline_results',
                        help='Output directory')

    # ── LLM parameters ──
    parser.add_argument('--llm_api_key', default=None,
                        help='LLM API key (or set OPENAI_API_KEY env)')
    parser.add_argument('--llm_model', default=None,
                        help='LLM model name (default from config)')
    parser.add_argument('--llm_base_url', default=None,
                        help='LLM API base URL (default from config)')
    parser.add_argument('--column_stats_file', default=None,
                        help='Column stats file for LLM filter')
    parser.add_argument('--concurrency', type=int, default=10,
                        help='Max parallel LLM calls for Step 2')

    # ── Verification hyperparameters ──
    parser.add_argument('--verification_method', default='cluster',
                        choices=['per_edge', 'cluster'],
                        help='Edge verification method (default: cluster)')
    parser.add_argument('--cluster_size', type=int, default=5,
                        help='Max cluster size r (default: 5)')
    parser.add_argument('--no_pruning', action='store_true',
                        help='Disable graph pruning (use complete graph)')
    parser.add_argument('--max_iter', type=int, default=10,
                        help='Max validation iterations (default: 10)')
    parser.add_argument('--max_workers', type=int, default=5,
                        help='Concurrent workers for verification (default: 5)')

    # ── Transform Agent ──
    parser.add_argument('--enable_transform_agent', action='store_true',
                        help='Enable Layer 3 Transform Agent')

    # ── General ──
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite all caches')

    args = parser.parse_args()
    cfg = build_config_from_args(args)

    # Print configuration summary
    print("=" * 80)
    print("  EcoTable+ Pipeline Configuration")
    print("=" * 80)
    print(f"  Verification method : {cfg['verification_method']}")
    print(f"  Use pruning         : {cfg['use_pruning']}")
    print(f"  Cluster size (r)    : {cfg['cluster_size']}")
    print(f"  Max iterations      : {cfg['max_iter']}")
    print(f"  Max workers         : {cfg['max_workers']}")
    print(f"  LLM model           : {cfg.get('llm_model', 'gpt-4o')}")
    print(f"  Output directory    : {cfg['output_dir']}")
    print("=" * 80)

    asyncio.run(run_pipeline(cfg))


if __name__ == "__main__":
    main()
