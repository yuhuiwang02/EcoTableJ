"""
Shortest-path adapter for EcoTable Layer-2 (Join Evaluation).

Bypasses Layer-1 (schema linking cross-encoder + LLM filter) and Layer-4
(DeepJoin) by constructing the two inputs that `run_join_evaluation` needs
directly from the dataset ground-truth / schema-linking ground-truth files:

  1. candidate_edges.json  -> json_file  (DeepJoin-style output: table1/table2/
                                                    column1/column2/prob)
  2. queries_llm_filter.json -> queries_file (LLM-filter-style output:
                                                    detailed_results[].selected_tables)

For datasets WITHOUT a *_join_relationship_gt.csv (e.g. NYC), pass
--no-join-gt to synthesize candidate edges from the schema-linking related
tables (table-level fully-connected within the related-table universe).
These synthesized edges are NOT ground truth, so F1 against a GT is invalid;
they only serve to exercise the graph-build + Steiner + edge-validation pipeline.

Run:
    python make_shortest_path_inputs.py --dataset ad
    python make_shortest_path_inputs.py --dataset nyc --no-join-gt
"""
import argparse
import csv
import json
import os
from itertools import combinations


def build_candidate_edges(join_gt_csv: str, prob: float = 0.99):
    """Convert *_join_relationship_gt.csv into DeepJoin-style candidate edges.

    CSV columns: table1, table2, column1, column2, [flag]
    """
    edges = []
    with open(join_gt_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 4:
                continue
            t1, t2, c1, c2 = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
            if not t1 or not t2:
                continue
            edges.append({
                'table1': t1,
                'table2': t2,
                'column1': c1,
                'column2': c2,
                'prob': prob,
            })
    return edges


def build_candidate_edges_from_schema(queries: list, prob: float = 0.5):
    """Synthesize candidate edges by fully-connecting the related-table universe.

    Used when no join_relationship_gt.csv exists. Node ids are the subtable
    names exactly as they appear in query_related_table (which match the CSV
    file names under each dataset's tables/ directory).
    """
    universe = set()
    for q in queries:
        for t in q.get('selected_tables', []):
            universe.add(t)
    universe = sorted(universe)
    edges = []
    for t1, t2 in combinations(universe, 2):
        edges.append({
            'table1': t1,
            'table2': t2,
            'column1': '',
            'column2': '',
            'prob': prob,
        })
    return edges


def build_queries(schema_linking_gt_json: str):
    """Convert *_schema_linking_gt.json (query_related_table) into LLM-filter
    output format expected by load_queries_from_llm_filter().

    Expected structure (per item in detailed_results):
        {
          'selected_tables': ['table_a', 'table_b', ...],   # table names
          'llm_interaction': {'parsed_result': {'selected_tables': [...]}}  # optional
        }
    load_queries_from_llm_filter reads selected_tables first, then falls back to
    llm_interaction.parsed_result.selected_tables.
    """
    with open(schema_linking_gt_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    detailed_results = []
    skipped = 0
    for item in data:
        related = item.get('query_related_table', [])
        if not related or len(related) < 2:
            skipped += 1
            continue
        # table names only; extract_table_name() just strips a trailing .csv
        detailed_results.append({
            'query': item.get('query', ''),
            'selected_tables': list(related),
            'llm_interaction': {
                'parsed_result': {
                    'selected_tables': list(related)
                }
            }
        })
    return detailed_results, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
                        help='Dataset name, e.g. ad / platform / business / engagement / nyc')
    parser.add_argument('--root', default='.', help='Repo root directory')
    parser.add_argument('--prob', type=float, default=0.99,
                        help='Probability assigned to GT candidate edges')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit number of queries (0 = all). Used for small-sample runs.')
    parser.add_argument('--no-join-gt', action='store_true',
                        help='Synthesize candidate edges from schema-linking related tables '
                             '(for datasets without *_join_relationship_gt.csv, e.g. NYC). '
                             'Edges are NOT ground truth; F1 is invalid in this mode.')
    args = parser.parse_args()

    # Map dataset folder -> file prefix for schema-linking GT.
    # NOTE: NYC's file is literally 'schema_linking_gt_nyc.json' (lowercase suffix),
    # while others are '{Prefix}_schema_linking_gt.json'.
    schema_suffix_map = {'ad': 'Ads', 'platform': 'Platform', 'business': 'Business',
                         'engagement': 'Engagement', 'nyc': 'nyc'}
    cap = schema_suffix_map.get(args.dataset.lower(), args.dataset.capitalize())

    # join_relationship_gt prefix is capitalized (Ads/Platform/.../NYC) except nyc.
    join_prefix_map = {'ad': 'Ads', 'platform': 'Platform', 'business': 'Business',
                       'engagement': 'Engagement', 'nyc': 'NYC'}

    ds_dir = args.dataset  # keep original casing for the directory name
    join_gt_csv = os.path.join(args.root, ds_dir, f'{join_prefix_map[args.dataset.lower()]}_join_relationship_gt.csv')
    if args.dataset.lower() == 'nyc':
        schema_gt_json = os.path.join(args.root, ds_dir, f'schema_linking_gt_{cap}.json')
    else:
        schema_gt_json = os.path.join(args.root, ds_dir, f'{cap}_schema_linking_gt.json')

    if not os.path.exists(schema_gt_json):
        raise FileNotFoundError(schema_gt_json)

    queries, skipped = build_queries(schema_gt_json)

    if args.no_join_gt:
        if os.path.exists(join_gt_csv):
            print(f"[Adapter] WARNING: {join_gt_csv} exists but --no-join-gt was given; "
                  f"ignoring GT and synthesizing edges from schema linking.")
        edges = build_candidate_edges_from_schema(queries, prob=args.prob)
    else:
        if not os.path.exists(join_gt_csv):
            raise FileNotFoundError(
                f"{join_gt_csv} not found. This dataset has no join_relationship_gt.csv; "
                f"pass --no-join-gt to synthesize candidate edges from schema linking "
                f"(pipeline-only, F1 invalid)."
            )
        edges = build_candidate_edges(join_gt_csv, prob=args.prob)

    if args.limit and args.limit > 0:
        queries = queries[:args.limit]
        suffix = f'_top{args.limit}'
    else:
        suffix = ''

    edges_out = os.path.join(args.root, ds_dir, 'candidate_edges.json')
    queries_out = os.path.join(args.root, ds_dir, f'queries_llm_filter{suffix}.json')

    with open(edges_out, 'w', encoding='utf-8') as f:
        json.dump(edges, f, indent=2, ensure_ascii=False)
    with open(queries_out, 'w', encoding='utf-8') as f:
        json.dump({'metadata': {'model': 'gt-shortest-path' if not args.no_join_gt else 'schema-synth'},
                   'detailed_results': queries}, f, indent=2, ensure_ascii=False)

    print(f"[Adapter] candidate edges : {len(edges)} -> {edges_out}")
    print(f"[Adapter] queries         : {len(queries)} (skipped {skipped} with <2 tables) -> {queries_out}")


if __name__ == '__main__':
    main()
