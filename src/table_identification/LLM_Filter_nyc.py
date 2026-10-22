import json
import asyncio
import openai
import re
import os
import sys
from datetime import datetime
from colorama import init, Fore, Style
import numpy as np

init(autoreset=True)

TOTAL_PROMPT_TOKENS = 0
TOTAL_COMPLETION_TOKENS = 0


def make_serializable(obj, max_depth=10, current_depth=0):
    if current_depth > max_depth:
        return f"<max_depth_reached: {type(obj).__name__}>"
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): make_serializable(v, max_depth, current_depth + 1)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(item, max_depth, current_depth + 1)
                for item in obj]
    try:
        return str(obj)
    except:
        return f"<non-serializable: {type(obj).__name__}>"


# ── System Prompt (from technical report, enhanced with JSON constraint) ─────
SYSTEM_PROMPT = (
    "You are a data lake schema expert specializing in large-scale, open government "
    "data lakes. Your role is to identify, from a pool of candidate tables retrieved "
    "by a deep learning model, the minimal and semantically correct set of tables "
    "required to answer a given natural language query.\n\n"
    "You reason at the column level: a table is relevant if and only if its columns "
    "directly supply information requested by the query. In this data lake, tables may "
    "share naming prefixes and be organized by temporal scope or horizontal data "
    "partitioning; your reasoning must account for both aspects.\n\n"
    "Return valid JSON only, no markdown."
)

# ── Step 1 Prompt: Query Understanding ──────────────────────────────────────
STEP1_PROMPT_TEMPLATE = """\
Step 1: Query Understanding

Given the following natural language query, decompose it into a structured list of \
atomic semantic components. Each component should represent one self-contained \
information need, such as:
- A key entity or subject (e.g., "taxi trip", "building permit").
- A measurable attribute or metric (e.g., "trip duration", "fare").
- A filter or scoping condition (e.g., "yellow cab only").
- A temporal specification (e.g., "in 2017", "FY 2018-2019").

Pay particular attention to temporal references. If the query mentions a specific year, \
fiscal year, or date range, record it explicitly, as it will be used to resolve temporal \
ambiguity among candidate tables in the verification step.

Input
Query: {query}

Output Format (JSON Only)
{{
    "temporal_scope": "<year / fiscal year / date range, or null>",
    "semantic_components": [
        "component 1: <description>",
        "component 2: <description>"
    ]
}}"""

# ── Step 2 Prompt: Schema Reasoning and Alignment ──────────────────────────
STEP2_PROMPT_TEMPLATE = """\
Step 2: Schema Reasoning and Alignment

You are given the original query, its decomposed semantic components (including \
temporal scope), and a ranked list of candidate tables from the data lake. Each table \
entry includes its name and a sample of its column names.

For each candidate table, reason step-by-step to determine whether the table directly \
contributes at least one column that satisfies a semantic component of the query. \
Apply the following principles in order:

- Column-Level Matching: A table is relevant only if its columns directly provide \
data requested by the query. Do not rely on table names alone.
- Temporal Scope Consistency: If a temporal scope exists, discard tables whose names \
encode a different year or fiscal year, even if schemas match.
- Partition-Complete Selection: In this data lake, large datasets are horizontally \
partitioned into sub-tables that share a common base name with a numeric suffix \
(e.g., records, records_1, records_2). These sub-tables hold disjoint rows of the \
same logical table — they are NOT duplicates. A query that needs data from one \
partition almost always needs data from ALL partitions of that table, because the \
required rows could reside in any partition. \
RULE: Once you decide to keep ANY table whose name follows the pattern <base> or \
<base>_N, you MUST also keep EVERY other candidate table whose name matches <base> \
or <base>_M (where M is any number). Do NOT evaluate partitions independently — \
include the entire partition family as a unit.
- Dataset Coherence: Each candidate table name follows the pattern \
`<DatasetName>_<subtable>`. The Dataset field shown for each candidate identifies \
which parent dataset it belongs to. If the query explicitly references a specific \
dataset by name (e.g., "DOB NOW Job Application Filings", "2019 DOE High School \
Directory"), you MUST strongly prefer tables whose Dataset matches the referenced \
dataset. Tables from a different dataset — even one with a very similar name, \
overlapping columns, or a different year/version — should be discarded unless the \
query clearly requires cross-dataset information. Pay special attention to: \
(a) year/version differences in dataset names (e.g., "2018 DOE High School Directory" \
vs "2019 DOE High School Directory"); (b) subtle naming variants (e.g., "DOB NOW: \
Build – Job Application Filings" vs "DOB Job Application Filings"); \
(c) historical vs current versions (e.g., "LinkNYC Kiosk Status" vs \
"LinkNYC Kiosk Status (Historical)"). These are DIFFERENT datasets.
- Non-Redundancy: If two tables have identical schemas and partitions, retain only \
the one with broader coverage or the lower partition index. Do not deduplicate \
based on naming similarity alone.

Input
Query: {query}
Temporal Scope: {temporal_scope}
Semantic Components: {semantic_components}
Candidate Tables:
{candidate_tables_with_columns}

Output Format (JSON Only)
{{
    "table_reasoning": {{
        "<table_name>": {{
            "matched_components": ["component i"],
            "matched_columns": ["col_a"],
            "temporal_consistent": true,
            "verdict": "keep",
            "reason": "<one-sentence justification>"
        }}
    }},
    "selected_tables": ["table_1", "table_2"]
}}"""


def parse_json_response(raw: str) -> dict:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    content = raw.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    # Fallback: find the outermost JSON object
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(0))
    return json.loads(content)


class LLMTableFilter:

    def __init__(self, api_key, model="gpt-4o", base_url="https://www.dmxapi.com/v1"):
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model

    async def filter_candidates(self, query: str, candidates: list,
                                 column_info: dict) -> dict:
        """
        Two-step LLM filter (Paper V6 best version):
          Step 1: Query Understanding -> temporal_scope + semantic_components
          Step 2: Schema Reasoning    -> selected_tables (with Step 1 context)
        """
        global TOTAL_PROMPT_TOKENS, TOTAL_COMPLETION_TOKENS

        llm_interaction = {
            "step1_prompt": None,
            "step1_response": None,
            "step2_prompt": None,
            "step2_response": None,
            "parsed_result": None,
            "error": None,
            "usage": {"step1": None, "step2": None}
        }

        # ── Step 1: Query Understanding ──────────────────────────────────
        step1_prompt = STEP1_PROMPT_TEMPLATE.format(query=query)
        llm_interaction["step1_prompt"] = step1_prompt

        try:
            resp1 = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": step1_prompt},
                ],
                temperature=0,
                max_tokens=1024,
            )
            raw1 = resp1.choices[0].message.content.strip()
            llm_interaction["step1_response"] = raw1

            if resp1.usage:
                TOTAL_PROMPT_TOKENS += resp1.usage.prompt_tokens
                TOTAL_COMPLETION_TOKENS += resp1.usage.completion_tokens
                llm_interaction["usage"]["step1"] = {
                    "prompt_tokens": resp1.usage.prompt_tokens,
                    "completion_tokens": resp1.usage.completion_tokens,
                }

            parsed1 = parse_json_response(raw1)
            temporal_scope = parsed1.get("temporal_scope")
            semantic_components = parsed1.get("semantic_components", [])

        except Exception as e:
            llm_interaction["error"] = f"Step1: {e}"
            return {
                'filtered_tables': [t for t, s in candidates[:10]],
                'temporal_scope': 'error',
                'summary': {},
                'llm_interaction': llm_interaction
            }

        # ── Build candidate text with columns and Dataset field ──────────
        cand_lines = []
        for i, (table_name, score) in enumerate(candidates[:30], 1):
            cols = list(column_info.get(table_name, {}).keys())
            cols_str = ", ".join(cols[:30])
            if len(cols) > 30:
                cols_str += f" ... (+{len(cols) - 30} more)"
            # Extract dataset name from table name
            dataset_name = table_name
            for j, ch in enumerate(table_name):
                if ch == '_':
                    rest = table_name[j+1:]
                    if rest and rest[0].islower():
                        dataset_name = table_name[:j]
                        break
            cand_lines.append(
                f"{i}. `{table_name}` (Score: {score:.6f})\n"
                f"   Dataset: {dataset_name}\n"
                f"   Columns: {cols_str}"
            )
        cand_text = "\n".join(cand_lines)

        sc_str = json.dumps(semantic_components, ensure_ascii=False)

        # ── Step 2: Schema Reasoning (with Step 1 conversation history) ──
        step2_prompt = STEP2_PROMPT_TEMPLATE.format(
            query=query,
            temporal_scope=temporal_scope if temporal_scope else "null",
            semantic_components=sc_str,
            candidate_tables_with_columns=cand_text,
        )
        llm_interaction["step2_prompt"] = step2_prompt

        try:
            resp2 = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": step1_prompt},
                    {"role": "assistant", "content": raw1},
                    {"role": "user",      "content": step2_prompt},
                ],
                temperature=0.1,
                max_tokens=8000,
            )
            raw2 = resp2.choices[0].message.content.strip()
            llm_interaction["step2_response"] = raw2

            if resp2.usage:
                TOTAL_PROMPT_TOKENS += resp2.usage.prompt_tokens
                TOTAL_COMPLETION_TOKENS += resp2.usage.completion_tokens
                llm_interaction["usage"]["step2"] = {
                    "prompt_tokens": resp2.usage.prompt_tokens,
                    "completion_tokens": resp2.usage.completion_tokens,
                }

            parsed2 = parse_json_response(raw2)
            llm_interaction["parsed_result"] = parsed2
            selected_tables = parsed2.get('selected_tables', [])

            if len(selected_tables) == 0:
                selected_tables = [t for t, s in candidates[:10]]

            return {
                'filtered_tables': selected_tables,
                'temporal_scope': temporal_scope,
                'semantic_components': semantic_components,
                'summary': parsed2.get('table_reasoning', {}),
                'llm_interaction': llm_interaction
            }

        except Exception as e:
            llm_interaction["error"] = f"Step2: {e}"
            return {
                'filtered_tables': [t for t, s in candidates[:10]],
                'temporal_scope': temporal_scope,
                'semantic_components': semantic_components,
                'summary': {},
                'llm_interaction': llm_interaction
            }


def load_all_table_columns(stats_file: str) -> dict:
    with open(stats_file, 'r', encoding='utf-8') as f:
        stats = json.load(f)

    table_columns = {}
    for category in ['normal', 'no_overlap', 'no_join_cols']:
        for table_info in stats.get(category, []):
            parent_name = table_info['table_name']
            if 'node_mapping' in table_info:
                for node_id, node_info in table_info['node_mapping'].items():
                    subtable_name = node_info['subtable_file']
                    columns = node_info.get('columns', [])

                    table_columns[subtable_name] = columns

                    if subtable_name.startswith('node_'):
                        full_name = f"{parent_name}_{subtable_name}"
                        table_columns[full_name] = columns

    return table_columns


def prepare_data(results_file: str, all_columns: dict) -> list:

    with open(results_file, 'r', encoding='utf-8') as f:
        results_data = json.load(f)

    prepared_data = []
    for sample in results_data:
        column_info = {}
        for table_name, score in sample['candidates']:
            if table_name in all_columns:
                column_info[table_name] = {col: '' for col in all_columns[table_name]}

        prepared_data.append({
            'sample_id': sample['sample_id'],
            'query': sample['query'],
            'sql': sample.get('sql', ''),
            'candidates': sample['candidates'],
            'column_info': column_info,
            'ground_truth': sample['gt']
        })

    return prepared_data


async def main():

    API_KEY = ""
    API_BASE = ""
    MODEL = ""

    filter_obj = LLMTableFilter(
        api_key=API_KEY,
        model=MODEL,
        base_url=API_BASE
    )

    results_file = ''
    stats_file = ''

    all_data = prepare_data(results_file, all_columns)

    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else None
    if num_samples and num_samples < len(all_data):
        import random
        random.seed(42)
        sample_indices = random.sample(range(len(all_data)), num_samples)
        sample_data = [all_data[i] for i in sorted(sample_indices)]
    else:
        sample_data = all_data
        sample_indices = list(range(len(all_data)))


    results = []
    for idx, item in enumerate(sample_data, 1):

        filter_result = await filter_obj.filter_candidates(
            item['query'], item['candidates'], item['column_info']
        )

        selected_tables = filter_result['filtered_tables']

        gt_set = set(item['ground_truth'])
        pred_set = set(selected_tables)

        # Stage1 Top-30 metrics
        top30_set = set([t for t, s in item['candidates'][:30]])
        top30_tp = len(gt_set & top30_set)
        top30_fp = len(top30_set - gt_set)
        top30_fn = len(gt_set - top30_set)
        top30_precision = top30_tp / (top30_tp + top30_fp) if (top30_tp + top30_fp) > 0 else 0
        top30_recall = top30_tp / (top30_tp + top30_fn) if (top30_tp + top30_fn) > 0 else 0
        top30_f1 = 2 * top30_precision * top30_recall / (top30_precision + top30_recall) if (top30_precision + top30_recall) > 0 else 0

        # LLM Filter metrics
        tp = len(gt_set & pred_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"{Fore.GREEN}Selected ({len(selected_tables)}): {selected_tables}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}[Stage1 Top-30] P: {top30_precision:.4f}, R: {top30_recall:.4f}, F1: {top30_f1:.4f} ({len(top30_set)} tables){Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}[LLM Filter]    P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f} ({len(selected_tables)} tables){Style.RESET_ALL}")

        if fn > 0:
            print(f"{Fore.RED}Missing: {gt_set - pred_set}{Style.RESET_ALL}")
        if fp > 0:
            print(f"{Fore.YELLOW}Extra: {pred_set - gt_set}{Style.RESET_ALL}")

        results.append({
            'sample_id': item['sample_id'],
            'query': item['query'],
            'sql': item['sql'],
            'ground_truth': item['ground_truth'],
            'candidates_top30': item['candidates'][:30],
            'selected_tables': selected_tables,
            'num_selected': len(selected_tables),
            'year_detected': filter_result.get('temporal_scope', 'null'),
            'summary': filter_result['summary'],
            'llm_interaction': filter_result['llm_interaction'],
            'stage1_metrics': {
                'precision': top30_precision,
                'recall': top30_recall,
                'f1': top30_f1,
                'num_tables': len(top30_set)
            },
            'llm_filter_metrics': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
        })

        await asyncio.sleep(1.0)

    avg_s1_precision = np.mean([r['stage1_metrics']['precision'] for r in results])
    avg_s1_recall = np.mean([r['stage1_metrics']['recall'] for r in results])
    avg_s1_f1 = np.mean([r['stage1_metrics']['f1'] for r in results])

    avg_precision = np.mean([r['llm_filter_metrics']['precision'] for r in results])
    avg_recall = np.mean([r['llm_filter_metrics']['recall'] for r in results])
    avg_f1 = np.mean([r['llm_filter_metrics']['f1'] for r in results])
    avg_num_selected = np.mean([r['num_selected'] for r in results])

    cost_input = TOTAL_PROMPT_TOKENS / 1_000_000 * 2.5
    cost_output = TOTAL_COMPLETION_TOKENS / 1_000_000 * 10

    output_dir = ''
    output_file = os.path.join(output_dir, f'llm_filter_clean_863_{len(results)}samples_final.json')

    output_data = {
        'metadata': {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': MODEL,
            'source': results_file,
            'total_samples': len(results),
            'sample_indices': sample_indices,
            'token_usage': {
                'prompt_tokens': TOTAL_PROMPT_TOKENS,
                'completion_tokens': TOTAL_COMPLETION_TOKENS,
                'total_tokens': TOTAL_PROMPT_TOKENS + TOTAL_COMPLETION_TOKENS,
                'estimated_cost_usd': cost_input + cost_output
            }
        },
        'stage1_metrics': {
            'precision': avg_s1_precision,
            'recall': avg_s1_recall,
            'f1': avg_s1_f1
        },
        'llm_filter_metrics': {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1,
            'avg_num_selected': avg_num_selected
        },
        'detailed_results': make_serializable(results)
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
