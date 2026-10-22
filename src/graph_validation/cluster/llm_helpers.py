"""
LLM helper functions for cluster-based edge validation.

Provides:
  - call_llm(): Raw LLM API call wrapper
  - parse_json_response(): Extract JSON from LLM response text
  - parse_selected_pairs(): Parse "Table.col <-> Table.col" strings
  - serialize_table(): Serialize DataFrame to autoprep format
  - serialize_col_values(): Serialize column values for Phase2
  - build_candidate_pairs(): Heuristic column-pair matching
  - run_phase2_single(): Phase2 data-level validation for one edge
"""

import json
import re
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI


def call_llm(client: OpenAI, model: str, messages: list,
             temperature: float = 0.1, max_tokens: int = 8192) -> dict:
    """Call LLM API and return response + usage stats.

    NOTE: max_tokens is explicitly set (default 8192) to support reasoning models
    such as GLM-5.3 which consume output budget for internal `reasoning_content`.
    """
    start = time.time()
    resp = client.chat.completions.create(
        model=model, messages=messages,
        temperature=temperature, max_tokens=max_tokens,
    )
    content = resp.choices[0].message.content
    # Reasoning models (GLM-5.3 etc.) may put the real answer in `reasoning_content`
    # when `content` is empty; keep `content` authoritative but surface reasoning
    # length for debugging.
    reasoning = getattr(resp.choices[0].message, 'reasoning_content', None)
    return {
        'response': content if content else (reasoning or ''),
        'tokens': resp.usage.total_tokens,
        'input_tokens': resp.usage.prompt_tokens,
        'output_tokens': resp.usage.completion_tokens,
        'latency': time.time() - start,
    }


def parse_json_response(text: str) -> dict:
    """Extract the first JSON object from LLM response text."""
    if '```' in text:
        m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if m:
            text = m.group(1).strip()
    s, e = text.find('{'), text.rfind('}') + 1
    if s >= 0 and e > s:
        return json.loads(text[s:e])
    return {}


def parse_selected_pairs(selected_pairs: List[str]) -> List[Tuple[str, str]]:
    """Parse 'Table1.col1 <-> Table2.col2' into (col1, col2) tuples."""
    pairs = []
    for ps in selected_pairs:
        parts = ps.split("<->")
        if len(parts) == 2:
            pairs.append((
                parts[0].strip().split(".")[-1].strip(),
                parts[1].strip().split(".")[-1].strip(),
            ))
    return pairs


def serialize_table(df: pd.DataFrame, name: str, nrows: int = 5) -> str:
    """Serialize first nrows of a DataFrame in autoprep format."""
    d = df.head(nrows)
    ret = f"<title>\n{name}\ncol : " + ' | '.join(d.columns) + '\n'
    for i in range(len(d)):
        ret += f'row{i+1} : ' + ' | '.join(str(x) for x in d.iloc[i].values) + '\n'
    return ret.strip() + "\n"


def serialize_col_values(df1: pd.DataFrame, df2: pd.DataFrame,
                         c1: str, c2: str, max_vals: int = 200) -> str:
    """Serialize two columns' unique values for Phase2 comparison."""
    try:
        v1 = df1[c1].dropna().astype(str).unique()[:max_vals]
    except KeyError:
        v1 = []
    try:
        v2 = df2[c2].dropna().astype(str).unique()[:max_vals]
    except KeyError:
        v2 = []
    return (
        f"### Column Pair for Detailed Analysis\n"
        f"**Table 1 Column**: {c1}\n**Values** (n={len(v1)}):\n{list(v1)}\n\n"
        f"**Table 2 Column**: {c2}\n**Values** (n={len(v2)}):\n{list(v2)}\n"
    )


def build_candidate_pairs(df1: pd.DataFrame,
                          df2: pd.DataFrame) -> List[Tuple[str, str]]:
    """Heuristic: find column pairs with matching names."""
    cands = [
        (c1, c2)
        for c1 in df1.columns for c2 in df2.columns
        if c1.lower() == c2.lower()
        or c1.lower().replace('_', '') == c2.lower().replace('_', '')
    ]
    if not cands:
        cands = [(c1, c2) for c1 in df1.columns[:2] for c2 in df2.columns[:2]]
    return cands


# ── Phase-2 prompt template ──
PHASE2_PROMPT = """Analyze the column pair below. Return a JSON object:
{{
  "data_quality": "valid or invalid",
  "same_attribute": "yes or no",
  "alignment_method": "2-3 example value pairs if joinable, else N/A",
  "can_join": "1 or 0",
  "reason": "Brief explanation."
}}

---

{column_data}"""


def run_phase2_single(client: OpenAI, model: str,
                      df1: pd.DataFrame, df2: pd.DataFrame,
                      col_pairs: List[Tuple[str, str]],
                      max_pairs: int = 3):
    """Run Phase2 data-level validation for one edge.

    Returns:
        (is_joinable, phase2_details_list, total_tokens, total_latency)
    """
    total_tok = total_lat = 0
    details = []
    for c1, c2 in col_pairs[:max_pairs]:
        col_data = serialize_col_values(df1, df2, c1, c2)
        prompt = PHASE2_PROMPT.format(column_data=col_data)
        msgs = [
            {"role": "system", "content": "You are a rigorous data architect."},
            {"role": "user", "content": prompt},
        ]
        try:
            res = call_llm(client, model, msgs)
            total_tok += res['tokens']
            total_lat += res['latency']
            parsed = parse_json_response(res['response'])
            can_join = str(parsed.get('can_join', '0')).strip() == '1'
            details.append({
                'columns': [c1, c2], 'can_join': can_join,
                'raw_response': res['response'], 'parsed': parsed,
                'tokens': res['tokens'], 'latency': res['latency'],
            })
            if can_join:
                return True, details, total_tok, total_lat
        except Exception as e:
            details.append({'columns': [c1, c2], 'can_join': False, 'error': str(e)})
    return False, details, total_tok, total_lat
