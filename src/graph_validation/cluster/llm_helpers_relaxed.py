"""
Relaxed LLM helper functions for cluster-based edge validation.

This module is a drop-in replacement for llm_helpers.py with a relaxed Phase2
prompt that reduces false negatives by:
  - Accepting partial value overlap as sufficient evidence for joinability
  - Tolerating format differences (prefixes, case, whitespace, type casting)

Usage:
  In cluster_verifier.py or run scripts, replace:
    from .llm_helpers import ... run_phase2_single ...
  with:
    from .llm_helpers_relaxed import ... run_phase2_single ...

  Or point PROMPT_CLUSTER to prompt_cluster_relaxed.txt.
"""

import json
import re
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI


# ── Re-export unchanged helpers from original module ──
from .llm_helpers import (
    call_llm,
    parse_json_response,
    parse_selected_pairs,
    serialize_table,
    serialize_col_values,
    build_candidate_pairs,
)


# ── Relaxed Phase-2 prompt template ──
PHASE2_PROMPT_RELAXED = """Analyze the column pair below and determine whether they can serve as a join key.

**Important context**: In real-world data integration scenarios, tables that should be joined often have imperfect value alignment. Keep the following in mind:
- Their value sets may only **partially overlap** (even a small overlap is sufficient for a valid join).
- Values may differ in **format** (e.g., "BIN-12345" vs "12345", "NYC DOB" vs "DOB", leading/trailing spaces, case differences). Such formatting differences do NOT invalidate a join — they can be resolved by simple data cleaning.
- One column may contain a **subset** of the other's values, or both may share only a fraction of values. This still constitutes a valid join.

**Decision criteria**:
- `can_join = "1"`: The two columns refer to the **same real-world attribute** AND there is **any evidence of value correspondence** (even partial overlap, or overlap after reasonable normalization like trimming prefixes, case-folding, or type casting).
- `can_join = "0"`: The columns clearly refer to **different attributes**, OR the values are from **completely unrelated domains** with zero plausible correspondence even after normalization.

Return a JSON object:
{{
  "data_quality": "valid or invalid",
  "same_attribute": "yes or no",
  "overlap_evidence": "List 2-3 matching or near-matching value pairs if any exist, else N/A",
  "can_join": "1 or 0",
  "reason": "Brief explanation."
}}

---

{column_data}"""


def run_phase2_single(client: OpenAI, model: str,
                      df1: pd.DataFrame, df2: pd.DataFrame,
                      col_pairs: List[Tuple[str, str]],
                      max_pairs: int = 3):
    """Run Phase2 data-level validation for one edge (relaxed version).

    Identical logic to the original run_phase2_single, but uses the relaxed
    PHASE2_PROMPT_RELAXED that tolerates partial overlap and format differences.

    Returns:
        (is_joinable, phase2_details_list, total_tokens, total_latency)
    """
    total_tok = total_lat = 0
    details = []
    for c1, c2 in col_pairs[:max_pairs]:
        col_data = serialize_col_values(df1, df2, c1, c2)
        prompt = PHASE2_PROMPT_RELAXED.format(column_data=col_data)
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
