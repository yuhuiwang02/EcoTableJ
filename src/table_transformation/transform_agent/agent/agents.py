"""
PromptAgent: ReAct-style LLM agent for table transformation.

Implements the iterative Reason-Act-Observe loop described in EcoTable paper Section 5:
  1. Observe: GetLeftAnti + EditDistScan → unmatched samples + fuzzy candidates
  2. Reason: LLM analyzes and recommends a transformation operator
  3. Act: Execute the recommended operator on the DuckDB tables
  4. Evaluate: ValidateJoin computes containment ratio
  5. Loop: Repeat until LLM commits or max_iter reached

Each iteration's conversation and results are saved to disk for cost accounting.
"""

import json
import logging
import os
import re
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .models import call_llm, LLMCallRecord
from .prompts_v2 import ANALYSIS_SYSTEM_PROMPT
from .operators.observation import GetLeftAnti, EditDistScan
from .operators.evaluation import ValidateJoin, TokenRatio
from .operators.transformation import (
    TextNorm, ToDate, ToNumeric, FuzzyMap,
    SepSplit, Pivot, UnPivot,
)

logger = logging.getLogger("transform_agent")

# Operator registry: maps operator name → class
OPERATOR_REGISTRY = {
    "TextNorm": TextNorm,
    "ToDate": ToDate,
    "ToNumeric": ToNumeric,
    "FuzzyMap": FuzzyMap,
    "SepSplit": SepSplit,
    "Pivot": Pivot,
    "UnPivot": UnPivot,
}


def _parse_llm_json(text: str) -> dict:
    """Extract JSON object from LLM response (handles markdown fences)."""
    if '```' in text:
        m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if m:
            text = m.group(1).strip()
    s, e = text.find('{'), text.rfind('}') + 1
    if s >= 0 and e > s:
        try:
            return json.loads(text[s:e])
        except json.JSONDecodeError:
            pass
    return {}


def _format_observation(obs_result: dict, edit_candidates: list,
                        table1: str, table2: str,
                        join_col1: str, join_col2: str,
                        exact_ratio: float,
                        structural_hints: str = "") -> str:
    """Format observation results into a prompt string for the LLM."""
    parts = []
    parts.append(f"## Current State")
    parts.append(f"- **Table 1**: `{table1}`, join column: `{join_col1}`")
    parts.append(f"- **Table 2**: `{table2}`, join column: `{join_col2}`")
    parts.append(f"- **ExactRatio (containment)**: {exact_ratio:.4f}")
    parts.append(f"- **Matched values**: {obs_result['matched_count']}")
    parts.append(f"- **Table 1 unmatched rows**: {obs_result['table1_total_unmatched']}")
    parts.append(f"- **Table 2 unmatched rows**: {obs_result['table2_total_unmatched']}")

    # Table 1 sample (full rows with headers)
    t1_sample = obs_result.get('table1_sample')
    if t1_sample is not None and len(t1_sample) > 0:
        parts.append(f"\n### Table 1 unmatched sample (5 rows, all columns):")
        parts.append(t1_sample.to_string(index=False, max_colwidth=50))

    # Table 2 sample
    t2_sample = obs_result.get('table2_sample')
    if t2_sample is not None and len(t2_sample) > 0:
        parts.append(f"\n### Table 2 unmatched sample (5 rows, all columns):")
        parts.append(t2_sample.to_string(index=False, max_colwidth=50))

    # Edit distance candidates
    if edit_candidates:
        parts.append(f"\n### Top edit-distance candidates (fuzzy matches):")
        for v1, v2, dist in edit_candidates[:10]:
            parts.append(f"  `{v1}` ↔ `{v2}` (dist={dist:.3f})")

    # Structural hints
    if structural_hints:
        parts.append(f"\n### Structural hints:")
        parts.append(structural_hints)

    return "\n".join(parts)


def _detect_structural_hints(obs_result: dict, table1: str, table2: str,
                             join_col1: str, join_col2: str,
                             duckdb_path: str) -> str:
    """Detect structural patterns (pivot/unpivot/split) from table schemas."""
    import duckdb
    hints = []

    with duckdb.connect(duckdb_path) as con:
        for tname, jcol in [(table1, join_col1), (table2, join_col2)]:
            cols = con.execute(f"PRAGMA table_info('{tname}')").fetchdf()['name'].tolist()

            # Check for column names containing the join column name (UnPivot signal)
            related_cols = [c for c in cols if jcol.lower() in c.lower() and c != jcol]
            if len(related_cols) >= 2:
                hints.append(
                    f"**UnPivot hint**: Table `{tname}` has {len(related_cols)} columns "
                    f"related to join column `{jcol}`: {related_cols[:5]}. "
                    f"Consider UnPivot to consolidate them."
                )

            # Check for merged column names with separators (SepSplit signal)
            for c in cols:
                if '-' in c and jcol.lower() in c.lower():
                    hints.append(
                        f"**SepSplit hint**: Table `{tname}` has column `{c}` "
                        f"which appears to be a merged column containing `{jcol}`."
                    )

            # Check for key-value pattern (Pivot signal)
            if len(cols) <= 5:
                # Small number of columns might indicate long format
                nrows = con.execute(f'SELECT COUNT(*) FROM "{tname}"').fetchone()[0]
                ndistinct = con.execute(
                    f'SELECT COUNT(DISTINCT "{cols[0]}") FROM "{tname}" WHERE "{cols[0]}" IS NOT NULL'
                ).fetchone()[0] if cols else 0
                if nrows > 0 and ndistinct > 0 and nrows / ndistinct > 3:
                    hints.append(
                        f"**Pivot hint**: Table `{tname}` has only {len(cols)} columns "
                        f"and high row-to-distinct ratio ({nrows}/{ndistinct}). "
                        f"May be in long (key-value) format."
                    )

    return "\n".join(hints) if hints else ""


class PromptAgent:
    """ReAct-style LLM agent for table transformation."""

    def __init__(self, model: str = "gpt-4o", max_tokens: int = 2500,
                 top_p: float = 0.9, temperature: float = 0.1):
        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature

    def run_new_workflow(
        self,
        duckdb_path: str,
        table1: str,
        table2: str,
        join_col1: str,
        join_col2: str,
        max_iterations: int = 10,
        test_mode: bool = False,
        output_dir: str = ".",
        pair_idx: int = 0,
    ) -> Dict[str, Any]:
        """Run the iterative ReAct workflow for one join column pair.

        Args:
            duckdb_path: Path to the DuckDB file.
            table1, table2: Table names.
            join_col1, join_col2: Join column names.
            max_iterations: Max ReAct iterations.
            test_mode: If True, skip LLM calls and just observe.
            output_dir: Directory to save conversation logs.
            pair_idx: Index of this column pair (for file naming).

        Returns:
            Dict with keys: final_exact_ratio, iterations, operations_applied,
            total_llm_latency_ms, total_llm_tokens, total_llm_cost, conversation.
        """
        # Initialize operators
        get_left_anti = GetLeftAnti(duckdb_path)
        edit_dist_scan = EditDistScan(duckdb_path)
        validate_join = ValidateJoin(duckdb_path)

        # Track state
        conversation: List[Dict] = []
        operations_applied: List[Dict] = []
        total_llm_latency_ms = 0.0
        total_llm_tokens = 0
        total_llm_cost = 0.0
        best_ratio = 0.0

        # Initial evaluation
        initial_ratio = validate_join.execute(table1, table2, join_col1, join_col2)
        best_ratio = initial_ratio
        logger.info(f"  Initial ExactRatio: {initial_ratio:.4f}")

        if initial_ratio >= 1.0:
            logger.info(f"  Already perfect match, no transformation needed")
            return {
                "final_exact_ratio": initial_ratio,
                "initial_exact_ratio": initial_ratio,
                "iterations": 0,
                "operations_applied": [],
                "total_llm_latency_ms": 0.0,
                "total_llm_tokens": 0,
                "total_llm_cost": 0.0,
                "committed": True,
                "conversation": [],
            }

        # Current join column names (may change after SepSplit)
        cur_col1, cur_col2 = join_col1, join_col2

        for iteration in range(1, max_iterations + 1):
            logger.info(f"  --- Iteration {iteration}/{max_iterations} ---")

            # ── Phase 1: Observe ──
            try:
                obs_result = get_left_anti.execute(table1, table2, cur_col1, cur_col2)
            except Exception as e:
                logger.warning(f"  GetLeftAnti failed: {e}")
                obs_result = {
                    'matched_values': [], 'matched_count': 0,
                    'table1_sample': pd.DataFrame(), 'table2_sample': pd.DataFrame(),
                    'table1_total_unmatched': 0, 'table2_total_unmatched': 0,
                }

            try:
                edit_candidates = edit_dist_scan.execute(
                    table1, table2, cur_col1, cur_col2, top_k=10)
            except Exception as e:
                logger.warning(f"  EditDistScan failed: {e}")
                edit_candidates = []

            # Structural hints
            structural_hints = _detect_structural_hints(
                obs_result, table1, table2, cur_col1, cur_col2, duckdb_path)

            # Current ratio
            current_ratio = validate_join.execute(table1, table2, cur_col1, cur_col2)
            if current_ratio > best_ratio:
                best_ratio = current_ratio

            # Format observation for LLM
            obs_text = _format_observation(
                obs_result, edit_candidates, table1, table2,
                cur_col1, cur_col2, current_ratio, structural_hints)

            if test_mode:
                logger.info(f"  [TEST MODE] Observation:\n{obs_text[:500]}")
                conversation.append({
                    "iteration": iteration,
                    "observation": obs_text,
                    "test_mode": True,
                })
                continue

            # ── Phase 2: Reason (LLM call) ──
            messages = [
                {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": obs_text},
            ]

            try:
                record: LLMCallRecord = call_llm(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                )
                total_llm_latency_ms += record.latency_ms
                total_llm_tokens += record.total_tokens
                total_llm_cost += record.total_cost

                llm_response = record.response_text
                parsed = _parse_llm_json(llm_response)

            except Exception as e:
                logger.error(f"  LLM call failed: {e}")
                conversation.append({
                    "iteration": iteration,
                    "observation": obs_text,
                    "error": str(e),
                })
                break

            conversation.append({
                "iteration": iteration,
                "observation": obs_text[:2000],  # truncate for storage
                "llm_response": llm_response,
                "parsed": parsed,
                "exact_ratio": current_ratio,
                "llm_tokens": record.total_tokens,
                "llm_latency_ms": record.latency_ms,
                "llm_cost": record.total_cost,
            })

            # ── Check for commit ──
            if parsed.get("commit", False):
                logger.info(f"  LLM committed. Final ratio: {best_ratio:.4f}")
                break

            # ── Phase 3: Act (execute operator) ──
            rec = parsed.get("recommendation")
            if not rec:
                logger.warning(f"  No recommendation from LLM, stopping")
                break

            op_name = rec.get("operator", "")
            op_params = rec.get("parameters", {})
            target_table = rec.get("table", "table1")
            target_column = rec.get("column", cur_col1)

            if op_name not in OPERATOR_REGISTRY:
                logger.warning(f"  Unknown operator: {op_name}, skipping")
                conversation[-1]["action_error"] = f"Unknown operator: {op_name}"
                continue

            # Resolve target table name
            if target_table == "table1":
                actual_tables = [table1]
                actual_cols = [cur_col1]
            elif target_table == "table2":
                actual_tables = [table2]
                actual_cols = [cur_col2]
            elif target_table == "both":
                actual_tables = [table1, table2]
                actual_cols = [cur_col1, cur_col2]
            else:
                actual_tables = [table1]
                actual_cols = [cur_col1]

            op_success = False
            for at, ac in zip(actual_tables, actual_cols):
                try:
                    op_cls = OPERATOR_REGISTRY[op_name]
                    op_instance = op_cls(duckdb_path)

                    # Resolve column: use the actual join column if target_column
                    # matches the original join column name
                    resolved_col = target_column
                    if target_column in (join_col1, cur_col1):
                        resolved_col = ac
                    elif target_column in (join_col2, cur_col2):
                        resolved_col = ac

                    # Execute operator
                    if op_name in ("Pivot", "UnPivot"):
                        op_instance.execute(at, **op_params)
                    elif op_name == "FuzzyMap":
                        op_instance.execute(at, resolved_col, mapping=op_params.get("mapping", {}))
                    elif op_name == "SepSplit":
                        op_instance.execute(at, resolved_col, **op_params)
                        # After SepSplit, update current column name
                        sep = op_params.get("separator", "-")
                        keep_part = op_params.get("keep_part", 0)
                        split_from = op_params.get("split_from", "right")
                        if split_from == 'right':
                            name_parts = resolved_col.rsplit(sep, 1)
                        else:
                            name_parts = resolved_col.split(sep, 1)
                        if len(name_parts) == 2:
                            if isinstance(keep_part, str):
                                keep_part = 0 if keep_part.lower() in ('left', 'first', '0') else 1
                            new_col = name_parts[keep_part]
                            if at == table1:
                                cur_col1 = new_col
                            else:
                                cur_col2 = new_col
                            logger.info(f"  SepSplit: {resolved_col} → {new_col}")
                    else:
                        op_instance.execute(at, resolved_col, **op_params)

                    op_success = True
                    logger.info(f"  Applied {op_name} on {at}.{resolved_col}")

                except Exception as e:
                    logger.error(f"  Operator {op_name} failed on {at}.{resolved_col}: {e}")
                    conversation[-1]["action_error"] = str(e)

            if op_success:
                operations_applied.append({
                    "iteration": iteration,
                    "operator": op_name,
                    "table": target_table,
                    "column": target_column,
                    "parameters": op_params,
                })

            # ── Phase 4: Evaluate ──
            new_ratio = validate_join.execute(table1, table2, cur_col1, cur_col2)
            improvement = new_ratio - current_ratio
            if new_ratio > best_ratio:
                best_ratio = new_ratio
            logger.info(f"  ExactRatio: {current_ratio:.4f} → {new_ratio:.4f} "
                        f"(Δ={improvement:+.4f}, best={best_ratio:.4f})")

            conversation[-1]["new_exact_ratio"] = new_ratio
            conversation[-1]["improvement"] = improvement

            if new_ratio >= 1.0:
                logger.info(f"  Perfect match achieved!")
                break

        # ── Save conversation log ──
        conv_path = os.path.join(output_dir, f"conversation_{pair_idx}.json")
        conv_data = {
            "table1": table1,
            "table2": table2,
            "join_col1": join_col1,
            "join_col2": join_col2,
            "initial_exact_ratio": initial_ratio,
            "final_exact_ratio": best_ratio,
            "iterations": len(conversation),
            "operations_applied": operations_applied,
            "total_latency_ms": total_llm_latency_ms,
            "total_tokens": total_llm_tokens,
            "total_cost": total_llm_cost,
            "conversation": conversation,
        }
        os.makedirs(output_dir, exist_ok=True)
        with open(conv_path, 'w', encoding='utf-8') as f:
            json.dump(conv_data, f, indent=2, ensure_ascii=False, default=str)

        return {
            "final_exact_ratio": best_ratio,
            "initial_exact_ratio": initial_ratio,
            "iterations": len(conversation),
            "operations_applied": operations_applied,
            "total_llm_latency_ms": total_llm_latency_ms,
            "total_llm_tokens": total_llm_tokens,
            "total_llm_cost": total_llm_cost,
            "committed": parsed.get("commit", False) if 'parsed' in dir() else False,
            "conversation": conversation,
        }

    def discover_join_columns(
        self,
        duckdb_path: str,
        table1: str,
        table2: str,
        output_dir: str = ".",
    ) -> Dict[str, Any]:
        """Discover join column pairs for an edge without pre-identified join columns.

        Uses the Layer 2 prompt (Step 1) to analyze table schemas and find
        semantically valid join column pairs.

        Args:
            duckdb_path: Path to the DuckDB file.
            table1, table2: Table names.
            output_dir: Directory to save discovery results.

        Returns:
            Dict with keys: selected_pairs (list of [col1, col2]),
            same_scope (bool), reason (str), llm_cost, llm_tokens, llm_latency_ms.
        """
        import duckdb as _duckdb

        # Load the discovery prompt template
        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "prompt.txt"
        )
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()

        # Get table schemas and sample data from DuckDB
        with _duckdb.connect(duckdb_path) as con:
            def get_table_info(tname):
                cols = con.execute(f"PRAGMA table_info('{tname}')").fetchdf()
                col_names = cols['name'].tolist()
                # Get 5 sample rows
                sample = con.execute(f'SELECT * FROM "{tname}" LIMIT 5').fetchdf()
                # Format as text
                header = f"Table: {tname}\nColumns ({len(col_names)}): {', '.join(col_names)}\n"
                header += f"Sample rows (5):\n{sample.to_string(index=False, max_colwidth=40)}"
                return header, col_names

            t1_info, t1_cols = get_table_info(table1)
            t2_info, t2_cols = get_table_info(table2)

        # Build candidate list (all column pairs with same name)
        candidate_pairs = []
        for c1 in t1_cols:
            for c2 in t2_cols:
                if c1.lower() == c2.lower():
                    candidate_pairs.append(f"{table1}.{c1} <-> {table2}.{c2}")

        candidate_text = "Candidate column pairs (columns with matching names):\n"
        if candidate_pairs:
            candidate_text += "\n".join(f"  - {p}" for p in candidate_pairs)
        else:
            candidate_text += "  (No columns with matching names found. Please analyze schemas to discover join opportunities.)"

        # Fill prompt template
        prompt = prompt_template.replace("{TABLE_NAMES}", f"Table 1: {table1}\nTable 2: {table2}")
        prompt = prompt_template.replace("{TABLE_NAMES}", f"Table 1: {table1}\nTable 2: {table2}")
        prompt = prompt.replace("{TABLE1}", t1_info)
        prompt = prompt.replace("{TABLE2}", t2_info)
        prompt = prompt.replace("{CANDIDATE_LIST}", candidate_text)

        # Call LLM
        messages = [{"role": "user", "content": prompt}]

        try:
            record: LLMCallRecord = call_llm(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
            )

            parsed = _parse_llm_json(record.response_text)

            # Parse selected_pairs into [col1, col2] format
            selected_pairs = []
            for pair_str in parsed.get('selected_pairs', []):
                # Format: "Table1.ColA <-> Table2.ColB"
                parts = pair_str.split(' <-> ')
                if len(parts) == 2:
                    col1 = parts[0].split('.')[-1].strip()
                    col2 = parts[1].split('.')[-1].strip()
                    selected_pairs.append([col1, col2])

            result = {
                'selected_pairs': selected_pairs,
                'same_scope': parsed.get('same_scope', True),
                'reason': parsed.get('reason', ''),
                'llm_cost': record.total_cost,
                'llm_tokens': record.total_tokens,
                'llm_latency_ms': record.latency_ms,
                'raw_response': record.response_text,
            }

        except Exception as e:
            logger.error(f"  Join column discovery failed: {e}")
            result = {
                'selected_pairs': [],
                'same_scope': True,
                'reason': f'Error: {e}',
                'llm_cost': 0.0,
                'llm_tokens': 0,
                'llm_latency_ms': 0.0,
                'error': str(e),
            }

        # Save discovery result
        os.makedirs(output_dir, exist_ok=True)
        disc_path = os.path.join(output_dir, "join_column_discovery.json")
        with open(disc_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"  Discovered {len(result['selected_pairs'])} join column pairs "
                     f"(same_scope={result['same_scope']})")

        return result

    def replay_operations(self, duckdb_path: str, edge_result: dict):
        """Replay previously recorded operations on a fresh DuckDB.

        Used when resuming from cached results to bring the DuckDB
        to the same state as after the original run.
        """
        for pair_result in edge_result.get("results", []):
            if "error" in pair_result:
                continue
            for op in pair_result.get("operations_applied", []):
                op_name = op.get("operator", "")
                if op_name not in OPERATOR_REGISTRY:
                    continue
                op_params = op.get("parameters", {})
                target_table = op.get("table", "table1")
                target_column = op.get("column", "")

                source = edge_result.get("source", "")
                target = edge_result.get("target", "")

                if target_table == "table1":
                    tables = [source]
                elif target_table == "table2":
                    tables = [target]
                elif target_table == "both":
                    tables = [source, target]
                else:
                    tables = [source]

                for t in tables:
                    try:
                        op_cls = OPERATOR_REGISTRY[op_name]
                        op_instance = op_cls(duckdb_path)
                        if op_name in ("Pivot", "UnPivot"):
                            op_instance.execute(t, **op_params)
                        elif op_name == "FuzzyMap":
                            op_instance.execute(t, target_column,
                                                mapping=op_params.get("mapping", {}))
                        else:
                            op_instance.execute(t, target_column, **op_params)
                    except Exception as e:
                        logger.warning(f"Replay {op_name} on {t}.{target_column} failed: {e}")
