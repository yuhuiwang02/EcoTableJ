"""
Edge-level LLM verification cache.

Directory layout:
    {cache_dir}/
    ├── per_edge/
    │   ├── phase1/
    │   │   └── {t1}--{t2}.json          # Phase-1 result for one edge
    │   └── phase2/
    │       └── {t1}--{t2}__round{i}.json # Phase-2 result (one round)
    ├── cluster/
    │   └── cluster_{hash}.json           # Cluster Step-1 result
    └── index.json                        # Global index for quick lookup

Each cache file stores:
    - timestamp, model, prompt (first 500 chars), raw_response
    - parsed result (same_scope, selected_pairs, can_join, etc.)
    - tokens, input_tokens, output_tokens, latency
"""

import hashlib
import json
import os
import time
from typing import Dict, List, Optional, Tuple


def _edge_key(t1: str, t2: str) -> str:
    """Canonical edge key: sorted, joined by '--'. Safe for filenames."""
    a, b = sorted([t1, t2])
    # Replace path separators and other unsafe chars with _
    a_safe = a.replace('/', '_').replace('\\', '_')
    b_safe = b.replace('/', '_').replace('\\', '_')
    key = f"{a_safe}--{b_safe}"
    # Truncate and hash if too long
    if len(key) > 200:
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        key = key[:187] + "_" + h
    return key


def _cluster_key(tables: List[str], edges: List[Tuple[str, str]]) -> str:
    """Deterministic cluster key based on sorted tables + edges."""
    canon = json.dumps({
        'tables': sorted(tables),
        'edges': sorted([tuple(sorted(e)) for e in edges])
    }, sort_keys=True)
    return "cluster_" + hashlib.md5(canon.encode()).hexdigest()[:16]


class EdgeCache:
    """Read/write cache for LLM edge verification results."""

    def __init__(self, cache_dir: str, model: str = "gpt-4o"):
        self.cache_dir = os.path.join(cache_dir, model.replace("/", "_"))
        self.model = model
        self._ensure_dirs()
        self.stats = {'hits': 0, 'misses': 0, 'saves': 0}

    def _ensure_dirs(self):
        for sub in ["per_edge/phase1", "per_edge/phase2", "cluster"]:
            os.makedirs(os.path.join(self.cache_dir, sub), exist_ok=True)

    # ── Per-edge Phase-1 ──────────────────────────────────────────────────

    def get_phase1(self, t1: str, t2: str, round_idx: int = 0) -> Optional[Dict]:
        """Load cached Phase-1 result. Returns None on miss."""
        key = _edge_key(t1, t2)
        suffix = f"__round{round_idx}" if round_idx > 0 else ""
        path = os.path.join(self.cache_dir, "per_edge", "phase1", f"{key}{suffix}.json")
        if os.path.exists(path):
            self.stats['hits'] += 1
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        self.stats['misses'] += 1
        return None

    def save_phase1(self, t1: str, t2: str, result: Dict,
                    raw_response: str = "", prompt_summary: str = "",
                    round_idx: int = 0):
        """Save Phase-1 result to cache."""
        key = _edge_key(t1, t2)
        suffix = f"__round{round_idx}" if round_idx > 0 else ""
        path = os.path.join(self.cache_dir, "per_edge", "phase1", f"{key}{suffix}.json")
        data = {
            'edge': [t1, t2],
            'round_idx': round_idx,
            'model': self.model,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'prompt_summary': prompt_summary[:500],
            'raw_response': raw_response,
            'result': result,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.stats['saves'] += 1

    # ── Per-edge Phase-2 ──────────────────────────────────────────────────

    def get_phase2(self, t1: str, t2: str, round_idx: int = 0) -> Optional[Dict]:
        """Load cached Phase-2 result. Returns None on miss."""
        key = _edge_key(t1, t2)
        suffix = f"__round{round_idx}" if round_idx > 0 else ""
        path = os.path.join(self.cache_dir, "per_edge", "phase2", f"{key}{suffix}.json")
        if os.path.exists(path):
            self.stats['hits'] += 1
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        self.stats['misses'] += 1
        return None

    def save_phase2(self, t1: str, t2: str, result: Dict,
                    raw_responses: List[str] = None,
                    round_idx: int = 0):
        """Save Phase-2 result to cache."""
        key = _edge_key(t1, t2)
        suffix = f"__round{round_idx}" if round_idx > 0 else ""
        path = os.path.join(self.cache_dir, "per_edge", "phase2", f"{key}{suffix}.json")
        data = {
            'edge': [t1, t2],
            'round_idx': round_idx,
            'model': self.model,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'raw_responses': raw_responses or [],
            'result': result,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.stats['saves'] += 1

    # ── Cluster Step-1 ────────────────────────────────────────────────────

    def get_cluster(self, tables: List[str],
                    edges: List[Tuple[str, str]]) -> Optional[Dict]:
        """Load cached cluster Step-1 result. Returns None on miss."""
        key = _cluster_key(tables, edges)
        path = os.path.join(self.cache_dir, "cluster", f"{key}.json")
        if os.path.exists(path):
            self.stats['hits'] += 1
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        self.stats['misses'] += 1
        return None

    def save_cluster(self, tables: List[str],
                     edges: List[Tuple[str, str]],
                     result: Dict,
                     raw_response: str = "",
                     prompt_summary: str = ""):
        """Save cluster Step-1 result to cache."""
        key = _cluster_key(tables, edges)
        path = os.path.join(self.cache_dir, "cluster", f"{key}.json")
        data = {
            'tables': sorted(tables),
            'edges': sorted([list(e) for e in edges]),
            'model': self.model,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'prompt_summary': prompt_summary[:500],
            'raw_response': raw_response,
            'result': result,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.stats['saves'] += 1

    # ── Utilities ─────────────────────────────────────────────────────────

    def print_stats(self):
        print(f"[Cache] hits={self.stats['hits']}  misses={self.stats['misses']}  "
              f"saves={self.stats['saves']}  dir={self.cache_dir}")

    def count_cached_edges(self) -> int:
        """Count how many Phase-1 results are cached."""
        p1_dir = os.path.join(self.cache_dir, "per_edge", "phase1")
        if not os.path.isdir(p1_dir):
            return 0
        return len([f for f in os.listdir(p1_dir) if f.endswith('.json')])

    def count_cached_clusters(self) -> int:
        """Count how many cluster results are cached."""
        c_dir = os.path.join(self.cache_dir, "cluster")
        if not os.path.isdir(c_dir):
            return 0
        return len([f for f in os.listdir(c_dir) if f.endswith('.json')])
