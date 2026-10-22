"""Map NYC benchmark short table names to full graph node names."""

from __future__ import annotations

import os
from typing import Iterable, List, Optional


_NYC_CATEGORY_SUFFIXES = ("_no_join_cols", "_no_overlap", "_normal")


class GTNameMapper:
    def __init__(self, nyc_root: str):
        self.nyc_root = nyc_root
        self.basename_to_cat: dict[str, str] = {}
        self.basename_to_subtables: dict[str, set[str]] = {}
        self._load()
        self.basenames_sorted: list[str] = sorted(
            self.basename_to_cat.keys(), key=len, reverse=True
        )

    def _load(self) -> None:
        for d in os.listdir(self.nyc_root):
            dpath = os.path.join(self.nyc_root, d)
            if not os.path.isdir(dpath):
                continue
            cat = None
            base = None
            for suffix in _NYC_CATEGORY_SUFFIXES:
                if d.endswith(suffix):
                    cat = suffix[1:]
                    base = d[: -len(suffix)]
                    break
            if cat is None or base is None:
                continue
            self.basename_to_cat[base] = cat
            self.basename_to_subtables[base] = {
                f[:-4] for f in os.listdir(dpath) if f.endswith(".csv")
            }

    def extract_hint_basename(self, query_tables: Iterable[str]) -> Optional[str]:
        for table in query_tables:
            if not table or table.startswith("node_"):
                continue
            for basename in self.basenames_sorted:
                if table == basename or table.startswith(basename + "_"):
                    return basename
        return None

    def map_one(self, short: str, hint_basename: Optional[str] = None) -> Optional[str]:
        if not short:
            return None
        if short.startswith("node_") and hint_basename is not None:
            cat = self.basename_to_cat.get(hint_basename)
            if cat is not None:
                return f"{hint_basename}_{cat}/{short}"
            return None
        for basename in self.basenames_sorted:
            if short == basename:
                return None
            if short.startswith(basename + "_"):
                tail = short[len(basename) + 1 :]
                cat = self.basename_to_cat[basename]
                subtable = tail if tail.startswith("node_") else f"{basename}_{tail}"
                return f"{basename}_{cat}/{subtable}"
        return None

    def map_query_tables(self, short_list: Iterable[str]) -> tuple[List[str], List[str]]:
        short_list = list(short_list)
        hint = self.extract_hint_basename(short_list)
        mapped: list[str] = []
        unresolved: list[str] = []
        for table in short_list:
            full = self.map_one(table, hint_basename=hint)
            if full is None:
                unresolved.append(table)
            else:
                mapped.append(full)
        return mapped, unresolved
