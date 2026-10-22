"""
Paper Algorithm 1: Edge Clustering Strategy.

Groups candidate edges into clusters of size ≤ r, ensuring that critical edges
(demand d=3) appear in exactly 3 different clusters and ordinary edges (d=1)
appear in exactly 1 cluster.
"""

from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple

from .io_utils import Edge, canon_edge


def paper_clustering(e_new: List[Tuple[str, str, int]], r: int) -> List[dict]:
    """Cluster edges according to Algorithm 1 (degree-greedy).

    Args:
        e_new: List of (node_u, node_v, demand) tuples.
               demand=3 for critical edges, 1 for ordinary.
        r: Maximum cluster size (number of tables per cluster).

    Returns:
        List of cluster dicts, each with 'tables' and 'edges' keys.
    """
    demand: Dict[Edge, int] = {canon_edge(u, v): d for u, v, d in e_new}
    clusters: List[dict] = []

    while True:
        active = {e: d for e, d in demand.items() if d > 0}
        if not active:
            break

        # Step 1: initialize — pick edge with highest combined degree
        deg: Dict[str, int] = defaultdict(int)
        for (a, b) in active:
            deg[a] += 1
            deg[b] += 1
        best_edge = max(active, key=lambda e: (deg[e[0]] + deg[e[1]], e))
        cluster_tables = set(best_edge)

        # Step 2: grow the cluster up to size r
        while len(cluster_tables) < r:
            cg: Dict[str, int] = defaultdict(int)
            for (a, b), d in demand.items():
                if d <= 0:
                    continue
                if a in cluster_tables and b not in cluster_tables:
                    cg[b] += d
                elif b in cluster_tables and a not in cluster_tables:
                    cg[a] += d
            if not cg:
                break
            best_c = max(sorted(cg), key=lambda c: cg[c])
            if cg[best_c] <= 0:
                break
            cluster_tables.add(best_c)

        # Step 3: update remaining demand
        covered = []
        for a, b in combinations(sorted(cluster_tables), 2):
            e = canon_edge(a, b)
            if demand.get(e, 0) > 0:
                demand[e] -= 1
                covered.append(e)

        clusters.append({'tables': sorted(cluster_tables), 'edges': sorted(covered)})

    return clusters
