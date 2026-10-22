"""
Cluster-based verification method.

This package implements the cluster-based LLM verification pipeline
(Paper Algorithm 1 — Edge Clustering Strategy):
  - Critical edges (d=3) are distributed across 3 different clusters
  - Ordinary edges (d=1) appear in 1 cluster
  - Each cluster is verified once by LLM (Step1 cluster prompt + Phase2 data validation)
  - Cross-cluster majority voting determines final edge decisions

Modules:
    io_utils          - Canonical edge, safe naming, graph/table loading
    clustering        - Algorithm 1 (degree-greedy edge clustering)
    llm_helpers       - LLM API wrapper, serialization, Phase2 validation
    cluster_verifier  - Single-cluster verification + cross-cluster voting
    iterative_runner  - Full iterative pipeline with resume support

Output structure (per r value):
  outputs/edge_cluster_validation/r{R}/
  ├── iteration_1/
  │   ├── clusters/cluster_1.json ... cluster_N.json
  │   ├── per_edge/edge_001/ ... edge_NNN/
  │   │   ├── edge_info.json
  │   │   ├── validation_1.json ... validation_K.json
  │   └── iteration_summary.json
  └── final_summary.json

Usage from script:
    from src.graph_validation.cluster.iterative_runner import run_iterative_validation
    from src.graph_validation.cluster.io_utils import load_pruned_graph
"""

from .io_utils import canon_edge, load_pruned_graph, load_table_dfs
from .clustering import paper_clustering
from .cluster_verifier import verify_cluster, cross_cluster_vote
from .iterative_runner import (
    run_iterative_validation,
    build_query_paths,
    collect_e_new,
)
