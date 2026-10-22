"""
Graph Validation Package

Provides two verification methods for join edge validation:

1. **Per-edge verification** (`per_edge/`):
   - Phase 1 (schema-level): LLM identifies candidate join column pairs
   - Phase 2 (data-level): LLM validates each column pair with actual data
   - Critical edges get 3 independent rounds with majority voting

2. **Cluster-based verification** (`cluster/`):
   - Paper Algorithm 1: Edge Clustering Strategy
   - Critical edges (d=3) distributed across 3 different clusters
   - Each cluster verified once; cross-cluster majority voting
   - Concurrent cluster verification for efficiency

Shared utilities live in `common/`:
   - load_file: Graph loading, path finding, pruning
   - tree_process: Steiner tree 2-approximation
   - DeepJoin: Column-pair scoring model

Evaluation utilities live in `evaluation/`.

The main orchestrator is `join_evaluation.py` (Steiner tree + dispatch).
The iterative cluster pipeline is in `scripts/run_iterative_validation.py`.
"""

# Lazy imports are used to avoid circular dependencies and slow startup.
# External code can still use:
#   from src.graph_validation.load_file import extract_table_name
#   from src.graph_validation.edge_cache import EdgeCache
# because the original files remain as shims (they are also copied into subpackages).
