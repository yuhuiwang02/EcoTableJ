"""
Per-edge verification method.

This module implements the per-edge LLM verification pipeline:
  - Phase 1 (schema-level): LLM identifies candidate join column pairs
  - Phase 2 (data-level): LLM validates each column pair with actual data values
  - Critical edges get 3 independent verification rounds with majority voting

Modules:
    llm_verifier        - LLM API wrapper for Phase1/Phase2 calls
    edge_validator      - Per-edge validation orchestrator with critical-edge voting
    verification_utils  - File naming and storage helpers
    edge_cache          - Legacy per-edge cache (flat file layout)
    subgraph_manager    - Subgraph construction and caching
"""

from .llm_verifier import LLMVerifier
from .edge_validator import EdgeValidator
from .verification_utils import (
    get_edge_filename,
    get_verification_filepath,
    check_verification_exists,
    load_verification_result,
    save_verification_result,
    save_verification_visualization,
)
from .edge_cache import EdgeCache
