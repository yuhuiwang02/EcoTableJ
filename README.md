# EcoTable: Cost-effective Table Integration in Data Lakes

This repository provides the official implementation of the paper: **"EcoTable: Cost-effective Table Integration in Data Lakes for Natural Language Queries"**.

EcoTable is a natural language-driven framework designed to bridge the gap between raw data lake files (e.g., CSV/Parquet) and analytical requirements by discovering query-specific schemas and performing automated data transformations.

## 🚀 Key Features

* **Query-Driven Integration**: Tailors database schemas to specific natural language queries rather than relying on a static, pre-defined ETL process.
* **Hybrid Architecture**: Combines lightweight deep learning models (e.g., DeBERTa-v3) for efficiency with LLMs for high-precision semantic reasoning.
* **Cost-Aware Optimization**: Minimizes LLM invocation costs through Steiner tree-based join path identification and global result caching.
* **Scalable Pruning**: Features a t-spanner based edge-pruning strategy to handle massive data lakes containing thousands of tables.
* **Parallel Transformation**: Optimizes execution latency via an edge-coloring-based parallelization strategy for independent table joins.

---

## 📁 Repository Structure

### 1. Source Code (`src/`)
* **`table_identification/`**: Two-stage schema linking — Stage 1 PLM-based (DeBERTa-v3) coarse ranking (`schema_linking_DL_crossencoder.py`), Stage 2 LLM-based fine-grained verification (`LLM_Filter_nyc.py`).
* **`graph_validation/`**: DeepJoin column-pair scoring (`DeepJoin.py`), Steiner-tree join path discovery, t-spanner edge pruning, and iterative LLM edge verification (`join_evaluation.py`, `load_file.py`, `subgraph_manager.py`).
* **`table_transformation/`**: ReAct-style transform agent for code generation and execution (`transform_agent/`).

### 2. Datasets (`data/benchmarks/`)
Five core benchmarks featuring real-world industrial data and a large-scale noisy data lake:
* **`ad/`**: Advertisement spend and click-through tracking datasets.
* **`business/`**: Financial metrics and sales operations data.
* **`engagement/`**: User interaction and retention records.
* **`platform/`**: Engineering activity and system metadata.
* **`NYC/`**: **NYC Data Lake**, a massive benchmark with 1,214 tables and 800 queries designed for scalability and robustness testing. Includes `schema_linking_gt_nyc.json` (ground-truth schema linking labels).

### 3. Scripts (`scripts/`)
Standalone inference runners for the heavy stages (schema linking, DeepJoin) plus an auto-chaining helper (`auto_run_deepjoin_after_schema.sh`).

---

## 🛠️ Requirements

* **Environment**: Python 3.8+ (tested on 3.11), CUDA 12.x for GPU inference
* **Backends**:
    * PLMs: `microsoft/deberta-v3-large` (Stage 1 schema linking), `sentence-transformers/all-mpnet-base-v2` (DeepJoin column embeddings)
    * LLMs: GPT-4o via OpenAI-compatible API (used for Stage 2 LLM filter, join verification, and table transformation)
* **Required Libraries**: see `requirements.txt`

```bash
pip install -r requirements.txt
```

> **Note**: Trained model weights (`outputs/stage1_model/`, `outputs/stage2_deepjoin_model/`) and inference artifacts (`outputs/pipeline_results/`) are **not included** in this repository — they are excluded via `.gitignore`. The HuggingFace model cache (`deberta-v3-large`, `all-mpnet-base-v2`) must be available locally or downloaded on first run.

---

## 🔑 API Key Setup

All LLM-based components (Stage 2 filter, join verifier, transform agent) use a single OpenAI-compatible endpoint (`https://www.dmxapi.com/v1`). Set the API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

The default `base_url` is already configured to the proxy in code. To use the official OpenAI endpoint instead, change `DEFAULT_BASE_URL` in `src/table_transformation/transform_agent/agent/models.py` and the `base_url` defaults in `LLM_Filter_nyc.py` / `llm_verifier.py` to `https://api.openai.com/v1`.

---

## ▶️ Running the Pipeline

The full pipeline has 6 stages: (1) Schema Linking → (2) LLM Filter → (3) Table Name Conversion → (4) DeepJoin Prediction → (5) Join Evaluation (LLM verification + t-spanner pruning) → (6) Transform Agent.

### Quick start (full pipeline)

```bash
export OPENAI_API_KEY="your-key"

python3 -m src.pipeline_main \
  --queries_file data/queries.json \
  --schema_model outputs/stage1_model/best_model.pt \
  --test_data_file data/queries.json \
  --csv_dataset_root data/benchmarks/NYC \
  --deepjoin_model_dir outputs/stage2_deepjoin_model \
  --ground_truth_csv data/training/stage2/trainDataForLayer2.csv \
  --tables_dir data/benchmarks/NYC \
  --llm_api_key $OPENAI_API_KEY \
  --output_dir outputs/pipeline_results
```

### Standalone inference scripts

Two helper scripts in `scripts/` run the most compute-intensive stages independently and persist only the compact artifacts needed downstream:

| Script | Input | Output | Notes |
|--------|-------|--------|-------|
| `scripts/run_schema_linking_inference.py` | `schema_linking_gt_nyc.json` (800 queries) | `outputs/pipeline_results/step1_schema_linking_top30.json` | Per-query top-30 candidate tables (compact) |
| `scripts/run_deepjoin_predict.py` | all NYC CSV tables | `outputs/pipeline_results/step4_deepjoin_top6.json` | Per-table-pair top-6 join column pairs |

A pre-built, LLM-verified pruned join graph is available at `data/pruned_graph_join_cols_version.json` (614 KB) and can be reused as the Step 5 output / Step 6 input to skip the costly LLM verification pass.
