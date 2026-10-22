# EcoTable: Cost-effective Table Integration in Data Lakes

This repository contains the implementation and reproducibility package for **EcoTable**, a cost-effective table integration framework for natural language queries over data lakes.

The public repository is organized to include only the components needed to reproduce the main experiments: source code, configuration files, benchmark data, and dependency specifications. Intermediate outputs, logs, figures, model checkpoints, and paper-specific analysis artifacts are intentionally excluded.

## Repository Structure

```text
configs/        Default experiment configuration
data/           Benchmark and training data used by the reproducibility workflow
src/            EcoTable source code
requirements.txt
README.md
```

## Environment Setup

EcoTable requires Python 3.8+ and a CUDA-enabled GPU for the neural inference stages.

Install dependencies:

```bash
pip install -r requirements.txt
```

Set the LLM API key before running LLM-based stages:

```bash
export OPENAI_API_KEY="your-api-key"
```

The default OpenAI-compatible endpoint and model settings are configured in `configs/default.json`. Update the configuration file if you need to use a different provider or model.

## Main Experiment Reproduction

The unified entry point is `src.pipeline_main`. It runs the main EcoTable workflow with configurable schema linking, table filtering, graph validation, and optional table transformation stages.

### EcoTable+ Main Setting

```bash
python -m src.pipeline_main \
  --config configs/default.json \
  --schema_cache outputs/pipeline_results/step1_schema_linking_top30.json \
  --deepjoin_cache outputs/pipeline_results/step4_deepjoin_top6.json \
  --tables_dir data/benchmarks/NYC \
  --ground_truth_csv data/training/stage2/trainDataForLayer2.csv \
  --gt_file data/benchmarks/NYC/schema_linking_gt_nyc.json \
  --output_dir outputs/pipeline_results \
  --llm_api_key "$OPENAI_API_KEY"
```

This command uses the default EcoTable+ graph validation setting from `configs/default.json`, including cluster-based edge verification and graph pruning.

### EcoTable Original-Style Baseline

```bash
python -m src.pipeline_main \
  --config configs/default.json \
  --verification_method per_edge \
  --no_pruning \
  --schema_cache outputs/pipeline_results/step1_schema_linking_top30.json \
  --deepjoin_cache outputs/pipeline_results/step4_deepjoin_top6.json \
  --tables_dir data/benchmarks/NYC \
  --ground_truth_csv data/training/stage2/trainDataForLayer2.csv \
  --gt_file data/benchmarks/NYC/schema_linking_gt_nyc.json \
  --output_dir outputs/pipeline_results_baseline \
  --llm_api_key "$OPENAI_API_KEY"
```

### Cluster Size Ablation

Use `--cluster_size` to reproduce the cluster-size sensitivity experiment:

```bash
python -m src.pipeline_main \
  --config configs/default.json \
  --cluster_size 5 \
  --schema_cache outputs/pipeline_results/step1_schema_linking_top30.json \
  --deepjoin_cache outputs/pipeline_results/step4_deepjoin_top6.json \
  --tables_dir data/benchmarks/NYC \
  --ground_truth_csv data/training/stage2/trainDataForLayer2.csv \
  --gt_file data/benchmarks/NYC/schema_linking_gt_nyc.json \
  --output_dir outputs/pipeline_results_r5 \
  --llm_api_key "$OPENAI_API_KEY"
```

Repeat the command with different `--cluster_size` values as needed.

### Optional Table Transformation Stage

To enable the final transformation agent, add `--enable_transform_agent`:

```bash
python -m src.pipeline_main \
  --config configs/default.json \
  --enable_transform_agent \
  --schema_cache outputs/pipeline_results/step1_schema_linking_top30.json \
  --deepjoin_cache outputs/pipeline_results/step4_deepjoin_top6.json \
  --tables_dir data/benchmarks/NYC \
  --ground_truth_csv data/training/stage2/trainDataForLayer2.csv \
  --gt_file data/benchmarks/NYC/schema_linking_gt_nyc.json \
  --output_dir outputs/pipeline_results_transform \
  --llm_api_key "$OPENAI_API_KEY"
```

## Reproducibility Notes

- **Model checkpoints** are not included. Place trained checkpoints under the paths expected by your command or update `configs/default.json` accordingly.
- **Intermediate artifacts** such as cached predictions, validation outputs, logs, and figures are generated locally under `outputs/` and are not tracked by Git.
- **Experiment results and cost calculations** are not included in the repository. Re-run the commands above to regenerate outputs in your local environment.
- **API-dependent stages** may vary slightly across LLM providers or model versions.
