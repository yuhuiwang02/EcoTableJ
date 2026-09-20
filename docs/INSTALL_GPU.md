# EcoTable GPU 环境安装文档

适用环境（已在用户服务器确认）：
- GPU: Tesla T4 (16GB)
- Driver: 535.161.07  →  CUDA 上限 12.2
- nvcc: 12.2.140 (toolkit)
- Python: 3.11 / pip 23.3.1

> 为什么用 cu121 而不是 cu122？PyTorch 官方预编译 wheel 没有 cu122，
> cu121 是 ≤12.2 驱动下可用的最高 CUDA-build，T4 + CUDA 12.2 完全兼容。

---

## 步骤 1：建独立环境（推荐，可选但建议）

```bash
conda create -n ecotable python=3.11 -y
conda activate ecotable
# 若不用 conda，也可 python -m venv venv && source venv/bin/activate
```

## 步骤 2：安装 CUDA 版 torch（关键，不能从 PyPI 直接装）

```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
```

验证 GPU 可见：
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望输出类似: 2.2.2+cu121 True Tesla T4
```

## 步骤 3：安装其余依赖

```bash
cd EcoTable
pip install -r requirements.txt
```

## 步骤 4：确认全部 import 通过

```bash
cd EcoTable/src
PYTHONPATH=$(pwd)/.. python -c "import utils.join_evaluation, utils.llm_verifier, utils.edge_validator, utils.DeepJoin; print('IMPORT_OK')"
```

## 步骤 5：模型权重

- **DeepJoin** 用 `sentence-transformers/all-mpnet-base-v2`（HF 可下，无需微调）。
  仓库 `src/utils/DeepJoin.py` 里把本地路径改成 `'all-mpnet-base-v2'` 或你的本地缓存路径。
- **第一层 cross-encoder** `microsoft/deberta-v3-large` 需微调权重（仓库未给）。
  最短跑通路径：用数据集的 `*_schema_linking_gt.json` 作为第一层理想输出，
  跳过 DeBERTa，直接喂给第二层（见 README / pipeline 适配器）。
- **NYC 数据集** 1214 张表，建议最后跑；先小数据集验证链路。

## 步骤 6：API Key（不入库，走环境变量）

```bash
export OPENAI_API_KEY="sk-..."        # 官方 OpenAI
# llm_verifier.py 默认 base_url 需改回官方: https://api.openai.com/v1
# 真实调用前先 test_mode=True 验证链路（只生成 prompt 不调 API）。
```

## 已知坑修复清单（已在仓库修复，供核对）

1. `edge_validator.py` 的 `identify_critical_edges` 原 `from testTrainQuality_new import ...`
   → 改为从 `utils.load_file` 导入（`find_k_shortest_paths`, `extract_table_name`），
   缺的 `path_to_edges` 用本地小函数补。
2. 导入路径统一：把 `src` 加入 `PYTHONPATH` 并统一用绝对导入 `from utils...`。
3. `DeepJoin.py` 硬编码本地路径 → 改 `'all-mpnet-base-v2'`；
   `schema_linking_DL_crossencoder.py` 的 `model_path=''` → 填 `'microsoft/deberta-v3-large'`。
4. `llm_verifier.py` base_url 改回官方 `https://api.openai.com/v1`，model `gpt-4o`。

## 验证命令（test_mode 不消耗 token）

```bash
cd EcoTable/src
PYTHONPATH=$(pwd)/.. python -m src.pipeline_main --test_mode
# 或单独跑第二层：
PYTHONPATH=$(pwd)/.. python -c "from utils.join_evaluation import run_join_evaluation; ..."
```
