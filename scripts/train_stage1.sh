#!/bin/bash
# Stage 1: DeBERTa CrossEncoder Training Script
# Usage: bash scripts/train_stage1.sh

set -e

cd /data/workspace/EcoTableJ

echo "=========================================="
echo "Stage 1: DeBERTa-v3-large CrossEncoder Training (3 epochs)"
echo "=========================================="
echo "Train data: data/training/stage1/trainDataForStage1.json"
echo "Test data:  data/queries.json"
echo "CSV root:   data/benchmarks/NYC"
echo "Output:     outputs/stage1_model/"
echo "=========================================="

mkdir -p outputs/stage1_model

# Launch training
nohup python3 -m src.table_identification.schema_linking_DL_crossencoder \
    > outputs/stage1_model/train.log 2>&1 &

TRAIN_PID=$!
echo "Training started with PID: $TRAIN_PID"
echo "Monitor with: tail -f outputs/stage1_model/train.log"
echo "Check GPU:    nvidia-smi"
