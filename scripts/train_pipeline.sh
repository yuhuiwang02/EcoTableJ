#!/bin/bash
# Automated Pipeline: Stage 1 (DeBERTa) -> Stage 2 (DeepJoin)
# Usage: nohup bash scripts/train_pipeline.sh > outputs/pipeline.log 2>&1 &

set -e

PROJECT_ROOT="/data/workspace/EcoTableJ"
STAGE1_LOG="$PROJECT_ROOT/outputs/stage1_model/train.log"
STAGE2_MODEL_DIR="$PROJECT_ROOT/outputs/stage2_deepjoin_model"
STAGE2_LOG="$PROJECT_ROOT/outputs/stage2_deepjoin_model/train.log"
STAGE2_LINKS="$PROJECT_ROOT/data/training/stage2/trainDataForLayer2.csv"
STAGE2_TABLES="$PROJECT_ROOT/data/benchmarks/NYC"

echo "=========================================="
echo "  EcoTableJ Automated Training Pipeline"
echo "=========================================="
echo "  Stage 1: DeBERTa CrossEncoder (monitoring)"
echo "  Stage 2: DeepJoin (auto-launch after Stage 1)"
echo "=========================================="

# ---- Step 1: Wait for Stage 1 to finish ----
echo "[Pipeline] Checking Stage 1 training status..."

while true; do
    STAGE1_PID=$(pgrep -f "schema_linking_DL_crossencoder" || true)
    if [ -z "$STAGE1_PID" ]; then
        echo "[Pipeline] Stage 1 process finished."
        break
    fi
    EPOCH_INFO=$(grep -oP "Epoch \d+/\d+" "$STAGE1_LOG" 2>/dev/null | tail -1 || echo "unknown")
    echo "[Pipeline] Stage 1 running (PID: $STAGE1_PID) | $EPOCH_INFO | $(date '+%H:%M:%S')"
    sleep 60
done

# Verify Stage 1 completed
if [ -f "$PROJECT_ROOT/outputs/stage1_model/best_model.pt" ]; then
    echo "[Pipeline] Stage 1 completed successfully!"
    grep -E "Train Loss|Val Loss" "$STAGE1_LOG" 2>/dev/null | tail -6 || true
else
    echo "[Pipeline] ERROR: Stage 1 failed (no best_model.pt)"
    exit 1
fi

# ---- Step 2: Launch Stage 2 DeepJoin Training ----
echo ""
echo "=========================================="
echo "[Pipeline] Starting Stage 2: DeepJoin Training"
echo "  Tables:    $STAGE2_TABLES"
echo "  Links:     $STAGE2_LINKS"
echo "  Model dir: $STAGE2_MODEL_DIR"
echo "=========================================="

mkdir -p "$STAGE2_MODEL_DIR"

cd "$PROJECT_ROOT"
python3 -m src.graph_validation.DeepJoin \
    --mode train \
    --tables "$STAGE2_TABLES" \
    --links "$STAGE2_LINKS" \
    --model_dir "$STAGE2_MODEL_DIR" \
    --isfolder \
    2>&1 | tee "$STAGE2_LOG"

# ---- Step 3: Verify ----
if [ -d "$STAGE2_MODEL_DIR/sentence_transformer" ]; then
    echo "[Pipeline] Stage 2 completed successfully!"
    ls -lh "$STAGE2_MODEL_DIR/"
else
    echo "[Pipeline] ERROR: Stage 2 failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "  All training complete!"
echo "  Stage 1: outputs/stage1_model/best_model.pt"
echo "  Stage 2: outputs/stage2_deepjoin_model/"
echo "=========================================="
