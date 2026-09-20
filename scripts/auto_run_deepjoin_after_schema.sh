#!/bin/bash
# Auto pipeline: Wait for Schema Linking to finish, then run DeepJoin prediction
# Usage: nohup bash scripts/auto_run_deepjoin_after_schema.sh > outputs/pipeline_results/auto_pipeline.log 2>&1 &

PROJECT_ROOT="/data/workspace/EcoTableJ"
SCHEMA_LOG="$PROJECT_ROOT/outputs/pipeline_results/schema_linking_inference.log"
SCHEMA_OUTPUT="$PROJECT_ROOT/outputs/pipeline_results/step1_schema_linking_top30.json"
DEEPJOIN_OUTPUT="$PROJECT_ROOT/outputs/pipeline_results/step4_deepjoin_top6.json"

echo "[$(date)] Auto pipeline started. Waiting for Schema Linking to finish..."

while true; do
    if ! pgrep -f "run_schema_linking_inference" > /dev/null 2>&1; then
        echo "[$(date)] Schema Linking process ended."
        
        if [ -f "$SCHEMA_OUTPUT" ]; then
            SIZE=$(du -h "$SCHEMA_OUTPUT" | cut -f1)
            echo "[$(date)] Schema Linking output found: $SCHEMA_OUTPUT ($SIZE)"
            echo "[$(date)] Starting DeepJoin prediction..."
            
            cd "$PROJECT_ROOT"
            python3 -u scripts/run_deepjoin_predict.py
            
            if [ -f "$DEEPJOIN_OUTPUT" ]; then
                SIZE=$(du -h "$DEEPJOIN_OUTPUT" | cut -f1)
                echo "[$(date)] DeepJoin output found: $DEEPJOIN_OUTPUT ($SIZE)"
                echo "[$(date)] Both scripts completed successfully!"
            else
                echo "[$(date)] ERROR: DeepJoin output not found!"
            fi
        else
            echo "[$(date)] ERROR: Schema Linking output not found! Check log:"
            tail -20 "$SCHEMA_LOG"
        fi
        
        break
    fi
    
    sleep 60
done

echo "[$(date)] Auto pipeline finished."
