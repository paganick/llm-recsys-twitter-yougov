#!/usr/bin/env bash
# Full pipeline: Steps 1–5
# Usage: bash run_pipeline.sh [--tweets PATH] [--survey PATH] [--provider PROVIDER]

set -euo pipefail
cd "$(dirname "$0")/.."

TWEETS="${TWEETS:-datasets/tweets.csv}"
SURVEY="${SURVEY:-datasets/survey.csv}"
PROVIDER="${PROVIDER:-anthropic}"

echo "=========================================="
echo " LLM Recommendation Bias — Full Pipeline"
echo "=========================================="
echo " Tweets:   $TWEETS"
echo " Survey:   $SURVEY"
echo " Provider: $PROVIDER"
echo "=========================================="

# Step 1
echo ""
echo "[Step 1] Prepare dataset"
python pipeline/prepare_dataset.py --tweets "$TWEETS" --survey "$SURVEY"

# Step 2
echo ""
echo "[Step 2] Run LLM recommendation experiments"
python pipeline/run_llm_recommendation.py --provider "$PROVIDER"

# Step 3
echo ""
echo "[Step 3] Compute text features"
for exp_dir in outputs/experiments/${PROVIDER}_*; do
    python pipeline/compute_text_features.py --experiment-dir "$exp_dir"
done

# Steps 4–5
echo ""
bash pipeline/run_analysis.sh
