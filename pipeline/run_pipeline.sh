#!/usr/bin/env bash
# Full pipeline: Steps 1–5
#
# Usage:
#   TWEETS=datasets/tweets.csv SURVEY=datasets/survey.csv PROVIDER=anthropic \
#       bash pipeline/run_pipeline.sh

set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v apptainer &>/dev/null; then
    module load apptainer 2>/dev/null \
        || { echo "ERROR: apptainer not found. Run: module load apptainer"; exit 1; }
fi

TWEETS="${TWEETS:-datasets/tweets.csv}"
SURVEY="${SURVEY:-datasets/survey.csv}"
PROVIDER="${PROVIDER:-anthropic}"
SIF="${SIF:-llm-recsys-twitter.sif}"
APT="apptainer exec $SIF python"

echo "=========================================="
echo " LLM Recommendation Bias — Full Pipeline"
echo "=========================================="
echo " Container: $SIF"
echo " Tweets:    $TWEETS"
echo " Survey:    $SURVEY"
echo " Provider:  $PROVIDER"
echo "=========================================="

echo ""
echo "[Step 1] prepare_dataset.py"
$APT pipeline/prepare_dataset.py --tweets "$TWEETS" --survey "$SURVEY"

echo ""
echo "[Step 2] run_llm_recommendation.py"
$APT pipeline/run_llm_recommendation.py --provider "$PROVIDER"

echo ""
echo "[Step 3] compute_text_features.py"
for exp_dir in outputs/experiments/${PROVIDER}_*; do
    $APT pipeline/compute_text_features.py --experiment-dir "$exp_dir"
done

echo ""
SIF=$SIF bash pipeline/run_analysis.sh
