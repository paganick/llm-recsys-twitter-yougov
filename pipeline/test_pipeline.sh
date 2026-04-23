#!/usr/bin/env bash
# End-to-end pipeline test using example data and fake LLM/NLP calls.
# No API keys or GPU required.
#
# Requires the Apptainer container. Before running, either:
#   (a) Copy from the other repo (packages are identical, fast):
#       cp ../llm-recsys-clean/llm-recsys.sif llm-recsys-twitter.sif
#   (b) Build from scratch (~20-30 min):
#       apptainer build llm-recsys-twitter.sif llm-recsys-twitter.def
#
# Usage:
#   bash pipeline/test_pipeline.sh

set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v apptainer &>/dev/null; then
    module load apptainer 2>/dev/null \
        || { echo "ERROR: apptainer not found. Run: module load apptainer"; exit 1; }
fi

SIF="llm-recsys-twitter.sif"
APT="apptainer exec $SIF python"

if [ ! -f "$SIF" ]; then
    echo "ERROR: $SIF not found."
    echo "  Copy: cp ../llm-recsys-clean/llm-recsys.sif llm-recsys-twitter.sif"
    echo "  Or build: apptainer build $SIF llm-recsys-twitter.def"
    exit 1
fi

SEED=$(python3 -c "import random; print(random.randint(1, 99999))")

echo "======================================================"
echo "  LLM Recommendation Bias — pipeline test (fake mode)"
echo "  Seed: $SEED"
echo "======================================================"

echo ""
echo "[Step 1] prepare_dataset.py"
$APT pipeline/prepare_dataset.py \
    --tweets  datasets/examples/tweets.csv \
    --survey  datasets/examples/survey.csv \
    --pool-size 200 \
    --seed "$SEED"

echo ""
echo "[Step 2] run_llm_recommendation.py --fake (all 3 providers)"
for provider in anthropic openai gemini; do
    $APT pipeline/run_llm_recommendation.py \
        --provider $provider \
        --n-trials 20 \
        --sample-size 50 \
        --fake \
        --seed "$SEED"
done

echo ""
echo "[Step 3] compute_text_features.py --fake (all 3 providers)"
for exp_dir in outputs/experiments/*/; do
    $APT pipeline/compute_text_features.py \
        --experiment-dir "$exp_dir" \
        --fake
done

echo ""
echo "[Step 4] compute_bias_metrics.py"
$APT pipeline/compute_bias_metrics.py

echo ""
echo "[Step 5] generate_figures.py"
$APT pipeline/generate_figures.py

echo ""
echo "======================================================"
echo "  Test complete."
echo "  Outputs:"
echo "    outputs/pools/twitter_pool.csv"
echo "    outputs/experiments/{provider}_*/post_level_data.csv"
echo "    analysis_outputs/pool_vs_recommended_summary.csv"
echo "    analysis_outputs/visualizations/paper_plots_final/"
echo "======================================================"
