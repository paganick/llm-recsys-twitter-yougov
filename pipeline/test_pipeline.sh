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

echo "======================================================"
echo "  LLM Recommendation Bias — pipeline test (fake mode)"
echo "======================================================"

echo ""
echo "[Step 1] prepare_dataset.py"
$APT pipeline/prepare_dataset.py \
    --tweets  datasets/examples/tweets.csv \
    --survey  datasets/examples/survey.csv \
    --pool-size 30 \
    --seed 42

echo ""
echo "[Step 2] run_llm_recommendation.py --fake"
$APT pipeline/run_llm_recommendation.py \
    --provider anthropic \
    --n-trials 5 \
    --sample-size 20 \
    --fake

echo ""
echo "[Step 3] compute_text_features.py --fake"
$APT pipeline/compute_text_features.py \
    --experiment-dir outputs/experiments/anthropic_claude-sonnet-4-5 \
    --fake

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
echo "    outputs/experiments/anthropic_claude-sonnet-4-5/post_level_data.csv"
echo "    analysis_outputs/pool_vs_recommended_summary.csv"
echo "    analysis_outputs/visualizations/paper_plots_final/"
echo "======================================================"
