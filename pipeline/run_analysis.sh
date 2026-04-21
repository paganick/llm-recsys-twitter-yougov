#!/usr/bin/env bash
# Steps 4–5: bias metrics and figures.
# Run after experiment CSVs have been enriched with text features.

set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v apptainer &>/dev/null; then
    module load apptainer 2>/dev/null \
        || { echo "ERROR: apptainer not found. Run: module load apptainer"; exit 1; }
fi

SIF="${SIF:-llm-recsys-twitter.sif}"
APT="apptainer exec $SIF python"

echo "[Step 4] compute_bias_metrics.py"
$APT pipeline/compute_bias_metrics.py

echo ""
echo "[Step 5] generate_figures.py"
$APT pipeline/generate_figures.py

echo ""
echo "✓ Done. Figures saved to analysis_outputs/visualizations/paper_plots_final/"
