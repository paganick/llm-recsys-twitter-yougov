#!/usr/bin/env bash
# Analysis steps only (Steps 4–5): bias metrics and figures.
# Run this after the experiment CSVs have been enriched with text features.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "[Step 4] Compute bias metrics"
python pipeline/compute_bias_metrics.py

echo ""
echo "[Step 5] Generate figures"
python pipeline/generate_figures.py

echo ""
echo "✓ Analysis complete. Figures saved to analysis_outputs/visualizations/paper_plots_final/"
