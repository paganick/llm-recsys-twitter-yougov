#!/usr/bin/env python3
"""
Evaluate LLM demographic inference accuracy.

For each provider, compares inferred demographic attributes
(from demographic_inference.csv) against ground-truth survey demographics
(from outputs/pools/author_features.csv), stratified by context level.

Outputs (written to analysis_outputs/demographic_inference/):
  accuracy_by_attribute.csv   — accuracy per (provider, context_level, attribute)
  accuracy_heatmap.png        — heatmap: attributes × context levels, one panel per provider

Usage
-----
    python pipeline/compute_demographic_inference.py
    python pipeline/compute_demographic_inference.py --fake
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT     = Path(__file__).parent.parent
POOLS    = ROOT / "outputs" / "pools"
EXP_DIR  = ROOT / "outputs" / "experiments"
OUT_DIR  = ROOT / "analysis_outputs" / "demographic_inference"

PROVIDERS = ["anthropic", "openai", "gemini", "google"]
_PROVIDER_ALIAS = {"google": "gemini"}

# Map inferred attribute key → ground-truth column in author_features.csv
ATTRIBUTE_MAP = {
    "gender":         "author_gender",
    "age":            "author_age",
    "race":           "author_race",
    "ideology":       "author_ideology",
    "partisanship":   "author_partisanship",
    "education":      "author_education",
    "income":         "author_income",
    "marital_status": "author_marital_status",
    "religiosity":    "author_religiosity",
}

# Normalize ground-truth values to match the LLM's output vocabulary.
# (Survey labels differ in capitalisation and sometimes wording.)
GROUND_TRUTH_NORM = {
    "author_ideology": {
        "very liberal":      "left",
        "liberal":           "center-left",
        "moderate":          "center",
        "conservative":      "center-right",
        "very conservative": "right",
    },
    "author_income": {
        "$30-60k":   "$30-60k",
        "$60-100k":  "$60-100k",
        "$100k+":    "$100k+",
        "<$30k":     "<$30k",
    },
}


def _normalize(series: pd.Series, col: str) -> pd.Series:
    """Lowercase + apply any column-specific remapping."""
    s = series.astype(str).str.strip().str.lower()
    mapping = GROUND_TRUTH_NORM.get(col, {})
    if mapping:
        s = s.map(lambda v: mapping.get(v, v))
    return s


def load_ground_truth(pools_dir: Path) -> pd.DataFrame:
    """Load post_features (post_id→author_id) + author_features (demographics)."""
    post_df   = pd.read_csv(pools_dir / "post_features.csv",   low_memory=False,
                            usecols=["post_id", "author_id"])
    author_df = pd.read_csv(pools_dir / "author_features.csv", low_memory=False)
    return post_df.merge(author_df, on="author_id", how="left")


def compute_accuracy(df: pd.DataFrame, attr_key: str, gt_col: str) -> dict:
    """
    Compare inferred_<attr_key> against ground-truth <gt_col>.

    Returns a dict with:
      n_total      — rows with a ground-truth value
      n_predicted  — rows where the LLM gave a non-'unknown' answer
      accuracy     — correct / n_predicted  (excludes 'unknown' predictions)
      accuracy_all — correct / n_total      (treats 'unknown' as wrong)
      pct_unknown  — fraction predicted as 'unknown'
    """
    inferred = df[f"inferred_{attr_key}"].astype(str).str.strip().str.lower()
    truth    = _normalize(df[gt_col].dropna(), gt_col)

    # Align on index
    both = pd.DataFrame({"inferred": inferred, "truth": truth}).dropna()
    n_total = len(both)
    if n_total == 0:
        return {"n_total": 0, "n_predicted": 0,
                "accuracy": float("nan"), "accuracy_all": float("nan"),
                "pct_unknown": float("nan")}

    unknown_mask   = both["inferred"] == "unknown"
    n_unknown      = unknown_mask.sum()
    n_predicted    = n_total - n_unknown
    correct_mask   = both["inferred"] == both["truth"]
    n_correct      = correct_mask.sum()

    return {
        "n_total":      n_total,
        "n_predicted":  int(n_predicted),
        "accuracy":     round(n_correct / n_predicted, 4) if n_predicted else float("nan"),
        "accuracy_all": round(n_correct / n_total,     4),
        "pct_unknown":  round(n_unknown / n_total,     4),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fake", action="store_true",
                        help="Read from outputs_fake/ instead of outputs/")
    parser.add_argument("--pools-dir", type=Path, default=None,
                        help="Pools directory for ground truth (default: outputs/pools)")
    parser.add_argument("--experiments-dir", type=Path, default=None,
                        help="Experiments directory to read from "
                             "(default: outputs/experiments)")
    parser.add_argument("--analysis-dir", type=Path, default=OUT_DIR.parent,
                        help="Directory for analysis outputs "
                             "(default: analysis_outputs)")
    args = parser.parse_args()

    base = ROOT / ("outputs_fake" if args.fake else "outputs")
    pools_dir = Path(args.pools_dir) if args.pools_dir else base / "pools"
    exp_dir   = Path(args.experiments_dir) if args.experiments_dir else base / "experiments"
    out_dir_eff = Path(args.analysis_dir) / "demographic_inference"
    out_dir_eff.mkdir(parents=True, exist_ok=True)

    print("Loading ground truth ...")
    gt = load_ground_truth(pools_dir)
    print(f"  {len(gt):,} posts with author info")

    rows = []

    for exp_path in sorted(exp_dir.iterdir()):
        inf_csv = exp_path / "demographic_inference.csv"
        if not inf_csv.exists():
            continue

        provider = exp_path.name.split("_")[0]
        if provider not in PROVIDERS:
            continue
        provider = _PROVIDER_ALIAS.get(provider, provider)

        print(f"\nProvider: {provider.upper()}  ({exp_path.name})")
        inf_df = pd.read_csv(inf_csv, low_memory=False)
        print(f"  {len(inf_df):,} rows, context levels: "
              f"{sorted(inf_df['context_level'].unique())}")

        merged = inf_df.merge(gt, on="post_id", how="left")

        for cl in sorted(inf_df["context_level"].unique()):
            sub = merged[merged["context_level"] == cl]
            for attr_key, gt_col in ATTRIBUTE_MAP.items():
                if gt_col not in sub.columns:
                    continue
                stats = compute_accuracy(sub, attr_key, gt_col)
                rows.append({
                    "provider":      provider,
                    "context_level": cl,
                    "attribute":     attr_key,
                    **stats,
                })
                print(f"  {cl:14s} | {attr_key:15s}: "
                      f"acc={stats['accuracy']:.2%}  "
                      f"(all={stats['accuracy_all']:.2%}, "
                      f"unknown={stats['pct_unknown']:.1%})")

    if not rows:
        print("\nNo demographic_inference.csv files found. "
              "Run with --infer-demographics first.")
        sys.exit(1)

    acc_df = pd.DataFrame(rows)
    out_csv = out_dir_eff / "accuracy_by_attribute.csv"
    acc_df.to_csv(out_csv, index=False)
    print(f"\n✓ {out_csv}")

    # ── Heatmap: accuracy_all, attributes × context_levels, one panel per provider ──
    providers = acc_df["provider"].unique()
    n = len(providers)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)

    for ax, prov in zip(axes[0], providers):
        pdata = acc_df[acc_df["provider"] == prov]
        pivot = pdata.pivot(index="attribute", columns="context_level",
                            values="accuracy_all")
        sns.heatmap(
            pivot, ax=ax,
            vmin=0, vmax=1, center=0.5,
            cmap="RdYlGn", annot=True, fmt=".2f", linewidths=0.5,
            cbar_kws={"label": "Accuracy (incl. unknown)"},
        )
        ax.set_title(prov.title(), fontweight="bold")
        ax.set_xlabel("Context level")
        ax.set_ylabel("Attribute")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    plt.suptitle("Demographic Inference Accuracy vs Ground Truth",
                 fontweight="bold", fontsize=13, y=1.02)
    plt.tight_layout()
    out_png = out_dir_eff / "accuracy_heatmap.png"
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"✓ {out_png}")


if __name__ == "__main__":
    main()
