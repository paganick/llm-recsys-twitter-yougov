#!/usr/bin/env python3
"""
Generate paper figures for the Twitter/X LLM Recommendation Bias Study.

Reads from analysis_outputs/ and writes figures to:
    analysis_outputs/visualizations/paper_plots_final/

Figures generated
-----------------
01  Aggregated bias bar plot (all features, ordered by magnitude)
02  Bias-by-prompt R² heatmap (features × 6 prompt styles)
03  Normalized bias-by-prompt heatmap (z-score within features)
04  Demographic directional bias heatmap (partisanship, ideology, gender, race)
05  Content/safety bias by prompt × model (polarization, sentiment, toxicity)
06  Directional bias bar charts by model (polarization, sentiment, toxicity)
07  Feature importance by model (SHAP absolute values)
08  Primary topic bias heatmap by model × prompt style
09  Directional bias for selected continuous features

Usage
-----
    python generate_figures.py
    python generate_figures.py --base-dir /path/to/repo
"""

import argparse
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# ============================================================================
# CONFIGURATION
# ============================================================================

PROVIDERS = ["openai", "anthropic", "gemini"]
PROVIDER_LABELS = {
    "openai":    "GPT-4o-mini",
    "anthropic": "Claude Sonnet",
    "gemini":    "Gemini Flash",
}
PROVIDER_COLORS = {
    "openai":    "#10A37F",
    "anthropic": "#C87533",
    "gemini":    "#4285F4",
}
PROMPT_STYLES = ["general", "popular", "engaging", "informative", "controversial", "neutral"]

DEMOGRAPHIC_FEATURES = [
    "author_gender", "author_partisanship", "author_ideology", "author_race",
]
CONTENT_FEATURES = ["polarization_score", "sentiment_polarity", "toxicity"]

FEATURE_DISPLAY_NAMES = {
    "author_gender":         "Author: Gender",
    "author_partisanship":   "Author: Partisanship",
    "author_ideology":       "Author: Ideology",
    "author_race":           "Author: Race",
    "author_age":            "Author: Age",
    "author_education":      "Author: Education",
    "author_income":         "Author: Income",
    "author_marital_status": "Author: Marital Status",
    "author_religiosity":    "Author: Religiosity",
    "sentiment_polarity":    "Sentiment: Polarity",
    "sentiment_subjectivity":"Sentiment: Subjectivity",
    "has_emoji":             "Style: Has Emoji",
    "has_hashtag":           "Style: Has Hashtag",
    "has_mention":           "Style: Has Mention",
    "has_url":               "Style: Has URL",
    "avg_word_length":       "Text: Avg Word Length",
    "text_length":           "Text: Length",
    "word_count":            "Text: Word Count",
    "polarization_score":    "Content: Polarization",
    "primary_topic":         "Content: Primary Topic",
    "toxicity":              "Toxicity: Score",
    "is_reply":              "Tweet: Is Reply",
    "is_retweet":            "Tweet: Is Retweet",
    "is_quote":              "Tweet: Is Quote",
    "user_followers_count":  "User: Followers",
    "user_friends_count":    "User: Friends",
    "user_statuses_count":   "User: Statuses",
    "user_verified":         "User: Verified",
    "user_account_age_days": "User: Account Age",
    "engagement_score":      "User: Engagement Score",
}

DIVERGING_CMAP  = "RdYlBu_r"
SEQUENTIAL_CMAP = "viridis"

DIRECTIONAL_COLORS = {
    "negative": "#D55E00",
    "positive": "#009E73",
    "neutral":  "#999999",
}

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10})


def fmt(name: str) -> str:
    return FEATURE_DISPLAY_NAMES.get(name, name.replace("_", " ").title())


def dir_color(v: float) -> str:
    if v < -0.01:
        return DIRECTIONAL_COLORS["negative"]
    elif v > 0.01:
        return DIRECTIONAL_COLORS["positive"]
    return DIRECTIONAL_COLORS["neutral"]


# ============================================================================
# PATHS
# ============================================================================

def _init_paths(base_dir: Path):
    global OUT, SUMMARY_CSV, DIR_BIAS_CSV, IMPORTANCE_CSV
    analysis = base_dir / "analysis_outputs"
    OUT = analysis / "visualizations" / "paper_plots_final"
    OUT.mkdir(parents=True, exist_ok=True)
    SUMMARY_CSV    = analysis / "pool_vs_recommended_summary.csv"
    DIR_BIAS_CSV   = analysis / "directional_bias_data.csv"
    IMPORTANCE_CSV = analysis / "feature_importance_data.csv"


# ============================================================================
# FIGURE 01 — Aggregated bias bar plot
# ============================================================================

def fig01_aggregated_bias(summary: pd.DataFrame):
    agg = (
        summary.groupby("feature")["bias"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    agg["label"] = agg["feature"].map(fmt)

    fig, ax = plt.subplots(figsize=(10, max(4, len(agg) * 0.4)))
    colors = [PROVIDER_COLORS["anthropic"] if v > 0 else DIRECTIONAL_COLORS["neutral"]
              for v in agg["bias"]]
    ax.barh(agg["label"], agg["bias"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Mean Bias (Cohen's d / Cramér's V)", fontsize=11)
    ax.set_title("Average Bias Magnitude by Feature (across all providers × prompt styles)",
                 fontsize=12, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    out_path = OUT / "01_aggregated_bias.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    agg.to_csv(out_path.with_suffix(".csv"), index=False)
    print(f"  Saved: {out_path.name}")


# ============================================================================
# FIGURE 02 — Bias-by-prompt heatmap
# ============================================================================

def fig02_bias_heatmap(summary: pd.DataFrame):
    pivot = summary.pivot_table(
        values="bias", index="feature", columns="prompt_style", aggfunc="mean"
    )
    pivot = pivot[[s for s in PROMPT_STYLES if s in pivot.columns]]
    pivot.index = [fmt(f) for f in pivot.index]

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.5)))
    sns.heatmap(
        pivot, ax=ax, cmap=SEQUENTIAL_CMAP, annot=True, fmt=".3f",
        linewidths=0.5, cbar_kws={"label": "Bias (Cohen's d / Cramér's V)"},
    )
    ax.set_title("Bias by Feature × Prompt Style (mean across providers)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Prompt Style")
    ax.set_ylabel("")
    plt.tight_layout()
    out_path = OUT / "02_bias_heatmap_by_prompt.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ============================================================================
# FIGURE 03 — Normalized (z-score) bias heatmap
# ============================================================================

def fig03_normalized_heatmap(summary: pd.DataFrame):
    pivot = summary.pivot_table(
        values="bias", index="feature", columns="prompt_style", aggfunc="mean"
    )
    pivot = pivot[[s for s in PROMPT_STYLES if s in pivot.columns]]
    z = pivot.sub(pivot.mean(axis=1), axis=0).div(pivot.std(axis=1).replace(0, 1), axis=0)
    z.index = [fmt(f) for f in z.index]

    fig, ax = plt.subplots(figsize=(10, max(4, len(z) * 0.5)))
    sns.heatmap(
        z, ax=ax, cmap=DIVERGING_CMAP, center=0, annot=True, fmt=".2f",
        linewidths=0.5, cbar_kws={"label": "Z-score (within feature)"},
    )
    ax.set_title("Normalized Bias by Prompt Style (z-score within feature)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Prompt Style")
    ax.set_ylabel("")
    plt.tight_layout()
    out_path = OUT / "03_normalized_bias_heatmap.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ============================================================================
# FIGURE 04 — Demographic directional bias heatmap
# ============================================================================

def fig04_demographic_directional(dir_bias: pd.DataFrame):
    demo_feats = [f for f in DEMOGRAPHIC_FEATURES if f in dir_bias["feature"].unique()]
    if not demo_feats:
        print("  Fig 04: no demographic features found — skipping.")
        return

    rows = []
    for feat in demo_feats:
        sub = dir_bias[dir_bias["feature"] == feat]
        sub = sub[sub["category"] != "mean"]
        agg = (
            sub.groupby(["provider", "category"])["directional_bias"]
            .mean()
            .reset_index()
        )
        agg["feature"] = feat
        rows.append(agg)

    df = pd.concat(rows, ignore_index=True)
    df["label"] = df["feature"].map(fmt) + " = " + df["category"].astype(str)

    pivot = df.pivot_table(values="directional_bias", index="label", columns="provider", aggfunc="mean")
    pivot.columns = [PROVIDER_LABELS.get(c, c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(9, max(4, len(pivot) * 0.45)))
    sns.heatmap(
        pivot, ax=ax, cmap=DIVERGING_CMAP, center=0, annot=True, fmt=".3f",
        linewidths=0.5, cbar_kws={"label": "Directional Bias (rec − pool)"},
    )
    ax.set_title("Demographic Directional Bias (mean across prompt styles)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Provider")
    ax.set_ylabel("")
    plt.tight_layout()
    out_path = OUT / "04_demographic_directional_bias.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    df.to_csv(out_path.with_suffix(".csv"), index=False)
    print(f"  Saved: {out_path.name}")


# ============================================================================
# FIGURE 05 — Content/safety bias by prompt × model
# ============================================================================

def fig05_content_safety_heatmap(summary: pd.DataFrame):
    sub = summary[summary["feature"].isin(CONTENT_FEATURES)].copy()
    if sub.empty:
        print("  Fig 05: no content/safety features found — skipping.")
        return

    sub["provider_label"] = sub["provider"].map(PROVIDER_LABELS)
    pivot = sub.pivot_table(
        values="bias", index="feature", columns=["provider_label", "prompt_style"], aggfunc="mean"
    )
    pivot.index = [fmt(f) for f in pivot.index]

    fig, ax = plt.subplots(figsize=(14, max(3, len(pivot) * 0.8)))
    sns.heatmap(
        pivot, ax=ax, cmap=SEQUENTIAL_CMAP, annot=True, fmt=".3f",
        linewidths=0.5, cbar_kws={"label": "Bias"},
    )
    ax.set_title("Content/Safety Feature Bias by Provider × Prompt Style", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = OUT / "05_content_safety_bias_heatmap.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ============================================================================
# FIGURE 06 — Directional bias bar charts by model (content features)
# ============================================================================

def fig06_directional_bars(dir_bias: pd.DataFrame):
    for feat in CONTENT_FEATURES:
        sub = dir_bias[(dir_bias["feature"] == feat) & (dir_bias["category"] == "mean")]
        if sub.empty:
            continue

        agg = sub.groupby("provider")["directional_bias"].mean().reset_index()
        agg["color"] = agg["directional_bias"].map(dir_color)
        agg["label"] = agg["provider"].map(PROVIDER_LABELS)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(agg["label"], agg["directional_bias"], color=agg["color"],
                edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"Directional Bias — {fmt(feat)}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Directional Bias (recommended − pool mean)")
        plt.tight_layout()
        out_path = OUT / f"06_{feat}_directional_by_model.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path.name}")


# ============================================================================
# FIGURE 07 — Feature importance by model (SHAP)
# ============================================================================

def fig07_feature_importance(importance: pd.DataFrame):
    if importance.empty or "shap_importance" not in importance.columns:
        print("  Fig 07: no importance data — skipping.")
        return

    fig, axes = plt.subplots(1, len(PROVIDERS), figsize=(5 * len(PROVIDERS), 6), sharey=False)
    if len(PROVIDERS) == 1:
        axes = [axes]

    for ax, provider in zip(axes, PROVIDERS):
        sub = importance[importance["provider"] == provider]
        if sub.empty:
            ax.set_title(PROVIDER_LABELS.get(provider, provider))
            continue
        agg = sub.groupby("feature")["shap_importance"].mean().sort_values(ascending=True)
        agg.index = [fmt(f) for f in agg.index]
        ax.barh(agg.index, agg.values, color=PROVIDER_COLORS.get(provider, "#666"))
        ax.set_title(PROVIDER_LABELS.get(provider, provider), fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean |SHAP|")

    plt.suptitle("Feature Importance by Provider (SHAP, mean across prompt styles)",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = OUT / "07_feature_importance_by_model.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ============================================================================
# FIGURE 08 — Primary topic bias heatmap
# ============================================================================

def fig08_topic_bias(dir_bias: pd.DataFrame):
    sub = dir_bias[dir_bias["feature"] == "primary_topic"]
    if sub.empty:
        print("  Fig 08: no primary_topic data — skipping.")
        return

    pivot = sub.pivot_table(
        values="directional_bias", index="category", columns="provider", aggfunc="mean"
    )
    pivot.columns = [PROVIDER_LABELS.get(c, c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(8, max(4, len(pivot) * 0.5)))
    sns.heatmap(
        pivot, ax=ax, cmap=DIVERGING_CMAP, center=0, annot=True, fmt=".3f",
        linewidths=0.5, cbar_kws={"label": "Directional Bias"},
    )
    ax.set_title("Primary Topic Directional Bias by Provider", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = OUT / "08_topic_bias_by_model.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ============================================================================
# FIGURE 09 — Continuous feature directional bias
# ============================================================================

def fig09_continuous_directional(dir_bias: pd.DataFrame):
    cont_feats = [
        f for f in ["avg_word_length", "polarization_score", "sentiment_polarity", "toxicity"]
        if f in dir_bias["feature"].unique()
    ]
    if not cont_feats:
        print("  Fig 09: no continuous features found — skipping.")
        return

    fig, axes = plt.subplots(1, len(cont_feats), figsize=(4.5 * len(cont_feats), 4))
    if len(cont_feats) == 1:
        axes = [axes]

    for ax, feat in zip(axes, cont_feats):
        sub = dir_bias[(dir_bias["feature"] == feat) & (dir_bias["category"] == "mean")]
        agg = sub.groupby("provider")["directional_bias"].mean().reset_index()
        agg["label"] = agg["provider"].map(PROVIDER_LABELS)
        agg["color"] = agg["directional_bias"].map(dir_color)
        ax.barh(agg["label"], agg["directional_bias"], color=agg["color"],
                edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_title(fmt(feat), fontsize=10, fontweight="bold")
        ax.set_xlabel("Directional Bias")

    plt.suptitle("Directional Bias for Selected Continuous Features",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = OUT / "09_continuous_directional_bias.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--base-dir", type=Path, default=Path(__file__).parent.parent)
    _args, remaining = _parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[_parser],
    )
    args = parser.parse_args(remaining, namespace=_args)
    _init_paths(args.base_dir)

    print(f"Reading analysis outputs from {SUMMARY_CSV.parent} ...")
    if not SUMMARY_CSV.exists():
        print(f"ERROR: {SUMMARY_CSV} not found.")
        print(f"       Run first: python compute_bias_metrics.py")
        import sys; sys.exit(1)

    summary    = pd.read_csv(SUMMARY_CSV)
    dir_bias   = pd.read_csv(DIR_BIAS_CSV)
    importance = pd.read_csv(IMPORTANCE_CSV) if IMPORTANCE_CSV.exists() else pd.DataFrame()

    print(f"  Summary: {len(summary):,} rows | "
          f"Directional: {len(dir_bias):,} rows | "
          f"Importance: {len(importance):,} rows")
    print(f"\nGenerating figures → {OUT}/\n")

    fig01_aggregated_bias(summary)
    fig02_bias_heatmap(summary)
    fig03_normalized_heatmap(summary)
    fig04_demographic_directional(dir_bias)
    fig05_content_safety_heatmap(summary)
    fig06_directional_bars(dir_bias)
    fig07_feature_importance(importance)
    fig08_topic_bias(dir_bias)
    fig09_continuous_directional(dir_bias)

    print(f"\n✓ Done. All figures saved to {OUT}/")


if __name__ == "__main__":
    main()
