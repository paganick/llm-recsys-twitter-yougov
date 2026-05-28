#!/usr/bin/env python3
"""
generate_figures.py — All paper figures for the Twitter/X LLM Recommendation Bias Study.

Reads from analysis_outputs/ and writes to:
    analysis_outputs/visualizations/paper_plots_final/

Figures
-------
01   Aggregated R² bar plot (all features, ordered by magnitude)
02   Bias-by-prompt R² heatmap (features × prompt styles)
03   Normalized bias-by-prompt heatmap (z-score within features)
04   Demographic directional bias (partisanship, ideology, gender, race)
05   Content/safety bias by prompt × model heatmap
06   Content/safety directional bias bar charts by model
07   Feature importance by model (SHAP, absolute values)
08a  Primary topic bias heatmap by model
08b  Primary topic bias heatmap by prompt style
09a  Avg word length directional bias by model / prompt
09b  Polarization directional bias by model / prompt
09c  Sentiment polarity directional bias by model / prompt
09d  Toxicity directional bias by model / prompt
10   Demographic bias by model (per demographic variable)

Usage
-----
    python generate_figures.py
    python generate_figures.py --base-dir /path/to/repo
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

def _init_paths(base_dir: Path):
    global BASE, ANALYSIS, OUT, SUMMARY_CSV, DIR_BIAS_CSV, IMPORTANCE_CSV, POOL_CSV
    BASE     = base_dir
    ANALYSIS = BASE / "analysis_outputs"
    OUT      = ANALYSIS / "visualizations" / "paper_plots_final"
    if OUT.exists():
        import shutil
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    SUMMARY_CSV    = ANALYSIS / "pool_vs_recommended_summary.csv"
    DIR_BIAS_CSV   = ANALYSIS / "directional_bias_data.csv"
    IMPORTANCE_CSV = ANALYSIS / "feature_importance_data.csv"
    POOL_CSV       = BASE / "outputs" / "pools" / "twitter_pool.csv"

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--base-dir", type=Path, default=Path(__file__).parent.parent)
_args, _ = _parser.parse_known_args()
_init_paths(_args.base_dir)

# ============================================================================
# SHARED CONSTANTS
# ============================================================================

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"]  = 150
plt.rcParams["savefig.dpi"] = 300

PROVIDER_ORDER  = ["anthropic", "openai", "gemini", "google"]
PROVIDER_LABELS = {
    "anthropic": "Claude Sonnet 4.5",
    "openai":    "GPT-4o-mini",
    "gemini":    "Gemini 2.0 Flash",
    "google":    "Gemini 2.0 Flash",
}
_PROVIDER_PREFERRED_ORDER = ["anthropic", "openai", "gemini", "google"]

PROMPT_ORDER  = ["neutral", "general", "popular", "engaging", "informative", "controversial"]
PROMPT_LABELS = {p: p.capitalize() for p in PROMPT_ORDER}

AUTHOR_FEATURES = ["author_gender", "author_partisanship", "author_ideology", "author_race"]

FEATURES_ALL = {
    "author":          [
        "author_gender", "author_partisanship", "author_ideology", "author_race",
        "author_age", "author_education", "author_income", "author_marital_status", "author_religiosity",
    ],
    "text_metrics":    ["avg_word_length", "text_length"],
    "sentiment":       ["sentiment_polarity", "sentiment_subjectivity"],
    "style":           ["has_emoji", "has_hashtag", "has_mention", "has_url"],
    "content":         ["polarization_score", "primary_topic"],
    "toxicity":        ["toxicity"],
    "author_metadata": ["user_followers_count", "user_friends_count", "user_statuses_count", "user_favourites_count"],
    "post_metadata":   ["favorite_count", "retweet_count", "retweeted"],
}

FEATURE_DISPLAY = {
    "author_gender":          "Author: Gender",
    "author_partisanship":    "Author: Partisanship",
    "author_ideology":        "Author: Ideology",
    "author_race":            "Author: Race",
    "author_age":             "Author: Age",
    "author_education":       "Author: Education",
    "author_income":          "Author: Income",
    "author_marital_status":  "Author: Marital Status",
    "author_religiosity":     "Author: Religiosity",
    "avg_word_length":        "Text: Avg Word Length",
    "text_length":            "Text: Character Length",
    "polarization_score":     "Content: Polarization Score",
    "primary_topic":          "Content: Primary Topic",
    "sentiment_polarity":     "Sentiment: Polarity",
    "sentiment_subjectivity": "Sentiment: Subjectivity",
    "has_emoji":              "Style: Has Emoji",
    "has_hashtag":            "Style: Has Hashtag",
    "has_mention":            "Style: Has Mention",
    "has_url":                "Style: Has URL",
    "toxicity":               "Toxicity: Toxicity",
    "user_followers_count":   "Author Meta: Followers",
    "user_friends_count":     "Author Meta: Following",
    "user_statuses_count":    "Author Meta: Tweet Count",
    "user_favourites_count":  "Author Meta: Likes Given",
    "favorite_count":         "Post Meta: Likes",
    "retweet_count":          "Post Meta: Retweets",
    "retweeted":              "Post Meta: Is Retweeted",
}

CATEGORY_COLORS = {
    "author":          ["#4A1A00", "#6B2F0A", "#8B4513", "#A0522D", "#CD853F", "#C8965A", "#DEB887", "#E8C99A", "#F5DEB3"],
    "text_metrics":    ["#1E90FF"],
    "content":         ["#32CD32", "#3CB371"],
    "sentiment":       ["#FFD700", "#FFA500"],
    "style":           ["#9370DB", "#8A2BE2", "#9400D3", "#9932CC"],
    "toxicity":        ["#DC143C"],
    "author_metadata": ["#008080", "#20B2AA", "#48D1CC", "#7FFFD4"],
    "post_metadata":   ["#FF6347", "#FF7F50", "#FFA07A"],
}

DIVG_COLORS = [
    "#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#F7F7F7",
    "#FFFFFF",
    "#FEE0D2", "#FCBBA1", "#FC9272", "#FB6A4A", "#DE2D26",
]
CMAP_DIVG = LinearSegmentedColormap.from_list("diverging", DIVG_COLORS, N=256)
CMAP_WR   = LinearSegmentedColormap.from_list(
    "white_red",
    ["#FFFFFF", "#FFF5F0", "#FEE0D2", "#FCBBA1", "#FC9272",
     "#FB6A4A", "#EF3B2C", "#CB181D", "#A50F15", "#67000D"],
    N=256,
)

RQ3_METRICS = {
    "polarization_score": {
        "short_name": "Polarization",
        "ylabel":     "Polarization Bias\n(Recommended − Pool)",
    },
    "sentiment_polarity": {
        "short_name": "Sentiment",
        "ylabel":     "Sentiment Polarity Bias\n(Recommended − Pool)",
    },
    "toxicity": {
        "short_name": "Toxicity",
        "ylabel":     "Toxicity Bias\n(Recommended − Pool)",
    },
}

TOPIC_DISPLAY = {
    "news_&_social_concern":    "News &\nSocial Concern",
    "diaries_&_daily_life":     "Diaries &\nDaily Life",
    "sports":                   "Sports",
    "business_&_entrepreneurs": "Business &\nEntrepreneurs",
    "celebrity_&_pop_culture":  "Celebrity &\nPop Culture",
    "film_tv_&_video":          "Film, TV\n& Video",
}
TOP_N_TOPICS = 3

# ============================================================================
# HELPERS
# ============================================================================

def fmt(feature):
    return FEATURE_DISPLAY.get(feature, feature.replace("_", " ").title())

def get_category(feature):
    for cat, feats in FEATURES_ALL.items():
        if feature in feats:
            return cat
    return "other"

def get_color(feature, idx=0):
    colors = CATEGORY_COLORS.get(get_category(feature), ["#888888"])
    return colors[idx % len(colors)]

def to_r2(row):
    if pd.isna(row["bias"]) or pd.isna(row["metric"]):
        return np.nan
    v = abs(row["bias"])
    if row["metric"] == "Cohen's d":
        return (v ** 2) / (v ** 2 + 4)
    elif row["metric"] == "Cramér's V":
        return v ** 2
    return np.nan

def _filter_context(df: pd.DataFrame, context_level: str = "none") -> pd.DataFrame:
    """Filter to a single context_level if the column exists."""
    if "context_level" in df.columns:
        return df[df["context_level"] == context_level].copy()
    return df


def load_summary(context_level: str = "none"):
    df = pd.read_csv(SUMMARY_CSV)
    df = _filter_context(df, context_level)
    df["r_squared"] = df.apply(to_r2, axis=1)
    return df.dropna(subset=["r_squared"])


def load_summary_all():
    """Load all context levels (for new figures 11–13)."""
    df = pd.read_csv(SUMMARY_CSV)
    df["r_squared"] = df.apply(to_r2, axis=1)
    if "context_level" not in df.columns:
        df["context_level"] = "none"
    return df.dropna(subset=["r_squared"])

def make_r2_annot(pivot_r2, pivot_sig, mean_row_name="Mean Across Features"):
    annot = np.empty_like(pivot_r2, dtype=object)
    for i in range(pivot_r2.shape[0]):
        rn = pivot_r2.index[i]
        for j in range(pivot_r2.shape[1]):
            val = pivot_r2.iloc[i, j]
            sig = pivot_sig.iloc[i, j] if not pd.isna(pivot_sig.iloc[i, j]) else 0
            if pd.isna(val):
                annot[i, j] = ""
            elif rn == mean_row_name:
                annot[i, j] = f"{val:.3f}"
            elif sig > 0.75:
                annot[i, j] = f"{val:.3f}***"
            elif sig > 0.60:
                annot[i, j] = f"{val:.3f}**"
            elif sig > 0.50:
                annot[i, j] = f"{val:.3f}*"
            else:
                annot[i, j] = f"{val:.3f}"
    return annot

def _load_metric_bias(feature_name, context_level="none"):
    df = pd.read_csv(DIR_BIAS_CSV)
    df = _filter_context(df, context_level)
    mask = (df["feature"] == feature_name) & (
        df["feature_type"].isin(["numerical", "binary"])
    )
    return df[mask][["provider", "prompt_style", "directional_bias"]].copy()

# ============================================================================
# FIGURE 01 — Aggregated R² bar plot
# ============================================================================

def plot_01_aggregated_bar(comp_df):
    print("\n" + "="*70)
    print("FIGURE 01 — Aggregated R² bar plot")
    print("="*70)

    agg = comp_df.groupby("feature").agg(
        r_squared=("r_squared", "mean"),
        significant=("significant", "mean"),
    ).reset_index()
    agg["category"]        = agg["feature"].apply(get_category)
    agg["feature_display"] = agg["feature"].apply(fmt)
    agg = agg.sort_values("r_squared", ascending=False).reset_index(drop=True)

    colors, cat_counts = [], {}
    for _, row in agg.iterrows():
        idx = cat_counts.get(row["category"], 0)
        colors.append(get_color(row["feature"], idx))
        cat_counts[row["category"]] = idx + 1

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(agg["feature_display"], agg["r_squared"],
           color=colors, edgecolor="black", alpha=0.8, linewidth=0.5)

    for i, row in agg.iterrows():
        marker = ("***" if row["significant"] > 0.75 else
                  "**"  if row["significant"] > 0.60 else
                  "*"   if row["significant"] > 0.50 else None)
        if marker:
            ax.text(i, row["r_squared"], marker, ha="center", va="bottom",
                    fontsize=12, fontweight="bold")

    ax.set_ylabel("Average R²", fontsize=12, fontweight="bold")
    ax.set_title("Average Bias per Feature (R²)\n(* p<0.05 >50%, ** >60%, *** >75%)",
                 fontweight="bold", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    plt.xticks(rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    legend_elements = []
    for cat in ["author", "text_metrics", "sentiment", "style", "content", "toxicity", "author_metadata", "post_metadata"]:
        feats = FEATURES_ALL.get(cat, [])
        if not feats:
            continue
        label = cat.replace("_", " ").title()
        legend_elements.append(Patch(facecolor=get_color(feats[0], 0),
                                     edgecolor="black", label=label))
    ax.legend(handles=legend_elements, loc="upper right",
              title="Feature Category", fontsize=11, title_fontsize=11)

    plt.tight_layout()
    fig.savefig(OUT / "01_aggregated_r2_bar_plot.png", bbox_inches="tight")
    plt.close()
    print("  ✓ 01_aggregated_r2_bar_plot.png")

    agg[["feature", "feature_display", "category", "r_squared", "significant"]].to_csv(
        OUT / "01_aggregated_r2_bar_plot_data.csv", index=False)
    print("  ✓ 01_aggregated_r2_bar_plot_data.csv")

# ============================================================================
# FIGURE 02 — Bias-by-prompt R² heatmap
# ============================================================================

def plot_02_bias_by_prompt(comp_df):
    print("\n" + "="*70)
    print("FIGURE 02 — Bias-by-prompt R² heatmap")
    print("="*70)

    agg_p = comp_df.groupby(["feature", "prompt_style"]).agg(
        r_squared=("r_squared", "mean"), significant=("significant", "mean")
    ).reset_index()
    agg_a = comp_df.groupby("feature").agg(
        r_squared=("r_squared", "mean"), significant=("significant", "mean")
    ).reset_index()
    agg_a["prompt_style"] = "Average"

    combined  = pd.concat([agg_p, agg_a], ignore_index=True)
    pivot_r2  = combined.pivot(index="feature", columns="prompt_style", values="r_squared")
    pivot_sig = combined.pivot(index="feature", columns="prompt_style", values="significant")

    col_order = PROMPT_ORDER + ["Average"]
    col_order = [c for c in col_order if c in pivot_r2.columns]
    pivot_r2  = pivot_r2[col_order].sort_values("Average", ascending=False)
    pivot_sig = pivot_sig[col_order].reindex(pivot_r2.index)

    mean_vals = {c: pivot_r2[c].mean() for c in col_order}
    pivot_r2  = pd.concat([pivot_r2,
                            pd.Series(mean_vals, name="Mean Across Features").to_frame().T])
    pivot_sig = pd.concat([pivot_sig,
                            pd.Series({c: np.nan for c in col_order},
                                      name="Mean Across Features").to_frame().T])

    pivot_r2.index  = [fmt(f) if f != "Mean Across Features" else f for f in pivot_r2.index]
    pivot_sig.index = pivot_r2.index
    pivot_r2.columns  = [c.title() for c in pivot_r2.columns]
    pivot_sig.columns = pivot_r2.columns

    annot = make_r2_annot(pivot_r2, pivot_sig)

    fig, ax = plt.subplots(figsize=(12, max(8, len(pivot_r2) * 0.6)))
    sns.heatmap(pivot_r2, annot=annot, fmt="", cmap=CMAP_WR,
                vmin=0, vmax=pivot_r2.max().max(), ax=ax,
                cbar_kws={"label": "R² (Variance Explained)"},
                linewidths=0.5, linecolor="lightgray",
                annot_kws={"fontsize": 15})
    ax.collections[0].colorbar.ax.yaxis.label.set_size(18)
    ax.collections[0].colorbar.ax.tick_params(labelsize=14)

    avg_col = pivot_r2.columns.get_loc("Average")
    ax.axvline(x=avg_col,   color="black", linewidth=3)
    ax.axvline(x=avg_col+1, color="black", linewidth=3)
    for i in range(len(pivot_r2)):
        ax.add_patch(plt.Rectangle((avg_col, i), 1, 1,
                                   fill=True, facecolor="lightgray", alpha=0.2,
                                   edgecolor="black", linewidth=3, zorder=0))
    ax.axhline(y=len(pivot_r2)-1, color="black", linewidth=3)

    ax.set_title(
        "Bias by Prompt Style ($R^2$) — Aggregated across Models\n"
        "(* p<0.05 >50%, ** >60%, *** >75%)",
        fontweight="bold", fontsize=15, pad=20)
    ax.set_xlabel("Prompt Style", fontsize=15, fontweight="bold")
    ax.set_ylabel("Feature",      fontsize=15, fontweight="bold")
    ax.tick_params(labelsize=15)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    fig.savefig(OUT / "02_bias_by_prompt_heatmap.png", bbox_inches="tight")
    plt.close()
    print("  ✓ 02_bias_by_prompt_heatmap.png")

    out_rows = []
    for feat_disp in pivot_r2.index:
        feat = next((k for k, v in FEATURE_DISPLAY.items() if v == feat_disp), feat_disp)
        row  = {"feature": feat, "feature_display": feat_disp}
        for col in pivot_r2.columns:
            row[col.lower()] = pivot_r2.loc[feat_disp, col]
        out_rows.append(row)
    pd.DataFrame(out_rows).to_csv(OUT / "02_bias_by_prompt_heatmap_data.csv", index=False)
    print("  ✓ 02_bias_by_prompt_heatmap_data.csv")

# ============================================================================
# FIGURE 03 — Normalized bias-by-prompt heatmap
# ============================================================================

def plot_03_normalized_bias(comp_df):
    print("\n" + "="*70)
    print("FIGURE 03 — Normalized bias-by-prompt heatmap")
    print("="*70)

    agg_p = comp_df.groupby(["feature", "prompt_style"]).agg(
        bias=("bias", "mean"), significant=("significant", "mean")
    ).reset_index()
    agg_a = comp_df.groupby("feature").agg(
        bias=("bias", "mean"), significant=("significant", "mean")
    ).reset_index()
    agg_a["prompt_style"] = "Average"

    combined  = pd.concat([agg_p, agg_a], ignore_index=True)
    pivot_b   = combined.pivot(index="feature", columns="prompt_style", values="bias")
    pivot_sig = combined.pivot(index="feature", columns="prompt_style", values="significant")

    avail_prompts = [p for p in PROMPT_ORDER if p in pivot_b.columns]
    pivot_b   = pivot_b[avail_prompts + ["Average"]]
    pivot_sig = pivot_sig[avail_prompts + ["Average"]]

    pivot_norm = pivot_b[avail_prompts].copy()
    for feature in pivot_norm.index:
        vals = pivot_b.loc[feature, avail_prompts].values.astype(float)
        mu, sd = vals.mean(), vals.std()
        pivot_norm.loc[feature, avail_prompts] = (vals - mu) / sd if sd > 0 else np.zeros_like(vals)

    avg_r2   = comp_df.groupby("feature")["r_squared"].mean()
    ordering = avg_r2.reindex(pivot_norm.index).sort_values(ascending=False).index
    pivot_norm = pivot_norm.reindex(ordering)
    pivot_sig_p = pivot_sig[avail_prompts].reindex(ordering)

    pivot_norm.index    = [fmt(f) for f in pivot_norm.index]
    pivot_sig_p.index   = pivot_norm.index
    pivot_norm.columns  = [c.title() for c in pivot_norm.columns]
    pivot_sig_p.columns = pivot_norm.columns

    annot = np.empty_like(pivot_norm, dtype=object)
    for i in range(pivot_norm.shape[0]):
        for j in range(pivot_norm.shape[1]):
            val = pivot_norm.iloc[i, j]
            sig = pivot_sig_p.iloc[i, j] if not pd.isna(pivot_sig_p.iloc[i, j]) else 0
            if pd.isna(val):
                annot[i, j] = ""
            elif sig > 0.75:
                annot[i, j] = f"{val:.1f}***"
            elif sig > 0.60:
                annot[i, j] = f"{val:.1f}**"
            elif sig > 0.50:
                annot[i, j] = f"{val:.1f}*"
            else:
                annot[i, j] = f"{val:.1f}"

    flat    = pivot_norm.values.flatten()
    flat    = flat[~np.isnan(flat.astype(float))]
    max_abs = max(abs(flat.min()), abs(flat.max())) if len(flat) > 0 else 1.0

    fig, ax = plt.subplots(figsize=(12, max(8, len(pivot_norm) * 0.6)))
    sns.heatmap(pivot_norm, annot=annot, fmt="", cmap=CMAP_DIVG,
                center=0, vmin=-max_abs, vmax=max_abs, ax=ax,
                cbar_kws={"label": "Normalized Bias (z-score)\n← Reduced | Enhanced →"},
                linewidths=0.5, linecolor="lightgray",
                annot_kws={"fontsize": 15})
    ax.collections[0].colorbar.ax.yaxis.label.set_size(16)
    ax.collections[0].colorbar.ax.tick_params(labelsize=15)

    ax.set_title(
        "Normalized Bias by Prompt Style — Aggregated across Models\n"
        "(red = enhanced, blue = reduced; * p<0.05 >50%, ** >60%, *** >75%)",
        fontweight="bold", fontsize=15, pad=20)
    ax.set_xlabel("Prompt Style", fontsize=16, fontweight="bold")
    ax.set_ylabel("Feature",      fontsize=16, fontweight="bold")
    ax.tick_params(labelsize=15)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    fig.savefig(OUT / "03_bias_by_prompt_normalized_heatmap.png", bbox_inches="tight")
    plt.close()
    print("  ✓ 03_bias_by_prompt_normalized_heatmap.png")

    out_rows = []
    for feat_disp in pivot_norm.index:
        feat = next((k for k, v in FEATURE_DISPLAY.items() if v == feat_disp), feat_disp)
        row  = {"feature": feat, "feature_display": feat_disp}
        for col in pivot_norm.columns:
            row[f"{col.lower()}_normalized"] = pivot_norm.loc[feat_disp, col]
        out_rows.append(row)
    pd.DataFrame(out_rows).to_csv(
        OUT / "03_bias_by_prompt_normalized_heatmap_data.csv", index=False)
    print("  ✓ 03_bias_by_prompt_normalized_heatmap_data.csv")

# ============================================================================
# FIGURE 04 — Demographic directional bias (one sub-panel per feature)
# ============================================================================

# Per-feature display config.  Only "display_name" is required.
# "preferred_order" fixes the column ordering for features where sequence is
# meaningful (e.g. ideology low→high); any data categories not listed are
# appended alphabetically.  Features with no preferred_order are sorted
# alphabetically from the data.
_DEMO_FEATURE_CONFIG = {
    "author_gender": {
        "display_name":   "Author Gender",
        "preferred_order": ["male", "female", "non-binary", "other", "unknown"],
    },
    "author_partisanship": {
        "display_name":   "Author Partisanship",
        "preferred_order": ["Democrat", "Republican", "Independent", "Other", "unknown"],
    },
    "author_ideology": {
        "display_name":   "Author Ideology",
        "preferred_order": ["left", "center-left", "center", "center-right", "right", "unknown"],
    },
    "author_race": {
        "display_name":   "Author Race",
    },
    "author_age": {
        "display_name":    "Author Age",
        "preferred_order": ["18-24", "25-34", "35-44", "45-54", "55-64", "65+", "unknown"],
    },
    "author_education": {
        "display_name":    "Author Education",
        "preferred_order": ["less than high school", "high school", "some college",
                            "college", "postgraduate", "unknown"],
    },
    "author_income": {
        "display_name":    "Author Income",
        "preferred_order": ["<$30k", "$30-60k", "$60-100k", "$100k+",
                            "Prefer not to say", "unknown"],
    },
    "author_marital_status": {
        "display_name":    "Author Marital Status",
        "preferred_order": ["single", "married", "divorced", "widowed", "other", "unknown"],
    },
    "author_religiosity": {
        "display_name":    "Author Religiosity",
        "preferred_order": ["not religious", "somewhat religious", "religious",
                            "very religious", "unknown"],
    },
}


def _resolve_categories(finfo: dict, data_cats, pool_props: dict | None = None) -> list:
    """Return ordered category list.

    - If preferred_order is set: use that order, append any unlisted cats by
      pool proportion (or alphabetically if no pool data).
    - If no preferred_order: sort all cats by pool proportion descending (most
      frequent first), falling back to alphabetical.
    """
    preferred = finfo.get("preferred_order", [])
    data_set  = set(str(c) for c in data_cats)

    def by_freq(c):
        return -(pool_props.get(c, 0) if pool_props else 0)

    if preferred:
        ordered   = [c for c in preferred if c in data_set]
        remaining = sorted((c for c in data_set if c not in preferred), key=by_freq)
        return ordered + remaining
    else:
        return sorted(data_set, key=by_freq)


def _cat_label(cat: str) -> str:
    return str(cat).replace("_", " ").title()


def _load_demo_bias(context_level="none"):
    dir_bias = pd.read_csv(DIR_BIAS_CSV)
    dir_bias = _filter_context(dir_bias, context_level)
    # Only normalise categorical demographic features (not numerical like age)
    cat_demo_features = [
        f for f in _DEMO_FEATURE_CONFIG
        if f in dir_bias["feature"].unique()
        and dir_bias.loc[dir_bias["feature"] == f, "feature_type"].iloc[0]
            in ("categorical", "binary")
    ]
    normalized_rows = []
    for feature in cat_demo_features:
        fdata = dir_bias[dir_bias["feature"] == feature].copy()
        for (prov, prompt), grp in fdata.groupby(["provider", "prompt_style"]):
            bias_sum   = grp["directional_bias"].sum()
            correction = bias_sum / len(grp) if abs(bias_sum) > 1e-10 else 0
            grp = grp.copy()
            grp["directional_bias"] -= correction
            normalized_rows.append(grp)
    dir_bias = dir_bias[~dir_bias["feature"].isin(cat_demo_features)]
    return pd.concat([dir_bias] + normalized_rows, ignore_index=True)


def plot_04_demographics(context_level="none"):
    print("\n" + "="*70)
    print("FIGURE 04 — Demographic directional bias")
    print("="*70)

    dir_bias = _load_demo_bias(context_level)

    # Pool proportions from pool CSV
    pool_props: dict = {}
    if POOL_CSV.exists():
        pool = pd.read_csv(POOL_CSV).drop_duplicates("post_id")
        for feat in _DEMO_FEATURE_CONFIG:
            if feat in pool.columns:
                vc = pool[feat].value_counts(normalize=True, dropna=False)
                vc.index = vc.index.map(lambda x: "unknown" if pd.isna(x) else x)
                pool_props[feat] = vc.to_dict()

    available = [f for f in _DEMO_FEATURE_CONFIG if f in dir_bias["feature"].unique()]
    if not available:
        print("  No demographic features found — skipping.")
        return

    all_panels = {}
    for feat in available:
        finfo     = _DEMO_FEATURE_CONFIG[feat]
        sub       = dir_bias[(dir_bias["feature"] == feat) &
                             (dir_bias["category"] != "mean")]

        mean_agg  = sub.groupby(["provider", "category"])["directional_bias"].mean().reset_index()
        std_agg   = sub.groupby(["provider", "category"])["directional_bias"].std().reset_index()
        piv_m     = mean_agg.pivot(index="provider", columns="category", values="directional_bias")
        piv_s     = std_agg.pivot( index="provider", columns="category", values="directional_bias")

        fp         = pool_props.get(feat, {})
        avail_cats = _resolve_categories(finfo, piv_m.columns, fp)
        avail_cats = [c for c in avail_cats if c in piv_m.columns]
        piv_m = piv_m[avail_cats].reindex(PROVIDER_ORDER)
        piv_s = piv_s[avail_cats].reindex(PROVIDER_ORDER)

        avg_row = pd.Series(piv_m.mean(axis=0), name="Average")
        std_row = pd.Series(piv_m.std(axis=0),  name="Average")
        piv_m   = pd.concat([piv_m, avg_row.to_frame().T])
        piv_s   = pd.concat([piv_s, std_row.to_frame().T])

        col_labels = []
        for col in avail_cats:
            lbl = _cat_label(col)
            if col in fp:
                lbl += f"\n({fp[col]*100:.1f}%)"
            col_labels.append(lbl)

        piv_disp         = piv_m.copy()
        piv_disp.columns = col_labels
        piv_disp.index   = [
            p if p == "Average" else PROVIDER_LABELS.get(p, p)
            for p in piv_disp.index
        ]

        annot = np.empty_like(piv_disp.values, dtype=object)
        for i in range(piv_disp.shape[0]):
            for j in range(piv_disp.shape[1]):
                v = piv_disp.iloc[i, j]
                s = piv_s.iloc[i, j]
                if pd.isna(v):
                    annot[i, j] = ""
                elif pd.isna(s):
                    annot[i, j] = f"{v:.3f}"
                else:
                    annot[i, j] = f"{v:.3f}\n±{s:.3f}"

        all_vals = [x for x in piv_disp.values.flatten() if not pd.isna(x)]
        max_abs  = max(abs(min(all_vals)), abs(max(all_vals))) if all_vals else 0.1
        max_abs  = max(max_abs, 1e-6)  # guard against degenerate vmin=vmax=0
        all_panels[feat] = (piv_disp, piv_s, piv_m, avail_cats, annot, max_abs, finfo)

        rows = []
        for p in PROVIDER_ORDER + ["Average"]:
            if p not in piv_m.index:
                continue
            row = {"provider": p}
            for cat in avail_cats:
                row[cat]          = piv_m.loc[p, cat]
                row[f"{cat}_std"] = piv_s.loc[p, cat]
            rows.append(row)
        pd.DataFrame(rows).to_csv(OUT / f"04_{feat}_data.csv", index=False)
        print(f"  ✓ 04_{feat}_data.csv")

    from matplotlib import gridspec as mgridspec

    ROW_SIZE   = 5
    rows_feat  = [available[i:i + ROW_SIZE] for i in range(0, len(available), ROW_SIZE)]
    nrows      = len(rows_feat)
    fig_width  = max(14, max(
        sum(len(all_panels[f][3]) for f in row) * 1.8 for row in rows_feat
    ))
    fig_height = 5.5 * nrows + 0.5 * max(0, nrows - 1)
    fig        = plt.figure(figsize=(fig_width, fig_height))
    outer_gs   = mgridspec.GridSpec(nrows, 1, figure=fig, hspace=0.8)

    all_axes_list = []
    for r_idx, row_feats in enumerate(rows_feat):
        row_widths = [len(all_panels[f][3]) for f in row_feats]
        inner_gs   = mgridspec.GridSpecFromSubplotSpec(
            1, len(row_feats),
            subplot_spec=outer_gs[r_idx],
            width_ratios=row_widths,
            wspace=0.4,
        )
        row_axes = [fig.add_subplot(inner_gs[0, j]) for j in range(len(row_feats))]
        all_axes_list.append((row_feats, row_axes))

    fig.suptitle(
        "Author Demographic Directional Bias (mean across prompt styles)",
        fontweight="bold", fontsize=18, y=1.01)

    cbar_label = "Directional Bias\n← Under | Over-represented →"
    last_mesh  = None

    for row_feats, row_axes in all_axes_list:
        for idx, (feat, ax) in enumerate(zip(row_feats, row_axes)):
            piv_disp, _, _, avail_cats, annot, max_abs, finfo = all_panels[feat]

            sns.heatmap(piv_disp, annot=annot, fmt="", cmap=CMAP_DIVG,
                        center=0, vmin=-max_abs, vmax=max_abs, ax=ax,
                        cbar=False,
                        linewidths=0.5, linecolor="gray", annot_kws={"fontsize": 16})

            if ax.collections:
                last_mesh = ax.collections[0]

            ax.axhline(y=len(piv_disp) - 1, color="black", linewidth=2.5)
            ax.set_title(finfo["display_name"], fontsize=17, fontweight="bold", pad=12)
            ax.set_xlabel("Category", fontsize=17, fontweight="bold")
            if idx == 0:
                ax.set_ylabel("Model", fontsize=17, fontweight="bold")
                ax.tick_params(axis="y", labelsize=17)
                plt.setp(ax.get_yticklabels(), rotation=0, ha="right")
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
                ax.tick_params(axis="y", left=False)
            ax.tick_params(axis="x", labelsize=17)
            plt.setp(ax.get_xticklabels(), rotation=40, ha="right")

    flat_axes = [ax for _, row_axes in all_axes_list for ax in row_axes]
    if last_mesh is not None:
        cbar = fig.colorbar(last_mesh, ax=flat_axes, shrink=0.6, pad=0.01)
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label(cbar_label, fontsize=15, fontweight="bold")
    fig.savefig(OUT / "04_demographics_directional_bias_heatmap.png",
                bbox_inches="tight", dpi=300)
    plt.close()
    print("  ✓ 04_demographics_directional_bias_heatmap.png")

# ============================================================================
# FIGURE 05 — Content/safety heatmap (prompt × model)
# ============================================================================

def plot_05_content_safety_heatmap(context_level="none"):
    print("\n" + "="*70)
    print("FIGURE 05 — Content/safety heatmap (prompt × model)")
    print("="*70)

    fig, axes = plt.subplots(1, 3, figsize=(20, 8.05))
    all_rows  = []

    for idx, (feat, minfo) in enumerate(RQ3_METRICS.items()):
        ax   = axes[idx]
        data = _load_metric_bias(feat, context_level)

        agg_m = data.groupby(["provider", "prompt_style"])["directional_bias"].mean().reset_index()
        piv_m = agg_m.pivot(index="prompt_style", columns="provider",
                             values="directional_bias")
        piv_m = piv_m.reindex(index=PROMPT_ORDER, columns=PROVIDER_ORDER)

        std_col_vals = piv_m.std(axis=1).values
        std_row_vals = piv_m.std(axis=0).values
        overall_std  = float(np.nanstd(piv_m.values))

        avg_col = piv_m.mean(axis=1)
        avg_row = pd.Series(piv_m.mean(axis=0), name="Average")
        piv_m   = pd.concat([piv_m, avg_row.to_frame().T])

        avg_col["Average"] = avg_col.mean()
        piv_m["Average"]   = avg_col

        short = {p: PROVIDER_LABELS[p] for p in PROVIDER_ORDER}
        short["Average"] = "Average"
        piv_m.columns = [short.get(p, p) for p in piv_m.columns]
        piv_m.index   = [PROMPT_LABELS.get(p, p) for p in piv_m.index]

        n_rows, n_cols = piv_m.shape
        annot = np.empty_like(piv_m, dtype=object)
        for i in range(n_rows):
            for j in range(n_cols):
                val = piv_m.iloc[i, j]
                if pd.isna(val):
                    annot[i, j] = ""
                elif i == n_rows - 1 and j == n_cols - 1:
                    annot[i, j] = f"{val:.3f}\n±{overall_std:.3f}"
                elif i == n_rows - 1:
                    annot[i, j] = f"{val:.3f}\n±{std_row_vals[j]:.3f}"
                elif j == n_cols - 1:
                    annot[i, j] = f"{val:.3f}\n±{std_col_vals[i]:.3f}"
                else:
                    annot[i, j] = f"{val:.3f}"

        max_abs = max(abs(piv_m.min().min()), abs(piv_m.max().max()))
        sns.heatmap(piv_m, annot=annot, fmt="", cmap=CMAP_DIVG,
                    center=0, vmin=-max_abs, vmax=max_abs, ax=ax,
                    cbar=True,
                    cbar_kws={"label": minfo["ylabel"].replace("\n", " ")},
                    linewidths=0.5, linecolor="gray", annot_kws={"fontsize": 16})
        ax.collections[0].colorbar.ax.yaxis.label.set_size(18)
        ax.collections[0].colorbar.ax.tick_params(labelsize=16)
        ax.axhline(y=len(piv_m)-1, color="black", linewidth=2.5)
        ax.axvline(x=len(piv_m.columns)-1, color="black", linewidth=2.5)
        ax.set_title(minfo["short_name"], fontweight="bold", fontsize=20, pad=12)
        ax.set_xlabel("Model", fontsize=18, fontweight="bold")
        ax.set_ylabel("Prompt Style" if idx == 0 else "", fontsize=18, fontweight="bold")
        ax.tick_params(labelsize=17)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=15)
        if idx > 0:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center", fontsize=15)

        for p_orig in PROVIDER_ORDER:
            for prompt in PROMPT_ORDER:
                val = data[(data["provider"] == p_orig) & (data["prompt_style"] == prompt)][
                    "directional_bias"].mean()
                all_rows.append({"feature": feat, "provider": p_orig,
                                  "provider_display": PROVIDER_LABELS[p_orig],
                                  "prompt_style": prompt,
                                  "prompt_display": PROMPT_LABELS[prompt],
                                  "directional_bias": val})

    fig.suptitle(
        "Content and Safety Directional Bias by Model and Prompt Style",
        fontweight="bold", fontsize=24, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "05_content_safety_bias_by_prompt_and_model_heatmap.png",
                bbox_inches="tight", dpi=300)
    plt.close()
    print("  ✓ 05_content_safety_bias_by_prompt_and_model_heatmap.png")

    pd.DataFrame(all_rows).to_csv(OUT / "05_content_safety_bias_heatmap_data.csv", index=False)
    print("  ✓ 05_content_safety_bias_heatmap_data.csv")

# ============================================================================
# FIGURE 06 — Content/safety bar charts (prompt grid, bars by provider)
# ============================================================================

def plot_06_content_safety_bars(context_level="none"):
    print("\n" + "="*70)
    print("FIGURE 06 — Content/safety bar charts by model")
    print("="*70)

    all_rows = []
    for feat, minfo in RQ3_METRICS.items():
        data  = _load_metric_bias(feat, context_level)
        if data.empty:
            print(f"  skipped {feat} (no data)")
            continue
        vals  = data["directional_bias"].dropna()
        if vals.empty:
            print(f"  skipped {feat} (all NaN)")
            continue
        y_min = vals.min()
        y_max = vals.max()
        y_rng = y_max - y_min if y_max != y_min else 1.0
        y_lim = (y_min - 0.1*y_rng, y_max + 0.1*y_rng)

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        fig.suptitle(
            f"{minfo['short_name']} Directional Bias by Model\n"
            "(One panel per Prompt Style)",
            fontweight="bold", fontsize=16, y=0.98)

        pcolors = [
            "#C87533",   # anthropic — amber
            "#10A37F",   # openai    — green
            "#4285F4",   # gemini    — blue
        ]

        for panel_idx, prompt in enumerate(PROMPT_ORDER):
            ax    = axes[panel_idx]
            pdata = data[data["prompt_style"] == prompt]
            x     = np.arange(len(PROVIDER_ORDER))

            vals = []
            for p in PROVIDER_ORDER:
                sub = pdata[pdata["provider"] == p]["directional_bias"]
                vals.append(sub.values[0] if len(sub) > 0 else 0)

            ax.bar(x, vals, 0.55, color=pcolors, alpha=0.85,
                   edgecolor="black", linewidth=0.5)
            ax.axhline(y=0, color="black", linewidth=0.8, alpha=0.3)
            ax.set_title(PROMPT_LABELS[prompt], fontweight="bold", fontsize=13)
            ax.set_ylabel(minfo["ylabel"], fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels([PROVIDER_LABELS[p].split()[0] for p in PROVIDER_ORDER],
                               fontsize=10)
            ax.set_ylim(*y_lim)
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(OUT / f"06_{feat}_by_model.png", bbox_inches="tight", dpi=300)
        plt.close()
        print(f"  ✓ 06_{feat}_by_model.png")

        out = data.copy()
        out["feature"]          = feat
        out["provider_display"] = out["provider"].map(PROVIDER_LABELS)
        out["prompt_display"]   = out["prompt_style"].map(PROMPT_LABELS)
        all_rows.append(out)

    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(
            OUT / "06_content_safety_bars_data.csv", index=False)
        print("  ✓ 06_content_safety_bars_data.csv")

# ============================================================================
# FIGURE 07 — Feature importance by model (SHAP heatmap)
# ============================================================================

def plot_07_feature_importance(context_level="none"):
    print("\n" + "="*70)
    print("FIGURE 07 — Feature importance by model (SHAP, absolute)")
    print("="*70)

    try:
        df = pd.read_csv(IMPORTANCE_CSV)
    except Exception:
        df = pd.DataFrame()
    if df.empty:
        print("  skipped (no SHAP data — install shap and re-run compute_bias_metrics.py)")
        return
    df = _filter_context(df, context_level)
    if "feature" in df.columns and "shap_importance" in df.columns:
        df_long = df[["feature", "provider", "shap_importance"]].copy()
    else:
        shap_cols = [c for c in df.columns if c.startswith("shap_") and c != "shap_file"]
        rows = []
        for _, row in df.iterrows():
            for feat in [c.replace("shap_", "") for c in shap_cols]:
                rows.append({"provider": row["provider"], "feature": feat,
                              "shap_importance": row[f"shap_{feat}"]})
        df_long = pd.DataFrame(rows)

    agg   = df_long.groupby(["feature", "provider"])["shap_importance"].mean().reset_index()
    pivot = agg.pivot_table(values="shap_importance", index="provider",
                            columns="feature", aggfunc="mean").reindex(PROVIDER_ORDER)

    avg_row = pd.Series(pivot.mean(axis=0), name="Average\n(across models)")
    pivot   = pd.concat([pivot, avg_row.to_frame().T])

    sorted_feats = pivot.loc["Average\n(across models)"].sort_values(ascending=False).index.tolist()
    pivot = pivot[sorted_feats]

    pivot.index   = [idx if "Average" in str(idx) else PROVIDER_LABELS.get(idx, idx)
                     for idx in pivot.index]
    pivot.columns = [fmt(f) for f in pivot.columns]

    annot = np.empty_like(pivot, dtype=object)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            annot[i, j] = "" if pd.isna(val) else f"{val:.3f}"

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(pivot, annot=annot, fmt="", cmap=CMAP_WR,
                vmin=0, vmax=pivot.max().max(), ax=ax,
                cbar_kws={"label": "SHAP Importance"},
                linewidths=0.5, linecolor="lightgray",
                annot_kws={"fontsize": 15})
    ax.axhline(y=len(pivot)-1, color="black", linewidth=2.5)
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(15)
    cbar.ax.tick_params(labelsize=15)

    ax.set_title("Feature Importance by Model\n(Aggregated across prompt styles)",
                 fontweight="bold", fontsize=16, pad=15)
    ax.set_xlabel("Feature",        fontsize=16, fontweight="bold")
    ax.set_ylabel("Model", fontsize=16, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=15)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=15)

    plt.tight_layout()
    fig.savefig(OUT / "07_feature_importance_by_model.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("  ✓ 07_feature_importance_by_model.png")

    pivot.to_csv(OUT / "07_feature_importance_by_model_data.csv")
    print("  ✓ 07_feature_importance_by_model_data.csv")

# ============================================================================
# FIGURES 08a/b — Primary topic directional bias
# ============================================================================

def _topic_col_labels(pt, topics):
    pool = pt.groupby("category")["prop_pool"].mean()
    out  = []
    for t in topics:
        disp = TOPIC_DISPLAY.get(t, t.replace("_", " ").title())
        pct  = pool.get(t, np.nan)
        out.append(f"{disp}\n({pct*100:.1f}%)" if not np.isnan(pct) else disp)
    return out

def _make_topic_annot(piv, std=None):
    annot = np.empty_like(piv.values, dtype=object)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            if pd.isna(v):
                annot[i, j] = ""
            elif std is not None and not pd.isna(std.iloc[i, j]):
                annot[i, j] = f"{v:+.3f}\n±{std.iloc[i, j]:.3f}"
            else:
                annot[i, j] = f"{v:+.3f}"
    return annot

def _draw_topic_heatmap(ax, piv, annot, vmax, col_lbls, row_lbls,
                        show_cbar=True, show_ylabel=True,
                        ylabel="", title="", annot_fs=15):
    piv_disp         = piv.copy()
    piv_disp.columns = col_lbls
    piv_disp.index   = row_lbls
    im = sns.heatmap(piv_disp, annot=annot, fmt="", cmap=CMAP_DIVG,
                     center=0, vmin=-vmax, vmax=vmax, ax=ax,
                     linewidths=0.5, linecolor="gray",
                     cbar=show_cbar,
                     cbar_kws={"label": "Directional Bias", "shrink": 1.0} if show_cbar else {},
                     annot_kws={"fontsize": annot_fs})
    if show_cbar and im.collections:
        im.collections[0].colorbar.ax.yaxis.label.set_size(15)
        im.collections[0].colorbar.ax.tick_params(labelsize=15)
    ax.set_title(title, fontweight="bold", fontsize=16, pad=16)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel if show_ylabel else "", fontsize=16, fontweight="bold")
    if not show_ylabel:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=15)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)


def plot_08_topic_heatmaps(context_level="none"):
    print("\n" + "="*70)
    print("FIGURES 08a/b — Primary topic directional bias")
    print("="*70)

    df = pd.read_csv(DIR_BIAS_CSV)
    df = _filter_context(df, context_level)
    pt = df[df["feature"] == "primary_topic"].copy()
    if pt.empty:
        print("  No primary_topic data — skipping.")
        return

    top_topics = (
        pt.groupby("category")["prop_pool"].mean()
        .sort_values(ascending=False)
        .head(TOP_N_TOPICS)
        .index.tolist()
    )
    print(f"  Top {TOP_N_TOPICS} topics: {top_topics}")
    pt = pt[pt["category"].isin(top_topics)].copy()

    csv_data = pt.groupby(["provider", "prompt_style", "category"])[
        "directional_bias"].mean().reset_index()
    csv_data["category_display"] = csv_data["category"].map(
        lambda t: TOPIC_DISPLAY.get(t, t.replace("_", " ").title()))
    csv_data.to_csv(OUT / "08_topic_bias_data.csv", index=False)
    print("  ✓ 08_topic_bias_data.csv")

    vmax_global = max(abs(pt["directional_bias"].min()), abs(pt["directional_bias"].max()))

    # 08a: rows=models, cols=topics
    mean_m = pt.groupby(["provider", "category"])["directional_bias"].mean().reset_index()
    std_m  = pt.groupby(["provider", "category"])["directional_bias"].std().reset_index()
    piv_m  = mean_m.pivot(index="provider",  columns="category",
                          values="directional_bias").reindex(index=PROVIDER_ORDER, columns=top_topics)
    spiv_m = std_m.pivot( index="provider",  columns="category",
                          values="directional_bias").reindex(index=PROVIDER_ORDER, columns=top_topics)

    fig, ax = plt.subplots(figsize=(10, 4))
    _draw_topic_heatmap(ax, piv_m, _make_topic_annot(piv_m, spiv_m), vmax_global,
                        col_lbls=_topic_col_labels(pt, top_topics),
                        row_lbls=[PROVIDER_LABELS[p] for p in PROVIDER_ORDER],
                        show_cbar=True, show_ylabel=True,
                        ylabel="Model", title="Primary Topic Directional Bias by Model")
    fig.suptitle("(Averaged across Prompt Styles, ±SD shown in annotations)",
                 fontsize=13, y=1.04)
    plt.tight_layout()
    fig.savefig(OUT / "08a_topic_bias_by_model.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("  ✓ 08a_topic_bias_by_model.png")

    # 08b: rows=prompts, cols=topics
    mean_p = pt.groupby(["prompt_style", "category"])["directional_bias"].mean().reset_index()
    std_p  = pt.groupby(["prompt_style", "category"])["directional_bias"].std().reset_index()
    piv_p  = mean_p.pivot(index="prompt_style", columns="category",
                          values="directional_bias").reindex(index=PROMPT_ORDER, columns=top_topics)
    spiv_p = std_p.pivot( index="prompt_style", columns="category",
                          values="directional_bias").reindex(index=PROMPT_ORDER, columns=top_topics)

    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_topic_heatmap(ax, piv_p, _make_topic_annot(piv_p, spiv_p), vmax_global,
                        col_lbls=_topic_col_labels(pt, top_topics),
                        row_lbls=[PROMPT_LABELS[p] for p in PROMPT_ORDER],
                        show_cbar=True, show_ylabel=True,
                        ylabel="Prompt Style", title="Primary Topic Directional Bias by Prompt Style")
    fig.suptitle("(Averaged across Models, ±SD shown in annotations)",
                 fontsize=13, y=1.04)
    plt.tight_layout()
    fig.savefig(OUT / "08b_topic_bias_by_prompt.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("  ✓ 08b_topic_bias_by_prompt.png")

# ============================================================================
# FIGURE 09 — Raw directional bias (one figure per metric)
# ============================================================================

_METRICS_09 = {
    "avg_word_length": {
        "title":  "Average Word Length Directional Bias by Model and Prompt Style",
        "ylabel": "Directional Bias (chars/word)\n← Shorter | Longer →",
    },
    "text_length": {
        "title":  "Post Character Length Directional Bias by Model and Prompt Style",
        "ylabel": "Directional Bias (chars)\n← Shorter | Longer Posts →",
    },
    "polarization_score": {
        "title":  "Polarization Directional Bias by Model and Prompt Style",
        "ylabel": "Directional Bias\n← Less | More Polarized →",
    },
    "sentiment_polarity": {
        "title":  "Sentiment Polarity Directional Bias by Model and Prompt Style",
        "ylabel": "Directional Bias\n← More Negative | More Positive →",
    },
    "toxicity": {
        "title":  "Toxicity Directional Bias by Model and Prompt Style",
        "ylabel": "Directional Bias\n← Less | More Toxic →",
    },
    # Style (binary)
    "has_emoji": {
        "title":  "Emoji Usage Directional Bias by Model and Prompt Style",
        "ylabel": "Directional Bias (proportion)\n← Fewer | More Posts with Emoji →",
    },
    "has_hashtag": {
        "title":  "Hashtag Usage Directional Bias by Model and Prompt Style",
        "ylabel": "Directional Bias (proportion)\n← Fewer | More Posts with Hashtag →",
    },
    "has_mention": {
        "title":  "Mention Usage Directional Bias by Model and Prompt Style",
        "ylabel": "Directional Bias (proportion)\n← Fewer | More Posts with Mention →",
    },
    "has_url": {
        "title":  "URL Usage Directional Bias by Model and Prompt Style",
        "ylabel": "Directional Bias (proportion)\n← Fewer | More Posts with URL →",
    },
    # Author metadata
    "user_followers_count": {
        "title":  "Author Followers Directional Bias by Model and Prompt Style",
        "ylabel": "Directional Bias (followers)\n← Fewer | More Followers →",
    },
    "user_friends_count": {
        "title":  "Author Following Count Directional Bias by Model and Prompt Style",
        "ylabel": "Directional Bias (following)\n← Fewer | More Following →",
    },
    "user_statuses_count": {
        "title":  "Author Tweet Count Directional Bias by Model and Prompt Style",
        "ylabel": "Directional Bias (tweets)\n← Fewer | More Tweets →",
    },
    "user_favourites_count": {
        "title":  "Author Likes Given Directional Bias by Model and Prompt Style",
        "ylabel": "Directional Bias (likes given)\n← Fewer | More Likes Given →",
    },
    # Post engagement metadata
    "favorite_count": {
        "title":  "Post Likes Directional Bias by Model and Prompt Style",
        "ylabel": "Directional Bias (likes)\n← Fewer | More Likes →",
    },
    "retweet_count": {
        "title":  "Post Retweets Directional Bias by Model and Prompt Style",
        "ylabel": "Directional Bias (retweets)\n← Fewer | More Retweets →",
    },
    "retweeted": {
        "title":  "Post Is-Retweeted Directional Bias by Model and Prompt Style",
        "ylabel": "Directional Bias\n← Less | More Often Retweeted →",
    },
}


def _plot_09_single_metric(feature, minfo, context_level="none"):
    data = _load_metric_bias(feature, context_level)
    if data.empty:
        return None

    all_vals = data["directional_bias"].dropna()
    max_abs  = max(abs(all_vals.min()), abs(all_vals.max())) if len(all_vals) else 1.0

    agg_m = data.groupby(["provider", "prompt_style"])["directional_bias"].mean().reset_index()
    piv_m = agg_m.pivot(index="prompt_style", columns="provider", values="directional_bias")
    piv_m = piv_m.reindex(index=PROMPT_ORDER, columns=PROVIDER_ORDER)

    std_col_vals = piv_m.std(axis=1).values
    std_row_vals = piv_m.std(axis=0).values
    overall_std  = float(np.nanstd(piv_m.values))

    avg_col             = piv_m.mean(axis=1)
    avg_row             = pd.Series(piv_m.mean(axis=0), name="Average")
    piv_m               = pd.concat([piv_m, avg_row.to_frame().T])
    avg_col["Average"]  = avg_col.mean()
    piv_m["Average"]    = avg_col

    short = {p: PROVIDER_LABELS[p] for p in PROVIDER_ORDER}
    short["Average"] = "Average"
    piv_m.columns = [short.get(p, p) for p in piv_m.columns]
    piv_m.index   = [PROMPT_LABELS.get(p, p) for p in piv_m.index]

    n_rows, n_cols = piv_m.shape
    annot = np.empty_like(piv_m, dtype=object)
    for i in range(n_rows):
        for j in range(n_cols):
            val = piv_m.iloc[i, j]
            if pd.isna(val):
                annot[i, j] = ""
            elif i == n_rows - 1 and j == n_cols - 1:
                annot[i, j] = f"{val:.3f}\n±{overall_std:.3f}"
            elif i == n_rows - 1:
                annot[i, j] = f"{val:.3f}\n±{std_row_vals[j]:.3f}"
            elif j == n_cols - 1:
                annot[i, j] = f"{val:.3f}\n±{std_col_vals[i]:.3f}"
            else:
                annot[i, j] = f"{val:.3f}"

    fig, ax = plt.subplots(figsize=(10, 8.05))
    sns.heatmap(piv_m, annot=annot, fmt="", cmap=CMAP_DIVG,
                center=0, vmin=-max_abs, vmax=max_abs, ax=ax,
                cbar=True,
                cbar_kws={"label": minfo["ylabel"]},
                linewidths=0.5, linecolor="gray", annot_kws={"fontsize": 15})

    if ax.collections:
        cbar = ax.collections[0].colorbar
        if cbar:
            cbar.ax.tick_params(labelsize=16)
            cbar.set_label(minfo["ylabel"], fontsize=17, fontweight="bold")

    ax.axhline(y=n_rows - 1, color="black", linewidth=2.5)
    ax.axvline(x=n_cols - 1, color="black", linewidth=2.5)
    ax.set_title(minfo["title"], fontweight="bold", fontsize=16, pad=10)
    ax.set_xlabel("Model",        fontsize=18, fontweight="bold")
    ax.set_ylabel("Prompt Style", fontsize=18, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=15)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center", fontsize=15)

    plt.tight_layout()
    return fig


def plot_09_raw_bias_heatmaps(context_level="none"):
    print("\n" + "="*70)
    print("FIGURES 09a–p — Raw directional bias (one figure per metric)")
    print("="*70)

    labels = {
        "avg_word_length":      "09a",
        "text_length":          "09b",
        "polarization_score":   "09c",
        "sentiment_polarity":   "09d",
        "toxicity":             "09e",
        "has_emoji":            "09f",
        "has_hashtag":          "09g",
        "has_mention":          "09h",
        "has_url":              "09i",
        "user_followers_count": "09j",
        "user_friends_count":   "09k",
        "user_statuses_count":  "09l",
        "user_favourites_count":"09m",
        "favorite_count":       "09n",
        "retweet_count":        "09o",
        "retweeted":            "09p",
    }
    for feature, minfo in _METRICS_09.items():
        fig = _plot_09_single_metric(feature, minfo, context_level)
        if fig is None:
            print(f"  skipped {feature} (no data)")
            continue
        tag   = labels[feature]
        fname = f"{tag}_raw_bias_{feature}.png"
        fig.savefig(OUT / fname, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"  ✓ {fname}")

        data    = _load_metric_bias(feature, context_level)
        mean_df = data.groupby(["provider", "prompt_style"])["directional_bias"].mean().reset_index()
        std_df  = data.groupby(["provider", "prompt_style"])["directional_bias"].std().reset_index()
        mean_df = mean_df.rename(columns={"directional_bias": "mean_bias"})
        std_df  = std_df.rename(columns={"directional_bias": "std_bias"})
        csv_df  = mean_df.merge(std_df, on=["provider", "prompt_style"])
        csv_df["provider_label"] = csv_df["provider"].map(PROVIDER_LABELS)
        csv_df.to_csv(OUT / f"{tag}_raw_bias_{feature}.csv", index=False)
        print(f"  ✓ {tag}_raw_bias_{feature}.csv")

# ============================================================================
# FIGURE 10 — Demographic bias by model (one figure per demographic variable)
# ============================================================================

def plot_10_demographic_by_model(context_level="none"):
    print("\n" + "="*70)
    print("FIGURE 10 — Demographic bias by model")
    print("="*70)

    dir_bias = _load_demo_bias(context_level)

    for feature, finfo in _DEMO_FEATURE_CONFIG.items():
        fdata = dir_bias[(dir_bias["feature"] == feature) &
                         (dir_bias["category"] != "mean")]
        if len(fdata) == 0:
            print(f"  skipped {feature} (no data)")
            continue

        avail      = _resolve_categories(finfo, fdata["category"].unique())
        avail      = [c for c in avail if c in fdata["category"].unique()]
        pool_pct   = fdata.groupby("category")["prop_pool"].mean()
        col_labels = [
            f"{_cat_label(c)}\n({pool_pct.get(c, 0) * 100:.1f}%)"
            for c in avail
        ]

        all_vals = fdata["directional_bias"].dropna()
        max_abs  = max(abs(all_vals.min()), abs(all_vals.max())) if len(all_vals) else 1.0
        max_abs  = max(max_abs, 1e-6)

        fig, axes = plt.subplots(1, len(PROVIDER_ORDER), figsize=(20, 8.05))

        for idx, provider in enumerate(PROVIDER_ORDER):
            ax        = axes[idx]
            prov_data = fdata[fdata["provider"] == provider]

            mean_agg = prov_data.groupby(["prompt_style", "category"])["directional_bias"].mean().reset_index()
            std_agg  = prov_data.groupby(["prompt_style", "category"])["directional_bias"].std().reset_index()

            piv_m = mean_agg.pivot(index="prompt_style", columns="category", values="directional_bias")
            piv_s = std_agg.pivot( index="prompt_style", columns="category", values="directional_bias")

            piv_m = piv_m.reindex(index=PROMPT_ORDER, columns=avail)
            piv_s = piv_s.reindex(index=PROMPT_ORDER, columns=avail)

            std_col_vals = piv_m.std(axis=1).values
            std_row_vals = piv_m.std(axis=0).values
            overall_std  = float(np.nanstd(piv_m.values))

            avg_col            = piv_m.mean(axis=1)
            avg_row            = pd.Series(piv_m.mean(axis=0), name="Average")
            piv_m              = pd.concat([piv_m, avg_row.to_frame().T])
            avg_col["Average"] = avg_col.mean()
            piv_m["Average"]   = avg_col

            piv_m.columns = col_labels + ["Average"]
            piv_m.index   = [PROMPT_LABELS.get(p, p) for p in piv_m.index]

            n_rows, n_cols = piv_m.shape
            annot = np.empty_like(piv_m, dtype=object)
            for i in range(n_rows):
                for j in range(n_cols):
                    val = piv_m.iloc[i, j]
                    if pd.isna(val):
                        annot[i, j] = ""
                    elif i == n_rows - 1 and j == n_cols - 1:
                        annot[i, j] = f"{val:.3f}\n±{overall_std:.3f}"
                    elif i == n_rows - 1:
                        annot[i, j] = f"{val:.3f}\n±{std_row_vals[j]:.3f}"
                    elif j == n_cols - 1:
                        annot[i, j] = f"{val:.3f}\n±{std_col_vals[i]:.3f}"
                    else:
                        annot[i, j] = f"{val:.3f}"

            show_cbar  = (idx == len(PROVIDER_ORDER) - 1)
            cbar_label = "Directional Bias\n← Under | Over-represented →"
            sns.heatmap(piv_m, annot=annot, fmt="", cmap=CMAP_DIVG,
                        center=0, vmin=-max_abs, vmax=max_abs, ax=ax,
                        cbar=show_cbar,
                        cbar_kws={"label": cbar_label} if show_cbar else {},
                        linewidths=0.5, linecolor="gray", annot_kws={"fontsize": 15})

            if show_cbar and ax.collections:
                cbar = ax.collections[0].colorbar
                if cbar:
                    cbar.ax.tick_params(labelsize=16)
                    cbar.set_label(cbar_label, fontsize=17, fontweight="bold")

            ax.axhline(y=n_rows - 1, color="black", linewidth=2.5)
            ax.axvline(x=n_cols - 1, color="black", linewidth=2.5)
            ax.set_title(PROVIDER_LABELS[provider], fontweight="bold", fontsize=20, pad=10)
            ax.set_xlabel("Category", fontsize=18, fontweight="bold")
            ax.set_ylabel("Prompt Style" if idx == 0 else "", fontsize=18, fontweight="bold")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=15)
            if idx > 0:
                ax.set_yticklabels([])
            else:
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center", fontsize=15)

        fig.suptitle(
            f"{finfo['display_name']} — Directional Bias by Model and Prompt Style",
            fontweight="bold", fontsize=20, y=1.02)
        plt.tight_layout(w_pad=3)

        slug  = feature.replace("author_", "")
        fname = f"10_demo_bias_{slug}_by_model.png"
        fig.savefig(OUT / fname, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"  ✓ {fname}")

        mean_df = fdata.groupby(["provider", "prompt_style", "category"])["directional_bias"].mean().reset_index()
        std_df  = fdata.groupby(["provider", "prompt_style", "category"])["directional_bias"].std().reset_index()
        mean_df = mean_df.rename(columns={"directional_bias": "mean_bias"})
        std_df  = std_df.rename(columns={"directional_bias": "std_bias"})
        csv_df  = mean_df.merge(std_df, on=["provider", "prompt_style", "category"])
        csv_df["provider_label"] = csv_df["provider"].map(PROVIDER_LABELS)
        csv_df.to_csv(OUT / f"10_demo_bias_{slug}_by_model.csv", index=False)
        print(f"  ✓ 10_demo_bias_{slug}_by_model.csv")

# ============================================================================
# POOL DISTRIBUTIONS CSV
# ============================================================================

def save_pool_distributions(context_level="none"):
    print("\n" + "="*70)
    print("SAVING pool_distributions.csv")
    print("="*70)

    df   = pd.read_csv(DIR_BIAS_CSV)
    df   = _filter_context(df, context_level)
    rows = []

    cat  = df[df["feature_type"] == "categorical"]
    for (feature, category), grp in cat.groupby(["feature", "category"]):
        rows.append({
            "feature": feature, "feature_type": "categorical",
            "category": category,
            "pool_proportion":     round(grp["prop_pool"].mean(), 6),
            "pool_proportion_std": round(grp["prop_pool"].std(),  6),
            "pool_mean": np.nan, "pool_mean_std": np.nan,
        })

    cont = df[df["feature_type"] == "continuous"]
    for feature, grp in cont.groupby("feature"):
        rows.append({
            "feature": feature, "feature_type": "continuous",
            "category": np.nan,
            "pool_proportion": np.nan, "pool_proportion_std": np.nan,
            "pool_mean":     round(grp["mean_pool"].mean(), 6),
            "pool_mean_std": round(grp["mean_pool"].std(),  6),
        })

    pd.DataFrame(rows).sort_values(
        ["feature_type", "feature", "category"]
    ).to_csv(OUT / "pool_distributions.csv", index=False)
    print("  ✓ pool_distributions.csv")

# ============================================================================
# FIGURES 11–13 — Metadata context-level experiment
# ============================================================================

_AUTHOR_META_FEATURES = [
    "user_followers_count", "user_friends_count",
    "user_statuses_count",  "user_favourites_count",
]
_POST_META_FEATURES = ["favorite_count", "retweet_count", "retweeted"]

_CONTEXT_DISPLAY = {
    "none":        "None\n(text only)",
    "author":      "Author\nmetadata",
    "post":        "Post\nmetadata",
    "author_post": "Author+Post\nmetadata",
}

_META_FEATURE_DISPLAY = {
    "user_followers_count":  "Followers",
    "user_friends_count":    "Following",
    "user_statuses_count":   "Tweet count",
    "user_favourites_count": "Likes given",
    "favorite_count":        "Post likes",
    "retweet_count":         "Retweets",
    "retweeted":             "Is retweeted",
}


def _meta_bias_pivot(all_df, features, context_levels):
    """Build pivot (rows = feature × provider, cols = context_level) of mean R²."""
    sub = all_df[
        all_df["feature"].isin(features) &
        all_df["context_level"].isin(context_levels)
    ].copy()
    agg = sub.groupby(["feature", "provider", "context_level"])["r_squared"].mean().reset_index()
    return agg


def plot_11_author_metadata_bias(all_df):
    print("\n" + "="*70)
    print("FIGURE 11 — Author metadata bias by context level × model")
    print("="*70)

    context_levels = ["author", "author_post"]
    available = [f for f in _AUTHOR_META_FEATURES if f in all_df["feature"].unique()]
    if not available:
        print("  No author metadata features found — skipping.")
        return

    agg = _meta_bias_pivot(all_df, available, context_levels)
    if agg.empty:
        print("  No data for author/author_post context levels — skipping.")
        return

    pivot = agg.pivot_table(
        index=["feature", "provider"],
        columns="context_level",
        values="r_squared",
        aggfunc="mean",
    ).reindex(columns=context_levels)

    row_labels = [
        f"{_META_FEATURE_DISPLAY.get(f, f)}\n({PROVIDER_LABELS.get(p, p).split()[0]})"
        for f, p in pivot.index
    ]
    col_labels = [_CONTEXT_DISPLAY.get(c, c) for c in context_levels]

    fig, ax = plt.subplots(figsize=(6, max(4, len(pivot) * 0.55)))
    sns.heatmap(pivot.values, annot=True, fmt=".3f", cmap=CMAP_WR,
                vmin=0, vmax=max(0.01, pivot.values[~pd.isna(pivot.values)].max()),
                ax=ax, cbar_kws={"label": "R²"},
                linewidths=0.5, linecolor="lightgray", annot_kws={"fontsize": 12})
    ax.set_xticks(np.arange(len(col_labels)) + 0.5)
    ax.set_xticklabels(col_labels, fontsize=12)
    ax.set_yticks(np.arange(len(row_labels)) + 0.5)
    ax.set_yticklabels(row_labels, fontsize=11, rotation=0, ha="right")
    ax.set_title("Author Metadata Bias by Context Level\n(R², mean across prompt styles)",
                 fontweight="bold", fontsize=13, pad=10)
    ax.set_xlabel("Context level", fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT / "11_author_metadata_bias_by_context.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("  ✓ 11_author_metadata_bias_by_context.png")
    agg.to_csv(OUT / "11_author_metadata_bias_data.csv", index=False)
    print("  ✓ 11_author_metadata_bias_data.csv")


def plot_12_post_metadata_bias(all_df):
    print("\n" + "="*70)
    print("FIGURE 12 — Post metadata bias by context level × model")
    print("="*70)

    context_levels = ["post", "author_post"]
    available = [f for f in _POST_META_FEATURES if f in all_df["feature"].unique()]
    if not available:
        print("  No post metadata features found — skipping.")
        return

    agg = _meta_bias_pivot(all_df, available, context_levels)
    if agg.empty:
        print("  No data for post/author_post context levels — skipping.")
        return

    pivot = agg.pivot_table(
        index=["feature", "provider"],
        columns="context_level",
        values="r_squared",
        aggfunc="mean",
    ).reindex(columns=context_levels)

    row_labels = [
        f"{_META_FEATURE_DISPLAY.get(f, f)}\n({PROVIDER_LABELS.get(p, p).split()[0]})"
        for f, p in pivot.index
    ]
    col_labels = [_CONTEXT_DISPLAY.get(c, c) for c in context_levels]

    fig, ax = plt.subplots(figsize=(6, max(4, len(pivot) * 0.55)))
    vals = pivot.values
    vmax = max(0.01, vals[~np.isnan(vals.astype(float))].max()) if vals.size > 0 else 0.01
    sns.heatmap(vals, annot=True, fmt=".3f", cmap=CMAP_WR,
                vmin=0, vmax=vmax,
                ax=ax, cbar_kws={"label": "R²"},
                linewidths=0.5, linecolor="lightgray", annot_kws={"fontsize": 12})
    ax.set_xticks(np.arange(len(col_labels)) + 0.5)
    ax.set_xticklabels(col_labels, fontsize=12)
    ax.set_yticks(np.arange(len(row_labels)) + 0.5)
    ax.set_yticklabels(row_labels, fontsize=11, rotation=0, ha="right")
    ax.set_title("Post Metadata Bias by Context Level\n(R², mean across prompt styles)",
                 fontweight="bold", fontsize=13, pad=10)
    ax.set_xlabel("Context level", fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT / "12_post_metadata_bias_by_context.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("  ✓ 12_post_metadata_bias_by_context.png")
    agg.to_csv(OUT / "12_post_metadata_bias_data.csv", index=False)
    print("  ✓ 12_post_metadata_bias_data.csv")


def plot_13_context_level_delta(all_df):
    """Delta heatmap: R² change vs none baseline, for each feature × provider × context_level."""
    print("\n" + "="*70)
    print("FIGURE 13 — Δ bias vs none baseline (all features, all context levels)")
    print("="*70)

    non_none = [c for c in all_df["context_level"].unique() if c != "none"]
    if not non_none:
        print("  Only 'none' context level present — skipping.")
        return

    none_df = all_df[all_df["context_level"] == "none"].copy()
    none_base = none_df.groupby(["feature", "provider"])["r_squared"].mean()

    rows = []
    for cl in sorted(non_none):
        cl_df = all_df[all_df["context_level"] == cl]
        agg = cl_df.groupby(["feature", "provider"])["r_squared"].mean()
        delta = (agg - none_base).dropna().reset_index()
        delta.columns = ["feature", "provider", "delta_r2"]
        delta["context_level"] = cl
        rows.append(delta)

    if not rows:
        print("  No data — skipping.")
        return

    delta_all = pd.concat(rows, ignore_index=True)
    delta_all["feature_display"]   = delta_all["feature"].apply(
        lambda f: _META_FEATURE_DISPLAY.get(f, fmt(f))
    )
    delta_all["provider_display"]  = delta_all["provider"].map(PROVIDER_LABELS)
    delta_all["row_label"] = (delta_all["feature_display"] + "\n("
                              + delta_all["provider_display"].str.split().str[0] + ")")
    delta_all["col_label"] = delta_all["context_level"].map(
        lambda c: _CONTEXT_DISPLAY.get(c, c)
    )

    pivot = delta_all.pivot_table(
        index="row_label", columns="col_label", values="delta_r2", aggfunc="mean"
    )
    col_order = [_CONTEXT_DISPLAY.get(c, c) for c in sorted(non_none)
                 if _CONTEXT_DISPLAY.get(c, c) in pivot.columns]
    pivot = pivot.reindex(columns=col_order)

    vals = pivot.values.flatten()
    vals = vals[~pd.isna(vals)]
    max_abs = max(abs(vals).max(), 1e-6) if len(vals) > 0 else 0.1

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 2.5),
                                    max(6, len(pivot) * 0.45)))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap=CMAP_DIVG,
                center=0, vmin=-max_abs, vmax=max_abs, ax=ax,
                cbar_kws={"label": "ΔR² (vs. none baseline)"},
                linewidths=0.5, linecolor="lightgray", annot_kws={"fontsize": 11})
    ax.set_title("Bias Change vs. No-Context Baseline (ΔR²)\n"
                 "(red = more bias with metadata visible; blue = less bias)",
                 fontweight="bold", fontsize=13, pad=10)
    ax.set_xlabel("Context level", fontsize=12)
    ax.set_ylabel("Feature / Model", fontsize=12)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(OUT / "13_context_level_delta_heatmap.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("  ✓ 13_context_level_delta_heatmap.png")
    delta_all.to_csv(OUT / "13_context_level_delta_data.csv", index=False)
    print("  ✓ 13_context_level_delta_data.csv")


# ============================================================================
# FIGURES 14–15 — Metadata directional bias across context levels
# ============================================================================

def _metadata_directional_pivot(features, context_levels):
    """Load directional_bias_data.csv and return a pivot of mean directional bias.

    Rows: (feature, provider) — Cols: context_level
    Only uses rows where feature_type is numerical/binary and category == 'mean'.
    Values are averaged across prompt styles.
    """
    dir_df = pd.read_csv(DIR_BIAS_CSV)
    if "context_level" not in dir_df.columns:
        dir_df["context_level"] = "none"

    available = [f for f in features if f in dir_df["feature"].unique()]
    if not available:
        return None, []

    sub = dir_df[
        dir_df["feature"].isin(available) &
        dir_df["feature_type"].isin(["numerical", "binary"]) &
        (dir_df["category"] == "mean") &
        dir_df["context_level"].isin(context_levels)
    ].copy()

    if sub.empty:
        return None, []

    agg = sub.groupby(["feature", "provider", "context_level"])["directional_bias"].mean().reset_index()
    pivot = agg.pivot_table(
        index=["feature", "provider"],
        columns="context_level",
        values="directional_bias",
        aggfunc="mean",
    ).reindex(columns=context_levels)
    return pivot, agg


def _draw_directional_heatmap(pivot, row_labels, col_labels, title, cbar_label, out_path):
    vals = pivot.values.astype(float)
    max_abs = max(np.nanmax(np.abs(vals)), 1e-6)

    fig, ax = plt.subplots(figsize=(max(6, len(col_labels) * 2), max(4, len(pivot) * 0.55)))
    sns.heatmap(vals, annot=True, fmt=".2g", cmap=CMAP_DIVG,
                center=0, vmin=-max_abs, vmax=max_abs,
                ax=ax, cbar_kws={"label": cbar_label},
                linewidths=0.5, linecolor="lightgray", annot_kws={"fontsize": 12})
    if ax.collections:
        cbar = ax.collections[0].colorbar
        if cbar:
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label(cbar_label, fontsize=13, fontweight="bold")
    ax.set_xticks(np.arange(len(col_labels)) + 0.5)
    ax.set_xticklabels(col_labels, fontsize=12)
    ax.set_yticks(np.arange(len(row_labels)) + 0.5)
    ax.set_yticklabels(row_labels, fontsize=11, rotation=0, ha="right")
    ax.set_title(title, fontweight="bold", fontsize=13, pad=10)
    ax.set_xlabel("Context level", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()


def _plot_metadata_directional_per_feature(fig_num, features, context_levels, title_prefix, out_prefix):
    """Create one heatmap per feature (providers × context levels)."""
    dir_df = pd.read_csv(DIR_BIAS_CSV)
    if "context_level" not in dir_df.columns:
        dir_df["context_level"] = "none"

    available = [f for f in features if f in dir_df["feature"].unique()]
    if not available:
        print("  No data — skipping.")
        return pd.DataFrame()

    sub = dir_df[
        dir_df["feature"].isin(available) &
        dir_df["feature_type"].isin(["numerical", "binary"]) &
        (dir_df["category"] == "mean") &
        dir_df["context_level"].isin(context_levels)
    ].copy()

    if sub.empty:
        print("  No data — skipping.")
        return pd.DataFrame()

    agg = sub.groupby(["feature", "provider", "context_level"])["directional_bias"].mean().reset_index()
    col_labels = [_CONTEXT_DISPLAY.get(c, c) for c in context_levels]

    letters = "abcdefghijklmnopqrstuvwxyz"
    for i, feat in enumerate(available):
        feat_agg = agg[agg["feature"] == feat]
        pivot = feat_agg.pivot_table(
            index="provider", columns="context_level",
            values="directional_bias", aggfunc="mean",
        ).reindex(index=PROVIDER_ORDER, columns=context_levels)

        row_labels = [PROVIDER_LABELS.get(p, p) for p in pivot.index]
        feat_display = _META_FEATURE_DISPLAY.get(feat, feat)
        suffix = letters[i]

        _draw_directional_heatmap(
            pivot, row_labels, col_labels,
            title=f"{title_prefix}: {feat_display}\n"
                  "(mean recommended − mean pool, averaged across prompt styles;\n"
                  " red = LLM selects higher values, blue = selects lower)",
            cbar_label="Mean Rec − Mean Pool",
            out_path=OUT / f"{out_prefix}{suffix}_{feat}.png",
        )
        slug = f"{out_prefix}{suffix}_{feat}.png"
        print(f"  ✓ {slug}")

    return agg


def plot_14_author_metadata_directional(all_df):
    print("\n" + "="*70)
    print("FIGURE 14 — Author metadata directional bias by context level (per feature)")
    print("="*70)

    context_levels = [c for c in ["none", "author", "post", "author_post"]
                      if c in all_df["context_level"].unique()]
    agg = _plot_metadata_directional_per_feature(
        14, _AUTHOR_META_FEATURES, context_levels,
        "Author Metadata Directional Bias", "14",
    )
    if not agg.empty:
        agg.to_csv(OUT / "14_author_metadata_directional_data.csv", index=False)
        print("  ✓ 14_author_metadata_directional_data.csv")


def plot_15_post_metadata_directional(all_df):
    print("\n" + "="*70)
    print("FIGURE 15 — Post metadata directional bias by context level (per feature)")
    print("="*70)

    context_levels = [c for c in ["none", "author", "post", "author_post"]
                      if c in all_df["context_level"].unique()]
    agg = _plot_metadata_directional_per_feature(
        15, _POST_META_FEATURES, context_levels,
        "Post Engagement Directional Bias", "15",
    )
    if not agg.empty:
        agg.to_csv(OUT / "15_post_metadata_directional_data.csv", index=False)
        print("  ✓ 15_post_metadata_directional_data.csv")


# ============================================================================
# MAIN
# ============================================================================

def main():
    global OUT
    import shutil

    print("=" * 70)
    print("GENERATING ALL PAPER PLOTS")
    print(f"Output → {OUT}")
    print("=" * 70)

    missing = [p for p in [SUMMARY_CSV, DIR_BIAS_CSV] if not p.exists()]
    if missing:
        print("\nERROR — missing input files:")
        for p in missing:
            print(f"  {p}")
        return

    # All context levels for new figures
    all_df = load_summary_all()
    n_cl = all_df["context_level"].nunique() if "context_level" in all_df.columns else 1
    context_levels = sorted(all_df["context_level"].unique()) if "context_level" in all_df.columns else ["none"]
    print(f"Loaded {len(all_df)} total rows across {n_cl} context level(s): {context_levels}")

    # Restrict PROVIDER_ORDER to providers actually present in the data
    dir_df = pd.read_csv(DIR_BIAS_CSV)
    providers_in_data = set(dir_df["provider"].unique())
    PROVIDER_ORDER[:] = [p for p in _PROVIDER_PREFERRED_ORDER if p in providers_in_data]
    print(f"Providers in data: {PROVIDER_ORDER}")

    root_out = OUT
    use_subfolders = n_cl > 1

    # Figures 01–10: one set per context level, each in its own subfolder (or root if only one)
    for cl in context_levels:
        if use_subfolders:
            OUT = root_out / cl
            if OUT.exists():
                shutil.rmtree(OUT)
            OUT.mkdir(parents=True, exist_ok=True)
            print(f"\n{'='*70}")
            print(f"  Context level: '{cl}'  →  {OUT}")
            print(f"{'='*70}")
        else:
            OUT = root_out  # already created by _init_paths

        comp_df = load_summary(context_level=cl)
        print(f"  Loaded {len(comp_df)} rows for context_level='{cl}'")

        save_pool_distributions(context_level=cl)
        plot_01_aggregated_bar(comp_df)
        plot_02_bias_by_prompt(comp_df)
        plot_03_normalized_bias(comp_df)
        plot_04_demographics(context_level=cl)
        plot_05_content_safety_heatmap(context_level=cl)
        plot_06_content_safety_bars(context_level=cl)
        if IMPORTANCE_CSV.exists():
            plot_07_feature_importance(context_level=cl)
        else:
            print("\nFIGURE 07 skipped — feature_importance_data.csv not found")
        plot_08_topic_heatmaps(context_level=cl)
        plot_09_raw_bias_heatmaps(context_level=cl)
        plot_10_demographic_by_model(context_level=cl)

    # Figures 11–15: cross-context comparisons, always in root
    OUT = root_out
    if n_cl > 1:
        plot_11_author_metadata_bias(all_df)
        plot_12_post_metadata_bias(all_df)
        plot_13_context_level_delta(all_df)
        plot_14_author_metadata_directional(all_df)
        plot_15_post_metadata_directional(all_df)
    else:
        print("\nFIGURES 11–15 skipped — only one context level in data")

    print("\n" + "=" * 70)
    print("ALL DONE")
    print("=" * 70)
    print(f"Root: {root_out}")
    for f in sorted(root_out.iterdir()):
        if f.is_dir():
            n = sum(1 for _ in f.glob("*.png"))
            print(f"  {f.name}/  ({n} figures)")
        else:
            print(f"  {f.name:<65} {f.stat().st_size // 1024:>4} KB")


if __name__ == "__main__":
    main()
