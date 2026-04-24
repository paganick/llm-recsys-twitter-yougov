#!/usr/bin/env python3
"""
Compute bias metrics for the Twitter/X LLM Recommendation Bias Study.

Analyzes bias across all available features, 3 LLM providers (Anthropic /
OpenAI / Gemini), 6 prompt styles, and up to 4 context levels
(none / author / post / author_post), giving up to 72 conditions total.

Reads per-condition post_level_data.csv files from outputs/experiments/ and
produces three aggregated output files in analysis_outputs/:
  - pool_vs_recommended_summary.csv   (Cohen's d / Cramér's V per feature)
  - directional_bias_data.csv         (directional bias per category)
  - feature_importance_data.csv       (Random Forest SHAP + AUROC)

Usage
-----
    python compute_bias_metrics.py
    python compute_bias_metrics.py --experiments-dir outputs/experiments
"""

import argparse
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

try:
    import shap as _shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================

# Features grouped by category — edit here to add/remove features
FEATURES = {
    # Ground-truth survey demographics (all YouGov fields)
    "demographics": [
        "author_gender",
        "author_partisanship",
        "author_ideology",
        "author_race",
        "author_age",
        "author_education",
        "author_income",
        "author_marital_status",
        "author_religiosity",
    ],
    # Computed NLP features
    "sentiment": [
        "sentiment_polarity",
        "sentiment_subjectivity",
    ],
    # Style indicators (pre-computed from tweet text)
    "style": [
        "has_emoji",
        "has_hashtag",
        "has_mention",
        "has_url",
    ],
    # Text metrics (pre-computed from tweet text)
    "text_metrics": [
        "avg_word_length",
    ],
    # Content (NLP-computed)
    "content": [
        "polarization_score",
        "primary_topic",
    ],
    # Safety (NLP-computed)
    "toxicity": [
        "toxicity",
    ],
    # Author social-graph metadata (exposed to LLM in author/author_post context levels)
    "author_metadata": [
        "user_followers_count",
        "user_friends_count",
        "user_statuses_count",
        "user_favourites_count",
    ],
    # Post engagement metadata (exposed to LLM in post/author_post context levels)
    "post_metadata": [
        "favorite_count",
        "retweet_count",
        "retweeted",
    ],
    # --- Extended features (Phase 2) ---
    # Uncomment to include once data contains these columns:
    #
    # "text_metrics_extended": [
    #     "text_length",
    #     "word_count",
    # ],
    # "tweet_type": [
    #     "is_reply",
    #     "is_retweet",
    #     "is_quote",
    # ],
}

FEATURE_TYPES = {
    # Demographics (categorical)
    "author_gender":       "categorical",
    "author_partisanship": "categorical",
    "author_ideology":     "categorical",
    "author_race":         "categorical",
    # Demographics extended (all categorical — age is binned by prepare_dataset.py)
    "author_age":              "categorical",
    "author_education":        "categorical",
    "author_income":           "categorical",
    "author_marital_status":   "categorical",
    "author_religiosity":      "categorical",
    # Sentiment (numerical)
    "sentiment_polarity":    "numerical",
    "sentiment_subjectivity":"numerical",
    # Style (binary)
    "has_emoji":   "binary",
    "has_hashtag": "binary",
    "has_mention": "binary",
    "has_url":     "binary",
    # Text metrics (numerical)
    "avg_word_length": "numerical",
    "text_length":     "numerical",
    "word_count":      "numerical",
    # Content (mixed)
    "polarization_score": "numerical",
    "primary_topic":      "categorical",
    # Safety (numerical)
    "toxicity": "numerical",
    # Tweet type (binary)
    "is_reply":   "binary",
    "is_retweet": "binary",
    "is_quote":   "binary",
    # User account (numerical / binary)
    "user_followers_count":  "numerical",
    "user_friends_count":    "numerical",
    "user_statuses_count":   "numerical",
    "user_favourites_count": "numerical",
    "user_verified":         "binary",
    "user_account_age_days": "numerical",
    "engagement_score":      "numerical",
    # Post engagement metadata
    "favorite_count": "numerical",
    "retweet_count":  "numerical",
    "retweeted":      "binary",
}

# Category ordering hints for features where order is meaningful.
# Only used for documentation / downstream consumers — compute_directional_bias
# derives the actual category list directly from the data.
CATEGORY_ORDER_HINTS = {
    "author_ideology": ["very liberal", "liberal", "moderate", "conservative", "very conservative"],
}

FEATURE_DISPLAY_NAMES = {
    "author_gender":       "Author: Gender",
    "author_partisanship": "Author: Partisanship",
    "author_ideology":     "Author: Ideology",
    "author_race":         "Author: Race",
    "author_age":          "Author: Age",
    "author_education":    "Author: Education",
    "author_income":       "Author: Income",
    "author_marital_status":  "Author: Marital Status",
    "author_religiosity":  "Author: Religiosity",
    "sentiment_polarity":    "Sentiment: Polarity",
    "sentiment_subjectivity":"Sentiment: Subjectivity",
    "has_emoji":   "Style: Has Emoji",
    "has_hashtag": "Style: Has Hashtag",
    "has_mention": "Style: Has Mention",
    "has_url":     "Style: Has URL",
    "avg_word_length": "Text: Avg Word Length",
    "text_length":     "Text: Length",
    "word_count":      "Text: Word Count",
    "polarization_score": "Content: Polarization",
    "primary_topic":      "Content: Primary Topic",
    "toxicity":           "Toxicity: Score",
    "is_reply":    "Tweet: Is Reply",
    "is_retweet":  "Tweet: Is Retweet",
    "is_quote":    "Tweet: Is Quote",
    "user_followers_count":  "Author: Followers",
    "user_friends_count":    "Author: Following",
    "user_statuses_count":   "Author: Tweet Count",
    "user_favourites_count": "Author: Likes Given",
    "user_verified":         "Author: Verified",
    "user_account_age_days": "Author: Account Age",
    "engagement_score":      "Author: Engagement Score",
    "favorite_count": "Post: Likes",
    "retweet_count":  "Post: Retweets",
    "retweeted":      "Post: Is Retweeted",
}

PROVIDERS    = ["openai", "anthropic", "gemini"]
PROMPT_STYLES = ["general", "popular", "engaging", "informative", "controversial", "neutral"]

OUTPUT_DIR = Path("analysis_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def flat_features(features_dict=FEATURES):
    return [f for grp in features_dict.values() for f in grp]


def format_feature_name(name):
    return FEATURE_DISPLAY_NAMES.get(name, name.replace("_", " ").title())


def compute_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def compute_cramers_v(pool_vals, rec_vals):
    try:
        pool_vals = pool_vals.reset_index(drop=True)
        rec_vals  = rec_vals.reset_index(drop=True)
        combined  = pd.concat([pool_vals, rec_vals], ignore_index=True)
        labels    = pd.Series(["pool"] * len(pool_vals) + ["rec"] * len(rec_vals))
        contingency = pd.crosstab(combined, labels)
        if contingency.shape[0] <= 1 or contingency.shape[1] <= 1:
            return 0.0
        chi2, _, _, _ = chi2_contingency(contingency)
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        if min_dim == 0 or n == 0:
            return 0.0
        return float(np.sqrt(chi2 / (n * min_dim)))
    except Exception:
        return 0.0


def compute_bias_metric(pool_vals, rec_vals, feature_type):
    """Return (bias_value, p_value, metric_name)."""
    pool_vals = pool_vals.dropna()
    rec_vals  = rec_vals.dropna()
    if len(pool_vals) < 10 or len(rec_vals) < 10:
        return 0.0, 1.0, "insufficient_data"

    if feature_type in ("numerical", "binary"):
        pool_arr = pool_vals.astype(float).values
        rec_arr  = rec_vals.astype(float).values
        d = compute_cohens_d(rec_arr, pool_arr)
        _, p = ttest_ind(rec_arr, pool_arr)
        return abs(d), float(p), "Cohen's d"
    else:  # categorical
        v = compute_cramers_v(pool_vals, rec_vals)
        pool_arr = pool_vals.astype(float).values if feature_type == "binary" else pool_vals.values
        # p-value via chi-square
        try:
            combined = pd.concat([pool_vals, rec_vals], ignore_index=True)
            labels   = pd.Series(["pool"] * len(pool_vals) + ["rec"] * len(rec_vals))
            ct = pd.crosstab(combined, labels)
            _, p, _, _ = chi2_contingency(ct)
        except Exception:
            p = 1.0
        return v, float(p), "Cramér's V"


def compute_directional_bias(pool_vals, rec_vals, feature_type):
    """Return list of (category, directional_bias, prop_pool, prop_rec, ...) dicts."""
    pool_vals = pool_vals.dropna()
    rec_vals  = rec_vals.dropna()
    rows = []

    if feature_type in ("numerical", "binary"):
        pool_mean = float(pool_vals.astype(float).mean())
        rec_mean  = float(rec_vals.astype(float).mean())
        pool_std  = float(pool_vals.astype(float).std())
        rec_std   = float(rec_vals.astype(float).std())
        rows.append({
            "category":         "mean",
            "directional_bias": rec_mean - pool_mean,
            "mean_pool":        pool_mean,
            "mean_recommended": rec_mean,
            "std_pool":         pool_std,
            "std_recommended":  rec_std,
        })
    else:
        all_cats = sorted(set(pool_vals.unique()) | set(rec_vals.unique()))
        n_pool = len(pool_vals)
        n_rec  = len(rec_vals)
        for cat in all_cats:
            pp = float((pool_vals == cat).sum() / n_pool) if n_pool else 0.0
            rp = float((rec_vals  == cat).sum() / n_rec)  if n_rec  else 0.0
            rows.append({
                "category":         cat,
                "directional_bias": rp - pp,
                "prop_pool":        pp,
                "prop_recommended": rp,
            })
    return rows


def load_experiment_data(experiments_dir: Path, provider: str) -> pd.DataFrame | None:
    exp_dirs = list(experiments_dir.glob(f"{provider}_*"))
    if not exp_dirs:
        return None
    return pd.read_csv(exp_dirs[0] / "post_level_data.csv")


def get_available_features(df: pd.DataFrame) -> list:
    """Return configured features that are actually present in df."""
    all_configured = flat_features()
    return [f for f in all_configured if f in df.columns]


# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

def compute_feature_importance(df: pd.DataFrame, features: list) -> dict:
    """Train a Random Forest and compute SHAP + AUROC for one condition.

    Returns an empty dict (skipped) if shap is not importable.
    """
    if not SHAP_AVAILABLE:
        return {}

    df = df.copy()

    # Encode categorical columns
    X_cols = []
    for f in features:
        if f not in df.columns:
            continue
        ftype = FEATURE_TYPES.get(f, "numerical")
        if ftype == "categorical":
            le = LabelEncoder()
            df[f + "_enc"] = le.fit_transform(df[f].astype(str).fillna("unknown"))
            X_cols.append(f + "_enc")
        else:
            df[f] = pd.to_numeric(df[f], errors="coerce").fillna(0)
            X_cols.append(f)

    if not X_cols:
        return {}

    X = df[X_cols].values
    y = df["selected"].values

    if len(np.unique(y)) < 2:
        return {}

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    y_prob = rf.predict_proba(X)[:, 1]
    auroc = float(roc_auc_score(y, y_prob))

    explainer = _shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(X)
    # Older shap: list of arrays per class. Newer shap: 3D array (samples, features, classes).
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    elif shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]
    shap_importance = np.abs(shap_vals).mean(axis=0)

    result = {}
    for i, col in enumerate(X_cols):
        orig_name = col.replace("_enc", "")
        result[orig_name] = {
            "rf_importance":   float(rf.feature_importances_[i]),
            "shap_importance": float(shap_importance[i]),
            "auroc":           auroc,
            "n_samples":       len(y),
            "n_positive":      int(y.sum()),
            "n_negative":      int((1 - y).sum()),
        }
    return result


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--experiments-dir", type=Path, default=Path("outputs/experiments"),
                        help="Directory containing experiment subdirectories (default: outputs/experiments)")
    args = parser.parse_args()

    experiments_dir = args.experiments_dir
    if not experiments_dir.exists():
        print(f"ERROR: {experiments_dir} not found.")
        print(f"       Run first: python run_llm_recommendation.py --provider anthropic")
        import sys; sys.exit(1)

    summary_rows      = []
    dir_bias_rows     = []
    importance_rows   = []

    if not SHAP_AVAILABLE:
        print("WARNING: shap could not be imported — feature_importance_data.csv will be empty.")
        print("         Install a compatible version: pip install shap")

    for provider in PROVIDERS:
        print(f"\n{'='*60}")
        print(f"Provider: {provider.upper()}")
        df = load_experiment_data(experiments_dir, provider)
        if df is None:
            print(f"  No data found — skipping.")
            continue

        # Handle data without context_level column (backward compat)
        if "context_level" not in df.columns:
            df["context_level"] = "none"

        context_levels = sorted(df["context_level"].unique())
        features = get_available_features(df)
        print(f"  Features found: {features}")
        print(f"  Context levels: {context_levels}")

        for style in PROMPT_STYLES:
            for context_level in context_levels:
                sub = df[
                    (df["prompt_style"] == style) &
                    (df["context_level"] == context_level)
                ].copy()
                if sub.empty:
                    continue

                pool_df = sub[sub["selected"] == 0]
                rec_df  = sub[sub["selected"] == 1]

                # --------------------------------------------------------------
                # Bias summary
                for feature in features:
                    ftype = FEATURE_TYPES.get(feature, "numerical")
                    bias, p, metric = compute_bias_metric(
                        pool_df[feature], rec_df[feature], ftype
                    )
                    summary_rows.append({
                        "feature":        feature,
                        "provider":       provider,
                        "prompt_style":   style,
                        "context_level":  context_level,
                        "bias":           bias,
                        "p_value":        p,
                        "metric":         metric,
                        "significant":    p < 0.05,
                    })

                    # Directional bias rows
                    dir_rows = compute_directional_bias(
                        pool_df[feature], rec_df[feature], ftype
                    )
                    for row in dir_rows:
                        row.update({
                            "feature":       feature,
                            "provider":      provider,
                            "prompt_style":  style,
                            "context_level": context_level,
                            "feature_type":  ftype,
                        })
                        dir_bias_rows.append(row)

                # --------------------------------------------------------------
                # Feature importance (one RF per style × context_level × provider)
                imp = compute_feature_importance(sub, features)
                for feat, stats in imp.items():
                    importance_rows.append({
                        "feature":       feat,
                        "provider":      provider,
                        "prompt_style":  style,
                        "context_level": context_level,
                        **stats,
                    })

    # --------------------------------------------------------------------------
    # Save outputs
    summary_df    = pd.DataFrame(summary_rows)
    dir_bias_df   = pd.DataFrame(dir_bias_rows)
    importance_df = pd.DataFrame(importance_rows)

    summary_df.to_csv(OUTPUT_DIR / "pool_vs_recommended_summary.csv", index=False)
    dir_bias_df.to_csv(OUTPUT_DIR / "directional_bias_data.csv",      index=False)
    importance_df.to_csv(OUTPUT_DIR / "feature_importance_data.csv",  index=False)

    print(f"\n{'='*60}")
    print(f"✓ Outputs saved to {OUTPUT_DIR}/")
    print(f"  pool_vs_recommended_summary.csv  ({len(summary_df):,} rows)")
    print(f"  directional_bias_data.csv         ({len(dir_bias_df):,} rows)")
    print(f"  feature_importance_data.csv        ({len(importance_df):,} rows)")
    print(f"\n  Next step: python generate_figures.py")


if __name__ == "__main__":
    main()
