#!/usr/bin/env python3
"""
Logistic regression: predictors of post recommendation.

Layer 1 — 4 pooled models (one per context level), pooling all providers and
          prompt styles with those as fixed effects.
          → LaTeX table of odds ratios (4 columns).

Layer 2 — 18 sub-models (3 providers × 6 prompt styles) per context level.
          → Coefficient heatmap per context level (Figure 16).

Usage
-----
    python pipeline/logistic_regression.py [--fake]

    --fake   use outputs_fake/ instead of outputs/
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

# ============================================================================
# PATHS & CONSTANTS
# ============================================================================

ROOT = Path(__file__).parent.parent

CONTEXT_LEVELS  = ["none", "author", "post", "author_post"]
CONTEXT_LABELS  = {
    "none": "No context", "author": "Author", "post": "Post", "author_post": "Author+Post"
}
PROVIDER_ORDER  = ["anthropic", "openai", "google"]
PROVIDER_LABELS = {"anthropic": "Claude", "openai": "GPT-4o", "google": "Gemini"}
PROMPT_ORDER    = ["general", "popular", "engaging", "informative", "controversial", "neutral"]
PROMPT_LABELS   = {p: p.capitalize() for p in PROMPT_ORDER}

# Features that benefit from log1p (right-skewed counts)
LOG_FEATURES = [
    "user_followers_count", "user_friends_count", "user_statuses_count",
    "user_favourites_count", "favorite_count", "retweet_count",
]
# Features used on the original scale (still z-scored)
LINEAR_FEATURES = [
    "avg_word_length",
    "sentiment_polarity", "sentiment_subjectivity",
    "polarization_score", "toxicity",
]
BINARY_FEATURES = [
    "has_emoji", "has_hashtag", "has_mention", "has_url",
]
# Categorical features: reference category for dummy encoding
CATEGORICAL_CONFIG = {
    "author_gender":         {"ref": "male"},
    "author_partisanship":   {"ref": "Independent"},
    "author_ideology":       {"ref": "center"},
    "author_age":            {"ref": "25-34"},
    "author_education":      {"ref": "college"},
    "author_income":         {"ref": "$30-60k"},
    "author_marital_status": {"ref": "single"},
    "author_religiosity":    {"ref": "not religious"},
    "author_race":           {"ref": "white"},
}

# Display names for the table
FEATURE_DISPLAY = {
    "avg_word_length":        "Avg word length",
    "sentiment_polarity":     "Sentiment polarity",
    "sentiment_subjectivity": "Sentiment subjectivity",
    "polarization_score":     "Polarization",
    "toxicity":               "Toxicity",
    "user_followers_count":   "Followers (log)",
    "user_friends_count":     "Following (log)",
    "user_statuses_count":    "Tweet count (log)",
    "user_favourites_count":  "Likes given (log)",
    "favorite_count":         "Post likes (log)",
    "retweet_count":          "Post retweets (log)",
    "has_emoji":              "Has emoji",
    "has_hashtag":            "Has hashtag",
    "has_mention":            "Has mention",
    "has_url":                "Has URL",
    "retweeted":              "Is retweeted",
}
# Category labels within demographic dummies will be formatted as "Feature: value"

# Feature groups for table section headers
FEATURE_GROUPS = [
    ("Text",            ["avg_word_length"]),
    ("Content",         ["polarization_score", "toxicity"]),
    ("Sentiment",       ["sentiment_polarity", "sentiment_subjectivity"]),
    ("Style",           ["has_emoji", "has_hashtag", "has_mention", "has_url"]),
    ("Post metadata",   ["favorite_count", "retweet_count"]),
    ("Author metadata", ["user_followers_count", "user_friends_count",
                         "user_statuses_count", "user_favourites_count"]),
    ("Demographics",    list(CATEGORICAL_CONFIG.keys())),
]

DIVG_COLORS = [
    "#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#F7F7F7",
    "#FFFFFF",
    "#FEE0D2", "#FCBBA1", "#FC9272", "#FB6A4A", "#DE2D26",
]
CMAP_DIVG = LinearSegmentedColormap.from_list("divg", DIVG_COLORS, N=256)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(fake: bool = False) -> pd.DataFrame:
    base = "outputs_fake" if fake else "outputs"
    exp_root = ROOT / base / "experiments"

    frames = []
    for exp_dir in sorted(exp_root.iterdir()):
        csv = exp_dir / "post_level_data.csv"
        if not csv.exists():
            continue
        provider = exp_dir.name.split("_")[0]
        if provider not in PROVIDER_ORDER:
            continue
        df = pd.read_csv(csv, low_memory=False)
        df["provider"] = provider
        frames.append(df)
        print(f"  Loaded {len(df):,} rows from {exp_dir.name}")

    combined = pd.concat(frames, ignore_index=True)
    combined["selected"] = pd.to_numeric(combined["selected"], errors="coerce").fillna(0).astype(int)
    # Unique trial identifier across providers/prompts/context levels
    combined["trial_group"] = (
        combined["provider"] + "|" +
        combined["prompt_style"].fillna("") + "|" +
        combined["context_level"].fillna("none") + "|" +
        combined["trial_id"].astype(str)
    )
    return combined


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def _to_numeric_bool(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.map(lambda x: 1 if x is True or x == "True"
              else (0 if x is False or x == "False" else x)),
        errors="coerce",
    )


def fit_scalers(df: pd.DataFrame) -> dict:
    """Compute log1p + z-score parameters from the full dataset."""
    params = {}
    for f in LOG_FEATURES:
        if f not in df.columns:
            continue
        col = np.log1p(pd.to_numeric(df[f], errors="coerce"))
        params[f] = {"log": True, "mean": col.mean(), "std": col.std()}
    for f in LINEAR_FEATURES:
        if f not in df.columns:
            continue
        col = pd.to_numeric(df[f], errors="coerce")
        params[f] = {"log": False, "mean": col.mean(), "std": col.std()}
    return params


def apply_scalers(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()
    for f, p in params.items():
        if f not in df.columns:
            continue
        col = pd.to_numeric(df[f], errors="coerce")
        if p["log"]:
            col = np.log1p(col)
        σ = p["std"] if p["std"] > 0 else 1.0
        df[f] = (col - p["mean"]) / σ
    for f in BINARY_FEATURES:
        if f in df.columns:
            df[f] = _to_numeric_bool(df[f])
    return df


def build_X(df: pd.DataFrame, scaler_params: dict,
            feature_names: list[str],
            add_fixed_effects: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """
    Build the design matrix. Returns (X_with_const, list_of_feature_col_names).
    feature_names: base feature names to include (continuous + binary + categorical keys).
    """
    df = apply_scalers(df, scaler_params)
    parts = []
    feat_cols = []  # track which columns correspond to model features (excl. FE + const)

    # Continuous + binary
    for f in feature_names:
        if f in CATEGORICAL_CONFIG:
            continue
        if f not in df.columns:
            continue
        col = pd.to_numeric(df[f], errors="coerce") if f not in BINARY_FEATURES else df[f]
        frac_valid = col.notna().mean()
        if frac_valid < 0.3:  # skip mostly-missing features
            continue
        parts.append(col.rename(f))
        feat_cols.append(f)

    # Categorical dummies
    for f, cfg in CATEGORICAL_CONFIG.items():
        if f not in feature_names or f not in df.columns:
            continue
        ref = cfg["ref"]
        vals = df[f].fillna("unknown").astype(str)
        # Drop reference and "unknown" categories
        cats = sorted(v for v in vals.unique() if v not in (ref, "unknown"))
        for cat in cats:
            col_name = f"{f}::{cat}"
            parts.append((vals == cat).astype(float).rename(col_name))
            feat_cols.append(col_name)

    # Fixed effects
    if add_fixed_effects:
        prov_dummies   = pd.get_dummies(df["provider"],    prefix="FE_prov",   drop_first=True)
        prompt_dummies = pd.get_dummies(df["prompt_style"], prefix="FE_prompt", drop_first=True)
        parts += [prov_dummies, prompt_dummies]

    X = pd.concat(parts, axis=1).astype(float)
    # Drop zero-variance columns (constant → collinear with intercept)
    varying = X.std() > 0
    zero_cols = list(X.columns[~varying])
    if zero_cols:
        feat_cols = [f for f in feat_cols if f not in zero_cols]
        X = X.loc[:, varying]
    X = sm.add_constant(X, prepend=True, has_constant="add")
    return X, feat_cols


# ============================================================================
# MODEL FITTING
# ============================================================================

def fit_logit(y: pd.Series, X: pd.DataFrame,
              groups: pd.Series | None = None):
    """Fit logistic regression with optional clustered SEs."""
    mask = y.notna() & X.notna().all(axis=1)
    y_, X_ = y[mask], X[mask]
    if y_.sum() < 10 or (1 - y_).sum() < 10:
        return None
    try:
        model = sm.Logit(y_, X_)
        if groups is not None:
            res = model.fit(disp=False, method="lbfgs", maxiter=1000,
                            cov_type="cluster",
                            cov_kwds={"groups": groups[mask]})
        else:
            res = model.fit(disp=False, method="lbfgs", maxiter=1000)
        return res
    except Exception as e:
        print(f"    fit error: {e}")
        return None


def extract_coefs(result, feat_cols: list[str]) -> pd.DataFrame:
    """Extract coefficient, OR, CI, p-value for each feature column."""
    rows = []
    for f in feat_cols:
        if f not in result.params.index:
            continue
        coef = result.params[f]
        se   = result.bse[f]
        pval = result.pvalues[f]
        rows.append({
            "feature": f,
            "coef":    coef,
            "se":      se,
            "or":      np.exp(coef),
            "ci_lo":   np.exp(coef - 1.96 * se),
            "ci_hi":   np.exp(coef + 1.96 * se),
            "pvalue":  pval,
        })
    return pd.DataFrame(rows)


def stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


# ============================================================================
# LAYER 1 — POOLED MODELS → LATEX TABLE
# ============================================================================

def run_pooled_models(df: pd.DataFrame, scaler_params: dict,
                      all_features: list[str]) -> dict:
    results = {}
    for cl in CONTEXT_LEVELS:
        sub = df[df["context_level"] == cl].copy()
        if len(sub) < 500:
            print(f"  Skipping {cl} (too few rows: {len(sub)})")
            continue
        print(f"  Fitting pooled model for context_level='{cl}' (N={len(sub):,}) ...")
        X, feat_cols = build_X(sub, scaler_params, all_features, add_fixed_effects=True)
        y = sub["selected"].reindex(X.index)
        groups = sub["trial_group"].reindex(X.index)
        res = fit_logit(y, X, groups=groups)
        if res is None:
            print(f"    WARN: model did not converge for {cl}")
            continue
        coef_df = extract_coefs(res, feat_cols)
        results[cl] = {
            "result":    res,
            "coef_df":   coef_df,
            "nobs":      int(res.nobs),
            "prsquared": res.prsquared,
            "aic":       res.aic,
        }
    return results


def make_latex_table(pooled_results: dict, out_path: Path):
    cls = [cl for cl in CONTEXT_LEVELS if cl in pooled_results]
    header_labels = [CONTEXT_LABELS[cl] for cl in cls]

    lines = []
    lines += [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Logistic Regression: Predictors of Post Recommendation (Odds Ratios)}",
        r"\label{tab:logistic_regression}",
        r"\begin{threeparttable}",
        r"\begin{tabular}{l" + "c" * len(cls) + "}",
        r"\toprule",
        " & " + " & ".join(f"\\textbf{{{lbl}}}" for lbl in header_labels) + r" \\",
        r"\midrule",
    ]

    def fmt_cell(coef_df, fname):
        row = coef_df[coef_df["feature"] == fname]
        if row.empty:
            return "---"
        r = row.iloc[0]
        s = stars(r["pvalue"])
        return f"{r['or']:.3f}{s}"

    def fmt_ci(coef_df, fname):
        row = coef_df[coef_df["feature"] == fname]
        if row.empty:
            return ""
        r = row.iloc[0]
        return f"[{r['ci_lo']:.3f},\\ {r['ci_hi']:.3f}]"

    def add_feature_rows(fname, display_name):
        cells = [fmt_cell(pooled_results[cl]["coef_df"], fname) for cl in cls]
        cis   = [fmt_ci(pooled_results[cl]["coef_df"], fname)   for cl in cls]
        lines.append(f"\\quad {display_name} & " + " & ".join(cells) + r" \\")
        lines.append("& " + " & ".join(f"\\scriptsize{{{c}}}" for c in cis) + r" \\[2pt]")

    for group_name, group_feats in FEATURE_GROUPS:
        # Check if any feature in this group has results
        all_feat_names = []
        for f in group_feats:
            if f in CATEGORICAL_CONFIG:
                # Collect all dummy column names across models
                for cl in cls:
                    cdf = pooled_results[cl]["coef_df"]
                    dummies = [c for c in cdf["feature"] if c.startswith(f"{f}::")]
                    all_feat_names += dummies
            else:
                all_feat_names.append(f)

        has_data = any(
            not pooled_results[cl]["coef_df"][
                pooled_results[cl]["coef_df"]["feature"] == fn
            ].empty
            for cl in cls
            for fn in all_feat_names
        )
        if not has_data:
            continue

        lines.append(f"\\multicolumn{{{len(cls)+1}}}{{l}}{{\\textit{{{group_name}}}}}" + r" \\")

        for f in group_feats:
            if f in CATEGORICAL_CONFIG:
                # Get all dummy values
                dummy_names = []
                for cl in cls:
                    cdf = pooled_results[cl]["coef_df"]
                    dummy_names += [c for c in cdf["feature"] if c.startswith(f"{f}::")]
                dummy_names = sorted(set(dummy_names))
                for dn in dummy_names:
                    cat_val = dn.split("::", 1)[1]
                    pretty = f.replace("author_", "").replace("_", " ").title()
                    add_feature_rows(dn, f"{pretty}: {cat_val}")
            else:
                if any(
                    not pooled_results[cl]["coef_df"][
                        pooled_results[cl]["coef_df"]["feature"] == f
                    ].empty
                    for cl in cls
                ):
                    add_feature_rows(f, FEATURE_DISPLAY.get(f, f))

        lines.append(r"\midrule")

    # Footer stats
    n_row   = " & ".join(f"{pooled_results[cl]['nobs']:,}" for cl in cls)
    r2_row  = " & ".join(f"{pooled_results[cl]['prsquared']:.3f}" for cl in cls)
    aic_row = " & ".join(f"{pooled_results[cl]['aic']:,.0f}" for cl in cls)
    lines += [
        f"$N$ & {n_row}" + r" \\",
        f"McFadden $R^2$ & {r2_row}" + r" \\",
        f"AIC & {aic_row}" + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}\footnotesize",
        r"\item Odds ratios. 95\% confidence intervals in brackets.",
        r"\item $^{*}p<0.05$,\ $^{**}p<0.01$,\ $^{***}p<0.001$.",
        r"\item All models include provider and prompt style fixed effects.",
        r"\item Standard errors clustered by trial.",
        r"\item Continuous features are standardised (log-transformed then z-scored for count variables).",
        r"\item Reference categories: male gender, Independent partisanship, center ideology,",
        r"\item \quad 25--34 age, college education, \$30--60k income, single marital status,",
        r"\item \quad not religious, white race.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]

    out_path.write_text("\n".join(lines))
    print(f"  ✓ {out_path.name}")


# ============================================================================
# LAYER 2 — SUB-MODELS → COEFFICIENT HEATMAP
# ============================================================================

def run_submodels(df: pd.DataFrame, scaler_params: dict,
                  scalar_features: list[str]) -> dict:
    """
    For each context_level × provider × prompt_style, fit a logistic regression
    and return a dict: context_level → DataFrame(feature, provider, prompt_style, coef, pvalue).
    Only scalar (continuous + binary) features — no fixed effects, no categorical dummies.
    """
    all_rows = []
    for cl in CONTEXT_LEVELS:
        for prov in PROVIDER_ORDER:
            for prompt in PROMPT_ORDER:
                sub = df[
                    (df["context_level"] == cl) &
                    (df["provider"] == prov) &
                    (df["prompt_style"] == prompt)
                ].copy()
                if len(sub) < 200:
                    continue
                X, feat_cols = build_X(sub, scaler_params, scalar_features,
                                       add_fixed_effects=False)
                y = sub["selected"].reindex(X.index)
                res = fit_logit(y, X)
                if res is None:
                    continue
                for f in feat_cols:
                    if f not in res.params.index:
                        continue
                    all_rows.append({
                        "context_level": cl,
                        "provider":      prov,
                        "prompt_style":  prompt,
                        "feature":       f,
                        "coef":          res.params[f],
                        "pvalue":        res.pvalues[f],
                    })
    return pd.DataFrame(all_rows)


def plot_heatmap_16(submodel_df: pd.DataFrame, out_dir: Path):
    """Figure 16: coefficient heatmap per context level."""
    scalar_feats = (
        [f for f in LOG_FEATURES    if f in submodel_df["feature"].unique()] +
        [f for f in LINEAR_FEATURES  if f in submodel_df["feature"].unique()] +
        [f for f in BINARY_FEATURES  if f in submodel_df["feature"].unique()]
    )
    feat_labels = [FEATURE_DISPLAY.get(f, f) for f in scalar_feats]
    conditions  = [
        f"{PROVIDER_LABELS.get(p, p)}\n{PROMPT_LABELS.get(s, s)}"
        for p in PROVIDER_ORDER for s in PROMPT_ORDER
    ]

    n_cl = len(CONTEXT_LEVELS)
    fig, axes = plt.subplots(1, n_cl, figsize=(7 * n_cl, max(6, len(scalar_feats) * 0.55)))

    for ax_idx, cl in enumerate(CONTEXT_LEVELS):
        ax  = axes[ax_idx]
        sub = submodel_df[submodel_df["context_level"] == cl]

        mat   = np.full((len(scalar_feats), len(PROVIDER_ORDER) * len(PROMPT_ORDER)), np.nan)
        pmat  = np.ones_like(mat)
        col_i = {(p, s): i * len(PROMPT_ORDER) + j
                 for i, p in enumerate(PROVIDER_ORDER)
                 for j, s in enumerate(PROMPT_ORDER)}

        for _, row in sub.iterrows():
            fi = next((i for i, f in enumerate(scalar_feats) if f == row["feature"]), None)
            ci = col_i.get((row["provider"], row["prompt_style"]))
            if fi is not None and ci is not None:
                mat[fi, ci]  = row["coef"]
                pmat[fi, ci] = row["pvalue"]

        max_abs = max(np.nanmax(np.abs(mat)), 1e-6)
        sns.heatmap(mat, ax=ax, cmap=CMAP_DIVG, center=0,
                    vmin=-max_abs, vmax=max_abs,
                    linewidths=0.3, linecolor="lightgray",
                    cbar=(ax_idx == n_cl - 1),
                    cbar_kws={"label": "Coefficient (log-odds)", "shrink": 0.8})

        # Significance stars
        for fi in range(len(scalar_feats)):
            for ci in range(len(conditions)):
                p = pmat[fi, ci]
                s = stars(p)
                if s:
                    ax.text(ci + 0.5, fi + 0.5, s, ha="center", va="center",
                            fontsize=7, color="black", fontweight="bold")

        ax.set_title(CONTEXT_LABELS[cl], fontweight="bold", fontsize=13)
        ax.set_yticks(np.arange(len(scalar_feats)) + 0.5)
        ax.set_yticklabels(feat_labels, fontsize=9, rotation=0, ha="right")
        ax.set_xticks(np.arange(len(conditions)) + 0.5)
        ax.set_xticklabels(conditions, fontsize=7, rotation=90)

        # Vertical lines separating providers
        for i in range(1, len(PROVIDER_ORDER)):
            ax.axvline(x=i * len(PROMPT_ORDER), color="black", linewidth=1.5)

        if ax_idx > 0:
            ax.set_yticklabels([])

    fig.suptitle(
        "Feature Coefficients by Provider × Prompt Style × Context Level\n"
        "(logistic regression, standardised features; *, **, *** = p<.05/.01/.001)",
        fontweight="bold", fontsize=13, y=1.01,
    )
    plt.tight_layout()
    out_path = out_dir / "16_coefficient_heatmap.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  ✓ 16_coefficient_heatmap.png")


def _heatmap_aggregated(submodel_df, scalar_feats, group_col, group_order, group_labels,
                        title, fname, out_dir):
    """Generic aggregated coefficient heatmap: features × group, one panel per context level."""
    feat_labels = [FEATURE_DISPLAY.get(f, f) for f in scalar_feats]
    n_cl = len(CONTEXT_LEVELS)
    fig, axes = plt.subplots(1, n_cl, figsize=(4 * n_cl, max(6, len(scalar_feats) * 0.55)))

    for ax_idx, cl in enumerate(CONTEXT_LEVELS):
        ax  = axes[ax_idx]
        sub = submodel_df[submodel_df["context_level"] == cl]

        # Average coefficients across the non-grouped dimension
        agg = sub.groupby(["feature", group_col]).agg(
            coef=("coef", "mean"), pvalue=("pvalue", "mean")
        ).reset_index()

        mat  = np.full((len(scalar_feats), len(group_order)), np.nan)
        pmat = np.ones_like(mat)
        g_idx = {g: i for i, g in enumerate(group_order)}

        for _, row in agg.iterrows():
            fi = next((i for i, f in enumerate(scalar_feats) if f == row["feature"]), None)
            gi = g_idx.get(row[group_col])
            if fi is not None and gi is not None:
                mat[fi, gi]  = row["coef"]
                pmat[fi, gi] = row["pvalue"]

        max_abs = max(np.nanmax(np.abs(mat)), 1e-6)
        sns.heatmap(mat, ax=ax, cmap=CMAP_DIVG, center=0,
                    vmin=-max_abs, vmax=max_abs,
                    linewidths=0.3, linecolor="lightgray",
                    cbar=(ax_idx == n_cl - 1),
                    cbar_kws={"label": "Mean coefficient (log-odds)", "shrink": 0.8})

        for fi in range(len(scalar_feats)):
            for gi in range(len(group_order)):
                s = stars(pmat[fi, gi])
                if s:
                    ax.text(gi + 0.5, fi + 0.5, s, ha="center", va="center",
                            fontsize=9, color="black", fontweight="bold")

        ax.set_title(CONTEXT_LABELS[cl], fontweight="bold", fontsize=13)
        ax.set_yticks(np.arange(len(scalar_feats)) + 0.5)
        ax.set_yticklabels(feat_labels if ax_idx == 0 else [], fontsize=10, rotation=0, ha="right")
        ax.set_xticks(np.arange(len(group_order)) + 0.5)
        ax.set_xticklabels(group_labels, fontsize=10, rotation=30, ha="right")

    fig.suptitle(title, fontweight="bold", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / fname, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  ✓ {fname}")


def plot_heatmap_by_model(submodel_df: pd.DataFrame, out_dir: Path):
    """Figure 17: coefficients averaged across prompts, one column per model."""
    scalar_feats = (
        [f for f in LOG_FEATURES    if f in submodel_df["feature"].unique()] +
        [f for f in LINEAR_FEATURES  if f in submodel_df["feature"].unique()] +
        [f for f in BINARY_FEATURES  if f in submodel_df["feature"].unique()]
    )
    group_labels = [PROVIDER_LABELS.get(p, p) for p in PROVIDER_ORDER]
    _heatmap_aggregated(
        submodel_df, scalar_feats,
        group_col="provider", group_order=PROVIDER_ORDER, group_labels=group_labels,
        title="Feature Coefficients by Model × Context Level\n"
              "(averaged across prompt styles; *, **, *** = p<.05/.01/.001)",
        fname="17_coefficient_heatmap_by_model.png",
        out_dir=out_dir,
    )


CONTEXT_COLORS = {
    "none":        "#888888",
    "author":      "#2166AC",
    "post":        "#33A02C",
    "author_post": "#984EA3",
}


def plot_forest_20(table_df: pd.DataFrame, out_dir: Path):
    """Figure 20: forest plot of pooled-model coefficients with 95% CIs."""
    all_feats = set(table_df["feature"].unique())

    feat_rows = []   # (feat_col_or_None, display_label, y_pos, is_header)
    y = 0.0
    for ig, (g_name, g_feats) in enumerate(FEATURE_GROUPS):
        if ig > 0:
            y += 0.6
        feat_rows.append((None, f"▸ {g_name}", y, True))
        y += 0.9
        for f in g_feats:
            if f in CATEGORICAL_CONFIG:
                dummies = sorted(c for c in all_feats if c.startswith(f"{f}::"))
                for dn in dummies:
                    cat_val = dn.split("::", 1)[1]
                    base = f.replace("author_", "").replace("_", " ").title()
                    feat_rows.append((dn, f"  {base}: {cat_val}", y, False))
                    y += 1.0
            elif f in all_feats:
                feat_rows.append((f, f"  {FEATURE_DISPLAY.get(f, f)}", y, False))
                y += 1.0
    total_h = y

    offsets = {"none": -0.27, "author": -0.09, "post": 0.09, "author_post": 0.27}
    fig, ax = plt.subplots(figsize=(10, max(10, total_h * 0.30)))

    for cl in CONTEXT_LEVELS:
        cl_sub = table_df[table_df["context_level"] == cl].set_index("feature")
        color  = CONTEXT_COLORS[cl]
        first  = True
        for feat_col, _, ypos, is_header in feat_rows:
            if is_header or feat_col not in cl_sub.index:
                continue
            row  = cl_sub.loc[feat_col]
            or_  = row["or"]
            ci_lo, ci_hi = row["ci_lo"], row["ci_hi"]
            pval = row["pvalue"]
            yy   = ypos + offsets[cl]
            ax.plot([ci_lo, ci_hi], [yy, yy], color=color, lw=1.0, alpha=0.55)
            ax.scatter([or_], [yy], color=color,
                       s=28 if pval < 0.05 else 12,
                       marker="D" if pval < 0.05 else "o",
                       zorder=5,
                       label=CONTEXT_LABELS[cl] if first else None)
            first = False

    ax.set_yticks([r[2] for r in feat_rows])
    ax.set_yticklabels([r[1] for r in feat_rows], fontsize=7)
    tick_labels = ax.get_yticklabels()
    for i, (_, _, _, is_header) in enumerate(feat_rows):
        if is_header and i < len(tick_labels):
            tick_labels[i].set_fontweight("bold")
            tick_labels[i].set_fontsize(8.5)
            tick_labels[i].set_color("dimgray")

    ax.axvline(1.0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("Odds Ratio  (OR > 1 = more likely to be recommended)", fontsize=10)
    ax.set_ylim(total_h + 0.5, -1.0)
    ax.grid(axis="x", alpha=0.25)

    handles = [
        plt.Line2D([0], [0], marker="D", color="w",
                   markerfacecolor=CONTEXT_COLORS[cl], markersize=7,
                   label=CONTEXT_LABELS[cl])
        for cl in CONTEXT_LEVELS
    ]
    ax.legend(handles=handles, title="Context level", loc="lower right",
              fontsize=9, title_fontsize=9, framealpha=0.9)
    ax.set_title(
        "Logistic Regression — All Feature Odds Ratios (pooled models)\n"
        "95% CI bars;  ◆ = p<0.05,  ● = n.s.;  dashed line = OR 1.0 (null)",
        fontweight="bold", fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out_dir / "20_forest_plot.png", bbox_inches="tight", dpi=200)
    plt.close()
    print("  ✓ 20_forest_plot.png")


def plot_context_effect_21(table_df: pd.DataFrame, out_dir: Path):
    """Figure 21: how revealing author/post context changes bias on matching features."""
    xticks  = list(range(len(CONTEXT_LEVELS)))
    xlabels = [CONTEXT_LABELS[cl] for cl in CONTEXT_LEVELS]

    def _coef_series(feat_col):
        vals = []
        for cl in CONTEXT_LEVELS:
            r = table_df[(table_df["feature"] == feat_col) & (table_df["context_level"] == cl)]
            vals.append(float(r["coef"].iloc[0]) if not r.empty else np.nan)
        return vals

    def _demo_abs_series(feat_name):
        dummies = [c for c in table_df["feature"].unique() if c.startswith(f"{feat_name}::")]
        vals = []
        for cl in CONTEXT_LEVELS:
            sub = table_df[(table_df["feature"].isin(dummies)) & (table_df["context_level"] == cl)]
            vals.append(float(sub["coef"].abs().mean()) if not sub.empty else np.nan)
        return vals

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    def _shade_author(ax):
        for xi, cl in enumerate(CONTEXT_LEVELS):
            if "author" in cl:
                ax.axvspan(xi - 0.42, xi + 0.42, alpha=0.10, color="#2166AC", zorder=0)

    def _shade_post(ax):
        for xi, cl in enumerate(CONTEXT_LEVELS):
            if "post" in cl:
                ax.axvspan(xi - 0.42, xi + 0.42, alpha=0.10, color="#33A02C", zorder=0)

    def _fmt_ax(ax, title, ylabel):
        ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.7)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.grid(axis="y", alpha=0.25)

    # ── Panel A: author account scalars ──────────────────────────────────────
    ax = axes[0, 0]
    feats_a = [
        ("user_followers_count",  "Followers (log)"),
        ("user_friends_count",    "Following (log)"),
        ("user_statuses_count",   "Tweet count (log)"),
        ("user_favourites_count", "Likes given (log)"),
    ]
    for (fc, lbl), col in zip(feats_a, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]):
        ax.plot(xticks, _coef_series(fc), marker="o", lw=1.8, ms=6, color=col, label=lbl)
    _shade_author(ax)
    _fmt_ax(ax, "A  Author Account Features\n(revealed when context includes Author)",
            "Log-odds coefficient")
    ax.legend(fontsize=8, framealpha=0.85)

    # ── Panel B: author demographics (mean |coef| across dummies) ─────────────
    ax = axes[0, 1]
    demo_feats  = list(CATEGORICAL_CONFIG.keys())
    demo_labels = [f.replace("author_", "").replace("_", " ").title() for f in demo_feats]
    cmap_b = plt.cm.tab10
    for i, (feat, lbl) in enumerate(zip(demo_feats, demo_labels)):
        ax.plot(xticks, _demo_abs_series(feat), marker="s", lw=1.6, ms=5,
                color=cmap_b(i / max(len(demo_feats) - 1, 1)), label=lbl)
    _shade_author(ax)
    _fmt_ax(ax,
            "B  Author Demographic Features\n"
            "(revealed when context includes Author)\n"
            "Mean |coefficient| across category dummies",
            "Mean |coefficient| across dummies")
    ax.legend(fontsize=7, framealpha=0.85, ncol=2)

    # ── Panel C: post engagement ──────────────────────────────────────────────
    ax = axes[1, 0]
    feats_c = [
        ("favorite_count", "Post likes (log)"),
        ("retweet_count",  "Post retweets (log)"),
    ]
    for (fc, lbl), col in zip(feats_c, ["#e377c2", "#17becf"]):
        ax.plot(xticks, _coef_series(fc), marker="o", lw=1.8, ms=6, color=col, label=lbl)
    _shade_post(ax)
    _fmt_ax(ax, "C  Post Engagement Features\n(revealed when context includes Post)",
            "Log-odds coefficient")
    ax.legend(fontsize=8, framealpha=0.85)

    # ── Panel D: text/style features (control) ────────────────────────────────
    ax = axes[1, 1]
    feats_d = [
        ("avg_word_length",        "Avg word length"),
        ("sentiment_polarity",     "Sentiment polarity"),
        ("sentiment_subjectivity", "Sent. subjectivity"),
        ("polarization_score",     "Polarization"),
        ("toxicity",               "Toxicity"),
        ("has_emoji",              "Has emoji"),
        ("has_hashtag",            "Has hashtag"),
        ("has_mention",            "Has mention"),
        ("has_url",                "Has URL"),
    ]
    cmap_d = plt.cm.Set2
    for i, (fc, lbl) in enumerate(feats_d):
        ax.plot(xticks, _coef_series(fc), marker="^", lw=1.4, ms=5,
                color=cmap_d(i % 8), label=lbl, alpha=0.85)
    _fmt_ax(ax, "D  Text & Style Features (control)\n(always visible — no shading)",
            "Log-odds coefficient")
    ax.legend(fontsize=7, framealpha=0.85, ncol=2)

    fig.text(
        0.5, -0.015,
        "Blue shading = context levels where author info is provided in prompt. "
        "Green shading = context levels where post engagement info is provided.\n"
        "Key question: do author/post feature coefficients increase when the corresponding "
        "context is revealed?",
        ha="center", fontsize=9, style="italic", color="dimgray",
    )
    fig.suptitle(
        "Effect of Context Level on Feature Bias\n"
        "(pooled logistic regression, one model per context level)",
        fontweight="bold", fontsize=13,
    )
    plt.tight_layout()
    fig.savefig(out_dir / "21_context_effect.png", bbox_inches="tight", dpi=200)
    plt.close()
    print("  ✓ 21_context_effect.png")


def plot_demographic_zoom_22(table_df: pd.DataFrame, out_dir: Path):
    """Figure 22: OR for each category dummy of each demographic feature across context levels."""
    demo_feats = list(CATEGORICAL_CONFIG.keys())  # 9 features
    xticks  = list(range(len(CONTEXT_LEVELS)))
    xlabels = [CONTEXT_LABELS[cl] for cl in CONTEXT_LEVELS]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    for ax, feat in zip(axes.flatten(), demo_feats):
        feat_label = feat.replace("author_", "").replace("_", " ").title()
        ref_cat    = CATEGORICAL_CONFIG[feat]["ref"]
        dummies    = sorted(c for c in table_df["feature"].unique() if c.startswith(f"{feat}::"))

        colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(dummies), 1)))

        for j, dn in enumerate(dummies):
            cat_val = dn.split("::", 1)[1]
            yvals, err_lo, err_hi = [], [], []
            for cl in CONTEXT_LEVELS:
                r = table_df[(table_df["feature"] == dn) & (table_df["context_level"] == cl)]
                if r.empty:
                    yvals.append(np.nan); err_lo.append(np.nan); err_hi.append(np.nan)
                else:
                    or_, lo, hi = float(r["or"].iloc[0]), float(r["ci_lo"].iloc[0]), float(r["ci_hi"].iloc[0])
                    yvals.append(or_)
                    err_lo.append(or_ - lo)
                    err_hi.append(hi - or_)

            sig_any = any(
                table_df[(table_df["feature"] == dn) & (table_df["context_level"] == cl)]["pvalue"].iloc[0] < 0.05
                for cl in CONTEXT_LEVELS
                if not table_df[(table_df["feature"] == dn) & (table_df["context_level"] == cl)].empty
            )
            lw = 2.0 if sig_any else 1.0
            ls = "-" if sig_any else "--"
            ax.errorbar(xticks, yvals, yerr=[err_lo, err_hi],
                        marker="o", lw=lw, ls=ls, ms=5, capsize=3,
                        color=colors[j], label=cat_val, alpha=0.85)

        # Shade author context columns
        for xi, cl in enumerate(CONTEXT_LEVELS):
            if "author" in cl:
                ax.axvspan(xi - 0.42, xi + 0.42, alpha=0.10, color="#2166AC", zorder=0)

        ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.6)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=8, rotation=20, ha="right")
        ax.set_ylabel("Odds Ratio", fontsize=8)
        ax.set_title(f"{feat_label}  (ref: {ref_cat})", fontweight="bold", fontsize=10)
        ax.legend(fontsize=7, framealpha=0.85, loc="best",
                  ncol=2 if len(dummies) > 4 else 1)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Demographic Bias by Category and Context Level  (Odds Ratios)\n"
        "Solid line = significant in ≥1 context level;  "
        "blue shading = author info provided in prompt",
        fontweight="bold", fontsize=13,
    )
    plt.tight_layout()
    fig.savefig(out_dir / "22_demographic_zoom.png", bbox_inches="tight", dpi=200)
    plt.close()
    print("  ✓ 22_demographic_zoom.png")


def plot_heatmap_by_prompt(submodel_df: pd.DataFrame, out_dir: Path):
    """Figure 18: coefficients averaged across models, one column per prompt style."""
    scalar_feats = (
        [f for f in LOG_FEATURES    if f in submodel_df["feature"].unique()] +
        [f for f in LINEAR_FEATURES  if f in submodel_df["feature"].unique()] +
        [f for f in BINARY_FEATURES  if f in submodel_df["feature"].unique()]
    )
    group_labels = [PROMPT_LABELS.get(s, s) for s in PROMPT_ORDER]
    _heatmap_aggregated(
        submodel_df, scalar_feats,
        group_col="prompt_style", group_order=PROMPT_ORDER, group_labels=group_labels,
        title="Feature Coefficients by Prompt Style × Context Level\n"
              "(averaged across models; *, **, *** = p<.05/.01/.001)",
        fname="18_coefficient_heatmap_by_prompt.png",
        out_dir=out_dir,
    )


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake", action="store_true")
    args = parser.parse_args()

    out_dir = ROOT / "analysis_outputs" / "logistic_regression"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LOGISTIC REGRESSION ANALYSIS")
    print("=" * 70)

    print("\nLoading data ...")
    df = load_data(fake=args.fake)
    print(f"Total rows: {len(df):,}  |  selected: {df['selected'].sum():,}")

    print("\nFitting scalers on full dataset ...")
    scaler_params = fit_scalers(df)

    all_scalar   = LOG_FEATURES + LINEAR_FEATURES + BINARY_FEATURES
    all_features = all_scalar + list(CATEGORICAL_CONFIG.keys())

    # ---- Layer 1: pooled models ----------------------------------------
    print("\n" + "=" * 70)
    print("LAYER 1 — Pooled models (one per context level)")
    print("=" * 70)
    pooled_results = run_pooled_models(df, scaler_params, all_features)

    # Save CSV
    csv_rows = []
    for cl, res in pooled_results.items():
        tmp = res["coef_df"].copy()
        tmp["context_level"] = cl
        csv_rows.append(tmp)
    if csv_rows:
        pd.concat(csv_rows).to_csv(out_dir / "logistic_regression_table.csv", index=False)
        print(f"  ✓ logistic_regression_table.csv")

    # LaTeX table
    if pooled_results:
        make_latex_table(pooled_results, out_dir / "logistic_regression_table.tex")

    # ---- Layer 2: sub-models -------------------------------------------
    print("\n" + "=" * 70)
    print("LAYER 2 — Sub-models (provider × prompt per context level)")
    print("=" * 70)
    submodel_df = run_submodels(df, scaler_params, all_scalar)
    if not submodel_df.empty:
        submodel_df.to_csv(out_dir / "logistic_regression_submodels.csv", index=False)
        print(f"  ✓ logistic_regression_submodels.csv")
        plot_heatmap_16(submodel_df, out_dir)
        plot_heatmap_by_model(submodel_df, out_dir)
        plot_heatmap_by_prompt(submodel_df, out_dir)

    # ---- Forest plot & context effect (read from saved CSV) ----------------
    print("\n" + "=" * 70)
    print("FIGURES 20 & 21 — Visual summaries")
    print("=" * 70)
    table_df = pd.read_csv(out_dir / "logistic_regression_table.csv")
    plot_forest_20(table_df, out_dir)
    plot_context_effect_21(table_df, out_dir)
    plot_demographic_zoom_22(table_df, out_dir)

    print("\n" + "=" * 70)
    print("DONE → analysis_outputs/logistic_regression/")
    print("=" * 70)


if __name__ == "__main__":
    main()
