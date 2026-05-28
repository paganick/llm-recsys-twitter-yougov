#!/usr/bin/env python3
"""
Compute a mixed-type association matrix for shown posts.

One cell per feature pair using the appropriate statistic:
  - Pearson |r|          for numerical × numerical and numerical × binary
  - Cramér's V           for categorical × categorical and categorical × binary
  - Correlation ratio η  for categorical × numerical

Produces:
  19_association_matrix.png   — all features in one 27×27 heatmap (semantic grouping)

Sub-group heatmaps:
  19a_text_style.png           — Generated: text style (Pearson r, signed)
  19b_content_semantics.png    — Generated: content & semantics (Pearson r, signed)
  19c_author_account.png       — Author: account data (Pearson r, signed)
  19d_author_demographics.png  — Author: demographics (Cramér's V, 0–1)
  19e_post_engagement.png      — Post: engagement (Pearson r, signed)

Usage
-----
    python pipeline/compute_feature_correlations.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT    = Path(__file__).parent.parent
EXP_DIR = ROOT / "outputs" / "experiments"
OUT_DIR = ROOT / "analysis_outputs" / "feature_correlations"

# ── feature definitions ───────────────────────────────────────────────────────

LOG_COLS = {
    "user_followers_count", "user_friends_count", "user_statuses_count",
    "user_favourites_count", "favorite_count", "retweet_count",
}

NUMERICAL = [
    "text_length", "avg_word_length", "word_count",
    "sentiment_polarity", "sentiment_subjectivity",
    "polarization_score", "toxicity",
    "user_followers_count", "user_friends_count",
    "user_statuses_count", "user_favourites_count",
    "favorite_count", "retweet_count",
]

BINARY = ["has_emoji", "has_hashtag", "has_mention", "has_url"]

CATEGORICAL = [
    "author_gender", "author_partisanship", "author_ideology",
    "author_race", "author_age", "author_education",
    "author_income", "author_marital_status", "author_religiosity",
    "primary_topic",
]

DISPLAY = {
    "text_length":            "Text length",
    "avg_word_length":        "Avg word length",
    "word_count":             "Word count",
    "sentiment_polarity":     "Sentiment polarity",
    "sentiment_subjectivity": "Sent. subjectivity",
    "polarization_score":     "Polarization",
    "toxicity":               "Toxicity",
    "has_emoji":              "Has emoji",
    "has_hashtag":            "Has hashtag",
    "has_mention":            "Has mention",
    "has_url":                "Has URL",
    "user_followers_count":   "Followers (log)",
    "user_friends_count":     "Following (log)",
    "user_statuses_count":    "Tweet count (log)",
    "user_favourites_count":  "Likes given (log)",
    "favorite_count":         "Post likes (log)",
    "retweet_count":          "Retweets (log)",
    "author_gender":          "Gender",
    "author_partisanship":    "Partisanship",
    "author_ideology":        "Ideology",
    "author_race":            "Race",
    "author_age":             "Age",
    "author_education":       "Education",
    "author_income":          "Income",
    "author_marital_status":  "Marital status",
    "author_religiosity":     "Religiosity",
    "primary_topic":          "Topic",
}

# ── semantic groups for main matrix ordering ──────────────────────────────────
# Ordered list of (group_label, [features])
SEMANTIC_GROUPS = [
    ("Generated\n(Text style)",     ["text_length", "avg_word_length", "word_count",
                                     "has_emoji", "has_hashtag", "has_mention", "has_url"]),
    ("Generated\n(Content)",        ["sentiment_polarity", "sentiment_subjectivity",
                                     "polarization_score", "toxicity", "primary_topic"]),
    ("Author\n(Account)",           ["user_followers_count", "user_friends_count",
                                     "user_statuses_count", "user_favourites_count"]),
    ("Post\n(Engagement)",          ["favorite_count", "retweet_count"]),
    ("Author\n(Demographics)",      ["author_gender", "author_partisanship", "author_ideology",
                                     "author_race", "author_age", "author_education",
                                     "author_income", "author_marital_status", "author_religiosity"]),
]

# ── sub-group heatmap definitions ─────────────────────────────────────────────
SUB_GROUPS = [
    {
        "key":      "text_style",
        "letter":   "a",
        "title":    "Generated Features — Text Style (Pearson r)",
        "features": ["text_length", "avg_word_length", "word_count",
                     "has_emoji", "has_hashtag", "has_mention", "has_url"],
        "plot":     "pearson",
    },
    {
        "key":      "content_semantics",
        "letter":   "b",
        "title":    "Generated Features — Content & Semantics (Pearson r)",
        "features": ["sentiment_polarity", "sentiment_subjectivity",
                     "polarization_score", "toxicity"],
        "plot":     "pearson",
    },
    {
        "key":      "author_account",
        "letter":   "c",
        "title":    "Author Account Features (Pearson r)",
        "features": ["user_followers_count", "user_friends_count",
                     "user_statuses_count", "user_favourites_count"],
        "plot":     "pearson",
    },
    {
        "key":      "author_demographics",
        "letter":   "d",
        "title":    "Author Demographic Features (Cramér's V)",
        "features": ["author_gender", "author_partisanship", "author_ideology",
                     "author_race", "author_age", "author_education",
                     "author_income", "author_marital_status", "author_religiosity"],
        "plot":     "cramersv",
    },
    {
        "key":      "post_engagement",
        "letter":   "e",
        "title":    "Post Engagement Features (Pearson r)",
        "features": ["favorite_count", "retweet_count"],
        "plot":     "pearson",
    },
]


# ── association statistics ────────────────────────────────────────────────────

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    x, y = x[mask].astype(str), y[mask].astype(str)
    if len(x) < 2:
        return np.nan
    ct = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum()
    min_dim = min(ct.shape) - 1
    if min_dim == 0 or n == 0:
        return np.nan
    return float(np.sqrt(chi2 / (n * min_dim)))


def correlation_ratio(cat: pd.Series, num: pd.Series) -> float:
    """η (eta) — how much variance in `num` is explained by `cat`."""
    mask = cat.notna() & num.notna()
    c, v = cat[mask].astype(str), num[mask].astype(float)
    if len(v) < 2:
        return np.nan
    overall_mean = v.mean()
    ss_total = ((v - overall_mean) ** 2).sum()
    if ss_total == 0:
        return 0.0
    ss_between = sum(
        len(grp) * (grp.mean() - overall_mean) ** 2
        for _, grp in v.groupby(c)
    )
    return float(np.sqrt(ss_between / ss_total))


def pearson_abs(a: pd.Series, b: pd.Series) -> float:
    mask = a.notna() & b.notna()
    if mask.sum() < 3:
        return np.nan
    r, _ = stats.pearsonr(a[mask].astype(float), b[mask].astype(float))
    return abs(float(r))


def pairwise_association(df: pd.DataFrame, features: list, feat_types: dict) -> pd.DataFrame:
    """Compute n×n association matrix (all values 0–1)."""
    n = len(features)
    mat = np.full((n, n), np.nan)

    for i, fi in enumerate(features):
        mat[i, i] = 1.0
        for j, fj in enumerate(features):
            if j >= i:
                continue
            ti, tj = feat_types[fi], feat_types[fj]
            si, sj = df[fi], df[fj]

            if ti in ("numerical", "binary") and tj in ("numerical", "binary"):
                v = pearson_abs(si, sj)
            elif ti == "categorical" and tj == "categorical":
                v = cramers_v(si, sj)
            elif ti == "categorical" and tj in ("numerical", "binary"):
                v = correlation_ratio(si, sj)
            elif ti in ("numerical", "binary") and tj == "categorical":
                v = correlation_ratio(sj, si)
            else:
                v = np.nan

            mat[i, j] = v
            mat[j, i] = v  # symmetric

    labels = [DISPLAY.get(f, f) for f in features]
    return pd.DataFrame(mat, index=labels, columns=labels)


# ── data loading ─────────────────────────────────────────────────────────────

def load_shown_posts() -> pd.DataFrame:
    frames = []
    for d in sorted(EXP_DIR.iterdir()):
        csv = d / "post_level_data.csv"
        if csv.exists():
            frames.append(pd.read_csv(csv, engine="python", on_bad_lines="warn"))
    all_df = pd.concat(frames, ignore_index=True)
    agg = {c: "first" for c in all_df.columns if c not in ("post_id", "selected")}
    agg["selected"] = "max"
    return all_df.groupby("post_id", sort=False).agg(agg).reset_index()


def prepare_df(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return cleaned df and feature-type dict."""
    out = {}
    feat_types = {}

    for col in NUMERICAL:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if col in LOG_COLS:
            s = np.log1p(s.clip(lower=0))
        out[col] = s
        feat_types[col] = "numerical"

    for col in BINARY:
        if col not in df.columns:
            continue
        out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        feat_types[col] = "binary"

    for col in CATEGORICAL:
        if col not in df.columns:
            continue
        out[col] = df[col].astype(str).replace("nan", np.nan)
        feat_types[col] = "categorical"

    return pd.DataFrame(out), feat_types


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_association_matrix(assoc: pd.DataFrame, boundaries: list,
                            group_labels: list, title: str, out_path: Path, n_posts: int):
    n = len(assoc)
    mask = np.triu(np.ones_like(assoc, dtype=bool), k=1)

    fig_size = max(12, n * 0.52)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))

    sns.heatmap(
        assoc, mask=mask, ax=ax,
        cmap="YlOrRd", vmin=0, vmax=1,
        linewidths=0.4, linecolor="white",
        annot=(n <= 30), fmt=".2f",
        annot_kws={"fontsize": 7},
        cbar_kws={"label": "Association strength (0–1)", "shrink": 0.6},
        square=True,
    )

    for b in boundaries:
        ax.axhline(b, color="black", linewidth=1.5)
        ax.axvline(b, color="black", linewidth=1.5)

    tick_fs = 9 if n <= 30 else 7
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=tick_fs)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=tick_fs)

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([(boundaries[i-1] if i > 0 else 0) + (b - (boundaries[i-1] if i > 0 else 0)) / 2
                    for i, b in enumerate(boundaries + [n])])
    ax2.set_yticklabels(group_labels, fontsize=9, fontstyle="italic")
    ax2.tick_params(length=0)

    method_note = (
        "Pearson |r| (numerical×numerical/binary),  "
        "Cramér's V (categorical×categorical/binary),  "
        "η correlation ratio (categorical×numerical)"
    )
    ax.set_title(
        f"{title}\n(N = {n_posts:,} unique shown posts)\n"
        f"[{method_note}]",
        fontweight="bold", fontsize=10, pad=10,
    )
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  ✓ {out_path.name}  ({n}×{n})")


def plot_pearson_subgroup(df: pd.DataFrame, features: list, title: str, out_path: Path):
    cols = [f for f in features if f in df.columns]
    mat = pd.DataFrame({DISPLAY.get(c, c): df[c] for c in cols}).astype(float)
    corr = mat.corr()
    n = len(corr)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig_w = max(7, n * 0.6 + 2)
    fig_h = max(5, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(corr, mask=mask, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                linewidths=0.4, linecolor="white",
                annot=True, fmt=".2f", annot_kws={"fontsize": 8},
                cbar_kws={"label": "Pearson r", "shrink": 0.7}, square=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_title(f"{title}\n(N = {len(mat):,} unique shown posts)",
                 fontweight="bold", fontsize=11, pad=10)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  ✓ {out_path.name}  ({n}×{n})")


def plot_cramersv_subgroup(df: pd.DataFrame, features: list, feat_types: dict,
                           title: str, out_path: Path):
    cols = [f for f in features if f in df.columns]
    assoc = pairwise_association(df[cols], cols, feat_types)
    n = len(assoc)
    mask = np.triu(np.ones_like(assoc, dtype=bool), k=1)

    fig_w = max(7, n * 0.6 + 2)
    fig_h = max(5, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(assoc, mask=mask, ax=ax, cmap="YlOrRd", vmin=0, vmax=1,
                linewidths=0.4, linecolor="white",
                annot=True, fmt=".2f", annot_kws={"fontsize": 8},
                cbar_kws={"label": "Cramér's V (0–1)", "shrink": 0.7}, square=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_title(f"{title}\n(N = {len(df):,} unique shown posts)",
                 fontweight="bold", fontsize=11, pad=10)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  ✓ {out_path.name}  ({n}×{n})")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading unique shown posts ...")
    raw = load_shown_posts()
    print(f"  {len(raw):,} unique posts")

    df, feat_types = prepare_df(raw)

    # ── main association matrix (semantic grouping) ──
    print("\nComputing mixed-type association matrix ...")
    all_features = []
    boundaries = []
    group_labels = []
    for label, feats in SEMANTIC_GROUPS:
        present = [f for f in feats if f in df.columns]
        if not present:
            continue
        all_features.extend(present)
        boundaries.append(len(all_features))
        group_labels.append(label)
    boundaries = boundaries[:-1]  # last group has no trailing line

    assoc = pairwise_association(df, all_features, feat_types)

    plot_association_matrix(
        assoc, boundaries, group_labels,
        title="Feature Association Matrix — All Shown Posts",
        out_path=OUT_DIR / "19_association_matrix.png",
        n_posts=len(df),
    )

    # ── per-group sub-heatmaps ──
    print("\nComputing sub-group heatmaps ...")
    for grp in SUB_GROUPS:
        out_path = OUT_DIR / f"19{grp['letter']}_{grp['key']}.png"
        if grp["plot"] == "pearson":
            cols = [f for f in grp["features"] if f in df.columns]
            # For Pearson: all numerical/binary — combine into float df
            sub = {}
            for col in cols:
                s = pd.to_numeric(raw[col], errors="coerce") if col not in df.columns else df[col].copy()
                if col in LOG_COLS and col not in df.columns:
                    s = np.log1p(s.clip(lower=0))
                if feat_types.get(col) == "binary":
                    s = s.fillna(0)
                sub[col] = df[col] if col in df.columns else s
            plot_pearson_subgroup(
                pd.DataFrame({c: df[c] for c in cols if c in df.columns}),
                cols, grp["title"], out_path,
            )
        elif grp["plot"] == "cramersv":
            plot_cramersv_subgroup(df, grp["features"], feat_types, grp["title"], out_path)

    print(f"\n✓ Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
