"""
Shared data loading utility for analysis scripts.

Loads and joins the three output tables:
  outputs/pools/post_features.csv    — one row per shown post (with NLP features)
  outputs/pools/author_features.csv  — one row per author
  outputs/experiments/*/trial_results.csv — one row per (post × condition)

Returns a single DataFrame with all columns, ready for bias analysis.
"""

from pathlib import Path
import pandas as pd


# Canonical provider names used throughout the analysis.
# "google" is the legacy name from early experiments; normalised to "gemini" on load.
PROVIDERS = ["anthropic", "openai", "gemini", "google"]
_PROVIDER_ALIAS = {"google": "gemini"}


def load_data(base: str = "outputs", fake: bool = False,
              migrated: bool = False) -> pd.DataFrame:
    """
    Load and join post_features, author_features, and trial_results.

    Parameters
    ----------
    base : str
        Root outputs directory ("outputs" or "outputs_fake").
    fake : bool
        Shortcut: if True, sets base="outputs_fake".

    Returns
    -------
    pd.DataFrame
        One row per (post × trial × condition), all feature columns included.
    """
    if migrated:
        base = "outputs_migrated"
    elif fake:
        base = "outputs_fake"

    root      = Path(base)
    # Migrated data has its own pools; otherwise pools are always canonical.
    pools_dir = root / "pools" if migrated else Path("outputs") / "pools"
    exp_dir   = root / "experiments"

    # ── Feature tables ────────────────────────────────────────────────────────
    post_path   = pools_dir / "post_features.csv"
    author_path = pools_dir / "author_features.csv"

    if not post_path.exists():
        raise FileNotFoundError(
            f"{post_path} not found. Run prepare_pools.py then compute_text_features.py first."
        )
    if not author_path.exists():
        raise FileNotFoundError(
            f"{author_path} not found. Run prepare_pools.py first."
        )

    post_df   = pd.read_csv(post_path,   low_memory=False)
    author_df = pd.read_csv(author_path, low_memory=False)
    print(f"  post_features:   {len(post_df):,} posts")
    print(f"  author_features: {len(author_df):,} authors")

    # ── Trial results (one file per provider) ─────────────────────────────────
    frames = []
    for d in sorted(exp_dir.iterdir()):
        csv = d / "trial_results.csv"
        if not csv.exists():
            continue
        provider = d.name.split("_")[0]
        if provider not in PROVIDERS:
            continue
        provider = _PROVIDER_ALIAS.get(provider, provider)
        df = pd.read_csv(csv, low_memory=False)
        df["provider"] = provider
        df["model"]    = "_".join(d.name.split("_")[1:])
        frames.append(df)
        print(f"  trial_results:   {len(df):,} rows  ({d.name})")

    if not frames:
        raise FileNotFoundError(
            f"No trial_results.csv found under {exp_dir}. "
            "Run run_llm_recommendation.py first."
        )

    trials = pd.concat(frames, ignore_index=True)
    trials["selected"] = (
        pd.to_numeric(trials["selected"], errors="coerce").fillna(0).astype(int)
    )
    if "context_level" not in trials.columns:
        trials["context_level"] = "none"

    # ── Join ──────────────────────────────────────────────────────────────────
    # Drop author_id from post_df — it's already in trials and would collide.
    post_df_for_join = post_df.drop(columns=["author_id"], errors="ignore")
    combined = (
        trials
        .merge(post_df_for_join, on="post_id",   how="left")
        .merge(author_df,        on="author_id", how="left")
    )

    combined["trial_group"] = (
        combined["provider"]                        + "|" +
        combined["prompt_style"].fillna("")         + "|" +
        combined["context_level"].fillna("none")    + "|" +
        combined["trial_id"].astype(str)
    )

    return combined
