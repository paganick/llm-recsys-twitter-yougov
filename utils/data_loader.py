"""
Shared data loading utility for analysis scripts.

Supports two experiment formats:

New format (outputs_v2/):
  pools/post_features.csv    — one row per shown post (with all features)
  pools/author_features.csv  — one row per author (stable demographics)
  pools/trial_*.csv          — full trial pool per trial (same pool across conditions)
  experiments/*/trial_results.csv — selected post_ids per condition

Legacy format (outputs/experiments/*/post_level_data.csv):
  Flat file with one row per (post × trial × condition), all features already
  merged. Each condition has its own pool (different posts per condition), so
  pool reconstruction from trial_*.csv files is not applicable.

Returns a single DataFrame with one row per (post × trial × condition), ready
for bias analysis. Both formats can coexist under the same experiments_dir.
"""

from pathlib import Path
import pandas as pd


PROVIDERS       = ["anthropic", "openai", "gemini", "google"]
_PROVIDER_ALIAS = {"google": "gemini"}


def load_data(fake: bool = False,
              pools_dir: Path = None,
              experiments_dir: Path = None,
              analysis_dir: Path = None) -> pd.DataFrame:
    """
    Parameters
    ----------
    fake : bool
        If True, reads from outputs_fake/ instead of outputs/.
    pools_dir : Path, optional
        Override the pools directory (default: outputs/pools).
    experiments_dir : Path, optional
        Override the experiments directory (default: outputs/experiments).
    analysis_dir : Path, optional
        Unused; accepted for API symmetry with callers.
    """
    base      = Path("outputs_fake" if fake else "outputs")
    pools_dir = Path(pools_dir) if pools_dir else base / "pools"
    exp_dir   = Path(experiments_dir) if experiments_dir else base / "experiments"

    # ── Classify experiment directories ──────────────────────────────────────
    new_dirs    = []
    legacy_dirs = []
    for d in sorted(exp_dir.iterdir()):
        if not d.is_dir():
            continue
        provider = d.name.split("_")[0]
        if provider not in PROVIDERS:
            continue
        if (d / "trial_results.csv").exists():
            new_dirs.append(d)
        elif (d / "post_level_data.csv").exists():
            legacy_dirs.append(d)

    frames = []

    # ── New format ────────────────────────────────────────────────────────────
    if new_dirs:
        post_path   = pools_dir / "post_features.csv"
        author_path = pools_dir / "author_features.csv"

        if not post_path.exists():
            raise FileNotFoundError(
                f"{post_path} not found. "
                "Run prepare_pools.py then compute_text_features.py first."
            )
        if not author_path.exists():
            raise FileNotFoundError(
                f"{author_path} not found. Run prepare_pools.py first."
            )

        post_df   = pd.read_csv(post_path,   engine="python", on_bad_lines="warn")
        author_df = pd.read_csv(author_path, engine="python", on_bad_lines="warn")
        print(f"  post_features:   {len(post_df):,} posts")
        print(f"  author_features: {len(author_df):,} authors")

        trial_pool_files = sorted(pools_dir.glob("trial_*.csv"))
        if not trial_pool_files:
            raise FileNotFoundError(
                f"No trial_*.csv files found under {pools_dir}. "
                "Run prepare_pools.py first."
            )
        trial_pools = {}
        for f in trial_pool_files:
            tid = int(f.stem.split("_")[1])
            trial_pools[tid] = pd.read_csv(
                f, engine="python", on_bad_lines="warn", usecols=["post_id"]
            )
        pool_stack = pd.concat(
            [df.assign(trial_id=tid) for tid, df in trial_pools.items()],
            ignore_index=True,
        )

        post_with_author = post_df[["post_id", "author_id"]]
        post_features    = post_df.drop(columns=["author_id"], errors="ignore")

        for d in new_dirs:
            provider = _PROVIDER_ALIAS.get(d.name.split("_")[0], d.name.split("_")[0])
            sel = pd.read_csv(d / "trial_results.csv", engine="python", on_bad_lines="warn")
            sel["selected"] = 1
            print(f"  trial_results:   {len(sel):,} selected rows  ({d.name})")

            conditions = sel[["trial_id", "prompt_style", "context_level"]].drop_duplicates()
            shown      = pool_stack.merge(conditions, on="trial_id", how="inner")
            sel_keys   = sel[["post_id", "trial_id", "prompt_style", "context_level", "selected"]]
            shown      = shown.merge(sel_keys,
                                     on=["post_id", "trial_id", "prompt_style", "context_level"],
                                     how="left")
            shown["selected"] = shown["selected"].fillna(0).astype(int)
            shown["provider"] = provider
            shown["model"]    = "_".join(d.name.split("_")[1:])

            shown = (
                shown
                .merge(post_with_author, on="post_id", how="left")
                .merge(post_features,   on="post_id", how="left")
                .merge(author_df,       on="author_id", how="left")
            )
            frames.append(shown)

    # ── Legacy format (flat post_level_data.csv, per-condition pools) ─────────
    for d in legacy_dirs:
        df = pd.read_csv(d / "post_level_data.csv", low_memory=False)
        print(f"  post_level_data: {len(df):,} rows  ({d.name})")

        if "user_id" in df.columns and "author_id" not in df.columns:
            df = df.rename(columns={"user_id": "author_id"})

        provider = d.name.split("_")[0]
        df["provider"] = _PROVIDER_ALIAS.get(provider, provider)
        df["model"]    = "_".join(d.name.split("_")[1:])
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No trial_results.csv or post_level_data.csv found under {exp_dir}. "
            "Run run_llm_recommendation.py first."
        )

    combined = pd.concat(frames, ignore_index=True)
    if "context_level" not in combined.columns:
        combined["context_level"] = "none"

    combined["trial_group"] = (
        combined["provider"]                      + "|" +
        combined["prompt_style"].fillna("")       + "|" +
        combined["context_level"].fillna("none")  + "|" +
        combined["trial_id"].astype(str)
    )

    if "sentiment_polarity" in combined.columns:
        combined["abs_sentiment_polarity"] = combined["sentiment_polarity"].abs()

    return combined
