"""
Shared data loading utility for analysis scripts.

Loads and joins:
  outputs_v2/pools/post_features.csv    — one row per shown post (with all features)
  outputs_v2/pools/author_features.csv  — one row per author (stable demographics)
  outputs_v2/pools/trial_*.csv          — full trial pools (all shown posts per trial)
  outputs_v2/experiments/*/trial_results.csv — selected post_ids per condition

trial_results.csv stores only the *selected* posts to minimise storage. The full
pool (shown + not shown) is reconstructed here by crossing trial_results with the
trial pool files, marking selected=1 for hits and selected=0 for the rest.

Returns a single DataFrame with one row per (post × trial × condition), ready
for bias analysis.
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

    # ── Feature tables ────────────────────────────────────────────────────────
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

    # ── Trial pool files (full shown pool per trial) ──────────────────────────
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

    # Stack all pools: one row per (post_id, trial_id)
    pool_stack = pd.concat(
        [df.assign(trial_id=tid) for tid, df in trial_pools.items()],
        ignore_index=True,
    )

    # ── Trial results (selected posts per condition, per provider) ────────────
    frames = []
    for d in sorted(exp_dir.iterdir()):
        csv = d / "trial_results.csv"
        if not csv.exists():
            continue
        provider = d.name.split("_")[0]
        if provider not in PROVIDERS:
            continue
        provider = _PROVIDER_ALIAS.get(provider, provider)

        sel = pd.read_csv(csv, engine="python", on_bad_lines="warn")
        sel["selected"] = 1
        print(f"  trial_results:   {len(sel):,} selected rows  ({d.name})")

        # Conditions present in this provider's results
        conditions = sel[["trial_id", "prompt_style", "context_level"]].drop_duplicates()

        # Cross trial pools with conditions: all shown posts × all conditions
        shown = pool_stack.merge(conditions, on="trial_id", how="inner")

        # Mark selected
        sel_keys = sel[["post_id", "trial_id", "prompt_style", "context_level", "selected"]]
        shown = shown.merge(sel_keys,
                            on=["post_id", "trial_id", "prompt_style", "context_level"],
                            how="left")
        shown["selected"] = shown["selected"].fillna(0).astype(int)

        shown["provider"] = provider
        shown["model"]    = "_".join(d.name.split("_")[1:])
        frames.append(shown)

    if not frames:
        raise FileNotFoundError(
            f"No trial_results.csv found under {exp_dir}. "
            "Run run_llm_recommendation.py first."
        )

    trials = pd.concat(frames, ignore_index=True)
    if "context_level" not in trials.columns:
        trials["context_level"] = "none"

    # ── Join features ─────────────────────────────────────────────────────────
    # Get author_id from post_features, then merge all features
    post_with_author = post_df[["post_id", "author_id"]]
    post_features    = post_df.drop(columns=["author_id"], errors="ignore")

    combined = (
        trials
        .merge(post_with_author, on="post_id", how="left")
        .merge(post_features,   on="post_id", how="left")
        .merge(author_df,       on="author_id", how="left")
    )

    combined["trial_group"] = (
        combined["provider"]                      + "|" +
        combined["prompt_style"].fillna("")       + "|" +
        combined["context_level"].fillna("none")  + "|" +
        combined["trial_id"].astype(str)
    )

    return combined
