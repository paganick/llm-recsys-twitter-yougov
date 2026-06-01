#!/usr/bin/env python3
"""
Pre-generate shared trial pools for the LLM recommendation experiment.

Produces N_TRIALS pool files (trial_000.csv … trial_099.csv) so that every
model, prompt style, and context level evaluates *identical* post sets.

Strategy
--------
1. Load the master pool and sort by date (oldest → newest).
2. Split into N_TRIALS equal-count temporal buckets (~552 posts each for a
   ~55k pool).  Equal-count guarantees every bucket has enough posts; equal-
   time intervals would produce sparse buckets in low-activity periods.
3. From each bucket t, sample POOL_SIZE posts with a fixed seed derived only
   from t.  All experimental conditions share this seed, so they all receive
   the same POOL_SIZE posts for trial t.
4. Save each pool to outputs/pools/trial_{t:03d}.csv.

Usage
-----
    python pipeline/prepare_pools.py

    # Custom sizes:
    python pipeline/prepare_pools.py --n-trials 100 --pool-size 100

    # Preview what would be generated without writing files:
    python pipeline/prepare_pools.py --dry-run
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

POOLS_DIR  = Path("outputs/pools")
POOL_FILE  = POOLS_DIR / "twitter_pool.csv"

N_TRIALS   = 100
POOL_SIZE  = 100
SEED_BASE  = 42   # trial t gets seed SEED_BASE + t

# Columns that capture a snapshot at tweet time — they belong in post_features
# (not author_features) because they can change between tweets.
# Simple text features (has_url, word_count, etc.) are computed in Step 3
# (compute_text_features.py) and passed through here if already present in input.
POST_FEATURE_COLS = [
    "created_at",
    "has_url", "has_hashtag", "has_mention", "has_emoji",
    "word_count", "avg_word_length",
    "is_reply", "is_retweet", "is_quote",
    "user_followers_count", "user_friends_count",
    "user_statuses_count", "user_favourites_count",
    "user_verified", "user_account_age_days", "engagement_score",
    "favorite_count", "retweet_count", "retweeted",
]

# Stable survey-derived demographics — one row per author.
AUTHOR_FEATURE_COLS = [
    "author_gender", "author_partisanship", "author_ideology", "author_race",
    "author_age", "author_education", "author_income",
    "author_marital_status", "author_religiosity",
]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-trials",  type=int, default=N_TRIALS,
                        help=f"Number of trial pools to generate (default: {N_TRIALS})")
    parser.add_argument("--pool-size", type=int, default=POOL_SIZE,
                        help=f"Posts per trial pool (default: {POOL_SIZE})")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print bucket statistics and exit without writing files")
    parser.add_argument("--pools-dir", type=Path, default=POOLS_DIR,
                        help=f"Directory for pool files (default: {POOLS_DIR})")
    args = parser.parse_args()

    pools_dir = args.pools_dir
    pool_file = pools_dir / "twitter_pool.csv"

    if not pool_file.exists():
        print(f"ERROR: master pool not found at {pool_file}")
        print("       Place twitter_pool.csv there before running.")
        sys.exit(1)

    print(f"Loading {pool_file} …")
    pool = pd.read_csv(pool_file, engine="python", on_bad_lines="warn")
    print(f"  {len(pool):,} posts loaded")

    # Rename user_id → author_id for consistency with the three-table schema.
    if "user_id" in pool.columns and "author_id" not in pool.columns:
        pool = pool.rename(columns={"user_id": "author_id"})

    # Strip embedded newlines from text before any further processing.
    pool["text"] = pool["text"].astype(str).str.replace(r"\r?\n", " ", regex=True).str.strip()

    pools_dir.mkdir(parents=True, exist_ok=True)

    # Sort by date then post_id as tiebreaker — fully deterministic regardless
    # of platform or pandas version (avoids unstable-sort ambiguity on ties).
    if "created_at" not in pool.columns:
        print("WARNING: no created_at column — sorting by post_id only")
        pool = pool.sort_values("post_id", kind="mergesort").reset_index(drop=True)
    else:
        dates = pd.to_datetime(pool["created_at"], format="mixed", errors="coerce")
        n_bad = dates.isna().sum()
        if n_bad:
            print(f"  WARNING: {n_bad:,} posts have unparseable created_at — dropping them")
            pool = pool[dates.notna()].copy()
            dates = dates[dates.notna()]
        sort_cols = ["_date", "post_id"] if "post_id" in pool.columns else ["_date"]
        pool = (pool.assign(_date=dates)
                    .sort_values(sort_cols, kind="mergesort")
                    .drop(columns="_date")
                    .reset_index(drop=True))

    n      = len(pool)
    n_t    = args.n_trials
    p_size = args.pool_size

    # Equal-count split: bucket t gets rows [start_t, end_t)
    bucket_size = n // n_t
    remainder   = n % n_t

    buckets = []
    start = 0
    for t in range(n_t):
        # Distribute the remainder one row at a time across the first buckets
        end = start + bucket_size + (1 if t < remainder else 0)
        buckets.append((start, end))
        start = end

    min_bucket = min(e - s for s, e in buckets)
    max_bucket = max(e - s for s, e in buckets)

    print(f"\nBucket statistics:")
    print(f"  Trials:       {n_t}")
    print(f"  Posts/bucket: {min_bucket}–{max_bucket}  "
          f"(need ≥ {p_size} — {'OK' if min_bucket >= p_size else 'TOO FEW'})")

    if min_bucket < p_size:
        print(f"ERROR: smallest bucket has {min_bucket} posts but pool-size={p_size}.")
        print("       Reduce --pool-size or --n-trials.")
        sys.exit(1)

    if args.dry_run:
        print("\n--dry-run: no files written.")
        return

    existing = sorted(pools_dir.glob("trial_*.csv"))
    if existing:
        print(f"\n{len(existing)} existing trial file(s) found — they will be overwritten.")

    print(f"\nWriting {n_t} pool files to {pools_dir} …")
    samples = []
    for t, (s, e) in enumerate(buckets):
        bucket = pool.iloc[s:e]
        sample = bucket.sample(n=p_size, random_state=SEED_BASE + t)
        out    = pools_dir / f"trial_{t:03d}.csv"
        sample.to_csv(out, index=False)
        samples.append(sample)
        if (t + 1) % 10 == 0 or t == n_t - 1:
            print(f"  {t+1}/{n_t} written")

    # ── Write post_features.csv and author_features.csv ───────────────────────
    # Only include posts/authors that actually appear in at least one trial.
    shown = pd.concat(samples, ignore_index=True).drop_duplicates(subset="post_id").copy()
    post_cols = (["post_id", "author_id", "text"]
                 + [c for c in POST_FEATURE_COLS if c in pool.columns])
    shown[post_cols].to_csv(pools_dir / "post_features.csv", index=False)
    print(f"  post_features.csv:   {len(shown):,} unique shown posts")

    author_cols = ["author_id"] + [c for c in AUTHOR_FEATURE_COLS if c in pool.columns]
    author_df = shown[author_cols].drop_duplicates(subset="author_id")
    author_df.to_csv(pools_dir / "author_features.csv", index=False)
    print(f"  author_features.csv: {len(author_df):,} unique shown authors")

    print(f"\n✓ Done.")
    print(f"  {n_t} trial pools of {p_size} posts each → {pools_dir}/trial_*.csv")
    print(f"  post_features.csv + author_features.csv → {pools_dir}/")
    if "created_at" in pool.columns:
        print(f"  Date range: {pool['created_at'].iloc[buckets[0][0]]}"
              f" → {pool['created_at'].iloc[buckets[-1][1]-1]}")
    print(f"\n  Next step: python pipeline/compute_text_features.py")


if __name__ == "__main__":
    main()
