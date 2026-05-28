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
    args = parser.parse_args()

    if not POOL_FILE.exists():
        print(f"ERROR: master pool not found at {POOL_FILE}")
        print("       Run first:  python pipeline/prepare_dataset.py --tweets <path> --survey <path>")
        sys.exit(1)

    print(f"Loading {POOL_FILE} …")
    pool = pd.read_csv(POOL_FILE, engine="python", on_bad_lines="warn")
    print(f"  {len(pool):,} posts loaded")

    # Sort by date; posts without a parseable date go to the end
    if "created_at" not in pool.columns:
        print("WARNING: no created_at column — using original row order as proxy for date")
    else:
        dates = pd.to_datetime(pool["created_at"], errors="coerce")
        pool = pool.assign(_date=dates).sort_values("_date", na_position="last").drop(columns="_date")
        pool = pool.reset_index(drop=True)

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

    POOLS_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(POOLS_DIR.glob("trial_*.csv"))
    if existing:
        print(f"\n{len(existing)} existing trial file(s) found — they will be overwritten.")

    print(f"\nWriting {n_t} pool files to {POOLS_DIR} …")
    for t, (s, e) in enumerate(buckets):
        bucket = pool.iloc[s:e]
        sample = bucket.sample(n=p_size, random_state=SEED_BASE + t)
        out    = POOLS_DIR / f"trial_{t:03d}.csv"
        sample.to_csv(out, index=False)
        if (t + 1) % 10 == 0 or t == n_t - 1:
            print(f"  {t+1}/{n_t} written")

    print(f"\n✓ Done. {n_t} pools of {p_size} posts each in {POOLS_DIR}/")
    print(f"  Date range covered: "
          f"{pool['created_at'].iloc[buckets[0][0]] if 'created_at' in pool.columns else 'n/a'}"
          f" → "
          f"{pool['created_at'].iloc[buckets[-1][1]-1] if 'created_at' in pool.columns else 'n/a'}")


if __name__ == "__main__":
    main()
