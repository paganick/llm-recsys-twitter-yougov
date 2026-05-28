#!/usr/bin/env python3
"""
Prepare the Twitter/X post pool for LLM recommendation experiments.

Merges tweet data with survey demographic data, anonymises identifiers, and
writes a pool CSV ready for run_llm_recommendation.py.

Input files (see --help for paths)
-----------------------------------
  --tweets   CSV with tweet content and pre-computed metadata
  --survey   CSV with ground-truth survey demographics, joined on user_id
             (pass --tweets-only if the input is already a merged file)

Expected tweet columns (required):
  user_id, text

Expected tweet columns (optional, passed through to pool):
  tweet_id, created_at, has_url, has_hashtag, has_mention, has_emoji,
  text_length, word_count, avg_word_length,
  is_reply, is_retweet, is_quote,
  user_followers_count, user_friends_count, user_statuses_count,
  user_favourites_count, user_verified, user_account_age_days, engagement_score,
  favorite_count, retweet_count, retweeted

Expected survey columns (required when --survey is provided):
  user_id, author_gender, author_partisanship, author_ideology, author_race

Expected survey columns (optional):
  author_age, author_education, author_income,
  author_marital_status, author_religiosity

Outputs
-------
  outputs/pools/twitter_pool.csv    post_id, user_id, text + all metadata
  outputs/pools/twitter_id_map.csv  user_id → original_user_id  (internal only)

Usage
-----
    # Separate tweet + survey files:
    python prepare_dataset.py --tweets data/tweets.csv --survey data/survey.csv

    # Pre-merged file (tweet + demographics already joined):
    python prepare_dataset.py --tweets data/prepared.csv --tweets-only

    # With custom pool size and seed:
    python prepare_dataset.py --tweets data/tweets.csv --survey data/survey.csv \\
        --pool-size 5000 --seed 42
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

OUT_DIR = Path("outputs/pools")

# Metadata columns passed through from tweet data to the pool (if present)
TWEET_METADATA_COLS = [
    # Temporal
    "created_at",
    # Style flags
    "has_url", "has_hashtag", "has_mention", "has_emoji",
    # Text metrics
    "text_length", "word_count", "avg_word_length",
    # Tweet type
    "is_reply", "is_retweet", "is_quote",
    # Author metadata
    "user_followers_count", "user_friends_count", "user_statuses_count",
    "user_favourites_count",
    "user_verified", "user_account_age_days", "engagement_score",
    # Post engagement metadata
    "favorite_count", "retweet_count", "retweeted",
]

# Demographic columns from survey data
SURVEY_COLS = [
    "author_gender", "author_partisanship", "author_ideology", "author_race",
    "author_age", "author_education", "author_income",
    "author_marital_status", "author_religiosity",
]


def load_tweets(path: Path) -> pd.DataFrame:
    """Load tweet CSV; normalise text/user columns."""
    # Use python engine + quotechar to handle embedded newlines in tweet text
    df = pd.read_csv(path, engine="python", on_bad_lines="warn")

    # Normalise user identifier column
    if "user_id" not in df.columns:
        for candidate in ("userid", "author_id", "screen_name", "username"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "user_id"})
                break
        else:
            print(f"ERROR: cannot find a user identifier column in {path}")
            sys.exit(1)

    # Normalise text column
    if "text" not in df.columns:
        for candidate in ("tweet", "content", "message", "body"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "text"})
                break
        else:
            print(f"ERROR: cannot find a text column in {path}")
            sys.exit(1)

    df["user_id"] = df["user_id"].astype(str)
    df["text"] = df["text"].astype(str)
    return df


def load_survey(path: Path) -> pd.DataFrame:
    """Load survey CSV; normalise user identifier."""
    df = pd.read_csv(path)
    if "user_id" not in df.columns:
        for candidate in ("userid", "author_id", "screen_name", "username"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "user_id"})
                break
        else:
            print(f"ERROR: cannot find a user identifier column in {path}")
            sys.exit(1)
    df["user_id"] = df["user_id"].astype(str)
    return df


def merge_data(tweets: pd.DataFrame, survey: pd.DataFrame) -> pd.DataFrame:
    """Inner-join tweets with survey data on user_id."""
    before = len(tweets)
    survey_cols_present = ["user_id"] + [c for c in SURVEY_COLS if c in survey.columns]
    merged = tweets.merge(survey[survey_cols_present], on="user_id", how="inner")
    n_matched = merged["user_id"].nunique()
    n_survey = survey["user_id"].nunique()
    print(f"  Tweets before merge:      {before:,}")
    print(f"  Survey respondents:       {n_survey:,}")
    print(f"  Users with both records:  {n_matched:,}")
    print(f"  Tweets after merge:       {len(merged):,}")
    missing = [c for c in SURVEY_COLS if c not in merged.columns]
    if missing:
        print(f"  WARNING: survey columns not found: {missing}")
    return merged


def anonymise(df: pd.DataFrame) -> tuple:
    """Assign anonymous post_id and user_id; return (pool, id_map)."""
    unique_users = sorted(df["user_id"].unique())
    user_map = {u: f"user_{i+1:05d}" for i, u in enumerate(unique_users)}

    df = df.copy().reset_index(drop=True)
    df["original_user_id"] = df["user_id"]
    df["user_id"] = df["user_id"].map(user_map)
    df["post_id"] = [f"post_{i+1:05d}" for i in range(len(df))]

    id_map = pd.DataFrame({
        "user_id": list(user_map.values()),
        "original_user_id": list(user_map.keys()),
    })
    return df, id_map


def build_pool_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Select and order columns for the pool CSV."""
    # Core columns always present
    core = ["post_id", "user_id", "text"]

    # Survey demographic columns (keep those present)
    demo_cols = [c for c in SURVEY_COLS if c in df.columns]

    # Tweet metadata columns (keep those present)
    meta_cols = [c for c in TWEET_METADATA_COLS if c in df.columns]

    keep = core + demo_cols + meta_cols
    return df[keep]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tweets", required=True,
                        help="Path to tweet CSV (or pre-merged CSV with --tweets-only)")
    parser.add_argument("--survey",
                        help="Path to survey demographics CSV (omit with --tweets-only)")
    parser.add_argument("--tweets-only", action="store_true",
                        help="Input file already contains merged tweet + demographics")
    parser.add_argument("--pool-size", type=int, default=None,
                        help="Max tweets to sample into pool (default: unlimited)")
    parser.add_argument("--max-chars", type=int, default=280,
                        help="Max tweet length in characters (default: 280)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--require-demographics", action="store_true",
                        help="Only keep users with all core demographic columns filled")
    args = parser.parse_args()

    tweets_path = Path(args.tweets)
    if not tweets_path.exists():
        print(f"ERROR: {tweets_path} not found.")
        sys.exit(1)

    print(f"\nLoading tweets from {tweets_path} ...")
    df = load_tweets(tweets_path)
    print(f"  {len(df):,} rows loaded, {df['user_id'].nunique():,} unique users")

    if not args.tweets_only:
        if not args.survey:
            print("ERROR: --survey is required unless --tweets-only is set.")
            sys.exit(1)
        survey_path = Path(args.survey)
        if not survey_path.exists():
            print(f"ERROR: {survey_path} not found.")
            sys.exit(1)
        print(f"\nLoading survey data from {survey_path} ...")
        survey = load_survey(survey_path)
        print(f"  {len(survey):,} survey respondents loaded")

        print("\nMerging ...")
        df = merge_data(df, survey)
    else:
        print("  Using pre-merged data (--tweets-only mode)")
        found = [c for c in SURVEY_COLS if c in df.columns]
        print(f"  Found {len(found)}/{len(SURVEY_COLS)} demographic columns")

    # Bin author_age into standard demographic ranges if it is numeric
    if "author_age" in df.columns and pd.api.types.is_numeric_dtype(df["author_age"]):
        bins   = [0, 24, 34, 44, 54, 64, 999]
        labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        df["author_age"] = pd.cut(
            df["author_age"], bins=bins, labels=labels, right=True
        ).astype(str).replace("nan", pd.NA)
        print(f"\nAuthor age binned into ranges: {labels}")

    # Length filter
    before = len(df)
    df = df[df["text"].str.len() <= args.max_chars].copy()
    print(f"\nLength filter (≤{args.max_chars} chars): {before - len(df):,} removed, {len(df):,} remain")

    # Optionally require core demographics to be non-null
    CORE_DEMO_COLS = ["author_gender", "author_partisanship", "author_ideology", "author_race"]
    if args.require_demographics:
        present_core = [c for c in CORE_DEMO_COLS if c in df.columns]
        if present_core:
            before = len(df)
            df = df.dropna(subset=present_core)
            df = df[~df[present_core].isin(["unknown", "Unknown", ""]).any(axis=1)]
            print(f"Demographic filter: {before - len(df):,} removed, {len(df):,} remain")

    # Sample
    if args.pool_size is not None and args.pool_size < len(df):
        df = df.sample(n=args.pool_size, random_state=args.seed).reset_index(drop=True)
        print(f"Sampled {args.pool_size:,} posts (seed={args.seed})")
    else:
        cap_msg = f" (cap={args.pool_size:,})" if args.pool_size else ""
        print(f"Using all {len(df):,} posts{cap_msg}")

    # Anonymise
    df, id_map = anonymise(df)
    pool = build_pool_csv(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pool.to_csv(OUT_DIR / "twitter_pool.csv", index=False)
    id_map.to_csv(OUT_DIR / "twitter_id_map.csv", index=False)

    demo_present = [c for c in SURVEY_COLS if c in pool.columns]
    meta_present = [c for c in TWEET_METADATA_COLS if c in pool.columns]

    print(f"\n{'='*55}")
    print(f"  Pool:       {OUT_DIR}/twitter_pool.csv  ({len(pool):,} posts, {pool['user_id'].nunique():,} users)")
    print(f"  ID map:     {OUT_DIR}/twitter_id_map.csv  (internal — do not redistribute)")
    print(f"  Demographics: {demo_present}")
    print(f"  Metadata:     {meta_present}")
    print(f"\n  Next step: python run_llm_recommendation.py --provider anthropic")


if __name__ == "__main__":
    main()
