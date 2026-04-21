#!/usr/bin/env python3
"""
Compute NLP-derived text features for posts in an experiment output.

Features computed per post (not already in the pool from prepare_dataset.py):
  - sentiment_polarity, sentiment_subjectivity  (VADER)
  - primary_topic, polarization_score           (Cardiff NLP RoBERTa)
  - toxicity                                    (Detoxify)

Style indicators (has_emoji, has_hashtag, has_mention, has_url) and text
metrics (avg_word_length, text_length, word_count) are expected to already
be present in the pool CSV (pre-computed from the raw data).  If they are
missing, they will be computed here as a fallback.

Features are cached in outputs/cache/twitter_features.parquet, keyed on
post_id, so repeated runs only recompute new posts.

Input:  outputs/experiments/{provider}_{model}/post_level_data.csv
Cache:  outputs/cache/twitter_features.parquet
Output: outputs/experiments/{provider}_{model}/post_level_data.csv  (enriched)

Usage
-----
    python compute_text_features.py \\
        --experiment-dir outputs/experiments/anthropic_claude-sonnet-4-5
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.text_features import infer_tweet_metadata

CACHE_DIR = Path("outputs/cache")
CACHE_FILE = CACHE_DIR / "twitter_features.parquet"

# Features computed by this script (NLP-derived)
COMPUTED_FEATURES = [
    "sentiment_polarity", "sentiment_subjectivity",
    "primary_topic", "polarization_score",
    "toxicity",
]


def load_cache() -> pd.DataFrame:
    if CACHE_FILE.exists():
        return pd.read_parquet(CACHE_FILE)
    return pd.DataFrame(columns=["post_id"])


def save_cache(cache: pd.DataFrame):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.to_parquet(CACHE_FILE, index=False)


TOPICS = [
    "news_&_social_concern", "arts_&_culture", "sports",
    "science_&_technology", "business_&_entrepreneurs", "politics",
    "celebrity_&_pop_culture", "diaries_&_daily_life", "family",
    "fitness_&_health", "food_&_dining", "travel_&_adventure",
    "gaming", "learning_&_educational", "other",
]


def fake_features(post_ids: pd.Series, seed: int = 42) -> pd.DataFrame:
    """Generate plausible-looking random feature values for testing."""
    import numpy as np
    rng = np.random.default_rng(seed)
    n = len(post_ids)
    return pd.DataFrame({
        "post_id":               post_ids.values,
        "sentiment_polarity":    rng.uniform(-1, 1, n).round(4),
        "sentiment_subjectivity":rng.uniform(0, 1, n).round(4),
        "primary_topic":         rng.choice(TOPICS, n),
        "polarization_score":    rng.uniform(0, 1, n).round(4),
        "toxicity":              rng.uniform(0, 0.3, n).round(4),
    })


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--experiment-dir", required=True,
                        help="Path to experiment directory containing post_level_data.csv")
    parser.add_argument("--fake", action="store_true",
                        help="Assign random feature values without loading any NLP models. "
                             "Use together with --fake in run_llm_recommendation.py for "
                             "end-to-end pipeline testing without API keys or GPU.")
    args = parser.parse_args()

    exp_dir  = Path(args.experiment_dir)
    csv_path = exp_dir / "post_level_data.csv"

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        print(f"       Run first:  python run_llm_recommendation.py --provider <provider>")
        sys.exit(1)

    print(f"Loading experiment data from {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df):,} rows, {df['post_id'].nunique():,} unique post_ids")

    unique_posts = df[["post_id", "text"]].drop_duplicates("post_id")

    if args.fake:
        print(f"\n--fake mode: assigning random feature values (no NLP models loaded)")
        new_rows = fake_features(unique_posts["post_id"])
        cache = new_rows
    else:
        cache = load_cache()
        cached_ids = set(cache["post_id"]) if not cache.empty else set()
        missing = unique_posts[~unique_posts["post_id"].isin(cached_ids)]

        if missing.empty:
            print(f"All {len(unique_posts):,} posts already in cache — skipping computation.")
        else:
            print(f"\nCache: {len(cached_ids):,} posts already computed, "
                  f"{len(missing):,} new posts to process.")
            features_df = infer_tweet_metadata(
                missing,
                text_column="text",
                sentiment_method="vader",
                topic_method="roberta",
                include_gender=False,
                include_political=False,
            )
            keep_cols = ["post_id"] + [c for c in COMPUTED_FEATURES if c in features_df.columns]
            fallback_cols = [
                c for c in features_df.columns
                if c not in keep_cols and c not in ("text",)
            ]
            keep_cols = keep_cols + fallback_cols
            new_rows = features_df[[c for c in keep_cols if c in features_df.columns]].copy()

            cache = pd.concat([cache, new_rows], ignore_index=True)
            cache = cache.drop_duplicates(subset=["post_id"], keep="last")
            save_cache(cache)
            print(f"  Cache updated: {len(cache):,} posts total → {CACHE_FILE}")

    # Merge into experiment CSV
    feature_cols = [c for c in cache.columns if c != "post_id"]
    new_cols = [c for c in feature_cols if c not in df.columns]
    if new_cols:
        df = df.merge(cache[["post_id"] + new_cols], on="post_id", how="left")
    else:
        df = df.drop(columns=[c for c in feature_cols if c in df.columns], errors="ignore")
        df = df.merge(cache[["post_id"] + feature_cols], on="post_id", how="left")

    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved enriched data ({len(df):,} rows, {len(df.columns)} columns) to {csv_path}")
    print(f"  Next step: python compute_bias_metrics.py")


if __name__ == "__main__":
    main()
