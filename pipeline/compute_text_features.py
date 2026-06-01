#!/usr/bin/env python3
"""
Compute all text-derived features for the posts in post_features.csv.

Simple text features computed per post (fast, no ML models):
  - has_url, has_hashtag, has_mention, has_emoji
  - word_count, avg_word_length

NLP features computed per post (requires model downloads on first run):
  - sentiment_polarity, sentiment_subjectivity  (VADER)
  - primary_topic, polarization_score           (Cardiff NLP RoBERTa)
  - toxicity                                    (Detoxify)

Features are cached in outputs/cache/twitter_features.parquet, keyed on
post_id, so repeated runs only recompute new posts.

Input:  outputs/pools/post_features.csv
Cache:  outputs/cache/twitter_features.parquet
Output: outputs/pools/post_features.csv  (enriched in-place)

Usage
-----
    python pipeline/compute_text_features.py
    python pipeline/compute_text_features.py --fake   # no NLP models, random values
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.text_features import infer_tweet_metadata

POOLS_DIR  = Path("outputs/pools")
POST_FILE  = POOLS_DIR / "post_features.csv"
CACHE_DIR  = Path("outputs/cache")
CACHE_FILE = CACHE_DIR / "twitter_features.parquet"

COMPUTED_FEATURES = [
    "sentiment_polarity", "sentiment_subjectivity",
    "primary_topic", "polarization_score",
    "toxicity",
]

_TOXICITY_BATCH = 512


def load_cache() -> pd.DataFrame:
    if CACHE_FILE.exists():
        return pd.read_parquet(CACHE_FILE)
    return pd.DataFrame(columns=["post_id"])


def save_cache(cache: pd.DataFrame):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.to_parquet(CACHE_FILE, index=False)


def fake_features(post_ids: pd.Series) -> pd.DataFrame:
    import random
    rng = random.Random(42)
    topics = [
        "news_&_social_concern", "arts_&_culture", "sports",
        "science_&_technology", "business_&_entrepreneurs", "politics",
        "celebrity_&_pop_culture", "diaries_&_daily_life", "other",
    ]
    rows = []
    for pid in post_ids:
        rows.append({
            "post_id":                pid,
            "sentiment_polarity":     round(rng.uniform(-1, 1), 4),
            "sentiment_subjectivity": round(rng.uniform(0, 1), 4),
            "primary_topic":          rng.choice(topics),
            "polarization_score":     round(rng.uniform(0, 1), 4),
            "toxicity":               round(rng.uniform(0, 0.3), 4),
        })
    return pd.DataFrame(rows)


def _fill_null_features(cache, texts_df, null_cols):
    new = infer_tweet_metadata(
        texts_df, text_column="text",
        sentiment_method="vader", topic_method="roberta",
        include_gender=False, include_political=False,
    )
    for col in null_cols:
        if col in new.columns:
            cache = cache.merge(
                new[["post_id", col]].rename(columns={col: col + "_new"}),
                on="post_id", how="left",
            )
            mask = cache[col].isna()
            cache.loc[mask, col] = cache.loc[mask, col + "_new"]
            cache = cache.drop(columns=[col + "_new"])
    return cache


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pools-dir", type=Path, default=POOLS_DIR,
                        help=f"Directory containing post_features.csv (default: {POOLS_DIR})")
    parser.add_argument("--fake", action="store_true",
                        help="Assign random feature values without loading any NLP models. "
                             "Use together with --fake in run_llm_recommendation.py for "
                             "end-to-end pipeline testing without API keys or GPU.")
    args = parser.parse_args()

    post_file = args.pools_dir / "post_features.csv"

    if not post_file.exists():
        print(f"ERROR: {post_file} not found.")
        print("       Run first:  python pipeline/prepare_pools.py")
        sys.exit(1)

    print(f"Loading {post_file} ...")
    df = pd.read_csv(post_file, engine="python", on_bad_lines="warn")
    df["text"] = df["text"].astype(str).str.replace(r"\r?\n", " ", regex=True).str.strip()
    print(f"  {len(df):,} posts, {df['post_id'].nunique():,} unique post_ids")

    # ── Simple text features (fast, no ML) ───────────────────────────────────
    text = df["text"]
    simple = {}
    if "has_url" not in df.columns:
        simple["has_url"]     = text.str.contains(r"https?://", regex=True).astype(int)
    if "has_hashtag" not in df.columns:
        simple["has_hashtag"] = text.str.contains(r"#\w+", regex=True).astype(int)
    if "has_mention" not in df.columns:
        simple["has_mention"] = text.str.contains(r"@\w+", regex=True).astype(int)
    if "has_emoji" not in df.columns:
        simple["has_emoji"]   = text.str.contains(
            r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
            r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF☀-⛿✀-➿]",
            regex=True).astype(int)
    if "word_count" not in df.columns:
        simple["word_count"]      = text.str.split().str.len().fillna(0).astype(int)
    if "avg_word_length" not in df.columns:
        simple["avg_word_length"] = text.apply(
            lambda t: round(sum(len(w) for w in t.split()) / len(t.split()), 3)
            if t.split() else 0.0)
    if simple:
        for col, vals in simple.items():
            df[col] = vals
        print(f"  Simple text features computed: {list(simple)}")

    if args.fake:
        print(f"\n--fake mode: assigning random feature values (no NLP models loaded)")
        new_rows = fake_features(df["post_id"])
        cache = new_rows
    else:
        cache = load_cache()
        cached_ids = set(cache["post_id"]) if not cache.empty else set()
        missing = df[~df["post_id"].isin(cached_ids)][["post_id", "text"]].drop_duplicates("post_id")

        if missing.empty:
            print(f"All {len(df):,} posts already in cache — skipping computation.")

            null_cols = [c for c in COMPUTED_FEATURES
                         if c in cache.columns and cache[c].isna().any()]
            if null_cols:
                null_ids = cache[cache[null_cols].isna().any(axis=1)]["post_id"]
                null_texts = df[df["post_id"].isin(null_ids)][["post_id", "text"]]
                if not null_texts.empty:
                    print(f"  Re-filling {len(null_ids):,} posts with null columns: {null_cols}")
                    cache = _fill_null_features(cache, null_texts, null_cols)
                    save_cache(cache)
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
            fallback_cols = [c for c in features_df.columns
                             if c not in keep_cols and c != "text"]
            new_rows = features_df[[c for c in keep_cols + fallback_cols
                                    if c in features_df.columns]].copy()
            cache = pd.concat([cache, new_rows], ignore_index=True)
            cache = cache.drop_duplicates(subset=["post_id"], keep="last")
            save_cache(cache)
            print(f"  Cache updated: {len(cache):,} posts total → {CACHE_FILE}")

    # Merge NLP features into post_features.csv
    feature_cols = [c for c in cache.columns if c != "post_id"]
    new_cols = [c for c in feature_cols if c not in df.columns]
    if new_cols:
        df = df.merge(cache[["post_id"] + new_cols], on="post_id", how="left")
    else:
        df = df.drop(columns=[c for c in feature_cols if c in df.columns], errors="ignore")
        df = df.merge(cache[["post_id"] + feature_cols], on="post_id", how="left")

    df.to_csv(post_file, index=False)
    print(f"\n✓ Saved enriched post_features ({len(df):,} posts, {len(df.columns)} columns)"
          f" → {post_file}")
    print(f"  Next step: python pipeline/run_llm_recommendation.py --provider anthropic")


if __name__ == "__main__":
    main()
