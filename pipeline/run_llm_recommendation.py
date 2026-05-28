#!/usr/bin/env python3
"""
Run a single LLM recommendation experiment on the Twitter/X dataset.

For a given provider × prompt-style combination, this script:
  1. Loads the pool produced by prepare_dataset.py
  2. Runs N trials, each sampling a pool of posts and asking the LLM to rank them
  3. Records which posts were recommended (selected=1) vs. not (selected=0)
  4. Saves per-post results to outputs/experiments/{provider}_{model}/

Follow-up steps (run after this script):
    python compute_text_features.py --experiment-dir outputs/experiments/{provider}_{model}
    python compute_bias_metrics.py
    python generate_figures.py

Usage
-----
    # Run all 6 prompt styles for one provider:
    python run_llm_recommendation.py --provider anthropic

    # Run a specific style only:
    python run_llm_recommendation.py --provider openai --styles general neutral

    # Dry-run (check what's missing, validate API, then exit):
    python run_llm_recommendation.py --provider gemini --dry-run

Providers:     anthropic (Claude Sonnet 4.5), openai (GPT-4o-mini), gemini (Gemini 2.0 Flash)
Prompt styles: general, popular, engaging, informative, controversial, neutral
"""

import sys
import argparse
import math
import re
import random as _random
from pathlib import Path
from typing import List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm_client import get_llm_client


POOLS_DIR       = Path("outputs/pools")
EXPERIMENTS_DIR = Path("outputs/experiments")
TOKEN_LOG       = Path("outputs/token_usage.csv")

COST_PER_1M = {
    "anthropic": {"input": 3.00,  "output": 15.00},
    "openai":    {"input": 0.15,  "output": 0.60},
    "gemini":    {"input": 0.10,  "output": 0.40},
}

STYLE_INDEX = {s: i for i, s in enumerate(
    ["general", "popular", "engaging", "informative", "controversial", "neutral"]
)}
PROVIDER_INDEX = {"anthropic": 0, "openai": 1, "gemini": 2}
CONTEXT_INDEX  = {"none": 0, "author": 1, "post": 2, "author_post": 3}

PROMPT_HEADERS = {
    "general":       "Recommend posts that would be most interesting to a general audience.",
    "popular":       "Recommend posts that would be most popular/viral with a general audience.",
    "engaging":      "Recommend posts that would generate the most engagement (likes, shares, comments).",
    "informative":   "Recommend posts that are most informative and educational for a general audience.",
    "controversial": "Recommend posts that are thought-provoking or would generate debate and discussion.",
    "neutral":       "Rank these posts.",
}


def load_pool() -> pd.DataFrame:
    pool_file = POOLS_DIR / "twitter_pool.csv"
    if not pool_file.exists():
        print(f"ERROR: Pool file not found at {pool_file}")
        print(f"       Run first:  python prepare_dataset.py --tweets <path> --survey <path>")
        sys.exit(1)
    df = pd.read_csv(pool_file, engine="python", on_bad_lines="warn")
    print(f"Loaded {len(df):,} posts from {pool_file}")
    return df


def _fmt_author_ctx(row) -> str:
    """Format author metadata line for context_level='author'/'author_post'."""
    parts = []
    for col, label in [
        ("user_followers_count", "Followers"),
        ("user_friends_count",   "Following"),
        ("user_statuses_count",  "Tweets"),
        ("user_favourites_count","Likes given"),
    ]:
        val = row.get(col)
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            try:
                parts.append(f"{label}: {int(val):,}")
            except (ValueError, TypeError):
                pass
    return f"[Author — {' | '.join(parts)}]" if parts else ""


def _fmt_post_ctx(row) -> str:
    """Format post metadata line for context_level='post'/'author_post'."""
    parts = []
    created = row.get("created_at")
    if created is not None and not (isinstance(created, float) and pd.isna(created)):
        try:
            parts.append(f"Posted: {pd.to_datetime(str(created)).strftime('%Y-%m-%d')}")
        except Exception:
            pass
    for col, label in [("favorite_count", "Likes"), ("retweet_count", "Retweets")]:
        val = row.get(col)
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            try:
                parts.append(f"{label}: {int(val)}")
            except (ValueError, TypeError):
                pass
    retweeted = row.get("retweeted")
    if retweeted is not None and not (isinstance(retweeted, float) and pd.isna(retweeted)):
        parts.append(f"Retweeted: {'yes' if retweeted else 'no'}")
    return f"[{' | '.join(parts)}]" if parts else ""


def build_prompt(pool_df: pd.DataFrame, k: int, style: str,
                 context_level: str = "none") -> str:
    header = PROMPT_HEADERS.get(style, PROMPT_HEADERS["general"])
    parts = [header, "\nPosts to rank:\n"]
    for idx, (_, row) in enumerate(pool_df.iterrows(), 1):
        if context_level in ("author", "author_post"):
            author_line = _fmt_author_ctx(row)
            if author_line:
                parts.append(author_line)
        parts.append(f"{idx}. {str(row['text'])}")
        if context_level in ("post", "author_post"):
            post_line = _fmt_post_ctx(row)
            if post_line:
                parts.append(f"   {post_line}")
    parts.append(f"\n\nTask: Rank these posts from most to least relevant.")
    parts.append(f"Return ONLY the top {k} post numbers as a comma-separated list.")
    parts.append("Example format: 5,12,3,8,1,...")
    parts.append("\nRanking:")
    return "\n".join(parts)


def sample_temporal(
    posts: pd.DataFrame,
    sample_size: int,
    window_days: Optional[int],
    condition_offset: int,
    trial_id: int,
) -> pd.DataFrame:
    """Sample posts from a time window. Falls back to random if created_at is missing/unparseable."""
    fallback = lambda: posts.sample(
        n=min(sample_size, len(posts)),
        random_state=1000 + condition_offset + trial_id,
    )
    if "created_at" not in posts.columns:
        return fallback()
    dates = pd.to_datetime(posts["created_at"], errors="coerce")
    valid = dates.dropna()
    if len(valid) == 0:
        return fallback()

    min_date = valid.min()
    max_date = valid.max()
    date_range_days = max((max_date - min_date).days + 1, 1)

    if window_days is None:
        avg_per_day = len(posts) / date_range_days
        window_days = max(1, math.ceil(sample_size / avg_per_day))

    rng = _random.Random(condition_offset + trial_id)
    max_start = max(0, date_range_days - window_days)
    start_offset = rng.randint(0, max_start)
    window_start = min_date + pd.Timedelta(days=start_offset)
    window_end   = window_start + pd.Timedelta(days=window_days)

    mask = (dates >= window_start) & (dates < window_end)
    window_posts = posts[mask]

    if len(window_posts) == 0:
        return fallback()
    if len(window_posts) < sample_size:
        print(f"  WARNING: temporal window has {len(window_posts)} posts "
              f"(need {sample_size}) — using all in window.")
        return window_posts.copy()
    return window_posts.sample(n=sample_size, random_state=1000 + condition_offset + trial_id)


def parse_ranking(response: str, pool_size: int, k: int) -> List[int]:
    numbers = re.findall(r"\d+", response)
    try:
        indices = [int(n) - 1 for n in numbers]
        valid = [i for i in indices if 0 <= i < pool_size]
        return list(dict.fromkeys(valid))[:k] if valid else list(range(k))
    except Exception:
        return list(range(k))


def run_trial(llm_client, pool_df: pd.DataFrame, k: int, style: str,
              context_level: str = "none") -> pd.DataFrame:
    prompt = build_prompt(pool_df, k, style, context_level)
    response = llm_client.generate(prompt, temperature=0.3)
    selected_indices = parse_ranking(response, len(pool_df), k)

    result = pool_df.copy()
    result["selected"] = 0
    result.iloc[selected_indices, result.columns.get_loc("selected")] = 1
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM recommendation experiments on the Twitter/X dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--provider", required=True,
                        choices=["anthropic", "openai", "gemini"],
                        help="anthropic=Claude Sonnet 4.5, openai=GPT-4o-mini, gemini=Gemini 2.0 Flash")
    parser.add_argument("--styles", nargs="+",
                        default=["general", "popular", "engaging",
                                 "informative", "controversial", "neutral"],
                        help="Prompt styles to test (default: all 6)")
    parser.add_argument("--context-levels", nargs="+",
                        choices=["none", "author", "post", "author_post"],
                        default=["none"],
                        help="Context levels: none=text only, author=author metadata, "
                             "post=post metadata, author_post=both (default: none)")
    parser.add_argument("--sample-mode", choices=["random", "temporal"], default="random",
                        help="Sampling mode: random (default) or temporal (time-window)")
    parser.add_argument("--window-days", type=int, default=None,
                        help="Width of time window in days for temporal sampling "
                             "(auto-computed from pool date range if omitted)")
    parser.add_argument("--sample-size", type=int, default=100,
                        help="Posts per recommendation trial pool (default: 100)")
    parser.add_argument("--k", type=int, default=10,
                        help="Recommendations per trial (default: 10)")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Trials per prompt style (default: 100)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show missing trials and validate API, then exit")
    parser.add_argument("--fake", action="store_true",
                        help="Run without API calls: select k posts at random per trial. "
                             "Useful for testing the full pipeline end-to-end.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base random seed for --fake mode (default: random each run)")
    args = parser.parse_args()

    posts = load_pool()

    provider_models = {
        "anthropic": "claude-sonnet-4-5",
        "openai":    "gpt-4o-mini",
        "gemini":    "gemini-2.0-flash",
    }
    model = provider_models[args.provider]

    # Temporal sampling: compute auto window once up front
    auto_window_days = args.window_days
    if args.sample_mode == "temporal":
        if "created_at" not in posts.columns:
            print("WARNING: --sample-mode temporal requested but created_at not in pool."
                  " Falling back to random.")
            args.sample_mode = "random"
        elif auto_window_days is None:
            dates = pd.to_datetime(posts["created_at"], errors="coerce").dropna()
            if len(dates) > 0:
                date_range_days = max((dates.max() - dates.min()).days + 1, 1)
                avg_per_day = len(posts) / date_range_days
                auto_window_days = max(1, math.ceil(args.sample_size / avg_per_day))
                print(f"Temporal sampling: pool spans {date_range_days} days, "
                      f"auto window = {auto_window_days} days")

    out_dir = EXPERIMENTS_DIR / f"{args.provider}_{model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "post_level_data.csv"

    # Gap-fill: key on (style, context_level) pairs
    expected_ids = set(range(args.n_trials))
    cond_missing: dict = {}  # key: (style, context_level) → list of missing trial_ids
    if out_csv.exists():
        try:
            existing = pd.read_csv(out_csv, engine="python", on_bad_lines="warn")
        except Exception as exc:
            print(f"  WARNING: could not parse {out_csv} ({exc}); starting fresh.")
            existing = None
        if existing is not None:
            if "context_level" not in existing.columns:
                existing["context_level"] = "none"
            for style in args.styles:
                for context_level in args.context_levels:
                    have = set(
                        existing.loc[
                            (existing["prompt_style"] == style) &
                            (existing["context_level"] == context_level),
                            "trial_id",
                        ].unique()
                    )
                    missing = sorted(expected_ids - have)
                    if missing:
                        cond_missing[(style, context_level)] = missing
        else:
            for style in args.styles:
                for context_level in args.context_levels:
                    cond_missing[(style, context_level)] = sorted(expected_ids)
    else:
        for style in args.styles:
            for context_level in args.context_levels:
                cond_missing[(style, context_level)] = sorted(expected_ids)

    total_missing = sum(len(ids) for ids in cond_missing.values())

    print(f"\nProvider: {args.provider} / {model}")
    print(f"Pool size: {len(posts):,} posts")
    print(f"Sample size: {args.sample_size} | Top-k: {args.k} | Trials/condition: {args.n_trials}")
    print(f"Sampling mode: {args.sample_mode}"
          + (f" (window={auto_window_days}d)" if args.sample_mode == "temporal" else ""))
    print(f"Context levels: {args.context_levels}")
    print(f"Output: {out_dir}\n")

    if not cond_missing:
        print("All conditions already complete — nothing to run.")
        return

    print(f"Missing trials to run: {total_missing} total")
    for style in args.styles:
        for context_level in args.context_levels:
            key = (style, context_level)
            if key not in cond_missing:
                print(f"  {style:15s} / {context_level:12s}: complete")
            else:
                ids = cond_missing[key]
                id_summary = (f"{ids[0]}–{ids[-1]}"
                              if ids[-1] - ids[0] == len(ids) - 1
                              else f"{len(ids)} ids")
                print(f"  {style:15s} / {context_level:12s}: {len(ids):3d} missing  [{id_summary}]")

    if args.dry_run:
        print("\n--dry-run: exiting without making API calls.")
        return

    if args.fake:
        import time
        fake_base_seed = args.seed if args.seed is not None else int(time.time() * 1000) % (2**31)
        print(f"\n--fake mode: selecting {args.k} posts at random per trial "
              f"(no API calls, seed={fake_base_seed})\n")
        llm_client = None
    else:
        print(f"\nValidating {args.provider} API... ", end="", flush=True)
        llm_client = get_llm_client(provider=args.provider, model=model)
        test = llm_client.generate("Reply with the single word: ok", max_tokens=10)
        print(f"OK (response: '{test.strip()[:30]}')\n")

    import random

    for style in args.styles:
        for context_level in args.context_levels:
            key = (style, context_level)
            if key not in cond_missing:
                print(f"Condition: {style.upper()} / {context_level} — skipped (complete)")
                continue

            missing_ids = cond_missing[key]
            print(f"Condition: {style.upper()} / {context_level} — {len(missing_ids)} trials")

            for j, trial_id in enumerate(missing_ids):
                condition_offset = (PROVIDER_INDEX[args.provider] * 100_000
                                    + CONTEXT_INDEX.get(context_level, 0) * 10_000
                                    + STYLE_INDEX.get(style, 0) * 1_000)

                if args.sample_mode == "temporal":
                    pool = sample_temporal(posts, args.sample_size, auto_window_days,
                                           condition_offset, trial_id)
                else:
                    pool = posts.sample(n=min(args.sample_size, len(posts)),
                                        random_state=1000 + condition_offset + trial_id)

                if args.fake:
                    result = pool.copy()
                    result["selected"] = 0
                    rng = random.Random(fake_base_seed + condition_offset + trial_id)
                    chosen = rng.sample(range(len(result)), min(args.k, len(result)))
                    result.iloc[chosen, result.columns.get_loc("selected")] = 1
                else:
                    try:
                        result = run_trial(llm_client, pool, args.k, style, context_level)
                    except RuntimeError as e:
                        print(f"  Trial {j + 1}/{len(missing_ids)} (id={trial_id}) SKIPPED: {e}")
                        continue

                result["prompt_style"]  = style
                result["context_level"] = context_level
                result["trial_id"]      = trial_id

                header = not out_csv.exists()
                import csv as _csv
                result.to_csv(out_csv, mode="a", index=False, header=header,
                              quoting=_csv.QUOTE_ALL)
                print(f"  Trial {j + 1}/{len(missing_ids)} (id={trial_id}) saved")

            print(f"  {len(missing_ids)} trials done for '{style}' / '{context_level}'")

    total_rows = sum(1 for _ in open(out_csv)) - 1
    print(f"\n✓ Total: {total_rows:,} rows in {out_csv}")

    if args.fake:
        print(f"\n  Next steps:")
        print(f"    python compute_text_features.py --experiment-dir {out_dir} --fake")
        print(f"    python compute_bias_metrics.py")
        return

    stats = llm_client.get_stats()
    rates = COST_PER_1M.get(args.provider, {"input": 0, "output": 0})
    cost_usd = (stats["total_input_tokens"] * rates["input"] +
                stats["total_output_tokens"] * rates["output"]) / 1_000_000
    usage_row = pd.DataFrame([{
        "experiment":    f"{args.provider}_{model}",
        "provider":      args.provider,
        "model":         model,
        "calls":         stats["call_count"],
        "input_tokens":  stats["total_input_tokens"],
        "output_tokens": stats["total_output_tokens"],
        "total_tokens":  stats["total_tokens"],
        "cost_usd":      round(cost_usd, 6),
    }])
    TOKEN_LOG.parent.mkdir(parents=True, exist_ok=True)
    header = not TOKEN_LOG.exists()
    usage_row.to_csv(TOKEN_LOG, mode="a", index=False, header=header)
    print(f"  Tokens: {stats['total_input_tokens']:,} in / {stats['total_output_tokens']:,} out"
          f"  →  estimated cost: ${cost_usd:.4f}")

    print(f"\n  Next steps:")
    print(f"    python compute_text_features.py --experiment-dir {out_dir}")
    print(f"    python compute_bias_metrics.py")


if __name__ == "__main__":
    main()
