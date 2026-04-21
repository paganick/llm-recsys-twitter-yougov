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
import re
from pathlib import Path
from typing import List

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
    df = pd.read_csv(pool_file)
    print(f"Loaded {len(df):,} posts from {pool_file}")
    return df


def build_prompt(pool_df: pd.DataFrame, k: int, style: str) -> str:
    header = PROMPT_HEADERS.get(style, PROMPT_HEADERS["general"])
    parts = [header, "\nPosts to rank:\n"]
    for idx, (_, row) in enumerate(pool_df.iterrows(), 1):
        parts.append(f"{idx}. {str(row['text'])}")
    parts.append(f"\n\nTask: Rank these posts from most to least relevant.")
    parts.append(f"Return ONLY the top {k} post numbers as a comma-separated list.")
    parts.append("Example format: 5,12,3,8,1,...")
    parts.append("\nRanking:")
    return "\n".join(parts)


def parse_ranking(response: str, pool_size: int, k: int) -> List[int]:
    numbers = re.findall(r"\d+", response)
    try:
        indices = [int(n) - 1 for n in numbers]
        valid = [i for i in indices if 0 <= i < pool_size]
        return list(dict.fromkeys(valid))[:k] if valid else list(range(k))
    except Exception:
        return list(range(k))


def run_trial(llm_client, pool_df: pd.DataFrame, k: int, style: str) -> pd.DataFrame:
    prompt = build_prompt(pool_df, k, style)
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
    args = parser.parse_args()

    posts = load_pool()

    provider_models = {
        "anthropic": "claude-sonnet-4-5",
        "openai":    "gpt-4o-mini",
        "gemini":    "gemini-2.0-flash",
    }
    model = provider_models[args.provider]

    out_dir = EXPERIMENTS_DIR / f"{args.provider}_{model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "post_level_data.csv"

    # Compute missing trial IDs per style (handles gap-filling)
    expected_ids = set(range(args.n_trials))
    style_missing: dict = {}
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        for style in args.styles:
            have = set(existing.loc[existing["prompt_style"] == style, "trial_id"].unique())
            missing = sorted(expected_ids - have)
            if missing:
                style_missing[style] = missing
    else:
        for style in args.styles:
            style_missing[style] = sorted(expected_ids)

    total_missing = sum(len(ids) for ids in style_missing.values())

    print(f"\nProvider: {args.provider} / {model}")
    print(f"Pool size: {len(posts):,} posts")
    print(f"Sample size: {args.sample_size} | Top-k: {args.k} | Trials/style: {args.n_trials}")
    print(f"Output: {out_dir}\n")

    if not style_missing:
        print("All styles already complete — nothing to run.")
        return

    print(f"Missing trials to run: {total_missing} total")
    for style in args.styles:
        if style not in style_missing:
            print(f"  {style:15s}: complete")
        else:
            ids = style_missing[style]
            id_summary = f"{ids[0]}–{ids[-1]}" if ids[-1] - ids[0] == len(ids) - 1 else f"{len(ids)} ids"
            print(f"  {style:15s}: {len(ids):3d} missing  [{id_summary}]")

    if args.dry_run:
        print("\n--dry-run: exiting without making API calls.")
        return

    if args.fake:
        print(f"\n--fake mode: selecting {args.k} posts at random per trial (no API calls)\n")
        llm_client = None
    else:
        print(f"\nValidating {args.provider} API... ", end="", flush=True)
        llm_client = get_llm_client(provider=args.provider, model=model)
        test = llm_client.generate("Reply with the single word: ok", max_tokens=10)
        print(f"OK (response: '{test.strip()[:30]}')\n")

    import random

    for style in args.styles:
        if style not in style_missing:
            print(f"Prompt style: {style.upper()} — skipped (complete)")
            continue

        missing_ids = style_missing[style]
        print(f"Prompt style: {style.upper()} — {len(missing_ids)} trials")

        for j, trial_id in enumerate(missing_ids):
            pool = posts.sample(n=min(args.sample_size, len(posts)), random_state=1000 + trial_id)
            if args.fake:
                result = pool.copy()
                result["selected"] = 0
                rng = random.Random(trial_id)
                chosen = rng.sample(range(len(result)), min(args.k, len(result)))
                result.iloc[chosen, result.columns.get_loc("selected")] = 1
            else:
                try:
                    result = run_trial(llm_client, pool, args.k, style)
                except RuntimeError as e:
                    print(f"  Trial {j + 1}/{len(missing_ids)} (id={trial_id}) SKIPPED: {e}")
                    continue
            result["prompt_style"] = style
            result["trial_id"] = trial_id

            header = not out_csv.exists()
            result.to_csv(out_csv, mode="a", index=False, header=header)
            print(f"  Trial {j + 1}/{len(missing_ids)} (id={trial_id}) saved")

        print(f"  {len(missing_ids)} trials done for style '{style}'")

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
