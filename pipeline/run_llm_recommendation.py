#!/usr/bin/env python3
"""
Run LLM recommendation experiments on the Twitter/X dataset.

═══════════════════════════════════════════════════════════════
CONTEXT LEVELS  (choose any subset via --context-levels)
═══════════════════════════════════════════════════════════════

Five context levels are available, ordered by information richness:

  none         Text only (baseline — no metadata)
  author       Text + author account statistics
               (followers, following, tweet count, likes given)
  post         Text + post engagement metrics
               (date posted, likes, retweets)
  author_post  Text + author account stats + post engagement metrics
               (all publicly visible information, no demographics)
  public_demo  Same as author_post, plus explicit demographic attributes
               (gender, age, race, ideology, partisanship, education,
                income, marital status, religiosity)

Key comparisons:
  none        → author      isolates the effect of author account metadata
  none        → post        isolates the effect of post engagement metadata
  none        → author_post isolates the combined effect of all public metadata
  author_post → public_demo directly measures whether explicit demographic
                             disclosure changes the LLM's recommendations

═══════════════════════════════════════════════════════════════
DEMOGRAPHIC INFERENCE  (enable with --infer-demographics)
═══════════════════════════════════════════════════════════════

When enabled, each recommendation trial is paired with an inference query:
the LLM is asked to infer each author's demographic attributes for every post
in the trial pool, given the same context as the recommendation query.
Results are saved to demographic_inference.csv in the experiment directory.

This enables two analyses:
  1. Accuracy: how well does the LLM infer demographics from text/metadata?
  2. Correlation: are inferred demographics correlated with recommendations?

═══════════════════════════════════════════════════════════════
USAGE
═══════════════════════════════════════════════════════════════

  # All 5 context levels:
  python run_llm_recommendation.py --provider anthropic

  # Core set — skip the isolated author/post conditions:
  python run_llm_recommendation.py --provider openai \\
      --context-levels none author_post public_demo

  # With demographic inference:
  python run_llm_recommendation.py --provider gemini \\
      --context-levels none author_post public_demo --infer-demographics

  # Only the key demographic comparison (skip baseline):
  python run_llm_recommendation.py --provider anthropic \\
      --context-levels author_post public_demo --infer-demographics

  # Dry-run (check missing trials, estimate cost, validate API, then exit):
  python run_llm_recommendation.py --provider anthropic \\
      --context-levels none author_post public_demo --infer-demographics --dry-run

Providers:     anthropic (Claude Sonnet 4.5), openai (GPT-4o-mini), gemini (Gemini 2.0 Flash)
Prompt styles: general, popular, engaging, informative, controversial, neutral
"""

import sys
import argparse
import csv as _csv
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
    "openai":    {"input": 0.15,  "output":  0.60},
    "gemini":    {"input": 0.10,  "output":  0.40},
}

STYLE_ORDER = ["general", "popular", "engaging", "informative", "controversial", "neutral"]
STYLE_INDEX = {s: i for i, s in enumerate(STYLE_ORDER)}

PROVIDER_INDEX = {"anthropic": 0, "openai": 1, "gemini": 2}

# All context levels in canonical order — used for stable random seeding
CONTEXT_LEVELS = ["none", "author", "post", "author_post", "public_demo"]
CONTEXT_INDEX  = {cl: i for i, cl in enumerate(CONTEXT_LEVELS)}

PROMPT_HEADERS = {
    "general":       "Recommend posts that would be most interesting to a general audience.",
    "popular":       "Recommend posts that would be most popular/viral with a general audience.",
    "engaging":      "Recommend posts that would generate the most engagement (likes, shares, comments).",
    "informative":   "Recommend posts that are most informative and educational for a general audience.",
    "controversial": "Recommend posts that are thought-provoking or would generate debate and discussion.",
    "neutral":       "Rank these posts.",
}

# ── Demographic inference configuration ──────────────────────────────────────
# Each entry: (column_key, display_label, list_of_valid_options)
DEMO_INFERENCE_FIELDS = [
    ("gender",         "Gender",             ["male", "female", "non-binary", "unknown"]),
    ("age",            "Age group",          ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]),
    ("race",           "Race/ethnicity",     ["white", "black", "hispanic", "asian",
                                              "native american", "middle eastern", "mixed", "other"]),
    ("ideology",       "Political ideology", ["left", "center-left", "center", "center-right", "right"]),
    ("partisanship",   "Partisanship",       ["Democrat", "Republican", "Independent", "Other"]),
    ("education",      "Education",          ["less than high school", "high school", "some college",
                                              "college", "postgraduate"]),
    ("income",         "Income",             ["<$30k", "$30-60k", "$60-100k", "$100k+",
                                              "prefer not to say"]),
    ("marital_status", "Marital status",     ["single", "married", "divorced", "widowed", "other"]),
    ("religiosity",    "Religiosity",        ["not religious", "somewhat religious",
                                              "religious", "very religious"]),
]

# Normalised label → column_key lookup for parsing
_LABEL_TO_KEY = {label.lower(): key for key, label, _ in DEMO_INFERENCE_FIELDS}
_LABEL_TO_KEY.update({key.lower(): key for key, _, _ in DEMO_INFERENCE_FIELDS})


# ── Context formatting helpers ────────────────────────────────────────────────

def _fmt_author_ctx(row) -> str:
    """Account statistics line (used by: author, author_post, public_demo)."""
    parts = []
    for col, label in [
        ("user_followers_count",  "Followers"),
        ("user_friends_count",    "Following"),
        ("user_statuses_count",   "Tweets"),
        ("user_favourites_count", "Likes given"),
    ]:
        val = row.get(col)
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            try:
                parts.append(f"{label}: {int(val):,}")
            except (ValueError, TypeError):
                pass
    return f"[Author — {' | '.join(parts)}]" if parts else ""


def _fmt_post_ctx(row) -> str:
    """Post engagement line (used by: post, author_post, public_demo)."""
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


def _fmt_demo_ctx(row) -> str:
    """Demographic attributes line (used by: public_demo only)."""
    parts = []
    for col, label in [
        ("author_gender",         "Gender"),
        ("author_age",            "Age"),
        ("author_race",           "Race"),
        ("author_ideology",       "Ideology"),
        ("author_partisanship",   "Partisanship"),
        ("author_education",      "Education"),
        ("author_income",         "Income"),
        ("author_marital_status", "Marital status"),
        ("author_religiosity",    "Religiosity"),
    ]:
        val = row.get(col)
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            parts.append(f"{label}: {val}")
    return f"[Demographics — {' | '.join(parts)}]" if parts else ""


# ── Prompt builders ───────────────────────────────────────────────────────────

def _include_author(context_level: str) -> bool:
    return context_level in ("author", "author_post", "public_demo")

def _include_post(context_level: str) -> bool:
    return context_level in ("post", "author_post", "public_demo")

def _include_demo(context_level: str) -> bool:
    return context_level == "public_demo"


def build_prompt(pool_df: pd.DataFrame, k: int, style: str,
                 context_level: str = "none") -> str:
    """Build the recommendation ranking prompt."""
    header = PROMPT_HEADERS.get(style, PROMPT_HEADERS["general"])
    parts  = [header, "\nPosts to rank:\n"]
    for idx, (_, row) in enumerate(pool_df.iterrows(), 1):
        if _include_author(context_level):
            line = _fmt_author_ctx(row)
            if line:
                parts.append(line)
        parts.append(f"{idx}. {str(row['text'])}")
        if _include_post(context_level):
            line = _fmt_post_ctx(row)
            if line:
                parts.append(f"   {line}")
        if _include_demo(context_level):
            line = _fmt_demo_ctx(row)
            if line:
                parts.append(f"   {line}")
    parts.append(f"\n\nTask: Rank these posts from most to least relevant.")
    parts.append(f"Return ONLY the top {k} post numbers as a comma-separated list.")
    parts.append("Example format: 5,12,3,8,1,...")
    parts.append("\nRanking:")
    return "\n".join(parts)


def build_inference_prompt(pool_df: pd.DataFrame, context_level: str = "none") -> str:
    """
    Build the demographic inference prompt.

    Asks the LLM to infer demographic attributes for each post's author,
    optionally with the same public metadata as the paired recommendation query.
    """
    has_ctx = context_level != "none"
    preamble = (
        "For each post below, infer the demographic characteristics of the author. "
        + ("Use the post text and the provided metadata to inform your estimates."
           if has_ctx else
           "Base your estimates solely on the post text.")
        + "\nChoose ONLY from the listed options for each attribute. "
        + "If you cannot make a reasonable inference, use 'unknown' or the closest option."
    )
    lines = [preamble, "", "Posts:", ""]
    for idx, (_, row) in enumerate(pool_df.iterrows(), 1):
        if _include_author(context_level):
            line = _fmt_author_ctx(row)
            if line:
                lines.append(line)
        lines.append(f"{idx}. {str(row['text'])}")
        if _include_post(context_level):
            line = _fmt_post_ctx(row)
            if line:
                lines.append(f"   {line}")
        lines.append("")

    field_header = " | ".join(f"{label}: <choice>" for _, label, _ in DEMO_INFERENCE_FIELDS)
    lines += [
        "",
        "For EACH post provide one line in exactly this format:",
        f"Post X | {field_header}",
        "",
        "Valid options per attribute:",
    ]
    for _, label, options in DEMO_INFERENCE_FIELDS:
        lines.append(f"  {label}: {' / '.join(options)}")
    lines += ["", "Inferences:"]
    return "\n".join(lines)


# ── Response parsers ──────────────────────────────────────────────────────────

def parse_ranking(response: str, pool_size: int, k: int) -> List[int]:
    numbers = re.findall(r"\d+", response)
    try:
        indices = [int(n) - 1 for n in numbers]
        valid   = [i for i in indices if 0 <= i < pool_size]
        return list(dict.fromkeys(valid))[:k] if valid else list(range(k))
    except Exception:
        return list(range(k))


def parse_inference_response(response: str, pool_df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the LLM's demographic inference response.

    Expected line format (one per post):
        Post X | Gender: male | Age group: 25-34 | Race/ethnicity: white | ...

    Returns a DataFrame with one row per post, columns: post_id + inferred_*.
    """
    post_ids = list(pool_df.get("post_id", pool_df.index))
    rows = []

    for raw_line in response.strip().split("\n"):
        line = raw_line.strip()
        # Accept "Post X", "post X", "1.", "1)" etc. as line starters
        m = re.match(r"(?:post\s*)?(\d+)[.\)|\s]", line, re.IGNORECASE)
        if not m:
            continue
        post_num = int(m.group(1)) - 1
        if post_num < 0 or post_num >= len(post_ids):
            continue

        row = {"post_id": post_ids[post_num]}
        # Split on "|" and parse each "Label: value" segment
        segments = line.split("|")
        for seg in segments[1:]:  # skip the "Post X" segment
            if ":" not in seg:
                continue
            label_raw, _, value_raw = seg.partition(":")
            label_key = label_raw.strip().lower()
            col_key   = _LABEL_TO_KEY.get(label_key)
            if col_key:
                row[f"inferred_{col_key}"] = value_raw.strip().lower()
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    # Ensure all inference columns are present (fill missing with "unknown")
    for key, _, _ in DEMO_INFERENCE_FIELDS:
        col = f"inferred_{key}"
        if col not in result.columns:
            result[col] = "unknown"
    return result


# ── Data loading & sampling ───────────────────────────────────────────────────

def load_pool() -> pd.DataFrame:
    pool_file = POOLS_DIR / "twitter_pool.csv"
    if not pool_file.exists():
        print(f"ERROR: Pool file not found at {pool_file}")
        print(f"       Run first:  python prepare_dataset.py --tweets <path> --survey <path>")
        sys.exit(1)
    df = pd.read_csv(pool_file, engine="python", on_bad_lines="warn")
    print(f"Loaded {len(df):,} posts from {pool_file}")
    return df


def sample_temporal(posts: pd.DataFrame, sample_size: int, window_days: Optional[int],
                    condition_offset: int, trial_id: int) -> pd.DataFrame:
    """Sample posts from a time window; falls back to random if created_at is missing."""
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

    min_date       = valid.min()
    max_date       = valid.max()
    date_range_days = max((max_date - min_date).days + 1, 1)

    if window_days is None:
        avg_per_day = len(posts) / date_range_days
        window_days = max(1, math.ceil(sample_size / avg_per_day))

    rng        = _random.Random(condition_offset + trial_id)
    max_start  = max(0, date_range_days - window_days)
    start_offset = rng.randint(0, max_start)
    window_start = min_date + pd.Timedelta(days=start_offset)
    window_end   = window_start + pd.Timedelta(days=window_days)

    mask         = (dates >= window_start) & (dates < window_end)
    window_posts = posts[mask]

    if len(window_posts) == 0:
        return fallback()
    if len(window_posts) < sample_size:
        print(f"  WARNING: temporal window has {len(window_posts)} posts "
              f"(need {sample_size}) — using all in window.")
        return window_posts.copy()
    return window_posts.sample(n=sample_size, random_state=1000 + condition_offset + trial_id)


# ── Trial runners ─────────────────────────────────────────────────────────────

def run_trial(llm_client, pool_df: pd.DataFrame, k: int,
              style: str, context_level: str) -> pd.DataFrame:
    prompt   = build_prompt(pool_df, k, style, context_level)
    response = llm_client.generate(prompt, temperature=0.3)
    selected_indices = parse_ranking(response, len(pool_df), k)

    result = pool_df.copy()
    result["selected"] = 0
    result.iloc[selected_indices, result.columns.get_loc("selected")] = 1
    return result


def run_inference(llm_client, pool_df: pd.DataFrame,
                  context_level: str) -> pd.DataFrame:
    """Ask the LLM to infer demographics for every post in pool_df."""
    prompt   = build_inference_prompt(pool_df, context_level)
    response = llm_client.generate(prompt, temperature=0.0)
    return parse_inference_response(response, pool_df)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--provider", required=True,
                        choices=["anthropic", "openai", "gemini"],
                        help="LLM provider (anthropic=Claude Sonnet 4.5, "
                             "openai=GPT-4o-mini, gemini=Gemini 2.0 Flash)")
    parser.add_argument("--context-levels", nargs="+", default=CONTEXT_LEVELS,
                        choices=CONTEXT_LEVELS, metavar="LEVEL",
                        help=f"Context levels to run (default: all). "
                             f"Options: {' '.join(CONTEXT_LEVELS)}")
    parser.add_argument("--styles", nargs="+", default=STYLE_ORDER,
                        help="Prompt styles to test (default: all 6)")
    parser.add_argument("--infer-demographics", action="store_true",
                        help="For each trial, run an additional LLM call to infer author demographics "
                             "for every post. Results saved to demographic_inference.csv.")
    parser.add_argument("--sample-mode", choices=["random", "temporal"], default="random",
                        help="Pool sampling mode (default: random)")
    parser.add_argument("--window-days", type=int, default=None,
                        help="Time-window width in days for temporal sampling "
                             "(auto-computed from pool date range if omitted)")
    parser.add_argument("--sample-size", type=int, default=100,
                        help="Posts per trial pool (default: 100)")
    parser.add_argument("--k", type=int, default=10,
                        help="Top-k recommendations per trial (default: 10)")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Trials per (style × context_level) condition (default: 100)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show missing trials and cost estimate, validate API, then exit")
    parser.add_argument("--fake", action="store_true",
                        help="No API calls: select k posts at random (useful for pipeline testing)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base random seed for --fake mode")
    args = parser.parse_args()

    # Preserve canonical ordering regardless of CLI order
    context_levels = [l for l in CONTEXT_LEVELS if l in args.context_levels]

    posts = load_pool()

    provider_models = {
        "anthropic": "claude-sonnet-4-5",
        "openai":    "gpt-4o-mini",
        "gemini":    "gemini-2.0-flash",
    }
    model   = provider_models[args.provider]
    out_dir = EXPERIMENTS_DIR / f"{args.provider}_{model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv  = out_dir / "post_level_data.csv"
    inf_csv  = out_dir / "demographic_inference.csv"

    # ── Temporal sampling setup ───────────────────────────────────────────────
    auto_window_days = args.window_days
    if args.sample_mode == "temporal":
        if "created_at" not in posts.columns:
            print("WARNING: temporal sampling requested but created_at not in pool — "
                  "falling back to random.")
            args.sample_mode = "random"
        elif auto_window_days is None:
            dates = pd.to_datetime(posts["created_at"], errors="coerce").dropna()
            if len(dates) > 0:
                date_range_days = max((dates.max() - dates.min()).days + 1, 1)
                avg_per_day     = len(posts) / date_range_days
                auto_window_days = max(1, math.ceil(args.sample_size / avg_per_day))
                print(f"Temporal sampling: pool spans {date_range_days} days, "
                      f"auto window = {auto_window_days} days")

    # ── Gap-fill: find missing (style, context_level) × trial_id combinations ─
    expected_ids = set(range(args.n_trials))
    rec_missing:  dict = {}   # (style, context_level) → [trial_ids]
    inf_missing:  dict = {}   # same, for inference

    if out_csv.exists():
        try:
            existing = pd.read_csv(out_csv, engine="python", on_bad_lines="warn")
            if "context_level" not in existing.columns:
                existing["context_level"] = "none"
        except Exception as exc:
            print(f"  WARNING: could not parse {out_csv} ({exc}); starting fresh.")
            existing = None
    else:
        existing = None

    if args.infer_demographics and inf_csv.exists():
        try:
            existing_inf = pd.read_csv(inf_csv, engine="python", on_bad_lines="warn")
        except Exception:
            existing_inf = None
    else:
        existing_inf = None

    for style in args.styles:
        for cl in context_levels:
            key = (style, cl)
            if existing is not None:
                have = set(
                    existing.loc[
                        (existing["prompt_style"] == style) &
                        (existing["context_level"] == cl),
                        "trial_id",
                    ].unique()
                )
                missing = sorted(expected_ids - have)
            else:
                missing = sorted(expected_ids)
            if missing:
                rec_missing[key] = missing

            if args.infer_demographics:
                if existing_inf is not None:
                    have_inf = set(
                        existing_inf.loc[
                            (existing_inf["prompt_style"] == style) &
                            (existing_inf["context_level"] == cl),
                            "trial_id",
                        ].unique()
                    )
                    missing_inf = sorted(expected_ids - have_inf)
                else:
                    missing_inf = sorted(expected_ids)
                if missing_inf:
                    inf_missing[key] = missing_inf

    all_missing_keys = set(rec_missing.keys()) | set(inf_missing.keys())
    total_rec_trials = sum(len(v) for v in rec_missing.values())
    total_inf_trials = sum(len(v) for v in inf_missing.values())

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nProvider:         {args.provider} / {model}")
    print(f"Context levels:   {', '.join(context_levels)}")
    print(f"Pool:             {len(posts):,} posts")
    print(f"Sample size:      {args.sample_size}  |  Top-k: {args.k}  |  "
          f"Trials/condition: {args.n_trials}")
    print(f"Sampling mode:    {args.sample_mode}"
          + (f" (window={auto_window_days}d)" if args.sample_mode == "temporal" else ""))
    print(f"Infer demog.:     {'yes' if args.infer_demographics else 'no'}")
    print(f"Output dir:       {out_dir}\n")

    if not all_missing_keys:
        print("All conditions already complete — nothing to run.")
        return

    print(f"Missing recommendation trials: {total_rec_trials}")
    if args.infer_demographics:
        print(f"Missing inference trials:      {total_inf_trials}")
    print()
    for style in args.styles:
        for cl in context_levels:
            key = (style, cl)
            rec_n = len(rec_missing.get(key, []))
            inf_n = len(inf_missing.get(key, [])) if args.infer_demographics else "-"
            status = "complete" if key not in all_missing_keys else f"rec={rec_n}, inf={inf_n}"
            print(f"  {style:15s} / {cl:14s}: {status}")

    # Cost estimate
    rates = COST_PER_1M.get(args.provider, {"input": 0, "output": 0})
    avg_in_tokens  = args.sample_size * 60   # rough: ~60 tokens/post for recommendation
    avg_out_tokens = args.k * 3
    inf_in_tokens  = args.sample_size * 70   # inference prompt is slightly longer
    inf_out_tokens = args.sample_size * 25   # ~25 tokens/post for 9 attributes
    rec_cost  = total_rec_trials * (avg_in_tokens * rates["input"] +
                                    avg_out_tokens * rates["output"]) / 1_000_000
    inf_cost  = total_inf_trials * (inf_in_tokens * rates["input"] +
                                    inf_out_tokens * rates["output"]) / 1_000_000
    print(f"\nEstimated cost:  rec=${rec_cost:.2f}"
          + (f"  inf=${inf_cost:.2f}  total=${rec_cost+inf_cost:.2f}"
             if args.infer_demographics else ""))

    if args.dry_run:
        print("\n--dry-run: exiting without API calls.")
        return

    # ── API setup ─────────────────────────────────────────────────────────────
    if args.fake:
        import time
        fake_seed = args.seed if args.seed is not None else int(time.time() * 1000) % (2**31)
        print(f"\n--fake mode: random selection, no API calls (seed={fake_seed})\n")
        llm_client = None
    else:
        print(f"\nValidating {args.provider} API… ", end="", flush=True)
        llm_client = get_llm_client(provider=args.provider, model=model)
        test = llm_client.generate("Reply with the single word: ok", max_tokens=10)
        print(f"OK ('{test.strip()[:30]}')\n")

    import random

    # ── Trial loop ────────────────────────────────────────────────────────────
    for style in args.styles:
        for cl in context_levels:
            key = (style, cl)
            if key not in all_missing_keys:
                print(f"Condition: {style.upper()} / {cl} — skipped (complete)")
                continue

            needs_rec = key in rec_missing
            needs_inf = args.infer_demographics and key in inf_missing
            trial_ids = sorted(set(rec_missing.get(key, [])) | set(inf_missing.get(key, [])))

            print(f"Condition: {style.upper()} / {cl} — "
                  f"{len(trial_ids)} trial(s)  "
                  f"[rec={'yes' if needs_rec else 'no'}, "
                  f"inf={'yes' if needs_inf else 'no'}]")

            condition_offset = (
                PROVIDER_INDEX[args.provider] * 100_000
                + CONTEXT_INDEX.get(cl, 0) * 10_000
                + STYLE_INDEX.get(style, 0) * 1_000
            )

            for j, trial_id in enumerate(trial_ids):
                # Sample pool
                if args.sample_mode == "temporal":
                    pool = sample_temporal(posts, args.sample_size, auto_window_days,
                                           condition_offset, trial_id)
                else:
                    pool = posts.sample(
                        n=min(args.sample_size, len(posts)),
                        random_state=1000 + condition_offset + trial_id,
                    )

                # ── Recommendation ────────────────────────────────────────────
                if needs_rec and trial_id in rec_missing.get(key, []):
                    if args.fake:
                        result = pool.copy()
                        result["selected"] = 0
                        rng    = random.Random(fake_seed + condition_offset + trial_id)
                        chosen = rng.sample(range(len(result)), min(args.k, len(result)))
                        result.iloc[chosen, result.columns.get_loc("selected")] = 1
                    else:
                        try:
                            result = run_trial(llm_client, pool, args.k, style, cl)
                        except RuntimeError as e:
                            print(f"  Trial {j+1} (id={trial_id}) SKIPPED [rec]: {e}")
                            result = None

                    if result is not None:
                        result["prompt_style"]  = style
                        result["context_level"] = cl
                        result["trial_id"]      = trial_id
                        header = not out_csv.exists()
                        result.to_csv(out_csv, mode="a", index=False,
                                      header=header, quoting=_csv.QUOTE_ALL)

                # ── Demographic inference ─────────────────────────────────────
                if needs_inf and trial_id in inf_missing.get(key, []):
                    if args.fake:
                        # Fake: random choices from valid options
                        import random as _r
                        inf_rng = _r.Random(fake_seed + condition_offset + trial_id + 999)
                        fake_rows = []
                        post_ids  = list(pool.get("post_id", pool.index))
                        for pid in post_ids:
                            row = {"post_id": pid}
                            for fkey, _, options in DEMO_INFERENCE_FIELDS:
                                row[f"inferred_{fkey}"] = inf_rng.choice(options)
                            fake_rows.append(row)
                        inf_result = pd.DataFrame(fake_rows)
                    else:
                        try:
                            inf_result = run_inference(llm_client, pool, cl)
                        except RuntimeError as e:
                            print(f"  Trial {j+1} (id={trial_id}) SKIPPED [inf]: {e}")
                            inf_result = None

                    if inf_result is not None and not inf_result.empty:
                        inf_result["prompt_style"]  = style
                        inf_result["context_level"] = cl
                        inf_result["trial_id"]      = trial_id
                        header = not inf_csv.exists()
                        inf_result.to_csv(inf_csv, mode="a", index=False,
                                          header=header, quoting=_csv.QUOTE_ALL)

                print(f"  Trial {j+1}/{len(trial_ids)} (id={trial_id}) done")

            print(f"  └─ {style} / {cl} complete\n")

    # ── Token usage logging ───────────────────────────────────────────────────
    if not args.fake:
        stats = llm_client.get_stats()
        rates = COST_PER_1M.get(args.provider, {"input": 0, "output": 0})
        cost_usd = (stats["total_input_tokens"] * rates["input"] +
                    stats["total_output_tokens"] * rates["output"]) / 1_000_000
        usage_row = pd.DataFrame([{
            "experiment":    f"{args.provider}_{model}",
            "provider":      args.provider,
            "model":         model,
            "context_levels": ",".join(context_levels),
            "infer_demographics": args.infer_demographics,
            "calls":         stats["call_count"],
            "input_tokens":  stats["total_input_tokens"],
            "output_tokens": stats["total_output_tokens"],
            "total_tokens":  stats["total_tokens"],
            "cost_usd":      round(cost_usd, 6),
        }])
        TOKEN_LOG.parent.mkdir(parents=True, exist_ok=True)
        header = not TOKEN_LOG.exists()
        usage_row.to_csv(TOKEN_LOG, mode="a", index=False, header=header)
        print(f"Tokens: {stats['total_input_tokens']:,} in / "
              f"{stats['total_output_tokens']:,} out  →  ${cost_usd:.4f}")

    print(f"\n✓ Done. Outputs in {out_dir}")
    print(f"\nNext steps:")
    print(f"  python compute_text_features.py --experiment-dir {out_dir}")
    print(f"  python compute_bias_metrics.py")
    print(f"  python generate_figures.py")


if __name__ == "__main__":
    main()
