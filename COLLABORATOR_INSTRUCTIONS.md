# Experiment Instructions — LLM Recommendation Bias Study

## Overview

This pipeline runs LLM recommendation experiments on a Twitter/X dataset and
computes bias metrics across three providers, six prompt styles, and five context
levels. The steps below go from `twitter_pool.csv` to `analysis_outputs_v2/`.

**Please share back only the contents of `analysis_outputs_v2/`** — raw data and
experiment files must remain local.

---

## What to share back

| Path | Share? |
|------|--------|
| `analysis_outputs_v2/` (PNGs, summary CSVs) | **Yes** |
| `outputs_v2/experiments/` | No — keep local |
| `outputs_v2/pools/` | No — keep local |
| `outputs_v2/token_usage.csv` | Optional (cost tracking only) |
| `outputs/cache/` | No — NLP feature cache, created automatically |

---

## Prerequisites

### Python environment

```bash
# Core data / analysis
pip install pandas numpy scipy scikit-learn statsmodels

# Plotting
pip install matplotlib seaborn

# NLP — sentiment
pip install textblob vaderSentiment

# NLP — toxicity (CPU install; drop --index-url if a GPU is available)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install detoxify

# NLP — topic / polarization (Cardiff NLP RoBERTa via HuggingFace)
pip install transformers

# LLM provider SDKs
pip install anthropic            # Anthropic / Claude
pip install openai               # OpenAI / GPT
pip install google-generativeai  # Google / Gemini

# Feature importance (optional — needed for Figure 07 and feature_importance_data.csv)
pip install shap
```

> **Note on detoxify / transformers:** both download model weights (~500 MB each)
> on first use. Internet access is needed for the first run of
> `compute_text_features.py`.
>
> **Note on shap:** if not installed, `feature_importance_data.csv` will be
> empty and Figure 07 will be skipped. All other outputs are unaffected.

### API keys

Set the following environment variables **before running any step**. All three
are needed (one per provider):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="AIza..."
```

---

## Step 1 — Prepare `twitter_pool.csv`

Place your input file at `outputs_v2/pools/twitter_pool.csv`. One row per tweet.

**Required columns:**

| Column | Type | Description |
|---|---|---|
| `post_id` | string | Unique post identifier |
| `user_id` | string | Anonymised author identifier |
| `text` | string | Tweet text |
| `created_at` | string | Post timestamp (needed for temporal trial sampling) |
| `author_gender` | string | Survey demographic |
| `author_partisanship` | string | Survey demographic |
| `author_ideology` | string | Survey demographic |
| `author_race` | string | Survey demographic |
| `author_age` | string | Survey demographic (binned: 18-24, 25-34, …, 65+) |
| `author_education` | string | Survey demographic |
| `author_income` | string | Survey demographic |
| `author_marital_status` | string | Survey demographic |
| `author_religiosity` | string | Survey demographic |

**Optional metadata** (needed to run the `author`, `post`, `author_post`, and
`public_demo` context levels — without them only the `none` level is meaningful):

| Column | Type | Description |
|---|---|---|
| `user_followers_count` | int | Follower count at time of tweet |
| `user_friends_count` | int | Following count at time of tweet |
| `user_statuses_count` | int | Tweet count at time of tweet |
| `user_favourites_count` | int | Likes-given count at time of tweet |
| `favorite_count` | int | Likes received on this post |
| `retweet_count` | int | Retweets received |
| `retweeted` | bool | Whether the post is a retweet |

> Note: `user_*` counts are per-post (not per-author) because they reflect the
> author's state at the time of posting and change over time.
>
> Note: the dataset must contain at least 10,000 posts. Posts with a missing or
> unparseable `created_at` are dropped before sampling.

---

## Step 2 — Generate shared trial pools and feature files

Sorts posts by date, splits into 100 temporal buckets, and samples 100 posts
from each. Writes:

- `outputs_v2/pools/trial_000.csv` … `trial_099.csv` — 100 trial pools, 100
  posts each, containing all columns from `twitter_pool.csv`
- `outputs_v2/pools/post_features.csv` — one row per shown post (10,000 rows);
  text and NLP features are added in Step 3
- `outputs_v2/pools/author_features.csv` — one row per author (stable
  survey demographics only)

```bash
python pipeline/prepare_pools.py --pools-dir outputs_v2/pools
```

---

## Step 3 — Compute text features

Enriches `post_features.csv` with all text-derived features for the 10,000 posts
shown in trials. Run **once** — not once per provider.

Simple features (fast, no ML models): `has_url`, `has_hashtag`, `has_mention`,
`has_emoji`, `word_count`, `avg_word_length`.

NLP features (require model downloads ~500 MB on first run): sentiment polarity,
sentiment subjectivity, primary topic, polarization score, toxicity.

Computed features are cached under `outputs/cache/` so interrupted runs resume.

```bash
python pipeline/compute_text_features.py --pools-dir outputs_v2/pools
```

---

## Step 4 — Run the LLM recommendation experiments

Run once per provider. Each call covers all 6 prompt styles × 5 context levels
× 100 trials (3,000 LLM calls per provider). The script resumes automatically
if interrupted.

```bash
python pipeline/run_llm_recommendation.py --provider anthropic \
    --pools-dir outputs_v2/pools \
    --experiments-dir outputs_v2/experiments

python pipeline/run_llm_recommendation.py --provider openai \
    --pools-dir outputs_v2/pools \
    --experiments-dir outputs_v2/experiments

python pipeline/run_llm_recommendation.py --provider gemini \
    --pools-dir outputs_v2/pools \
    --experiments-dir outputs_v2/experiments
```

**Context levels** (all run by default):
- `none` — text only
- `author` — text + author account stats
- `post` — text + post engagement
- `author_post` — text + author + post (all public metadata)
- `public_demo` — same as author_post, plus explicit demographics

Each provider writes
`outputs_v2/experiments/<provider>_<model>/trial_results.csv` — one row per
**selected** post per (trial × prompt style × context level), with columns
`post_id`, `trial_id`, `prompt_style`, `context_level`. The full pool of shown
posts for each trial is in the corresponding `trial_*.csv` file; the analysis
scripts reconstruct the complete (selected / not selected) picture automatically.

**Check cost before running** (no API calls made):
```bash
python pipeline/run_llm_recommendation.py --provider anthropic \
    --pools-dir outputs_v2/pools \
    --experiments-dir outputs_v2/experiments \
    --dry-run
```

---

## Step 5 — Run demographic inference

For each trial pool, ask the model to guess each post author's demographic
attributes from the tweet text alone. This runs **after** Step 4 — the script
detects that recommendations are already complete and only runs the inference
(100 LLM calls per provider).

```bash
python pipeline/run_llm_recommendation.py --provider anthropic \
    --pools-dir outputs_v2/pools \
    --experiments-dir outputs_v2/experiments \
    --infer-demographics

python pipeline/run_llm_recommendation.py --provider openai \
    --pools-dir outputs_v2/pools \
    --experiments-dir outputs_v2/experiments \
    --infer-demographics

python pipeline/run_llm_recommendation.py --provider gemini \
    --pools-dir outputs_v2/pools \
    --experiments-dir outputs_v2/experiments \
    --infer-demographics
```

Each provider writes
`outputs_v2/experiments/<provider>_<model>/demographic_inference.csv` — one row
per (post × trial) with the model's guessed demographic attributes.

---

## Step 6 — Run the analysis pipeline

All five scripts below read from `outputs_v2/` and write to `analysis_outputs_v2/`.

```bash
# Bias metrics (Cohen's d, Cramér's V, directional bias)
python pipeline/compute_bias_metrics.py \
    --pools-dir outputs_v2/pools \
    --experiments-dir outputs_v2/experiments \
    --analysis-dir analysis_outputs_v2

# Logistic regression models
python pipeline/logistic_regression.py \
    --pools-dir outputs_v2/pools \
    --experiments-dir outputs_v2/experiments \
    --analysis-dir analysis_outputs_v2

# Feature association matrix
python pipeline/compute_feature_correlations.py \
    --pools-dir outputs_v2/pools \
    --analysis-dir analysis_outputs_v2

# All paper figures
python pipeline/generate_figures.py \
    --pools-dir outputs_v2/pools \
    --analysis-dir analysis_outputs_v2

# Demographic inference accuracy (inferred vs ground-truth demographics)
python pipeline/compute_demographic_inference.py \
    --pools-dir outputs_v2/pools \
    --experiments-dir outputs_v2/experiments \
    --analysis-dir analysis_outputs_v2
```

Results are written to `analysis_outputs_v2/`. **Please share this folder back.**

---

## Data confidentiality

The raw data and experiment outputs contain individual-level information and
must not be shared or uploaded anywhere. Only the aggregated outputs in
`analysis_outputs_v2/` (figures and summary CSVs) should be returned.
