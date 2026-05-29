# Experiment Instructions — LLM Recommendation Bias Study

## Overview

This pipeline runs LLM recommendation experiments on a Twitter/X dataset and
computes bias metrics across three providers, six prompt styles, and up to five
context levels. The steps below cover the full pipeline from raw data to
analysis outputs.

**Please share back only the contents of `analysis_outputs/`** — raw data and
experiment files must remain local.

---

## What to share back

| Path | Share? |
|------|--------|
| `analysis_outputs/` (PNGs, summary CSVs) | **Yes** |
| `outputs/experiments/` | No — keep local |
| `outputs/pools/` | No — keep local |
| `outputs/token_usage.csv` | Optional (cost tracking only) |
| `outputs/cache/` | No |

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

# NLP — toxicity (CPU install; drop the --index-url flag if a GPU is available)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install detoxify

# NLP — topic / polarization (Cardiff NLP RoBERTa via HuggingFace)
pip install transformers

# LLM provider SDKs (install only those you need)
pip install anthropic            # Anthropic / Claude
pip install openai               # OpenAI / GPT
pip install google-generativeai  # Google / Gemini
```

> **Note on detoxify / transformers:** both download model weights on first
> use (~500 MB each). Internet access is needed for the first run of
> `compute_text_features.py`.

### API keys

```bash
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."
export GOOGLE_API_KEY="..."
```

---

## Step 1 — Prepare the dataset

Two input files are needed:
- A tweet CSV with at minimum `user_id` and `text` columns
- A survey CSV with `user_id` and demographic columns

```bash
python pipeline/prepare_dataset.py \
    --tweets /path/to/tweets.csv \
    --survey /path/to/survey.csv
```

Writes `outputs/pools/twitter_pool.csv`.

---

## Step 2 — Generate shared trial pools

Creates 100 pre-sampled post pools (one per trial) so all providers and context
levels evaluate **identical post sets**. Run once.

```bash
python pipeline/prepare_pools.py
```

Expected output: `outputs/pools/post_features.csv`, `outputs/pools/author_features.csv`,
and `outputs/pools/trial_000.csv` … `trial_099.csv`.

---

## Step 3 — Compute text features

Patches NLP features (sentiment, toxicity, topic, polarization) into
`outputs/pools/post_features.csv`. Run once after Step 2.

```bash
python pipeline/compute_text_features.py
```

---

## Step 4 — Run the LLM experiments

Run once per provider. Each command covers all 6 prompt styles × 5 context
levels × 100 trials. The script resumes automatically if interrupted.

```bash
# Anthropic (Claude Sonnet 4.6)
python pipeline/run_llm_recommendation.py --provider anthropic

# OpenAI (GPT-5)
python pipeline/run_llm_recommendation.py --provider openai

# Google (Gemini 3.5 Flash)
python pipeline/run_llm_recommendation.py --provider gemini
```

**Context levels** (all run by default):
- `none` — text only
- `author` — text + author account stats
- `post` — text + post engagement
- `author_post` — text + author + post (all public metadata)
- `public_demo` — same as author_post, plus explicit demographics

**Dry-run** (check missing trials and estimate cost, no API calls):
```bash
python pipeline/run_llm_recommendation.py --provider anthropic --dry-run
```

This step writes one `post_level_data.csv` per provider under `outputs/experiments/`.

---

## Step 5 — Migrate to analysis format

The analysis scripts expect a normalised three-table layout. Run the migration
script once after all providers are done:

```bash
python pipeline/migrate_to_new_format.py
```

This reads from `outputs/experiments/` and writes:
- `outputs_migrated/pools/post_features.csv` — one row per unique post
- `outputs_migrated/pools/author_features.csv` — one row per author
- `outputs_migrated/experiments/<provider>/trial_results.csv` — one row per shown post × trial

> **Note:** `migrate_to_new_format.py` is a local utility script shared
> separately — it is not in the git repository.

---

## Step 6 — Run the analysis pipeline

All four scripts accept `--migrated` to read from `outputs_migrated/`
and write results to `analysis_outputs/`.

```bash
python pipeline/compute_bias_metrics.py        --migrated
python pipeline/logistic_regression.py         --migrated
python pipeline/compute_feature_correlations.py --migrated
python pipeline/generate_figures.py            --migrated
```

Results are written to `analysis_outputs/`. Please share this folder back.

---

## Data confidentiality

The raw data and experiment outputs contain individual-level information and
must not be shared or uploaded anywhere. Only the aggregated outputs in
`analysis_outputs/` (figures and summary CSVs) should be returned.
