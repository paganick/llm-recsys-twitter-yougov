# LLM Recommendation Bias — Twitter/X

Audits systematic bias in LLM-based content recommendation on Twitter/X data
with **ground-truth survey demographics** (YouGov-linked).

The pipeline tests three LLM providers × six prompt styles (18 conditions),
comparing which posts are recommended against the full pool across demographic,
textual, and content dimensions.

---

## Key differences from the multi-platform study

| Dimension | Multi-platform repo | This repo |
|---|---|---|
| Datasets | Twitter, Bluesky, Reddit | Twitter/X only |
| Demographics | LLM-inferred (bios) | Ground-truth survey (YouGov) |
| Demographic features | gender, political, minority | gender, partisanship, ideology, race + extensible |
| Conditions | 54 (3 × 3 × 6) | 18 (3 × 6) |
| Additional features | — | tweet type, user account metadata *(Phase 2)* |

---

## Pipeline overview

```
Step 1  prepare_dataset.py       Merge tweets + survey, anonymise, build pool
Step 2  run_llm_recommendation.py  Query LLMs (3 providers × 6 styles × 100 trials)
Step 3  compute_text_features.py  Add NLP features (sentiment, topic, toxicity)
Step 4  compute_bias_metrics.py   Compute Cohen's d / Cramér's V + SHAP importance
Step 5  generate_figures.py       Produce paper-ready figures
```

---

## Data format

### Tweet CSV (required columns)
| Column | Type | Description |
|---|---|---|
| `user_id` | str | User identifier (will be anonymised) |
| `text` | str | Tweet text |

### Tweet CSV (optional columns passed through to pool)
| Column | Type | Description |
|---|---|---|
| `has_url`, `has_hashtag`, `has_mention`, `has_emoji` | int | Style indicators |
| `text_length`, `word_count`, `avg_word_length` | float | Text metrics |
| `is_reply`, `is_retweet`, `is_quote` | int | Tweet type |
| `user_followers_count`, `user_friends_count` | int | Account metadata |
| `user_statuses_count`, `user_verified` | int | Account metadata |
| `user_account_age_days`, `engagement_score` | float | Account metadata |

### Survey CSV (required columns)
| Column | Type | Description |
|---|---|---|
| `user_id` | str | Must match tweet CSV |
| `author_gender` | str | e.g. male / female / non-binary |
| `author_partisanship` | str | e.g. Democrat / Republican / Independent |
| `author_ideology` | str | e.g. liberal / moderate / conservative |
| `author_race` | str | e.g. White / Black / Hispanic / Asian |

### Survey CSV (optional columns)
`author_age`, `author_education`, `author_income`, `author_marital_status`, `author_religiosity`

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API keys
source config.yaml   # after filling in config.yaml from config.yaml.example

# 3. Prepare the pool (separate tweet + survey files)
python pipeline/prepare_dataset.py \
    --tweets datasets/tweets.csv \
    --survey datasets/survey.csv

# — OR — if data is already merged:
python pipeline/prepare_dataset.py \
    --tweets datasets/prepared.csv --tweets-only

# 4. Run LLM experiments (repeat for openai, gemini)
python pipeline/run_llm_recommendation.py --provider anthropic

# 5. Compute text features
python pipeline/compute_text_features.py \
    --experiment-dir outputs/experiments/anthropic_claude-sonnet-4-5

# 6. Compute bias metrics and generate figures
python pipeline/compute_bias_metrics.py
python pipeline/generate_figures.py
```

Or run the full pipeline in one go:
```bash
TWEETS=datasets/tweets.csv SURVEY=datasets/survey.csv PROVIDER=anthropic \
    bash pipeline/run_pipeline.sh
```

---

## Outputs

```
outputs/
  pools/
    twitter_pool.csv          post_id, user_id, text + demographics + metadata
    twitter_id_map.csv        user_id → original_user_id  (internal, do not share)
  experiments/
    {provider}_{model}/
      post_level_data.csv     60,000 rows per provider (100 posts × 100 trials × 6 styles)
  cache/
    twitter_features.parquet  cached NLP features keyed on post_id
  token_usage.csv             API usage and estimated cost per run

analysis_outputs/
  pool_vs_recommended_summary.csv   bias magnitude per feature × condition
  directional_bias_data.csv         category-level directional bias
  feature_importance_data.csv       RF + SHAP importance per condition
  visualizations/
    paper_plots_final/              01–09 paper figures (PNG)
    1_distributions/
    2_bias_heatmaps/
    3_directional_bias/
    4_feature_importance/
```

---

## Extending to additional features (Phase 2)

All tweet metadata columns present in the pool CSV are automatically
propagated through the pipeline. To include them in the bias analysis:

1. Ensure the column is in `datasets/tweets.csv` before running Step 1.
2. Uncomment the relevant block in `FEATURES` in `pipeline/compute_bias_metrics.py`.
3. Add a `FEATURE_TYPES` entry if the column is new.

Re-run Steps 4–5 to update the analysis (no need to re-run the LLM experiments).

---

## Reproducibility

Steps 4–5 (`compute_bias_metrics.py`, `generate_figures.py`) operate entirely
on the pre-generated experiment CSVs and can be re-run without API access or
raw data.
