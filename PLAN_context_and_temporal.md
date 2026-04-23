# Implementation Plan: Temporal Sampling + Metadata Context Experiment

## Overview

Two related extensions to the pipeline:

1. **Temporal sampling** — replace random pool sampling with time-window sampling
   across all conditions (existing + new)
2. **Metadata context experiment** — re-run all 6 prompt styles with 3 additional
   context levels that expose author/post metadata to the LLM, then measure bias
   w.r.t. those metadata features

---

## 1. Temporal Sampling

### Motivation

Current random sampling draws 100 posts from the full pool independently per trial,
so the same posts repeat across trials and the "feed" has no temporal coherence.
Time-window sampling gives each trial a realistic slice of content from a narrow
time period.

### Changes to `prepare_dataset.py`

- Pass `created_at` through to `twitter_pool.csv` if present in the tweet CSV
  (currently it may be dropped). No other changes needed here.
- Raise (or remove) the default `--pool-size` cap. Current default is 5000;
  change to unlimited (keep `--pool-size` as an optional hard cap for testing).

### Changes to `run_llm_recommendation.py`

Add two new arguments:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--sample-mode` | choice | `random` | `random` or `temporal` |
| `--window-days` | int | auto | Width of the time window in days (see below) |

**Temporal sampling logic (per trial):**

1. Check that `created_at` is present in the pool; if not, warn and fall back to `random`.
2. Parse `created_at` to datetime.
3. Compute auto window size if `--window-days` is not set:
   ```
   date_range_days = (max_date - min_date).days + 1
   avg_posts_per_day = len(pool) / date_range_days
   window_days = ceil(sample_size / avg_posts_per_day)
   ```
   Print the computed window so the user can inspect it.
4. For each trial: pick a uniformly random start date in
   `[min_date, max_date - window_days]`, collect all posts in that window,
   then sample `sample_size` from them (random if window > sample_size,
   all if window < sample_size with a warning).
5. Seed the window-start draw from `condition_offset + trial_id` for reproducibility.

**Applies to all conditions** — both the existing 6 prompt styles and the new
context-level conditions introduced in section 2.

---

## 2. Metadata Context Experiment

### Motivation

Test whether exposing author or post metadata to the LLM during ranking shifts
which content gets recommended, and measure the resulting bias w.r.t. those
metadata features.

### New metadata columns (passed through from tweet CSV)

**Author metadata** (added to pool if present in input):

| Column | Type |
|---|---|
| `user_statuses_count` | int (tweet count) |
| `user_followers_count` | int |
| `user_friends_count` | int (following) |
| `user_favourites_count` | int (likes given) |

**Post metadata** (added to pool if present in input):

| Column | Type |
|---|---|
| `created_at` | datetime string |
| `favorite_count` | int |
| `retweet_count` | int |
| `retweeted` | bool |

> **Note:** `user_description` (bio) is intentionally excluded for now — it
> contains free text that raises anonymisation concerns.

### Changes to `prepare_dataset.py`

Pass all columns above through to `twitter_pool.csv` if present in the tweet CSV.
No transformation needed (they are used as-is in prompts and as bias features).

### Changes to `run_llm_recommendation.py`

**New argument:**

```
--context-levels  [none author post author_post]  (multi-value, default: none)
```

Running `--context-levels none author post author_post` executes all four levels
in one invocation. The `context_level` column is added to `post_level_data.csv`.

**Prompt format per context level:**

All levels keep the existing prompt style header (general / popular / engaging /
informative / controversial / neutral) unchanged. Only the per-post formatting
changes.

- **`none`** (current behaviour):
  ```
  1. tweet text
  ```

- **`author`** — author metadata prepended to each post:
  ```
  [Author — Followers: 1,234 | Following: 567 | Tweets: 8,901 | Likes given: 45,678]
  1. tweet text
  ```
  Fields omitted silently if missing from the pool.

- **`post`** — post metadata appended after each post:
  ```
  1. tweet text
     [Posted: 2023-03-15 | Likes: 45 | Retweets: 12 | Retweeted: no]
  ```
  `created_at` formatted as YYYY-MM-DD. Fields omitted if missing.

- **`author_post`** — both combined:
  ```
  [Author — Followers: 1,234 | Following: 567 | Tweets: 8,901 | Likes given: 45,678]
  1. tweet text
     [Posted: 2023-03-15 | Likes: 45 | Retweets: 12 | Retweeted: no]
  ```

**Output:** `post_level_data.csv` gains a `context_level` column. The gap-filling
/ resume logic must be extended to key on `(prompt_style, context_level)` pairs,
not just `prompt_style`.

**Conditions:** 4 context levels × 6 prompt styles = 24 conditions per provider,
72 total (up from 18).

### Changes to `compute_bias_metrics.py`

Add two new feature groups to `FEATURES`:

```python
"author_metadata": [
    "user_followers_count",
    "user_friends_count",
    "user_statuses_count",
    "user_favourites_count",
],
"post_metadata": [
    "favorite_count",
    "retweet_count",
    "retweeted",
],
```

Add corresponding `FEATURE_TYPES` entries (all numerical except `retweeted` which
is binary).

**Context-level dimension:** bias metrics must be computed per
`(provider, prompt_style, context_level)` triple. The output CSVs
(`pool_vs_recommended_summary.csv`, `directional_bias_data.csv`,
`feature_importance_data.csv`) gain a `context_level` column.

### Changes to `generate_figures.py`

The `context_level` column is a new grouping dimension alongside `provider` and
`prompt_style`. Proposed approach:

- Existing figures (01–10) filter to `context_level == "none"` so they remain
  directly comparable to the pre-extension results.
- New figures (11+) facet by context level:
  - **Figure 11** — bias for author metadata features by context level × model
    (only meaningful for `author` and `author_post` conditions)
  - **Figure 12** — bias for post metadata features by context level × model
    (only meaningful for `post` and `author_post` conditions)
  - **Figure 13** — delta heatmap: bias change relative to `none` baseline,
    for each feature × context level (shows what exposing metadata adds)

Exact figure design to be decided when implementing.

---

## 3. Test pipeline updates (`test_pipeline.sh`)

- Add `created_at`, engagement, and author metadata columns to
  `datasets/examples/tweets.csv` (synthetic values).
- Run Step 2 with `--sample-mode temporal --context-levels none author post author_post`.
- Keep trial count small (20) so the test completes quickly.

---

## 4. File change summary

| File | Changes |
|---|---|
| `prepare_dataset.py` | Pass through `created_at`, engagement, author metadata; raise pool-size cap |
| `run_llm_recommendation.py` | Add `--sample-mode`, `--window-days`, `--context-levels`; temporal sampling logic; metadata prompt formatting; gap-fill keyed on `(style, context_level)` |
| `compute_bias_metrics.py` | Add `author_metadata` and `post_metadata` feature groups; handle `context_level` dimension in output |
| `generate_figures.py` | Filter existing figures to `context_level == "none"`; add figures 11–13 for metadata bias |
| `datasets/examples/tweets.csv` | Add synthetic `created_at`, engagement, author metadata columns |
| `pipeline/test_pipeline.sh` | Add temporal + context-level flags to Step 2 invocation |

---

## 5. Open questions / decisions deferred

- **`user_description` (bio):** excluded for now; revisit when anonymisation
  strategy is clearer.
- **Figure 13 (delta) design:** exact layout TBD at implementation time.
- **Whether to split `post_level_data.csv` by context level** or keep one file
  with a `context_level` column. Current plan: one file (simpler, consistent with
  existing structure).
