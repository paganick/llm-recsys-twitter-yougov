# Research Notes

## Known Design Issues

### [CRITICAL] Models do not share the same evaluation stimulus set

**What we found:**
Each model's experiment (Anthropic, OpenAI, Google) was run independently with its own random sampling. `trial_id=0` for Anthropic is a completely different 100-post pool than `trial_id=0` for Google or OpenAI. Post sets differ not just across trial IDs but within the same trial ID across models (verified empirically — 0/20 spot-checked combos matched).

**Why it matters:**
Differences in bias scores across models may partly reflect different post distributions seen, not just different LLM behavior. Direct model-to-model comparisons are confounded.

**What needs to be fixed:**
Re-run the experiment so all models evaluate the same (trial × context_level × prompt_style) pools. Concretely: pre-generate the 2,400 post pools (100 trials × 4 context levels × 6 prompt styles) once, save them, then feed the same pools to all models.

**Current workaround:**
Bias metrics are computed per-model against each model's own pool (recommended vs. shown), so within-model results are valid. Cross-model comparisons should be treated as approximate.

---

### [MINOR] ~46% of (trial × context × prompt) combos have fewer than 100 posts

**What we found:**
~1,100 out of 2,400 combos per model have fewer than 100 posts (min = 1). The pattern is identical across all three models, so this is by design: temporal sampling sometimes produces sparse time windows with fewer than 100 available posts.

**Impact:** Low — within-model directional bias is still well-identified. Just means some trials contribute fewer observations.

---

## Pending Computations

### Toxicity scores not yet computed

Detoxify is installed but toxicity is all-NaN in the cache and all experiment CSVs. The cache was built before detoxify was installed.

**Fix ready to run:**
`compute_text_features.py` now has a `_fill_null_features` helper that runs detoxify in batches of 512 and patches the cache in-place. Run:

```bash
python pipeline/compute_text_features.py --experiment-dir outputs/experiments/anthropic_claude-sonnet-4-5
python pipeline/compute_bias_metrics.py
python pipeline/generate_figures.py
```

The cache update from the first command covers all posts seen by any model (cache is shared), so only one experiment dir needs to be passed.
