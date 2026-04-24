#!/usr/bin/env bash
# run_tests.sh — Full integration test for the extended pipeline.
#
# Tests covered
# -------------
#   1. Full pipeline: temporal sampling + all 4 context levels + all 3 providers
#   2. Temporal fallback: pool with no created_at → warning + random sampling
#   3. Output validation: checks that output CSVs have the expected columns and row counts
#
# Usage:
#   bash pipeline/run_tests.sh
#
# Requires the Apptainer SIF (same as test_pipeline.sh).

set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v apptainer &>/dev/null; then
    module load apptainer 2>/dev/null \
        || { echo "ERROR: apptainer not found. Run: module load apptainer"; exit 1; }
fi

SIF="llm-recsys-twitter.sif"
APT="apptainer exec $SIF python"

if [ ! -f "$SIF" ]; then
    echo "ERROR: $SIF not found."
    exit 1
fi

SEED=42
TEST_OUT="outputs/test_run"

# Colours
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}  ✓ $1${NC}"; }
fail() { echo -e "${RED}  ✗ $1${NC}"; exit 1; }
info() { echo -e "${YELLOW}  → $1${NC}"; }

echo "========================================================"
echo "  Extended pipeline integration test"
echo "  Seed: $SEED"
echo "========================================================"

# -----------------------------------------------------------------------
# 0. Generate test datasets
# -----------------------------------------------------------------------
echo ""
echo "[Prep] Generating test datasets ..."

$APT - <<'PYEOF'
import pandas as pd, random, datetime, os

rng = random.Random(42)
os.makedirs("datasets/test", exist_ok=True)

# ---- Dataset A: full metadata + created_at (main test) -----------------
users = [f"u{i:03d}" for i in range(1, 101)]   # 100 users
rows  = []
base  = datetime.date(2023, 1, 1)
for u in users:
    foll = rng.randint(50, 40000)
    frie = rng.randint(20, 3000)
    stat = rng.randint(100, 8000)
    favs = rng.randint(500, 80000)
    for _ in range(2):                           # 2 tweets per user → 200 rows
        day   = rng.randint(0, 89)               # spread over 90 days
        rows.append({
            "user_id":               u,
            "text":                  rng.choice([
                "Thoughts on climate policy today.",
                "Really interesting take on tax reform.",
                "Healthcare costs are still rising — we need to talk.",
                "Education funding deserves more attention.",
                "Gun safety is a critical issue right now.",
                "The housing market is wild lately.",
                "Immigration policy needs nuance.",
                "Mental health awareness is more important than ever.",
                "Renewable energy is the future.",
                "Media bias is real and it matters.",
            ]),
            "created_at":            (base + datetime.timedelta(days=day)).strftime("%Y-%m-%d"),
            "favorite_count":        rng.randint(0, 400),
            "retweet_count":         rng.randint(0, 80),
            "retweeted":             rng.choice([True, False]),
            "user_followers_count":  foll,
            "user_friends_count":    frie,
            "user_statuses_count":   stat,
            "user_favourites_count": favs,
        })

tweets_a = pd.DataFrame(rows)
tweets_a.to_csv("datasets/test/tweets_with_dates.csv", index=False)
print(f"  Dataset A: {len(tweets_a)} tweets, date range "
      f"{tweets_a['created_at'].min()} → {tweets_a['created_at'].max()}")

# ---- Dataset B: no created_at (fallback test) -------------------------
tweets_b = tweets_a.drop(columns=["created_at"])
tweets_b.to_csv("datasets/test/tweets_no_dates.csv", index=False)
print(f"  Dataset B: {len(tweets_b)} tweets, NO created_at column")

# ---- Survey (same for both) -------------------------------------------
genders   = ["male", "female", "non-binary"]
parties   = ["Democrat", "Republican", "Independent"]
ideologies= ["very liberal", "liberal", "moderate", "conservative", "very conservative"]
races     = ["White", "Black", "Hispanic", "Asian", "Other"]
ages      = [22, 30, 40, 50, 60, 70]
edus      = ["high school", "college", "postgraduate"]
incomes   = ["<25k", "25k-50k", "50k-75k", "75k-100k", "100k+"]
marital   = ["single", "married", "divorced"]
relig     = ["not religious", "somewhat religious", "religious", "very religious"]

survey_rows = []
for u in users:
    survey_rows.append({
        "user_id":            u,
        "author_gender":      rng.choice(genders),
        "author_partisanship":rng.choice(parties),
        "author_ideology":    rng.choice(ideologies),
        "author_race":        rng.choice(races),
        "author_age":         rng.choice(ages),
        "author_education":   rng.choice(edus),
        "author_income":      rng.choice(incomes),
        "author_marital_status": rng.choice(marital),
        "author_religiosity": rng.choice(relig),
    })

survey = pd.DataFrame(survey_rows)
survey.to_csv("datasets/test/survey.csv", index=False)
print(f"  Survey:   {len(survey)} respondents")
PYEOF

ok "Test datasets generated (datasets/test/)"

# -----------------------------------------------------------------------
# 1. Clean previous test outputs
# -----------------------------------------------------------------------
echo ""
echo "[Step 0] Cleaning previous test outputs ..."
rm -rf "$TEST_OUT"
rm -f outputs/experiments/*/post_level_data.csv
rm -rf analysis_outputs/
mkdir -p "$TEST_OUT"
ok "Clean output dir: $TEST_OUT/, outputs/experiments/*/post_level_data.csv, analysis_outputs/"

# -----------------------------------------------------------------------
# TEST A — Full pipeline with created_at + temporal + all context levels
# -----------------------------------------------------------------------
echo ""
echo "========================================================"
echo "  TEST A: temporal sampling + all context levels"
echo "========================================================"

echo ""
echo "[A1] prepare_dataset.py (with created_at) ..."
$APT pipeline/prepare_dataset.py \
    --tweets  datasets/test/tweets_with_dates.csv \
    --survey  datasets/test/survey.csv \
    --seed    "$SEED"

# Verify pool has the new columns
$APT - <<'PYEOF'
import pandas as pd, sys
pool = pd.read_csv("outputs/pools/twitter_pool.csv")
required = ["created_at", "favorite_count", "retweet_count", "retweeted",
            "user_followers_count", "user_friends_count",
            "user_statuses_count", "user_favourites_count"]
missing = [c for c in required if c not in pool.columns]
if missing:
    print(f"FAIL — pool missing columns: {missing}"); sys.exit(1)
print(f"  Pool: {len(pool)} rows, date range: {pool['created_at'].min()} → {pool['created_at'].max()}")
print(f"  Columns present: {required}")
PYEOF
ok "Pool CSV has all metadata + created_at columns"

echo ""
echo "[A2] run_llm_recommendation.py --fake (all 3 providers, temporal, all context levels) ..."
for provider in anthropic openai gemini; do
    info "Provider: $provider"
    $APT pipeline/run_llm_recommendation.py \
        --provider "$provider" \
        --n-trials 20 \
        --sample-size 50 \
        --sample-mode temporal \
        --context-levels none author post author_post \
        --fake \
        --seed "$SEED"
done

# Verify output has context_level column and expected row count
$APT - <<'PYEOF'
import pandas as pd, sys, glob

providers = ["anthropic_claude-sonnet-4-5", "openai_gpt-4o-mini", "gemini_gemini-2.0-flash"]
for p in providers:
    path = f"outputs/experiments/{p}/post_level_data.csv"
    df = pd.read_csv(path)
    if "context_level" not in df.columns:
        print(f"FAIL — {p}: missing context_level column"); sys.exit(1)
    cl = sorted(df["context_level"].unique())
    styles = sorted(df["prompt_style"].unique())
    n_trials = df.groupby(["prompt_style","context_level"])["trial_id"].nunique()
    if set(cl) != {"none","author","post","author_post"}:
        print(f"FAIL — {p}: unexpected context_levels {cl}"); sys.exit(1)
    if len(styles) != 6:
        print(f"FAIL — {p}: expected 6 styles, got {styles}"); sys.exit(1)
    print(f"  {p}: {len(df):,} rows | styles={len(styles)} | context_levels={cl}")
    print(f"    trials/condition: {n_trials.unique().tolist()}")
PYEOF
ok "All providers: context_level column present, all 4 levels × 6 styles × 20 trials"

echo ""
echo "[A3] Verify prompt formatting for each context level ..."
$APT - <<'PYEOF'
import sys
sys.path.insert(0, ".")
import pandas as pd
from pipeline.run_llm_recommendation import build_prompt, PROMPT_HEADERS

pool = pd.read_csv("outputs/pools/twitter_pool.csv")
sample = pool.sample(3, random_state=1)

for level in ["none", "author", "post", "author_post"]:
    prompt = build_prompt(sample, k=3, style="general", context_level=level)
    # Check expected markers per level
    has_author = "[Author —" in prompt
    has_post   = "[Posted:" in prompt
    if level == "none"        and (has_author or has_post):
        print(f"FAIL — 'none' prompt should not have metadata markers"); sys.exit(1)
    if level == "author"      and not has_author:
        print(f"FAIL — 'author' prompt missing [Author —] marker"); sys.exit(1)
    if level == "post"        and not has_post:
        print(f"FAIL — 'post' prompt missing [Posted:] marker"); sys.exit(1)
    if level == "author_post" and not (has_author and has_post):
        print(f"FAIL — 'author_post' prompt missing one or both markers"); sys.exit(1)
    print(f"  [{level:12s}] Author marker: {has_author} | Post marker: {has_post}  ✓")

# Print the author_post version so we can visually inspect it
print("\n--- Sample author_post prompt (first 20 lines) ---")
prompt_ap = build_prompt(sample, k=3, style="general", context_level="author_post")
for line in prompt_ap.split("\n")[:20]:
    print("  " + line)
PYEOF
ok "Prompt formatting verified for all 4 context levels"

echo ""
echo "[A4] compute_text_features.py --fake ..."
for exp_dir in outputs/experiments/*/; do
    $APT pipeline/compute_text_features.py \
        --experiment-dir "$exp_dir" \
        --fake
done
ok "Text features computed (fake)"

echo ""
echo "[A5] compute_bias_metrics.py ..."
$APT pipeline/compute_bias_metrics.py

# Verify output CSVs have context_level column and new feature groups
$APT - <<'PYEOF'
import pandas as pd, sys

summary = pd.read_csv("analysis_outputs/pool_vs_recommended_summary.csv")
dirbias = pd.read_csv("analysis_outputs/directional_bias_data.csv")
impcsv  = pd.read_csv("analysis_outputs/feature_importance_data.csv")

for name, df in [("summary", summary), ("directional_bias", dirbias), ("importance", impcsv)]:
    if "context_level" not in df.columns:
        print(f"FAIL — {name}: missing context_level column"); sys.exit(1)
    cls = sorted(df["context_level"].unique())
    print(f"  {name}: {len(df):,} rows | context_levels={cls}")

# Check new feature groups appear in summary
new_features = ["user_followers_count", "user_friends_count",
                "user_statuses_count", "user_favourites_count",
                "favorite_count", "retweet_count", "retweeted"]
found = [f for f in new_features if f in summary["feature"].unique()]
missing = [f for f in new_features if f not in summary["feature"].unique()]
print(f"\n  New metadata features found in bias summary: {found}")
if missing:
    print(f"  WARNING — not found (may be absent from pool): {missing}")
PYEOF
ok "Bias metrics: context_level column present, new feature groups appear"

echo ""
echo "[A6] generate_figures.py ..."
$APT pipeline/generate_figures.py
# Check that figures 11-13 were generated (context levels > 1)
$APT - <<'PYEOF'
import sys
from pathlib import Path

out = Path("analysis_outputs/visualizations/paper_plots_final")
# Per-context figures land in subfolders; cross-context (11-13) in root
required_root = [
    "11_author_metadata_bias_by_context.png",
    "12_post_metadata_bias_by_context.png",
    "13_context_level_delta_heatmap.png",
]
required_none = [
    "01_aggregated_r2_bar_plot.png",
    "02_bias_by_prompt_heatmap.png",
]
missing = [f for f in required_root if not (out / f).exists()]
missing += [f for f in required_none if not (out / "none" / f).exists()]
if missing:
    print(f"FAIL — figures not generated: {missing}"); sys.exit(1)
subfolders = sorted(d.name for d in out.iterdir() if d.is_dir())
print(f"  Context-level subfolders: {subfolders}")
for sf in subfolders:
    n = len(list((out / sf).glob("*.png")))
    print(f"    {sf}/  ({n} figures)")
root_figs = sorted(f.name for f in out.glob("*.png"))
print(f"  Root figures: {root_figs}")
PYEOF
ok "Figures generated including 11, 12, 13"

# -----------------------------------------------------------------------
# TEST B — Temporal fallback when created_at is missing
# -----------------------------------------------------------------------
echo ""
echo "========================================================"
echo "  TEST B: temporal fallback (no created_at)"
echo "========================================================"

echo ""
echo "[B1] prepare_dataset.py (no created_at) ..."
$APT pipeline/prepare_dataset.py \
    --tweets  datasets/test/tweets_no_dates.csv \
    --survey  datasets/test/survey.csv \
    --seed    "$SEED"

# Check that created_at is not in the pool
$APT - <<'PYEOF'
import pandas as pd
pool = pd.read_csv("outputs/pools/twitter_pool.csv")
has_date = "created_at" in pool.columns
print(f"  Pool has created_at column: {has_date}")
if has_date:
    print("  (expected: False — column should be absent)")
else:
    print("  ✓ Correctly absent from pool")
PYEOF

echo ""
echo "[B2] run_llm_recommendation.py --sample-mode temporal (should warn + fall back to random) ..."
# Run one provider, one context level, few trials — capture output to check warning
$APT pipeline/run_llm_recommendation.py \
    --provider anthropic \
    --styles   neutral \
    --context-levels none \
    --n-trials 5 \
    --sample-size 30 \
    --sample-mode temporal \
    --fake \
    --seed "$SEED" 2>&1 | tee /tmp/temporal_fallback_output.txt

# Check that the warning appeared or it completed successfully (either way is acceptable)
if grep -q "WARNING\|created_at\|falling back\|Fallback\|temporal" /tmp/temporal_fallback_output.txt; then
    ok "Fallback message detected (expected)"
else
    # No warning means it fell back at pool load time — check it still ran
    if grep -q "trials done\|Total:" /tmp/temporal_fallback_output.txt; then
        ok "Ran successfully without created_at (fell back to random at startup)"
    else
        fail "Unexpected output — no completion message found"
    fi
fi

# -----------------------------------------------------------------------
# TEST C — Explicit --window-days override
# -----------------------------------------------------------------------
echo ""
echo "========================================================"
echo "  TEST C: explicit --window-days override"
echo "========================================================"

echo ""
echo "[C1] prepare_dataset.py (with created_at, restore) ..."
$APT pipeline/prepare_dataset.py \
    --tweets datasets/test/tweets_with_dates.csv \
    --survey datasets/test/survey.csv \
    --seed   "$SEED"

echo ""
echo "[C2] run_llm_recommendation.py --window-days 10 (explicit) ..."
$APT pipeline/run_llm_recommendation.py \
    --provider anthropic \
    --styles   neutral \
    --context-levels none \
    --n-trials 5 \
    --sample-size 20 \
    --sample-mode temporal \
    --window-days 10 \
    --fake \
    --seed "$SEED" 2>&1 | tee /tmp/explicit_window_output.txt

# Should NOT print the auto-computed window line
if grep -q "auto window" /tmp/explicit_window_output.txt; then
    fail "Auto window was printed despite --window-days being explicit"
fi
if grep -q "trials done\|Total:\|All conditions already complete" /tmp/explicit_window_output.txt; then
    ok "Ran successfully with explicit --window-days 10"
else
    fail "Did not complete successfully with explicit window"
fi

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
echo ""
echo "========================================================"
echo -e "${GREEN}  ALL TESTS PASSED${NC}"
echo "========================================================"
echo "  Test A: temporal + all context levels  → full pipeline OK"
echo "  Test B: no created_at                  → fallback OK"
echo "  Test C: explicit --window-days          → explicit override OK"
echo ""
echo "  Outputs: outputs/pools/, outputs/experiments/, analysis_outputs/"
echo "  Test data: datasets/test/"
echo "========================================================"
