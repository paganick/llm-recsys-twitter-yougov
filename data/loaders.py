"""
Data loading utilities for the Twitter/X dataset.
"""

from pathlib import Path
import pandas as pd


def load_pool(pools_dir: str | Path = "outputs/pools") -> pd.DataFrame:
    """Load the anonymised post pool produced by prepare_dataset.py."""
    path = Path(pools_dir) / "twitter_pool.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Pool not found at {path}. "
            "Run: python pipeline/prepare_dataset.py --tweets <path> --survey <path>"
        )
    return pd.read_csv(path)


def load_experiment(experiments_dir: str | Path, provider: str) -> pd.DataFrame | None:
    """Load the post-level experiment CSV for a given provider."""
    experiments_dir = Path(experiments_dir)
    matches = list(experiments_dir.glob(f"{provider}_*"))
    if not matches:
        return None
    return pd.read_csv(matches[0] / "post_level_data.csv")


def load_all_experiments(experiments_dir: str | Path = "outputs/experiments") -> pd.DataFrame:
    """Load and concatenate all experiment CSVs, adding a 'provider' column."""
    experiments_dir = Path(experiments_dir)
    dfs = []
    for exp_dir in sorted(experiments_dir.iterdir()):
        csv = exp_dir / "post_level_data.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            provider = exp_dir.name.split("_")[0]
            if "provider" not in df.columns:
                df.insert(0, "provider", provider)
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError(
            f"No experiment CSVs found in {experiments_dir}. "
            "Run: python pipeline/run_llm_recommendation.py --provider <provider>"
        )
    return pd.concat(dfs, ignore_index=True)
