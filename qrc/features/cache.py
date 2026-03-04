"""Simple parquet-based data cache."""

import os
import time

import pandas as pd


class DataCache:
    """File-based DataFrame cache using Parquet."""

    def is_fresh(self, cache_path: str, max_age_days: int = 1) -> bool:
        """Return True if the cache file exists and is recent enough."""
        if not os.path.exists(cache_path):
            return False
        mtime = os.path.getmtime(cache_path)
        age_seconds = time.time() - mtime
        return age_seconds < max_age_days * 86400

    def save(self, df: pd.DataFrame, cache_path: str) -> None:
        """Save DataFrame to parquet, creating directories as needed."""
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        df.to_parquet(cache_path)

    def load(self, cache_path: str) -> pd.DataFrame:
        """Load DataFrame from parquet."""
        return pd.read_parquet(cache_path)
