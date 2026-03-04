"""Assemble all feature sources into a single DataFrame."""

import pandas as pd


def assemble_features(
    equity_df: pd.DataFrame,
    ff_df: pd.DataFrame,
    fred_df: pd.DataFrame,
    shiller_df: pd.DataFrame,
) -> pd.DataFrame:
    """Align and merge all feature DataFrames.

    Returns:
        DataFrame with columns:
        [RV, RVq, RVa, MKT, HML, SMB, STR, DP, EP, TB, INF, DEF, IP]
    """
    # Ensure all have month-end index
    dfs = []
    for df in [equity_df, ff_df, fred_df, shiller_df]:
        df = df.copy()
        df.index = pd.DatetimeIndex(df.index)
        df.index = df.index + pd.offsets.MonthEnd(0)
        dfs.append(df)

    # Outer merge on index
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.join(df, how="outer")

    # Forward fill up to 3 months, then drop remaining NaN
    merged = merged.ffill(limit=3)
    merged = merged.dropna()

    column_order = [
        "RV", "RVq", "RVa", "MKT", "HML", "SMB", "STR",
        "DP", "EP", "TB", "INF", "DEF", "IP",
    ]
    # Only keep columns that exist
    cols = [c for c in column_order if c in merged.columns]
    return merged[cols]
