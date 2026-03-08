"""Fama-French factor data from pandas-datareader."""

import pandas as pd
import pandas_datareader.data as web


def _fetch_fama_french_impl(start: str, end: str) -> pd.DataFrame:
    """Fetch monthly Fama-French factors: MKT, HML, SMB, STR.

    Returns:
        DataFrame with columns [MKT, HML, SMB, STR], monthly DatetimeIndex.
    """
    # 3 factors
    ff3 = web.DataReader("F-F_Research_Data_Factors", "famafrench", start, end)
    ff3_monthly = ff3[0]  # First table is monthly data

    # Momentum factor
    mom = web.DataReader("F-F_Momentum_Factor", "famafrench", start, end)
    mom_monthly = mom[0]

    # Build result
    df = pd.DataFrame(index=ff3_monthly.index)
    df["MKT"] = ff3_monthly["Mkt-RF"] / 100.0
    df["HML"] = ff3_monthly["HML"] / 100.0
    df["SMB"] = ff3_monthly["SMB"] / 100.0
    df["STR"] = mom_monthly.iloc[:, 0] / 100.0

    # Convert PeriodIndex to DatetimeIndex (month end)
    if isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.to_timestamp(how="end")

    df = df.loc[start:end]
    return df[["MKT", "HML", "SMB", "STR"]]


def fetch_fama_french(start: str, end: str) -> pd.DataFrame:
    """Public wrapper that uses st.cache_data when inside a Streamlit session."""
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx() is not None:
        import streamlit as st

        return st.cache_data(ttl=86400)(_fetch_fama_french_impl)(start, end)
    return _fetch_fama_french_impl(start, end)
