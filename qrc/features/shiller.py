"""Shiller PE/dividend data from public Excel file."""

import os

import pandas as pd

SHILLER_URL = "https://shillerdata.com/ie_data.xls"


def _fetch_shiller_features_impl(
    start: str, end: str, cache_dir: str = ".cache"
) -> pd.DataFrame:
    """Fetch D/P and E/P ratios from Shiller's public dataset.

    Returns:
        DataFrame with columns [DP, EP], monthly DatetimeIndex.
    """
    cache_path = os.path.join(cache_dir, "shiller.parquet")

    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
    else:
        # Download and parse
        raw = pd.read_excel(SHILLER_URL, sheet_name="Data", skiprows=7)

        # The date column is typically the first column (fractional year)
        raw = raw.dropna(subset=[raw.columns[0]])
        date_col = raw.columns[0]
        price_col = raw.columns[1]

        # Find dividend and earnings columns
        div_col = None
        earn_col = None
        for col in raw.columns:
            col_str = str(col).strip().lower()
            if "dividend" in col_str and div_col is None:
                div_col = col
            elif "earning" in col_str and earn_col is None:
                earn_col = col

        if div_col is None or earn_col is None:
            # Fallback: use positional columns (D is col 5, E is col 6 typically)
            div_col = raw.columns[5] if len(raw.columns) > 5 else None
            earn_col = raw.columns[6] if len(raw.columns) > 6 else None

        # Convert fractional year to datetime
        def frac_year_to_date(fy):
            try:
                fy = float(fy)
                year = int(fy)
                month = round((fy - year) * 12) + 1
                if month > 12:
                    month = 12
                return pd.Timestamp(year=year, month=month, day=1)
            except (ValueError, TypeError):
                return pd.NaT

        raw["date"] = raw[date_col].apply(frac_year_to_date)
        raw = raw.dropna(subset=["date"])
        raw = raw.set_index("date")

        price = pd.to_numeric(raw[price_col], errors="coerce")
        dividends = pd.to_numeric(raw[div_col], errors="coerce")
        earnings = pd.to_numeric(raw[earn_col], errors="coerce")

        df = pd.DataFrame(index=raw.index)
        df["DP"] = dividends / price
        df["EP"] = earnings / price
        df = df.dropna()

        # Convert to month-end
        df.index = df.index + pd.offsets.MonthEnd(0)
        df = df[~df.index.duplicated(keep="last")]

        # Cache
        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(cache_path)

    df = df.loc[start:end]
    return df[["DP", "EP"]]


def fetch_shiller_features(
    start: str, end: str, cache_dir: str = ".cache"
) -> pd.DataFrame:
    """Public wrapper that uses st.cache_data when inside a Streamlit session."""
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx() is not None:
        import streamlit as st

        return st.cache_data(ttl=86400)(_fetch_shiller_features_impl)(
            start, end, cache_dir
        )
    return _fetch_shiller_features_impl(start, end, cache_dir)
