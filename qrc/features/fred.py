"""Macroeconomic features from FRED."""

import os

import pandas as pd
import pandas_datareader.data as web
import streamlit as st
from dotenv import load_dotenv


@st.cache_data(ttl=86400)
def fetch_fred_features(
    start: str, end: str, api_key: str | None = None
) -> pd.DataFrame:
    """Fetch macro features from FRED: TB, INF, IP, DEF.

    Args:
        start: Start date string.
        end: End date string.
        api_key: FRED API key. Falls back to FRED_API_KEY env var.

    Returns:
        DataFrame with columns [TB, INF, IP, DEF], monthly DatetimeIndex.
    """
    if api_key is None:
        load_dotenv()
        api_key = os.environ.get("FRED_API_KEY")

    if api_key:
        os.environ["FRED_API_KEY"] = api_key

    # 3-Month T-Bill rate
    tb3ms = web.DataReader("TB3MS", "fred", start, end)
    tb = tb3ms["TB3MS"] / 100.0 / 12.0  # annualised to monthly decimal

    # CPI -> year-on-year inflation
    cpi = web.DataReader("CPIAUCSL", "fred", start, end)
    inf = cpi["CPIAUCSL"].pct_change(12)

    # Industrial Production growth (month-on-month)
    indpro = web.DataReader("INDPRO", "fred", start, end)
    ip = indpro["INDPRO"].pct_change(1)

    # Default spread: BAA - GS10
    baa = web.DataReader("BAA", "fred", start, end)
    gs10 = web.DataReader("GS10", "fred", start, end)
    def_spread = (baa["BAA"] - gs10["GS10"]) / 100.0

    df = pd.DataFrame(
        {"TB": tb, "INF": inf, "IP": ip, "DEF": def_spread}
    )

    # Resample to month end
    df = df.resample("ME").last()
    df = df.dropna()
    return df[["TB", "INF", "IP", "DEF"]]
