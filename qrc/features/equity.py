"""Equity features from Yahoo Finance: RV, RVq, RVa."""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=86400)
def fetch_equity_features(
    symbol: str, start: str, end: str
) -> pd.DataFrame:
    """Download daily prices and compute monthly realized volatility features.

    Returns:
        DataFrame with columns [RV, RVq, RVa], DatetimeIndex at month end.
    """
    ticker = yf.Ticker(symbol)
    daily = ticker.history(start=start, end=end, auto_adjust=True)

    if daily.empty:
        raise ValueError(f"No data returned for {symbol}")

    # Daily log returns
    close = daily["Close"]
    log_ret = np.log(close / close.shift(1)).dropna()

    # Monthly realized volatility: sqrt(sum of squared daily returns)
    monthly_rv = log_ret.groupby(pd.Grouper(freq="ME")).apply(
        lambda x: np.sqrt((x**2).sum())
    )
    monthly_rv.name = "RV"

    df = monthly_rv.to_frame()
    df["RVq"] = df["RV"].rolling(3).mean()
    df["RVa"] = df["RV"].rolling(12).mean()
    df = df.dropna()

    return df[["RV", "RVq", "RVa"]]
