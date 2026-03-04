"""Shared Plotly chart builders and Streamlit components."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def rv_line_chart(
    dates: pd.DatetimeIndex,
    actual: np.ndarray,
    predicted: np.ndarray,
    model_name: str,
) -> go.Figure:
    """Line chart comparing actual vs predicted RV."""
    fig = go.Figure()
    n = min(len(dates), len(actual), len(predicted))
    fig.add_trace(go.Scatter(
        x=dates[:n], y=actual[:n],
        mode="lines", name="Actual RV",
        line=dict(color="#636EFA"),
    ))
    fig.add_trace(go.Scatter(
        x=dates[:n], y=predicted[:n],
        mode="lines", name=f"{model_name} Predicted",
        line=dict(color="#EF553B", dash="dash"),
    ))
    fig.update_layout(
        title=f"Realized Volatility: Actual vs {model_name}",
        xaxis_title="Date",
        yaxis_title="Realized Volatility",
        template="plotly_white",
    )
    return fig


def metric_card(label: str, value: float, delta: float | None = None) -> None:
    """Display a Streamlit metric card."""
    st.metric(label=label, value=f"{value:.6f}", delta=f"{delta:.6f}" if delta else None)


def model_table(
    metrics_df: pd.DataFrame, mcs_results: dict | None = None
) -> None:
    """Display a styled model comparison table."""
    df = metrics_df.copy()
    if mcs_results:
        df["MCS p-value"] = df.index.map(lambda x: mcs_results.get(x, float("nan")))
        df["In MCS"] = df["MCS p-value"].apply(
            lambda p: "Yes" if p >= 0.05 else "No" if not np.isnan(p) else "-"
        )
    st.dataframe(df.style.highlight_min(subset=["MSE"], color="#d4edda"), use_container_width=True)


def shap_bar_chart(shap_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of mean absolute SHAP values."""
    mean_abs = shap_df.abs().mean().sort_values(ascending=True)
    fig = go.Figure(go.Bar(
        x=mean_abs.values,
        y=mean_abs.index,
        orientation="h",
        marker_color="#AB63FA",
    ))
    fig.update_layout(
        title="Feature Importance (Mean |SHAP|)",
        xaxis_title="Mean |SHAP value|",
        yaxis_title="Feature",
        template="plotly_white",
    )
    return fig
