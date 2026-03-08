"""Forecast page: actual vs predicted RV."""

import numpy as np
import streamlit as st

from qrc.dashboard.components import PLOTLY_STATIC_CONFIG, metric_card, rv_line_chart
from qrc.evaluation.metrics import mse, qlike


def render_forecast_page(context) -> None:
    """Render the forecast comparison page."""
    st.header("Forecast Results")

    if context.predictions is None or context.raw_df is None:
        st.warning("No forecast results available. Run the pipeline first.")
        return

    train_idx = context.train_idx
    dates = context.raw_df.index[train_idx + 1:]
    y_true = context.raw_df["RV"].values[train_idx + 1:]

    model_type = context.config.get("model", "QR2")

    # Metrics cards
    col1, col2 = st.columns(2)
    for name in [model_type]:
        if name in context.predictions:
            preds = context.predictions[name]
            n = min(len(y_true), len(preds))
            with col1:
                metric_card(f"{name} MSE", mse(y_true[:n], preds[:n]))
            with col2:
                preds_pos = np.clip(preds[:n], 1e-12, None)
                metric_card(f"{name} QLIKE", qlike(y_true[:n], preds_pos))

    # Toggle QR1 vs QR2
    show_both = st.checkbox("Show QR1 vs QR2 comparison", value=False)
    models_to_show = ["QR1", "QR2"] if show_both else [model_type]

    for name in models_to_show:
        if name in context.predictions:
            preds = context.predictions[name]
            n = min(len(dates), len(y_true), len(preds))
            fig = rv_line_chart(dates[:n], y_true[:n], preds[:n], name)
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_STATIC_CONFIG)
