"""Backtest page: rolling MSE and QLIKE charts."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from qrc.evaluation.metrics import mse, qlike


def render_backtest_page(context) -> None:
    """Render the backtest analysis page."""
    st.header("Backtest Analysis")

    if context.predictions is None or context.raw_df is None:
        st.warning("No backtest results available. Run the pipeline first.")
        return

    train_idx = context.train_idx
    y_true = context.raw_df["RV"].values[train_idx + 1:]
    dates = context.raw_df.index[train_idx + 1:]
    model_type = context.config.get("model", "QR2")

    for name in [model_type]:
        if name not in context.predictions:
            continue
        preds = context.predictions[name]
        n = min(len(y_true), len(preds))

        # Rolling MSE (window of 12 months)
        window = 12
        rolling_mse = []
        rolling_qlike = []
        rolling_dates = []
        for i in range(window, n):
            yt = y_true[i - window:i]
            yp = preds[i - window:i]
            rolling_mse.append(mse(yt, yp))
            yp_pos = np.clip(yp, 1e-12, None)
            rolling_qlike.append(qlike(yt, yp_pos))
            if i < len(dates):
                rolling_dates.append(dates[i])

        # Rolling MSE chart
        fig_mse = go.Figure()
        fig_mse.add_trace(go.Scatter(
            x=rolling_dates, y=rolling_mse,
            mode="lines", name=f"{name} Rolling MSE",
            line=dict(color="#636EFA"),
        ))
        fig_mse.update_layout(
            title=f"{name}: Rolling MSE (12-month window)",
            xaxis_title="Date", yaxis_title="MSE",
            template="plotly_white",
        )
        st.plotly_chart(fig_mse, use_container_width=True)

        # Rolling QLIKE chart
        fig_ql = go.Figure()
        fig_ql.add_trace(go.Scatter(
            x=rolling_dates, y=rolling_qlike,
            mode="lines", name=f"{name} Rolling QLIKE",
            line=dict(color="#EF553B"),
        ))
        fig_ql.update_layout(
            title=f"{name}: Rolling QLIKE (12-month window)",
            xaxis_title="Date", yaxis_title="QLIKE",
            template="plotly_white",
        )
        st.plotly_chart(fig_ql, use_container_width=True)

    # Summary table
    if context.metrics is not None:
        st.subheader("Out-of-Sample Summary")
        qr_metrics = context.metrics.loc[
            context.metrics.index.isin(["QR1", "QR2"])
        ]
        if not qr_metrics.empty:
            st.dataframe(qr_metrics, use_container_width=True)
