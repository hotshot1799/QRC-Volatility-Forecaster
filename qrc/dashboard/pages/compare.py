"""Compare page: all models side by side."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from qrc.dashboard.components import model_table


def render_compare_page(context) -> None:
    """Render the model comparison page."""
    st.header("Model Comparison")

    if context.metrics is None:
        st.warning("No comparison results available. Run the pipeline first.")
        return

    # Main table with MCS p-values
    st.subheader("All Models — Sorted by MSE")
    model_table(context.metrics, context.mcs_results)

    # MSE bar chart
    fig_bar = go.Figure(go.Bar(
        x=context.metrics.index,
        y=context.metrics["MSE"],
        marker_color=[
            "#00CC96" if name == "QR2" else "#636EFA"
            for name in context.metrics.index
        ],
    ))
    fig_bar.update_layout(
        title="MSE by Model",
        xaxis_title="Model",
        yaxis_title="MSE",
        template="plotly_white",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # DM test heatmap
    if context.dm_results is not None:
        st.subheader("Diebold-Mariano Test p-values")
        fig_dm = go.Figure(go.Heatmap(
            z=context.dm_results.values,
            x=context.dm_results.columns,
            y=context.dm_results.index,
            colorscale="RdYlGn",
            zmin=0, zmax=1,
            text=np.round(context.dm_results.values, 3),
            texttemplate="%{text}",
        ))
        fig_dm.update_layout(
            title="DM Test p-values (row = baseline, col = challenger)",
            template="plotly_white",
        )
        st.plotly_chart(fig_dm, use_container_width=True)
