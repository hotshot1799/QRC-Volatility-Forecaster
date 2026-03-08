"""Shapley page: SHAP feature importance."""

import streamlit as st

from qrc.dashboard.components import PLOTLY_STATIC_CONFIG, shap_bar_chart


def render_shapley_page(context) -> None:
    """Render the SHAP feature importance page."""
    st.header("Feature Importance (SHAP)")

    if context.shap_values is None:
        st.warning(
            "No SHAP results available. Enable 'Run SHAP analysis' in the sidebar "
            "and run the pipeline again."
        )
        return

    fig = shap_bar_chart(context.shap_values)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_STATIC_CONFIG)

    # Selected features
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Selected Features")
        if context.selected_features:
            for f in context.selected_features:
                st.write(f"- {f}")

    with col2:
        st.subheader("About SHAP Values")
        st.write(
            "SHAP (SHapley Additive exPlanations) values measure each feature's "
            "contribution to the model's prediction. Higher absolute SHAP values "
            "indicate features that have a greater impact on the quantum reservoir's "
            "volatility forecast. The values are computed using a model-agnostic "
            "KernelExplainer applied to the reservoir's measurement-to-prediction "
            "mapping."
        )
