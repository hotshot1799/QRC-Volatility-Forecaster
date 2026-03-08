"""Main Streamlit dashboard for QRC Volatility Forecaster."""

import streamlit as st

from qrc.dashboard.pages.forecast import render_forecast_page
from qrc.dashboard.pages.backtest import render_backtest_page
from qrc.dashboard.pages.compare import render_compare_page
from qrc.dashboard.pages.shapley import render_shapley_page
from qrc.pipeline.context import PipelineContext
from qrc.pipeline.runner import ALL_BENCHMARKS, PipelineRunner, load_config

st.set_page_config(
    page_title="QRC Volatility Forecaster",
    page_icon="📈",
    layout="wide",
)

st.title("QRC Volatility Forecaster")
st.caption("Quantum Reservoir Computing for S&P 500 Realized Volatility")

# Sidebar controls
with st.sidebar:
    st.header("Configuration")
    symbol = st.text_input("Symbol", value="^GSPC")
    model = st.selectbox("Model", ["QR2", "QR1"])
    start_date = st.text_input("Start Date", value="1950-01-01")
    end_date = st.text_input("End Date", value="2017-12-31")

    st.divider()
    st.subheader("Benchmark Models")
    selected_benchmarks = st.multiselect(
        "Select benchmarks to run",
        options=ALL_BENCHMARKS,
        default=["HAR", "AR1"],
    )

    st.divider()
    st.subheader("Optional Analysis")
    run_shap = st.checkbox("Run SHAP analysis", value=False)
    run_mcs_flag = st.checkbox("Run Model Confidence Set", value=False)

    st.divider()
    run_button = st.button("Run Forecast", type="primary")

# Session state
if "context" not in st.session_state:
    st.session_state.context = None

if run_button:
    config = load_config()
    config["symbol"] = symbol
    config["model"] = model
    config["start_date"] = start_date
    config["end_date"] = end_date

    progress_bar = st.progress(0.0, text="Initializing pipeline...")

    def _update_progress(fraction: float, label: str) -> None:
        progress_bar.progress(min(fraction, 1.0), text=label)

    runner = PipelineRunner(
        config,
        run_shap=run_shap,
        run_mcs=run_mcs_flag,
        selected_benchmarks=selected_benchmarks,
        progress_callback=_update_progress,
    )
    ctx = runner.run()
    st.session_state.context = ctx

    progress_bar.empty()

    if ctx.errors:
        st.error(f"Pipeline encountered errors: {ctx.errors}")
    else:
        st.success("Pipeline completed successfully!")

# Render pages — only when context is available (lazy)
if st.session_state.context is not None:
    ctx = st.session_state.context
    tab1, tab2, tab3, tab4 = st.tabs([
        "Forecast", "Backtest", "Compare", "SHAP",
    ])
    with tab1:
        render_forecast_page(ctx)
    with tab2:
        render_backtest_page(ctx)
    with tab3:
        render_compare_page(ctx)
    with tab4:
        render_shapley_page(ctx)
else:
    st.info("Configure parameters in the sidebar and click 'Run Forecast' to start.")
