"""Pipeline runner: orchestrates the full QRC forecasting pipeline."""

import logging
import os

import numpy as np
import pandas as pd
import yaml

from qrc.core.ensemble import QR2Ensemble
from qrc.core.hamiltonian import IsingHamiltonian
from qrc.core.readout import RidgeReadout
from qrc.core.reservoir import QuantumReservoir
from qrc.evaluation.dm_test import dm_matrix
from qrc.evaluation.mcs import model_confidence_set
from qrc.evaluation.metrics import compute_all_metrics, mse
from qrc.features.assembler import assemble_features
from qrc.features.cache import DataCache
from qrc.features.preprocessor import Preprocessor
from qrc.models.arx import AR1Model, AR3Model, ARMAXModel
from qrc.models.classical_rc import ClassicalRCModel, ClassicalRCXModel
from qrc.models.har import HARModel, HARXModel
from qrc.models.lstm import LSTMModel, LSTMXModel
from qrc.pipeline.context import PipelineContext
from qrc.pipeline.states import PipelineState

logger = logging.getLogger(__name__)


def load_config(path: str = "config/default.yaml") -> dict:
    """Load YAML config."""
    with open(path) as f:
        return yaml.safe_load(f)


_hamiltonian_cache: dict[tuple, tuple] = {}


def _get_get_cached_hamiltonian(n_qubits: int, random_seed: int, tau: float):
    """Build IsingHamiltonian and compute unitary, cached across reruns."""
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx() is not None:
        import streamlit as st

        @st.cache_resource
        def _st_cached(n_qubits, random_seed, tau):
            ham = IsingHamiltonian(n_qubits, random_seed)
            U = ham.get_unitary(tau)
            return ham, U

        return _st_cached(n_qubits, random_seed, tau)

    key = (n_qubits, random_seed, tau)
    if key not in _hamiltonian_cache:
        ham = IsingHamiltonian(n_qubits, random_seed)
        U = ham.get_unitary(tau)
        _hamiltonian_cache[key] = (ham, U)
    return _hamiltonian_cache[key]


ALL_BENCHMARKS = ["HAR", "HARX", "AR1", "AR3", "ARMAX", "LSTM", "LSTMX", "RC", "RCX"]


class PipelineRunner:
    """Orchestrates the full QRC volatility forecasting pipeline."""

    def __init__(
        self,
        config: dict,
        context: PipelineContext | None = None,
        run_shap: bool = False,
        run_mcs: bool = False,
        selected_benchmarks: list[str] | None = None,
        progress_callback=None,
    ):
        self.config = config
        self.ctx = context or PipelineContext(config=config)
        self._cache = DataCache()
        self._preprocessor = Preprocessor()
        self.run_shap = run_shap
        self.run_mcs = run_mcs
        self.selected_benchmarks = selected_benchmarks if selected_benchmarks is not None else ALL_BENCHMARKS
        self._progress_callback = progress_callback

    def _transition(self, new_state: PipelineState) -> None:
        logger.info("Pipeline: %s -> %s", self.ctx.state.name, new_state.name)
        self.ctx.state = new_state

    def _report_progress(self, fraction: float, label: str) -> None:
        if self._progress_callback:
            self._progress_callback(fraction, label)

    def run(self) -> PipelineContext:
        """Execute the full pipeline."""
        try:
            self._report_progress(0.0, "Checking cache / fetching data...")
            self._transition(PipelineState.CACHE_CHECK)
            self._cache_check()

            self._report_progress(0.10, "Assembling features...")
            self._transition(PipelineState.ASSEMBLE)
            self._assemble()

            self._report_progress(0.15, "Preprocessing data...")
            self._transition(PipelineState.PREPROCESS)
            self._preprocess()

            self._report_progress(0.20, "Selecting features...")
            self._transition(PipelineState.SELECT_FEATURES)
            self._select_features()

            self._report_progress(0.30, "Building quantum reservoir...")
            self._transition(PipelineState.BUILD_RESERVOIR)
            self._build_reservoir()

            self._report_progress(0.35, "Encoding & evolving...")
            self._transition(PipelineState.ENCODE_EVOLVE)
            # Encoding and evolution happen inside forecast

            self._transition(PipelineState.MEASURE)
            # Measurement happens inside forecast

            self._transition(PipelineState.TRAIN_READOUT)
            # Training happens inside forecast

            self._report_progress(0.40, "Running QRC forecast...")
            self._transition(PipelineState.FORECAST)
            self._forecast()

            self._report_progress(0.55, "Running benchmark models...")
            self._transition(PipelineState.RUN_BENCHMARKS)
            self._run_benchmarks()

            self._report_progress(0.75, "Computing metrics...")
            self._transition(PipelineState.COMPUTE_METRICS)
            self._compute_metrics()

            if self.run_mcs:
                self._report_progress(0.80, "Running Model Confidence Set...")
                self._transition(PipelineState.RUN_MCS)
                self._run_mcs()
            else:
                self.ctx.mcs_results = None

            self._report_progress(0.85, "Running Diebold-Mariano tests...")
            self._transition(PipelineState.RUN_DM)
            self._run_dm()

            if self.run_shap:
                self._report_progress(0.90, "Computing SHAP values...")
                self._transition(PipelineState.COMPUTE_SHAPLEY)
                self._compute_shapley()
            else:
                self.ctx.shap_values = None

            self._report_progress(0.95, "Rendering...")
            self._transition(PipelineState.RENDER)
            self._report_progress(1.0, "Done!")
            self._transition(PipelineState.DONE)

        except Exception as e:
            logger.exception("Pipeline error: %s", e)
            self.ctx.errors.append(str(e))
            self._transition(PipelineState.ERROR)
            self.ctx.state = PipelineState.IDLE

        return self.ctx

    def run_single(self, symbol: str | None = None, model: str | None = None) -> PipelineContext:
        """Convenience wrapper for CLI/dashboard use."""
        if symbol:
            self.config["symbol"] = symbol
        if model:
            self.config["model"] = model
        return self.run()

    def _cache_check(self) -> None:
        cache_dir = self.config.get("cache_dir", ".cache")
        cache_path = os.path.join(cache_dir, "assembled.parquet")
        if self.config.get("cache_data") and self._cache.is_fresh(cache_path):
            logger.info("Loading cached assembled data")
            self.ctx.raw_df = self._cache.load(cache_path)
        else:
            self._fetch_all()
            if self.config.get("cache_data") and self.ctx.raw_df is not None:
                self._cache.save(self.ctx.raw_df, cache_path)

    def _fetch_all(self) -> None:
        start = self.config["start_date"]
        end = self.config["end_date"]

        self._transition(PipelineState.FETCH_EQUITY)
        from qrc.features.equity import fetch_equity_features
        equity_df = fetch_equity_features(self.config["symbol"], start, end)

        self._transition(PipelineState.FETCH_FF)
        from qrc.features.fama_french import fetch_fama_french
        ff_df = fetch_fama_french(start, end)

        self._transition(PipelineState.FETCH_FRED)
        from qrc.features.fred import fetch_fred_features
        fred_df = fetch_fred_features(start, end)

        self._transition(PipelineState.FETCH_SHILLER)
        from qrc.features.shiller import fetch_shiller_features
        shiller_df = fetch_shiller_features(start, end)

        self.ctx.raw_df = assemble_features(equity_df, ff_df, fred_df, shiller_df)

    def _assemble(self) -> None:
        # Already assembled in _cache_check or _fetch_all
        if self.ctx.raw_df is None:
            raise RuntimeError("No data available after fetch/cache stage")

    def _preprocess(self) -> None:
        train_window = self.config.get("train_window", 571)
        self.ctx.scaled_df, self.ctx.train_idx = self._preprocessor.fit_transform(
            self.ctx.raw_df, train_window
        )

    def _select_features(self) -> None:
        n_input = self.config.get("n_input", 7)
        # Use all available features (excluding RV as target)
        all_features = [c for c in self.ctx.scaled_df.columns if c != "RV"]

        if len(all_features) <= n_input:
            self.ctx.selected_features = all_features
        else:
            # Simple selection: use forward selection
            from qrc.selection.forward_select import forward_feature_select

            def _eval_features(feature_names: list[str]) -> float:
                """Quick MSE evaluation on training data."""
                cfg = self.config
                n_qubits = cfg["n_qubits"]
                n_in = len(feature_names)
                n_hid = n_qubits - n_in
                if n_hid <= 0:
                    n_hid = 1
                    n_in = n_qubits - 1

                tau = cfg.get("tau", 1.0)
                seed = cfg.get("random_seed", 42)
                ham, U = _get_cached_hamiltonian(n_qubits, seed, tau)
                res = QuantumReservoir(n_in, n_hid, U)

                train = self.ctx.scaled_df.iloc[:self.ctx.train_idx]
                features = train[feature_names[:n_in]].values
                target = train["RV"].values

                M = res.run_sequence(features, cfg.get("memory_depth", 3))
                readout = RidgeReadout(cfg.get("ridge_delta", 1e-8))
                readout.fit(M, target)
                pred = readout.predict(M)
                return mse(target, pred)

            self.ctx.selected_features = forward_feature_select(
                self.ctx.scaled_df,
                "RV",
                n_input,
                _eval_features,
            )

    def _build_reservoir(self) -> None:
        cfg = self.config
        n_qubits = cfg["n_qubits"]
        tau = cfg.get("tau", 1.0)
        seed = cfg.get("random_seed", 42)

        ham, U = _get_cached_hamiltonian(n_qubits, seed, tau)
        self.ctx.hamiltonian = ham
        self.ctx.unitary = U

    def _forecast(self) -> None:
        """Rolling-window QRC forecast."""
        cfg = self.config
        n_qubits = cfg["n_qubits"]
        n_input = min(len(self.ctx.selected_features), cfg.get("n_input", 7))
        n_hidden = n_qubits - n_input
        tau = cfg.get("tau", 1.0)
        memory_depth = cfg.get("memory_depth", 3)
        delta = cfg.get("ridge_delta", 1e-8)
        model_type = cfg.get("model", "QR2")

        features = self.ctx.scaled_df[self.ctx.selected_features[:n_input]].values
        rv_target = self.ctx.scaled_df["RV"].values
        train_end = self.ctx.train_idx
        T = len(rv_target)

        ham = self.ctx.hamiltonian
        U_full = ham.get_unitary(tau)

        predictions = []
        for t in range(train_end, T - 1):
            train_features = features[:t]
            train_rv = rv_target[:t]

            res1 = QuantumReservoir(n_input, n_hidden, U_full)
            M1 = res1.run_sequence(train_features, memory_depth)

            if model_type == "QR2":
                U_half = ham.get_unitary(tau / 2.0)
                res2 = QuantumReservoir(n_input, n_hidden, U_half)
                M2 = res2.run_sequence(train_features, memory_depth)
                M_train = np.hstack([M1, M2])
            else:
                M_train = M1

            readout = RidgeReadout(delta)
            readout.fit(M_train, train_rv)

            # Predict next step
            test_feat = features[t:t + 1]
            res1_test = QuantumReservoir(n_input, n_hidden, U_full)
            M1_test = res1_test.run_sequence(
                np.vstack([train_features[-memory_depth:], test_feat]),
                memory_depth,
            )

            if model_type == "QR2":
                res2_test = QuantumReservoir(n_input, n_hidden, U_half)
                M2_test = res2_test.run_sequence(
                    np.vstack([train_features[-memory_depth:], test_feat]),
                    memory_depth,
                )
                M_test = np.hstack([M1_test[-1:], M2_test[-1:]])
            else:
                M_test = M1_test[-1:]

            pred = readout.predict(M_test)
            predictions.append(pred[0])

        model_name = model_type
        self.ctx.predictions = self.ctx.predictions or {}
        self.ctx.predictions[model_name] = np.array(predictions)

        # Store the measurement matrix from the last window for SHAP
        self.ctx.measurement_matrix = M_train if model_type == "QR2" else M1

    def _run_benchmarks(self) -> None:
        """Run selected classical benchmark models."""
        rv = self.ctx.raw_df["RV"].values
        train_end = self.ctx.train_idx
        T = len(rv)

        # Exogenous features (unscaled)
        exog_cols = [c for c in self.ctx.raw_df.columns if c not in ["RV", "RVq", "RVa"]]
        exog = self.ctx.raw_df[exog_cols].values if exog_cols else None

        all_benchmarks = {
            "HAR": HARModel,
            "HARX": HARXModel,
            "AR1": AR1Model,
            "AR3": AR3Model,
            "ARMAX": ARMAXModel,
            "LSTM": LSTMModel,
            "LSTMX": LSTMXModel,
            "RC": ClassicalRCModel,
            "RCX": ClassicalRCXModel,
        }

        self.ctx.predictions = self.ctx.predictions or {}

        for name in self.selected_benchmarks:
            if name not in all_benchmarks:
                continue
            model_cls = all_benchmarks[name]
            model = model_cls()
            logger.info("Running benchmark: %s", name)
            try:
                needs_exog = name in ("HARX", "ARMAX", "LSTMX", "RCX")
                train_rv = rv[:train_end]
                train_x = exog[:train_end] if needs_exog and exog is not None else None
                test_rv = rv[train_end:]
                test_x = exog[train_end:] if needs_exog and exog is not None else None

                if needs_exog and train_x is not None:
                    model.fit(train_rv, train_x)
                    preds = model.predict(test_rv, test_x)
                else:
                    model.fit(train_rv)
                    preds = model.predict(test_rv)

                self.ctx.predictions[name] = np.array(preds).flatten()
            except Exception as e:
                logger.warning("Benchmark %s failed: %s", name, e)

    def _compute_metrics(self) -> None:
        rv = self.ctx.raw_df["RV"].values
        train_end = self.ctx.train_idx
        y_true = rv[train_end + 1:]  # +1 because we predict one step ahead
        self.ctx.metrics = compute_all_metrics(self.ctx.predictions, y_true)

    def _run_mcs(self) -> None:
        rv = self.ctx.raw_df["RV"].values
        train_end = self.ctx.train_idx
        y_true = rv[train_end + 1:]

        # Build per-period loss matrix
        losses = {}
        for name, preds in self.ctx.predictions.items():
            n = min(len(y_true), len(preds))
            losses[name] = (y_true[:n] - preds[:n]) ** 2

        # Align to same length
        min_len = min(len(v) for v in losses.values()) if losses else 0
        loss_matrix = pd.DataFrame({k: v[:min_len] for k, v in losses.items()})

        alpha = self.config.get("mcs_alpha", 0.05)
        self.ctx.mcs_results = model_confidence_set(
            loss_matrix, alpha=alpha,
            random_seed=self.config.get("random_seed", 42),
        )

    def _run_dm(self) -> None:
        rv = self.ctx.raw_df["RV"].values
        train_end = self.ctx.train_idx
        y_true = rv[train_end + 1:]
        self.ctx.dm_results = dm_matrix(self.ctx.predictions, y_true, mse)

    def _compute_shapley(self) -> None:
        # Skip SHAP if no measurement matrix
        if self.ctx.measurement_matrix is None:
            return
        try:
            from qrc.selection.shapley import compute_shapley

            M = self.ctx.measurement_matrix
            rv = self.ctx.raw_df["RV"].values
            train_end = self.ctx.train_idx
            train_rv = rv[:train_end]

            readout = RidgeReadout(self.config.get("ridge_delta", 1e-8))
            readout.fit(M[:len(train_rv)], train_rv[:M.shape[0]])

            feature_names = [f"q{i}" for i in range(M.shape[1])]
            self.ctx.shap_values, _ = compute_shapley(
                readout.predict, M[:len(train_rv)], feature_names
            )
        except Exception as e:
            logger.warning("SHAP computation failed: %s", e)
