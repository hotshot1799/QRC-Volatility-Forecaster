"""Tests for feature processing modules."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from qrc.features.cache import DataCache
from qrc.features.assembler import assemble_features
from qrc.features.preprocessor import Preprocessor


class TestPreprocessor:
    def test_scales_to_pi_range(self):
        df = pd.DataFrame({
            "RV": [0.01, 0.02, 0.03, 0.04, 0.05],
            "X1": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        pp = Preprocessor()
        scaled, idx = pp.fit_transform(df, train_end_idx=4)

        # Training data (fit window) should be in [-pi, pi]
        train_vals = scaled.iloc[:4].values
        assert train_vals.min() >= -np.pi - 1e-10
        assert train_vals.max() <= np.pi + 1e-10
        # Out-of-sample data may exceed [-pi, pi] — this is correct behaviour
        assert idx == 4

    def test_inverse_transform_rv(self):
        df = pd.DataFrame({
            "RV": [0.01, 0.02, 0.03, 0.04, 0.05],
            "X1": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        pp = Preprocessor()
        scaled, _ = pp.fit_transform(df, train_end_idx=4)

        original_rv = df["RV"].values[:4]
        scaled_rv = scaled["RV"].values[:4]
        recovered = pp.inverse_transform_rv(scaled_rv)
        np.testing.assert_allclose(recovered, original_rv, atol=1e-10)


class TestAssembler:
    def test_correct_column_count(self):
        dates = pd.date_range("2000-01-31", periods=24, freq="ME")
        equity = pd.DataFrame(
            {"RV": np.random.rand(24), "RVq": np.random.rand(24), "RVa": np.random.rand(24)},
            index=dates,
        )
        ff = pd.DataFrame(
            {"MKT": np.random.rand(24), "HML": np.random.rand(24),
             "SMB": np.random.rand(24), "STR": np.random.rand(24)},
            index=dates,
        )
        fred = pd.DataFrame(
            {"TB": np.random.rand(24), "INF": np.random.rand(24),
             "IP": np.random.rand(24), "DEF": np.random.rand(24)},
            index=dates,
        )
        shiller = pd.DataFrame(
            {"DP": np.random.rand(24), "EP": np.random.rand(24)},
            index=dates,
        )
        result = assemble_features(equity, ff, fred, shiller)
        assert len(result.columns) == 13


class TestDataCache:
    def test_save_load_roundtrip(self):
        cache = DataCache()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.parquet")
            cache.save(df, path)
            loaded = cache.load(path)
            pd.testing.assert_frame_equal(df, loaded)

    def test_is_fresh(self):
        cache = DataCache()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.parquet")
            df = pd.DataFrame({"a": [1]})
            cache.save(df, path)
            assert cache.is_fresh(path, max_age_days=1)
            assert not cache.is_fresh(os.path.join(tmpdir, "nonexistent.parquet"))
