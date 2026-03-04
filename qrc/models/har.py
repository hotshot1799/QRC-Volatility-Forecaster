"""HAR and HARX volatility models."""

import numpy as np
import pandas as pd
import statsmodels.api as sm


class HARModel:
    """Heterogeneous Autoregressive model for realized volatility.

    Regressors: RV_{t-1}, RV_weekly (5-day avg), RV_monthly (22-day avg).
    """

    def __init__(self):
        self._model = None

    def _build_har_features(self, rv: np.ndarray) -> np.ndarray:
        """Build HAR lag features from an RV series."""
        T = len(rv)
        X = np.zeros((T, 3), dtype=np.float64)
        for t in range(22, T):
            X[t, 0] = rv[t - 1]  # daily lag
            X[t, 1] = np.mean(rv[max(0, t - 5):t])  # weekly avg
            X[t, 2] = np.mean(rv[max(0, t - 22):t])  # monthly avg
        return X

    def fit(self, train_rv: np.ndarray, train_x: np.ndarray | None = None) -> None:
        """Fit HAR model on training RV data."""
        X = self._build_har_features(train_rv)
        valid = slice(22, len(train_rv))
        X_valid = sm.add_constant(X[valid])
        y_valid = train_rv[valid]
        self._model = sm.OLS(y_valid, X_valid).fit()

    def predict(self, test_rv: np.ndarray, test_x: np.ndarray | None = None) -> np.ndarray:
        """Predict next-step RV."""
        X = self._build_har_features(test_rv)
        valid = slice(22, len(test_rv))
        X_valid = sm.add_constant(X[valid])
        return self._model.predict(X_valid)


class HARXModel:
    """HAR model augmented with exogenous features."""

    def __init__(self):
        self._model = None

    def _build_harx_features(
        self, rv: np.ndarray, exog: np.ndarray
    ) -> np.ndarray:
        T = len(rv)
        har = np.zeros((T, 3), dtype=np.float64)
        for t in range(22, T):
            har[t, 0] = rv[t - 1]
            har[t, 1] = np.mean(rv[max(0, t - 5):t])
            har[t, 2] = np.mean(rv[max(0, t - 22):t])
        return np.hstack([har, exog])

    def fit(self, train_rv: np.ndarray, train_x: np.ndarray) -> None:
        X = self._build_harx_features(train_rv, train_x)
        valid = slice(22, len(train_rv))
        X_valid = sm.add_constant(X[valid])
        y_valid = train_rv[valid]
        self._model = sm.OLS(y_valid, X_valid).fit()

    def predict(self, test_rv: np.ndarray, test_x: np.ndarray) -> np.ndarray:
        X = self._build_harx_features(test_rv, test_x)
        valid = slice(22, len(test_rv))
        X_valid = sm.add_constant(X[valid])
        return self._model.predict(X_valid)
