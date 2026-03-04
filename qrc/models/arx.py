"""AR(1), AR(3), and ARMAX models for realized volatility."""

import numpy as np
import statsmodels.api as sm


class AR1Model:
    """AR(1) model: regress RV_t on RV_{t-1}."""

    def __init__(self):
        self._model = None

    def fit(self, train_rv: np.ndarray, train_x: np.ndarray | None = None) -> None:
        y = train_rv[1:]
        X = sm.add_constant(train_rv[:-1].reshape(-1, 1))
        self._model = sm.OLS(y, X).fit()

    def predict(self, test_rv: np.ndarray, test_x: np.ndarray | None = None) -> np.ndarray:
        X = sm.add_constant(test_rv[:-1].reshape(-1, 1))
        return self._model.predict(X)


class AR3Model:
    """AR(3) model: regress RV_t on RV_{t-1}, RV_{t-2}, RV_{t-3}."""

    def __init__(self):
        self._model = None

    def _build_lags(self, rv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T = len(rv)
        X = np.column_stack([rv[2:T - 1], rv[1:T - 2], rv[:T - 3]])
        y = rv[3:]
        return X, y

    def fit(self, train_rv: np.ndarray, train_x: np.ndarray | None = None) -> None:
        X, y = self._build_lags(train_rv)
        X = sm.add_constant(X)
        self._model = sm.OLS(y, X).fit()

    def predict(self, test_rv: np.ndarray, test_x: np.ndarray | None = None) -> np.ndarray:
        X, _ = self._build_lags(test_rv)
        X = sm.add_constant(X)
        return self._model.predict(X)


class ARMAXModel:
    """ARMAX: 3 lags of RV + all exogenous features."""

    def __init__(self):
        self._model = None

    def fit(self, train_rv: np.ndarray, train_x: np.ndarray) -> None:
        T = len(train_rv)
        lags = np.column_stack([train_rv[2:T - 1], train_rv[1:T - 2], train_rv[:T - 3]])
        exog = train_x[3:]
        X = sm.add_constant(np.hstack([lags, exog]))
        y = train_rv[3:]
        self._model = sm.OLS(y, X).fit()

    def predict(self, test_rv: np.ndarray, test_x: np.ndarray) -> np.ndarray:
        T = len(test_rv)
        lags = np.column_stack([test_rv[2:T - 1], test_rv[1:T - 2], test_rv[:T - 3]])
        exog = test_x[3:]
        X = sm.add_constant(np.hstack([lags, exog]))
        return self._model.predict(X)
