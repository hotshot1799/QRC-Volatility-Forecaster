"""Classical Echo State Network (reservoir computing) models via reservoirpy."""

import numpy as np

try:
    from reservoirpy.nodes import ESN, Reservoir, Ridge
except ImportError:
    ESN = None
    Reservoir = None
    Ridge = None


class ClassicalRCModel:
    """Classical RC: 50 neurons, no exogenous features."""

    def __init__(self, units: int = 50, leak_rate: float = 0.6,
                 spectral_radius: float = 0.9, ridge: float = 1e-8,
                 seed: int = 42):
        self.units = units
        self.leak_rate = leak_rate
        self.spectral_radius = spectral_radius
        self.ridge_val = ridge
        self.seed = seed
        self._model = None

    def fit(self, train_rv: np.ndarray, train_x: np.ndarray | None = None) -> None:
        if Reservoir is None:
            raise ImportError("reservoirpy is required for classical RC models")

        reservoir = Reservoir(
            self.units,
            lr=self.leak_rate,
            sr=self.spectral_radius,
            seed=self.seed,
        )
        readout = Ridge(ridge=self.ridge_val)
        self._model = reservoir >> readout

        X = train_rv[:-1].reshape(-1, 1)
        y = train_rv[1:].reshape(-1, 1)
        self._model.fit(X, y)

    def predict(self, test_rv: np.ndarray, test_x: np.ndarray | None = None) -> np.ndarray:
        X = test_rv[:-1].reshape(-1, 1)
        preds = self._model.run(X)
        return np.array(preds).flatten()


class ClassicalRCXModel:
    """Classical RC with exogenous features: 20 neurons."""

    def __init__(self, units: int = 20, leak_rate: float = 0.6,
                 spectral_radius: float = 0.9, ridge: float = 1e-8,
                 seed: int = 42):
        self.units = units
        self.leak_rate = leak_rate
        self.spectral_radius = spectral_radius
        self.ridge_val = ridge
        self.seed = seed
        self._model = None

    def fit(self, train_rv: np.ndarray, train_x: np.ndarray) -> None:
        if Reservoir is None:
            raise ImportError("reservoirpy is required for classical RC models")

        reservoir = Reservoir(
            self.units,
            lr=self.leak_rate,
            sr=self.spectral_radius,
            seed=self.seed,
        )
        readout = Ridge(ridge=self.ridge_val)
        self._model = reservoir >> readout

        X = np.hstack([train_rv[:-1].reshape(-1, 1), train_x[:-1]])
        y = train_rv[1:].reshape(-1, 1)
        self._model.fit(X, y)

    def predict(self, test_rv: np.ndarray, test_x: np.ndarray) -> np.ndarray:
        X = np.hstack([test_rv[:-1].reshape(-1, 1), test_x[:-1]])
        preds = self._model.run(X)
        return np.array(preds).flatten()
