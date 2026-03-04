"""Ridge regression readout layer."""

import numpy as np


class RidgeReadout:
    """Linear readout with Tikhonov (ridge) regularisation."""

    def __init__(self, delta: float = 1e-8):
        self.delta = delta
        self.W_out: np.ndarray | None = None

    def fit(self, M: np.ndarray, rv_targets: np.ndarray) -> None:
        """Solve for the output weight matrix.

        W_out = rv_targets @ M.T @ inv(M @ M.T + delta * I)

        Args:
            M: measurement matrix, shape (T_train, n_features).
            rv_targets: target RV values, shape (T_train,).
        """
        # M: (T, D), targets: (T,)
        # W_out = targets^T M (M^T M + delta I)^{-1}  => shape (D,)
        MtM = M.T @ M  # (D, D)
        reg = self.delta * np.eye(MtM.shape[0], dtype=np.float64)
        self.W_out = np.linalg.solve(MtM + reg, M.T @ rv_targets)

    def predict(self, M: np.ndarray) -> np.ndarray:
        """Predict using the fitted readout.

        Args:
            M: measurement matrix, shape (T, n_features).

        Returns:
            Predictions, shape (T,).
        """
        if self.W_out is None:
            raise RuntimeError("Readout not fitted. Call fit() first.")
        return M @ self.W_out
