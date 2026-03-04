"""Feature scaling to [-pi, pi]."""

import numpy as np
import pandas as pd


class Preprocessor:
    """Min-max scaler that maps features to [-pi, +pi]."""

    def __init__(self):
        self.min_vals: np.ndarray | None = None
        self.max_vals: np.ndarray | None = None
        self.columns: list[str] | None = None

    def fit_transform(
        self, df: pd.DataFrame, train_end_idx: int
    ) -> tuple[pd.DataFrame, int]:
        """Fit scaler on training data, transform entire dataset.

        Args:
            df: Full feature DataFrame.
            train_end_idx: Index of last training sample (exclusive).

        Returns:
            Tuple of (scaled DataFrame, train_end_idx).
        """
        self.columns = list(df.columns)
        train = df.iloc[:train_end_idx]

        self.min_vals = train.values.min(axis=0)
        self.max_vals = train.values.max(axis=0)

        # Avoid division by zero
        ranges = self.max_vals - self.min_vals
        ranges[ranges == 0] = 1.0

        # Scale to [0, 1] then to [-pi, pi]
        normalised = (df.values - self.min_vals) / ranges
        scaled = normalised * 2 * np.pi - np.pi

        scaled_df = pd.DataFrame(scaled, index=df.index, columns=df.columns)
        return scaled_df, train_end_idx

    def inverse_transform_rv(self, scaled_rv: np.ndarray) -> np.ndarray:
        """Undo scaling for the RV column only."""
        if self.min_vals is None:
            raise RuntimeError("Preprocessor not fitted.")
        rv_idx = self.columns.index("RV")
        rv_min = self.min_vals[rv_idx]
        rv_max = self.max_vals[rv_idx]
        rv_range = rv_max - rv_min if rv_max != rv_min else 1.0

        # Reverse: scaled = norm * 2pi - pi  =>  norm = (scaled + pi) / (2pi)
        normalised = (scaled_rv + np.pi) / (2 * np.pi)
        return normalised * rv_range + rv_min
