"""Forecast evaluation metrics: MSE and QLIKE."""

import numpy as np
import pandas as pd


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((y_true - y_pred) ** 2))


def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """QLIKE loss: mean(y_true/y_pred - log(y_true/y_pred) - 1).

    Both y_true and y_pred must be positive.
    """
    ratio = y_true / y_pred
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def compute_all_metrics(
    predictions_dict: dict[str, np.ndarray], y_true: np.ndarray
) -> pd.DataFrame:
    """Compute MSE and QLIKE for all models.

    Args:
        predictions_dict: {model_name: predictions array}.
        y_true: Actual values.

    Returns:
        DataFrame with columns [MSE, QLIKE], indexed by model name.
    """
    rows = {}
    for name, y_pred in predictions_dict.items():
        # Align lengths
        n = min(len(y_true), len(y_pred))
        yt = y_true[:n]
        yp = y_pred[:n]

        # Clamp predictions to positive for QLIKE
        yp_pos = np.clip(yp, 1e-12, None)

        rows[name] = {"MSE": mse(yt, yp), "QLIKE": qlike(yt, yp_pos)}

    return pd.DataFrame(rows).T.sort_values("MSE")
