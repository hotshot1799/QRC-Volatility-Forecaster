"""SHAP-based feature importance for the quantum reservoir."""

import numpy as np
import pandas as pd
import shap


def compute_shapley(
    model_predict_fn,
    feature_matrix: np.ndarray,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Compute SHAP values using KernelExplainer.

    Args:
        model_predict_fn: Callable that takes feature array and returns predictions.
        feature_matrix: Array of shape (T, n_features).
        feature_names: List of feature names.

    Returns:
        Tuple of (shap_values_df, mean_abs_shap per feature).
    """
    # Use a small background sample for efficiency
    n_bg = min(50, feature_matrix.shape[0])
    background = shap.sample(feature_matrix, n_bg)

    explainer = shap.KernelExplainer(model_predict_fn, background)
    shap_values = explainer.shap_values(feature_matrix)

    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    mean_abs.name = "mean_abs_shap"

    return shap_df, mean_abs
