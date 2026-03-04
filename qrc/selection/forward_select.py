"""Forward feature selection for quantum reservoir."""

import numpy as np
from tqdm import tqdm


def forward_feature_select(
    scaled_df,
    target_col: str,
    n_features: int,
    reservoir_builder,
) -> list[str]:
    """Greedy forward selection of features by MSE on training data.

    Args:
        scaled_df: Scaled DataFrame with all candidate features.
        target_col: Name of the target column (e.g. "RV").
        n_features: Number of features to select.
        reservoir_builder: Callable(feature_names) -> (predict_fn, mse)
            that builds a reservoir model on the given features and
            returns the training MSE.

    Returns:
        Ordered list of selected feature names.
    """
    candidates = [c for c in scaled_df.columns if c != target_col]
    selected: list[str] = []

    for step in tqdm(range(n_features), desc="Forward selection"):
        best_mse = np.inf
        best_feature = None

        for feat in candidates:
            trial = selected + [feat]
            mse = reservoir_builder(trial)
            if mse < best_mse:
                best_mse = mse
                best_feature = feat

        if best_feature is None:
            break

        selected.append(best_feature)
        candidates.remove(best_feature)

    return selected
