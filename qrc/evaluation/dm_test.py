"""Diebold-Mariano test with Newey-West HAC variance."""

import numpy as np
import pandas as pd
from scipy import stats


def _newey_west_variance(d: np.ndarray, h: int) -> float:
    """Newey-West HAC variance estimator for loss differential series."""
    T = len(d)
    d_mean = np.mean(d)
    d_centered = d - d_mean

    # Autocovariance at lag 0
    gamma_0 = np.sum(d_centered**2) / T

    # Add autocovariances with Bartlett kernel
    nw_var = gamma_0
    bandwidth = max(1, h - 1)
    for k in range(1, bandwidth + 1):
        weight = 1.0 - k / (bandwidth + 1.0)
        gamma_k = np.sum(d_centered[k:] * d_centered[:-k]) / T
        nw_var += 2 * weight * gamma_k

    return nw_var / T


def dm_test(
    loss_baseline: np.ndarray,
    loss_challenger: np.ndarray,
    h: int = 1,
) -> tuple[float, float]:
    """Diebold-Mariano test.

    Tests H0: E[d_t] = 0 where d_t = loss_baseline_t - loss_challenger_t.
    A negative DM stat means challenger has higher loss (baseline is better).

    Args:
        loss_baseline: Per-period losses for the baseline model.
        loss_challenger: Per-period losses for the challenger model.
        h: Forecast horizon.

    Returns:
        (DM statistic, two-sided p-value).
    """
    d = loss_baseline - loss_challenger
    d_mean = np.mean(d)
    var_d = _newey_west_variance(d, h)

    if var_d < 1e-15:
        return 0.0, 1.0

    dm_stat = d_mean / np.sqrt(var_d)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))

    return float(dm_stat), float(p_value)


def dm_matrix(
    predictions_dict: dict[str, np.ndarray],
    y_true: np.ndarray,
    loss_fn,
) -> pd.DataFrame:
    """Pairwise DM tests for all model pairs.

    Returns:
        DataFrame of p-values (row = baseline, col = challenger).
    """
    names = list(predictions_dict.keys())
    n = len(names)
    pvals = np.ones((n, n), dtype=np.float64)

    # Compute per-period losses
    losses = {}
    for name, preds in predictions_dict.items():
        min_len = min(len(y_true), len(preds))
        yt = y_true[:min_len]
        yp = preds[:min_len]
        losses[name] = (yt - yp) ** 2  # per-period squared loss

    for i in range(n):
        for j in range(n):
            if i != j:
                min_len = min(len(losses[names[i]]), len(losses[names[j]]))
                _, pval = dm_test(
                    losses[names[i]][:min_len],
                    losses[names[j]][:min_len],
                )
                pvals[i, j] = pval

    return pd.DataFrame(pvals, index=names, columns=names)
