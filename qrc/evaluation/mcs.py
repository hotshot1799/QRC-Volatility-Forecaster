"""Model Confidence Set (Hansen, Lunde, Nason 2011)."""

import numpy as np
import pandas as pd


def model_confidence_set(
    loss_matrix: pd.DataFrame,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
) -> dict[str, float]:
    """Compute the Model Confidence Set.

    Args:
        loss_matrix: DataFrame of shape (T, n_models) with per-period losses.
        alpha: Significance level.
        n_bootstrap: Number of bootstrap replications.
        random_seed: For reproducibility.

    Returns:
        Dict mapping surviving model names to their p-values.
    """
    rng = np.random.RandomState(random_seed)
    models = list(loss_matrix.columns)
    T = len(loss_matrix)
    p_values = {m: 1.0 for m in models}
    surviving = list(models)

    while len(surviving) > 1:
        losses = loss_matrix[surviving].values  # (T, n_surviving)
        n_surv = len(surviving)

        # Pairwise loss differentials
        d_bar = np.zeros((n_surv, n_surv))
        for i in range(n_surv):
            for j in range(n_surv):
                d_bar[i, j] = np.mean(losses[:, i] - losses[:, j])

        # Average loss of each model relative to the set
        t_bar = np.mean(d_bar, axis=1)  # (n_surv,)

        # Bootstrap variance estimate
        t_bar_boot = np.zeros((n_bootstrap, n_surv))
        for b in range(n_bootstrap):
            idx = rng.choice(T, size=T, replace=True)
            boot_losses = losses[idx]
            d_bar_b = np.zeros((n_surv, n_surv))
            for i in range(n_surv):
                for j in range(n_surv):
                    d_bar_b[i, j] = np.mean(boot_losses[:, i] - boot_losses[:, j])
            t_bar_boot[b] = np.mean(d_bar_b, axis=1)

        var_t = np.var(t_bar_boot, axis=0, ddof=1)
        var_t = np.maximum(var_t, 1e-12)

        # Range statistic: max standardised t_bar
        t_stat = t_bar / np.sqrt(var_t)
        TR = np.max(t_stat)

        # Bootstrap distribution of TR
        TR_boot = np.zeros(n_bootstrap)
        for b in range(n_bootstrap):
            t_stat_b = (t_bar_boot[b] - t_bar) / np.sqrt(var_t)
            TR_boot[b] = np.max(t_stat_b)

        p_val = np.mean(TR_boot >= TR)

        if p_val < alpha:
            # Eliminate the worst model
            worst_idx = np.argmax(t_stat)
            worst_model = surviving[worst_idx]
            p_values[worst_model] = p_val
            surviving.pop(worst_idx)
        else:
            break

    # Surviving models keep p_value = 1.0 (or last computed)
    return p_values
