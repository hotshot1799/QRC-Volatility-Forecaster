"""QR2 ensemble: two reservoirs at different evolution times."""

import numpy as np

from qrc.core.reservoir import QuantumReservoir


class QR2Ensemble:
    """Run two quantum reservoirs at tau and tau/2, concatenate measurements."""

    def __init__(self, reservoir_tau: QuantumReservoir, reservoir_tau_half: QuantumReservoir):
        self.reservoir_tau = reservoir_tau
        self.reservoir_tau_half = reservoir_tau_half

    def run(self, feature_matrix: np.ndarray, memory_depth: int) -> np.ndarray:
        """Run both reservoirs and concatenate measurement matrices.

        Returns:
            M: shape (T, 2 * n_total).
        """
        M1 = self.reservoir_tau.run_sequence(feature_matrix, memory_depth)
        M2 = self.reservoir_tau_half.run_sequence(feature_matrix, memory_depth)
        return np.hstack([M1, M2])
