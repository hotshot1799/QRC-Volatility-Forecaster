"""Transverse-field Ising Hamiltonian for quantum reservoir computing."""

import numpy as np
from scipy.linalg import expm

# Pauli matrices
X_PAULI = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
Z_PAULI = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
I2 = np.eye(2, dtype=np.float64)


def _multi_kron(matrices: list[np.ndarray]) -> np.ndarray:
    """Compute the tensor product of a list of matrices."""
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result


def _single_qubit_op(op: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """Embed a single-qubit operator into the full Hilbert space."""
    ops = [I2] * n_qubits
    ops[qubit] = op
    return _multi_kron(ops)


def _two_qubit_op(
    op: np.ndarray, qubit_i: int, qubit_j: int, n_qubits: int
) -> np.ndarray:
    """Embed a two-qubit XX interaction into the full Hilbert space."""
    ops = [I2] * n_qubits
    ops[qubit_i] = op
    ops[qubit_j] = op
    return _multi_kron(ops)


class IsingHamiltonian:
    """Fully-connected transverse-field Ising Hamiltonian.

    H = sum_{i<j} J_ij (X_i X_j) + v * sum_i Z_i
    """

    def __init__(self, n_qubits: int, random_seed: int = 42):
        self.n_qubits = n_qubits
        self.dim = 2**n_qubits
        self.random_seed = random_seed
        self._unitary_cache: dict[float, np.ndarray] = {}

        rng = np.random.RandomState(random_seed)

        # Build Hamiltonian
        H = np.zeros((self.dim, self.dim), dtype=np.float64)

        # XX interaction terms with random couplings
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                J_ij = rng.uniform(0.0, 1.0)
                H += J_ij * _two_qubit_op(X_PAULI, i, j, n_qubits)

        # Transverse field term: v * sum_i Z_i, v=1.0
        v = 1.0
        for i in range(n_qubits):
            H += v * _single_qubit_op(Z_PAULI, i, n_qubits)

        self._H = H

    def get_matrix(self) -> np.ndarray:
        """Return the Hamiltonian matrix."""
        return self._H.copy()

    def get_unitary(self, tau: float) -> np.ndarray:
        """Return the time-evolution unitary U = exp(-i H tau). Cached per tau."""
        if tau not in self._unitary_cache:
            U = expm(-1j * self._H * tau)
            self._unitary_cache[tau] = U.astype(np.complex128)
        return self._unitary_cache[tau]
