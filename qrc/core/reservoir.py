"""Quantum reservoir: encode, evolve, measure."""

import numpy as np


def _ry_gate(theta: float) -> np.ndarray:
    """RY rotation gate."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


class QuantumReservoir:
    """Quantum reservoir driven by a fixed unitary."""

    def __init__(self, n_input: int, n_hidden: int, unitary: np.ndarray):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_total = n_input + n_hidden
        self.U = unitary.astype(np.complex128)
        self.dim_input = 2**n_input
        self.dim_hidden = 2**n_hidden
        self.dim_total = 2**self.n_total

        # Pauli Z for each qubit (pre-build for measure)
        I2 = np.eye(2, dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        self._Z_ops = []
        for q in range(self.n_total):
            ops = [I2] * self.n_total
            ops[q] = Z
            full = ops[0]
            for m in ops[1:]:
                full = np.kron(full, m)
            self._Z_ops.append(full)

    def encode(self, x_vec: np.ndarray) -> np.ndarray:
        """Encode feature vector into input density matrix via RY rotations.

        Args:
            x_vec: array of length n_input (scaled features).

        Returns:
            rho_input: density matrix of shape (dim_input, dim_input).
        """
        # Start from |0> for each qubit, apply RY(x_j)
        state = np.array([1.0, 0.0], dtype=np.complex128)
        rho_q0 = np.outer(state, state.conj())

        rho_list = []
        for j in range(self.n_input):
            ry = _ry_gate(x_vec[j])
            q_state = ry @ state
            rho_list.append(np.outer(q_state, q_state.conj()))

        # Tensor product of all input qubit states
        rho_input = rho_list[0]
        for rho_q in rho_list[1:]:
            rho_input = np.kron(rho_input, rho_q)

        return rho_input

    def evolve(self, rho_input: np.ndarray, rho_hidden: np.ndarray) -> np.ndarray:
        """Evolve the joint input+hidden state under the unitary.

        Returns:
            rho_evolved: density matrix of shape (dim_total, dim_total).
        """
        rho_total = np.kron(rho_input, rho_hidden)
        rho_evolved = self.U @ rho_total @ self.U.conj().T
        return rho_evolved

    def partial_trace_input(self, rho_evolved: np.ndarray) -> np.ndarray:
        """Trace out input qubits, returning the hidden subsystem density matrix.

        Returns:
            rho_hidden: density matrix of shape (dim_hidden, dim_hidden).
        """
        d_in = self.dim_input
        d_hid = self.dim_hidden
        # Reshape to (d_in, d_hid, d_in, d_hid) and trace over input indices
        rho_reshaped = rho_evolved.reshape(d_in, d_hid, d_in, d_hid)
        rho_hidden = np.einsum("ijkj->ik", rho_reshaped)
        return rho_hidden

    def measure(self, rho_full: np.ndarray) -> np.ndarray:
        """Compute Pauli-Z expectation values for all qubits.

        Returns:
            Array of length n_total with <Z_j> values.
        """
        expectations = np.zeros(self.n_total, dtype=np.float64)
        for j, Z_j in enumerate(self._Z_ops):
            expectations[j] = np.trace(Z_j @ rho_full).real
        return expectations

    def run_sequence(
        self, feature_matrix: np.ndarray, memory_depth: int
    ) -> np.ndarray:
        """Run the reservoir over a time series.

        Args:
            feature_matrix: shape (T, n_input) — scaled feature values.
            memory_depth: number of lag steps (k) per timestep.

        Returns:
            M: measurement matrix of shape (T, n_total).
        """
        T = feature_matrix.shape[0]
        M = np.zeros((T, self.n_total), dtype=np.float64)

        # Initial hidden state: |0...0><0...0|
        zero_state = np.zeros(self.dim_hidden, dtype=np.complex128)
        zero_state[0] = 1.0
        rho_hidden = np.outer(zero_state, zero_state.conj())

        for t in range(T):
            # Memory loop: re-encode and evolve k times
            rho_h = rho_hidden.copy()
            start = max(0, t - memory_depth + 1)
            for lag in range(start, t + 1):
                rho_input = self.encode(feature_matrix[lag])
                rho_evolved = self.evolve(rho_input, rho_h)
                rho_h = self.partial_trace_input(rho_evolved)

            # Measure on the full evolved state from the last step
            rho_input_final = self.encode(feature_matrix[t])
            rho_full = self.evolve(rho_input_final, rho_h)
            M[t] = self.measure(rho_full)

            # Update hidden state for next timestep
            rho_hidden = self.partial_trace_input(rho_full)

        return M
