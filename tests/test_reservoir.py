"""Tests for quantum reservoir core modules."""

import numpy as np
import pytest

from qrc.core.hamiltonian import IsingHamiltonian
from qrc.core.reservoir import QuantumReservoir


class TestIsingHamiltonian:
    def test_matrix_dimensions(self):
        n = 4
        ham = IsingHamiltonian(n, random_seed=42)
        H = ham.get_matrix()
        assert H.shape == (2**n, 2**n)

    def test_matrix_hermitian(self):
        ham = IsingHamiltonian(4, random_seed=42)
        H = ham.get_matrix()
        np.testing.assert_allclose(H, H.T, atol=1e-12)

    def test_unitary_is_unitary(self):
        ham = IsingHamiltonian(4, random_seed=42)
        U = ham.get_unitary(1.0)
        I = np.eye(U.shape[0])
        product = U @ U.conj().T
        np.testing.assert_allclose(product, I, atol=1e-10)

    def test_unitary_caching(self):
        ham = IsingHamiltonian(4, random_seed=42)
        U1 = ham.get_unitary(1.0)
        U2 = ham.get_unitary(1.0)
        assert U1 is U2  # Same object (cached)

    def test_different_tau_different_unitary(self):
        ham = IsingHamiltonian(4, random_seed=42)
        U1 = ham.get_unitary(1.0)
        U2 = ham.get_unitary(0.5)
        assert not np.allclose(U1, U2)


class TestQuantumReservoir:
    @pytest.fixture
    def reservoir(self):
        n_qubits = 4
        n_input = 2
        n_hidden = 2
        ham = IsingHamiltonian(n_qubits, random_seed=42)
        U = ham.get_unitary(1.0)
        return QuantumReservoir(n_input, n_hidden, U)

    def test_encode_density_matrix(self, reservoir):
        x = np.array([0.5, -0.3])
        rho = reservoir.encode(x)
        assert rho.shape == (4, 4)  # 2^2 x 2^2
        # Should be a valid density matrix: trace = 1
        np.testing.assert_allclose(np.trace(rho).real, 1.0, atol=1e-12)

    def test_partial_trace_reduces_dimension(self, reservoir):
        x = np.array([0.5, -0.3])
        rho_in = reservoir.encode(x)
        zero = np.zeros(reservoir.dim_hidden, dtype=np.complex128)
        zero[0] = 1.0
        rho_hid = np.outer(zero, zero.conj())
        rho_evolved = reservoir.evolve(rho_in, rho_hid)
        rho_traced = reservoir.partial_trace_input(rho_evolved)
        assert rho_traced.shape == (reservoir.dim_hidden, reservoir.dim_hidden)
        np.testing.assert_allclose(np.trace(rho_traced).real, 1.0, atol=1e-10)

    def test_measure_returns_real_array(self, reservoir):
        dim = reservoir.dim_total
        rho = np.eye(dim, dtype=np.complex128) / dim
        m = reservoir.measure(rho)
        assert m.shape == (reservoir.n_total,)
        assert m.dtype == np.float64

    def test_run_sequence_shape(self, reservoir):
        T = 10
        features = np.random.randn(T, reservoir.n_input)
        M = reservoir.run_sequence(features, memory_depth=2)
        assert M.shape == (T, reservoir.n_total)
        assert M.dtype == np.float64
