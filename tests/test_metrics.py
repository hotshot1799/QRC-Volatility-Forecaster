"""Tests for evaluation metrics."""

import numpy as np
import pytest

from qrc.evaluation.metrics import mse, qlike
from qrc.evaluation.dm_test import dm_test


class TestMSE:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mse(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        expected = np.mean((y_true - y_pred) ** 2)
        assert mse(y_true, y_pred) == pytest.approx(expected)

    def test_symmetric(self):
        y1 = np.array([1.0, 2.0])
        y2 = np.array([3.0, 4.0])
        assert mse(y1, y2) == pytest.approx(mse(y2, y1))


class TestQLIKE:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert qlike(y, y) == pytest.approx(0.0)

    def test_positive(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        result = qlike(y_true, y_pred)
        assert result >= 0.0


class TestDMTest:
    def test_returns_p_value_in_range(self):
        rng = np.random.RandomState(42)
        loss1 = rng.rand(100)
        loss2 = rng.rand(100)
        stat, p = dm_test(loss1, loss2)
        assert 0.0 <= p <= 1.0

    def test_identical_losses(self):
        losses = np.ones(100)
        stat, p = dm_test(losses, losses)
        assert stat == pytest.approx(0.0)
        assert p == pytest.approx(1.0)
