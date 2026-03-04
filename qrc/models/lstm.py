"""LSTM and LSTMX models for realized volatility (PyTorch, CPU only)."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class LSTMModel:
    """LSTM: 2 layers, 60 hidden units, input = lagged RV only."""

    def __init__(self, n_lags: int = 3, hidden_size: int = 60, num_layers: int = 2,
                 lr: float = 0.001, epochs: int = 100):
        self.n_lags = n_lags
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self._net = None

    def _make_sequences(self, rv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for t in range(self.n_lags, len(rv)):
            X.append(rv[t - self.n_lags:t].reshape(-1, 1))
            y.append(rv[t])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def fit(self, train_rv: np.ndarray, train_x: np.ndarray | None = None) -> None:
        X, y = self._make_sequences(train_rv)
        self._net = _LSTMNet(1, self.hidden_size, self.num_layers)
        optimiser = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        self._net.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                pred = self._net(xb)
                loss = loss_fn(pred, yb)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

    def predict(self, test_rv: np.ndarray, test_x: np.ndarray | None = None) -> np.ndarray:
        X, _ = self._make_sequences(test_rv)
        self._net.eval()
        with torch.no_grad():
            return self._net(torch.from_numpy(X)).numpy()


class LSTMXModel:
    """LSTMX: 1 layer, 50 hidden units, input = lagged RV + exogenous."""

    def __init__(self, n_lags: int = 3, hidden_size: int = 50, num_layers: int = 1,
                 lr: float = 0.001, epochs: int = 100):
        self.n_lags = n_lags
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self._net = None
        self._input_size = None

    def _make_sequences(
        self, rv: np.ndarray, exog: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for t in range(self.n_lags, len(rv)):
            rv_lags = rv[t - self.n_lags:t].reshape(-1, 1)
            exog_lags = exog[t - self.n_lags:t]
            seq = np.hstack([rv_lags, exog_lags])
            X.append(seq)
            y.append(rv[t])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def fit(self, train_rv: np.ndarray, train_x: np.ndarray) -> None:
        X, y = self._make_sequences(train_rv, train_x)
        self._input_size = X.shape[2]
        self._net = _LSTMNet(self._input_size, self.hidden_size, self.num_layers)
        optimiser = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        self._net.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                pred = self._net(xb)
                loss = loss_fn(pred, yb)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

    def predict(self, test_rv: np.ndarray, test_x: np.ndarray) -> np.ndarray:
        X, _ = self._make_sequences(test_rv, test_x)
        self._net.eval()
        with torch.no_grad():
            return self._net(torch.from_numpy(X)).numpy()
