# QRC Volatility Forecaster

Quantum Reservoir Computing (QRC) for S&P 500 monthly realized volatility forecasting, based on the academic paper arXiv:2505.13933 (Li, Mukhopadhyay, Bayat, Habibnia, 2025).

## Overview

This project implements a 10-qubit transverse-field Ising Hamiltonian as a fixed quantum reservoir to forecast S&P 500 monthly realized volatility. It compares the QRC approach against 8 classical benchmarks: HAR, HARX, AR1, AR3, ARMAX, LSTM, LSTMX, RC, and RCX.

The full stack is **100% open source** — no paid APIs, no Qiskit. The only external credential needed is a free FRED API key.

## Installation

```bash
pip install -e .
```

## Configuration

```bash
cp .env.example .env
# Edit .env and add your FRED API key
```

Get a free FRED API key at: https://fred.stlouisfed.org/api/key

## Usage

### Streamlit Dashboard

```bash
streamlit run qrc/dashboard/app.py
```

### CLI

```bash
# Run a forecast
qrc forecast --symbol ^GSPC --model QR2

# Compare all models
qrc compare --symbol ^GSPC

# Run backtest
qrc backtest --symbol ^GSPC --model QR2

# SHAP feature importance
qrc shapley --symbol ^GSPC --model QR2
```

### Run Tests

```bash
pytest tests/
```

## Architecture

- **`qrc/core/`** — Quantum reservoir: Ising Hamiltonian, reservoir dynamics, ensemble (QR2), ridge readout
- **`qrc/features/`** — Data fetching (yfinance, Fama-French, FRED, Shiller) and preprocessing
- **`qrc/selection/`** — Forward feature selection and SHAP analysis
- **`qrc/models/`** — Classical benchmarks (HAR, AR, LSTM, Echo State Network)
- **`qrc/evaluation/`** — MSE, QLIKE, Model Confidence Set, Diebold-Mariano test
- **`qrc/pipeline/`** — State machine orchestrating the full pipeline
- **`qrc/dashboard/`** — Streamlit UI with Plotly charts
- **`qrc/cli/`** — Typer CLI with Rich output

## How It Works

1. **Data Collection**: Monthly features from Yahoo Finance (RV, RVq, RVa), Fama-French factors (MKT, HML, SMB, STR), FRED macro data (TB, INF, IP, DEF), and Shiller ratios (DP, EP)
2. **Preprocessing**: Min-max scaling to [-pi, +pi] fitted on training window only
3. **Feature Selection**: Greedy forward selection using reservoir MSE
4. **Quantum Reservoir**: Features encoded via RY rotations into input qubits, evolved under a fixed Ising unitary, Pauli-Z expectations measured
5. **QR2 Ensemble**: Two reservoirs at tau and tau/2, measurements concatenated
6. **Readout**: Ridge regression maps measurements to RV predictions
7. **Rolling Forecast**: Train on months 1..t, predict t+1, advance by 1 (245 out-of-sample windows)
8. **Evaluation**: MSE, QLIKE, Model Confidence Set, pairwise Diebold-Mariano tests
