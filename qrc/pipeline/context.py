"""Pipeline context: shared state across pipeline stages."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from qrc.pipeline.states import PipelineState


@dataclass
class PipelineContext:
    # Config
    config: dict

    # Data
    raw_df: pd.DataFrame | None = None
    scaled_df: pd.DataFrame | None = None
    selected_features: list | None = None
    train_idx: int | None = None

    # Quantum
    hamiltonian: object = None
    unitary: np.ndarray | None = None
    measurement_matrix: np.ndarray | None = None

    # Results
    predictions: dict | None = None  # {model_name: np.ndarray}
    metrics: pd.DataFrame | None = None
    mcs_results: dict | None = None
    dm_results: pd.DataFrame | None = None
    shap_values: pd.DataFrame | None = None

    # State tracking
    state: PipelineState = PipelineState.IDLE
    errors: list = field(default_factory=list)
