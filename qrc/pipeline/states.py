"""Pipeline state machine: states and transitions."""

from enum import Enum, auto


class PipelineState(Enum):
    IDLE = auto()
    CACHE_CHECK = auto()
    FETCH_EQUITY = auto()
    FETCH_FF = auto()
    FETCH_FRED = auto()
    FETCH_SHILLER = auto()
    ASSEMBLE = auto()
    PREPROCESS = auto()
    SELECT_FEATURES = auto()
    BUILD_RESERVOIR = auto()
    ENCODE_EVOLVE = auto()
    MEASURE = auto()
    TRAIN_READOUT = auto()
    FORECAST = auto()
    RUN_BENCHMARKS = auto()
    COMPUTE_METRICS = auto()
    RUN_MCS = auto()
    RUN_DM = auto()
    COMPUTE_SHAPLEY = auto()
    RENDER = auto()
    DONE = auto()
    ERROR = auto()


TRANSITIONS: dict[PipelineState, list[PipelineState]] = {
    PipelineState.IDLE: [PipelineState.CACHE_CHECK],
    PipelineState.CACHE_CHECK: [
        PipelineState.FETCH_EQUITY,
        PipelineState.ASSEMBLE,
    ],
    PipelineState.FETCH_EQUITY: [PipelineState.FETCH_FF],
    PipelineState.FETCH_FF: [PipelineState.FETCH_FRED],
    PipelineState.FETCH_FRED: [PipelineState.FETCH_SHILLER],
    PipelineState.FETCH_SHILLER: [PipelineState.ASSEMBLE],
    PipelineState.ASSEMBLE: [PipelineState.PREPROCESS],
    PipelineState.PREPROCESS: [PipelineState.SELECT_FEATURES],
    PipelineState.SELECT_FEATURES: [PipelineState.BUILD_RESERVOIR],
    PipelineState.BUILD_RESERVOIR: [PipelineState.ENCODE_EVOLVE],
    PipelineState.ENCODE_EVOLVE: [PipelineState.MEASURE],
    PipelineState.MEASURE: [PipelineState.TRAIN_READOUT],
    PipelineState.TRAIN_READOUT: [PipelineState.FORECAST],
    PipelineState.FORECAST: [PipelineState.RUN_BENCHMARKS],
    PipelineState.RUN_BENCHMARKS: [PipelineState.COMPUTE_METRICS],
    PipelineState.COMPUTE_METRICS: [
        PipelineState.RUN_MCS,
        PipelineState.RUN_DM,
        PipelineState.COMPUTE_SHAPLEY,
    ],
    PipelineState.RUN_MCS: [PipelineState.RENDER],
    PipelineState.RUN_DM: [PipelineState.RENDER],
    PipelineState.COMPUTE_SHAPLEY: [PipelineState.RENDER],
    PipelineState.RENDER: [PipelineState.DONE],
    PipelineState.DONE: [PipelineState.IDLE],
    PipelineState.ERROR: [PipelineState.IDLE],
}
