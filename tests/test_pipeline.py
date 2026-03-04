"""Tests for pipeline state machine and runner."""

import numpy as np
import pandas as pd
import pytest

from qrc.pipeline.states import PipelineState, TRANSITIONS
from qrc.pipeline.context import PipelineContext


class TestPipelineStates:
    def test_all_states_have_transitions(self):
        for state in PipelineState:
            assert state in TRANSITIONS, f"{state} missing from TRANSITIONS"

    def test_idle_transitions_to_cache_check(self):
        assert PipelineState.CACHE_CHECK in TRANSITIONS[PipelineState.IDLE]

    def test_done_transitions_to_idle(self):
        assert PipelineState.IDLE in TRANSITIONS[PipelineState.DONE]

    def test_error_transitions_to_idle(self):
        assert PipelineState.IDLE in TRANSITIONS[PipelineState.ERROR]


class TestPipelineContext:
    def test_default_state_is_idle(self):
        ctx = PipelineContext(config={})
        assert ctx.state == PipelineState.IDLE

    def test_errors_start_empty(self):
        ctx = PipelineContext(config={})
        assert ctx.errors == []

    def test_error_reset(self):
        ctx = PipelineContext(config={})
        ctx.state = PipelineState.ERROR
        ctx.errors.append("test error")
        # Reset
        ctx.state = PipelineState.IDLE
        ctx.errors.clear()
        assert ctx.state == PipelineState.IDLE
        assert len(ctx.errors) == 0
