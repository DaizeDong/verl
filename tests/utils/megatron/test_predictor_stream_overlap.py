"""Smoke tests for the predictor side-stream overlap helpers in router_replay_patch.

These tests exercise only the host-side bookkeeping (env flag, lazy stream init guard,
async-ratio queue + flush) so they pass on CPU-only hosts. The actual CUDA stream
scheduling in patched_forward needs a GPU, which is covered by integration runs.
"""

import importlib
import os
import sys

import pytest

pytest.importorskip("megatron.core")

from verl.utils.megatron import router_replay_patch as rrp


def _reload_module():
    importlib.reload(rrp)


def test_env_flag_default_off(monkeypatch):
    monkeypatch.delenv("VERL_PREDICTOR_STREAM_OVERLAP", raising=False)
    assert rrp._predictor_stream_overlap_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE", "Yes"])
def test_env_flag_truthy(monkeypatch, value):
    monkeypatch.setenv("VERL_PREDICTOR_STREAM_OVERLAP", value)
    assert rrp._predictor_stream_overlap_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_env_flag_falsy(monkeypatch, value):
    monkeypatch.setenv("VERL_PREDICTOR_STREAM_OVERLAP", value)
    assert rrp._predictor_stream_overlap_enabled() is False


def test_get_predictor_stream_returns_none_without_cuda(monkeypatch):
    """On CPU-only hosts, the helper must short-circuit and return None rather than crash."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rrp.RouterReplay._predictor_stream = None
    assert rrp.RouterReplay.get_predictor_stream() is None


def test_async_ratio_queue_flush_resolves_to_floats():
    """Queue a couple of zero-d tensors and verify the flush materializes them
    into the float tracker that wandb consumes."""
    import torch

    rrp.RouterReplay._pending_bias_ratio_tensors.clear()
    rrp.RouterReplay.predictive_bias_ratio_tracker.clear()

    rrp.RouterReplay.record_predictive_bias_ratio_async(0, torch.tensor(0.25))
    rrp.RouterReplay.record_predictive_bias_ratio_async(7, torch.tensor(0.5))

    assert len(rrp.RouterReplay._pending_bias_ratio_tensors) == 2
    assert rrp.RouterReplay.predictive_bias_ratio_tracker == []

    rrp.RouterReplay._flush_pending_predictive_metrics()

    assert rrp.RouterReplay._pending_bias_ratio_tensors == []
    assert rrp.RouterReplay.predictive_bias_ratio_tracker == [(0, 0.25), (7, 0.5)]


def test_get_and_clear_predictive_metrics_drains_pending():
    """End-to-end: queue -> get_and_clear -> avg ratio reflects queued tensors."""
    import torch

    rrp.RouterReplay._pending_bias_ratio_tensors.clear()
    rrp.RouterReplay.predictive_bias_ratio_tracker.clear()

    rrp.RouterReplay.record_predictive_bias_ratio_async(0, torch.tensor(0.4))
    rrp.RouterReplay.record_predictive_bias_ratio_async(1, torch.tensor(0.6))

    metrics = rrp.RouterReplay.get_and_clear_predictive_metrics()
    assert metrics.get("predictive_bias_to_logits_ratio") == pytest.approx(0.5)
    # Trackers cleared after consumption.
    assert rrp.RouterReplay.predictive_bias_ratio_tracker == []
    assert rrp.RouterReplay._pending_bias_ratio_tensors == []


def test_sync_predictor_stream_noop_when_uninitialized():
    """Without CUDA / before first use, sync must be a no-op rather than raising."""
    rrp.RouterReplay._predictor_stream = None
    # Should not raise.
    rrp.RouterReplay.sync_predictor_stream()
