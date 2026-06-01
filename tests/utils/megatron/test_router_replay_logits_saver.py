import pytest
import torch

pytest.importorskip("megatron.core")

from megatron.core import parallel_state as mpu

from verl.utils.megatron.router_replay_patch import RouterReplay, RouterReplayCacheAction
from verl.utils.megatron.router_replay_saver import RouterReplayLogitsSaver
from verl.utils.megatron import router_replay_utils as rru


def _empty_logits_cache():
    return {
        "compute_log_prob": [],
        "training": [],
        "router_weights": {},
        "global_token_ids": [],
        "predictive_bias": [],
        "predictive_bias_token_ids": [],
    }


def _patch_single_rank(monkeypatch):
    monkeypatch.setattr(mpu, "get_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(mpu, "get_pipeline_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(mpu, "get_data_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: 1)


def test_router_replay_logits_save_keeps_non_prefix_positions_aligned(tmp_path, monkeypatch):
    _patch_single_rank(monkeypatch)

    monkeypatch.setattr(RouterReplay, "logits_cache", _empty_logits_cache())
    monkeypatch.setattr(RouterReplay, "enable_logits_recording", True)
    monkeypatch.setattr(RouterReplay, "current_cache_action", RouterReplayCacheAction.COMPUTE_LOG_PROB)
    monkeypatch.setattr(RouterReplay, "current_sample_indices", torch.tensor([2, 5], dtype=torch.long))
    monkeypatch.setattr(RouterReplay, "current_full_token_count", 6)
    monkeypatch.setattr(RouterReplay, "sampled_log_prob_token_ids", {103, 106})
    monkeypatch.setattr(RouterReplay, "logits_save_sample_rate", 0.1)

    token_ids = torch.tensor([101, 102, 103, 104, 105, 106], dtype=torch.long)
    logits = torch.tensor(
        [
            [0.0, 10.0, 20.0],
            [1.0, 11.0, 21.0],
            [2.0, 12.0, 22.0],
            [3.0, 13.0, 23.0],
            [4.0, 14.0, 24.0],
            [5.0, 15.0, 25.0],
        ],
        dtype=torch.float32,
    )
    predictive_bias = torch.tensor(
        [
            [210.0, 211.0],
            [250.0, 251.0],
        ],
        dtype=torch.float32,
    )

    selected_rows = torch.tensor([2, 5], dtype=torch.long)
    RouterReplay.logits_cache["global_token_ids"].append(token_ids.index_select(0, selected_rows))
    RouterReplay.set_predictive_bias_token_ids(token_ids.index_select(0, selected_rows))
    RouterReplay.record_logits(logits, layer_idx=7)
    RouterReplay.record_predictive_bias(predictive_bias, layer_idx=7)

    logits_data = RouterReplay.get_and_clear_logits_cache()
    saver = RouterReplayLogitsSaver(str(tmp_path))
    saver._save_logits_sync(logits_data, "training_7_mini0")

    saved_path = tmp_path / "7" / "training_7_mini0_tp0_pp0.pt"
    saved = torch.load(saved_path, map_location="cpu")

    expected_logits = logits.index_select(0, selected_rows)
    expected_bias = predictive_bias.unsqueeze(1)
    expected_token_ids = token_ids.index_select(0, selected_rows)

    assert torch.equal(saved["global_token_ids"], expected_token_ids)
    assert torch.equal(saved["predictive_bias_token_ids"], expected_token_ids)
    assert torch.equal(saved["compute_log_prob"][7], expected_logits)
    assert torch.equal(saved["predictive_bias"][7], expected_bias)
    assert not torch.equal(saved["compute_log_prob"][7], logits[:2])


class _DummyRouter:
    def __init__(self):
        self.bias_predictor = None
        self.recorded_old_bias = None
        self.recorded_old_inputs = None
        self.recorded_old_logits = None
        self.recorded_valid_mask = None

    def set_predictive_bias(self, bias):
        self.recorded_old_bias = bias

    def clear_predictive_bias(self):
        self.recorded_old_bias = None

    def set_predictive_data(self, inputs, logits, valid_mask, loss_scale=1.0):
        self.recorded_old_inputs = inputs
        self.recorded_old_logits = logits
        self.recorded_valid_mask = valid_mask


def test_set_r3_predictive_bias_data_uses_explicit_positions(monkeypatch):
    routers = [_DummyRouter(), _DummyRouter()]
    monkeypatch.setattr(rru.RouterReplayHelper, "get_micro_batch_router_list", lambda tf_config, vp_rank=None: routers)
    monkeypatch.setattr(rru, "get_current_rank_layer_info", lambda tf_config, vp_rank=None: {"start": 0})
    monkeypatch.setattr(RouterReplay, "current_predictive_bias_token_ids", None)
    monkeypatch.setattr(RouterReplay, "current_predictive_bias_token_ids_recorded", False)

    tf_config = type("Cfg", (), {"params_dtype": torch.float32})()
    attention_mask = torch.tensor([[0, 1, 1, 1, 1, 1]], dtype=torch.bool)
    global_token_ids = torch.tensor([[900, 901, 902, 903, 904, 905]], dtype=torch.long)
    old_token_positions = [torch.tensor([1, 4], dtype=torch.int32)]
    old_bias = [
        torch.tensor(
            [
                [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]],
                [[40.0, 41.0, 42.0], [50.0, 51.0, 52.0]],
            ],
            dtype=torch.float32,
        ).numpy()
    ]

    rru.set_router_predictive_bias_data(
        old_bias,
        attention_mask,
        tf_config,
        old_token_positions_list=old_token_positions,
        global_token_ids=global_token_ids,
    )

    torch.testing.assert_close(routers[0].recorded_old_bias, torch.tensor([[10.0, 11.0, 12.0], [40.0, 41.0, 42.0]]))
    torch.testing.assert_close(routers[1].recorded_old_bias, torch.tensor([[20.0, 21.0, 22.0], [50.0, 51.0, 52.0]]))
    assert torch.equal(RouterReplay.current_predictive_bias_token_ids, torch.tensor([902, 905]))


def test_set_predictive_data_uses_fallback_recorded_positions(monkeypatch):
    routers = [_DummyRouter(), _DummyRouter()]
    monkeypatch.setattr(rru.RouterReplayHelper, "get_micro_batch_router_list", lambda tf_config, vp_rank=None: routers)

    tf_config = type("Cfg", (), {"params_dtype": torch.float32})()
    attention_mask = torch.tensor([[0, 1, 1, 1, 1, 1]], dtype=torch.bool)
    old_token_positions = [torch.tensor([1, 4], dtype=torch.int32)]
    old_inputs = [
        torch.tensor(
            [
                [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]],
                [[40.0, 41.0, 42.0], [50.0, 51.0, 52.0]],
            ],
            dtype=torch.float32,
        ).numpy()
    ]
    old_logits = [
        torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            dtype=torch.float32,
        ).numpy()
    ]

    rru.set_router_predictive_data(
        old_inputs,
        old_logits,
        attention_mask,
        tf_config,
        old_token_positions_list=old_token_positions,
    )

    assert torch.equal(routers[0].recorded_valid_mask, torch.tensor([False, True, False, False, True]))
    torch.testing.assert_close(routers[0].recorded_old_inputs.squeeze(1), torch.tensor([[10.0, 11.0, 12.0], [40.0, 41.0, 42.0]]))
    torch.testing.assert_close(routers[1].recorded_old_logits.squeeze(1), torch.tensor([[3.0, 4.0], [7.0, 8.0]]))


def test_set_predictive_data_skips_downsampled_inputs_without_positions(monkeypatch):
    routers = [_DummyRouter(), _DummyRouter()]
    monkeypatch.setattr(rru.RouterReplayHelper, "get_micro_batch_router_list", lambda tf_config, vp_rank=None: routers)

    tf_config = type("Cfg", (), {"params_dtype": torch.float32})()
    attention_mask = torch.tensor([[0, 1, 1, 1, 1, 1]], dtype=torch.bool)
    old_inputs = [torch.zeros((2, 2, 3), dtype=torch.float32).numpy()]
    old_logits = [torch.zeros((2, 2, 2), dtype=torch.float32).numpy()]

    rru.set_router_predictive_data(
        old_inputs,
        old_logits,
        attention_mask,
        tf_config,
        old_token_positions_list=None,
    )

    assert routers[0].recorded_old_inputs is None
    assert routers[1].recorded_old_logits is None
