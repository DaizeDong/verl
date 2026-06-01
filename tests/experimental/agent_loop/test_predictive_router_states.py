import types

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopWorkerBase, _InternalAgentLoopOutput
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop


class _FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **kwargs):
        del messages, add_generation_prompt, tokenize, kwargs
        return [11, 12, 13]


class _FakeServerManager:
    async def generate(self, **kwargs):
        del kwargs
        return types.SimpleNamespace(
            token_ids=[21, 22, 23],
            log_probs=[-0.1, -0.2, -0.3],
            routed_experts=np.ones((5, 2, 2), dtype=np.int64),
            router_inputs=np.ones((5, 2, 4), dtype=np.float16),
            router_logits=np.ones((5, 2, 3), dtype=np.float16) * 2,
            router_bias=np.ones((5, 2, 3), dtype=np.float16) * 3,
            router_token_positions=np.arange(5, dtype=np.int32),
        )


def _make_internal_output(router_inputs=None, router_logits=None, router_bias=None, router_token_positions=None):
    return _InternalAgentLoopOutput(
        prompt_ids=torch.tensor([[1, 2]]),
        response_ids=torch.tensor([[3, 4]]),
        input_ids=torch.tensor([[1, 2, 3, 4]]),
        position_ids=torch.tensor([[0, 1, 2, 3]]),
        response_mask=torch.tensor([[1, 1]]),
        attention_mask=torch.tensor([[1, 1, 1, 1]]),
        response_logprobs=None,
        routed_experts=None,
        router_inputs=router_inputs,
        router_logits=router_logits,
        router_bias=router_bias,
        router_token_positions=router_token_positions,
        multi_modal_inputs=None,
        multi_modal_data={},
        reward_score=None,
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields={},
    )


@pytest.mark.asyncio
async def test_single_turn_agent_loop_preserves_router_states():
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": 16,
                    "response_length": 2,
                }
            },
            "data": {},
        }
    )
    loop = SingleTurnAgentLoop(
        trainer_config=types.SimpleNamespace(config=config),
        server_manager=_FakeServerManager(),
        tokenizer=_FakeTokenizer(),
        processor=None,
    )

    output = await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}])

    assert output.router_inputs is not None
    assert output.router_logits is not None
    assert output.router_bias is not None
    assert output.router_token_positions is not None
    assert output.router_inputs.shape == (5, 2, 4)
    assert output.router_logits.shape == (5, 2, 3)
    assert output.router_bias.shape == (5, 2, 3)
    assert output.router_token_positions.shape == (5,)


def test_postprocess_only_emits_router_state_keys_when_samples_exist():
    worker = AgentLoopWorkerBase.__new__(AgentLoopWorkerBase)
    worker.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "router_replay": {
                        "mode": "R3",
                        "enable_bias_predictor": True,
                    }
                }
            }
        }
    )

    valid_inputs = np.ones((4, 2, 8), dtype=np.float16)
    valid_logits = np.ones((4, 2, 3), dtype=np.float16)
    valid_bias = np.ones((4, 2, 3), dtype=np.float16)
    valid_positions = np.arange(4, dtype=np.int32)
    batch = worker._postprocess(
        [
            _make_internal_output(valid_inputs, valid_logits, valid_bias, valid_positions),
            _make_internal_output(None, None, None),
        ]
    )

    assert "old_inputs" in batch.non_tensor_batch
    assert "old_logits" in batch.non_tensor_batch
    assert "old_bias" in batch.non_tensor_batch
    assert "old_token_positions" in batch.non_tensor_batch
    assert batch.non_tensor_batch["old_inputs"][0] is not None
    assert batch.non_tensor_batch["old_inputs"][1] is None
    assert np.array_equal(batch.non_tensor_batch["old_token_positions"][0], valid_positions)


def test_postprocess_skips_all_none_router_state_keys():
    worker = AgentLoopWorkerBase.__new__(AgentLoopWorkerBase)
    worker.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "router_replay": {
                        "mode": "R3",
                        "enable_bias_predictor": True,
                    }
                }
            }
        }
    )

    batch = worker._postprocess(
        [
            _make_internal_output(None, None, None),
            _make_internal_output(None, None, None),
        ]
    )

    assert "old_inputs" not in batch.non_tensor_batch
    assert "old_logits" not in batch.non_tensor_batch
    assert "old_bias" not in batch.non_tensor_batch
    assert "old_token_positions" not in batch.non_tensor_batch
