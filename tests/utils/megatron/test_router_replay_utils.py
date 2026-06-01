import pytest
import torch

pytest.importorskip("megatron.core")

from verl.utils.megatron.router_replay_utils import (
    build_predictive_valid_mask,
    restore_predictive_states_to_batch_order,
)


def test_build_predictive_valid_mask_prefix_lengths():
    attention_mask = torch.tensor(
        [
            [0, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )

    valid_mask, selected_lens = build_predictive_valid_mask(
        attention_mask=attention_mask,
        valid_indices=[0, 1],
        old_lengths=[2, 3],
        old_token_positions_list=None,
    )

    assert selected_lens == [2, 3]
    assert valid_mask.tolist() == [True, True, False, True, True, True, False]


def test_build_predictive_valid_mask_uses_explicit_positions():
    attention_mask = torch.tensor(
        [
            [0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0],
        ],
        dtype=torch.bool,
    )
    old_token_positions = [
        torch.tensor([0, 2], dtype=torch.int32),
        torch.tensor([1, 2], dtype=torch.int32),
    ]

    valid_mask, selected_lens = build_predictive_valid_mask(
        attention_mask=attention_mask,
        valid_indices=[0, 1],
        old_lengths=[2, 2],
        old_token_positions_list=old_token_positions,
    )

    assert selected_lens == [2, 2]
    assert valid_mask.tolist() == [True, False, True, False, False, True, True, False]


def test_build_predictive_valid_mask_handles_no_valid_samples():
    attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)

    valid_mask, selected_lens = build_predictive_valid_mask(
        attention_mask=attention_mask,
        valid_indices=[],
        old_lengths=[],
        old_token_positions_list=None,
    )

    assert selected_lens == []
    assert valid_mask.tolist() == [False, False]


def test_restore_predictive_states_to_batch_order_dynamic_indices():
    old_inputs = ["sample3-inputs", "sample0-inputs", "sample2-inputs"]
    old_logits = ["sample3-logits", "sample0-logits", "sample2-logits"]
    old_positions = ["sample3-positions", "sample0-positions", "sample2-positions"]
    sampled_masks = torch.tensor([True, False, True, True])
    indices = [[3], [1], [0], [2]]

    restored_inputs, restored_logits, restored_positions = restore_predictive_states_to_batch_order(
        old_inputs,
        old_logits,
        old_positions,
        sampled_masks,
        indices=indices,
    )

    assert restored_inputs == ["sample0-inputs", None, "sample2-inputs", "sample3-inputs"]
    assert restored_logits == ["sample0-logits", None, "sample2-logits", "sample3-logits"]
    assert restored_positions == ["sample0-positions", None, "sample2-positions", "sample3-positions"]


def test_restore_predictive_states_to_batch_order_sequential_masks():
    old_inputs = ["sample0-inputs", "sample2-inputs"]
    old_logits = ["sample0-logits", "sample2-logits"]
    old_positions = ["sample0-positions", "sample2-positions"]

    restored_inputs, restored_logits, restored_positions = restore_predictive_states_to_batch_order(
        old_inputs,
        old_logits,
        old_positions,
        [True, False, True],
    )

    assert restored_inputs == ["sample0-inputs", None, "sample2-inputs"]
    assert restored_logits == ["sample0-logits", None, "sample2-logits"]
    assert restored_positions == ["sample0-positions", None, "sample2-positions"]
