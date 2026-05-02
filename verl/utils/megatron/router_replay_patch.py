# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
import logging
import os
import types
import warnings
from enum import Enum
from functools import wraps

import torch
from torch import no_grad

from verl.utils.memory_utils import get_system_memory_info

try:
    import psutil
except ImportError:
    psutil = None

try:
    from megatron.core.transformer.moe.moe_utils import (
        MoEAuxLossAutoScaler,
        apply_random_logits,
        apply_router_token_dropping,
        compute_routing_scores_for_aux_loss,
        group_limited_topk,
    )
    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
except ImportError:
    warnings.warn("NPU not support router replay for now.", stacklevel=2)
    MoEAlltoAllTokenDispatcher = None
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# https://github.com/THUDM/slime/blob/main/slime/utils/routing_replay.py


class RouterReplayAction(Enum):
    RECORD = "record"
    REPLAY_FORWARD = "replay_forward"
    REPLAY_BACKWARD = "replay_backward"


class RouterReplayCacheAction(Enum):
    """Enum for logits cache recording phases."""

    COMPUTE_LOG_PROB = "compute_log_prob"
    TRAINING = "training"


class RouterPredictiveAction(Enum):
    """Enum for router predictive actions."""

    DISABLED = "disabled"
    RECORD = "record"  # R2 log_prob阶段：记录inputs和logits
    SKIP_PREDICTIVE = "skip_predictive"  # training ministep==0：跳过loss计算
    COMPUTE_PREDICTIVE_LOSS = "compute_predictive_loss"  # training ministep>=1：计算loss
    R3_COLLECT_STATS = "r3_collect_stats"  # R3 compute_log_prob阶段：用SGLang返回的bias计算统计


def _r3_predictive_diag_enabled() -> bool:
    return os.getenv("VERL_DEBUG_R3_PREDICTIVE_DIAG", "").lower() in {"1", "true", "yes", "on"}


def _predictive_sync_tokens() -> int:
    value = os.getenv("VERL_PREDICTIVE_SYNC_TOKENS", "").strip()
    if not value or value.lower() in {"null", "none"}:
        return 512
    try:
        parsed = int(value)
    except ValueError:
        logger.warning(
            "[Predictive Routing Replay] Invalid VERL_PREDICTIVE_SYNC_TOKENS=%r; fallback to 512",
            value,
        )
        return 512
    return max(1, parsed)


def _relative_l2(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    lhs_f = lhs.detach().float()
    rhs_f = rhs.detach().float()
    denom = rhs_f.norm().item()
    if denom <= 1e-12:
        return float("inf") if lhs_f.norm().item() > 1e-12 else 0.0
    return (lhs_f - rhs_f).norm().item() / denom


class RouterReplay:
    """
    A class to manage the recording and replaying of MoE routing decisions.
    It holds all router instances and provides static methods to globally
    control recording and replaying.
    """

    # Static variable to hold all router instances, one per MoE layer.
    router_instances = []

    # Global logits cache for recording
    # Structure: {"compute_log_prob": [], "training": [], "router_weights": {},
    #             "global_token_ids": [], "predictive_bias": [], "predictive_bias_token_ids": []}
    # Each list contains tuples of (layer_idx, tensor_cpu), router_weights stores parameter tensors
    # global_token_ids: list of token_ids_tensor
    # predictive_bias: list of (layer_idx, delta_logits_cpu) - bias after scaling
    # predictive_bias_token_ids: token ids aligned row-by-row with R3_COLLECT_STATS predictive_bias
    logits_cache = {
        "compute_log_prob": [],
        "training": [],
        "router_weights": {},
        "global_token_ids": [],
        "predictive_bias": [],
        "predictive_bias_token_ids": [],
    }

    # Flag to enable/disable logits recording
    enable_logits_recording = False

    # Current cache action phase
    current_cache_action = None

    # Current token indices for alignment
    current_token_indices = None

    # Logits save-time sampling state
    # When < 1.0, compute_log_prob subsamples tokens and training phases filter to the same set via global_token_ids.
    # Cleared in get_and_clear_logits_cache() (on save) so each save cycle re-samples independently.
    logits_save_sample_rate = 1.0  # 1.0 = no sampling
    sampled_log_prob_token_ids = None  # set of int token IDs sampled during compute_log_prob
    current_sample_indices = None  # per-micro-batch indices to subsample logits in record_logits
    current_full_token_count = None  # full-sequence valid token count for current micro-batch (used to
                                     # detect SP-sharded logits in record_logits/record_predictive_bias)
    current_predictive_bias_token_ids = None
    current_predictive_bias_token_ids_recorded = False

    # Predictive loss tracking for wandb logging
    replay_topk_accuracy_tracker = []  # List of (layer_idx, accuracy_value)
    predictive_loss_tracker = []  # List of (layer_idx, loss_value)
    predictive_bias_ratio_tracker = []  # List of (layer_idx, ratio_value)
    predictive_topk_accuracy_tracker = []  # List of (layer_idx, accuracy_value)

    def __init__(self):
        """Initializes a RouterReplay instance for a specific layer."""
        self.router_replay_action = None  # Router replay action for this layer
        self.recorded_topk_idx = None  # For recording
        self.target_topk_idx = None  # For replay
        self.replay_backward_list = []  # List of tensors for backward pass replay
        self.layer_idx = len(RouterReplay.router_instances)  # Layer index

        # 🔎 Predictive routing replay (bias predictor)
        self.predictive_action = None
        self.recorded_old_inputs = None  # Stored router inputs from log_prob phase
        self.recorded_old_logits = None  # Stored router logits from log_prob phase
        self.recorded_old_bias = None   # Stored delta_logits from SGLang (R3 mode only)

        RouterReplay.router_instances.append(self)

    """routing replay management"""

    def set_target_indices(self, topk_indices: torch.Tensor):
        """Sets the target topk indices for replay."""
        self.target_topk_idx = topk_indices
        self.replay_backward_list.append(topk_indices)

    def record_indices(self, topk_indices: torch.Tensor):
        """Records the topk indices."""
        self.recorded_topk_idx = topk_indices

    def clear_indices(self):
        """Clears the recorded and target topk indices."""
        self.recorded_topk_idx = None
        self.target_topk_idx = None
        self.replay_backward_list = []

    @staticmethod
    def clear_global_indices():
        """Clears the recorded and target topk indices in all instances."""
        for router in RouterReplay.router_instances:
            router.clear_indices()

    def set_router_replay_action(self, router_replay_action: RouterReplayAction):
        """Sets the router replay action for this layer."""
        self.router_replay_action = router_replay_action

    def clear_router_replay_action(self):
        """Clears the router replay action for this layer."""
        self.router_replay_action = None

    @staticmethod
    def set_global_router_replay_action(router_replay_action: RouterReplayAction):
        """Sets the router replay action for all router instances."""
        for router in RouterReplay.router_instances:
            router.set_router_replay_action(router_replay_action)

    @staticmethod
    def clear_global_router_replay_action():
        """Clears the router replay action for all router instances."""
        for router in RouterReplay.router_instances:
            router.clear_router_replay_action()

    """router cache saving management"""

    @staticmethod
    def set_cache_action(cache_action: RouterReplayCacheAction):
        """Set the current cache action phase."""
        # Reset sampled_log_prob_token_ids at the start of each new compute_log_prob cycle
        # (so next step's sampling is independent). Training phases inherit the set from
        # the preceding compute_log_prob so they can filter to the same tokens.
        if cache_action == RouterReplayCacheAction.COMPUTE_LOG_PROB:
            RouterReplay.sampled_log_prob_token_ids = None
        RouterReplay.current_cache_action = cache_action
        RouterReplay.enable_logits_recording = True

    @staticmethod
    def clear_cache_action():
        """Clear the current cache action phase."""
        RouterReplay.current_cache_action = None
        RouterReplay.enable_logits_recording = False

    @staticmethod
    def get_and_clear_logits_cache():
        """
        Get the current logits cache and clear it.
        Returns a dict with 'compute_log_prob', 'training', 'router_weights' and 'global_token_ids' keys.
        """
        cache = RouterReplay.logits_cache
        RouterReplay.logits_cache = {
            "compute_log_prob": [],
            "training": [],
            "router_weights": {},
            "global_token_ids": [],
            "predictive_bias": [],
            "predictive_bias_token_ids": [],
        }
        # Note: do NOT clear sampled_log_prob_token_ids here — it needs to persist from
        # compute_log_prob save through all training mini-step saves so training phases
        # can filter to the same token set. Cleared at start of next compute_log_prob (set_cache_action).
        RouterReplay.current_sample_indices = None
        RouterReplay.current_predictive_bias_token_ids = None
        RouterReplay.current_predictive_bias_token_ids_recorded = False
        return cache

    @staticmethod
    @no_grad()
    def set_predictive_bias_token_ids(token_ids: torch.Tensor | None):
        """Set token ids that align with R3_COLLECT_STATS predictive_bias rows."""
        if token_ids is None:
            RouterReplay.current_predictive_bias_token_ids = None
        else:
            RouterReplay.current_predictive_bias_token_ids = token_ids.detach().cpu().to(torch.long).contiguous()
        RouterReplay.current_predictive_bias_token_ids_recorded = False

    @staticmethod
    def clear_predictive_bias_token_ids():
        RouterReplay.current_predictive_bias_token_ids = None
        RouterReplay.current_predictive_bias_token_ids_recorded = False

    @staticmethod
    @no_grad()
    def record_global_token_ids(global_token_ids: torch.Tensor):
        """
        Record valid token IDs for the current micro-batch, applying save-time sampling.

        During compute_log_prob: uniformly subsample tokens at logits_save_sample_rate and
        accumulate their IDs into sampled_log_prob_token_ids. During training: filter tokens
        to only those whose IDs are in the sampled set. The resulting current_sample_indices
        is used by record_logits / record_predictive_bias to subsample the saved tensors.

        Must be called BEFORE record_logits within a micro-batch so indices are available.

        Args:
            global_token_ids: Tensor of valid token IDs (shape [num_tokens]).
        """
        # Default: no sampling indices; record_logits will save everything.
        RouterReplay.current_sample_indices = None
        RouterReplay.current_full_token_count = None

        if not RouterReplay.enable_logits_recording or RouterReplay.current_cache_action is None:
            logger.info(f"[record_global_token_ids] Skipping - enable_recording={RouterReplay.enable_logits_recording}, ")
            return

        ids_cpu = global_token_ids.detach().cpu().contiguous()
        sample_rate = RouterReplay.logits_save_sample_rate
        # Store the full-sequence valid token count so record_logits/record_predictive_bias
        # can detect SP-sharded tensors (shape[0] < full_count ⇒ need to gather before indexing).
        RouterReplay.current_full_token_count = int(ids_cpu.shape[0])

        if RouterReplay.current_cache_action == RouterReplayCacheAction.COMPUTE_LOG_PROB:
            # Subsample uniformly by linspace indices; accumulate IDs across micro-batches.
            n = ids_cpu.shape[0]
            if sample_rate is not None and 0.0 < sample_rate < 1.0 and n > 0:
                num_sample = max(1, int(round(n * sample_rate)))
                if num_sample < n:
                    indices = torch.round(torch.linspace(0, n - 1, num_sample)).long().clamp(0, n - 1)
                    RouterReplay.current_sample_indices = indices
                    ids_cpu = ids_cpu[indices]

            if RouterReplay.sampled_log_prob_token_ids is None:
                RouterReplay.sampled_log_prob_token_ids = set()
            RouterReplay.sampled_log_prob_token_ids.update(ids_cpu.tolist())

        elif RouterReplay.current_cache_action == RouterReplayCacheAction.TRAINING:
            # Keep only training tokens whose IDs were sampled in compute_log_prob.
            sampled_set = RouterReplay.sampled_log_prob_token_ids
            if sampled_set is not None and len(sampled_set) > 0 and sample_rate is not None and sample_rate < 1.0:
                ids_list = ids_cpu.tolist()
                mask = torch.tensor([tid in sampled_set for tid in ids_list], dtype=torch.bool)
                indices = mask.nonzero(as_tuple=False).squeeze(-1)
                RouterReplay.current_sample_indices = indices
                ids_cpu = ids_cpu[indices]

        RouterReplay.logits_cache["global_token_ids"].append(ids_cpu)
        logger.info(f"[record_global_token_ids] Recorded global token IDs of shape {ids_cpu.shape} (sample_rate={sample_rate}).")

    @staticmethod
    @no_grad()
    def record_logits(logits: torch.Tensor, layer_idx: int):
        """
        Record logits to cache (moved to CPU to save GPU memory).
        Records to the appropriate cache based on current_cache_action.

        Args:
            logits: The logits tensor from routing computation
            layer_idx: The layer index
        """
        # Debug: log first call
        if layer_idx == 0:
            logger.info(f"[record_logits] Layer 0: enable_recording={RouterReplay.enable_logits_recording}, "
                        f"cache_action={RouterReplay.current_cache_action}, "
                        f"logits_shape={logits.shape}")

        if not RouterReplay.enable_logits_recording or RouterReplay.current_cache_action is None:
            if layer_idx == 0:
                logger.info(f"[record_logits] Skipping - enable_recording={RouterReplay.enable_logits_recording}, "
                            f"cache_action={RouterReplay.current_cache_action}")
            return

        # Apply save-time sampling if indices were set by record_global_token_ids.
        indices = RouterReplay.current_sample_indices
        if indices is not None:
            if indices.device != logits.device:
                indices = indices.to(logits.device)
            # indices were built from the full-sequence valid-token count. If this router
            # is running in a sequence-parallel shard the local `logits` has shape[0] =
            # full_count / tp_size, and indexing with full-range indices would trigger the
            # CUDA `vectorized_gather_kernel` OOB assert. Detect this by comparing shape[0]
            # to the stored full_token_count and gather across the TP group first.
            full_count = RouterReplay.current_full_token_count
            tp_size = mpu.get_tensor_model_parallel_world_size()
            if tp_size > 1 and full_count is not None and logits.shape[0] < full_count:
                logits = gather_from_sequence_parallel_region(
                    logits, tensor_parallel_output_grad=False
                )
                # SP pads to a multiple of tp_size; truncate to the real valid-token count
                # so the index range matches exactly.
                if logits.shape[0] > full_count:
                    logits = logits[:full_count]
            logits = logits[indices]

        # Move to CPU to avoid GPU memory pressure
        # Make a contiguous copy to ensure clean memory layout
        logits_cpu = logits.detach().cpu().contiguous()

        # Force synchronization if on CUDA to ensure data is fully copied to CPU
        # This allows GPU memory to be freed immediately
        # if logits.is_cuda:
        #     torch.cuda.synchronize()

        if RouterReplay.current_cache_action == RouterReplayCacheAction.COMPUTE_LOG_PROB:
            RouterReplay.logits_cache["compute_log_prob"].append((layer_idx, logits_cpu))
            if layer_idx == 0:
                logger.info(f"[record_logits] Recorded to compute_log_prob cache. Current size: {len(RouterReplay.logits_cache['compute_log_prob'])}")
        elif RouterReplay.current_cache_action == RouterReplayCacheAction.TRAINING:
            RouterReplay.logits_cache["training"].append((layer_idx, logits_cpu))
            if layer_idx == 0:
                logger.info(f"[record_logits] Recorded to training cache. Current size: {len(RouterReplay.logits_cache['training'])}")

    @staticmethod
    @no_grad()
    def record_predictive_bias(delta_logits: torch.Tensor, layer_idx: int):
        """
        Record predictive bias (scaled delta_logits) to cache.
        Only records when logits recording is enabled.

        Args:
            delta_logits: The bias predictor output after scaling
            layer_idx: The layer index
        """
        if not RouterReplay.enable_logits_recording or RouterReplay.current_cache_action is None:
            return

        # R2 saved bias predictor output with a singleton sequence dimension: [tokens, 1, experts].
        # R3 stats collection receives per-layer bias from SGLang as [tokens, experts].
        # Normalize the saved artifact to the R2 shape so downstream comparisons stay consistent.
        if delta_logits.ndim == 2:
            delta_logits = delta_logits.unsqueeze(1)

        token_ids = RouterReplay.current_predictive_bias_token_ids
        if token_ids is not None:
            if int(token_ids.numel()) != int(delta_logits.shape[0]):
                logger.warning(
                    "[record_predictive_bias] predictive_bias token id count mismatch: ids=%s bias_tokens=%s. "
                    "Skipping predictive_bias save to avoid position-corrupted artifacts.",
                    int(token_ids.numel()),
                    int(delta_logits.shape[0]),
                )
                return

            # R3_COLLECT_STATS receives old_bias compacted to rollout-captured token
            # positions.  Full-token sample indices would index the wrong rows here.
            # Keep only captured tokens that also have saved current logits, and save
            # their ids separately so readers can join by token id.
            sampled_set = RouterReplay.sampled_log_prob_token_ids
            if (
                sampled_set is not None
                and len(sampled_set) > 0
                and RouterReplay.logits_save_sample_rate is not None
                and RouterReplay.logits_save_sample_rate < 1.0
            ):
                mask_cpu = torch.tensor([int(tid) in sampled_set for tid in token_ids.tolist()], dtype=torch.bool)
                mask = mask_cpu.to(delta_logits.device)
                delta_logits = delta_logits[mask]
                token_ids = token_ids[mask_cpu]

            if not RouterReplay.current_predictive_bias_token_ids_recorded:
                RouterReplay.logits_cache["predictive_bias_token_ids"].append(token_ids.detach().cpu().contiguous())
                RouterReplay.current_predictive_bias_token_ids_recorded = True
        else:
            # Apply save-time sampling along token dim (dim 0) if indices were set.
            indices = RouterReplay.current_sample_indices
            if indices is not None:
                if indices.device != delta_logits.device:
                    indices = indices.to(delta_logits.device)
                # Same SP-gather fix as record_logits: indices are full-sequence but delta_logits
                # may be SP-sharded.
                full_count = RouterReplay.current_full_token_count
                tp_size = mpu.get_tensor_model_parallel_world_size()
                if tp_size > 1 and full_count is not None and delta_logits.shape[0] < full_count:
                    delta_logits = gather_from_sequence_parallel_region(
                        delta_logits, tensor_parallel_output_grad=False
                    )
                    if delta_logits.shape[0] > full_count:
                        delta_logits = delta_logits[:full_count]
                delta_logits = delta_logits[indices]

        # Move to CPU to avoid GPU memory pressure
        delta_logits_cpu = delta_logits.detach().cpu().contiguous()

        # Record to predictive_bias cache (only during compute_log_prob phase)
        if RouterReplay.current_cache_action == RouterReplayCacheAction.COMPUTE_LOG_PROB:
            RouterReplay.logits_cache["predictive_bias"].append((layer_idx, delta_logits_cpu))

    @staticmethod
    def get_debug_info():
        """Get debug information about current state."""
        return {
            "enable_logits_recording": RouterReplay.enable_logits_recording,
            "current_cache_action": RouterReplay.current_cache_action,
            "num_router_instances": len(RouterReplay.router_instances),
            "cache_sizes": {
                "compute_log_prob": len(RouterReplay.logits_cache.get("compute_log_prob", [])),
                "training": len(RouterReplay.logits_cache.get("training", [])),
                "router_weights": len(RouterReplay.logits_cache.get("router_weights", [])),
                "predictive_bias": len(RouterReplay.logits_cache.get("predictive_bias", [])),
                "predictive_bias_token_ids": len(RouterReplay.logits_cache.get("predictive_bias_token_ids", [])),
            }
        }

    """predictive routing replay management"""

    def set_predictive_data(self, inputs: torch.Tensor, logits: torch.Tensor, valid_mask=None, loss_scale: float = 1.0):
        """Set old inputs and logits for this layer.

        Args:
            inputs: Old router inputs
            logits: Old router logits
            valid_mask: Optional boolean mask of shape [total_tokens] indicating which tokens belong to valid samples
            loss_scale: Scalar multiplier applied to predictive loss for this batch/layer.
        """
        self.recorded_old_inputs = inputs.detach() if inputs is not None else None
        self.recorded_old_logits = logits.detach() if logits is not None else None
        self.predictive_valid_mask = valid_mask.detach() if valid_mask is not None else None
        self.predictive_loss_scale = float(loss_scale)
        # For now this is the same as record_predictive_data, as we don't have backward yet.

    def get_predictive_data(self):
        """Get old inputs and logits for this layer."""
        return (
            self.recorded_old_inputs,
            self.recorded_old_logits,
            self.predictive_valid_mask,
            getattr(self, "predictive_loss_scale", 1.0),
        )

    def record_predictive_data(self, inputs: torch.Tensor, logits: torch.Tensor):
        """Record inputs and logits for this layer (like record_indices)."""
        # Keep on GPU for merge function, which will handle CPU transfer uniformly
        # Use .detach() to break gradient graph and reduce memory footprint

        # Detach and create contiguous copies to allow original tensors to be freed
        self.recorded_old_inputs = inputs.squeeze().detach().contiguous()
        self.recorded_old_logits = logits.squeeze().detach().contiguous()

    def clear_predictive_data(self):
        """Clear predictive data for this layer."""
        self.recorded_old_inputs = None
        self.recorded_old_logits = None
        self.predictive_valid_mask = None
        self.predictive_loss_scale = 1.0

    @staticmethod
    def clear_global_predictive_data():
        """Clear predictive data for all router instances."""
        for router in RouterReplay.router_instances:
            router.clear_predictive_data()

    def set_predictive_bias(self, bias: torch.Tensor):
        """Set old_bias (SGLang delta_logits) for this layer."""
        self.recorded_old_bias = bias.detach() if bias is not None else None

    def clear_predictive_bias(self):
        """Clear old_bias for this layer."""
        self.recorded_old_bias = None

    @staticmethod
    def clear_global_predictive_bias():
        """Clear old_bias for all router instances."""
        for router in RouterReplay.router_instances:
            router.clear_predictive_bias()
        RouterReplay.clear_predictive_bias_token_ids()

    def set_predictive_action(self, action: RouterPredictiveAction):
        """Set the predictive action for this layer."""
        self.predictive_action = action

    def clear_predictive_action(self):
        """Clear the predictive action for this layer."""
        self.predictive_action = None

    @staticmethod
    def set_global_predictive_action(action: RouterPredictiveAction):
        """Set the predictive action for all router instances."""
        for router in RouterReplay.router_instances:
            router.set_predictive_action(action)

    @staticmethod
    def clear_global_predictive_action():
        """Clear the predictive action for all router instances."""
        for router in RouterReplay.router_instances:
            router.clear_predictive_action()

    """(predictive) routing replay metrics logging"""

    @staticmethod
    def record_predictive_loss(layer_idx: int, loss_value: float):
        """Record predictive loss for wandb logging."""
        RouterReplay.predictive_loss_tracker.append((layer_idx, loss_value))

    @staticmethod
    def record_predictive_bias_ratio(layer_idx: int, ratio_value: float):
        """Record predictive bias-to-logits ratio for wandb logging."""
        RouterReplay.predictive_bias_ratio_tracker.append((layer_idx, ratio_value))

    @staticmethod
    def record_predictive_topk_accuracy(layer_idx: int, accuracy_value: float):
        """Record predictive top-k prediction accuracy for wandb logging."""
        RouterReplay.predictive_topk_accuracy_tracker.append((layer_idx, accuracy_value))

    @staticmethod
    def record_replay_topk_accuracy(layer_idx: int, accuracy_value: float):
        """Record replay top-k agreement for compatibility with the legacy replay path."""
        RouterReplay.replay_topk_accuracy_tracker.append((layer_idx, accuracy_value))

    @staticmethod
    def get_and_clear_predictive_metrics():
        """Get aggregated predictive metrics and clear trackers."""
        metrics = {}

        if RouterReplay.replay_topk_accuracy_tracker:
            avg_accuracy = sum(acc for _, acc in RouterReplay.replay_topk_accuracy_tracker) / len(RouterReplay.replay_topk_accuracy_tracker)
            metrics['replay_topk_accuracy'] = avg_accuracy
            RouterReplay.replay_topk_accuracy_tracker.clear()

        if RouterReplay.predictive_loss_tracker:
            avg_loss = sum(loss for _, loss in RouterReplay.predictive_loss_tracker) / len(RouterReplay.predictive_loss_tracker)
            metrics['predictive_loss'] = avg_loss
            RouterReplay.predictive_loss_tracker.clear()

        if RouterReplay.predictive_bias_ratio_tracker:
            avg_ratio = sum(ratio for _, ratio in RouterReplay.predictive_bias_ratio_tracker) / len(RouterReplay.predictive_bias_ratio_tracker)
            metrics['predictive_bias_to_logits_ratio'] = avg_ratio
            RouterReplay.predictive_bias_ratio_tracker.clear()

        if RouterReplay.predictive_topk_accuracy_tracker:
            avg_accuracy = sum(acc for _, acc in RouterReplay.predictive_topk_accuracy_tracker) / len(RouterReplay.predictive_topk_accuracy_tracker)
            metrics['predictive_topk_accuracy'] = avg_accuracy
            RouterReplay.predictive_topk_accuracy_tracker.clear()

        return metrics


@torch.no_grad()
def calculate_topk_accuracy(
    topk: int,
    logits1: torch.Tensor=None,
    logits2: torch.Tensor=None,
    topk_indices1: torch.Tensor=None,
    topk_indices2: torch.Tensor=None,
):
    if topk_indices1 is None:
        _, topk_indices1 = torch.topk(logits1, k=topk, dim=-1)
    if topk_indices2 is None:
        _, topk_indices2 = torch.topk(logits2, k=topk, dim=-1)
    topk_indices1_expanded = topk_indices1.unsqueeze(-1)  # [tokens, topk, 1]
    topk_indices2_expanded = topk_indices2.unsqueeze(-2)  # [tokens, 1, topk]
    matches = (topk_indices1_expanded == topk_indices2_expanded).any(dim=-1)  # [tokens, topk]
    accuracy = matches.float().mean().item()
    return accuracy


def _patched_topk_routing_with_score_function(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool,
    num_groups: int,
    group_topk: int,
    score_function: str,
    expert_bias: torch.Tensor,
    fused: bool,
    router_replay: RouterReplay,
    scaling_factor: float,
    layer_number: int = None,  # Added: for logits recording without router_replay
):
    """
    Patched version of topk_routing_with_score_function that supports router replay.
    """
    num_tokens, num_experts = logits.shape

    def _compute_topk(scores, topk, num_groups=None, group_topk=None):
        if group_topk:
            return group_limited_topk(
                scores=scores,
                topk=topk,
                num_tokens=num_tokens,
                num_experts=num_experts,
                num_groups=num_groups,
                group_topk=group_topk,
            )
        else:
            return torch.topk(scores, k=topk, dim=1)

    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        # Get layer_idx from router_replay or use layer_number directly
        # Determine layer_idx for logits recording
        if router_replay is not None:
            layer_idx = router_replay.layer_idx  # 0-indexed (from list position)
        elif layer_number is not None:
            layer_idx = layer_number - 1  # Convert from 1-indexed to 0-indexed
        else:
            layer_idx = 0  # Fallback

        routing_action = router_replay.router_replay_action if router_replay is not None else None

        # Record logits regardless of routing_action (if cache_action is set)
        # This allows recording even when router_replay is disabled
        if RouterReplay.enable_logits_recording and RouterReplay.current_cache_action is not None:
            RouterReplay.record_logits(scores, layer_idx)

        # Debug: log first call
        if layer_idx == 0:
            logger.info(f"[compute_topk] Layer 0: routing_action={routing_action}, "
                        f"router_replay={router_replay is not None}, layer_number={layer_number}, "
                        f"will_record_logits={RouterReplay.enable_logits_recording}")

        if routing_action is None:
            # No router replay, just compute topk normally
            return _compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)

        if routing_action == RouterReplayAction.RECORD:
            # Compute topk normally and record the indices
            probs, top_indices = _compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
            if router_replay is not None:
                router_replay.record_indices(top_indices)
            return probs, top_indices

        elif routing_action == RouterReplayAction.REPLAY_FORWARD:
            if router_replay is None or router_replay.target_topk_idx is None:
                # Fallback if replay data is not available
                probs, top_indices = _compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
                return probs, top_indices
            else:
                # Use the provided indices for replay
                top_indices = router_replay.target_topk_idx
                # Ensure indices are on the correct device
                top_indices = top_indices.to(scores.device)
                # Gather the scores for the replayed indices to get the probabilities
                probs = scores.gather(1, top_indices)
                # Calculate current top-k indices for accuracy calculation
                _, current_top_indices = _compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
                accuracy = calculate_topk_accuracy(topk=topk, logits1=top_indices, logits2=current_top_indices)
                RouterReplay.record_replay_topk_accuracy(layer_idx, accuracy)
            return probs, top_indices
        elif routing_action == RouterReplayAction.REPLAY_BACKWARD:
            if router_replay is None or not router_replay.replay_backward_list:
                # Fallback if replay data is not available
                return _compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)

            # Use the last recorded indices for backward replay
            top_indices = router_replay.replay_backward_list.pop(0)
            # Ensure indices are on the correct device
            top_indices = top_indices.to(scores.device)
            # Gather the scores for the replayed indices to get the probabilities
            probs = scores.gather(1, top_indices)
            return probs, top_indices
        else:  # Unknown action, fallback
            return _compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)

    if score_function == "softmax":
        if use_pre_softmax:
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            probs, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        else:
            scores, top_indices = compute_topk(logits, topk, num_groups, group_topk)
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(logits)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits.float()).type_as(logits)
        if expert_bias is not None:
            scores_for_routing = scores + expert_bias
            _, top_indices = compute_topk(scores_for_routing, topk, num_groups, group_topk)
            scores = torch.gather(scores, dim=1, index=top_indices).type_as(logits)
        else:
            scores, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    if scaling_factor:
        probs = probs * scaling_factor

    if torch.are_deterministic_algorithms_enabled():
        # build [num_tokens, num_experts] from [num_tokens, topk]
        routing_probs = torch.zeros_like(logits)
        rows = torch.arange(num_tokens, device=logits.device).unsqueeze(1)
        routing_probs.index_put_((rows, top_indices), probs, accumulate=False)

        routing_map = torch.zeros_like(logits, dtype=logits.dtype)
        routing_map.index_put_((rows, top_indices), torch.ones_like(probs, dtype=routing_map.dtype), accumulate=False)
        routing_map = routing_map.bool()
    else:
        # TODO Try using element-wise operations instead of scatter?
        routing_probs = torch.zeros_like(logits).scatter(1, top_indices, probs)
        routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()

    return routing_probs, routing_map


def _get_aux_loss_coeff(_self, aux_loss_type: str) -> float:
    """Return the aux loss coeff for the given auxiliary loss type."""
    if isinstance(_self.routing_type, str):
        if _self.routing_type == aux_loss_type:
            return _self.config.moe_aux_loss_coeff
    if isinstance(_self.routing_type, list):
        try:
            idx = _self.routing_type.index(aux_loss_type)
            return _self.config.moe_aux_loss_coeff[idx]
        except (ValueError, IndexError):
            return 0.0
    return 0.0


def _is_aux_loss_enabled(_self) -> bool:
    """Check if the auxiliary loss is enabled."""
    for aux_loss_type in ["aux_loss", "seq_aux_loss", "global_aux_loss"]:
        if _get_aux_loss_coeff(_self, aux_loss_type) > 0:
            return True
    return False


def patched_routing(self, logits: torch.Tensor, *args, **kwargs):
    """Top-k routing function

    Args:
        logits (torch.Tensor): Logits tensor after gating.

    Returns:
        probs (torch.Tensor): The probabilities of token to experts assignment.
        routing_map (torch.Tensor): The mapping of token to experts assignment,
            with shape [num_tokens, num_experts].
    """
    seq_length, bsz = logits.shape[:2]
    logits = logits.view(-1, self.config.num_moe_experts)
    # Note: Router weights recording moved to post-forward hooks in megatron_workers/actor
    # to ensure we capture updated weights after optimizer.step

    # Apply Z-Loss
    logits = self.apply_z_loss(logits)

    moe_router_fusion = getattr(self.config, "moe_router_fusion", False)

    # Calculate probs and routing_map for token dispatching
    if self.routing_type == "sinkhorn":
        probs, routing_map = self.sinkhorn_load_balancing(logits)
    else:
        probs, routing_map = _patched_topk_routing_with_score_function(
            logits=logits,
            topk=self.topk,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            num_groups=self.config.moe_router_num_groups,
            group_topk=self.config.moe_router_group_topk,
            scaling_factor=self.config.moe_router_topk_scaling_factor,
            score_function=self.score_function,
            expert_bias=self.expert_bias,
            fused=moe_router_fusion,
            router_replay=self.router_replay,
            layer_number=self.layer_number,  # Pass layer_number for logits recording
        )

    # Apply token dropping to probs and routing_map.
    if self.config.moe_expert_capacity_factor is not None:
        probs, routing_map = apply_router_token_dropping(
            probs,
            routing_map,
            router_topk=self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            drop_policy=self.config.moe_token_drop_policy,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
        )

    if not hasattr(self, "is_aux_loss_enabled"):
        self.is_aux_loss_enabled = types.MethodType(_is_aux_loss_enabled, self)

    # Apply each aux loss type and attach aux loss autograd function to probs
    if self.training and torch.is_grad_enabled() and self.is_aux_loss_enabled():
        # Calculate scores and routing_map for aux loss
        routing_map_for_aux_loss, scores_for_aux_loss = compute_routing_scores_for_aux_loss(
            logits, self.topk, self.score_function, fused=moe_router_fusion
        )
        probs = self._apply_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss)
        probs = self._apply_seq_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss, seq_length, bsz)
        probs = self._apply_global_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss)

    # Update expert bias and tokens_per_expert
    # Prevent extra local tokens accumulation on evaluation or activation recomputation
    if self.enable_expert_bias and torch.is_grad_enabled():
        with torch.no_grad():
            self.local_tokens_per_expert += routing_map.sum(dim=0)

    return probs, routing_map


"""predictive routing replay"""


class MoEPredictiveLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that triggers the backward pass and scales the grad for predictive loss."""

    main_loss_backward_scale: torch.Tensor = None

    @staticmethod
    def forward(ctx, output: torch.Tensor, predictive: torch.Tensor):
        """Preserve the predictive_loss by storing it in the context to avoid garbage collection.

        Args:
            output (torch.Tensor): The output tensor.
            predictive_loss (torch.Tensor): The predictive loss tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        ctx.save_for_backward(predictive)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for predictive loss..

        Args:
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled predictive loss
                                               gradient.
        """
        (predictive_loss,) = ctx.saved_tensors
        if MoEPredictiveLossAutoScaler.main_loss_backward_scale is None:
            MoEPredictiveLossAutoScaler.main_loss_backward_scale = torch.tensor(
                1.0, device=predictive_loss.device
            )
        predictive_loss_backward_scale = MoEPredictiveLossAutoScaler.main_loss_backward_scale
        scaled_predictive_loss_grad = torch.ones_like(predictive_loss) * predictive_loss_backward_scale
        return grad_output, scaled_predictive_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """set the scale of the aux loss.

        Args:
            scale (torch.Tensor): The scale value to set. Please ensure that the scale passed in
                                  matches the scale of the main_loss.
        """
        if MoEPredictiveLossAutoScaler.main_loss_backward_scale is None:
            MoEPredictiveLossAutoScaler.main_loss_backward_scale = scale
        else:
            MoEPredictiveLossAutoScaler.main_loss_backward_scale.copy_(scale)


# def apply_predictive_loss(self, probs: torch.Tensor, predictive_loss: torch.Tensor):
#     # Attach predictive loss for backprop
#     return MoEPredictiveLossAutoScaler.apply(probs, predictive_loss)


def patched_forward(self, input: torch.Tensor):
    """
    Forward pass of the router.

    Args:
        input (torch.Tensor): Input tensor.
    """
    self._maintain_float32_expert_bias()

    # Apply input jitter
    input = self.apply_input_jitter(input)
    logits = self.gating(input)

    if self.config.moe_router_force_load_balancing:
        # Apply force load balancing with random logits for benchmark
        logits = apply_random_logits(logits)

    # Router bias predictor
    # Managed by RouterReplay class for proper phase control
    if self.config.enable_router_bias_predictor:
        # logger.info(f"[Predictive Routing Replay] Layer {self.layer_number}: {get_system_memory_info()}")
        assert self.router_replay is not None and self.bias_predictor is not None

        predictive_routing_action = self.router_replay.predictive_action if self.router_replay is not None else None
        router_replay_action = self.router_replay.router_replay_action if self.router_replay is not None else None

        predictive_loss = None

        if router_replay_action == RouterReplayAction.REPLAY_BACKWARD:
            probs, routing_map = self.routing(logits)
        else:
            if predictive_routing_action is None:
                if self.layer_number == 1 and self.router_replay and self.router_replay.layer_idx == 0:
                    logger.info("[Predictive Routing Replay] No predictive action set. Using vanilla routing.")
                probs, routing_map = self.routing(logits)

            elif predictive_routing_action == RouterPredictiveAction.RECORD: # PR2 log_prob/record phase
                with torch.no_grad():
                    # Log_prob phase: record inputs/logits and predictor stats only.
                    # Do not apply the predictor to routing here: old_log_probs must match
                    # the rollout policy's uncorrected router decisions.
                    self.router_replay.record_predictive_data(input, logits)

                    delta_logits = self.bias_predictor(input)

                    # Track bias ratio: |delta_logits|_mean / |logits|_mean
                    layer_idx = self.router_replay.layer_idx if self.router_replay else 0
                    bias_ratio = (torch.abs(delta_logits).mean() / (torch.abs(logits).mean() + 1e-10)).item()
                    RouterReplay.record_predictive_bias_ratio(layer_idx, bias_ratio)

                    # Record predictive bias to logits cache if saving is enabled
                    if RouterReplay.enable_logits_recording:
                        RouterReplay.record_predictive_bias(delta_logits, layer_idx)

                probs, routing_map = self.routing(logits)

            elif predictive_routing_action == RouterPredictiveAction.SKIP_PREDICTIVE:
                # PR2/PR3 Training phase ministep=0
                # Skip predictive loss, use normal routing
                if self.layer_number == 1 and self.router_replay and self.router_replay.layer_idx == 0:
                     logger.info(f"[Predictive Routing Replay] Action is SKIP_PREDICTIVE. Skipping loss computation.")
                probs, routing_map = self.routing(logits)

            elif predictive_routing_action == RouterPredictiveAction.R3_COLLECT_STATS:
                # R3 compute_log_prob phase: use SGLang-provided delta_logits (old_bias) for statistics
                # No gradient needed; purely for logging bias_ratio and recording predictive_bias
                with torch.no_grad():
                    layer_idx = self.router_replay.layer_idx if self.router_replay else 0
                    delta_logits = self.router_replay.recorded_old_bias if self.router_replay else None
                    if delta_logits is not None:
                        delta_logits = delta_logits.to(logits.device)
                        # Track bias ratio: |delta_logits|_mean / |logits|_mean
                        bias_ratio = (torch.abs(delta_logits).mean() / (torch.abs(logits).mean() + 1e-10)).item()
                        RouterReplay.record_predictive_bias_ratio(layer_idx, bias_ratio)

                        # Record predictive bias to logits cache if saving is enabled
                        if RouterReplay.enable_logits_recording:
                            RouterReplay.record_predictive_bias(delta_logits, layer_idx)
                    elif self.layer_number == 1 and layer_idx == 0:
                        logger.debug("[Predictive Routing Replay] R3_COLLECT_STATS: no old_bias available for this sample, skipping stats.")
                probs, routing_map = self.routing(logits)

            elif predictive_routing_action == RouterPredictiveAction.COMPUTE_PREDICTIVE_LOSS:
                # PR2/PR3 Training phase ministep>=1: compute predictive loss
                if self.layer_number == 1 and self.router_replay and self.router_replay.layer_idx == 0:
                     logger.info(f"[Predictive Routing Replay] Action is COMPUTE_PREDICTIVE_LOSS. Computing loss...")

                # gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3)
                # logger.info(f"[Predictive Routing Replay] [Memory] (layer {self.layer_number}) Total GPU memory allocated before predictive loss computation: {gpu_mem:.2f} GB, {get_system_memory_info()}")
                old_inputs, old_logits, valid_mask, predictive_loss_scale = self.router_replay.get_predictive_data()

                # CRITICAL FIX: Check if we have valid data by checking tensor size (not None)
                # Empty tensors (shape [0, ...]) are created for processes without valid samples
                has_valid_data = (old_inputs is not None and old_logits is not None and old_inputs.shape[0] > 0 and old_logits.shape[0] > 0)

                if has_valid_data:
                    old_inputs = old_inputs.to(input.device)
                    old_logits = old_logits.to(logits.device)
                    valid_mask = valid_mask.to(input.device)
                    # gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3)
                    # logger.info(f"[Predictive Routing Replay] [Memory] (layer {self.layer_number}) Total GPU memory allocated after loading predictive data: {gpu_mem:.2f} GB, {get_system_memory_info()}")

                    # When sequence_parallel is enabled the router receives only the local
                    # SP shard of the current tokens (shape [total_tokens/tp, hidden]).
                    # The valid_mask and old_inputs/old_logits are stored in full (not split),
                    # so we must gather the current input/logits back to the full sequence
                    # before applying the mask.  Each rank computes the same full loss;
                    # this is redundant but correct, and avoids complex shard-level alignment.
                    current_input_full = input
                    current_logits_full = logits
                    if getattr(self.config, "sequence_parallel", False):
                        tp_size = mpu.get_tensor_model_parallel_world_size()
                        if tp_size > 1 and input.shape[0] < valid_mask.shape[0]:
                            # Input is SP-split (local shard is smaller than the full sequence).
                            # Gather all shards to recover the full-sequence tensor so that
                            # valid_mask (which is full-sequence sized) can be applied correctly.
                            current_input_full = gather_from_sequence_parallel_region(
                                input, tensor_parallel_output_grad=True
                            )
                            current_logits_full = gather_from_sequence_parallel_region(
                                logits, tensor_parallel_output_grad=False
                            )
                            # Megatron SP pads the full sequence up to a multiple of tp_size
                            # before splitting; the trailing pad tokens are not in valid_mask,
                            # so truncate the gathered tensors back to the unpadded length.
                            if current_input_full.shape[0] > valid_mask.shape[0]:
                                current_input_full = current_input_full[: valid_mask.shape[0]]
                                current_logits_full = current_logits_full[: valid_mask.shape[0]]

                    # Debug asserts: check shape matching
                    assert old_inputs.shape[-1] == current_input_full.shape[-1], f"hidden_size mismatch: old={old_inputs.shape[-1]}, current={current_input_full.shape[-1]}"
                    assert old_logits.shape[-1] == current_logits_full.shape[-1], f"num_experts mismatch: old={old_logits.shape[-1]}, current={current_logits_full.shape[-1]}"

                    # Apply token-level mask if provided (for downsampled data)
                    # valid_mask is at token level, matching the unpacked (full) input/logits shape
                    current_input = current_input_full  # TODO: not used, preserved for future enhancement
                    current_logits = current_logits_full
                    if valid_mask is not None:
                        # Filter current input and logits at token level to match old_inputs/old_logits
                        current_input = current_input_full[valid_mask]
                        current_logits = current_logits_full[valid_mask]
                        # logger.info(f"[Predictive Routing Replay] Applied token-level mask: {valid_mask.sum().item()}/{valid_mask.size(0)} valid tokens")
                        assert current_input.shape[0] == old_inputs.shape[0], f"Token count mismatch after masking: old={old_inputs.shape[0]}, current={current_input.shape[0]}"
                        assert current_logits.shape[0] == old_logits.shape[0], f"Token count mismatch after masking: old={old_logits.shape[0]}, current={current_logits.shape[0]}"

                    # Compute delta_logits and logits_diff
                    # Use old_inputs with current weights to get delta_logits
                    delta_logits = self.bias_predictor(old_inputs)
                    logits_diff = current_logits - old_logits

                    # Compute loss between delta_logits and logits_diff
                    if self.config.bias_predictor_loss_type == "l2":
                        # L2 loss on raw logits difference
                        predictive_loss = torch.nn.functional.mse_loss(delta_logits, logits_diff.detach(), reduction='mean')

                    elif self.config.bias_predictor_loss_type == "kl":
                        # KL divergence on logits_diff vs delta_logits distributions
                        ##############
                        # DEBUG
                        # import datetime
                        # SAVE_DIR = "/root/verl/debug/"
                        # os.makedirs(SAVE_DIR, exist_ok=True)
                        # TIME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        # torch.save({
                        #     'inputs': input.cpu(),
                        #     'logits': logits.cpu(),
                        #     'old_inputs': old_inputs.cpu(),
                        #     'old_logits': old_logits.cpu(),
                        #     'valid_mask': valid_mask.cpu() if valid_mask is not None else None,
                        #     'current_input': current_input.cpu(),
                        #     'current_logits': current_logits.cpu(),
                        #     'delta_logits': delta_logits.cpu(),
                        #     'logits_diff': logits_diff.cpu(),
                        # }, SAVE_DIR + f"predictive_routing_logits_{self.layer_number}_{TIME}.pt")
                        # logger.info(f"[Predictive Routing Replay] [Debug] Saved delta_logits and logits_diff for debugging at layer {self.layer_number} to {SAVE_DIR}predictive_routing_logits_{self.layer_number}_{TIME}.pt")
                        ##############
                        pred_log_probs = torch.log_softmax(delta_logits, dim=-1)
                        target_probs = torch.softmax(logits_diff, dim=-1)
                        predictive_loss = torch.nn.functional.kl_div(
                            pred_log_probs,
                            target_probs.detach(),
                            reduction='batchmean'
                        )

                    elif self.config.bias_predictor_loss_type == "kl-post":
                        # KL divergence on corrected vs uncorrected routing distributions
                        # This measures the KL between final routing decisions
                        pred_log_probs = torch.log_softmax(old_logits + delta_logits, dim=-1)
                        target_probs = torch.softmax(current_logits, dim=-1)
                        predictive_loss = torch.nn.functional.kl_div(
                            pred_log_probs,
                            target_probs.detach(),
                            reduction='batchmean'
                        )

                    else:
                        raise ValueError(f"Invalid loss type: {self.config.bias_predictor_loss_type}")

                    predictive_loss = predictive_loss * predictive_loss_scale

                    # Record predictive loss
                    layer_idx = self.router_replay.layer_idx if self.router_replay else 0
                    RouterReplay.record_predictive_loss(layer_idx, predictive_loss.item())

                    # Record top-k prediction accuracy
                    accuracy = calculate_topk_accuracy(topk=self.topk, logits1=old_logits + delta_logits, logits2=current_logits)
                    RouterReplay.record_predictive_topk_accuracy(layer_idx, accuracy)

                    if (
                        _r3_predictive_diag_enabled()
                        and self.config.enable_routing_replay
                        and self.layer_number in {1, self.config.num_layers}
                    ):
                        with torch.no_grad():
                            current_logits_on_old_inputs = self.gating(old_inputs)
                            old_topk_to_current = calculate_topk_accuracy(
                                topk=self.topk,
                                logits1=old_logits,
                                logits2=current_logits,
                            )
                            old_topk_to_current_on_old = calculate_topk_accuracy(
                                topk=self.topk,
                                logits1=old_logits,
                                logits2=current_logits_on_old_inputs,
                            )
                            pred_topk_to_current_on_old = calculate_topk_accuracy(
                                topk=self.topk,
                                logits1=old_logits + delta_logits,
                                logits2=current_logits_on_old_inputs,
                            )
                            current_on_old_to_current = calculate_topk_accuracy(
                                topk=self.topk,
                                logits1=current_logits_on_old_inputs,
                                logits2=current_logits,
                            )
                            logger.warning(
                                "[R3PredictiveDiag] layer=%s layer_idx=%s old_tokens=%s "
                                "acc_pred_vs_current=%.6f acc_old_vs_current=%.6f "
                                "acc_old_vs_current_on_old_inputs=%.6f "
                                "acc_pred_vs_current_on_old_inputs=%.6f "
                                "acc_current_on_old_inputs_vs_current=%.6f "
                                "input_rel_l2=%.6e logits_rel_l2_current=%.6e "
                                "logits_rel_l2_current_on_old_inputs=%.6e "
                                "delta_to_old_ratio=%.6e valid_mask_sum=%s predictive_loss_scale=%.6f",
                                self.layer_number,
                                layer_idx,
                                int(old_logits.shape[0]),
                                accuracy,
                                old_topk_to_current,
                                old_topk_to_current_on_old,
                                pred_topk_to_current_on_old,
                                current_on_old_to_current,
                                _relative_l2(current_input, old_inputs),
                                _relative_l2(current_logits, old_logits),
                                _relative_l2(current_logits_on_old_inputs, old_logits),
                                (torch.abs(delta_logits).mean() / (torch.abs(old_logits).mean() + 1e-10)).item(),
                                int(valid_mask.sum().item()) if valid_mask is not None else None,
                                predictive_loss_scale,
                            )

                    # gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3)
                    # logger.info(f"[Predictive Routing Replay] [Memory] (layer {self.layer_number}) Total GPU memory allocated after predictive loss computation: {gpu_mem:.2f} GB, {get_system_memory_info()}")
                else:
                    # Processes without local predictive samples still need to traverse the
                    # bias_predictor graph. A bare `param * 0` dummy loss skips that graph
                    # entirely, which can desynchronize long-running router/update regions at
                    # scale when other ranks execute a real predictor forward/backward.
                    #
                    # However, re-running the predictor on the full local token set here makes
                    # "all_none" ranks vastly slower than ranks that only replay the downsampled
                    # predictive tokens, which can trip NCCL watchdog timeouts at scale. Keep the
                    # graph alive with a bounded detached slice so every rank still exercises a
                    # similar predictor graph without replaying arbitrarily many tokens.
                    sync_tokens = min(int(input.shape[0]), _predictive_sync_tokens())
                    synthetic_inputs = input.detach()[:sync_tokens]
                    if synthetic_inputs.numel() == 0:
                        synthetic_shape = (1, *input.shape[1:])
                        synthetic_inputs = input.detach().new_zeros(synthetic_shape)
                    synthetic_delta_logits = self.bias_predictor(synthetic_inputs)
                    predictive_loss = (synthetic_delta_logits * 0.0).sum()

                    if self.layer_number == 1:
                        logger.warning(
                            "[Predictive Routing Replay] No valid predictive data, "
                            "creating synthetic zero-loss through bias_predictor for backward sync"
                        )

                probs, routing_map = self.routing(logits)

            else:
                raise ValueError(f"Invalid predictive routing action: {predictive_routing_action}")

        if predictive_loss is not None:
            # probs = self.apply_predictive_loss(probs, predictive_loss)
            # CRITICAL: All processes execute backward, including those with dummy loss
            # This ensures synchronization across all processes
            predictive_loss.backward()

            # [BPDiag] Immediately after inner backward: check that gradient reached main_grad.
            # DDP hook fires DURING backward and moves param.grad -> param.main_grad, then sets
            # param.grad=None.  So if main_grad is still zero here the hook never fired for bp.
            import os as _os
            if int(_os.environ.get('VERL_DEBUG_PREDICTOR_SYNC', '0')) >= 1:
                _loss_val = predictive_loss.item()
                # Print for every layer on every rank — but only for non-trivial cases to limit noise.
                # "all_none" path produces loss=0 and zero gradient intentionally; those are filtered.
                # For the "has_valid_data" path we always want to see whether main_grad was populated.
                _layer_idx = self.router_replay.layer_idx if self.router_replay else -1
                import torch.distributed as _dist
                _rank = _dist.get_rank() if _dist.is_initialized() else 0
                for _bp_name, _bp_param in self.bias_predictor.named_parameters():
                    _grad = _bp_param.grad
                    _main_grad = getattr(_bp_param, 'main_grad', None)
                    _grad_norm = _grad.detach().float().norm().item() if _grad is not None else float('nan')
                    _main_grad_norm = _main_grad.detach().float().norm().item() if _main_grad is not None else float('nan')
                    _has_main_grad_attr = _main_grad is not None
                    print(
                        f"[BPDiag][inner_bwd] rank={_rank} layer={_layer_idx} bp_param={_bp_name} "
                        f"shape={tuple(_bp_param.shape)} "
                        f"grad={'None' if _grad is None else f'{_grad_norm:.6e}'} "
                        f"main_grad={'None(no_attr)' if not _has_main_grad_attr else f'{_main_grad_norm:.6e}'} "
                        f"loss={_loss_val:.6e}",
                        flush=True,
                    )

            self.router_replay.clear_predictive_data()
            del old_inputs, old_logits, valid_mask

    else:
        # Vanilla, R2, R3 inference & training phases: Standard routing without bias predictor
        probs, routing_map = self.routing(logits)

    return probs, routing_map


def apply_router_replay_patch():
    """
    Applies the monkey patch for MoE Router Replay functionality.
    This patch dynamically adds router replay / predictive attributes to TransformerConfig
    and modifies the TopKRouter to support recording and replaying of routing decisions.
    """
    logger.info("Applying Router Replay Patch...")
    RouterReplay.router_instances.clear()
    try:
        sig = inspect.signature(TransformerConfig.__init__)
        native_params = sig.parameters
        params = list(sig.parameters.values())
    except Exception:
        sig = None
        native_params = {}
        params = []

    ext_attrs = {
        "enable_routing_replay": False,
        "enable_router_bias_predictor": False,
        "bias_predictor_loss_type": "kl",
        "bias_predictor_lr_mult": 1000.0,
    }

    for attr, default in ext_attrs.items():
        if attr not in native_params and sig is not None:
            new_param = inspect.Parameter(attr, inspect.Parameter.KEYWORD_ONLY, default=default)
            if params and params[-1].kind == inspect.Parameter.VAR_KEYWORD:
                params.insert(-1, new_param)
            else:
                params.append(new_param)

    if sig is not None:
        try:
            TransformerConfig.__init__.__signature__ = sig.replace(parameters=params)
        except Exception as e:
            logger.warning("Failed to update TransformerConfig signature metadata: %s", e)

    if not hasattr(TransformerConfig, "_verl_router_patched"):
        TransformerConfig.enable_routing_replay = ext_attrs["enable_routing_replay"]
        TransformerConfig.enable_router_bias_predictor = ext_attrs["enable_router_bias_predictor"]
        TransformerConfig.bias_predictor_loss_type = ext_attrs["bias_predictor_loss_type"]
        TransformerConfig.bias_predictor_lr_mult = ext_attrs["bias_predictor_lr_mult"]

        original_tf_config_init = TransformerConfig.__init__

        @wraps(original_tf_config_init)
        def patched_tf_config_init(self, *args, **kwargs):
            values = {}
            for attr, default in ext_attrs.items():
                if attr in native_params:
                    values[attr] = kwargs.get(attr, default)
                else:
                    values[attr] = kwargs.pop(attr, default)

            original_tf_config_init(self, *args, **kwargs)

            for attr, value in values.items():
                setattr(self, attr, value)

        TransformerConfig.__init__ = patched_tf_config_init
        TransformerConfig._verl_router_patched = True

    if hasattr(TopKRouter, "_router_replay_patched"):
        return

    original_init = TopKRouter.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.router_replay = None
        if getattr(self.config, "enable_routing_replay", False):
            self.router_replay = RouterReplay()

    if MoEAlltoAllTokenDispatcher is not None and not hasattr(MoEAlltoAllTokenDispatcher, "_preprocess_patched"):
        original_preprocess = MoEAlltoAllTokenDispatcher.preprocess

        def patched_preprocess(self, routing_map):
            result = original_preprocess(self, routing_map)
            if (
                getattr(self.config, "enable_routing_replay", False)
                and not self.drop_and_pad
                and self.config.moe_expert_capacity_factor is None
                and not (
                    getattr(self.config, "moe_router_padding_for_quantization", None)
                    or getattr(self.config, "moe_router_padding_for_fp8", None)
                )
            ):
                self.num_out_tokens = int(routing_map.sum().item())
            return result

        MoEAlltoAllTokenDispatcher.preprocess = patched_preprocess
        MoEAlltoAllTokenDispatcher._preprocess_patched = True

    TopKRouter.__init__ = patched_init
    TopKRouter.routing = patched_routing
    TopKRouter.forward = patched_forward
    TopKRouter._router_replay_patched = True

    logger.info(
        "Router Replay Patch applied successfully. enable_routing_replay=%s, enable_router_bias_predictor=%s",
        getattr(TransformerConfig, "enable_routing_replay", False),
        getattr(TransformerConfig, "enable_router_bias_predictor", False),
    )
