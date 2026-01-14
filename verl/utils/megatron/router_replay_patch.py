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
import os
import warnings
import logging
from enum import Enum

import torch
from torch import no_grad

from verl.utils.memory_utils import get_system_memory_info

try:
    import psutil
except ImportError:
    psutil = None

try:
    from megatron.core.transformer.moe.moe_utils import (
        apply_router_token_dropping,
        compute_routing_scores_for_aux_loss,
        group_limited_topk,
        apply_random_logits,
        MoEAuxLossAutoScaler,
    )
except ImportError:
    warnings.warn("NPU not support router replay for now.", stacklevel=2)
    pass
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
    RECORD = "record"  # log_prob阶段：记录inputs和logits
    SKIP_PREDICTIVE = "skip_predictive"  # training ministep==0：跳过loss计算
    COMPUTE_PREDICTIVE_LOSS = "compute_predictive_loss"  # training ministep>=1：计算loss


class RouterReplay:
    """
    A class to manage the recording and replaying of MoE routing decisions.
    It holds all router instances and provides static methods to globally
    control recording and replaying.
    """

    # Static variable to hold all router instances, one per MoE layer.
    router_instances = []

    # Global logits cache for recording
    # Structure: {"compute_log_prob": [], "training": [], "router_weights": {}, "global_token_ids": [], "predictive_bias": []}
    # Each list contains tuples of (layer_idx, tensor_cpu), router_weights stores parameter tensors
    # global_token_ids: list of token_ids_tensor
    # predictive_bias: list of (layer_idx, delta_logits_cpu) - bias after scaling
    logits_cache = {"compute_log_prob": [], "training": [], "router_weights": {}, "global_token_ids": [], "predictive_bias": []}

    # Flag to enable/disable logits recording
    enable_logits_recording = False

    # Current cache action phase
    current_cache_action = None

    # Current token indices for alignment
    current_token_indices = None

    # Predictive loss tracking for wandb logging
    predictive_loss_tracker = []  # List of (layer_idx, loss_value)
    bias_ratio_tracker = []  # List of (layer_idx, ratio_value)
    topk_accuracy_tracker = []  # List of (layer_idx, accuracy_value)

    @staticmethod
    def clear_global_indices():
        """Clears the recorded and target topk indices in all instances."""
        for router in RouterReplay.router_instances:
            router.clear_indices()

    def __init__(self):
        """Initializes a RouterReplay instance for a specific layer."""
        self.target_topk_idx = None  # For replay
        self.recorded_topk_idx = None  # For recording
        self.router_replay_action = None  # Router replay action for this layer
        self.replay_backward_list = []  # List of tensors for backward pass replay
        self.layer_idx = len(RouterReplay.router_instances)  # Layer index

        # Router bias predictor
        self.predictive_action = None
        self.recorded_old_inputs = None  # Stored router inputs from log_prob phase
        self.recorded_old_logits = None  # Stored router logits from log_prob phase

        RouterReplay.router_instances.append(self)

    def set_target_indices(self, topk_indices: torch.Tensor):
        """Sets the target topk indices for replay."""
        self.target_topk_idx = topk_indices
        self.replay_backward_list.append(topk_indices)

    def get_recorded_indices(self):
        """Returns the recorded topk indices."""
        return self.recorded_topk_idx

    def record_indices(self, topk_indices: torch.Tensor):
        """Records the topk indices."""
        self.recorded_topk_idx = topk_indices

    def clear_indices(self):
        """Clears the recorded and target topk indices."""
        self.recorded_topk_idx = None
        self.target_topk_idx = None
        self.replay_backward_list = []

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

    """cache saving management"""
    @staticmethod
    def set_cache_action(cache_action: RouterReplayCacheAction):
        """Set the current cache action phase."""
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
        RouterReplay.logits_cache = {"compute_log_prob": [], "training": [], "router_weights": {}, "global_token_ids": [], "predictive_bias": []}
        return cache

    @staticmethod
    @no_grad()
    def record_global_token_ids(global_token_ids: torch.Tensor):
        """
        Record valid token IDs for the current micro-batch.

        Args:
            token_ids: Tensor of valid token IDs (CPU, shape [num_tokens])
        """
        # Record global token IDs for the current micro-batch.
        if not RouterReplay.enable_logits_recording or RouterReplay.current_cache_action is None:
            logger.info(f"[record_global_token_ids] Skipping - enable_recording={RouterReplay.enable_logits_recording}, ")
            return

        ids_cpu = global_token_ids.detach().cpu().contiguous()
        RouterReplay.logits_cache["global_token_ids"].append(ids_cpu)
        logger.info(f"[record_global_token_ids] Recorded global token IDs of shape {ids_cpu.shape}. ")

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
            }
        }

    """predictive routing replay management"""

    def set_predictive_data(self, inputs: torch.Tensor, logits: torch.Tensor, valid_mask=None):
        """Set old inputs and logits for this layer.
        
        Args:
            inputs: Old router inputs
            logits: Old router logits
            valid_mask: Optional boolean mask of shape [total_tokens] indicating which tokens belong to valid samples
        """
        self.recorded_old_inputs = inputs
        self.recorded_old_logits = logits
        self.predictive_valid_mask = valid_mask
        # For now this is the same as record_predictive_data, as we don't have backward yet.

    def get_predictive_data(self):
        """Get old inputs and logits for this layer."""
        return self.recorded_old_inputs, self.recorded_old_logits, self.predictive_valid_mask

    def record_predictive_data(self, inputs: torch.Tensor, logits: torch.Tensor):
        """Record inputs and logits for this layer (like record_indices)."""
        # Keep on GPU for merge function, which will handle CPU transfer uniformly
        # Use .detach() to break gradient graph and reduce memory footprint
        
        # Check if inputs are non-zero (for debugging zero data issue)
        input_sum = inputs.sum().item()
        if input_sum == 0:
            logger.warning(f"[RouterReplayLayer] [Debug] Recording ALL-ZERO inputs for layer {self.layer_idx}!")
        else:
            logger.info(f"[RouterReplayLayer] [Debug] Recording inputs with sum {input_sum} for layer {self.layer_idx}.")
        
        # Detach and create contiguous copies to allow original tensors to be freed
        self.recorded_old_inputs = inputs.squeeze().detach().contiguous()
        self.recorded_old_logits = logits.squeeze().detach().contiguous()

    def clear_predictive_data(self):
        """Clear predictive data for this layer."""
        self.recorded_old_inputs = None
        self.recorded_old_logits = None
        self.predictive_valid_mask = None

    @staticmethod
    def clear_global_predictive_data():
        """Clear predictive data for all router instances."""
        for router in RouterReplay.router_instances:
            router.clear_predictive_data()

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

    @staticmethod
    def record_predictive_loss(layer_idx: int, loss_value: float):
        """Record predictive loss for wandb logging."""
        RouterReplay.predictive_loss_tracker.append((layer_idx, loss_value))

    @staticmethod
    def record_bias_ratio(layer_idx: int, ratio_value: float):
        """Record bias-to-logits ratio for wandb logging."""
        RouterReplay.bias_ratio_tracker.append((layer_idx, ratio_value))

    @staticmethod
    def record_topk_accuracy(layer_idx: int, accuracy_value: float):
        """Record top-k prediction accuracy for wandb logging."""
        RouterReplay.topk_accuracy_tracker.append((layer_idx, accuracy_value))

    @staticmethod
    def get_and_clear_predictive_metrics():
        """Get aggregated predictive metrics and clear trackers."""
        metrics = {}

        if RouterReplay.predictive_loss_tracker:
            avg_loss = sum(loss for _, loss in RouterReplay.predictive_loss_tracker) / len(RouterReplay.predictive_loss_tracker)
            metrics['predictive_loss'] = avg_loss
            RouterReplay.predictive_loss_tracker.clear()

        if RouterReplay.bias_ratio_tracker:
            avg_ratio = sum(ratio for _, ratio in RouterReplay.bias_ratio_tracker) / len(RouterReplay.bias_ratio_tracker)
            metrics['bias_to_logits_ratio'] = avg_ratio
            RouterReplay.bias_ratio_tracker.clear()

        if RouterReplay.topk_accuracy_tracker:
            avg_accuracy = sum(acc for _, acc in RouterReplay.topk_accuracy_tracker) / len(RouterReplay.topk_accuracy_tracker)
            metrics['topk_prediction_accuracy'] = avg_accuracy
            RouterReplay.topk_accuracy_tracker.clear()

        return metrics

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

        # Move to CPU to avoid GPU memory pressure
        delta_logits_cpu = delta_logits.detach().cpu().contiguous()

        # Record to predictive_bias cache (only during compute_log_prob phase)
        if RouterReplay.current_cache_action == RouterReplayCacheAction.COMPUTE_LOG_PROB:
            RouterReplay.logits_cache["predictive_bias"].append((layer_idx, delta_logits_cpu))


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
                return _compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)

            # Use the provided indices for replay
            top_indices = router_replay.target_topk_idx
            # Ensure indices are on the correct device
            top_indices = top_indices.to(scores.device)
            # Gather the scores for the replayed indices to get the probabilities
            probs = scores.gather(1, top_indices)
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


def patched_routing(self, logits: torch.Tensor):
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
            fused=self.config.moe_router_fusion,
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

    # Apply each aux loss type and attach aux loss autograd function to probs
    if self.training and torch.is_grad_enabled() and self.is_aux_loss_enabled():
        # Calculate scores and routing_map for aux loss
        routing_map_for_aux_loss, scores_for_aux_loss = compute_routing_scores_for_aux_loss(
            logits, self.topk, self.score_function, fused=self.config.moe_router_fusion
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


def apply_predictive_loss(self, probs: torch.Tensor, predictive_loss: torch.Tensor):
    if predictive_loss is None:
        return probs
    # Attach predictive loss for backprop
    return MoEPredictiveLossAutoScaler.apply(probs, predictive_loss)


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
        logger.info(f"[Predictive Routing] Layer {self.layer_number}: {get_system_memory_info()}")
        assert self.router_replay is not None and self.bias_predictor is not None

        predictive_routing_action = self.router_replay.predictive_action if self.router_replay is not None else None
        router_replay_action = self.router_replay.router_replay_action if self.router_replay is not None else None

        predictive_loss = None

        if router_replay_action == RouterReplayAction.REPLAY_BACKWARD:
            probs, routing_map = self.routing(logits)
        else:
            if predictive_routing_action == RouterPredictiveAction.RECORD:
                # Log_prob phase: record inputs and logits, apply bias correction
                self.router_replay.record_predictive_data(input, logits)

                # Apply bias correction (linear output, controlled by bias_scale)
                # Note: Can be changed to tanh or other activation if needed
                with torch.no_grad():
                    delta_logits = self.config.router_bias_scale * self.bias_predictor(input)
                corrected_logits = logits + delta_logits

                # Track bias ratio: |delta_logits|_mean / |logits|_mean
                with torch.no_grad():
                    bias_abs_mean = torch.abs(delta_logits).mean().item()
                    logits_abs_mean = torch.abs(logits).mean().item()
                    bias_ratio = bias_abs_mean / (logits_abs_mean + 1e-10)
                    layer_idx = self.router_replay.layer_idx if self.router_replay else 0
                    RouterReplay.record_bias_ratio(layer_idx, bias_ratio)

                    # Record predictive bias to logits cache if saving is enabled
                    if RouterReplay.enable_logits_recording:
                        RouterReplay.record_predictive_bias(delta_logits, layer_idx)

                probs, routing_map = self.routing(corrected_logits)

            elif predictive_routing_action == RouterPredictiveAction.SKIP_PREDICTIVE:
                # Training phase ministep=0
                # Skip predictive loss, use normal routing
                if self.layer_number == 1 and self.router_replay and self.router_replay.layer_idx == 0:
                     logger.info(f"[Predictive Routing Replay] Action is SKIP_PREDICTIVE. Skipping loss computation.")
                probs, routing_map = self.routing(logits)

            elif predictive_routing_action == RouterPredictiveAction.COMPUTE_PREDICTIVE_LOSS:
                # Training phase ministep>=1: compute predictive loss
                if self.layer_number == 1 and self.router_replay and self.router_replay.layer_idx == 0:
                     logger.info(f"[Predictive Routing Replay] Action is COMPUTE_PREDICTIVE_LOSS. Computing loss...")

                gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3)
                logger.info(f"[Memory] (layer {self.layer_number}) Total GPU memory allocated before predictive loss computation: {gpu_mem:.2f} GB, {get_system_memory_info()}")
                old_inputs, old_logits, valid_mask = self.router_replay.get_predictive_data()

                if old_inputs is not None and old_logits is not None:
                    old_inputs = old_inputs.to(input.device)
                    old_logits = old_logits.to(logits.device)
                    gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3)
                    logger.info(f"[Memory] (layer {self.layer_number}) Total GPU memory allocated after loading predictive data: {gpu_mem:.2f} GB, {get_system_memory_info()}")

                    # Debug asserts: check shape matching
                    assert old_inputs.shape[-1] == input.shape[-1], f"hidden_size mismatch: old={old_inputs.shape[-1]}, current={input.shape[-1]}"
                    assert old_logits.shape[-1] == logits.shape[-1], f"num_experts mismatch: old={old_logits.shape[-1]}, current={logits.shape[-1]}"

                    # Apply token-level mask if provided (for downsampled data)
                    # valid_mask is at token level, matching the unpacked input/logits shape
                    current_input = input # TODO: this is not used, it is preserved for future algorithm enhancement
                    current_logits = logits
                    if valid_mask is not None:
                        # Filter current input and logits at token level to match old_inputs/old_logits
                        current_input = input[valid_mask]
                        current_logits = logits[valid_mask]
                        logger.info(f"[Predictive Routing Replay] Applied token-level mask: {valid_mask.sum().item()}/{valid_mask.size(0)} valid tokens")
                        assert current_input.shape[0] == old_inputs.shape[0], f"Token count mismatch after masking: old={old_inputs.shape[0]}, current={current_input.shape[0]}"
                        assert current_logits.shape[0] == old_logits.shape[0], f"Token count mismatch after masking: old={old_logits.shape[0]}, current={current_logits.shape[0]}"

                    # Compute delta_logits and logits_diff
                    # Use old_inputs with current weights to get delta_logits
                    delta_logits = self.config.router_bias_scale * self.bias_predictor(old_inputs)
                    logits_diff = current_logits - old_logits

                    # Compute loss between delta_logits and logits_diff
                    if self.config.router_bias_predictor_loss_type == "l2":
                        # L2 loss on raw logits difference
                        predictive_loss = torch.nn.functional.mse_loss(delta_logits, logits_diff.detach(), reduction='mean')

                    elif self.config.router_bias_predictor_loss_type == "kl":
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
                        pred_probs = torch.softmax(delta_logits, dim=-1)
                        target_probs = torch.softmax(logits_diff, dim=-1)
                        predictive_loss = torch.nn.functional.kl_div(
                            torch.log(pred_probs + 1e-10),
                            target_probs.detach(),
                            reduction='batchmean'
                        )

                    elif self.config.router_bias_predictor_loss_type == "kl-post":
                        # KL divergence on corrected vs uncorrected routing distributions
                        # This measures the KL between final routing decisions
                        pred_log = torch.log_softmax(old_logits + delta_logits, dim=-1)
                        target_log = torch.log_softmax(current_logits, dim=-1)
                        predictive_loss = torch.sum(torch.exp(pred_log) * (pred_log - target_log.detach()), dim=-1).mean()

                    else:
                        raise ValueError(f"Invalid loss type: {self.config.router_bias_predictor_loss_type}")

                    # Record predictive loss for wandb logging
                    layer_idx = self.router_replay.layer_idx if self.router_replay else 0
                    RouterReplay.record_predictive_loss(layer_idx, predictive_loss.item())

                    # Compute top-k prediction accuracy
                    with torch.no_grad():
                        # Predicted top-k: use old_inputs + current bias predictor
                        _, predicted_topk_indices = torch.topk(old_logits + delta_logits, k=self.topk, dim=-1)

                        # True top-k: use current_logits if valid_mask is applied, otherwise use logits
                        _, true_topk_indices = torch.topk(current_logits, k=self.topk, dim=-1)

                        # Compute accuracy: percentage of matches
                        # Expand dims for broadcasting comparison
                        predicted_expanded = predicted_topk_indices.unsqueeze(-1)  # [tokens, topk, 1]
                        true_expanded = true_topk_indices.unsqueeze(-2)  # [tokens, 1, topk]
                        matches = (predicted_expanded == true_expanded).any(dim=-1)  # [tokens, topk]
                        accuracy = matches.float().mean().item()

                        RouterReplay.record_topk_accuracy(layer_idx, accuracy)

                    gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3)
                    logger.info(f"[Memory] (layer {self.layer_number}) Total GPU memory allocated after predictive loss computation: {gpu_mem:.2f} GB, {get_system_memory_info()}")
                else:
                    logger.info("Warning: Missing predictive data for router bias predictor loss computation.")

                probs, routing_map = self.routing(logits)

            else:
                # DISABLED : normal routing
                probs, routing_map = self.routing(logits)

        probs = self.apply_predictive_loss(probs, predictive_loss)

    else:
        # Standard routing without bias predictor
        probs, routing_map = self.routing(logits)

    return probs, routing_map


def apply_router_replay_patch():
    """
    Applies the monkey patch for MoE Router Replay functionality.
    This patch dynamically adds the 'enable_routing_replay' attribute to TransformerConfig
    and modifies the TopKRouter to support recording and replaying of routing decisions.
    
    Also supports router bias predictor for R2-only predicted routing replay.
    """
    logger.info("Applying Router Replay Patch...")
    # Clear router instances to avoid state leakage between model initializations.
    RouterReplay.router_instances.clear()
    # Step 1: Patch TransformerConfig to include the feature flags
    if not hasattr(TransformerConfig, "enable_routing_replay"):
        # Add class attribute with default value
        TransformerConfig.enable_routing_replay = False

        # Store original __init__ method
        original_tf_config_init = TransformerConfig.__init__

        # Define new __init__ method that safely handles enable_routing_replay parameter
        def patched_tf_config_init(self, *args, **kwargs):
            # Simple solution: remove the unknown parameter before calling original constructor
            enable_routing_replay = kwargs.pop("enable_routing_replay", TransformerConfig.enable_routing_replay)

            # Also handle router bias predictor parameters
            enable_router_bias_predictor = kwargs.pop("enable_router_bias_predictor", False)
            router_bias_scale = kwargs.pop("router_bias_scale", 1.0)
            router_bias_predictor_loss_type = kwargs.pop("router_bias_predictor_loss_type", "kl")

            # Call original constructor with remaining kwargs
            original_tf_config_init(self, *args, **kwargs)

            # Set the instance attributes
            self.enable_routing_replay = enable_routing_replay
            self.enable_router_bias_predictor = enable_router_bias_predictor
            self.router_bias_scale = router_bias_scale
            self.router_bias_predictor_loss_type = router_bias_predictor_loss_type

        # Apply the patch
        TransformerConfig.__init__ = patched_tf_config_init

    # Step 2: Patch TopKRouter only once to ensure idempotency.
    if hasattr(TopKRouter, "_router_replay_patched"):
        return

    original_init = TopKRouter.__init__

    # Step 3: Define the new __init__ method
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.router_replay = None
        if self.config.enable_routing_replay:
            self.router_replay = RouterReplay()

    # Step 4: Apply the patches
    TopKRouter.__init__ = patched_init
    TopKRouter.routing = patched_routing
    TopKRouter._router_replay_patched = True
    # predictive routing replay
    TopKRouter.apply_predictive_loss = apply_predictive_loss
    TopKRouter.forward = patched_forward

    logger.info(f"Router Replay Patch applied successfully. "
                f"enable_routing_replay={TransformerConfig.enable_routing_replay}, "
                f"enable_router_bias_predictor={getattr(TransformerConfig, 'enable_router_bias_predictor', False)}")
