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
import warnings
from enum import Enum

import torch
from torch import no_grad

try:
    from megatron.core.transformer.moe.moe_utils import (
        apply_router_token_dropping,
        compute_routing_scores_for_aux_loss,
        group_limited_topk,
    )
except ImportError:
    warnings.warn("NPU not support router replay for now.", stacklevel=2)
    pass
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig

# https://github.com/THUDM/slime/blob/main/slime/utils/routing_replay.py


class RouterReplayAction(Enum):
    RECORD = "record"
    REPLAY_FORWARD = "replay_forward"
    REPLAY_BACKWARD = "replay_backward"


class RouterReplayCacheAction(Enum):
    """Enum for logits cache recording phases."""
    COMPUTE_LOG_PROB = "compute_log_prob"
    TRAINING = "training"


class RouterReplay:
    """
    A class to manage the recording and replaying of MoE routing decisions.
    It holds all router instances and provides static methods to globally
    control recording and replaying.
    """

    # Static variable to hold all router instances, one per MoE layer.
    router_instances = []

    # Global logits cache for recording
    # Structure: {"compute_log_prob": [], "training": [], "router_weights": {}, "global_token_ids": []}
    # Each list contains tuples of (layer_idx, tensor_cpu), router_weights stores parameter tensors
    # global_token_ids: list of token_ids_tensor
    logits_cache = {"compute_log_prob": [], "training": [], "router_weights": {}, "global_token_ids": []}

    # Flag to enable/disable logits recording
    enable_logits_recording = False

    # Current cache action phase
    current_cache_action = None

    # Current token indices for alignment
    current_token_indices = None

    @staticmethod
    def set_replay_data(all_layers_topk_indices: list):
        """
        Distributes the topk indices for all layers to their respective RouterReplay instances.
        :param all_layers_topk_indices: A list of tensors, where each tensor contains the
                                        topk indices for a specific layer. The order
                                        must match the instantiation order of the routers.
        """
        if len(all_layers_topk_indices) != len(RouterReplay.router_instances):
            raise ValueError(
                f"The number of replay tensors ({len(all_layers_topk_indices)}) "
                f"does not match the number of router instances ({len(RouterReplay.router_instances)})."
            )
        for i, router_instance in enumerate(RouterReplay.router_instances):
            router_instance.set_target_indices(all_layers_topk_indices[i])

    @staticmethod
    def get_recorded_data() -> list:
        """
        Collects the recorded topk indices from all RouterReplay instances.
        :return: A list of tensors, each containing the recorded topk indices for a layer.
        """
        return [router.get_recorded_indices() for router in RouterReplay.router_instances]

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
        # Debug: check cache before clearing
        print(f"[get_and_clear_logits_cache] Before clear - "
                   f"compute_log_prob: {len(RouterReplay.logits_cache['compute_log_prob'])} items, "
                   f"training: {len(RouterReplay.logits_cache['training'])} items, "
                   f"router_weights: {len(RouterReplay.logits_cache['router_weights'])} items, "
                   f"global_token_ids: {len(RouterReplay.logits_cache['global_token_ids'])} items")

        cache = RouterReplay.logits_cache
        RouterReplay.logits_cache = {"compute_log_prob": [], "training": [], "router_weights": {}, "global_token_ids": []}
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
            print(f"[record_global_token_ids] Skipping - enable_recording={RouterReplay.enable_logits_recording}, ")
            return

        ids_cpu = global_token_ids.detach().cpu().contiguous()
        RouterReplay.logits_cache["global_token_ids"].append(ids_cpu)
        print(f"[record_global_token_ids] Recorded global token IDs of shape {ids_cpu.shape}. ")

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
            print(f"[record_logits] Layer 0: enable_recording={RouterReplay.enable_logits_recording}, "
                  f"cache_action={RouterReplay.current_cache_action}, "
                  f"logits_shape={logits.shape}")

        if not RouterReplay.enable_logits_recording or RouterReplay.current_cache_action is None:
            if layer_idx == 0:
                print(f"[record_logits] Skipping - enable_recording={RouterReplay.enable_logits_recording}, "
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
                print(f"[record_logits] Recorded to compute_log_prob cache. Current size: {len(RouterReplay.logits_cache['compute_log_prob'])}")
        elif RouterReplay.current_cache_action == RouterReplayCacheAction.TRAINING:
            RouterReplay.logits_cache["training"].append((layer_idx, logits_cpu))
            if layer_idx == 0:
                print(f"[record_logits] Recorded to training cache. Current size: {len(RouterReplay.logits_cache['training'])}")

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
            }
        }


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
        import logging
        logger = logging.getLogger(__name__)

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
            print(f"[compute_topk] Layer 0: routing_action={routing_action}, "
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


def apply_router_replay_patch():
    """
    Applies the monkey patch for MoE Router Replay functionality.
    This patch dynamically adds the 'enable_routing_replay' attribute to TransformerConfig
    and modifies the TopKRouter to support recording and replaying of routing decisions.
    """
    print("Applying Router Replay Patch...")
    # Clear router instances to avoid state leakage between model initializations.
    RouterReplay.router_instances.clear()
    # Step 1: Patch TransformerConfig to include the feature flag
    if not hasattr(TransformerConfig, "enable_routing_replay"):
        # Add class attribute with default value
        TransformerConfig.enable_routing_replay = False

        # Store original __init__ method
        original_tf_config_init = TransformerConfig.__init__

        # Define new __init__ method that safely handles enable_routing_replay parameter
        def patched_tf_config_init(self, *args, **kwargs):
            # Simple solution: remove the unknown parameter before calling original constructor
            enable_routing_replay = kwargs.pop("enable_routing_replay", TransformerConfig.enable_routing_replay)

            # Call original constructor with remaining kwargs
            original_tf_config_init(self, *args, **kwargs)

            # Set the instance attribute
            self.enable_routing_replay = enable_routing_replay

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
