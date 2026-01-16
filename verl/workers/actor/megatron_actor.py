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
"""
Megatron Actor.
In megatron actor, the differences are:
1. We only make minibatch

Note that our model doesn't have to be `MegatronModule` because we don't share embedding in the last layer
"""

import itertools
import logging
import os
from functools import partial
from typing import Iterable

import psutil
import numpy as np
import torch
import torch.distributed
from megatron.core import parallel_state as mpu
from megatron.core.distributed import finalize_model_grads

# from megatron.core.optimizer import DistributedOptimizer
from megatron.core.optimizer import DistributedOptimizer
from megatron.core.pipeline_parallel import get_forward_backward_func
from omegaconf import OmegaConf
from torch import nn

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id, get_torch_device
from verl.utils.megatron.pipeline_parallel import make_batch_generator
from verl.utils.megatron.router_replay_patch import RouterReplay, RouterReplayAction, RouterReplayCacheAction, RouterPredictiveAction
from verl.utils.megatron.router_replay_saver import RouterReplayLogitsSaver
from verl.utils.megatron.router_replay_utils import (
    RouterReplayHelper,
    merge_router_topk_indices,
    pp_gather,
    reorder_and_merge_vpp_layers,
    set_router_replay_data,
    merge_router_predictive_data,
    set_router_predictive_data,
)
from verl.utils.megatron.tensor_parallel import vocab_parallel_entropy, vocab_parallel_log_probs_from_logits
from verl.utils.megatron_utils import get_model_config, unwrap_model
from verl.utils.memory_utils import get_system_memory_info
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.profiler.profile import Profiler
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import broadcast_dict_tensor
from verl.workers.actor import BasePPOActor

__all__ = ["MegatronPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class MegatronPPOActor(BasePPOActor):
    def __init__(
        self,
        config,
        model_config,
        hf_config,
        tf_config,
        actor_module: nn.ModuleList,
        actor_optimizer: DistributedOptimizer,
    ):
        """MeagtronPPOActor class. This class implements the simple PPO logics when the model is built with Megatron.

        Args:
            config (OmegaConf): the basic config that contains the hyper-parameters of PPO Actor. It must contain

                ``ppo_micro_batch_size_per_gpu``: micro batch size when updating ppo.

                ``ppo_mini_batch_size``: minibatch size when updating ppo using the batch data.

                ``ppo_epochs``: number of epochs to update the actor using the batch data.

                ``shuffle``: whether to shuffle the data after each ppo epoch.

                ``clip_ratio``: clip ratio of the ppo algorithm. See https://arxiv.org/abs/1707.06347.

                ``entropy_coeff``: entropy coefficient of the PPO loss. See https://arxiv.org/abs/1707.06347.
            model_config (OmegaConf): model configuration. It must contains ``model_config.vocab_size`` and
                ``model_config.hidden_size``
            hf_config (PretrainedConfig): huggingface config
            tf_config (TransformerConfig): mcore transformer config
            actor_module (nn.ModuleList): actor module is a ModuleList that contains a list of nn.Module in this
                pp stage.
                each nn.Module in this rank holds a vpp module chunk. See https://arxiv.org/pdf/2104.04473.pdf for
                more details.
                The actor module has some constraints to follow in order to use the updating logics implemented here

                1. It must implement unpad_input before any computation and pad_input after all the computation.
                Remove padding is an
                optimization that removes the padding tokens. See unpad_input and pad_input function in flash-attn
                (https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py).

                2. Each pp stage must return the hidden state with the same shape [total_nnz, 1, hidden_size],
                where total_nnz is the number of valid tokens in this batch. If sequence parallel is enabled, the size
                of the hidden state is [total_nnz // tp, 1, hidden_size].
            actor_optimizer (DistributedOptimizer): currently, we only support DistributedOptimizer in Megatron.
                It implements
                zero1 optimizer that shards the optimizer state across dp ranks.

        >>> from megatron.training import get_model
        >>> from megatron.optimizer import get_megatron_optimizer
        >>> actor_module = get_model(megatron_actor_model_provider, wrap_with_ddp=True)
        >>> actor_module = nn.ModuleList(actor_module)
        >>> actor_optimizer = get_megatron_optimizer(actor_module)
        >>> actor = MegatronPPOActor(config=config,
        >>>                          model_config=actor_model_config,
        >>>                          hf_config=hf_config,
        >>>                          tf_config=tf_config,
        >>>                          actor_module=actor_module,
        >>>                          actor_optimizer=actor_optimizer)
        """
        super().__init__(config)
        self._validate_config(config)
        self.model_config = model_config
        self.hf_config = hf_config
        self.tf_config = tf_config
        self.actor_module = actor_module
        self.actor_optimizer: DistributedOptimizer = actor_optimizer
        self.use_torch_profiler = self.config.profiler.get("tool") == "torch"
        if self.use_torch_profiler:
            self.prof = Profiler(
                self.config.profiler, tool_config=self.config.profiler.get("tool_config", {}).get("torch", {})
            )
        else:
            self.prof = None
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if self.use_fused_kernels and not getattr(self.config, "overlap_moe_expert_parallel_comm", False):
            # do not patch if overlap_moe_expert_parallel_comm is enabled
            from verl.models.mcore.model_forward_fused import patch_fused_forward

            for model in self.actor_module:
                patch_fused_forward(model)

        self.optimizer_step_args = OmegaConf.create(
            {
                "skip_grad": None,
                "overlap_dp_param_comm": False,
                "overlap_dp_grad_comm": False,
                "gradient_accumulation_steps": 1,
                "sequence_parallel": self.tf_config.sequence_parallel,
                "DDP_impl": "local",
                "layernorm_allreduce_bucket_threshold": 0,
                "reduce_grads_use_alltoall": False,
            }
        )

        self.router_replay = self.config.router_replay
        self.enable_routing_replay = self.router_replay.mode != "disabled"
        self.enable_bias_predictor = self.router_replay.enable_bias_predictor

        # Initialize mini_layer_topk_idx_list if router replay is enabled
        if self.enable_routing_replay:
            self.mini_layer_topk_idx_list = []
            if self.enable_bias_predictor:
                self.mini_layer_old_inputs_list = []
                self.mini_layer_old_logits_list = []
                self.mini_layer_sampled_masks_list = []

        # Initialize logits saver if record_file is specified
        if self.router_replay.record_file not in [None, ""]:
            self.logits_saver = RouterReplayLogitsSaver(self.router_replay.record_file)
            self.enable_logits_saving = True
            self.training_step = 0  # Track training steps for file naming
            self.save_frequency = self.router_replay.save_frequency
            logger.info(f"[Routing Replay] Router logits saving enabled. Save directory: {self.router_replay.record_file}, frequency: every {self.save_frequency} step(s)")
        else:
            self.logits_saver = None
            self.enable_logits_saving = False
            self.save_frequency = 1

        config = get_model_config(self.actor_module[0])
        print(config)
        config.finalize_model_grads_func = finalize_model_grads

    def _validate_config(self, config) -> None:
        """Validate config options not implemented for Megatron backend"""
        assert config.get("ulysses_sequence_parallel_size", 1) == 1
        if config.get("shuffle", False):
            assert config.data_loader_seed is not None, "If shuffle dataloader, seed must be manually set"
        if config.megatron.tensor_model_parallel_size == 1:
            print("[Warining] Because actor tp size == 1, set sp to False")
            config.megatron.sequence_parallel = False
        self.config = config

    @GPUMemoryLogger(role="megatron actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            DataProto: torch.Tensor: the log_prob tensor
        """
        use_dynamic_bsz = data.meta_info.get("use_dynamic_bsz", False)
        micro_batch_size = data.meta_info.get("micro_batch_size", None)
        max_token_len = data.meta_info.get("max_token_len", None)
        if use_dynamic_bsz:
            assert max_token_len is not None, "max_token_len must be set when use_dynamic_bsz is True"
            max_token_len = max_token_len * self.config.megatron.context_parallel_size
        else:
            assert micro_batch_size is not None, (
                "micro batch size is needed for forward compute when use_dynamic_bsz is False"
            )

        def compute_logprobs_fn(output, data, use_dynamic_bsz=False, indices=None):
            response = data["responses"]
            response_length = response.size(1)
            log_probs = output["log_probs"][:, -response_length - 1 : -1].contiguous()
            return {"log_probs": log_probs}

        # We make recompute_old_log_prob by default here.
        # TODO (zhangchi.usc1992): actually, this function should only return log_prob and this logic should be
        # handled by user outside
        recompute_old_log_prob = self.config.get("recompute_old_log_prob", True)

        entropys = torch.Tensor()
        if recompute_old_log_prob:
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]

            # R3 mode: use routed_experts from previous step if available
            # For the first step, routed_experts won't exist, so we skip the check
            if self.enable_routing_replay and self.config.router_replay.mode == "R3":
                if "routed_experts" in data.batch.keys():
                    select_keys.append("routed_experts")
                else:
                    # First step in R3 mode: no routed_experts yet, will use R2-like recording
                    logger.warning("[compute_log_prob] R3 mode but routed_experts not in batch (likely first step). "
                                   "Will record routing decisions this step for replay in next step.")
            
            # Include global_token_ids if present (for alignment in update_policy phase)
            if "global_token_ids" in data.batch.keys():
                select_keys.append("global_token_ids")

            batch = data.select(batch_keys=select_keys).batch
            input_ids = batch["input_ids"]
            batch_size = input_ids.size(0)
            response = batch["responses"]
            response_length = response.size(1)
            with torch.no_grad():
                logger.info(f"[Data] [compute_log_prob] Total batch length: {batch_size}")
                output = self.forward_backward_batch(
                    data,
                    forward_only=True,
                    post_process_fn=compute_logprobs_fn,
                    calculate_entropy=calculate_entropy,
                    use_dynamic_bsz=use_dynamic_bsz,
                    micro_batch_size=micro_batch_size,
                    max_token_len=max_token_len,
                )
                if mpu.is_pipeline_last_stage(ignore_virtual=True):
                    # only on last rank. It should be on every tp rank
                    if calculate_entropy:
                        log_probs = [o[0]["log_probs"] for o in output["output"]]  # (bs, seq_size)
                    else:
                        log_probs = [o["log_probs"] for o in output["output"]]  # (bs, seq_size)
                    log_probs = torch.cat(log_probs, dim=0).to(torch.float32)
                    if use_dynamic_bsz:
                        indices = output["indices"]
                        indices = list(itertools.chain.from_iterable(indices))
                        assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
                        revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                        log_probs = log_probs[revert_indices]
                else:
                    log_probs = torch.empty(
                        size=(batch_size, response_length), dtype=torch.float32, device=input_ids.device
                    )
                log_probs = log_probs.to(get_device_id())
                # broadcast across pp ranks
                torch.distributed.broadcast(
                    tensor=log_probs,
                    src=mpu.get_pipeline_model_parallel_last_rank(),
                    group=mpu.get_pipeline_model_parallel_group(),
                    async_op=False,
                )
                log_probs = log_probs.to("cpu")
                if calculate_entropy:
                    # Note that o[0] is metrics, o[1] is entropy
                    if mpu.is_pipeline_last_stage(ignore_virtual=True):
                        entropys = torch.cat([o[1] for o in output["output"]], dim=0)
                        entropys = entropys.to(torch.float32)
                        if use_dynamic_bsz:
                            indices = output["indices"]
                            indices = list(itertools.chain.from_iterable(indices))
                            assert len(indices) == entropys.size(0), f"{len(indices)} vs. {entropys.size()}"
                            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                            entropys = entropys[revert_indices]
                    else:
                        entropys = torch.empty(
                            size=(batch_size, response_length), dtype=torch.float32, device=input_ids.device
                        )
                    # broadcast across pp ranks
                    entropys = entropys.to(get_device_id())
                    torch.distributed.broadcast(
                        tensor=entropys,
                        src=mpu.get_pipeline_model_parallel_last_rank(),
                        group=mpu.get_pipeline_model_parallel_group(),
                        async_op=False,
                    )
                    entropys = entropys.to("cpu")

                layers_topk_idx = None
                layers_predictive_states = None

                if RouterReplayHelper.is_r2_record_action(self.tf_config):
                    # (bs, max_seq_len/response_len,local_layer_num,topk)
                    layers_topk_idx = output["mini_layer_topk_idx_tensor"].to(torch.uint8)
                    if use_dynamic_bsz:
                        indices = output["indices"]
                        indices = list(itertools.chain.from_iterable(indices))
                        assert len(indices) == layers_topk_idx.size(0), f"{len(indices)} vs. {layers_topk_idx.size()}"
                        revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                        layers_topk_idx = layers_topk_idx[revert_indices]
                    layers_topk_idx = pp_gather(layers_topk_idx, self.tf_config)

                    if RouterReplayHelper.is_predictive_record_action(self.tf_config):
                        # NEW: Expect list of variable-shape tensors instead of single padded tensor
                        layers_old_inputs_list = output["mini_layer_old_inputs_list"]  # list of [num_tokens_i, layers, hidden]
                        layers_old_logits_list = output["mini_layer_old_logits_list"]  # list of [num_tokens_i, layers, num_experts]
                        sampled_masks = output["mini_layer_sampled_masks_tensor"]  # [num_batches * bs], bool tensor
                        logger.info(f"[Predictive Routing Replay] Shape of layers_old_inputs_list after compute_log_prob forward: {[t.shape if t is not None else None for t in layers_old_inputs_list]}")
                        logger.info(f"[Predictive Routing Replay] Shape of layers_old_logits_list after compute_log_prob forward: {[t.shape if t is not None else None for t in layers_old_logits_list]}")
                        logger.info(f"[Predictive Routing Replay] Shape of sampled_masks after compute_log_prob forward: {sampled_masks.shape}, values: {sampled_masks}")

                        # Memory debug
                        # inputs_mb = sum(t.numel() * t.element_size() for t in layers_old_inputs_list) / 1024 / 1024
                        # logits_mb = sum(t.numel() * t.element_size() for t in layers_old_logits_list) / 1024 / 1024
                        # logger.info(f"[Predictive Routing Replay] [Memory] compute_log_prob: old_inputs={inputs_mb:.2f}MB (list of {len(layers_old_inputs_list)} samples), old_logits={logits_mb:.2f}MB, {get_system_memory_info()}")

                        # Convert to numpy for Ray efficiency
                        # Each tensor is already on CPU and has shape [num_tokens_i, layers, hidden]
                        layers_old_inputs_list_np = [t.contiguous().numpy() for t in layers_old_inputs_list]
                        layers_old_logits_list_np = [t.contiguous().numpy() for t in layers_old_logits_list]

                        # Delete the torch tensors
                        del layers_old_inputs_list, layers_old_logits_list
                        import gc
                        gc.collect()
                        
                        # Note: dynamic_bsz reordering - if needed, reorder the list
                        if use_dynamic_bsz:
                            # TODO: check correctness
                            indices = output["indices"]  # [num_batches * bs], range from 0 to (num_batches * bs - 1)
                            indices_tensor = torch.tensor(list(itertools.chain.from_iterable(indices)) , dtype=torch.long, device=sampled_masks.device)  # Convert to tensor
                            sampled_indices = indices_tensor[sampled_masks]  # [num_batches * sampled_batch_size], range from 0 to (num_batches * bs - 1)
                            # Now we need to reassign the indices from 0 to (num_batches * sampled_batch_size - 1), while keeping the order
                            # Out target: [99, 20, 49, 2, 40, 75, 0, 8] => [7, 3, 5, 1, 4, 6, 0, 2]
                            indices_for_sorting = torch.argsort(sampled_indices)  # [99, 20, 49, 2, 40, 75, 0, 8] => [6, 3, 7, 1, 4, 2, 5, 0]
                            revert_indices = torch.tensor(get_reverse_idx(indices_for_sorting), dtype=torch.long)  # [6, 3, 7, 1, 4, 2, 5, 0] => [7, 3, 5, 1, 4, 6, 0, 2]
                            # Reorder the numpy lists
                            layers_old_inputs_list_np = [layers_old_inputs_list_np[i] for i in revert_indices]
                            layers_old_logits_list_np = [layers_old_logits_list_np[i] for i in revert_indices]

                        # Insert None for unsampled entries
                        full_inputs_list = []
                        full_logits_list = []
                        sampled_idx = 0
                        for i in range(len(sampled_masks)):
                            if sampled_masks[i]:
                                full_inputs_list.append(layers_old_inputs_list_np[sampled_idx])
                                full_logits_list.append(layers_old_logits_list_np[sampled_idx])
                                sampled_idx += 1
                            else:
                                full_inputs_list.append(None)
                                full_logits_list.append(None)
                        
                        # logger.info(f"[Predictive Routing Replay] [Downsample] Restored predictive data to full batch size: {len(sampled_masks)} (downsampled from {len(layers_old_inputs_list_np)})")

                        layers_predictive_states = (full_inputs_list, full_logits_list)

        # add empty cache after each compute
        get_torch_device().empty_cache()

        return log_probs, entropys, layers_topk_idx, layers_predictive_states

    def make_minibatch_iterator(self, data: DataProto) -> Iterable[DataProto]:
        """Make minibatch iterator for updating the actor

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64, where
                ``sequence_length = prompt_length + response_length``

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64

                ``responses``: tensor of shape [batch_size, response_length]. torch.int64. Note that
                responses = input_ids[:, -response_length:]

                ``old_log_probs``: tensor of shape [batch_size, response_length]. torch.float32. The log probability
                of responses.

                ``advantages``: tensor of shape [batch_size, response_length]. torch.float32. The advantages of
                responses.
                See PPO paper for details. https://arxiv.org/abs/1707.06347

        Returns:

        """
        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "response_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")
        self.has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        # router replay
        if self.enable_routing_replay:
            # Only include routed_experts if it actually exists in the data
            if "routed_experts" in data.batch.keys():
                select_keys.append("routed_experts")
            elif self.config.router_replay.mode == "R3":
                # R3 without routed_experts: first step, will use RECORD mode
                logger.warning("[make_minibatch_iterator] R3 mode but routed_experts not in batch. Training will record this step.")
        if self.enable_logits_saving:
            select_keys.append("global_token_ids")
        # Router bias predictor: include old_inputs and old_logits for training
        # These are stored in non_tensor_batch, not in batch
        non_tensor_batch_keys = []
        if self.enable_bias_predictor:
            if "old_inputs" in data.non_tensor_batch.keys():
                non_tensor_batch_keys.append("old_inputs")
                # logger.info(f"[Memory] [make_minibatch_iterator] Including old_inputs in non_tensor_batch")
            if "old_logits" in data.non_tensor_batch.keys():
                non_tensor_batch_keys.append("old_logits")
                # logger.info(f"[Memory] [make_minibatch_iterator] Including old_logits in non_tensor_batch")
        
        # logger.info(f"[Memory] [make_minibatch_iterator] Before data.select(): {get_system_memory_info()}")
        if self.has_multi_modal_inputs:
            data = data.select(select_keys, non_tensor_batch_keys=non_tensor_batch_keys + ["multi_modal_inputs"])
        else:
            data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_batch_keys)
        # logger.info(f"[Memory] [make_minibatch_iterator] After data.select(): {get_system_memory_info()}")

        # logger.info(f"[Memory] [make_minibatch_iterator] Before data.make_iterator(): {get_system_memory_info()}")
        iterator = data.make_iterator(
            mini_batch_size=self.config.ppo_mini_batch_size,
            epochs=self.config.ppo_epochs,
            seed=self.config.data_loader_seed,
        )
        # logger.info(f"[Memory] [make_minibatch_iterator] After data.make_iterator(): {get_system_memory_info()}")
        return iterator

    def forward_backward_batch(
        self,
        data: DataProto,
        forward_only=False,
        post_process_fn=None,
        calculate_entropy=False,
        use_dynamic_bsz=False,
        micro_batch_size=None,
        max_token_len=None,
        mini_batch_size=None,
    ):
        """
        We assume:
        - The model takes input: (input_ids, attention_mask, position_ids). No rmpad for the input
        - The communication shape is (total_nnz_pad_to_sp // tp_size, 1, hidden_size) if sequence parallel is enabled
        """
        # broadcast from last pp rank to all other pp ranks
        # TODO: actually, we just need to control the sampling order.
        data.to(get_device_id())
        data.batch = data.batch.contiguous()
        mini_batch = data
        broadcast_dict_tensor(
            mini_batch.batch,
            src=mpu.get_pipeline_model_parallel_last_rank(),
            group=mpu.get_pipeline_model_parallel_group(),
        )
        mini_batch.to("cpu")
        # split into micro-batches
        mini_batch.batch["attention_mask"] = mini_batch.batch["attention_mask"].to(bool)
        self.has_multi_modal_inputs = "multi_modal_inputs" in mini_batch.non_tensor_batch.keys()
        if self.has_multi_modal_inputs:
            mini_batch.batch["multi_modal_inputs"] = mini_batch.non_tensor_batch["multi_modal_inputs"]
            mini_batch.batch["multi_modal_inputs_idx"] = torch.Tensor(
                list(range(len(mini_batch.non_tensor_batch["multi_modal_inputs"])))
            ).to(torch.int64)

        if mini_batch.batch["position_ids"].dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
            mini_batch.batch["position_ids"] = mini_batch.batch["position_ids"][
                :, 0
            ]  # mcore patch recompute qwen2vl's pos ids during forward

        indices = None
        temperature = data.meta_info["temperature"]

        def _add_non_tensor_batch_to_micro_batches(micro_batches, mini_batch, indices=None):
            """
            Add old_inputs and old_logits from non_tensor_batch to each micro_batch's batch.
            
            Args:
                micro_batches: List of DataProto micro batches
                mini_batch: DataProto containing non_tensor_batch
                indices: Optional list of index lists for dynamic batching
            """
            if "old_inputs" not in mini_batch.non_tensor_batch and "old_logits" not in mini_batch.non_tensor_batch:
                logger.info(f"[Predictive Routing Replay] Skip adding non_tensor_batch due to no old_inputs or old_logits found.")
                return

            old_inputs_list = mini_batch.non_tensor_batch["old_inputs"]
            old_logits_list = mini_batch.non_tensor_batch["old_logits"]

            # Ensure they are lists (protocol.py might have converted them to object arrays)
            if isinstance(old_inputs_list, np.ndarray):
                old_inputs_list = old_inputs_list.tolist()
            if isinstance(old_logits_list, np.ndarray):
                old_logits_list = old_logits_list.tolist()

            # Split the lists according to indices or sequential split
            if indices is not None:
                # Dynamic batching: use indices to split
                for i, batch_idx in enumerate(indices):
                    micro_batches[i].non_tensor_batch["old_inputs"] = [old_inputs_list[idx] for idx in batch_idx]
                    micro_batches[i].non_tensor_batch["old_logits"] = [old_logits_list[idx] for idx in batch_idx]
            else:
                raise ValueError("Indices must be provided for dynamic batching")
        
        if use_dynamic_bsz:
            assert max_token_len is not None, "max_token_len must be set when use_dynamic_bsz is True"
            vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
            if vpp_size is not None and vpp_size > 1:
                microbatch_group_size_per_vp_stage = self.tf_config.microbatch_group_size_per_vp_stage
                micro_batches_td, indices = rearrange_micro_batches(
                    batch=mini_batch.batch,
                    num_batches_divided_by=microbatch_group_size_per_vp_stage,
                    max_token_len=max_token_len,
                )
                
                # Wrap TensorDicts in DataProto
                micro_batches = [
                    DataProto(batch=mb, meta_info=mini_batch.meta_info)
                    for mb in micro_batches_td
                ]
                
                # Add non_tensor_batch data to each micro_batch
                _add_non_tensor_batch_to_micro_batches(micro_batches, mini_batch, indices=indices)
                assert len(micro_batches) % self.tf_config.microbatch_group_size_per_vp_stage == 0, (
                    f"micro_batches {micro_batches} must be divisible by microbatch_group_size_per_vp_stage "
                    f"{microbatch_group_size_per_vp_stage} for megatron backend"
                )
            else:
                micro_batches_td, indices = rearrange_micro_batches(batch=mini_batch.batch, max_token_len=max_token_len)
                
                # Wrap TensorDicts in DataProto
                micro_batches = [
                    DataProto(batch=mb, meta_info=mini_batch.meta_info)
                    for mb in micro_batches_td
                ]
                
                # Add non_tensor_batch data to each micro_batch
                _add_non_tensor_batch_to_micro_batches(micro_batches, mini_batch, indices=indices)
            total_seqlen = max_token_len
        else:
            assert micro_batch_size is not None, (
                "micro_batch_size is needed to be passed in when not using dynamic batch size"
            )
            # Split the entire DataProto (including non_tensor_batch)
            micro_batches = mini_batch.split(micro_batch_size)
            
            # Ensure non_tensor_batch elements are lists if they are object arrays
            # Note: old_inputs/old_logits can be either numpy arrays or torch tensors
            for mb in micro_batches:
                if "old_inputs" in mb.non_tensor_batch:
                    val = mb.non_tensor_batch["old_inputs"]
                    if isinstance(val, np.ndarray):
                        # Convert object array to list (elements can be torch tensors or numpy arrays)
                        mb.non_tensor_batch["old_inputs"] = val.tolist()
                if "old_logits" in mb.non_tensor_batch:
                    val = mb.non_tensor_batch["old_logits"]
                    if isinstance(val, np.ndarray):
                        # Convert object array to list (elements can be torch tensors or numpy arrays)
                        mb.non_tensor_batch["old_logits"] = val.tolist()

            seq_len = micro_batches[0].batch["input_ids"].shape[1]
            total_seqlen = micro_batch_size * seq_len
        # compute input shapes for pp stages
        n_micro_batch = len(micro_batches)
        # logger.info(f"[Data] [forward_backward_batch] Total micro batches: {n_micro_batch}, batch size list: {[mb.batch['input_ids'].size(0) for mb in micro_batches]}")

        forward_backward_func = get_forward_backward_func()

        def loss_func(output, data, meta_info):
            # For memory efficiency
            # We move calculation of entropy to compute_log_probs, forward_only == True
            log_probs = None
            entropy = None
            if isinstance(output, dict):
                log_probs = output["log_probs"]
                if "entropy" in output:
                    entropy = output["entropy"]
            else:
                assert isinstance(output, torch.Tensor)
                log_probs = output

            device = log_probs.device
            metrics = {}
            if forward_only:
                if post_process_fn is None:
                    pass
                    # metrics["logits"] = output
                else:
                    stats = post_process_fn(output, data)
                    metrics.update(stats)
                if not calculate_entropy:
                    return torch.tensor(1.0, device=device), metrics

            responses = data["responses"]
            response_length = responses.size(1)
            response_mask = data["response_mask"].to(bool)
            loss_agg_mode = self.config.loss_agg_mode
            # compute policy loss
            log_prob = log_probs[:, -response_length - 1 : -1].contiguous()
            ret_entropy = None
            stats = {}
            if not forward_only:
                old_log_prob = data["old_log_probs"]
                advantages = data["advantages"]

                entropy_coeff = self.config.entropy_coeff
                loss_agg_mode = self.config.loss_agg_mode

                loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                policy_loss_fn = get_policy_loss_fn(loss_mode)

                # Extract pre-computed rollout correction weights if present
                # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                rollout_is_weights = data.get("rollout_is_weights", None)
                pg_loss, pg_metrics = policy_loss_fn(
                    old_log_prob=old_log_prob,
                    log_prob=log_prob,
                    advantages=advantages,
                    response_mask=response_mask,
                    loss_agg_mode=loss_agg_mode,
                    config=self.config,
                    rollout_is_weights=rollout_is_weights,
                )
                stats.update(pg_metrics)

                # Skip if using pure rollout correction mode (metrics already in pg_metrics)
                rollout_log_prob = data.get("rollout_log_probs", None)
                if loss_mode != "rollout_correction" and rollout_log_prob is not None:
                    # Compute metrics using CURRENT policy π_θ vs π_rollout
                    # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                    rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                        log_prob=log_prob,
                        rollout_log_prob=rollout_log_prob,
                        response_mask=response_mask,
                    )
                    stats.update(rollout_corr_metrics)

                stats["actor/pg_loss"] = pg_loss.detach().item()
                policy_loss = pg_loss

            if calculate_entropy:
                entropy = output["entropy"][:, -response_length - 1 : -1].contiguous()
                if not forward_only:
                    entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                    entropy_coeff = meta_info["entropy_coeff"]
                    policy_loss = pg_loss - entropy_coeff * entropy_loss
                else:
                    ret_entropy = entropy

            if forward_only:
                policy_loss = torch.tensor(1.0, device=device)
            else:
                if self.config.use_kl_loss:
                    ref_log_prob = data["ref_log_prob"]
                    # compute kl loss
                    kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                    kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)

                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    metrics["actor/kl_loss"] = kl_loss.detach().item()
                    metrics["actor/kl_coef"] = self.config.kl_loss_coef

                # return loss and stats

            append_to_dict(metrics, stats)
            return policy_loss, [metrics, ret_entropy]

        def forward_step(batch_iter, model, return_schedule_plan: bool = False):
            """
            Args:
                batch_iter: the batch iterator
                model: the model
                return_schedule_plan: whether to return the schedule plan, for 1f1b overlap
            """
            if return_schedule_plan:
                assert self.tf_config.overlap_moe_expert_parallel_comm, (
                    "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
                )
                # TODO: Fix this
                assert not calculate_entropy, "calculate_entropy must be disabled to return the schedule plan"
                from megatron.core.models.gpt.gpt_model import GPTModel

                assert isinstance(model, GPTModel), "model must be a GPTModel"
                assert self.use_fused_kernels, "use_fused_kernels must be enabled to return the schedule plan"
                # TODO: support VLM with MoE
                from verl.models.mcore.model_forward_1f1b_overlap import gptmodel_forward_1f1b_overlap

            # logger.info(f"[Memory] [forward_step] Before data generation (batch): {get_system_memory_info()}")
            batch: DataProto = next(batch_iter)
            batch.batch = batch.batch.contiguous()
            batch.batch = batch.batch.to(get_device_id())

            input_ids = batch.batch["input_ids"]
            attention_mask = batch.batch["attention_mask"].to(bool)
            position_ids = batch.batch["position_ids"]

            unwrapped_model = unwrap_model(model)
            if hasattr(unwrapped_model, "vp_stage"):
                vp_rank = unwrapped_model.vp_stage
            else:
                vp_rank = 0

            multi_modal_inputs = {}
            if "multi_modal_inputs" in batch.batch:
                from verl.utils.model import extract_multi_modal_inputs

                indices = batch.batch.get("multi_modal_inputs_idx", None)
                multi_modal_inputs = extract_multi_modal_inputs(batch.batch["multi_modal_inputs"], indices)
            responses = batch.batch["responses"]
            response_length = responses.size(1)
            label = position_ids.clone()
            label[:, -response_length - 1 : -1] = responses
            label_mask = attention_mask.clone()
            label_mask[:, : -response_length - 1] = False
            label_mask[:, -1] = False
            # logger.info(f"[Memory] [forward_step] After data generation (batch): {get_system_memory_info()}")

            if RouterReplayHelper.is_replay_backward_action(self.tf_config, vp_rank):
                router_instance_list = RouterReplayHelper.get_micro_batch_router_list(self.tf_config, vp_rank)
                for router in router_instance_list:
                    router.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
                # TODO: check if predictive routing replay needs to be set here

            if RouterReplayHelper.is_replay_forward_action(self.tf_config, vp_rank):
                # R3 mode: use routed_experts if available
                layers_topk_idx = batch.batch["routed_experts"]
                set_router_replay_data(layers_topk_idx, attention_mask, self.tf_config, vp_rank)

                if RouterReplayHelper.is_predictive_compute_loss_action(self.tf_config, vp_rank):
                    # logger.info(f"[Memory] [forward_step] Loading predictive data: {get_system_memory_info()}")

                    # Read old_inputs and old_logits from batch (per-sample, may contain None)
                    if "old_inputs" in batch.non_tensor_batch and "old_logits" in batch.non_tensor_batch:
                        set_router_predictive_data(
                            batch.non_tensor_batch["old_inputs"],  # list of ndarray or None
                            batch.non_tensor_batch["old_logits"],  # list of ndarray or None
                            attention_mask,
                            self.tf_config,
                            vp_rank,
                        )

            # logger.info(f"[Memory] [forward_step] After data generation (non_tensor_batch): {get_system_memory_info()}")

            from verl.models.mcore import get_mcore_forward_fn, get_mcore_forward_fused_fn

            if self.use_fused_kernels:
                forward_fn = get_mcore_forward_fused_fn(self.hf_config)
                if return_schedule_plan:
                    forward_fn = gptmodel_forward_1f1b_overlap
                # return dict of [logits, entropy]
                output = forward_fn(
                    model=model,
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    labels=label,
                    labels_mask=label_mask,
                    temperature=temperature,
                    multi_modal_inputs=multi_modal_inputs,
                )
            else:
                forward_fn = get_mcore_forward_fn(self.hf_config)

                def logits_processor(logits, label, label_mask):
                    assert logits.shape[:2] == label.shape[:2]
                    assert label.shape == label_mask.shape
                    logits.div_(temperature)
                    ret = {}
                    if calculate_entropy:
                        logits_bak = logits.clone()
                        # # disable the hint until the fused_kernel is optimized for triton>=3.3
                        # logger.warning_once(
                        #     "For memory-efficient computation, enable fused kernels via "
                        #     "`actor_rollout_ref.model.use_fused_kernels=True`. "
                        #     "The current `clone()` operation ensures correctness but increases memory usage."
                        # )
                        entropy = vocab_parallel_entropy(logits)
                        ret["entropy"] = entropy
                    else:
                        logits_bak = logits
                    log_probs = vocab_parallel_log_probs_from_logits(logits_bak, label)
                    log_probs = log_probs.masked_fill(~label_mask, 0.0)
                    ret["log_probs"] = log_probs
                    return ret

                logits_processor_args = {"label": label, "label_mask": label_mask}
                output = forward_fn(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    multi_modal_inputs=multi_modal_inputs,
                    logits_processor=logits_processor,
                    logits_processor_args=logits_processor_args,
                    data_format="thd" if self.config.megatron.use_remove_padding else "bshd",
                )

            # logger.info(f"[Memory] [forward_step] After model forward: {get_system_memory_info()}")

            if forward_only:
                meta_info = None
            else:
                clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                meta_info = {
                    "clip_ratio": self.config.clip_ratio,
                    "entropy_coeff": self.config.entropy_coeff,
                    "clip_ratio_c": clip_ratio_c,
                }

            if RouterReplayHelper.is_r2_record_action(self.tf_config, vp_rank):
                packed_seq_params = merge_router_topk_indices(
                    attention_mask, input_ids, self.mini_layer_topk_idx_list, self.tf_config, vp_rank
                )
                if RouterReplayHelper.is_predictive_record_action(self.tf_config, vp_rank):
                    # Use downsample_batch_size to reduce memory usage (1 = keep 1 sequence per micro-batch)
                    # logger.info(f"[Memory] [forward_step] Before merging predictive data: {get_system_memory_info()}")
                    # logger.info(f"[Memory] [forward_step] Current mini_layer lists sizes: inputs={len(self.mini_layer_old_inputs_list)}, logits={len(self.mini_layer_old_logits_list)}")

                    merge_router_predictive_data(
                        attention_mask,
                        input_ids,
                        self.mini_layer_old_inputs_list,
                        self.mini_layer_old_logits_list,
                        self.mini_layer_sampled_masks_list,
                        self.tf_config,
                        vp_rank,
                        packed_seq_params=packed_seq_params,
                        downsample_batch_size=self.config.router_replay.predictive_downsample_batch_size,
                        storage_dtype=self.config.router_replay.predictive_storage_dtype,
                    )
                    # logger.info(f"[Memory] [forward_step] After merging predictive routing replay data: {get_system_memory_info()}")

            # Record token_ids if present and logits saving is enabled
            if self.enable_logits_saving and RouterReplay.current_cache_action is not None:
                if "global_token_ids" in batch.batch:
                    global_token_ids = batch.batch["global_token_ids"]
                    # Only record valid tokens (remove padding)
                    valid_ids = torch.masked_select(global_token_ids.to(attention_mask.device), attention_mask)
                    logger.info(f"[Routing Replay] [forward_step] Recording {len(valid_ids)} valid global_token_ids (shape before mask: {global_token_ids.shape})")
                    RouterReplay.record_global_token_ids(valid_ids)
                else:
                    logger.info(f"[Routing Replay] [forward_step] WARNING: global_token_ids not in batch, cannot record!")

            if RouterReplayHelper.is_replay_forward_action(self.tf_config, vp_rank):
                router_instance_list = RouterReplayHelper.get_micro_batch_router_list(self.tf_config, vp_rank)
                for router in router_instance_list:
                    router.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)
                # TODO: check if predictive router replay action needs to be set

            return output, partial(loss_func, data=batch.batch, meta_info=meta_info)

        # batch should be a list of batches inside micro-batches
        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.actor_module))

        # TODO: we may use the new schedule instead
        # for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)
        # logger.info(f"[Memory] [forward_backward_batch] Start {get_system_memory_info()}")
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.actor_module,
                num_microbatches=n_micro_batch,
                seq_length=total_seqlen,  # no use when input_shapes was set
                micro_batch_size=1,  # no use when input_shapes was set
                forward_only=forward_only,
            )
        else:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.actor_module,
                num_microbatches=n_micro_batch,
                seq_length=total_seqlen,  # in use for pp = 1
                micro_batch_size=1,  # in use for pp = 1
                forward_only=forward_only,
            )
        # logger.info(f"[Memory] [forward_backward_batch] End {get_system_memory_info()}")
        
        # CRITICAL: Clean up GPU cache after all layers complete their backward
        # Each layer's predictive_loss.backward() accumulates intermediate tensors
        if self.enable_bias_predictor and RouterReplayHelper.is_predictive_compute_loss_action(self.tf_config):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # logger.info(f"[Memory] Cleaned GPU cache after all predictive backward passes")
        
        # loss_reduces contains the stats returned from loss_func

        if self.has_multi_modal_inputs:
            data.batch.pop("multi_modal_inputs")
            data.batch.pop("multi_modal_inputs_idx")
            data.non_tensor_batch.pop("multi_modal_inputs")

        losses_reduced = {"output": losses_reduced}
        if use_dynamic_bsz:
            losses_reduced["indices"] = indices
        if RouterReplayHelper.is_r2_record_action(self.tf_config):
            if self.tf_config.virtual_pipeline_model_parallel_size is not None:
                # config = self.actor_module[0].module.module.config
                vp_size = len(self.actor_module)
                microbatch_group_size_per_vp_stage = self.tf_config.microbatch_group_size_per_vp_stage
                bs = n_micro_batch
                losses_reduced["mini_layer_topk_idx_tensor"] = reorder_and_merge_vpp_layers(self.mini_layer_topk_idx_list, bs, vp_size, microbatch_group_size_per_vp_stage)
                if RouterReplayHelper.is_predictive_record_action(self.tf_config):
                    # logger.info(f"[Predictive Routing Replay] Merging predictive tensors with VPP size {vp_size}")
                    # logger.info(f"[Predictive Routing Replay] [Debug] mini_layer_old_inputs_list BEFORE reorder: len: {len(self.mini_layer_old_inputs_list)}, first element shape: {self.mini_layer_old_inputs_list[0].shape if len(self.mini_layer_old_inputs_list) > 0 else 'N/A'}")
                    # logger.info(f"[Predictive Routing Replay] [Debug] mini_layer_old_logits_list BEFORE reorder: len: {len(self.mini_layer_old_logits_list)}, first element shape: {self.mini_layer_old_logits_list[0].shape if len(self.mini_layer_old_logits_list) > 0 else 'N/A'}")
                    # logger.info(f"[Predictive Routing Replay] [Debug] mini_layer_sampled_masks_list BEFORE reorder: len: {len(self.mini_layer_sampled_masks_list)}, {self.mini_layer_sampled_masks_list}")
                    
                    # NEW: For list of variable-shape tensors, only reorder list elements
                    # No concat/reshape needed since each sample already has correct shape
                    if vp_size > 1:
                        # Apply VPP schedule reordering to the list order
                        from verl.utils.megatron.router_replay_utils import reorder_list_for_vpp
                        losses_reduced["mini_layer_old_inputs_list"] = reorder_list_for_vpp(self.mini_layer_old_inputs_list, bs, vp_size, microbatch_group_size_per_vp_stage)
                        losses_reduced["mini_layer_old_logits_list"] = reorder_list_for_vpp(self.mini_layer_old_logits_list, bs, vp_size, microbatch_group_size_per_vp_stage)
                    else:
                        losses_reduced["mini_layer_old_inputs_list"] = self.mini_layer_old_inputs_list
                        losses_reduced["mini_layer_old_logits_list"] = self.mini_layer_old_logits_list
                    
                    losses_reduced["mini_layer_sampled_masks_tensor"] = reorder_and_merge_vpp_layers(self.mini_layer_sampled_masks_list, bs, vp_size, microbatch_group_size_per_vp_stage)
                    # logger.info(f"[Predictive Routing Replay] [Debug] mini_layer_old_inputs_list after reorder: len: {len(losses_reduced['mini_layer_old_inputs_list'])}, first element shape: {losses_reduced['mini_layer_old_inputs_list'][0].shape if len(losses_reduced['mini_layer_old_inputs_list']) > 0 else 'N/A'}")
                    # logger.info(f"[Predictive Routing Replay] [Debug] mini_layer_old_logits_list after reorder: len: {len(losses_reduced['mini_layer_old_logits_list'])}, first element shape: {losses_reduced['mini_layer_old_logits_list'][0].shape if len(losses_reduced['mini_layer_old_logits_list']) > 0 else 'N/A'}")
                    # logger.info(f"[Predictive Routing Replay] [Debug] mini_layer_sampled_masks_tensor after reorder: len: {len(losses_reduced['mini_layer_sampled_masks_tensor'])}, {losses_reduced['mini_layer_sampled_masks_tensor']}")
                    # Memory debug
                    # inputs_mb = sum(t.numel() * t.element_size() for t in losses_reduced["mini_layer_old_inputs_list"]) / 1024 / 1024
                    # logits_mb = sum(t.numel() * t.element_size() for t in losses_reduced["mini_layer_old_logits_list"]) / 1024 / 1024
                    # logger.info(f"[Predictive Routing Replay] [Memory] Merged predictive tensors: old_inputs={inputs_mb:.2f}MB, old_logits={logits_mb:.2f}MB, {get_system_memory_info()}")
            else:
                losses_reduced["mini_layer_topk_idx_tensor"] = torch.cat(self.mini_layer_topk_idx_list, dim=0)
                if RouterReplayHelper.is_predictive_record_action(self.tf_config):
                    logger.info(f"[Predictive Routing Replay] Concatenating predictive tensors from mini-batches...")
                    # logger.info(f"[Predictive Routing Replay] [Debug] mini_layer_old_inputs_list BEFORE reorder & merge: len: {len(self.mini_layer_old_inputs_list)}, first element shape: {self.mini_layer_old_inputs_list[0].shape if len(self.mini_layer_old_inputs_list) > 0 else 'N/A'}")
                    # logger.info(f"[Predictive Routing Replay] [Debug] mini_layer_old_logits_list BEFORE reorder & merge: len: {len(self.mini_layer_old_logits_list)}, first element shape: {self.mini_layer_old_logits_list[0].shape if len(self.mini_layer_old_logits_list) > 0 else 'N/A'}")
                    # logger.info(f"[Predictive Routing Replay] [Debug] mini_layer_sampled_masks_list BEFORE reorder & merge: len: {len(self.mini_layer_sampled_masks_list)}, {self.mini_layer_sampled_masks_list}")
                    # NEW: Keep as list since each sample has different num_tokens (variable shapes)
                    # Just concatenate the lists from different micro-batches
                    losses_reduced["mini_layer_old_inputs_list"] = self.mini_layer_old_inputs_list
                    losses_reduced["mini_layer_old_logits_list"] = self.mini_layer_old_logits_list
                    losses_reduced["mini_layer_sampled_masks_tensor"] = torch.cat(self.mini_layer_sampled_masks_list, dim=0)
                    # logger.info(f"[Predictive Routing Replay] [Debug] mini_layer_old_inputs_list after reorder & merge: len: {len(losses_reduced['mini_layer_old_inputs_list'])}, first element shape: {losses_reduced['mini_layer_old_inputs_list'][0].shape if len(losses_reduced['mini_layer_old_inputs_list']) > 0 else 'N/A'}")
                    # logger.info(f"[Predictive Routing Replay] [Debug] mini_layer_old_logits_list after reorder & merge: len: {len(losses_reduced['mini_layer_old_logits_list'])}, first element shape: {losses_reduced['mini_layer_old_logits_list'][0].shape if len(losses_reduced['mini_layer_old_logits_list']) > 0 else 'N/A'}")
                    # logger.info(f"[Predictive Routing Replay] [Debug] mini_layer_sampled_masks_list after reorder & merge: len: {len(losses_reduced['mini_layer_sampled_masks_tensor'])}, {losses_reduced['mini_layer_sampled_masks_tensor']}")
                    # Memory debug
                    # inputs_mb = sum(t.numel() * t.element_size() for t in losses_reduced["mini_layer_old_inputs_list"]) / 1024 / 1024
                    # logits_mb = sum(t.numel() * t.element_size() for t in losses_reduced["mini_layer_old_logits_list"]) / 1024 / 1024
                    # logger.info(f"[Predictive Routing Replay] [Memory] Concatenated predictive tensors: old_inputs={inputs_mb:.2f}MB, old_logits={logits_mb:.2f}MB, {get_system_memory_info()}")
            # Clear mini-batch storage and explicitly free memory
            self.mini_layer_topk_idx_list = []
            if RouterReplayHelper.is_predictive_record_action(self.tf_config):
                # Explicitly clear large data structures
                self.mini_layer_old_inputs_list = []
                self.mini_layer_old_logits_list = []
                self.mini_layer_sampled_masks_list = []
            
            # Force garbage collection to free CPU memory
            import gc
            gc.collect()

        return losses_reduced

    @GPUMemoryLogger(role="megatron actor", logger=logger)
    def update_policy(self, dataloader: Iterable[DataProto], global_step: int = None) -> dict:
        """Update the policy with an iterator of DataProto

        Args:
            dataloader (Iterable[DataProto]): an iterator over the DataProto that returns by ``make_minibatch_iterator``
                The keys of each data batch is described in the make_minibatch_iterator.

        Returns:
            Dict: a dictionary containing the statistics. Note that the statistics are only valid in the last pp stage
            and users have to combine the output in each dp rank manually.

        """
        metrics = {}
        # step used for frequency and naming; prefer external global_step if provided
        step_for_save = self.training_step if global_step is None else global_step
        if self.use_torch_profiler and self.prof and self.prof.enable:
            self.prof.start()
        for mini_step, data in enumerate(dataloader):
            # logger.info(f"[Memory] [update_policy] Mini step {mini_step} START (after dataloader.next()): {get_system_memory_info()}")
            logger.info(f"[Data] [update_policy] Mini step {mini_step}, batch size: {data.batch['input_ids'].size(0)}")

            # OPTIMIZATION: Mini-step 0 uses SKIP_PREDICTIVE, doesn't need old_inputs/old_logits
            # Remove them to save memory during forward/backward
            if mini_step == 0 and self.config.router_replay.enable_bias_predictor:
                if "old_inputs" in data.non_tensor_batch or "old_logits" in data.non_tensor_batch:
                    # logger.info(f"[Memory] [update_policy] Mini-step 0: Removing old_inputs/old_logits to save memory (SKIP_PREDICTIVE doesn't need them)")
                    # logger.info(f"[Memory] Before removal: {get_system_memory_info()}")
                    data.non_tensor_batch.pop("old_inputs", None)
                    data.non_tensor_batch.pop("old_logits", None)
                    import gc
                    gc.collect()
                    # logger.info(f"[Memory] After removal and GC: {get_system_memory_info()}")
            
            should_save = self.enable_logits_saving and (step_for_save % self.save_frequency == 0)
            if should_save:
                RouterReplay.set_cache_action(RouterReplayCacheAction.TRAINING)
            else:
                RouterReplay.clear_cache_action()

            if self.config.router_replay.mode in ["R2", "R3"]:
                RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
                # Set predictive action based on ministep
                if self.config.router_replay.enable_bias_predictor:
                    if mini_step == 0:  # First ministep: skip predictive loss
                        RouterReplay.set_global_predictive_action(RouterPredictiveAction.SKIP_PREDICTIVE)
                    else:  # Later ministeps: compute predictive loss
                        assert mini_step <= 1, "Only 2 ministeps supported for now"
                        RouterReplay.set_global_predictive_action(RouterPredictiveAction.COMPUTE_PREDICTIVE_LOSS)

            self.actor_optimizer.zero_grad()
            # use use_contiguous_buffers_in_local_ddp and no overlap_dp_param_comm
            for chunk in self.actor_module:
                # if use distributed optimizer, zero grad buffer will be handled by optimizer
                chunk.zero_grad_buffer()

            calculate_entropy = self.config.entropy_coeff != 0
            if data.meta_info.get("micro_batch_size", None) is not None:
                micro_batch_size = data.meta_info["micro_batch_size"]
            else:
                micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
            max_token_len = None
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.config.megatron.context_parallel_size
            metric_micro_batch = self.forward_backward_batch(
                data,
                calculate_entropy=calculate_entropy,
                use_dynamic_bsz=self.config.use_dynamic_bsz,
                micro_batch_size=micro_batch_size,
                max_token_len=max_token_len,
                mini_batch_size=self.config.ppo_mini_batch_size,
            )
            metric_micro_batch = metric_micro_batch["output"]
            for metric in metric_micro_batch:
                # Note that o[0] is metrics, o[1] is entropy, o[2] is response_mask
                append_to_dict(metrics, metric[0])  # append the metric from this micro-batch to global metrics.

            # logger.info(f"[Memory] [update_policy] Before optimizer.step(): {get_system_memory_info()}")
            # if torch.cuda.is_available():
            #     gpu_mem = torch.cuda.memory_allocated() / (1024**3)
            #     gpu_reserved = torch.cuda.max_memory_reserved() / (1024**3)
            #     logger.info(f"[GPU Memory] Before optimizer.step() - Allocated: {gpu_mem:.2f}GB, Reserved: {gpu_reserved:.2f}GB")
            
            update_successful, grad_norm, num_zeros_in_grad = self.actor_optimizer.step()
            
            # logger.info(f"[Memory] [update_policy] After optimizer.step(): {get_system_memory_info()}, grad_norm={grad_norm}")
            # if torch.cuda.is_available():
            #     gpu_mem = torch.cuda.memory_allocated() / (1024**3)
            #     gpu_reserved = torch.cuda.max_memory_reserved() / (1024**3)
            #     logger.info(f"[GPU Memory] After optimizer.step() - Allocated: {gpu_mem:.2f}GB, Reserved: {gpu_reserved:.2f}GB")
            
            data = {"actor/grad_norm": grad_norm}
            append_to_dict(metrics, data)

            # 在 optimizer.step 后，重新记录 router_weights 以捕获更新后的参数
            if should_save:
                from megatron.core.transformer.moe.router import TopKRouter
                for module in self.actor_module:
                    for name, layer in module.named_modules():
                        if isinstance(layer, TopKRouter):
                            layer_idx = layer.layer_number
                            RouterReplay.logits_cache["router_weights"][layer_idx] = layer.weight.detach().cpu().contiguous()
                            logger.info(f"[Routing Replay] [update_policy] Post-step: Updated router_weights for layer {layer_idx}, shape={layer.weight.shape}")
                        else:
                            # logger.info(f"[Routing Replay] [update_policy] Module {name} is not TopKRouter, skipping.")
                            pass

            if update_successful:
                # allgather already execute in optimizer.step in new megatron
                pass
            else:
                raise NotImplementedError
            if self.use_torch_profiler and self.prof and self.prof.enable:
                self.prof.step()

            if self.config.router_replay.mode in ["R2", "R3"]:
                # logger.info(f"[Memory] [update_policy] Before clear router replay: {get_system_memory_info()}")
                RouterReplay.clear_global_router_replay_action()
                RouterReplay.clear_global_indices()
                # Clear predictive action after each ministep
                if self.config.router_replay.enable_bias_predictor:
                    RouterReplay.clear_global_predictive_action()
                    # logger.info(f"[Memory] [update_policy] Before clear_global_predictive_data: {get_system_memory_info()}")
                    RouterReplay.clear_global_predictive_data()
                    # logger.info(f"[Memory] [update_policy] After clear_global_predictive_data: {get_system_memory_info()}")

            # Save logits cache for this mini step (only when should_save)
            if should_save:
                logits_cache = RouterReplay.get_and_clear_logits_cache()

                if should_save and (logits_cache["compute_log_prob"] or logits_cache["training"]):
                    if mpu.get_tensor_model_parallel_world_size() > 1:
                        logits_cache = RouterReplayLogitsSaver.gather_logits_from_tp_group(logits_cache)

                    if mpu.get_tensor_model_parallel_rank() == 0:
                        if mpu.get_data_parallel_world_size() > 1:
                            logits_cache = RouterReplayLogitsSaver.gather_logits_from_dp_group(logits_cache)

                        if mpu.get_data_parallel_rank() == 0:
                            logger.info(
                                f"[Routing Replay] Logits cache sizes before save - compute_log_prob: {len(logits_cache['compute_log_prob'])}, "
                                f"training: {len(logits_cache['training'])}, "
                                f"global_token_ids: {len(logits_cache.get('global_token_ids', []))}"
                            )
                            step_name = f"training_{step_for_save}_mini{mini_step}"
                            self.logits_saver.save_logits_async(logits_cache, step_name)
                            logger.info(f"[Routing Replay] Scheduled async save for training step {step_for_save}, mini {mini_step}")
                else:
                    logger.debug(f"[Routing Replay] Skipping save for training step {step_for_save}, mini {mini_step} (frequency={self.save_frequency})")
            
            # logger.info(f"[Memory] [update_policy] Mini step {mini_step} END: {get_system_memory_info()}")
            # if torch.cuda.is_available():
            #     gpu_mem = torch.cuda.memory_allocated() / (1024**3)
            #     logger.info(f"[GPU Memory] Mini step {mini_step} END - Allocated: {gpu_mem:.2f}GB")

        # Increment training step after all mini steps
        if self.enable_logits_saving:
            if global_step is None:
                self.training_step += 1
            RouterReplay.clear_cache_action()
        
        # Collect and log predictive metrics for wandb
        if self.enable_bias_predictor:
            predictive_metrics = RouterReplay.get_and_clear_predictive_metrics()
            if predictive_metrics:
                for key, value in predictive_metrics.items():
                    metrics[f"router/{key}"] = value
                logger.info(f"[Predictive Routing Replay] Predictive metrics: {predictive_metrics}")

        # add empty cache after each compute
        if self.use_torch_profiler and self.prof and self.prof.enable:
            self.prof.stop_and_save()
            self.prof.stop_trace()
        get_torch_device().empty_cache()
        return metrics
