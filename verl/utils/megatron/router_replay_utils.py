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
Router Replay Utilities
Utilities for handling router replay functionality in Megatron models.
"""
import os

import logging

import numpy as np
import warnings
from typing import Optional

import torch

from verl.utils.memory_utils import get_system_memory_info

try:
    import psutil
except ImportError:
    psutil = None

try:
    from megatron.core.pipeline_parallel.utils import is_vp_first_stage, is_vp_last_stage
except ImportError:
    warnings.warn("NPU not support router replay for now.", stacklevel=2)
    pass

from megatron.core import parallel_state as mpu
from megatron.core.pipeline_parallel.schedules import get_schedule_table
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region, scatter_to_sequence_parallel_region
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from verl.models.mcore.util import postprocess_packed_seqs, preprocess_packed_seqs
from verl.utils.device import get_device_name
from verl.utils.megatron.router_replay_patch import RouterReplay, RouterReplayAction

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


def _get_system_memory_info():
    """Get system memory information (total and available) in GB."""
    if psutil is None:
        return "N/A (psutil not available)"
    try:
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024 ** 3)
        available_gb = mem.available / (1024 ** 3)
        used_gb = mem.used / (1024 ** 3)
        return f"System: {used_gb:.2f}GB/{total_gb:.2f}GB used, {available_gb:.2f}GB available"
    except Exception as e:
        return f"N/A (error: {e})"


# from megatron.core.transformer.transformer_block import get_num_layers_to_build
def get_num_layers_to_build(
    config: TransformerConfig, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
) -> int:
    """
    Determine the number of transformer layers to build for the current pipeline stage.
    Args:
        config (TransformerConfig): Configuration object containing transformer model parameters.
        vp_stage (Optional[int]): Virtual pipeline stage number.
        pp_rank (Optional[int]): Pipeline parallel rank.

    Returns:
        int: The number of layers to be built for the current pipeline stage.
    """
    # If we have a custom PP layout, straightforwardly
    # return the number of decoders in the layout array.
    if hasattr(config, "pipeline_model_parallel_layout") and config.pipeline_model_parallel_layout is not None:
        from megatron.core.transformer.enums import LayerType

        return config.pipeline_model_parallel_layout.get_num_layers_to_build(
            layer_type=LayerType.decoder, vp_stage=vp_stage
        )

    # Fallback for legacy tests.
    if pp_rank is None:
        pp_rank = mpu.get_pipeline_model_parallel_rank()

    is_first_pp_stage = pp_rank == 0
    is_last_pp_stage = pp_rank == config.pipeline_model_parallel_size - 1

    if config.num_layers_in_first_pipeline_stage is not None or config.num_layers_in_last_pipeline_stage is not None:
        assert not (config.account_for_embedding_in_pipeline_split or config.account_for_loss_in_pipeline_split), (
            " \
        Does not support standalone embedding stage and standalone loss stage with uneven pp"
        )
        # Number of layers to distribute over rest of pipeline stages
        layers_to_distribute = config.num_layers
        # Number of pipeline stages left for distributing transformer layers
        pipeline_stages_left = config.pipeline_model_parallel_size

        # If the uneven first (last) pipeline stage is enabled, remove the specified number
        # of layers to calculate the number of layers on each middle pipeline stage.
        if config.num_layers_in_first_pipeline_stage is not None:
            layers_to_distribute -= config.num_layers_in_first_pipeline_stage
            pipeline_stages_left -= 1

        if config.num_layers_in_last_pipeline_stage is not None:
            layers_to_distribute -= config.num_layers_in_last_pipeline_stage
            pipeline_stages_left -= 1

        # If pp_size <= 2, we do not have any intermediate pipeline stages, and we do not
        # need to check if the left over layers are divisible by the left over stages.
        if pipeline_stages_left > 0:
            assert layers_to_distribute % pipeline_stages_left == 0, (
                "With uneven pipelineing the left over layers must be divisible by left over stages"
            )
            num_layers_per_pipeline_rank = layers_to_distribute // pipeline_stages_left
        else:
            num_layers_per_pipeline_rank = 0

        # If the uneven first (last) pipeline stage is enabled, return the specified number
        # of layers for all virtual pipeline parallel stages within the first (last) pipeline
        # parallel stage.

        if is_first_pp_stage and config.num_layers_in_first_pipeline_stage is not None:
            num_layers_per_pipeline_rank = config.num_layers_in_first_pipeline_stage

        if is_last_pp_stage and config.num_layers_in_last_pipeline_stage is not None:
            num_layers_per_pipeline_rank = config.num_layers_in_last_pipeline_stage
    else:
        # Include the embedding layer and loss layer into pipeline parallelism partition
        num_layers = config.num_layers
        if config.account_for_embedding_in_pipeline_split:
            num_layers += 1

        if config.account_for_loss_in_pipeline_split:
            num_layers += 1

        assert num_layers % config.pipeline_model_parallel_size == 0, (
            "num_layers should be divisible by pipeline_model_parallel_size"
        )
        num_layers_per_pipeline_rank = num_layers // config.pipeline_model_parallel_size

    vp_size = config.virtual_pipeline_model_parallel_size
    if vp_size is not None and config.pipeline_model_parallel_size > 1:
        # Interleaved pipeline parallelism:
        # Number of layers in each model chunk is the number of layers in the stage,
        # divided by the number of model chunks in a stage.
        # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
        # layers to stages like (each list is a model chunk):
        # Stage 0: [0]  [2]  [4]  [6]
        # Stage 1: [1]  [3]  [5]  [7]
        # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
        # layers to stages like (each list is a model chunk):
        # Stage 0: [0, 1]  [4, 5]
        # Stage 1: [2, 3]  [6, 7]

        assert num_layers_per_pipeline_rank % vp_size == 0, (
            f"num_layers_per_pipeline_rank {num_layers_per_pipeline_rank} \
            should be divisible by vp_size {vp_size}"
        )
        num_layers_per_virtual_stage = num_layers_per_pipeline_rank // vp_size

        num_layers_to_build = num_layers_per_virtual_stage

    else:
        # Non-interleaved pipeline parallelism:
        # Each stage gets a contiguous set of layers.
        num_layers_to_build = num_layers_per_pipeline_rank

    # The embedding (or loss) layer cannot function as a standalone transformer layer
    # Reduce the number of layers to construct by 1 on the first (or last) stage if the
    # embedding (or loss) layer is included in the pipeline parallelism partition and placement.
    if config.account_for_embedding_in_pipeline_split:
        if is_vp_first_stage(vp_stage, vp_size) and is_first_pp_stage:
            num_layers_to_build -= 1
            assert num_layers_to_build >= 0, "Not enough layers in the first virtual pipeline stage"

    if config.account_for_loss_in_pipeline_split:
        if is_vp_last_stage(vp_stage, vp_size) and is_last_pp_stage:
            num_layers_to_build -= 1
            assert num_layers_to_build >= 0, "Not enough layers in the last virtual pipeline stage"

    return num_layers_to_build


def merge_router_topk_indices(attention_mask, input_ids, mini_layer_topk_idx_list, tf_config, vp_rank=None, packed_seq_params=None):
    """
    Merge recorded router top-k indices across sequence-parallel ranks for all router instances,
    then pack/unpack them to align with the original (batch, seq_len) layout and append the result.

    Args:
        attention_mask (torch.Tensor): Attention mask of shape [batch_size, seq_len]. Used to determine
            the valid token positions during pack/unpack.
        input_ids (torch.Tensor): Input token IDs of shape [batch_size, seq_len]. Used together with
            attention_mask for sequence packing/unpacking.
        mini_layer_topk_idx_list (list): A Python list to which the merged top-k indices tensor will be appended.
        tf_config: Megatron/Transformer engine configuration object. Used to locate router instances for
            the current micro-batch.
        vp_rank (Optional[int]): Virtual pipeline stage rank override. If None, the current VP rank from
            Megatron parallel state will be used.

    Returns:
        None: The function has side effects only; it appends a tensor of shape
        [1, dynamic_bs_all, layer_num, topk] to mini_layer_topk_idx_list.
    """
    with torch.no_grad():
        print(f"Packing router top-k indices for vp_rank={vp_rank}")
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        layers_topk_idx = []
        for router in router_instances_list:
            layers_topk_idx.append(router.recorded_topk_idx.to(torch.uint8))  # dynamic_bs, topk

        # layer_num, dynamic_bs, topk  -> dynamic_bs, layer_num, topk
        # print(f"Shape of layers_topk_idx before gather: {layers_topk_idx[0].shape}, total layers: {len(layers_topk_idx)}")
        layers_topk_idx = torch.stack(layers_topk_idx).permute(1, 0, 2).to(device_name)
        # dynamic_bs, layer_num, topk -> 1, dynamic_bs_all, layer_num, topk
        layers_topk_idx = (
            gather_from_sequence_parallel_region(layers_topk_idx, tensor_parallel_output_grad=False)
            .unsqueeze(0)
            .contiguous()
        )
        # print(f"Shape of layers_topk_idx after gather: {layers_topk_idx.shape}")

        batch_size, seq_len = attention_mask.shape[:2]
        if packed_seq_params is None:
            _, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=True)
        layers_topk_idx = postprocess_packed_seqs(
            layers_topk_idx, packed_seq_params, attention_mask, batch_size, seq_len, post_process=True
        )
        # print(f"Shape of layers_topk_idx after postprocess: {layers_topk_idx.shape}")
        # Move to CPU and explicitly delete GPU tensor
        cpu_topk_idx = layers_topk_idx.cpu()
        mini_layer_topk_idx_list.append(cpu_topk_idx)
        
        # Explicitly delete GPU tensors to free memory
        del layers_topk_idx
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Clear recorded topk indices from router instances to free GPU memory
        for router in router_instances_list:
            router.recorded_topk_idx = None

    return packed_seq_params

def set_router_replay_data(layers_topk_idx, attention_mask, tf_config, vp_rank=None):
    """
    Scatter the packed router top-k indices back to sequence-parallel ranks and update each local
    RouterReplay instance with target indices for replay mode.

    This function prepares the per-layer, per-sample top-k routing decisions (recorded during an earlier
    forward) so that subsequent replay passes can follow exactly the same routing.

    Args:
        layers_topk_idx (torch.Tensor): Router top-k indices with shape [bs, max_seq_len, layer_num, topk].
            This should be the merged output produced by merge_router_topk_indices.
        attention_mask (torch.Tensor): Attention mask [batch_size, seq_len] used for pack/unpack alignment.
        tf_config: Megatron/Transformer engine configuration object.
        vp_rank (Optional[int]): Virtual pipeline stage rank override. If None, the current VP rank from
            Megatron parallel state will be used.

    Returns:
        None: The function updates internal RouterReplay instances in-place.
    """
    with torch.no_grad():
        layers_topk_idx_rmpad, _ = preprocess_packed_seqs(layers_topk_idx, attention_mask, pre_process=True)
        layers_topk_idx_rmpad = layers_topk_idx_rmpad.contiguous()  # 1, dynamic_bs_all, layer_num, topk

        # 1, dynamic_bs_split, layer_num, topk
        layers_topk_idx_rmpad_split = scatter_to_sequence_parallel_region(
            layers_topk_idx_rmpad.to(device_name).squeeze(dim=0)
        ).unsqueeze(dim=0)

        # dynamic_bs_split, layer_num, topk -> layer_num, dynamic_bs_split, topk
        layers_topk_idx_reshape = layers_topk_idx_rmpad_split.permute(0, 2, 1, 3).squeeze(
            dim=0
        )  # layer_num, dynamic_bs_all, topk
        local_rank_info = get_current_rank_layer_info(tf_config, vp_rank)
        offset, _ = local_rank_info["start"], local_rank_info["end"]
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        for i, router in enumerate(router_instances_list):
            router.set_target_indices(layers_topk_idx_reshape[i + offset].to(torch.int64))


@torch.no_grad()
def merge_router_predictive_data(
    attention_mask,
    input_ids,
    mini_layer_old_inputs_list,
    mini_layer_old_logits_list,
    mini_layer_sampled_masks_list,
    tf_config,
    vp_rank=None,
    packed_seq_params=None,
    downsample_batch_size=None,
    storage_dtype='bf16'
):
    # TODO: check implementation correctness
    """
    Args:
        downsample_batch_size: Number of sequences to keep per micro-batch. Keeps the first N sequences.
            Set to None to keep all sequences (no downsampling).
        storage_dtype: Data type for storage ('fp32', 'bf16', 'fp16'). Lower precision saves memory.
    
    Returns:
        sampled_indices: Tensor of sampled batch indices (0 to downsample_batch_size-1, or 0 to batch_size-1 if no downsampling).
    """
    # CRITICAL: Force synchronization and cleanup to prevent resource accumulation and async operation backlog
    # print(f"[Predictive Routing Replay] [Debug] Synchronizing CUDA before merge...")
    # torch.cuda.synchronize()
    torch.cuda.empty_cache()
    # gpu_allocated_before = torch.cuda.memory_allocated() / (1024**3)
    # print(f"[Predictive Routing Replay] [Debug] GPU Memory before merge: {gpu_allocated_before:.2f}GB")
    
    print(f"Merging router predictive data...")
    print(f"Packing router old_inputs & old_logits for vp_rank={vp_rank}, downsample_batch_size={downsample_batch_size}, storage_dtype={storage_dtype}")
    router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
    layers_old_inputs = []
    layers_old_logits = []
    for router in router_instances_list:
        layers_old_inputs.append(router.recorded_old_inputs)  # dynamic_bs, hidden_size
        layers_old_logits.append(router.recorded_old_logits)  # dynamic_bs, num_experts

    # layer_num, dynamic_bs, hidden_size  -> dynamic_bs, layer_num, hidden_size
    # layer_num, dynamic_bs, num_experts  -> dynamic_bs, layer_num, num_experts
    # print(f"[Predictive Routing Replay] [Debug] Shape of layers_old_inputs before gather: {layers_old_inputs[0].shape}, sum: {layers_old_inputs[0].sum()}, total layers: {len(layers_old_inputs)}")
    # print(f"[Predictive Routing Replay] [Debug] Shape of layers_old_logits before gather: {layers_old_logits[0].shape}, sum: {layers_old_logits[0].sum()}, total layers: {len(layers_old_logits)}")
    
    # CHECKPOINT 1: Before torch.stack
    # print(f"[Predictive Routing Replay] [Debug] CHECKPOINT 1: Before torch.stack, device={device_name}")
    # torch.cuda.synchronize()
    
    layers_old_inputs = torch.stack(layers_old_inputs)
    # print(f"[Predictive Routing Replay] [Debug] CHECKPOINT 2: After stack inputs, shape={layers_old_inputs.shape}")
    
    layers_old_inputs = layers_old_inputs.permute(1, 0, 2)
    # print(f"[Predictive Routing Replay] [Debug] CHECKPOINT 3: After permute inputs")
    
    layers_old_inputs = layers_old_inputs.to(device_name)
    # print(f"[Predictive Routing Replay] [Debug] CHECKPOINT 4: After to(device) inputs, GPU={torch.cuda.memory_allocated()/(1024**3):.2f}GB")
    
    layers_old_logits = torch.stack(layers_old_logits).permute(1, 0, 2).to(device_name)
    # print(f"[Predictive Routing Replay] [Debug] CHECKPOINT 5: After processing logits, GPU={torch.cuda.memory_allocated()/(1024**3):.2f}GB")

    # Save original shapes before concat
    hidden_size = layers_old_inputs.shape[-1]
    num_experts = layers_old_logits.shape[-1]

    # CHECKPOINT 6: Before torch.cat
    # print(f"[Predictive Routing Replay] [Debug] CHECKPOINT 6: Before torch.cat, inputs={layers_old_inputs.shape}, logits={layers_old_logits.shape}")
    
    # dynamic_bs, layer_num, hidden_size -> 1, dynamic_bs_all, layer_num, hidden_size
    # dynamic_bs, layer_num, num_experts -> 1, dynamic_bs_all, layer_num, num_experts
    layers_merged_tensor = torch.cat([layers_old_inputs, layers_old_logits], dim=-1)
    # print(f"[Predictive Routing Replay] [Debug] CHECKPOINT 7: After torch.cat, merged shape={layers_merged_tensor.shape}")
    
    # CHECKPOINT 8: Before gather (collective communication - all processes must reach here)
    torch.cuda.synchronize()
    # print(f"[Predictive Routing Replay] [Debug] CHECKPOINT 8: Before gather_from_sequence_parallel_region (ALL PROCESSES MUST BE HERE)")
    
    layers_merged_tensor = (
        gather_from_sequence_parallel_region(layers_merged_tensor, tensor_parallel_output_grad=False)
        .unsqueeze(0)
        .contiguous()
    )
    
    # CHECKPOINT 9: After gather
    # torch.cuda.synchronize()
    # print(f"[Predictive Routing Replay] [Debug] CHECKPOINT 9: After gather, shape={layers_merged_tensor.shape}, GPU={torch.cuda.memory_allocated()/(1024**3):.2f}GB")
    
    layers_old_inputs, layers_old_logits = torch.split(layers_merged_tensor, [hidden_size, num_experts], dim=-1)
    # print(f"[Predictive Routing Replay] [Debug] Shape of layers_old_inputs after gather: {layers_old_inputs.shape}, sum: {layers_old_inputs.sum()}")
    # print(f"[Predictive Routing Replay] [Debug] Shape of layers_old_logits after gather: {layers_old_logits.shape}, sum: {layers_old_logits.sum()}")

    # ===== NEW: Keep unpacked format, split by sample without padding =====
    # Shape after gather: [1, total_valid_tokens, layers, hidden/experts]
    # We want to split into list of [num_tokens_i, layers, hidden] per sample
    
    batch_size = attention_mask.shape[0]
    if packed_seq_params is None:
        _, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=True)
    
    # Get token ranges per sample from packed_seq_params
    cu_seqlens = packed_seq_params.cu_seqlens_q_padded  # [bs+1]
    seqlens = attention_mask.sum(dim=1, dtype=torch.int32)  # [bs] - valid tokens per sample
    
    # Split by sample into list (each with different num_tokens)
    layers_old_inputs_list = []
    layers_old_logits_list = []
    total_tokens_before_split = 0
    for i in range(batch_size):
        start_idx = cu_seqlens[i].item()
        num_tokens = seqlens[i].item()
        end_idx = start_idx + num_tokens
        
        # Extract this sample's tokens: [num_tokens_i, layers, hidden]
        sample_inputs = layers_old_inputs[0, start_idx:end_idx, :, :].cpu()
        sample_logits = layers_old_logits[0, start_idx:end_idx, :, :].cpu()
        
        layers_old_inputs_list.append(sample_inputs)
        layers_old_logits_list.append(sample_logits)
        total_tokens_before_split += num_tokens
    
    # print(f"[Predictive Routing Replay] [Debug] Split into {len(layers_old_inputs_list)} samples without padding")
    # print(f"[Predictive Routing Replay] [Debug] Sample shapes: first={layers_old_inputs_list[0].shape}")
    # print(f"[Predictive Routing Replay] [Debug] Total tokens: {total_tokens_before_split}")
    
    # Memory usage debug info BEFORE downsampling
    # inputs_size_mb = sum(t.numel() * t.element_size() for t in layers_old_inputs_list) / 1024 / 1024
    # logits_size_mb = sum(t.numel() * t.element_size() for t in layers_old_logits_list) / 1024 / 1024
    # print(f"[Predictive Routing Replay] [Memory] BEFORE downsample - total size: inputs {inputs_size_mb:.2f} MB, logits {logits_size_mb:.2f} MB, {_get_system_memory_info()}")

    # Batch-level downsampling to reduce memory usage (by selecting first N samples)
    bs = len(layers_old_inputs_list)
    
    if downsample_batch_size is not None and downsample_batch_size >= bs:
        downsample_batch_size = None  # No downsampling needed

    if downsample_batch_size is None:
        # No downsampling needed - keep all batches
        downsample_mask = torch.ones((bs,), dtype=torch.bool, device='cpu')
        layers_old_inputs_sampled = layers_old_inputs_list
        layers_old_logits_sampled = layers_old_logits_list
        # print(f"[Predictive Routing Replay] [Downsample] No downsampling: batch_size ({bs}) <= downsample_batch_size ({downsample_batch_size})")
    else:
        # uniformly sample first N sequences
        step = bs // downsample_batch_size
        sampled_indices = [i * step for i in range(downsample_batch_size)]
        downsample_mask = torch.zeros((bs,), dtype=torch.bool, device='cpu')
        downsample_mask[sampled_indices] = True
        layers_old_inputs_sampled = [layers_old_inputs_list[i] for i in sampled_indices]
        layers_old_logits_sampled = [layers_old_logits_list[i] for i in sampled_indices]
        
        # inputs_size_mb_after = sum(t.numel() * t.element_size() for t in layers_old_inputs_sampled) / 1024 / 1024
        # logits_size_mb_after = sum(t.numel() * t.element_size() for t in layers_old_logits_sampled) / 1024 / 1024
        # print(f"[Predictive Routing Replay] [Downsample] Batch downsampling: kept first {downsample_batch_size}/{bs} sequences")
        # print(f"[Predictive Routing Replay] [Memory] AFTER downsample - inputs: {inputs_size_mb_after:.2f} MB (saved {inputs_size_mb - inputs_size_mb_after:.2f} MB), logits: {logits_size_mb_after:.2f} MB, {_get_system_memory_info()}")

    # Lower precision storage to save memory (convert dtype per sample)
    dtype_map = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}
    target_dtype = dtype_map.get(storage_dtype, torch.bfloat16)

    if len(layers_old_inputs_sampled) > 0 and target_dtype != layers_old_inputs_sampled[0].dtype:
        total_size_before = sum(t.numel() * t.element_size() for t in layers_old_inputs_sampled) / 1024 / 1024
        total_size_before += sum(t.numel() * t.element_size() for t in layers_old_logits_sampled) / 1024 / 1024
        
        for i in range(len(layers_old_inputs_sampled)):
            layers_old_inputs_sampled[i] = layers_old_inputs_sampled[i].to(target_dtype)
            layers_old_logits_sampled[i] = layers_old_logits_sampled[i].to(target_dtype)
        
        total_size_after = sum(t.numel() * t.element_size() for t in layers_old_inputs_sampled) / 1024 / 1024
        total_size_after += sum(t.numel() * t.element_size() for t in layers_old_logits_sampled) / 1024 / 1024
        # print(f"[Predictive Routing Replay] [Memory] Reduced precision to {target_dtype}: {total_size_before:.2f}MB → {total_size_after:.2f}MB (saved {total_size_before - total_size_after:.2f} MB), {_get_system_memory_info()}")

    # Append per-sample arrays to mini-batch lists
    for i in range(len(layers_old_inputs_sampled)):
        mini_layer_old_inputs_list.append(layers_old_inputs_sampled[i])
        mini_layer_old_logits_list.append(layers_old_logits_sampled[i])
    mini_layer_sampled_masks_list.append(downsample_mask)

    # Explicitly delete GPU tensors to free memory immediately
    # This ensures GPU memory is released before the next micro-batch
    # Note: layers_old_inputs_sampled and layers_old_logits_sampled are now already appended to mini_layer lists
    del layers_merged_tensor, layers_old_inputs, layers_old_logits
    del layers_old_inputs_list, layers_old_logits_list  # Delete the split-by-sample lists
    # downsample_mask is already appended, no need to delete here
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Calculate cumulative memory
    total_size_mb = sum(t.numel() * t.element_size() for t in mini_layer_old_inputs_list) / 1024 / 1024
    total_size_mb += sum(t.numel() * t.element_size() for t in mini_layer_old_logits_list) / 1024 / 1024
    # print(f"[Predictive Routing Replay] [Memory] Cumulative predictive data in list: {total_size_mb:.2f} MB ({len(mini_layer_old_inputs_list)} micro-batches), {_get_system_memory_info()}")

    # Clear recorded data from router instances to free GPU memory
    for router in router_instances_list:
        router.recorded_old_inputs = None
        router.recorded_old_logits = None

    # FINAL CHECKPOINT: Force synchronization and cleanup after merge completion
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    # gpu_allocated_after = torch.cuda.memory_allocated() / (1024 ** 3)
    # print(f"[Predictive Routing Replay] [Debug] CHECKPOINT FINAL: Merge completed. GPU Memory: {gpu_allocated_before:.2f}GB -> {gpu_allocated_after:.2f}GB (delta: {gpu_allocated_after - gpu_allocated_before:+.2f}GB)")
    # print(f"[Predictive Routing Replay] [Debug] merge_router_predictive_data COMPLETED successfully")


@torch.no_grad()
def set_router_predictive_data(
    old_inputs_list,
    old_logits_list,
    attention_mask,
    tf_config,
    vp_rank=None,
):
    """
    NEW: Simplified version that works with unpacked data (list of variable-shape tensors).
    
    Args:
        layers_old_inputs_list (list): List of tensors, each [num_tokens_i, layers, hidden]
        layers_old_logits_list (list): List of tensors, each [num_tokens_i, layers, num_experts]
        attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]
        tf_config: Transformer config
        vp_rank: Virtual pipeline rank
    """
    # Handle multiple possible formats after Ray serialization
    if isinstance(old_inputs_list, np.ndarray):
        old_inputs_list = list(old_inputs_list)
    if isinstance(old_logits_list, np.ndarray):
        old_logits_list = list(old_logits_list)

    # logger.info(f"old_inputs_list: type {type(old_inputs_list)}, length {len(old_inputs_list) if isinstance(old_inputs_list, list) else 'N/A'}")
    # if isinstance(old_inputs_list, list) and len(old_inputs_list) > 0 and old_inputs_list[0] is not None:
    #     logger.info(f"old_inputs_list[0]: type {type(old_inputs_list[0])}, shape {old_inputs_list[0].shape if hasattr(old_inputs_list[0], 'shape') else 'N/A'}")

    # Filter out None values
    valid_indices = []
    valid_old_inputs = []
    valid_old_logits = []
    for i, (old_input, old_logit) in enumerate(zip(old_inputs_list, old_logits_list)):
        if old_input is not None and old_logit is not None:
            valid_indices.append(i)

            # Handle Ray serialization artifacts
            if isinstance(old_input, list):
                # logger.info(f"[Ray Deserialization] Converting old_input from list to np.ndarray")
                old_input = np.array(old_input, dtype=np.float32)
            if isinstance(old_logit, list):
                # logger.info(f"[Ray Deserialization] Converting old_logit from list to np.ndarray")
                old_logit = np.array(old_logit, dtype=np.float32)

            # Convert numpy array to torch tensor: [num_tokens_i, layers, hidden]
            valid_old_inputs.append(torch.from_numpy(old_input.copy()))
            valid_old_logits.append(torch.from_numpy(old_logit.copy()))
    del old_inputs_list, old_logits_list  # Free memory

    if len(valid_old_inputs) > 0:
        logger.info(f"[Predictive Routing Replay] Loaded {len(valid_old_inputs)} valid samples (shapes: {[t.shape for t in valid_old_inputs[:3]]}...)")

        # Create token-level valid_mask for filtering current input/logits in router
        # valid_mask marks which tokens in the full unpacked batch belong to valid samples
        seq_lens = attention_mask.sum(dim=1, dtype=torch.int32)  # [batch_size]
        cumsum_lens = torch.cumsum(seq_lens, dim=0)  # [batch_size]

        total_valid_tokens = cumsum_lens[-1].item()  # total valid tokens (padding removed)
        valid_mask = torch.zeros(total_valid_tokens, dtype=torch.bool, device=attention_mask.device)

        # Mark tokens from valid samples using tensor operations
        if len(valid_indices) > 0:
            valid_indices_tensor = torch.tensor(valid_indices, device=attention_mask.device, dtype=torch.long)
            # Get start indices: cumsum_lens[i-1] for i > 0, else 0
            start_indices = torch.cat([
                torch.tensor([0], device=attention_mask.device, dtype=torch.long),
                cumsum_lens[:-1]
            ])[valid_indices_tensor]
            # Get end indices: cumsum_lens[i]
            end_indices = cumsum_lens[valid_indices_tensor]

            # Create index ranges for all valid samples
            range_lengths = end_indices - start_indices
            non_zero_mask = range_lengths > 0
            if non_zero_mask.any():
                filtered_starts = start_indices[non_zero_mask]
                filtered_lengths = range_lengths[non_zero_mask]

                # Create all indices at once
                expanded_starts = torch.repeat_interleave(filtered_starts, filtered_lengths)
                range_offsets = torch.cat([
                    torch.arange(length, device=attention_mask.device, dtype=torch.long)
                    for length in filtered_lengths
                ])
                all_indices = expanded_starts + range_offsets
                valid_mask[all_indices] = True
        logger.info(f"[Predictive Routing Replay] Created valid_mask with {valid_mask.sum().item()}/{total_valid_tokens} valid tokens, values: {valid_mask}")

        # Get router instances and compute dtype
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        if router_instances_list and hasattr(router_instances_list[0], 'bias_predictor') and router_instances_list[0].bias_predictor is not None:
            compute_dtype = router_instances_list[0].bias_predictor.weight.dtype
            # print(f"[Memory] Detected bias_predictor weight dtype: {compute_dtype}, {_get_system_memory_info()}")
        else:
            compute_dtype = tf_config.params_dtype
            # print(f"[Memory] Using tf_config.params_dtype: {compute_dtype}, {_get_system_memory_info()}")

        # Concatenate all valid samples into single tensor: [total_valid_tokens, layers, hidden]
        layers_old_inputs_concat = torch.cat([t.to(compute_dtype).to(device_name) for t in valid_old_inputs], dim=0)  # [total_tokens, layers, hidden]
        layers_old_logits_concat = torch.cat([t.to(compute_dtype).to(device_name) for t in valid_old_logits], dim=0)  # [total_tokens, layers, hidden]
        # print(f"[Predictive Routing Replay] [Debug] Concatenated {len(valid_old_inputs)} samples: inputs shape {layers_old_inputs_concat.shape}, logits shape {layers_old_logits_concat.shape}")

        # Scatter to sequence parallel if needed
        if tf_config.sequence_parallel:
            layers_old_inputs_concat = scatter_to_sequence_parallel_region(layers_old_inputs_concat)
            layers_old_logits_concat = scatter_to_sequence_parallel_region(layers_old_logits_concat)

        # Move to CPU to free GPU memory
        layers_old_inputs_concat = layers_old_inputs_concat.cpu()
        layers_old_logits_concat = layers_old_logits_concat.cpu()
        valid_mask = valid_mask.cpu()

        # Set to each router layer with valid_mask (created externally)
        local_rank_info = get_current_rank_layer_info(tf_config, vp_rank)
        offset = local_rank_info["start"]
        for i, router in enumerate(router_instances_list):
            router.set_predictive_data(
                layers_old_inputs_concat[:, i + offset, :].unsqueeze(1).contiguous(),  # [total_valid_tokens, 1, hidden]
                layers_old_logits_concat[:, i + offset, :].unsqueeze(1).contiguous(),  # [total_valid_tokens, 1, num_experts]
                valid_mask=valid_mask  # Token-level mask to filter current_input in router_replay_patch
            )
            logger.info(f"[Predictive Routing Replay] Set layer {i} predictive data with layers_old_inputs_concat shape {layers_old_inputs_concat.shape}, layers_old_logits_concat shape {layers_old_logits_concat.shape}, valid_mask {valid_mask}")

        # Explicitly delete large intermediate tensors to free GPU memory
        del layers_old_inputs_concat, layers_old_logits_concat
        del valid_old_inputs, valid_old_logits
        import gc
        gc.collect()
        # logger.info(f"[Memory] [forward_step] After set_router_predictive_data: {get_system_memory_info()}")

    else:
        logger.warning(f"[Predictive Routing Replay] No valid old_inputs/old_logits found in batch")

def reorder_list_for_vpp(
    micro_batch_list,
    num_microbatches: int,
    vpp_size: int,
    microbatch_group_size_per_vp_stage: int,
) -> list:
    """
    Reorder a list of elements according to VPP schedule (for variable-shape tensors).
    
    This function only reorders the list elements without concatenating them,
    making it suitable for lists of variable-shape tensors.
    
    Args:
        micro_batch_list: List of elements (can be tensors of different shapes)
        num_microbatches (int): Number of microbatches per pipeline stage (bs).
        vpp_size (int): Virtual pipeline parallel size (number of model chunks).
        microbatch_group_size_per_vp_stage (int): Number of consecutive microbatches processed per VPP stage.
    
    Returns:
        list: Reordered list of elements
    """
    if vpp_size <= 1:
        return micro_batch_list
    
    # Build schedule table
    schedule_table = get_schedule_table(num_microbatches, vpp_size, microbatch_group_size_per_vp_stage)
    
    # Group by model_chunk_id
    elements_by_chunk = [[] for _ in range(vpp_size)]
    for vidx, (_mb, chunk_id) in enumerate(schedule_table):
        elements_by_chunk[chunk_id].append(micro_batch_list[vidx])
    
    # Flatten in chunk order
    reordered_list = []
    for chunk_id in range(vpp_size):
        reordered_list.extend(elements_by_chunk[chunk_id])
    
    return reordered_list


def reorder_and_merge_vpp_layers(
    micro_batch_tensor_list,
    num_microbatches: int,
    vpp_size: int,
    microbatch_group_size_per_vp_stage: int,
) -> torch.Tensor:
    """
    Reorder and merge per-VPP layer blocks into a contiguous layer dimension.

    Given a tensor shaped as [bs*vpp_size, max_token_len, layer_num_per_vpp, topk], this function:
    1) Builds the schedule table for virtual microbatches and reorders the first dimension so that entries
       belonging to the same model chunk (VPP stage) become contiguous.
    2) Reshapes and merges the (vpp_size, layer_num_per_vpp) into a single layer dimension, producing
       [bs, max_token_len, layer_num, topk].

    Args:
        micro_batch_tensor_list : the list of Input tensor.
        num_microbatches (int): Number of microbatches per pipeline stage (bs).
        vpp_size (int): Virtual pipeline parallel size (number of model chunks).
        microbatch_group_size_per_vp_stage (int): Number of consecutive microbatches processed per VPP stage.

    Returns:
        torch.Tensor: Output tensor of shape [bs, max_token_len, layer_num, topk].

    Raises:
        ValueError: If input tensor dimensionality or expected sizes do not match.
        RuntimeError: If the computed output shape is unexpected or the schedule length mismatches.
    """
    # 1) Build schedule table: map each virtual_microbatch_id -> (microbatch_id, model_chunk_id)
    schedule_table = get_schedule_table(num_microbatches, vpp_size, microbatch_group_size_per_vp_stage)

    # 2) Group by model_chunk_id to build reorder indices so entries of the same chunk become contiguous along dim 0
    tensor_by_chunk = [[] for _ in range(vpp_size)]
    mini_tensor_list = []

    for vidx, (_mb, chunk_id) in enumerate(schedule_table):
        tensor_by_chunk[chunk_id].append(micro_batch_tensor_list[vidx])

    for chunk_id in range(vpp_size):
        mini_tensor_list.append(torch.cat(tensor_by_chunk[chunk_id], dim=0))

    out = torch.cat(mini_tensor_list, dim=2)
    return out


def get_current_rank_layer_info(tf_config, vp_rank=None):
    # When vp_rank is None, default to the current VP rank (or 0 if VP is disabled).
    """Return the local layer range/count for the current process and the full assignment table.

    Args:
        tf_config: Configuration object used by compute_pipeline_layer_assignment.
        vp_rank (Optional[int]): Explicit virtual pipeline stage rank to query. If None, uses
            mpu.get_virtual_pipeline_model_parallel_rank() when VP is enabled; otherwise 0.

    Returns:
        Tuple[dict, dict]: A tuple of (local_assignment, all_assignments) where local_assignment contains
        keys {"start", "end", "count"} for the current (pp_rank, vp_stage).
    """
    if vp_rank is None:
        vp_rank = 0
    num_layers_to_build = get_num_layers_to_build(tf_config, vp_stage=vp_rank)
    offset = get_transformer_layer_offset(tf_config, vp_stage=vp_rank)
    local = {}
    local["start"] = offset
    local["end"] = offset + num_layers_to_build
    local["count"] = num_layers_to_build
    return local


def pp_gather(local_layers_router_map, tf_config):
    # TODO: Consider non-uniform layer allocation cases.
    """
    Gather local router maps from all PP ranks into a global router map.

    Args:
        local_layers_router_map (torch.Tensor): Local router map of shape
            [bs, max_seq_len, local_num_layers, topk].
        tf_config: Configuration providing pipeline_model_parallel_size.

    Returns:
        torch.Tensor: Global router map of shape [bs, max_seq_len, num_layers, topk] placed on CPU.
    """
    pp_size = tf_config.pipeline_model_parallel_size
    if pp_size <= 1:
        return local_layers_router_map

    pp_group = mpu.get_pipeline_model_parallel_group()
    world_size = torch.distributed.get_world_size(pp_group)
    local_layers_router_map = local_layers_router_map.to(device_name)
    layers_topk_idx_global_list = [
        torch.empty(
            size=local_layers_router_map.shape,
            dtype=local_layers_router_map.dtype,
            device=local_layers_router_map.device,
        )
        for _ in range(world_size)
    ]
    torch.distributed.all_gather(
        tensor=local_layers_router_map,
        tensor_list=layers_topk_idx_global_list,
        group=pp_group,
        async_op=False,
    )
    vp_size = tf_config.virtual_pipeline_model_parallel_size
    if vp_size is not None:
        vpp_router_map_offset = [[] for _ in range(pp_size)]
        for pp_stage in range(pp_size):
            vpp_router_map_offset[pp_stage].append(0)
            for vp_stage in range(vp_size):
                num_layers_to_build = get_num_layers_to_build(tf_config, vp_stage, pp_stage)
                vpp_router_map_offset[pp_stage].append(num_layers_to_build + vpp_router_map_offset[pp_stage][-1])
        layers_topk_idx_global = []
        for vp_stage in range(vp_size):
            for pp_stage in range(pp_size):
                piece = slice(vpp_router_map_offset[pp_stage][vp_stage], vpp_router_map_offset[pp_stage][vp_stage + 1])
                layers_topk_idx_global.append(layers_topk_idx_global_list[pp_stage][:, :, piece, :])
        global_router_map = torch.cat(layers_topk_idx_global, dim=2).to("cpu")
    else:
        global_router_map = torch.cat(layers_topk_idx_global_list, dim=2).to("cpu")

    return global_router_map


class RouterReplayHelper:
    """Helper class to query router replay state and locate local RouterReplay instances."""

    @staticmethod
    def get_micro_batch_router_list(tf_config, vp_rank=None):
        """
        Return the list of RouterReplay instances corresponding to the current micro-batch and local
        (pp_rank, vp_stage) layer range.

        When virtual pipeline (VPP) is enabled, the local range for the PP rank is expanded to include
        all VP stages by multiplying the per-VP count by vp_size. The returned slice is taken from the
        global RouterReplay.router_instances list.

        Args:
            tf_config: Configuration object used to compute layer assignments.
            vp_rank (Optional[int]): Explicit virtual pipeline stage to query. If None, the current VP
                rank from Megatron parallel state is used when available.
        Returns:
            list: A contiguous sublist of RouterReplay.router_instances for the local layer range.
        """
        vp_size = tf_config.virtual_pipeline_model_parallel_size
        if vp_size is not None:
            vp_rank = 0 if vp_rank is None else vp_rank
            offset = 0
            for pre_vp_stage in range(vp_size):
                if pre_vp_stage == vp_rank:
                    break
                num_layers_to_build = get_num_layers_to_build(tf_config, pre_vp_stage)
                offset += num_layers_to_build
        else:
            offset = 0

        num_layers_to_build = get_num_layers_to_build(tf_config, vp_rank)
        router_instances_list = RouterReplay.router_instances[offset : offset + num_layers_to_build]
        return router_instances_list

    @staticmethod
    def is_r2_record_action(tf_config, vp_rank=None) -> bool:
        """Return True if the current router_replay_action is RECORD (R2) for the local router instances.

        This inspects the first local RouterReplay instance's router_replay_action and compares it to
        RouterReplayAction.RECORD.
        """
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        return router_instances_list and router_instances_list[0].router_replay_action == RouterReplayAction.RECORD

    @staticmethod
    def is_replay_forward_action(tf_config, vp_rank=None) -> bool:
        """Return True if the current router_replay_action is REPLAY_FORWARD for the local router instances.

        This inspects the first local RouterReplay instance's router_replay_action and compares it to
        RouterReplayAction.REPLAY_FORWARD.
        """
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        return (
            router_instances_list and router_instances_list[0].router_replay_action == RouterReplayAction.REPLAY_FORWARD
        )

    @staticmethod
    def is_replay_backward_action(tf_config, vp_rank=None) -> bool:
        """Return True if the current router_replay_action is REPLAY_BACKWARD for the local router instances.

        This inspects the first local RouterReplay instance's router_replay_action and compares it to
        RouterReplayAction.REPLAY_BACKWARD.
        """
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        return (
            router_instances_list
            and router_instances_list[0].router_replay_action == RouterReplayAction.REPLAY_BACKWARD
        )

    @staticmethod
    def is_predictive_record_action(tf_config, vp_rank=None) -> bool:
        """Check if current action is RECORD_FOR_PREDICTIVE (log_prob phase).
        
        This inspects the first local RouterReplay instance's predictive_action and compares it to
        RouterPredictiveAction.RECORD_FOR_PREDICTIVE.
        """
        from verl.utils.megatron.router_replay_patch import RouterPredictiveAction

        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        return (
            router_instances_list
            and router_instances_list[0].predictive_action == RouterPredictiveAction.RECORD
        )

    @staticmethod
    def is_predictive_compute_loss_action(tf_config, vp_rank=None) -> bool:
        """Check if current action is COMPUTE_PREDICTIVE_LOSS (training ministep>=1).
        
        This inspects the first local RouterReplay instance's predictive_action and compares it to
        RouterPredictiveAction.COMPUTE_PREDICTIVE_LOSS.
        """
        from verl.utils.megatron.router_replay_patch import RouterPredictiveAction

        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        return (
            router_instances_list
            and router_instances_list[0].predictive_action == RouterPredictiveAction.COMPUTE_PREDICTIVE_LOSS
        )

    @staticmethod
    def is_predictive_skip_action(tf_config, vp_rank=None) -> bool:
        """Check if current action is SKIP_PREDICTIVE (training ministep==0).
        
        This inspects the first local RouterReplay instance's predictive_action and compares it to
        RouterPredictiveAction.SKIP_PREDICTIVE.
        """
        from verl.utils.megatron.router_replay_patch import RouterPredictiveAction

        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        return (
            router_instances_list
            and router_instances_list[0].predictive_action == RouterPredictiveAction.SKIP_PREDICTIVE
        )
