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
Utilities for saving router replay logits to disk.
Handles distributed training scenarios (DP, TP, PP) and asynchronous saving.
"""

import logging
import os
import threading
import torch
from torch import no_grad
from typing import Dict, List, Tuple

from megatron.core import parallel_state as mpu

logger = logging.getLogger(__name__)


class RouterReplayLogitsSaver:
    """
    Handler for saving router replay logits with support for:
    - Distributed training (DP, TP, PP)
    - Asynchronous saving to reduce training latency
    - CPU memory management
    """

    def __init__(self, save_dir: str):
        """
        Args:
            save_dir: Directory to save logits files
        """
        self.save_dir = save_dir
        self.save_threads = []

        # Create directory only on global rank 0 (DP rank 0, TP rank 0, PP rank 0)
        if (mpu.get_data_parallel_rank() == 0 and
            mpu.get_tensor_model_parallel_rank() == 0 and
            mpu.get_pipeline_model_parallel_rank() == 0):
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"Router replay logits will be saved to: {save_dir}")

    def _save_logits_sync(self, logits_data: Dict[str, List[Tuple[int, torch.Tensor]]], step):
        """
        Synchronously save logits data to disk.
        
        Args:
            logits_data: Dict with 'compute_log_prob' and 'training' keys
            step: Current training step (int or str)
        """
        try:
            # Get parallel ranks
            tp_rank = mpu.get_tensor_model_parallel_rank()
            pp_rank = mpu.get_pipeline_model_parallel_rank()

            # Extract global step number from step string
            # step format: "log_prob_5" or "training_5_mini0"
            if isinstance(step, str):
                if step.startswith("log_prob_"):
                    global_step = step.split("_")[2]  # "log_prob_5" -> "5"
                elif step.startswith("training_"):
                    global_step = step.split("_")[1]  # "training_5_mini0" -> "5"
                else:
                    global_step = str(step)
            else:
                global_step = str(step)

            # Create step subdirectory (only on saving rank: DP0, TP0, PP0)
            step_dir = os.path.join(self.save_dir, global_step)
            # Use exist_ok to handle concurrent directory creation across ranks
            os.makedirs(step_dir, exist_ok=True)

            # Prepare filename without dp_rank (since we merge across DP)
            filename = f"{step}_tp{tp_rank}_pp{pp_rank}.pt"
            filepath = os.path.join(step_dir, filename)

            # Convert list of tuples to structured dict
            save_dict = {
                "step": step,
                "tp_rank": tp_rank,
                "pp_rank": pp_rank,
                "dp_world_size": mpu.get_data_parallel_world_size(),
                "compute_log_prob": {},
                "training": {},
                "router_weights": {},
                "global_token_ids": [],  # Direct list/tensor, no phase key
                "predictive_bias": {},  # Bias predictor output (scaled delta_logits)
            }

            # Organize by layer index
            for layer_idx, logits in logits_data.get("compute_log_prob", []):
                if layer_idx not in save_dict["compute_log_prob"]:
                    save_dict["compute_log_prob"][layer_idx] = []
                save_dict["compute_log_prob"][layer_idx].append(logits)

            for layer_idx, logits in logits_data.get("training", []):
                if layer_idx not in save_dict["training"]:
                    save_dict["training"][layer_idx] = []
                save_dict["training"][layer_idx].append(logits)

            for layer_idx, weights in logits_data.get("router_weights", {}).items():
                # Router weights are parameters; we keep the first occurrence
                if layer_idx not in save_dict["router_weights"]:
                    save_dict["router_weights"][layer_idx] = weights

            # Organize predictive_bias (bias predictor output after scaling)
            for layer_idx, bias in logits_data.get("predictive_bias", []):
                if layer_idx not in save_dict["predictive_bias"]:
                    save_dict["predictive_bias"][layer_idx] = []
                save_dict["predictive_bias"][layer_idx].append(bias)

            # Organize global_token_ids (now direct list of tensors, no phase wrapper)
            for ids in logits_data.get("global_token_ids", []):
                save_dict["global_token_ids"].append(ids)

            # Concatenate multiple micro-batches if present
            for phase in ["compute_log_prob", "training", "predictive_bias"]:
                for layer_idx in save_dict[phase]:
                    if len(save_dict[phase][layer_idx]) > 0:
                        save_dict[phase][layer_idx] = torch.cat(save_dict[phase][layer_idx], dim=0)

            if len(save_dict["global_token_ids"]) > 0:
                save_dict["global_token_ids"] = torch.cat(save_dict["global_token_ids"], dim=0)

            # Save to disk
            torch.save(save_dict, filepath)
            logger.info(f"Saved router replay logits to {filepath}")
            logger.info(f"  File contains: compute_log_prob={len(save_dict['compute_log_prob'])} layers, "
                        f"training={len(save_dict['training'])} layers, router_weights={len(save_dict['router_weights'])} layers, "
                        f"global_token_ids={save_dict['global_token_ids'].shape if isinstance(save_dict['global_token_ids'], torch.Tensor) else 0}, "
                        f"predictive_bias={len(save_dict['predictive_bias'])} layers")

        except Exception as e:
            logger.error(f"Failed to save router replay logits for step {step}: {e}")

    def save_logits_async(self, logits_data: Dict[str, List[Tuple[int, torch.Tensor]]], step):
        """
        Asynchronously save logits data to disk to reduce training latency.
        
        Args:
            logits_data: Dict with 'compute_log_prob' and 'training' keys
            step: Current training step (int or str)
        """
        # Debug: log what we're about to save
        logger.info(f"[save_logits_async] Step {step}: "
                    f"compute_log_prob={len(logits_data.get('compute_log_prob', []))} items, "
                    f"training={len(logits_data.get('training', []))} items")

        # Create a deep copy to avoid data corruption during async save
        logits_data_copy = {
            "compute_log_prob": [(idx, tensor.clone()) for idx, tensor in logits_data.get("compute_log_prob", [])],
            "training": [(idx, tensor.clone()) for idx, tensor in logits_data.get("training", [])],
            "router_weights": {idx: tensor.clone() for idx, tensor in logits_data.get("router_weights", {}).items()},
            "global_token_ids": [tensor.clone() for tensor in logits_data.get("global_token_ids", [])],
            "predictive_bias": [(idx, tensor.clone()) for idx, tensor in logits_data.get("predictive_bias", [])],
        }

        logger.info(f"[save_logits_async] After copy: "
                    f"compute_log_prob={len(logits_data_copy['compute_log_prob'])} items, "
                    f"training={len(logits_data_copy['training'])} items")

        # Start save thread
        save_thread = threading.Thread(
            target=self._save_logits_sync,
            args=(logits_data_copy, step),
            daemon=True
        )
        save_thread.start()
        self.save_threads.append(save_thread)

        # Clean up finished threads
        self.save_threads = [t for t in self.save_threads if t.is_alive()]

    def wait_all_saves(self):
        """Wait for all async save operations to complete."""
        for thread in self.save_threads:
            thread.join()
        self.save_threads = []
        logger.info("All router replay logits saved successfully")

    @staticmethod
    @no_grad()
    def gather_logits_from_tp_group(logits_data: Dict[str, List[Tuple[int, torch.Tensor]]]) -> Dict[str, List[Tuple[int, torch.Tensor]]]:
        """
        Note: Router logits are NOT sharded across TP ranks in Megatron MoE.
        
        In Megatron MoE:
        - Router logits shape: [tokens, num_experts]
        - num_experts dimension is NOT split by TP (Tensor Parallel)
        - Experts are split by EP (Expert Parallel), not TP
        - Each TP rank has identical router logits
        
        Therefore, we only need to save on TP rank 0 to avoid duplicate saves.
        No gathering is needed since all TP ranks have the same data.
        
        Args:
            logits_data: Local logits data (identical across TP ranks)
            
        Returns:
            logits_data on TP rank 0, empty dict on other ranks
        """
        tp_world_size = mpu.get_tensor_model_parallel_world_size()
        if tp_world_size == 1:
            return logits_data

        tp_rank = mpu.get_tensor_model_parallel_rank()

        # Router logits are identical across TP ranks, so only return data on rank 0
        # to avoid duplicate saves
        return logits_data if tp_rank == 0 else {"compute_log_prob": [], "training": [], "router_weights": {}, "global_token_ids": [], "predictive_bias": []}

    @staticmethod
    @no_grad()
    def gather_logits_from_dp_group(logits_data: Dict[str, List[Tuple[int, torch.Tensor]]], max_tokens: int = None) -> Dict[str, List[Tuple[int, torch.Tensor]]]:
        """
        Gather logits across data parallel group and concatenate along batch dimension.
        Only DP rank 0 will have the complete data.
        
        This version uses CPU-based gathering to avoid GPU OOM issues.
        Uses torch.distributed.gather_object to collect data directly on CPU.
        
        Args:
            logits_data: Local logits data from this DP rank (tensors on CPU)
            max_tokens: Maximum number of tokens to save globally. If set, each rank contributes
                       max_tokens // dp_world_size tokens.
            
        Returns:
            Gathered logits data (only valid on DP rank 0)
        """
        dp_world_size = mpu.get_data_parallel_world_size()

        # Calculate tokens to keep per rank if max_tokens is set
        tokens_per_rank = None
        if max_tokens is not None:
            tokens_per_rank = max_tokens // dp_world_size
            if tokens_per_rank == 0:
                logger.warning(f"max_tokens {max_tokens} is smaller than dp_world_size {dp_world_size}, setting to 1 per rank")
                tokens_per_rank = 1

        if dp_world_size == 1:
            # If downsampling is requested even with DP=1
            if tokens_per_rank is not None:
                downsampled_data = {
                    "compute_log_prob": [],
                    "training": [],
                    "router_weights": [],
                }
                for phase in ["compute_log_prob", "training", "router_weights"]:
                    for layer_idx, logits in logits_data.get(phase, []):
                        if logits.size(0) > tokens_per_rank:
                            logits = logits[:tokens_per_rank]
                        downsampled_data[phase].append((layer_idx, logits))
                return downsampled_data
            return logits_data

        dp_rank = mpu.get_data_parallel_rank()
        dp_group = mpu.get_data_parallel_group()

        gathered_data = {"compute_log_prob": [], "training": [], "router_weights": {}, "global_token_ids": [], "predictive_bias": []}

        try:

            for phase in ["compute_log_prob", "training", "router_weights", "global_token_ids", "predictive_bias"]:
                # Use appropriate default value based on phase data structure
                default_value = {} if phase == "router_weights" else []
                phase_data = logits_data.get(phase, default_value)

                # Even if empty, participate in collective operation
                # Prepare local data: concatenate micro-batches for each layer
                layer_dict = {}
                token_ids_list = []  # Special handling for token_ids

                if phase == "global_token_ids":
                    # global_token_ids: list of tensors (no tuple wrapper)
                    for tensor in phase_data:
                        token_ids_list.append(tensor)
                elif phase == "router_weights":
                    # router_weights: dict {layer_idx: tensor}
                    for layer_idx, logits in phase_data.items():
                        if layer_idx not in layer_dict:
                            layer_dict[layer_idx] = []
                        # Ensure logits are on CPU and contiguous
                        logits_cpu = logits.cpu().contiguous() if logits.is_cuda else logits.contiguous()
                        layer_dict[layer_idx].append(logits_cpu)
                else:
                    # compute_log_prob, training, predictive_bias: list of (layer_idx, tensor) tuples
                    for layer_idx, logits in phase_data:
                        if layer_idx not in layer_dict:
                            layer_dict[layer_idx] = []
                        # Ensure logits are on CPU and contiguous
                        logits_cpu = logits.cpu().contiguous() if logits.is_cuda else logits.contiguous()
                        layer_dict[layer_idx].append(logits_cpu)

                # Concatenate micro-batches for each layer
                local_layer_data = []

                if phase == "global_token_ids":
                    if token_ids_list:
                        combined_ids = torch.cat(token_ids_list, dim=0)
                        # We treat it as layer -1 or some special key to reuse logic, or just a list of tensors
                        # But wait, gather_object expects list of (layer_idx, tensor).
                        # Let's use a dummy index for token_ids
                        local_layer_data.append(("global_token_ids", combined_ids))
                else:
                    for layer_idx in sorted(layer_dict.keys()):
                        # For router_weights, we only take the first chunk (it's a parameter, doesn't split by micro-batch)
                        # predictive_bias should concatenate like logits (per-token output)
                        if phase == "router_weights":
                            local_logits = layer_dict[layer_idx][0]
                        else:
                            local_logits = torch.cat(layer_dict[layer_idx], dim=0)  # [local_tokens, num_experts]

                        # Apply downsampling per rank if requested
                        # Note: token_ids should also be downsampled if needed, but for now we skip downsampling for simplicity or apply same logic
                        if tokens_per_rank is not None and local_logits.size(0) > tokens_per_rank and phase != "router_weights":
                            local_logits = local_logits[:tokens_per_rank]

                        local_layer_data.append((layer_idx, local_logits))

                # Use gather_object to collect data on CPU (avoids GPU memory spike)
                # This works with gloo backend or when objects are serializable
                if dp_rank == 0:
                    gather_list = [None] * dp_world_size
                    torch.distributed.gather_object(local_layer_data, gather_list, dst=0, group=dp_group)

                    # Merge data from all DP ranks
                    layer_combined = {}
                    for rank_data in gather_list:
                        if rank_data is None:
                            continue
                        for layer_idx, logits in rank_data:
                            if layer_idx not in layer_combined:
                                layer_combined[layer_idx] = []
                            layer_combined[layer_idx].append(logits)

                    # Concatenate along batch/token dimension (for logits) or pick first (for router_weights)
                    if phase == "global_token_ids":
                        if "global_token_ids" in layer_combined:
                            # Concatenate token_ids from all ranks
                            combined_ids = torch.cat(layer_combined["global_token_ids"], dim=0)
                            gathered_data[phase].append(combined_ids)
                    elif phase == "router_weights":
                        for layer_idx in sorted(layer_combined.keys()):
                            # Pick first occurrence for router_weights (parameters don't split)
                            gathered_data[phase][layer_idx] = layer_combined[layer_idx][0]
                    elif phase == "predictive_bias":
                        for layer_idx in sorted(layer_combined.keys()):
                            # Concatenate predictive_bias from all ranks (like logits)
                            combined_bias = torch.cat(layer_combined[layer_idx], dim=0)
                            gathered_data[phase].append((layer_idx, combined_bias))
                    else:
                        for layer_idx in sorted(layer_combined.keys()):
                            combined_logits = torch.cat(layer_combined[layer_idx], dim=0)  # [total_tokens, num_experts]
                            gathered_data[phase].append((layer_idx, combined_logits))

                    logger.info(f"[gather_logits_from_dp_group] {phase}: gathered {len(layer_combined)} layers/items from {dp_world_size} DP ranks")
                else:
                    torch.distributed.gather_object(local_layer_data, None, dst=0, group=dp_group)

        except Exception as e:
            logger.error(f"[gather_logits_from_dp_group] Error during gathering: {e}")
            logger.error(f"  DP rank: {dp_rank}, DP world size: {dp_world_size}")
            raise

        return gathered_data if dp_rank == 0 else {"compute_log_prob": [], "training": [], "router_weights": {}, "global_token_ids": [], "predictive_bias": []}
