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
The main entry point to run the PPO algorithm
"""

import copy
import datetime
import logging
import math
import os
import re
import time
from contextlib import contextmanager, nullcontext
from typing import Any, Optional

import numpy as np
import psutil
import torch
import torch.distributed
from codetiming import Timer
from omegaconf import DictConfig, OmegaConf

try:
    from verl.workers.engine.mindspeed.transformer_impl import repatch
except ImportError:
    repatch = None
from megatron.core import parallel_state as mpu
from megatron.core.optimizer import ParamKey

from verl import DataProto
from verl.models.mcore import get_mcore_weight_converter
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import hf_tokenizer
from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
    set_expandable_segments,
)
from verl.utils.distributed import set_numa_affinity
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import deprecated
from verl.utils.megatron.router_replay_patch import (
    RouterPredictiveAction,
    RouterReplay,
    RouterReplayAction,
    RouterReplayCacheAction,
    apply_router_replay_patch,
)
from verl.utils.megatron_peft_utils import add_base_layer_suffix, build_peft_config_for_vllm
from verl.utils.megatron_utils import (
    load_megatron_model_to_gpu,
    load_megatron_optimizer,
    offload_megatron_model_to_cpu,
    offload_megatron_optimizer,
    per_tensor_generator,
    register_megatron_training_hooks,
)
from verl.utils.memory_utils import aggressive_empty_cache, get_system_memory_info
from verl.utils.model import get_hf_model_path, load_mcore_dist_weights, load_megatron_gptmodel_weights
from verl.utils.profiler import (
    DistProfiler,
    DistProfilerExtension,
    GPUMemoryLogger,
    ProfilerConfig,
    log_gpu_memory_usage,
    simple_timer,
)
from verl.utils.profiler.performance import reduce_timing, topk_reduce_ratio_min_max
from verl.utils.ray_utils import get_event_loop
from verl.utils.torch_functional import use_original_torch_compile
from verl.workers.actor.megatron_actor import MegatronPPOActor
from verl.workers.config import HFModelConfig, McoreCriticConfig, RolloutConfig
from verl.workers.critic.megatron_critic import MegatronPPOCritic
from verl.workers.rollout import get_rollout_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@contextmanager
def _hide_bias_predictors(modules):
    """Temporarily remove bias_predictor submodules from routers so mbridge doesn't see them.

    mbridge's load_weights iterates over model parameters and tries to map every Megatron
    param name to an HF name then load from the HF checkpoint.  The bias_predictor is a
    training-only parameter that doesn't exist in HF weights, so we hide it during loading
    and restore it afterwards (keeping its zero-initialized weights).
    """
    saved = []
    for m in modules:
        for module in m.modules():
            if hasattr(module, "bias_predictor") and module.bias_predictor is not None:
                saved.append((module, module.bias_predictor))
                module.bias_predictor = None
    try:
        yield
    finally:
        for module, bp in saved:
            module.bias_predictor = bp


def _bias_predictor_ckpt_debug_enabled() -> bool:
    return os.getenv("VERL_DEBUG_BIAS_PREDICTOR_CKPT", "").lower() in {"1", "true", "yes", "on"}


def _log_bias_predictor_stats(modules, stage: str, rank: int):
    if not _bias_predictor_ckpt_debug_enabled() or rank != 0:
        return
    lines = [f"[BiasPredictorDebug] [{stage}] rank={rank}"]
    for vpp_idx, model in enumerate(modules):
        m = model.module if hasattr(model, "module") else model
        while hasattr(m, "module"):
            m = m.module
        for name, param in m.named_parameters():
            if ".bias_predictor." not in name:
                continue
            t = param.detach().float()
            lines.append(
                f"    vpp={vpp_idx} {name} shape={tuple(t.shape)} dtype={param.dtype} "
                f"l2={t.norm().item():.6e} mean_abs={t.abs().mean().item():.6e} "
                f"max_abs={t.abs().max().item():.6e}"
            )
    if len(lines) == 1:
        lines.append("    <no bias_predictor params found>")
    logger.warning("\n".join(lines))


def _predictor_sync_debug_enabled() -> bool:
    return os.getenv("VERL_DEBUG_PREDICTOR_SYNC", "").lower() in {"1", "true", "yes", "on"}


def _format_predictor_sync_stats(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().float()
    return (
        f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"l2={tensor.norm().item():.6e} "
        f"maxabs={tensor.abs().max().item():.6e} "
        f"checksum={tensor.sum().item():.6e}"
    )


def _wrap_predictor_sync_debug(
    weights: Any,
    *,
    num_hidden_layers: int,
    logger: logging.Logger,
):
    if not _predictor_sync_debug_enabled():
        return weights

    target_layers = {0, max(num_hidden_layers - 1, 0)}
    pattern = re.compile(r"layers\.(\d+)\.mlp\.(bias_predictor|gate|router)\.weight$")

    def generator():
        for name, tensor in weights:
            match = pattern.search(name)
            if match is not None and int(match.group(1)) in target_layers:
                logger.info(
                    "[PredictorSync][actor-export] %s %s",
                    name,
                    _format_predictor_sync_stats(tensor),
                )
            yield name, tensor

    return generator()



def set_random_seed(seed, only_rollout=False):
    import random

    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if not only_rollout and get_torch_device().device_count() > 0:
        from megatron.core import tensor_parallel

        tensor_parallel.model_parallel_cuda_manual_seed(seed)
    # FIXME: torch cumsum not support deterministic (used in vllm sampler),
    # https://github.com/pytorch/pytorch/issues/89492
    # torch.use_deterministic_algorithms(True, warn_only=True)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


@deprecated("legacy worker implementation is deprecated and will be removed in v0.8.0")
class MegatronWorker(Worker):
    def _init_hf_config_and_tf_config(
        self,
        model_path,
        tokenizer_or_path,
        dtype,
        override_model_config,
        override_transformer_config,
        trust_remote_code=False,
        megatron_config=None,
        enable_mtp=False,
    ):
        from transformers import AutoConfig

        from verl.models.mcore import hf_to_mcore_config
        from verl.utils import hf_processor
        from verl.utils.model import update_model_config

        # Step 1: initialize the tokenizer
        self.local_path = copy_to_local(model_path)
        if tokenizer_or_path is None:
            self.tokenizer = hf_tokenizer(self.local_path, trust_remote_code=trust_remote_code)
            self.processor = hf_processor(self.local_path, trust_remote_code=trust_remote_code)
        elif isinstance(tokenizer_or_path, str):
            self.tokenizer = hf_tokenizer(copy_to_local(tokenizer_or_path), trust_remote_code=trust_remote_code)
            self.processor = hf_processor(copy_to_local(tokenizer_or_path), trust_remote_code=trust_remote_code)
        else:
            self.tokenizer = tokenizer_or_path
            self.processor = tokenizer_or_path

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template

        # Step 2: get the hf
        hf_config = AutoConfig.from_pretrained(self.local_path, trust_remote_code=trust_remote_code)

        # Step 3: override the hf config
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config.get("model_config", {}))
        self.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)

        # only actor need enable mtp
        if enable_mtp:
            assert hf_config.num_nextn_predict_layers > 0, "MTP requires at least one nextn_predict_layer"
            assert megatron_config.use_mbridge, "MTP requires use_mbridge to be True"
            override_transformer_config["mtp_loss_scaling_factor"] = self.config.model.mtp.mtp_loss_scaling_factor
        else:
            if hasattr(hf_config, "num_nextn_predict_layers"):
                hf_config.num_nextn_predict_layers = 0

        self.enable_mtp = enable_mtp

        update_model_config(hf_config, override_config_kwargs=override_config_kwargs)
        self.architectures = getattr(hf_config, "architectures", None)
        if self.rank == 0:
            print(f"Model config after override: {hf_config}")

        from verl.models.mcore.config_converter import mapping_string_to_attn_backend

        # todo: remove this line after mcore adopt mbridge 0.15, now for compatibility
        override_transformer_config = mapping_string_to_attn_backend(override_transformer_config)
        fp16 = dtype == torch.float16
        bf16 = dtype == torch.bfloat16
        if fp16:
            assert megatron_config.use_mbridge, "fp16 mode requires use_mbridge to be True"

        self.provider = None
        self.vanilla_bridge = megatron_config.get("vanilla_mbridge", True)
        if megatron_config.use_mbridge:
            if self.vanilla_bridge:
                from verl.models.mcore.mbridge import AutoBridge

                bridge = AutoBridge.from_config(hf_config, dtype=dtype)
                bridge.set_extra_args(**override_transformer_config)
                tf_config = bridge.config
                tf_config.fp16 = fp16
                tf_config.bf16 = bf16
            else:
                from verl.models.mcore.bridge import AutoBridge

                # Use Megatron-Bridge to convert HF config to Megatron config
                bridge = AutoBridge.from_hf_pretrained(self.local_path, trust_remote_code=trust_remote_code)
                # Get Megatron provider and configure it
                provider = bridge.to_megatron_provider(load_weights=False)

                # In case of invalid overrides, we need to make sure some critical params are set correctly
                provider.params_dtype = dtype

                # Ensure dtype settings propagate to Megatron-Bridge/TE
                provider.fp16 = fp16
                provider.bf16 = bf16

                # Pass distributed info
                provider.tensor_model_parallel_size = megatron_config.tensor_model_parallel_size
                provider.pipeline_model_parallel_size = megatron_config.pipeline_model_parallel_size
                provider.expert_model_parallel_size = megatron_config.expert_model_parallel_size
                provider.expert_tensor_parallel_size = megatron_config.expert_tensor_parallel_size
                provider.virtual_pipeline_model_parallel_size = megatron_config.virtual_pipeline_model_parallel_size
                provider.context_parallel_size = megatron_config.context_parallel_size
                provider.sequence_parallel = megatron_config.sequence_parallel

                # Match verl implementation (need variable_seq_lengths)
                from megatron.core.transformer.enums import AttnBackend

                provider.attention_backend = AttnBackend.flash
                provider.variable_seq_lengths = True
                provider.moe_token_dispatcher_type = "alltoall"
                provider.moe_router_load_balancing_type = "none"

                qat_enabled = getattr(self.config, "actor", {}).get("qat", {}).get("enable", False)
                if qat_enabled:
                    from verl.utils.modelopt import patch_provider_for_qat

                    patch_provider_for_qat(provider)

                # Apply transformer config overrides
                for key, value in override_transformer_config.items():
                    setattr(provider, key, value)

                provider.finalize()
                self.provider = provider
                tf_config = None  # Will be set after model creation
            self.bridge = bridge
        else:
            tf_config = hf_to_mcore_config(hf_config, dtype, **override_transformer_config)
            self.bridge = None

        if torch.distributed.get_rank() == 0:
            if tf_config is not None:
                print(f"TF config: {tf_config}")
        self.hf_config = hf_config
        self.tf_config = tf_config

        actor_config = getattr(self.config, "actor", None)
        qat_enabled = actor_config.get("qat", {}).get("enable", False) if actor_config is not None else False
        if qat_enabled:
            if not self.bridge or self.vanilla_bridge:
                raise ValueError(
                    "QAT (Quantization-Aware Training) requires Megatron bridge. "
                    "Please ensure 'actor.megatron.use_mbridge' is set to True and "
                    "'actor.megatron.vanilla_mbridge' is set to False in your configuration."
                )

        # Get PEFT config from model.lora if specified
        from verl.workers.config.megatron_peft import get_peft_cls

        self.peft_cls = get_peft_cls(
            model_config=self.config.model, bridge=self.bridge, provider=self.provider, dtype=dtype
        )


@deprecated("legacy worker implementation is deprecated and will be removed in v0.8.0")
class ActorRolloutRefWorker(MegatronWorker, DistProfilerExtension):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        Worker.__init__(self)
        self.config = config
        if repatch is not None:
            # NPU MindSpeed patch, will be refactored with MindSpeedEngine.
            repatch(self.config.actor.megatron.get("override_transformer_config", {}))

        self.role = role
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        # NOTE(sgm): We utilize colocate WorkerGroup by default.
        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel strategy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        if not torch.distributed.is_initialized():
            set_numa_affinity()
            rank = int(os.environ["LOCAL_RANK"])
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
            get_torch_device().set_device(rank)

            if self._is_actor or self._is_ref:
                mpu.initialize_model_parallel(
                    tensor_model_parallel_size=self.config.actor.megatron.tensor_model_parallel_size,
                    pipeline_model_parallel_size=self.config.actor.megatron.pipeline_model_parallel_size,
                    virtual_pipeline_model_parallel_size=self.config.actor.megatron.virtual_pipeline_model_parallel_size,
                    use_sharp=False,
                    context_parallel_size=self.config.actor.megatron.context_parallel_size,
                    expert_model_parallel_size=self.config.actor.megatron.expert_model_parallel_size,
                    expert_tensor_parallel_size=self.config.actor.megatron.expert_tensor_parallel_size,
                    nccl_communicator_config_path=None,
                )

        if self._is_actor or self._is_ref:
            is_collect = (
                mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
                and mpu.get_context_parallel_rank() == 0
            )
            self._register_dispatch_collect_info(
                mesh_name="actor", dp_rank=mpu.get_data_parallel_rank(), is_collect=is_collect
            )
        only_rollout = self._is_rollout and not self._is_actor

        self.enable_routing_replay = False
        self.enable_logits_saving = False

        if self._is_actor:
            self.router_replay = self.config.actor.router_replay
            self.enable_routing_replay = self.router_replay.mode != "disabled"

            # Initialize logits saver if record_file is specified (This is independent of router_replay)
            if self.router_replay.record_file not in [None, ""]:
                from verl.utils.megatron.router_replay_saver import RouterReplayLogitsSaver
                self.logits_saver = RouterReplayLogitsSaver(self.router_replay.record_file)
                self.enable_logits_saving = True
                self.save_frequency = self.router_replay.save_frequency
                logger.info(f"Router logits saving enabled in HybridEngine. Save directory: {self.router_replay.record_file}, frequency: every {self.save_frequency} step(s)")
            else:
                self.logits_saver = None
                self.enable_logits_saving = False
                self.save_frequency = 1
        else:
            self.logits_saver = None
            self.enable_logits_saving = False

        # Apply patch if either router_replay or logits_saving is enabled
        if self.enable_routing_replay or self.enable_logits_saving:
            apply_router_replay_patch()
            logger.info(f"Applied router replay patch. router_replay={self.enable_routing_replay}, logits_saving={self.enable_logits_saving}")

        set_random_seed(seed=self.config.actor.megatron.seed, only_rollout=only_rollout)

        if self._is_actor:
            omega_profiler_config = config.actor.get("profiler", {})
        elif self._is_rollout:
            # NOTE: In colocation mode, rollout config may not take effect (follow the actor config)
            # This is for extendability in AsyncRL cases
            omega_profiler_config = config.rollout.get("profiler", {})
        elif self._is_ref:
            omega_profiler_config = config.ref.get("profiler", {})
        else:
            raise ValueError(
                f"Invalid role {self.role}, should be one of "
                "['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']"
            )
        # omega_profiler_config is DictConfig
        # profiler_config is a ProfilerConfig dataclass
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory", "precision_debugger"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

        # TODO(sgm): Currently, we only support reference model param offload
        # will support other offload later
        self._is_offload_param = False
        self._is_offload_grad = False
        self._is_offload_optimizer = False

        # Initialize LoRA-related attributes (will be updated in _build_rollout if needed)
        self.base_sync_done = False
        self.peft_merge = False

        # normalize config
        if self._is_actor:
            original_ppo_mini_batch_size = self.config.actor.ppo_mini_batch_size
            dp_world_size = mpu.get_data_parallel_world_size()
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= dp_world_size
            if self.config.actor.ppo_mini_batch_size <= 0:
                raise ValueError(
                    "actor.ppo_mini_batch_size collapsed to zero after Megatron normalization: "
                    f"original={original_ppo_mini_batch_size}, rollout.n={self.config.rollout.n}, "
                    f"dp_world_size={dp_world_size}. Increase actor.ppo_mini_batch_size or reduce "
                    "data parallel world size."
                )
            if self.config.actor.get("ppo_micro_batch_size", None):
                self.config.actor.ppo_micro_batch_size //= dp_world_size
                self.config.rollout.log_prob_micro_batch_size //= dp_world_size
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size
                self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size

            self._is_offload_param = self.config.actor.megatron.get("param_offload", False)
            self._is_offload_grad = self.config.actor.megatron.get("grad_offload", False)
            self._is_offload_optimizer = self.config.actor.megatron.get("optimizer_offload", False)
        elif self._is_ref:
            if self.config.ref.get("log_prob_micro_batch_size", None):
                self.config.ref.log_prob_micro_batch_size //= mpu.get_data_parallel_world_size()
                self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size
            else:
                assert self.config.ref.get("log_prob_micro_batch_size_per_gpu", None) is not None, (
                    "Please note that in the ref policy configuration, `log_prob_micro_batch_size_per_gpu` and "
                    "`log_prob_micro_batch_size` should not be None at the same time."
                )
            self._ref_is_offload_param = self.config.ref.megatron.get("param_offload", False)

    def _build_model_optimizer(
        self, model_path, optim_config, override_model_config, override_transformer_config, override_ddp_config=None
    ):
        from verl.utils.megatron.optimizer import (
            get_megatron_optimizer,
            get_megatron_optimizer_param_scheduler,
            init_megatron_optim_config,
        )
        from verl.utils.megatron_utils import McoreModuleWrapperConfig, make_megatron_module
        from verl.utils.model import get_generation_config, print_model_size

        self._init_hf_config_and_tf_config(
            model_path,
            self.config.model.get("tokenizer_path") or model_path,
            self.dtype,
            override_model_config,
            override_transformer_config,
            self.config.model.get("trust_remote_code", False),
            self.config.actor.megatron if not self._is_ref else self.config.ref.megatron,
            self.config.model.get("mtp", {}).get("enable", False),
        )
        self.generation_config = get_generation_config(
            self.local_path,
            self.config.model.get("trust_remote_code", False),
        )

        if self._is_actor or self._is_rollout:
            wrap_config = McoreModuleWrapperConfig(
                is_value_model=False,  # actor is not value model
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                wrap_with_ddp=True,
                use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer,
            )
            actor_module, updated_tf_config = make_megatron_module(
                wrap_config=wrap_config,
                tf_config=self.tf_config,
                hf_config=self.hf_config,
                bridge=self.bridge,
                provider=self.provider,
                override_model_config=override_model_config,
                override_ddp_config=override_ddp_config,
                peft_cls=self.peft_cls,
                peft_config=self.config.model.get("lora", None),
            )
            self.tf_config = updated_tf_config
            print(f"actor_module: {len(actor_module)}")
            _log_bias_predictor_stats(actor_module, "actor:after_make_megatron_module", self.rank)
            if self.config.actor.load_weight:
                load_initial_dist_checkpointing = self.config.actor.megatron.use_dist_checkpointing
                if self.config.actor.megatron.load_initial_dist_checkpointing is not None:
                    load_initial_dist_checkpointing = self.config.actor.megatron.load_initial_dist_checkpointing
                if load_initial_dist_checkpointing:
                    try:
                        load_mcore_dist_weights(
                            actor_module,
                            self.config.actor.megatron.dist_checkpointing_path,
                            is_value_model=False,
                            prefix=self.config.actor.megatron.dist_checkpointing_prefix,
                            ignore_missing_bias_predictor=bool(
                                getattr(self.tf_config, "enable_router_bias_predictor", False)
                            ),
                        )
                    except Exception as e:
                        if self.bridge is None:
                            raise
                        local_model_path = get_hf_model_path(self.config)
                        logger.warning(
                            "Failed to load initial Megatron dist checkpoint from %s, "
                            "falling back to HF weights at %s. Error: %s: %s",
                            self.config.actor.megatron.dist_checkpointing_path,
                            local_model_path,
                            type(e).__name__,
                            e,
                        )
                        with _hide_bias_predictors(actor_module):
                            if self.vanilla_bridge:
                                self.bridge.load_weights(actor_module, local_model_path)
                            else:
                                self.bridge.load_hf_weights(actor_module, local_model_path)
                else:
                    if self.bridge is not None:
                        local_model_path = get_hf_model_path(self.config)
                        with _hide_bias_predictors(actor_module):
                            if self.vanilla_bridge:
                                self.bridge.load_weights(actor_module, local_model_path)
                            else:
                                self.bridge.load_hf_weights(actor_module, local_model_path)
                    else:
                        load_megatron_gptmodel_weights(
                            self.config, self.hf_config, actor_module, params_dtype=self.dtype, is_value_model=False
                        )

            _log_bias_predictor_stats(actor_module, "actor:after_load_weight_initial", self.rank)
            if self.rank == 0:
                print_model_size(actor_module[0])
            log_gpu_memory_usage("After MegatronPPOActor init", logger=logger)
            qat_config = self.config.actor.get("qat", {})
            if qat_config.get("enable", False):
                from verl.utils.modelopt import apply_qat_to_modules

                actor_module = apply_qat_to_modules(actor_module, qat_config)
        elif self._is_ref:
            wrap_config = McoreModuleWrapperConfig(
                is_value_model=False,  # ref is not value model
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                wrap_with_ddp=False,
                use_distributed_optimizer=self.config.ref.megatron.use_distributed_optimizer,
            )
            ref_module, updated_tf_config = make_megatron_module(
                wrap_config=wrap_config,
                tf_config=self.tf_config,
                hf_config=self.hf_config,
                bridge=self.bridge,
                provider=self.provider,
                override_model_config=override_model_config,
            )
            self.tf_config = updated_tf_config
            if self.config.ref.load_weight:  # should align with the actor:
                assert self.config.actor.load_weight == self.config.ref.load_weight
                print("load ref weight start")
                load_initial_dist_checkpointing = self.config.ref.megatron.use_dist_checkpointing
                if self.config.ref.megatron.load_initial_dist_checkpointing is not None:
                    load_initial_dist_checkpointing = self.config.ref.megatron.load_initial_dist_checkpointing
                if load_initial_dist_checkpointing:
                    try:
                        load_mcore_dist_weights(
                            ref_module,
                            self.config.ref.megatron.dist_checkpointing_path,
                            is_value_model=False,
                            prefix=self.config.ref.megatron.dist_checkpointing_prefix,
                            ignore_missing_bias_predictor=bool(
                                getattr(self.tf_config, "enable_router_bias_predictor", False)
                            ),
                        )
                    except Exception as e:
                        if self.bridge is None:
                            raise
                        local_model_path = get_hf_model_path(self.config)
                        logger.warning(
                            "Failed to load initial Megatron dist checkpoint from %s, "
                            "falling back to HF weights at %s. Error: %s: %s",
                            self.config.ref.megatron.dist_checkpointing_path,
                            local_model_path,
                            type(e).__name__,
                            e,
                        )
                        if self.vanilla_bridge:
                            self.bridge.load_weights(ref_module, local_model_path)
                        else:
                            self.bridge.load_hf_weights(ref_module, local_model_path)
                else:
                    if self.bridge is not None:
                        local_model_path = get_hf_model_path(self.config)
                        if self.vanilla_bridge:
                            self.bridge.load_weights(ref_module, local_model_path)
                        else:
                            self.bridge.load_hf_weights(ref_module, local_model_path)
                    else:
                        load_megatron_gptmodel_weights(
                            self.config, self.hf_config, ref_module, params_dtype=self.dtype, is_value_model=False
                        )
            log_gpu_memory_usage("After ref module init", logger=logger)
            return ref_module, self.hf_config

        # TODO: add more optimizer args into config
        if self._is_actor:
            optim_config_megatron = init_megatron_optim_config(
                optim_config,
                use_distributed_optimizer=wrap_config.use_distributed_optimizer,
                fp16=self.dtype == torch.float16,
            )
            
            # Build config_overrides for bias_predictor if enabled
            config_overrides = None
            if self.tf_config.enable_router_bias_predictor:
                bias_predictor_lr = optim_config_megatron.lr * self.tf_config.bias_predictor_lr_mult
                if math.isclose(
                    float(bias_predictor_lr),
                    float(optim_config_megatron.lr),
                    rel_tol=0.0,
                    abs_tol=0.0,
                ):
                    logger.info(
                        "[Bias Predictor] Using base optimizer config for predictor params: "
                        "base_lr=%s lr_mult=%s",
                        optim_config_megatron.lr,
                        self.tf_config.bias_predictor_lr_mult,
                    )
                else:
                    # Create specialized optimizer config for bias_predictor only when it truly
                    # differs from the default config. Megatron asserts if an override produces
                    # the same effective optimizer config as the default group.
                    bias_predictor_optim_config = copy.deepcopy(optim_config_megatron)
                    bias_predictor_optim_config.lr = bias_predictor_lr

                    # Use ParamKey to match parameters by attribute
                    bias_predictor_key = ParamKey(attr='is_bias_predictor')
                    config_overrides = {bias_predictor_key: bias_predictor_optim_config}

                    logger.info(
                        "[Bias Predictor] Setting separate learning rate: base_lr=%s, lr_mult=%s, "
                        "bias_predictor_lr=%s",
                        optim_config_megatron.lr,
                        self.tf_config.bias_predictor_lr_mult,
                        bias_predictor_optim_config.lr,
                    )
            
            actor_optimizer = get_megatron_optimizer(
                model=actor_module, 
                config=optim_config_megatron,
                config_overrides=config_overrides
            )
            actor_optimizer_scheduler = get_megatron_optimizer_param_scheduler(
                optimizer=actor_optimizer, config=optim_config
            )
        else:
            optim_config = None
            actor_optimizer = None
            actor_optimizer_scheduler = None

        log_gpu_memory_usage("After actor optimizer init", logger=logger)

        register_megatron_training_hooks(actor_module, actor_optimizer)

        return actor_module, actor_optimizer, actor_optimizer_scheduler, self.hf_config, optim_config

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        # 1. parse rollout and huggingface model config
        rollout_omega = OmegaConf.create(OmegaConf.to_container(self.config.rollout, resolve=False))
        if self._is_actor:
            actor_router_replay = OmegaConf.to_container(self.config.actor.router_replay, resolve=False)
            rollout_router_replay = rollout_omega.get("router_replay")
            if rollout_router_replay is None or rollout_router_replay.get("mode", "disabled") == "disabled":
                rollout_omega["router_replay"] = actor_router_replay
                if actor_router_replay.get("mode", "disabled") != "disabled":
                    logger.info(
                        "[RouterStates] Inherited actor.router_replay into rollout config: mode=%s enable_bias_predictor=%s",
                        actor_router_replay.get("mode"),
                        actor_router_replay.get("enable_bias_predictor"),
                    )
        rollout_config: RolloutConfig = omega_conf_to_dataclass(rollout_omega)

        # Convert megatron lora config to HFModelConfig
        model_config_dict = OmegaConf.to_container(self.config.model)
        model_config_dict.pop("lora", None)

        model_config: HFModelConfig = omega_conf_to_dataclass(
            OmegaConf.create(model_config_dict), dataclass_type=HFModelConfig
        )

        # 2. build rollout device mesh
        infer_tp = self.config.rollout.tensor_model_parallel_size * self.config.rollout.data_parallel_size
        infer_pp = self.config.rollout.pipeline_model_parallel_size
        infer_world_size = infer_tp * infer_pp
        dp = self.world_size // infer_world_size
        assert self.world_size % infer_world_size == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
        )
        rollout_device_mesh = init_device_mesh(
            get_device_name(), mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
        )

        self.rollout_device_mesh = rollout_device_mesh

        is_collect = (
            rollout_device_mesh["infer_tp"].get_local_rank() == 0
            and rollout_device_mesh["infer_pp"].get_local_rank() == 0
        )
        self._register_dispatch_collect_info(
            "rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect
        )

        # 4. build rollout model
        log_gpu_memory_usage(f"Before building {self.config.rollout.name} rollout", logger=logger)
        self.rollout = get_rollout_class(rollout_config.name, rollout_config.mode)(
            config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh
        )
        log_gpu_memory_usage(f"After building {self.config.rollout.name} rollout", logger=logger)

        # Initialize base_sync_done for LoRA
        self.base_sync_done: bool = "dummy" not in self.config.rollout.load_format
        self.peft_merge: bool = model_config.lora.get("merge", False)

        # 5. switch to trainer mode
        # NOTE: It's critical that hybrid engine in trainer mode initially to load checkpoint.
        # For async mode, we can't call run_until_complete here, so we will switch to trainer mode in AgentLoopManager.
        # Note: sync mode is deprecated and rejected in RolloutConfig.__post_init__

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        from verl.utils.torch_dtypes import PrecisionType

        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        if self._is_actor:
            override_transformer_config = OmegaConf.to_container(
                OmegaConf.create(self.config.actor.megatron.get("override_transformer_config", {}))
            )
            # Only set enable_routing_replay=True if router_replay mode is enabled
            # For logits-only recording (mode="disabled"), we use layer_number instead
            if self.enable_routing_replay:
                override_transformer_config["enable_routing_replay"] = True
                logger.info(f"[Rank {self.rank}] Set enable_routing_replay=True in transformer_config")
            # Set router bias predictor config if enabled
            if self.router_replay.enable_bias_predictor:
                override_transformer_config["enable_router_bias_predictor"] = True
                override_transformer_config["bias_predictor_loss_type"] = self.router_replay.bias_predictor_loss_type
                override_transformer_config["bias_predictor_lr_mult"] = self.router_replay.bias_predictor_lr_mult
                logger.info(f"[Rank {self.rank}] Router bias predictor enabled with loss_type={self.router_replay.bias_predictor_loss_type}, lr_mult={self.router_replay.bias_predictor_lr_mult}, inputs_storage_dtype={self.router_replay.predictive_inputs_storage_dtype}, logits_storage_dtype={self.router_replay.predictive_logits_storage_dtype}")

            # Log logits_saving status for debugging
            if self.enable_logits_saving:
                logger.info(f"[Rank {self.rank}] Logits saving enabled (router_replay={self.enable_routing_replay})")
            override_ddp_config = OmegaConf.to_container(
                OmegaConf.create(self.config.actor.megatron.get("override_ddp_config", {}))
            )
        elif self._is_ref:
            override_transformer_config = OmegaConf.to_container(
                OmegaConf.create(self.config.ref.megatron.get("override_transformer_config", {}))
            )
        else:
            override_transformer_config = {}
        self.param_dtype = PrecisionType.to_dtype(self.config.actor.megatron.dtype)
        log_gpu_memory_usage("Before init actor model and optimizer", logger=logger)
        self.dtype = PrecisionType.to_dtype(self.param_dtype)
        if self._is_actor:
            # we need the model for actor and rollout
            optim_config = self.config.actor.optim if self._is_actor else None
            (
                self.actor_module,
                self.actor_optimizer,
                self.actor_optimizer_scheduler,
                self.actor_model_config,
                self.actor_optim_config,
            ) = self._build_model_optimizer(
                model_path=self.config.model.path,
                optim_config=optim_config,
                override_model_config=override_model_config,
                override_transformer_config=override_transformer_config,
                override_ddp_config=override_ddp_config,
            )
            if self._is_offload_param:
                offload_megatron_model_to_cpu(self.actor_module)
                log_gpu_memory_usage("After offload actor params and grad during init", logger=logger)
            if self._is_offload_optimizer:
                offload_megatron_optimizer(self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            self.actor = MegatronPPOActor(
                config=actor_cfg,
                model_config=self.actor_model_config,
                hf_config=self.hf_config,
                tf_config=self.tf_config,
                actor_module=self.actor_module,
                actor_optimizer=self.actor_optimizer,
                mtp_config=self.config.model.mtp if self.config.model.mtp.enable else None,
            )
            logger.info(f"routing replay layers: {len(RouterReplay.router_instances)}")
            if self.enable_routing_replay and len(RouterReplay.router_instances) == 0:
                logger.error(f"[Rank {self.rank}] ❌ Router replay is enabled but no router instances found! Check if enable_routing_replay is set in transformer_config.")
            log_gpu_memory_usage("After MegatronPPOActor init", logger=logger)

        if self._is_rollout:
            with use_original_torch_compile():
                self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))
            log_gpu_memory_usage("After rollout init", logger=logger)

        if self._is_ref:
            self.ref_module, self.ref_model_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                optim_config=None,
                override_model_config=override_model_config,
                override_transformer_config=override_transformer_config,
            )
            log_gpu_memory_usage("After ref model init", logger=logger)
            self.ref_policy = MegatronPPOActor(
                config=self.config.ref,
                model_config=self.ref_model_config,
                hf_config=self.hf_config,
                tf_config=self.tf_config,
                actor_module=self.ref_module,
                actor_optimizer=None,
            )
            if self._ref_is_offload_param:
                offload_megatron_model_to_cpu(self.ref_module)
                log_gpu_memory_usage("After offload ref params during init", logger=logger)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_mananager = MegatronCheckpointManager(
                config=self.config,
                checkpoint_config=self.config.actor.checkpoint,
                model_config=self.actor_model_config,
                transformer_config=self.tf_config,
                role="actor",
                model=self.actor_module,
                arch=self.architectures[0],
                hf_config=self.hf_config,
                param_dtype=self.param_dtype,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                optimizer=self.actor_optimizer,
                optimizer_scheduler=self.actor_optimizer_scheduler,
                use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer,
                use_checkpoint_opt_param_scheduler=self.config.actor.optim.use_checkpoint_opt_param_scheduler,
                bridge=self.bridge,
                provider=self.provider,
                use_dist_checkpointing=self.config.actor.megatron.use_dist_checkpointing,
                peft_cls=self.peft_cls,
            )

            self.layer_name_mapping = {
                "qkv_layer_name": "self_attention.linear_qkv.",
                "gate_proj_layer_name": "linear_fc1.",
            }
            self.weight_converter = None
            if not self.config.actor.megatron.use_mbridge:
                self.weight_converter = get_mcore_weight_converter(self.actor_model_config, self.dtype)

        # Free cached GPU memory so colocated vLLM processes can see it via cudaMemGetInfo
        aggressive_empty_cache(force_sync=True)
        log_gpu_memory_usage("After init_model finish", logger=logger)

    async def rollout_mode(self):
        """Context switch hybridengine to rollout mode."""
        aggressive_empty_cache(force_sync=True)
        set_expandable_segments(False)

        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor.actor_module, load_grad=False)
            log_gpu_memory_usage("After load actor params during rollout_mode", logger=logger)

        # Build peft_config for vLLM LoRA support
        peft_config = None
        do_lora_base_sync = False
        if not self.peft_merge and self.peft_cls is not None:
            peft_config = build_peft_config_for_vllm(self.config.model.get("lora", {}))
            # set sleep level for LoRA adapter weights only sync
            # TODO: make this configurable so that users with small
            # main memory can trade sync time to avoid OOM
            self.rollout.sleep_level = 1

            do_lora_base_sync = (not self.base_sync_done) or (
                self.rollout.sleep_level != 1 and self.config.rollout.free_cache_engine
            )

        if self.bridge is not None:
            if self.vanilla_bridge:
                per_tensor_param = self.bridge.export_weights(self.actor.actor_module)
            elif not self.peft_merge and self.peft_cls is not None:
                # Only export adapter weights
                per_tensor_param = self.bridge.export_adapter_weights(self.actor.actor_module)
            else:
                per_tensor_param = self.bridge.export_hf_weights(self.actor.actor_module)
        else:
            per_tensor_param = per_tensor_generator(
                self.actor.actor_module,
                self.actor_model_config,
                self.weight_converter,
                self.tf_config,
                self.layer_name_mapping,
            )
        # Strip training-only params that confuse SGLang's weight loader.
        # Known offenders when mbridge bridge.py patch is applied for HF save:
        #   - bias_predictor.weight: training-only prediction head, no SGLang submodule
        #   - expert_bias: mbridge may export with name that collides with a linear layer
        # Set VERL_DEBUG_ROLLOUT_SYNC_NAMES=1 to log every name/shape going to the rollout backend
        # so we can identify additional culprits if SGLang weight_loader AssertionError recurs.
        import os as _os
        _dbg_sync_names = _os.environ.get("VERL_DEBUG_ROLLOUT_SYNC_NAMES", "0") == "1"

        def _filter_rollout_skip(gen):
            _skipped, _kept = 0, 0
            for name, tensor in gen:
                if "bias_predictor" in name or "expert_bias" in name or "_extra_state" in name:
                    if _dbg_sync_names:
                        logger.info("[rollout-sync][SKIP] %s %s", name, getattr(tensor, "shape", None))
                    _skipped += 1
                    continue
                if _dbg_sync_names:
                    logger.info("[rollout-sync][SEND] %s %s", name, getattr(tensor, "shape", None))
                _kept += 1
                yield name, tensor
            if _dbg_sync_names:
                logger.info("[rollout-sync] total kept=%d skipped=%d", _kept, _skipped)

        per_tensor_param = _filter_rollout_skip(per_tensor_param)
        per_tensor_param = _wrap_predictor_sync_debug(
            per_tensor_param,
            num_hidden_layers=getattr(self.hf_config, "num_hidden_layers", 1),
            logger=logger,
        )
        qat_config = self.config.actor.get("qat", {})
        if qat_config.get("enable", False):
            from verl.utils.modelopt import export_qat_weights

            qat_mode = qat_config.get("mode", "w4a16")
            per_tensor_param = export_qat_weights(per_tensor_param, self.actor.actor_module, qat_mode, self.bridge)

        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["weights"])
        if do_lora_base_sync:
            # Base layer sync
            per_tensor_param_lora_base = self.bridge.export_hf_weights(
                self.actor.actor_module, merge_adapter_weights=False
            )
            await self.rollout.update_weights(
                add_base_layer_suffix(per_tensor_param_lora_base, model_type=self.hf_config.model_type),
                peft_config=peft_config,
                base_sync_done=False,
            )

            # Mark base sync as done after first successful sync
            self.base_sync_done = True

        await self.rollout.update_weights(per_tensor_param, peft_config=peft_config, base_sync_done=True)
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor.actor_module)
        aggressive_empty_cache(force_sync=True)
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["kv_cache"])

        set_expandable_segments(True)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @GPUMemoryLogger(role="update_actor", logger=logger)
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto, global_step: int = None):
        assert self._is_actor
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
            log_gpu_memory_usage("After load actor params and grad during update_actor", logger=logger)
        if self._is_offload_optimizer:
            load_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage("After load actor optimizer during update_actor", logger=logger)

        from verl.utils.memory_utils import get_system_memory_info
        
        micro_batch_size = self.config.actor.ppo_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        
        # logger.info(f"[Memory] [update_actor] Before make_minibatch_iterator: {get_system_memory_info()}")
        data.print_size(prefix="[Memory] Input data size")
        dataloader = self.actor.make_minibatch_iterator(data=data)
        # logger.info(f"[Memory] [update_actor] After make_minibatch_iterator: {get_system_memory_info()}")
        
        with Timer(name="update_policy", logger=None) as timer:
            metrics = self.actor.update_policy(dataloader=dataloader, global_step=global_step)
        delta_time = timer.last
        global_num_tokens = data.meta_info["global_token_num"]
        images_seqlens = data.meta_info.get("images_seqlens", None)
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(
            global_num_tokens, delta_time, images_seqlens=images_seqlens
        )
        metrics["perf/mfu/actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
        metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
        metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
        metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)
        from verl.utils.megatron.optimizer import get_megatron_last_lr

        metrics["actor/lr"] = get_megatron_last_lr(self.actor_optimizer)
        self.actor_optimizer_scheduler.step(1)

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={"metrics": metrics})
        output = output.to("cpu")

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
            log_gpu_memory_usage("After offload actor params and grad during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        aggressive_empty_cache(force_sync=True)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    @GPUMemoryLogger(role="generate_sequences", logger=logger)
    @DistProfiler.annotate(color="red", role="rollout_generate")
    def generate_sequences(self, prompts: DataProto):
        assert self._is_rollout
        prompts = prompts.to(get_device_name())
        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)

        timing_generate = {}
        if self._is_actor:  # For rollout only, we do not switch context.
            loop = get_event_loop()
            loop.run_until_complete(self.rollout_mode())
            log_gpu_memory_usage("After switch to rollout mode", logger=logger)

        with simple_timer("generate_sequences", timing_generate):
            output = self.rollout.generate_sequences(prompts=prompts)

        if self._is_actor:
            loop.run_until_complete(self.trainer_mode())
            log_gpu_memory_usage("After switch to trainer mode", logger=logger)

        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = topk_reduce_ratio_min_max(
            timing_generate["generate_sequences"]
        )
        timing_generate = reduce_timing(timing_generate)
        timing_generate.update(
            {
                "generation_timing/max": timing_generate_max,
                "generation_timing/min": timing_generate_min,
                "generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")
        # clear kv cache
        aggressive_empty_cache(force_sync=True)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @GPUMemoryLogger(role="compute_ref_log_prob", logger=logger)
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    def compute_ref_log_prob(self, data: DataProto):
        if self.peft_cls is not None:
            # if is lora, actor without lora applied is the ref
            data.meta_info["is_lora"] = True
            return self.compute_log_prob(data)
        assert self._is_ref
        if self._ref_is_offload_param:
            load_megatron_model_to_gpu(self.ref_module, load_grad=False)
            log_gpu_memory_usage("After load ref params and grad during compute_ref_log_prob", logger=logger)
        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        output, _, _, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
        output = DataProto.from_dict(tensors={"ref_log_prob": output})
        output = output.to("cpu")
        if self._ref_is_offload_param:
            offload_megatron_model_to_cpu(self.ref_module)
            log_gpu_memory_usage("After offload ref params and grad during compute_ref_log_prob", logger=logger)
        aggressive_empty_cache(force_sync=True)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @GPUMemoryLogger(role="compute_log_prob", logger=logger)
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    def compute_log_prob(self, data: DataProto, global_step: Optional[int] = None):
        assert self._is_actor
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module, load_grad=False)
            log_gpu_memory_usage("After load actor params and grad during compute_log_prob", logger=logger)
        is_lora = data.meta_info.pop("is_lora", False)
        adapter_ctx = self.peft_cls.disable_adapter(self.actor_module) if is_lora else nullcontext()
        # we should always recompute old_log_probs when it is HybridEngine
        config_source = self.config.ref if is_lora else self.config.rollout
        data.meta_info["micro_batch_size"] = config_source.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = config_source.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = config_source.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature

        # Set cache action to COMPUTE_LOG_PROB phase
        # Determine whether to record/save this compute_log_prob based on save_frequency
        should_save = False
        if self.enable_logits_saving and global_step is not None:
            should_save = (global_step % self.save_frequency == 0)
        if should_save:
            RouterReplay.set_cache_action(RouterReplayCacheAction.COMPUTE_LOG_PROB)
            logger.info(f"[Rank {self.rank}] Enabled logits recording for COMPUTE_LOG_PROB. Debug: {RouterReplay.get_debug_info()}")
        else:
            RouterReplay.clear_cache_action()

        if self.enable_routing_replay and self.config.actor.router_replay.mode == "R2":
            RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
            # Set predictive action for log_prob phase
            if self.config.actor.router_replay.enable_bias_predictor:
                RouterReplay.set_global_predictive_action(RouterPredictiveAction.RECORD)

        if self.enable_routing_replay and self.config.actor.router_replay.mode == "R3":
            # R3 mode: Replay if routed_experts available, otherwise Record (for first step)
            if "routed_experts" in data.batch.keys():
                RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
                # R3 mode: Do NOT set predictive action in log_prob phase
                # Router states come from rollout, no need to record here
            else:
                # First step: use RECORD mode (fallback)
                logger.info(f"[Rank {self.rank}] R3 mode: routed_experts not in batch (first step). Using RECORD mode this time.")
                RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
                if self.config.actor.router_replay.enable_bias_predictor:
                    # First step needs RECORD since no rollout data yet
                    RouterReplay.set_global_predictive_action(RouterPredictiveAction.RECORD)

        with adapter_ctx:
            output, entropys, layers_topk_idx, layers_predictive_states = self.actor.compute_log_prob(
                data=data, calculate_entropy=not is_lora
            )
        tensors = {"ref_log_prob": output} if is_lora else {"old_log_probs": output}
        if not is_lora:
            tensors["entropys"] = entropys
        output = DataProto.from_dict(
            tensors=tensors,
            meta_info={"temperature": self.config.rollout.temperature},
        )
        # Save routed_experts for R2, or R3 when it recorded this step (for next step's replay)
        if self.config.actor.router_replay.mode == "R2":
            output.batch["routed_experts"] = layers_topk_idx
            if self.config.actor.router_replay.enable_bias_predictor:
                if layers_predictive_states is not None:
                    # Store to non_tensor_batch (as numpy arrays for Ray serialization efficiency)
                    output.non_tensor_batch["old_inputs"] = layers_predictive_states[0]  # list of numpy array [num_tokens_i, layers, hidden] (variable shape), length = bs
                    output.non_tensor_batch["old_logits"] = layers_predictive_states[1]  # list of numpy array [num_tokens_i, layers, num_experts] (variable shape), length = bs
                else:
                    logger.warning(f"[Rank {self.rank}] Bias predictor enabled but layers_predictive_states is None!")

        elif self.config.actor.router_replay.mode == "R3":
            # R3: only save if we recorded this step (i.e., routed_experts was not in input batch)
            if "routed_experts" not in data.batch.keys() and layers_topk_idx is not None:
                output.batch["routed_experts"] = layers_topk_idx
                
                # First step: also save router states if bias predictor is enabled
                if self.config.actor.router_replay.enable_bias_predictor and layers_predictive_states is not None:
                    output.non_tensor_batch["old_inputs"] = layers_predictive_states[0]
                    output.non_tensor_batch["old_logits"] = layers_predictive_states[1]
                    logger.info(f"[Rank {self.rank}] R3 first step: recorded routed_experts + router states")
                else:
                    logger.info(f"[Rank {self.rank}] R3 first step: recorded routed_experts for next step's replay")
            # Normal R3 runs: routed_experts and router states both come from rollout, no need to save new data

        if self.config.actor.router_replay.mode in ["R2", "R3"]:
            RouterReplay.clear_global_indices()
            RouterReplay.clear_global_router_replay_action()
            if self.config.actor.router_replay.enable_bias_predictor:
                RouterReplay.clear_global_predictive_data()
                RouterReplay.clear_global_predictive_action()

        # Save logits cache for this compute_log_prob call (only if should_save)
        if should_save:
            from verl.utils.megatron.router_replay_saver import RouterReplayLogitsSaver
            from megatron.core.transformer.moe.router import TopKRouter

            # Record router_weights after forward (captures weights at LogProb phase)
            for module in self.actor_module:
                for name, layer in module.named_modules():
                    if isinstance(layer, TopKRouter):
                        layer_idx = layer.layer_number
                        RouterReplay.logits_cache["router_weights"][layer_idx] = layer.weight.detach().cpu().contiguous()
                        logger.info(f"[compute_log_prob] Post-forward: Recorded router_weights for layer {layer_idx}, shape={layer.weight.shape}")
                    else:
                        # logger.info(f"[compute_log_prob] Module {name} is not TopKRouter, skipping.")
                        pass

            # Debug: Check state before getting cache
            debug_info = RouterReplay.get_debug_info()
            # logger.info(f"[Rank {self.rank}] Before get_and_clear_logits_cache: {debug_info}")
            # logger.info(f"[Memory] Before get_and_clear_logits_cache: GPU memory: allocated {get_torch_device().memory_allocated() / (1024**3):.2f} GB, CPU memory: {get_system_memory_info()}")
            logits_cache = RouterReplay.get_and_clear_logits_cache()
            # logger.info(f"[Memory] After get_and_clear_logits_cache: GPU memory: allocated {get_torch_device().memory_allocated() / (1024**3):.2f} GB, CPU memory: {get_system_memory_info()}")

            step_name = f"log_prob_{global_step}"
            if should_save:
                # Step 1: Gather logits across TP group (only TP rank 0 will have data after this)
                if mpu.get_tensor_model_parallel_world_size() > 1:
                    logits_cache = RouterReplayLogitsSaver.gather_logits_from_tp_group(logits_cache)

                # Step 2: Gather logits across DP group (only DP rank 0 will have data after this)
                # Only TP rank 0 participates in DP gathering
                if mpu.get_tensor_model_parallel_rank() == 0:
                    if mpu.get_data_parallel_world_size() > 1:
                        logits_cache = RouterReplayLogitsSaver.gather_logits_from_dp_group(logits_cache)

                    # Step 3: Save asynchronously (only on DP rank 0 and TP rank 0)
                    if mpu.get_data_parallel_rank() == 0:
                        self.logits_saver.save_logits_async(logits_cache, step_name)
                        logger.info(f"Scheduled async save for {step_name}")

            # Clear cache action
            RouterReplay.clear_cache_action()

        output = output.to("cpu")
        # clear kv cache
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
            log_gpu_memory_usage("After offload actor params and grad during compute_log_prob", logger=logger)
        aggressive_empty_cache(force_sync=True)
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, checkpoint_path, hdfs_path=None, del_local_after_load=True):
        # No checkpoint to load, just offload the model and optimizer to CPU
        if checkpoint_path is None:
            if self._is_offload_param:
                offload_megatron_model_to_cpu(self.actor_module)
            if self._is_offload_optimizer:
                offload_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage("After offload actor params and optimizer during load_checkpoint", logger=logger)
            return

        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
        _log_bias_predictor_stats(self.actor_module, f"actor:before_resume_load[{checkpoint_path}]", self.rank)
        self.checkpoint_mananager.load_checkpoint(
            local_path=checkpoint_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )
        _log_bias_predictor_stats(self.actor_module, f"actor:after_resume_load[{checkpoint_path}]", self.rank)
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_pretrained_model(self, checkpoint_path, del_local_after_load=True):
        pass

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, checkpoint_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
        # When save_contents includes 'optimizer' we must stage the distributed-optimizer
        # state back to GPU before generating the sharded state dict. With
        # optimizer_cpu_offload=True the on-device tensors have their storage resized to 0,
        # which would fail the internal shape assertion inside `sharded_state_dict`
        # (`assert tensors[key].shape == (gbuf_local_end - gbuf_local_start,)`).
        # Previously this was gated on `async_save`, but that is a bug: the same state is
        # read by both sync and async save paths. Always reload when about to save optimizer.
        need_optimizer_on_gpu = self._is_offload_optimizer and (
            self.checkpoint_mananager.should_save_optimizer
            or self.checkpoint_mananager.checkpoint_config.async_save
        )
        if need_optimizer_on_gpu:
            load_megatron_optimizer(self.actor_optimizer)
        self.checkpoint_mananager.save_checkpoint(
            local_path=checkpoint_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
        if need_optimizer_on_gpu:
            offload_megatron_optimizer(self.actor_optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def async_calls_finalize_fn_exec(self, blocking=False):
        from megatron.core.dist_checkpointing.strategies.base import async_calls

        async_calls.maybe_finalize_async_calls(blocking=blocking)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def start_profile(self, **kwargs) -> None:
        """Start profiling for the current rank in the current training step."""
        self.profiler.start(**kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def stop_profile(self) -> None:
        """Stop profiling for the current rank in the current training step."""
        self.profiler.stop()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def dump_memory_snapshot(self, tag: str = "manual", sub_dir: str = None) -> None:
        """Manually trigger a CUDA memory snapshot dump on all ranks."""
        # Memory snapshot is now handled by the profiler system
        # This method is kept for backward compatibility but delegates to profiler
        if hasattr(self, "profiler") and hasattr(self.profiler, "_impl"):
            try:
                # Try to use the profiler's memory snapshot functionality
                if hasattr(self.profiler._impl, "sampler"):
                    out_dir = OmegaConf.select(self.config, "actor.profiler.save_path") or "."
                    self.profiler._impl.sampler.dump_memory_snapshot(out_dir=out_dir, tag=tag, sub_dir=sub_dir)
            except Exception as e:
                # Log a warning if memory snapshot fails. This might be expected if the profiler doesn't support it.
                logger.warning(f"Failed to dump memory snapshot: {e}")


class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None):
        await self.rollout_mode()
        return True


class CriticWorker(MegatronWorker, DistProfilerExtension):
    def __init__(self, config: McoreCriticConfig):
        Worker.__init__(self)

        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory", "precision_debugger"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )
        self.config: McoreCriticConfig = config

        # NOTE(sgm): We utilize colocate WorkerGroup by default.
        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel strategy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        if not torch.distributed.is_initialized():
            set_numa_affinity()
            rank = int(os.environ["LOCAL_RANK"])
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
            get_torch_device().set_device(rank)

            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.config.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=self.config.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=self.config.megatron.virtual_pipeline_model_parallel_size,
                use_sharp=False,
                context_parallel_size=self.config.megatron.context_parallel_size,
                expert_model_parallel_size=self.config.megatron.expert_model_parallel_size,
                expert_tensor_parallel_size=self.config.megatron.expert_tensor_parallel_size,
                nccl_communicator_config_path=None,
            )

        is_collect = (
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
            and mpu.get_context_parallel_rank() == 0
        )
        self._register_dispatch_collect_info(
            mesh_name="critic", dp_rank=mpu.get_data_parallel_rank(), is_collect=is_collect
        )

        set_random_seed(seed=self.config.megatron.seed)

        # set FSDP offload params
        self._is_offload_param = self.config.megatron.param_offload
        self._is_offload_optimizer = self.config.megatron.optimizer_offload

        # normalize config
        original_ppo_mini_batch_size = self.config.ppo_mini_batch_size
        dp_world_size = mpu.get_data_parallel_world_size()
        self.config.ppo_mini_batch_size *= self.config.rollout_n
        self.config.ppo_mini_batch_size //= dp_world_size
        if self.config.ppo_mini_batch_size <= 0:
            raise ValueError(
                "ppo_mini_batch_size collapsed to zero after Megatron normalization: "
                f"original={original_ppo_mini_batch_size}, rollout_n={self.config.rollout_n}, "
                f"dp_world_size={dp_world_size}. Increase ppo_mini_batch_size or reduce "
                "data parallel world size."
            )
        if self.config.get("ppo_micro_batch_size", None):
            self.config.ppo_micro_batch_size //= dp_world_size
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size

        # TODO(sgm): support critic model offload

    def _build_critic_model_optimizer(
        self, model_path, optim_config, override_model_config, override_transformer_config, override_ddp_config
    ):
        from verl.utils.megatron.optimizer import (
            get_megatron_optimizer,
            get_megatron_optimizer_param_scheduler,
            init_megatron_optim_config,
        )
        from verl.utils.megatron_utils import McoreModuleWrapperConfig, make_megatron_module
        from verl.utils.model import print_model_size

        self._init_hf_config_and_tf_config(
            model_path,
            self.config.model.get("tokenizer_path") or model_path,
            self.dtype,
            override_model_config,
            override_transformer_config,
            self.config.model.get("trust_remote_code", False),
            self.config.megatron,
        )

        wrap_config = McoreModuleWrapperConfig(
            is_value_model=True,  # critic is value model
            share_embeddings_and_output_weights=False,
            wrap_with_ddp=True,
            use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
        )
        critic_module, updated_tf_config = make_megatron_module(
            wrap_config=wrap_config,
            tf_config=self.tf_config,
            hf_config=self.hf_config,
            bridge=self.bridge,
            provider=self.provider,
            override_model_config=override_model_config,
            override_ddp_config=override_ddp_config,
            peft_cls=self.peft_cls,
            peft_config=self.config.model.get("lora", None),
        )
        self.tf_config = updated_tf_config
        # note that here critic_module will be a list to be compatible with the construction of interleaved pp (vpp).
        # but here, we do not use pp (vpp) yet. For simplicity, we remove the list
        # critic_module = nn.ModuleList(critic_module)

        if self.config.load_weight:
            t0 = time.time()
            if self.config.megatron.use_dist_checkpointing:
                load_mcore_dist_weights(
                    critic_module,
                    self.config.megatron.dist_checkpointing_path,
                    is_value_model=True,
                    prefix=self.config.megatron.dist_checkpointing_prefix,
                )
            else:
                if self.bridge is not None:
                    local_model_path = get_hf_model_path(self.config)
                    if self.vanilla_bridge:
                        self.bridge.load_weights(critic_module, local_model_path)
                    else:
                        self.bridge.load_hf_weights(
                            critic_module, local_model_path, allowed_mismatched_params=["output_layer.weight"]
                        )
                else:
                    load_megatron_gptmodel_weights(
                        self.config, self.hf_config, critic_module, params_dtype=self.dtype, is_value_model=True
                    )
            t1 = time.time()
            if torch.distributed.get_rank() == 0:
                print(f"critic load_weight time: {t1 - t0}")
        if self.rank == 0:
            print_model_size(critic_module[0])

        # TODO: add more optimizer args into config
        optim_config_megatron = init_megatron_optim_config(
            optim_config,
            use_distributed_optimizer=wrap_config.use_distributed_optimizer,
            fp16=self.dtype == torch.float16,
        )
        critic_optimizer = get_megatron_optimizer(model=critic_module, config=optim_config_megatron)
        critic_optimizer_scheduler = get_megatron_optimizer_param_scheduler(
            optimizer=critic_optimizer, config=optim_config
        )
        get_torch_device().empty_cache()

        register_megatron_training_hooks(critic_module, critic_optimizer)

        return critic_module, critic_optimizer, critic_optimizer_scheduler, self.hf_config, optim_config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # create critic

        from verl.utils.torch_dtypes import PrecisionType

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)
        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        override_transformer_config = OmegaConf.to_container(
            OmegaConf.create(self.config.megatron.get("override_transformer_config", {}))
        )
        override_ddp_config = OmegaConf.to_container(
            OmegaConf.create(self.config.megatron.get("override_ddp_config", {}))
        )
        self.param_dtype = PrecisionType.to_dtype(self.config.megatron.dtype)
        self.dtype = PrecisionType.to_dtype(self.param_dtype)
        (
            self.critic_module,
            self.critic_optimizer,
            self.critic_optimizer_scheduler,
            self.critic_model_config,
            critic_optimizer_config,
        ) = self._build_critic_model_optimizer(
            model_path=self.config.model.path,
            optim_config=self.config.optim,
            override_model_config=override_model_config,
            override_transformer_config=override_transformer_config,
            override_ddp_config=override_ddp_config,
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.critic_optimizer)

        self.critic = MegatronPPOCritic(
            config=self.config,
            model_config=self.critic_model_config,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            critic_module=self.critic_module,
            critic_optimizer=self.critic_optimizer,
            critic_optimizer_config=critic_optimizer_config,
        )
        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_mananager = MegatronCheckpointManager(
            config=self.config,
            checkpoint_config=self.config.checkpoint,
            model_config=self.critic_model_config,
            transformer_config=self.tf_config,
            role="critic",
            model=self.critic_module,
            arch=self.architectures[0],
            hf_config=self.hf_config,
            param_dtype=self.param_dtype,
            share_embeddings_and_output_weights=False,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            optimizer=self.critic_optimizer,
            optimizer_scheduler=self.critic_optimizer_scheduler,
            use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
            use_checkpoint_opt_param_scheduler=self.config.optim.use_checkpoint_opt_param_scheduler,
            bridge=self.bridge,
            provider=self.provider,
            use_dist_checkpointing=self.config.megatron.use_dist_checkpointing,
            peft_cls=self.peft_cls,
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="cyan", role="compute_values")
    def compute_values(self, data: DataProto):
        micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        data = data.to(get_device_id())
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)
        values = self.critic.compute_values(data=data)
        output = DataProto.from_dict(tensors={"values": values})
        output = output.to("cpu")
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="pink", role="critic_update")
    def update_critic(self, data: DataProto):
        data = data.to(get_device_id())

        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)
        if self._is_offload_optimizer:
            load_megatron_optimizer(self.critic_optimizer)

        dataloader = self.critic.make_minibatch_iterator(data)
        with Timer(name="update_critic", logger=None) as timer:
            metrics = self.critic.update_critic(dataloader=dataloader)
        delta_time = timer.last
        global_num_tokens = data.meta_info["global_token_num"]
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        metrics["perf/mfu/critic"] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size
        from verl.utils.megatron.optimizer import get_megatron_last_lr

        metrics["critic/lr"] = get_megatron_last_lr(self.critic_optimizer)
        self.critic_optimizer_scheduler.step(1)

        output = DataProto(batch=None, meta_info={"metrics": metrics})

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.critic_optimizer)
        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, checkpoint_path, hdfs_path=None, del_local_after_load=True):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)
        self.checkpoint_mananager.load_checkpoint(
            local_path=checkpoint_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.critic_optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, checkpoint_path, hdfs_path=None, global_steps=0, max_ckpt_to_keep=None):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)
        need_optimizer_on_gpu = self._is_offload_optimizer and (
            self.checkpoint_mananager.should_save_optimizer
            or self.checkpoint_mananager.checkpoint_config.async_save
        )
        if need_optimizer_on_gpu:
            load_megatron_optimizer(self.critic_optimizer)
        self.checkpoint_mananager.save_checkpoint(
            local_path=checkpoint_path, hdfs_path=hdfs_path, global_step=global_steps, max_ckpt_to_keep=max_ckpt_to_keep
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        if need_optimizer_on_gpu:
            offload_megatron_optimizer(self.critic_optimizer)
