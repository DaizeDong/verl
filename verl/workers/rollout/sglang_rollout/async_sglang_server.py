# Copyright 2023-2024 SGLang Team
# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import asyncio
import dataclasses
import json
import logging
import os
from typing import Any, Optional

import ray
import sglang
import sglang.srt.entrypoints.engine
import torch
from ray.actor import ActorHandle
from sglang.srt.entrypoints.http_server import (
    ServerArgs,
    _GlobalState,
    _launch_subprocesses,
    app,
    set_global_state,
)
from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)
from sglang.srt.managers.tokenizer_manager import ServerStatus

from verl.single_controller.ray import RayClassWithInitArgs
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput
from verl.workers.rollout.sglang_rollout.sglang_rollout import ServerAdapter, _set_envs_and_config
from verl.workers.rollout.utils import get_free_port, is_valid_ipv6_address, run_unvicorn

logger = logging.getLogger(__file__)


def _rollout_router_replay_config(config):
    return getattr(config, "router_replay", None)


def _router_replay_get(router_replay_config, key: str, default=None):
    if router_replay_config is None:
        return default
    if isinstance(router_replay_config, dict):
        return router_replay_config.get(key, default)
    return getattr(router_replay_config, key, default)


def _should_return_router_states(config) -> bool:
    return bool(getattr(config, "enable_rollout_routing_replay", False))


def _should_enable_router_bias_predictor(config) -> bool:
    router_replay_config = _rollout_router_replay_config(config)
    return bool(_router_replay_get(router_replay_config, "enable_bias_predictor", False))
logger.setLevel(logging.INFO)


def _router_state_payload_present(value) -> bool:
    return value is not None and not (isinstance(value, str) and value == "")


def _normalize_router_state_quartet(
    router_inputs,
    router_logits,
    router_bias,
    router_token_positions,
    *,
    request_id: str,
):
    present = [
        _router_state_payload_present(router_inputs),
        _router_state_payload_present(router_logits),
        _router_state_payload_present(router_bias),
        _router_state_payload_present(router_token_positions),
    ]
    if not any(present):
        return None, None, None, None
    if not all(present):
        logger.warning(
            "[SGLang] Inconsistent router-state quartet for request_id=%s: "
            "inputs_present=%s logits_present=%s bias_present=%s token_positions_present=%s. "
            "Dropping router states.",
            request_id,
            present[0],
            present[1],
            present[2],
            present[3],
        )
        return None, None, None, None
    return router_inputs, router_logits, router_bias, router_token_positions


def _get_expected_router_token_count(meta_info: dict[str, Any], output_token_ids: list[int]) -> int:
    prompt_tokens = meta_info.get("prompt_tokens")
    completion_tokens = meta_info.get("completion_tokens")
    if prompt_tokens is None or completion_tokens is None:
        logger.warning(
            "[SGLang] Missing prompt_tokens/completion_tokens while decoding router states. "
            "Falling back to len(output_ids)=%s.",
            len(output_token_ids),
        )
        return len(output_token_ids)
    return max(prompt_tokens + completion_tokens - 1, 0)


def _coerce_router_state_array(value, *, expected_tokens: int, field_name: str, request_id: str):
    import numpy as np

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().numpy()
    elif isinstance(value, list):
        value = np.asarray(value)
    if not hasattr(value, "shape") or len(value.shape) == 0:
        logger.warning(
            "[SGLang] Invalid %s payload for request_id=%s: type=%s. Dropping router states.",
            field_name,
            request_id,
            type(value),
        )
        return None
    if value.shape[0] != expected_tokens:
        logger.warning(
            "[SGLang] %s token-count mismatch for request_id=%s: expected=%s actual=%s. "
            "Dropping router states.",
            field_name,
            request_id,
            expected_tokens,
            value.shape[0],
        )
        return None
    return value


def _decode_router_states_from_meta_info(meta_info: dict[str, Any], *, hf_config, output_token_ids: list[int], request_id: str):
    import numpy as np
    import pybase64

    router_inputs_base64, router_logits_base64, router_bias_base64, router_token_positions_base64 = (
        _normalize_router_state_quartet(
        meta_info.get("router_inputs"),
        meta_info.get("router_logits"),
        meta_info.get("router_bias"),
        meta_info.get("router_token_positions"),
        request_id=request_id,
    ))
    if router_inputs_base64 is None:
        return None, None, None, None

    num_tokens = _get_expected_router_token_count(meta_info, output_token_ids)
    hidden_size = hf_config.hidden_size
    num_layers = hf_config.num_hidden_layers
    num_experts = hf_config.num_local_experts

    def _decode_payload(base64_value: str, feature_size: int, field_name: str):
        raw = np.frombuffer(pybase64.b64decode(base64_value.encode("utf-8")), dtype=np.float16)
        expected_size = num_tokens * num_layers * feature_size
        if raw.size != expected_size:
            logger.warning(
                "[SGLang] %s size mismatch for request_id=%s: expected=%s actual=%s. Dropping router states.",
                field_name,
                request_id,
                expected_size,
                raw.size,
            )
            return None
        return raw.reshape(num_tokens, num_layers, feature_size)

    router_inputs = _decode_payload(router_inputs_base64, hidden_size, "router_inputs")
    router_logits = _decode_payload(router_logits_base64, num_experts, "router_logits")
    router_bias = _decode_payload(router_bias_base64, num_experts, "router_bias")
    router_token_positions = np.frombuffer(
        pybase64.b64decode(router_token_positions_base64.encode("utf-8")),
        dtype=np.int32,
    )
    if router_token_positions.size != num_tokens:
        logger.warning(
            "[SGLang] router_token_positions size mismatch for request_id=%s: expected=%s actual=%s. "
            "Dropping router states.",
            request_id,
            num_tokens,
            router_token_positions.size,
        )
        return None, None, None, None
    if router_inputs is None or router_logits is None or router_bias is None:
        return None, None, None, None
    return router_inputs, router_logits, router_bias, router_token_positions


@ray.remote(num_cpus=1)
class SGLangHttpServer:
    """SGLang http server in single node, this is equivalent to launch server with command line:
    ```
    python -m sglang.launch_server --node-rank 0 --nnode 1 ...
    ```

    Args:
        config (DictConfig): full config.
        rollout_mode (RolloutMode): rollout mode.
        replica_rank (int): replica rank, a replica may contain multiple nodes.
        node_rank (int): node rank.
        nnodes (int): number of nodes.
        cuda_visible_devices (str): cuda visible devices.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        nnodes: int,
        cuda_visible_devices: str,
    ):
        print(f"SGLang http server: {rollout_mode=}, {replica_rank=}, {node_rank=}, {nnodes=}, {cuda_visible_devices=}")
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        assert torch.cuda.is_available(), "SGLang http server should run on GPU node"

        self.config: RolloutConfig = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
        self.config.max_model_len = self.config.prompt_length + self.config.response_length
        self.rollout_mode = rollout_mode
        self.workers = workers

        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.nnodes = nnodes

        if self.rollout_mode != RolloutMode.HYBRID and self.config.load_format == "dummy":
            logger.warning(f"rollout mode is {self.rollout_mode}, load_format is dummy, set to auto")
            self.config.load_format = "auto"

        # used for http server
        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port = None

        # used for NCCL process group
        if self.node_rank == 0:
            self._master_address = self._server_address
            self._master_port, self._master_sock = get_free_port(self._server_address)
            logger.info(
                f"SGLangHttpServer, replica_rank: {self.replica_rank}, "
                f"master address: {self._master_address}, port: {self._master_port}"
            )
        else:
            self._master_address = None
            self._master_port = None

    def get_master_address(self):
        """Get master address and port for init NCCL process group."""
        return self._master_address, self._master_port

    def get_server_address(self):
        """Get http server address and port."""
        assert self._server_port is not None, "http server is not launched, port is None"
        return self._server_address, self._server_port

    async def launch_server(self, master_address: str = None, master_port: int = None):
        if self.node_rank != 0:
            assert master_address and master_port, "non-master node should provide master address and port"
            self._master_address = master_address
            self._master_port = master_port

        engine_kwargs = self.config.get("engine_kwargs", {}).get("sglang", {}) or {}
        attention_backend = engine_kwargs.pop("attention_backend", None)
        quantization = self.config.get("quantization", None)
        if quantization is not None:
            if quantization == "fp8":
                assert sglang.__version__ >= "0.5.5", "sglang>=0.5.5 is required for FP8 quantization"
                FP8_BLOCK_QUANT_KWARGS = {
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "quant_method": "fp8",
                    "weight_block_size": [128, 128],
                }
                fp8_block_quant_kwargs = dict(FP8_BLOCK_QUANT_KWARGS)
            else:
                raise ValueError(f"Currently only support fp8 quantization, got: {quantization}")
        dist_init_addr = (
            f"[{self._master_address}]:{self._master_port}"
            if is_valid_ipv6_address(self._master_address)
            else f"{self._master_address}:{self._master_port}"
        )

        args = {
            "model_path": self.model_config.local_path,
            "dtype": self.config.dtype,
            "mem_fraction_static": self.config.gpu_memory_utilization,
            "disable_cuda_graph": self.config.enforce_eager,
            "enable_memory_saver": True,
            "base_gpu_id": 0,
            "gpu_id_step": 1,
            "tp_size": self.config.tensor_model_parallel_size,
            "dp_size": self.config.data_parallel_size,
            "ep_size": self.config.expert_parallel_size,
            "node_rank": self.node_rank,
            "load_format": self.config.load_format,
            "dist_init_addr": dist_init_addr,
            "nnodes": self.nnodes,
            "trust_remote_code": self.model_config.trust_remote_code,
            "max_running_requests": self.config.get("max_num_seqs", None),
            "log_level": "error",
            "mm_attention_backend": "fa3",
            "attention_backend": attention_backend if attention_backend is not None else "fa3",
            "skip_tokenizer_init": self.config.skip_tokenizer_init,
            "skip_server_warmup": True,
            "quantization": quantization,
            "json_model_override_args": json.dumps({"quantization_config": fp8_block_quant_kwargs})
            if quantization == "fp8"
            else json.dumps({}),
            **engine_kwargs,
        }

        if self.config.prometheus.enable:
            if self.config.prometheus.served_model_name:
                # Extract model name from path if it's a full path
                served_model_name = self.config.prometheus.served_model_name
                if "/" in served_model_name:
                    # If it's a full path, extract the last part as model name
                    served_model_name = served_model_name.split("/")[-1]
                args["served_model_name"] = served_model_name

            # start sglang metrics
            args["enable_metrics"] = True

        # enable_weights_cpu_backup is supported in sglang>=0.5.3
        if "enable_weights_cpu_backup" in [f.name for f in dataclasses.fields(ServerArgs)]:
            enable_weights_cpu_backup = True if self.rollout_mode == RolloutMode.COLOCATED else False
            args["enable_weights_cpu_backup"] = enable_weights_cpu_backup

        if self.config.enable_rollout_routing_replay:
            args.update({"enable_return_routed_experts": True})

            if _should_return_router_states(self.config):
                args["enable_return_router_states"] = True
                args["enable_router_bias_predictor"] = _should_enable_router_bias_predictor(self.config)

                router_replay_config = _rollout_router_replay_config(self.config)
                if router_replay_config is not None:
                    args["router_states_sample_rate"] = _router_replay_get(
                        router_replay_config,
                        "predictive_r3_downsample_keep_rate",
                        args.get("router_states_sample_rate"),
                    )
                    args["router_states_max_seq_len"] = _router_replay_get(
                        router_replay_config,
                        "predictive_downsample_max_len_limit",
                        args.get("router_states_max_seq_len"),
                    )

                logger.warning(
                    "[SGLang] Enabled router states capture: bias_predictor=%s keep_rate=%s max_seq_len=%s",
                    args["enable_router_bias_predictor"],
                    args.get("router_states_sample_rate"),
                    args.get("router_states_max_seq_len"),
                )

        # NOTE: We can't directly call SGLang's launch_server since it's not an async function.
        # https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py
        sglang.srt.entrypoints.engine._set_envs_and_config = _set_envs_and_config
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
        server_args = ServerArgs(**args)
        self.tokenizer_manager, self.template_manager, self.scheduler_info, *_ = _launch_subprocesses(
            server_args=server_args
        )

        # In multi-node cases, non-zero rank nodes should not launch http server.
        if self.node_rank > 0:
            return

        set_global_state(
            _GlobalState(
                tokenizer_manager=self.tokenizer_manager,
                template_manager=self.template_manager,
                scheduler_info=self.scheduler_info,
            )
        )
        app.is_single_tokenizer_mode = True

        # Set warmup_thread_args to avoid AttributeError in lifespan function
        app.warmup_thread_args = (
            server_args,
            None,
            None,
        )

        # Manually add Prometheus middleware before starting server
        # This ensures /metrics endpoint is available immediately
        if server_args.enable_metrics:
            from sglang.srt.utils.common import add_prometheus_middleware

            add_prometheus_middleware(app)

        self._server_port, self._server_task = await run_unvicorn(app, server_args, self._server_address)
        self.tokenizer_manager.server_status = ServerStatus.Up

    async def wake_up(self):
        if self.rollout_mode == RolloutMode.HYBRID:
            # Call all workers to switch between trainer mode and rollout mode.
            await asyncio.gather(*[worker.wake_up.remote() for worker in self.workers])
        elif self.rollout_mode == RolloutMode.COLOCATED:
            # Directly call engine to wake up without sync weights.
            obj = ResumeMemoryOccupationReqInput(tags=["kv_cache", "weights"])
            await self.tokenizer_manager.resume_memory_occupation(obj, None)
            await self.tokenizer_manager.flush_cache()
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip wake_up in standalone mode")

    async def sleep(self):
        if self.rollout_mode == RolloutMode.HYBRID:
            await asyncio.gather(*[worker.sleep.remote() for worker in self.workers])
        elif self.rollout_mode == RolloutMode.COLOCATED:
            obj = ReleaseMemoryOccupationReqInput(tags=["kv_cache", "weights"])
            await self.tokenizer_manager.release_memory_occupation(obj, None)
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip sleep in standalone mode")

    async def clear_kv_cache(self):
        obj = ReleaseMemoryOccupationReqInput(tags=["kv_cache"])
        await self.tokenizer_manager.release_memory_occupation(obj, None)

    async def generate(
        self,
        prompt_ids: torch.Tensor,
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate sequence with token-in-token-out."""
        # TODO(@wuxibin): switch to `/generate` http endpoint once multi-modal support ready.
        max_new_tokens = min(self.config.response_length, self.config.max_model_len - len(prompt_ids) - 1)
        sampling_params["max_new_tokens"] = max_new_tokens
        return_logprob = sampling_params.pop("logprobs", False)

        request = {
            "rid": request_id,
            "input_ids": prompt_ids,
            "sampling_params": sampling_params,
            "return_logprob": return_logprob,
            "image_data": image_data,
            # TODO: support video input for sglang
            # video_data=video_data,
        }

        if self.config.enable_rollout_routing_replay:
            request.update({"return_routed_experts": True})

            if _should_return_router_states(self.config):
                request.update({"return_router_states": True})
                logger.warning(
                    "[RouterStates] Sending generate request rid=%s return_router_states=%s",
                    request_id,
                    request["return_router_states"],
                )

        generate_request = GenerateReqInput(**request)

        output = await self.tokenizer_manager.generate_request(generate_request, None).__anext__()
        if return_logprob:
            output_token_logprobs = output["meta_info"]["output_token_logprobs"]
            log_probs, token_ids = zip(
                *[(log_prob, token_ids) for log_prob, token_ids, _ in output_token_logprobs], strict=True
            )
        else:
            token_ids = output["output_ids"]
            log_probs = None

        routed_experts = None
        router_inputs = None
        router_logits = None
        router_bias = None
        router_token_positions = None
        meta_info = output.get("meta_info", {})
        if (
            meta_info.get("router_inputs") is not None
            or meta_info.get("router_logits") is not None
            or meta_info.get("router_bias") is not None
            or meta_info.get("router_token_positions") is not None
        ):
            logger.warning(
                "[RouterStates] Received router-state payloads for rid=%s inputs_present=%s logits_present=%s "
                "bias_present=%s positions_present=%s",
                request_id,
                meta_info.get("router_inputs") is not None,
                meta_info.get("router_logits") is not None,
                meta_info.get("router_bias") is not None,
                meta_info.get("router_token_positions") is not None,
            )
        
        if self.config.enable_rollout_routing_replay:
            if self.config.skip_tokenizer_init:
                routed_experts = meta_info.get("routed_experts", None)

                if _should_return_router_states(self.config):
                    router_inputs, router_logits, router_bias, router_token_positions = _normalize_router_state_quartet(
                        meta_info.get("router_inputs", None),
                        meta_info.get("router_logits", None),
                        meta_info.get("router_bias", None),
                        meta_info.get("router_token_positions", None),
                        request_id=request_id,
                    )
                    if router_inputs is not None:
                        expected_tokens = _get_expected_router_token_count(meta_info, list(token_ids))
                        router_inputs = _coerce_router_state_array(
                            router_inputs,
                            expected_tokens=expected_tokens,
                            field_name="router_inputs",
                            request_id=request_id,
                        )
                        router_logits = _coerce_router_state_array(
                            router_logits,
                            expected_tokens=expected_tokens,
                            field_name="router_logits",
                            request_id=request_id,
                        )
                        router_bias = _coerce_router_state_array(
                            router_bias,
                            expected_tokens=expected_tokens,
                            field_name="router_bias",
                            request_id=request_id,
                        )
                        router_token_positions = _coerce_router_state_array(
                            router_token_positions,
                            expected_tokens=expected_tokens,
                            field_name="router_token_positions",
                            request_id=request_id,
                        )
                        if (
                            router_inputs is None
                            or router_logits is None
                            or router_bias is None
                            or router_token_positions is None
                        ):
                            router_inputs = None
                            router_logits = None
                            router_bias = None
                            router_token_positions = None
            else:
                from sglang.srt.layers.moe.routed_experts_capturer import extract_routed_experts_from_meta_info

                hf_config = self.model_config.hf_config
                if not hasattr(hf_config, "num_hidden_layers") or not hasattr(hf_config, "num_experts_per_tok"):
                    raise AttributeError(
                        "enable_rollout_routing_replay is set, but hf_config is missing "
                        "'num_hidden_layers' or 'num_experts_per_tok'. This feature requires an MoE model "
                        "configuration that defines these attributes."
                    )
                routed_experts = extract_routed_experts_from_meta_info(output).reshape(
                    -1, hf_config.num_hidden_layers, hf_config.num_experts_per_tok
                )

                if _should_return_router_states(self.config):
                    router_inputs, router_logits, router_bias, router_token_positions = _decode_router_states_from_meta_info(
                        meta_info,
                        hf_config=hf_config,
                        output_token_ids=list(token_ids),
                        request_id=request_id,
                    )

        return TokenOutput(
            token_ids=token_ids, 
            log_probs=log_probs, 
            routed_experts=routed_experts,
            router_inputs=router_inputs,
            router_logits=router_logits,
            router_bias=router_bias,
            router_token_positions=router_token_positions,
        )


_rollout_worker_actor_cls = ray.remote(ServerAdapter)


class SGLangReplica(RolloutReplica):
    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        """Get rollout worker actor class for colocated and standalone mode."""
        worker_dict_cls = RayClassWithInitArgs(
            cls=_rollout_worker_actor_cls,
            config=self.config,
            model_config=self.model_config,
            device_mesh=None,
        )
        return worker_dict_cls

    async def launch_servers(self):
        """Launch http server in each node."""
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        # get (node_id, CUDA_VISIBLE_DEVICES) of all workers
        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (ray.get_runtime_context().get_node_id(), os.environ["CUDA_VISIBLE_DEVICES"])
                )
                for worker in self.workers
            ]
        )
        worker_cuda_visible_devices = [worker_info[1] for worker_info in worker_infos]
        worker_node_ids = [worker_info[0] for worker_info in worker_infos]

        # create server actor in each node with node affinity and cuda visible devices
        for node_rank in range(self.nnodes):
            workers = self.workers[node_rank * self.gpus_per_node : (node_rank + 1) * self.gpus_per_node]
            node_cuda_visible_devices = ",".join(
                worker_cuda_visible_devices[node_rank * self.gpus_per_node : (node_rank + 1) * self.gpus_per_node]
            )
            node_id = worker_node_ids[node_rank * self.gpus_per_node]
            name = (
                f"sglang_server_{self.replica_rank}_{node_rank}"
                if not self.is_reward_model
                else f"sglang_server_reward_{self.replica_rank}_{node_rank}"
            )
            server = SGLangHttpServer.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                runtime_env={"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}},
                name=name,
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                nnodes=self.nnodes,
                cuda_visible_devices=node_cuda_visible_devices,
            )
            self.servers.append(server)

        # launch http server in each node
        master_address, master_port = await self.servers[0].get_master_address.remote()
        await asyncio.gather(
            *[
                server.launch_server.remote(master_address=master_address, master_port=master_port)
                for server in self.servers
            ]
        )

        # get http server address from first server
        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )
