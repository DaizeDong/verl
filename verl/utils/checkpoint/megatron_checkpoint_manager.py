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
import json
import logging
import os
import random
from dataclasses import fields, is_dataclass
from enum import Enum

import megatron.core
import numpy as np
import torch
import torch.distributed
from megatron.core import dist_checkpointing, mpu, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedObject
from packaging import version
from transformers import GenerationConfig

from verl.models.weight_loader_registry import get_weight_saver
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.fs import is_non_local, local_mkdir_safe
from verl.utils.logger import log_with_rank
from verl.utils.megatron.dist_checkpointing import load_dist_checkpointing, save_dist_checkpointing
from verl.utils.megatron_utils import (
    get_dist_checkpoint_path,
    get_hf_model_checkpoint_path,
    get_transformer_config_checkpoint_path,
)

from .checkpoint_manager import BaseCheckpointManager

# Setup logging
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))
mcore_ge_014 = version.parse(megatron.core.__version__) >= version.parse("0.14.0")
if not mcore_ge_014:
    logger.warning(
        "Detected megatron.core %s, recommend upgrading to >= 0.14.0 for better checkpoint compatibility",
        megatron.core.__version__,
    )



def _bias_predictor_debug_enabled() -> bool:
    return os.getenv("VERL_DEBUG_BIAS_PREDICTOR_CKPT", "").lower() in {"1", "true", "yes", "on"}


def _iter_bias_predictor_params(models):
    """Yield (vpp_idx, fully-qualified-name, parameter) for every router bias_predictor weight."""
    for vpp_idx, model in enumerate(models):
        m = model.module if hasattr(model, "module") else model
        while hasattr(m, "module"):
            m = m.module
        for name, param in m.named_parameters():
            if ".bias_predictor." in name:
                yield vpp_idx, name, param


def _collect_bias_predictor_stats_from_model(models) -> str:
    stats = []
    for vpp_idx, name, param in _iter_bias_predictor_params(models):
        t = param.detach().float()
        stats.append(
            f"    vpp={vpp_idx} {name} shape={tuple(t.shape)} dtype={param.dtype} "
            f"l2={t.norm().item():.6e} mean_abs={t.abs().mean().item():.6e} "
            f"max_abs={t.abs().max().item():.6e}"
        )
    if not stats:
        return "    <no bias_predictor params found>"
    return "\n".join(stats)


def _iter_optimizer_chained_subs(optimizer):
    """Yield the leaf optimizer instances under `optimizer` that expose
    `param_to_fp32_param` (the HybridDeviceOptimizer holding the fp32 master).

    Real verl + Megatron hierarchy:
      ChainedOptimizer.chained_optimizers = [DistributedOptimizer, ...]
        DistributedOptimizer.optimizer = HybridDeviceOptimizer  <-- the HDO

    `get_megatron_optimizer` always wraps in a ChainedOptimizer (one
    DistributedOptimizer per param-group cluster — dense / expert / predictor
    via `ParamKey(attr="is_bias_predictor")`). So we recurse into
    `chained_optimizers` for chains, then walk down `.optimizer` references on
    each leaf until we find a node exposing `param_to_fp32_param`. Without the
    `.optimizer` walk this iterator stops at the DistributedOptimizer, which
    has neither `chained_optimizers` nor `param_to_fp32_param` — the original
    bug that made the bias_predictor backfill a silent no-op on resume.

    If nothing in the chain exposes `param_to_fp32_param`, falls back to
    yielding the original optimizer so callers that introspect duck-typed test
    doubles (and `param_to_fp32_param`-less optimizers) still get a chance to
    inspect param_groups.
    """
    if optimizer is None:
        return
    chained = getattr(optimizer, "chained_optimizers", None)
    if chained:
        for sub in chained:
            yield from _iter_optimizer_chained_subs(sub)
        return
    # Walk inner `.optimizer` references to find the node holding fp32 masters.
    # `seen` guards against pathological self-references.
    node = optimizer
    seen = set()
    while node is not None and id(node) not in seen:
        if hasattr(node, "param_to_fp32_param"):
            yield node
            return
        seen.add(id(node))
        node = getattr(node, "optimizer", None)
    yield optimizer


def _iter_bias_predictor_shards(optimizer):
    """Yield (shard_param, fp32_master) for every bias_predictor shard reachable
    from `optimizer`.

    Megatron's `DistributedOptimizer._build_model_and_main_param_groups` slices
    each `model_param` into a `shard_model_param` view of `model_param.view(-1)
    [start:end]`. When the wrapping params have `is_bias_predictor=True` it
    tags those shards via `shard_model_param._is_bp_shard = True`
    (`distrib_optimizer.py:413`). Those shards become the keys of
    `HybridDeviceOptimizer.param_to_fp32_param`, and they're what the inner
    optimizer copies fp32 -> bf16 from inside `step()`.

    We iterate from the optimizer side because — with
    `use_precision_aware_optimizer=True` — `model_param.main_param` is None
    (`distrib_optimizer.py:405-409` "main params are held by FusedAdam"), so
    there is no model->main bridge attribute on the *model* parameter to
    follow.
    """
    for sub in _iter_optimizer_chained_subs(optimizer):
        param_to_fp32 = getattr(sub, "param_to_fp32_param", None)
        if param_to_fp32 is None:
            continue
        for group in getattr(sub, "param_groups", []) or []:
            for shard in group.get("params", []) or []:
                if not getattr(shard, "_is_bp_shard", False):
                    continue
                fp32 = param_to_fp32.get(shard)
                if fp32 is None:
                    continue
                yield shard, fp32


def _iter_optimizer_model_chunks(optimizer):
    """Yield every model_chunk (DDP-wrapped) reachable from `optimizer`.

    Used to call `start_param_sync(force_sync=True)` after the bias_predictor
    master->shard copy so every DP rank's slice gets cross-rank propagated.
    Without this the rank-local shard write only populates 1/DP of the full
    param; the other slices stay at the HF-load value (zero for predictor).

    Why we don't walk `.optimizer`: `ChainedOptimizer.optimizer` is a property
    that asserts `len(chained_optimizers) == 1` (`optimizer.py:1104-1112`), and
    in real verl runs with `config_overrides` the chain has at least two
    sub-optimizers (dense + expert, plus the predictor override), so that
    access raises AssertionError. `model_chunks` lives directly on both
    `ChainedOptimizer` (aggregated in `optimizer.py:1090-1093`) and
    `DistributedOptimizer` (`distrib_optimizer.py:511`), so we only need to
    traverse `chained_optimizers` to reach every owner.
    """
    if optimizer is None:
        return
    seen = set()
    queue = [optimizer]
    while queue:
        node = queue.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        chunks = getattr(node, "model_chunks", None)
        if chunks:
            for chunk in chunks:
                if id(chunk) not in seen:
                    seen.add(id(chunk))
                    yield chunk
        chained = getattr(node, "chained_optimizers", None)
        if chained:
            queue.extend(chained)


def _all_gather_after_backfill(optimizer) -> int:
    """Force a synchronous DP all-gather so the per-rank shard writes from
    `_iter_bias_predictor_shards` propagate to every rank's full-size
    param buffer.

    With `use_distributed_optimizer=True`, `model_param.data` is a view of a
    bucket buffer whose `[rank_slice]` is owned by *this* rank — other slices
    only get filled by the post-step all-gather. After resume the HF load left
    every slice at zero; our master->shard copy only fixes this rank's slice;
    forward then sees 1/DP of the predictor's contribution (we observed
    ratio=1.03e-5, ~1/64 of the trained 7.3e-4 baseline at DP=64). A synchronous
    `start_param_sync(force_sync=True)` on each model_chunk runs the all-gather
    immediately and brings every slice into sync before the first forward.

    Returns the number of model_chunks synced.
    """
    n_synced = 0
    for chunk in _iter_optimizer_model_chunks(optimizer):
        if not hasattr(chunk, "start_param_sync"):
            continue
        try:
            chunk.start_param_sync(force_sync=True)
        except TypeError:
            # Older DDP wrappers without keyword-only force_sync.
            chunk.start_param_sync(True)
        n_synced += 1
    return n_synced


def _restore_bias_predictor_from_optimizer_master(models, optimizer) -> int:
    """Copy the optimizer's fp32 master copy back into the model for every router
    bias_predictor parameter, then DP all-gather so every rank sees the full
    restored param.

    Why this exists: with `use_dist_checkpointing=False` + mbridge HF save, model
    weights live in HF safetensors which doesn't carry router.bias_predictor.weight
    (it isn't part of the public Qwen3MoE schema). After resume, the actor's
    bias_predictor sits at zero-init for a full step until `optimizer.step()`
    eventually copies the restored fp32 master back into the model — long enough
    that the first rollout pushes a zero predictor to SGLang and
    `router/predictive_bias_to_logits_ratio` collapses to 0 for one step.

    The dist_ckpt-side optimizer state IS restored by `optimizer.load_state_dict`
    above, so the fp32 master already holds the correct value. We just need to
    propagate master -> model BEFORE the first forward.

    Two steps:
      1. Per rank: `shard.data.copy_(fp32.data)` writes the rank-local slice of
         the predictor weight buffer. (Each predictor shard is
         `model_param.view(-1)[start:end]`.)
      2. Force a synchronous DP all-gather so every other rank's slice gets the
         restored value too. Without (2) only 1/DP of the predictor weight is
         non-zero per rank, and the metric collapses to ~1/DP of the trained
         baseline (we observed exactly this — 1.03e-5 vs. 7.3e-4 at DP=64).

    Returns the number of shards that were updated (step 1 count).
    """
    # Decide whether this is an R3+predictor run *symmetrically across ranks*
    # before doing anything else. `_iter_bias_predictor_params` matches by
    # parameter name (".bias_predictor." substring), which is consistent across
    # every rank because the model definition is identical regardless of how
    # the optimizer distributes shards. We must NOT branch on local shard
    # ownership (`n_targets`) here — `_all_gather_after_backfill` issues a DP
    # collective via `start_param_sync(force_sync=True)`, and DP groups in
    # practice contain ranks that don't own any predictor shard locally (the
    # shard layout depends on bucket packing). On the previous run (job
    # 1612473) only ~39/64 ranks owned predictor shards, so gating the sync
    # call on `n_targets > 0` left ~25 ranks out of the collective and every
    # DP group hung until the 1800s NCCL timeout fired.
    has_predictor = any(True for _ in _iter_bias_predictor_params(models))
    if not has_predictor:
        return 0

    n_targets = 0
    n_restored = 0
    for shard, fp32 in _iter_bias_predictor_shards(optimizer):
        n_targets += 1
        with torch.no_grad():
            shard.data.copy_(fp32.data)
        n_restored += 1

    # Step 2: force a synchronous DP all-gather so other ranks' slices of every
    # predictor weight buffer get the restored value. Without this only 1/DP of
    # the weight is non-zero per rank and the next forward sees mostly zero.
    # MUST run on every rank (collective), independent of local shard count.
    n_synced = _all_gather_after_backfill(optimizer)

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    log_with_rank(
        f"[bias_predictor backfill] copied master -> model for {n_restored} "
        f"predictor shard(s) and force-synced {n_synced} model_chunk(s) so the "
        f"next forward sees the trained predictor instead of zero.",
        rank=rank,
        logger=logger,
    )
    return n_restored


def _optim_resume_debug_enabled() -> bool:
    return os.getenv("VERL_DEBUG_OPTIM_RESUME", "").lower() in {"1", "true", "yes", "on"}


def _build_shard_to_model_param_name_map(models, optimizer):
    """Reverse-map every optimizer shard back to the (vpp_idx, model_param_name)
    it slices from. Used by the resume-time diagnostic to print names like
    `decoder.layers.42.mlp.router.bias_predictor.weight` instead of opaque
    Tensor identities.

    The shard is `model_param.detach().view(-1)[start:end]` so the storage
    pointer matches the underlying model parameter. We hash by `data_ptr()`
    on the model side and probe each shard's `data_ptr()` to find a match.
    """
    by_ptr: dict = {}
    for vpp_idx, model in enumerate(models):
        m = model.module if hasattr(model, "module") else model
        while hasattr(m, "module"):
            m = m.module
        for name, p in m.named_parameters():
            try:
                by_ptr[p.data_ptr()] = (vpp_idx, name, tuple(p.shape), p.numel())
            except Exception:
                continue
    shard_id_to_info: dict = {}
    for hdo in _iter_optimizer_chained_subs(optimizer):
        for g in (getattr(hdo, "param_groups", []) or []):
            for shard in (g.get("params", []) or []):
                try:
                    ptr = shard.data_ptr()
                except Exception:
                    continue
                if ptr in by_ptr:
                    shard_id_to_info[id(shard)] = by_ptr[ptr]
    return shard_id_to_info


def _dump_optimizer_state_for_debug(models, optimizer, rank: int) -> None:
    """One-shot post-resume diagnostic dump of optimizer state per param group.

    Targeted at the klpost1e2 (lr_mult=100) actor-divergence regression: with
    `config_overrides={ParamKey(attr='is_bias_predictor'): override}` the
    optimizer ends up with a separate param group for predictor params, and we
    suspect the save/load path is mis-associating Adam state (`exp_avg`,
    `exp_avg_sq`, `step`, or `master_param`) — or, per the prior run's rank-0
    dump, the LR scheduler is restoring 1e-4 onto a non-predictor group.
    Symptom: klpost1e2 pg_loss/ppo_kl explode 100x baseline at step 101,
    klpost1e1 (lr_mult=10) is unaffected — divergence scales with lr_mult.

    For each leaf HDO reached via _iter_optimizer_chained_subs, dumps every
    param_group:
      - group index, lr, max_lr, min_lr, wd_mult, n_params, total numel, bp_shards
      - per-param: model-side name (resolved by data_ptr lookup),
        `_is_bp_shard` flag, shape, dtype, master norm, exp_avg norm,
        exp_avg_sq norm, step counter
    Runs on rank 0 AND any rank with bp shards (so we can see both perspectives).

    Set VERL_DEBUG_OPTIM_RESUME=1 in the wrapper to enable.
    """
    # Compute total bp shards on this rank. Only emit a dump if we're rank 0
    # OR we own predictor shards (so we get at least one rank from each
    # population: predictor-owning and non-predictor-owning).
    bp_shards_here = sum(
        1
        for hdo in _iter_optimizer_chained_subs(optimizer)
        for g in (getattr(hdo, "param_groups", []) or [])
        for p in (g.get("params", []) or [])
        if getattr(p, "_is_bp_shard", False)
    )
    if rank != 0 and bp_shards_here == 0:
        return

    shard_to_name = _build_shard_to_model_param_name_map(models, optimizer)

    lines = [f"[OptimResumeDebug] rank={rank} starting dump (bp_shards_here={bp_shards_here})"]
    n_hdos = 0
    for hdo in _iter_optimizer_chained_subs(optimizer):
        n_hdos += 1
        lines.append(f"[OptimResumeDebug] rank={rank} === HDO #{n_hdos} (type={type(hdo).__name__}) ===")
        param_to_fp32 = getattr(hdo, "param_to_fp32_param", None)
        state = getattr(hdo, "state", None)
        param_groups = getattr(hdo, "param_groups", []) or []
        for gi, group in enumerate(param_groups):
            params = group.get("params", []) or []
            total_numel = sum(p.numel() for p in params)
            bp_shards_in_group = sum(1 for p in params if getattr(p, "_is_bp_shard", False))
            lines.append(
                f"[OptimResumeDebug] rank={rank}   group #{gi}: "
                f"lr={group.get('lr')} max_lr={group.get('max_lr')} min_lr={group.get('min_lr')} "
                f"wd_mult={group.get('wd_mult')} n_params={len(params)} "
                f"numel={total_numel} bp_shards={bp_shards_in_group}"
            )
            # Sample up to first 3 predictor shards and first 3 non-predictor shards.
            sampled = []
            seen_bp = 0
            seen_nonbp = 0
            for p_idx, p in enumerate(params):
                is_bp = getattr(p, "_is_bp_shard", False)
                if is_bp and seen_bp < 3:
                    sampled.append((p_idx, p, is_bp))
                    seen_bp += 1
                elif (not is_bp) and seen_nonbp < 3:
                    sampled.append((p_idx, p, is_bp))
                    seen_nonbp += 1
                if seen_bp >= 3 and seen_nonbp >= 3:
                    break

            for p_idx, p, is_bp in sampled:
                name_info = shard_to_name.get(id(p))
                name_str = (
                    f"vpp={name_info[0]} name={name_info[1]} model_shape={name_info[2]}"
                    if name_info is not None
                    else "name=<unresolved>"
                )
                fp32 = param_to_fp32.get(p) if param_to_fp32 is not None else None
                st = state.get(p) if state is not None else None
                if st is None and fp32 is not None and state is not None:
                    st = state.get(fp32)
                exp_avg = st.get("exp_avg") if st else None
                exp_avg_sq = st.get("exp_avg_sq") if st else None
                step_v = st.get("step") if st else None
                master = st.get("master_param") if st else None
                if fp32 is not None and master is None:
                    master = fp32
                lines.append(
                    f"[OptimResumeDebug] rank={rank}     p_idx={p_idx} bp_shard={is_bp} "
                    f"{name_str} "
                    f"shape={tuple(p.shape)} dtype={p.dtype} "
                    f"param_norm={p.detach().float().norm().item():.6e} "
                    f"master_norm={master.detach().float().norm().item() if isinstance(master, torch.Tensor) else 'NA'} "
                    f"exp_avg_norm={exp_avg.detach().float().norm().item() if isinstance(exp_avg, torch.Tensor) else 'NA'} "
                    f"exp_avg_sq_norm={exp_avg_sq.detach().float().norm().item() if isinstance(exp_avg_sq, torch.Tensor) else 'NA'} "
                    f"step={step_v.item() if isinstance(step_v, torch.Tensor) else step_v}"
                )

    n_model_predictor = sum(1 for _ in _iter_bias_predictor_params(models))
    n_optim_bp_shards = sum(
        1
        for hdo in _iter_optimizer_chained_subs(optimizer)
        for g in (getattr(hdo, "param_groups", []) or [])
        for p in (g.get("params", []) or [])
        if getattr(p, "_is_bp_shard", False)
    )
    lines.append(
        f"[OptimResumeDebug] rank={rank} cross-check: model_side_predictor_params={n_model_predictor} "
        f"optimizer_bp_shards_reachable={n_optim_bp_shards}"
    )

    # Audit: which model-side params have `is_bias_predictor=True` attr? Should
    # be exactly the bias_predictor.weight params (48 layers). If any other
    # param has it set, that's how non-predictor params end up matched by
    # `_matches(ParamKey(attr='is_bias_predictor'))` -> OVERRIDE config -> lr=1e-4
    # group at init.
    if rank == 0:
        bp_attr_names = []
        non_bp_attr_names_with_flag = []
        for model in models:
            m = model.module if hasattr(model, "module") else model
            while hasattr(m, "module"):
                m = m.module
            for name, p in m.named_parameters():
                has_flag = bool(getattr(p, "is_bias_predictor", False))
                if has_flag and ".bias_predictor." in name:
                    bp_attr_names.append(name)
                elif has_flag and ".bias_predictor." not in name:
                    non_bp_attr_names_with_flag.append((name, tuple(p.shape)))
        lines.append(
            f"[OptimResumeDebug] rank={rank} attr-audit: "
            f"is_bias_predictor_set_correctly={len(bp_attr_names)} "
            f"is_bias_predictor_LEAKED_onto_non_predictor={len(non_bp_attr_names_with_flag)}"
        )
        if non_bp_attr_names_with_flag:
            lines.append(
                "[OptimResumeDebug] rank={rank} LEAKED params (first 10):".format(rank=rank)
            )
            for n, sh in non_bp_attr_names_with_flag[:10]:
                lines.append(f"[OptimResumeDebug]   - name={n} shape={sh}")
    log_with_rank("\n".join(lines), rank=rank, logger=logger)





_SKIP_CONFIG_VALUE = object()


def _to_json_safe_config_value(value, seen):
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if type(value) is torch.dtype or isinstance(value, Enum):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if callable(value):
        return _SKIP_CONFIG_VALUE
    if isinstance(value, list | tuple):
        value_id = id(value)
        if value_id in seen:
            return _SKIP_CONFIG_VALUE
        seen.add(value_id)
        converted = []
        for item in value:
            converted_item = _to_json_safe_config_value(item, seen)
            if converted_item is not _SKIP_CONFIG_VALUE:
                converted.append(converted_item)
        seen.remove(value_id)
        return converted
    if isinstance(value, dict):
        value_id = id(value)
        if value_id in seen:
            return _SKIP_CONFIG_VALUE
        seen.add(value_id)
        converted = {}
        for key, item in value.items():
            converted_key = _to_json_safe_config_value(key, seen)
            converted_item = _to_json_safe_config_value(item, seen)
            if converted_key is not _SKIP_CONFIG_VALUE and converted_item is not _SKIP_CONFIG_VALUE:
                converted[str(converted_key)] = converted_item
        seen.remove(value_id)
        return converted
    return _SKIP_CONFIG_VALUE


def _to_json_safe_config_dict(config_dict):
    json_safe_config = {}
    for key, value in config_dict.items():
        converted = _to_json_safe_config_value(value, set())
        if converted is not _SKIP_CONFIG_VALUE:
            json_safe_config[key] = converted
    return json_safe_config


def _config_to_shallow_dict(config):
    if is_dataclass(config):
        return {field.name: getattr(config, field.name) for field in fields(config) if hasattr(config, field.name)}
    return vars(config)


class MegatronCheckpointManager(BaseCheckpointManager):
    """
    Checkpoint manager for Megatron-LM distributed training.

    This class manages the saving and loading of model checkpoints in a Megatron-LM
    distributed training environment. It handles various aspects of checkpointing
    including model states, optimizer states, learning rate schedulers, and random
    number generator states, ensuring compatibility with HuggingFace formats.

    Key features:
    - Distributed checkpoint saving and loading using Megatron's dist_checkpointing
    - Support for tensor parallel, pipeline parallel, and data parallel configurations
    - Automatic handling of model state dictionaries across multiple pipeline stages
    - Integration with HuggingFace model configurations and tokenizers
    - Random number generator state management for reproducibility
    - Support for both synchronous and asynchronous checkpoint operations

    The manager automatically handles:
    - Directory structure creation based on global steps and process ranks
    - Model configuration and tokenizer saving in HuggingFace format
    - Optimizer and scheduler state persistence
    - CUDA RNG state management for deterministic training
    - Checkpoint cleanup and retention policies

    Args:
        model: The Megatron model instance to checkpoint
        optimizer: The optimizer instance (optional)
        lr_scheduler: The learning rate scheduler instance (optional)

    Attributes:
        model: Reference to the Megatron model being checkpointed
        optimizer: Reference to the optimizer (if provided)
        lr_scheduler: Reference to the learning rate scheduler (if provided)
        rank: Current process rank in the distributed setup

    Example:
        ```python
        checkpoint_manager = MegatronCheckpointManager(
            model=megatron_model,
            optimizer=optimizer,
            lr_scheduler=scheduler
        )

        checkpoint_manager.save_checkpoint(
            local_path="checkpoints/step_1000",
            global_step=1000
        )

        checkpoint_manager.load_checkpoint(
            local_path="checkpoints/step_1000"
        )
        ```
    """

    def __init__(
        self,
        config,
        checkpoint_config,
        model_config,
        transformer_config,
        role,
        model: torch.nn.ModuleList,
        arch: str,
        hf_config,
        param_dtype: torch.dtype,
        share_embeddings_and_output_weights: bool,
        processing_class,
        optimizer,
        optimizer_scheduler,
        use_distributed_optimizer: bool,
        use_checkpoint_opt_param_scheduler: bool = False,
        use_dist_checkpointing: bool = True,
        use_megatron_fsdp: bool = False,
        bridge=None,
        provider=None,
        peft_cls=None,
        **kwargs,
    ):
        super().__init__(
            model,
            optimizer=optimizer,
            lr_scheduler=optimizer_scheduler,
            processing_class=processing_class,
            checkpoint_config=checkpoint_config,
        )
        self.arch = arch
        self.config = config
        self.transformer_config = transformer_config
        self.role = role
        self.is_value_model = False
        if self.role in ["reward", "critic"]:
            self.is_value_model = True
        self.model_config = model_config
        self.hf_config = hf_config
        self.param_dtype = param_dtype
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.model_path = self.config.model.path
        self.use_distributed_optimizer = use_distributed_optimizer
        self.use_checkpoint_opt_param_scheduler = use_checkpoint_opt_param_scheduler
        self.bridge = bridge
        self.provider = provider
        self.vanilla_bridge = self.provider is None
        self.peft_cls = peft_cls
        self.use_megatron_fsdp = use_megatron_fsdp
        self.rank = torch.distributed.get_rank()
        # Megatron-Bridge is Okay to load/save HF checkpoint for value model as well
        self.use_dist_checkpointing = (
            use_dist_checkpointing or not self.bridge or (self.vanilla_bridge and self.is_value_model)
        )
        self.use_hf_checkpoint = not self.use_dist_checkpointing

        self.weight_saver = None
        if self.bridge is None:
            self.weight_saver = get_weight_saver(self.arch)

    def get_rng_state(self, use_dist_ckpt: bool = True, data_parallel_random_init: bool = False):
        """collect rng state across data parallel ranks"""
        rng_state = {
            "random_rng_state": random.getstate(),
            "np_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "rng_tracker_states": tensor_parallel.get_cuda_rng_tracker().get_states(),
        }

        if get_device_name() != "cpu":
            rng_state[f"{get_device_name()}_rng_state"] = get_torch_device().get_rng_state()

        rng_state_list = None
        if torch.distributed.is_initialized() and mpu.get_data_parallel_world_size() > 1 and data_parallel_random_init:
            rng_state_list = [None for i in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather_object(rng_state_list, rng_state, group=mpu.get_data_parallel_group())
        else:
            rng_state_list = [rng_state]

        if self.use_megatron_fsdp:
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            tp_rank = mpu.get_tensor_model_parallel_rank()
            return {f"({pp_rank}, {tp_rank})": rng_state_list}

        if use_dist_ckpt:
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            pp_size = mpu.get_pipeline_model_parallel_world_size()
            tp_rank = mpu.get_tensor_model_parallel_rank()
            tp_size = mpu.get_tensor_model_parallel_world_size()
            rng_state_list = ShardedObject(
                "rng_state",
                rng_state_list,
                (pp_size, tp_size),
                (pp_rank, tp_rank),
                replica_id=mpu.get_data_parallel_rank(with_context_parallel=True),
            )

        return rng_state_list

    def get_checkpoint_name(
        self,
        checkpoints_path,
        pipeline_parallel=None,
        tensor_rank=None,
        pipeline_rank=None,
        cp_rank=None,
        expert_parallel=None,
        expert_rank=None,
        return_base_dir=True,
        basename="model.pt",
    ):
        """Determine the directory name for this rank's checkpoint."""
        # Use both the tensor and pipeline MP rank.
        if pipeline_parallel is None:
            pipeline_parallel = mpu.get_pipeline_model_parallel_world_size() > 1
        if tensor_rank is None:
            tensor_rank = mpu.get_tensor_model_parallel_rank()
        if pipeline_rank is None:
            pipeline_rank = mpu.get_pipeline_model_parallel_rank()
        if cp_rank is None:
            cp_rank = mpu.get_context_parallel_rank()
        if expert_parallel is None:
            expert_parallel = mpu.get_expert_model_parallel_world_size() > 1
        if expert_rank is None:
            expert_rank = mpu.get_expert_model_parallel_rank()

        # Use both the tensor and pipeline MP rank. If using the distributed
        # optimizer, then the optimizer's path must additionally include the
        # data parallel rank.

        # due to the fact that models are identical across cp ranks, cp rank is not used in the checkpoint path
        if not pipeline_parallel:
            common_path = os.path.join(checkpoints_path, f"mp_rank_{tensor_rank:02d}")
        else:
            common_path = os.path.join(checkpoints_path, f"mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}")

        if expert_parallel:
            common_path = common_path + f"_{expert_rank:03d}"

        os.makedirs(common_path, exist_ok=True)

        if return_base_dir:
            return common_path
        return os.path.join(common_path, basename)

    def generate_state_dict(
        self,
        generate_model: bool = True,
        generate_optimizer: bool = True,
        generate_extra: bool = True,
        is_loading: bool = False,
        metadata: dict | None = None,
    ):
        # For save dist checkpointing
        state_dict = {}
        base_metadata = metadata or self._build_sharded_state_dict_metadata()

        should_generate_model_sections = generate_model or generate_optimizer

        # All ranks save model state dict when it is needed for either model checkpointing
        # or optimizer sharded_state_dict generation.
        if should_generate_model_sections:
            for vpp_rank, model in enumerate(self.model):
                if len(self.model) > 1:
                    mpu.set_virtual_pipeline_model_parallel_rank(vpp_rank)
                    key = f"model{vpp_rank}"
                else:
                    key = "model"

                if self.use_megatron_fsdp:
                    state_dict[key] = model.state_dict_for_save_checkpoint()
                else:
                    if hasattr(model, "module"):
                        model = model.module

                    # GPTModel's sharded_state_dict function when having mtp requires metadata['dp_cp_group']
                    model_metadata = dict(base_metadata)
                    model_metadata["dp_cp_group"] = mpu.get_data_parallel_group(with_context_parallel=True)
                    kwargs = {"metadata": model_metadata}
                    state_dict[key] = model.sharded_state_dict(**kwargs)

        # Optimizer State Dict
        if generate_optimizer:
            torch.distributed.barrier()
            sharded_state_dict_kwargs = {"is_loading": is_loading}
            if base_metadata is not None:
                # https://github.com/NVIDIA/Megatron-LM/blob/core_v0.14.0/megatron/core/optimizer/distrib_optimizer.py#L1109-L1123
                if mcore_ge_014:
                    sharded_state_dict_kwargs["metadata"] = base_metadata
            optimizer_sharded_states = self.optimizer.sharded_state_dict(state_dict, **sharded_state_dict_kwargs)
            state_dict["optimizer"] = optimizer_sharded_states

            if self.lr_scheduler is not None:
                lr_state_dict = self.lr_scheduler.state_dict()
                state_dict["lr_scheduler"] = lr_state_dict

        if not generate_model:
            for key in list(state_dict.keys()):
                if self._is_model_state_key(key):
                    state_dict.pop(key)

        # RNG States State Dict
        if generate_extra:
            torch.distributed.barrier()
            rng_state = self.get_rng_state()
            state_dict["rng_state"] = rng_state

        return state_dict

    def _build_sharded_state_dict_metadata(self) -> dict:
        """Builds metadata used for sharded_state_dict versioning.


        The whole content metadata is passed to ``sharded_state_dict`` model and optimizer methods
        and therefore affects only the logic behind sharded_state_dict creation.
        The content metadata should be minimalistic, ideally flat (or with a single nesting level)
        and with semantically meaningful flag names (e.g. `distrib_optim_sharding_type`).
        In particular, a simple integer (or SemVer) versioning flag (e.g. `metadata['version'] = 3.4`)
        is discouraged, because the metadata serves for all models and optimizers and it's practically
        impossible to enforce a linearly increasing versioning for this whole space.
        """
        metadata: dict = {}

        if not mcore_ge_014:
            # For backward compatibility with Megatron core < v0.14.0
            if self.use_distributed_optimizer:
                metadata["distrib_optim_sharding_type"] = "fully_sharded_model_space"
            return metadata

        if self.use_megatron_fsdp:
            metadata["distrib_optim_sharding_type"] = "fsdp_dtensor"
        elif self.use_distributed_optimizer:
            megatron_config = getattr(self.config, self.role, self.config).megatron
            dist_ckpt_optim_fully_reshardable = megatron_config.dist_ckpt_optim_fully_reshardable
            distrib_optim_fully_reshardable_mem_efficient = (
                megatron_config.distrib_optim_fully_reshardable_mem_efficient
            )
            if dist_ckpt_optim_fully_reshardable:
                metadata["distrib_optim_sharding_type"] = "fully_reshardable"
                metadata["distrib_optim_fully_reshardable_mem_efficient"] = (
                    distrib_optim_fully_reshardable_mem_efficient
                )
            else:
                metadata["distrib_optim_sharding_type"] = "dp_reshardable"

        metadata["singleton_local_shards"] = False
        metadata["chained_optim_avoid_prefix"] = True
        return metadata

    @staticmethod
    def _is_model_state_key(key: str) -> bool:
        return key == "model" or (key.startswith("model") and key[5:].isdigit())

    @staticmethod
    def _has_checkpoint_files(path: str) -> bool:
        return os.path.isdir(path) and any(os.scandir(path))

    def _raise_for_unsupported_peft_checkpoint_layout(self, local_path: str, dist_checkpoint_path: str):
        if self.peft_cls is None or not self.should_load_model or self._has_checkpoint_files(dist_checkpoint_path):
            return

        legacy_adapter_ckpt_path = os.path.join(local_path, "adapter_checkpoint")
        hf_adapter_ckpt_path = os.path.join(local_path, "huggingface", "adapter")

        if os.path.isdir(legacy_adapter_ckpt_path):
            raise RuntimeError(
                f"Found legacy PEFT checkpoint at {legacy_adapter_ckpt_path}, but checkpoint resume now expects "
                f"adapter weights in {dist_checkpoint_path}. Resave/convert the checkpoint or load the adapter via "
                "`lora.adapter_path`."
            )

        if os.path.isfile(os.path.join(hf_adapter_ckpt_path, "adapter_config.json")):
            raise RuntimeError(
                f"Found exported HF PEFT adapter at {hf_adapter_ckpt_path}, but `load_checkpoint()` resumes from "
                f"{dist_checkpoint_path}. HF adapter exports are not used for trainer resume; keep the distributed "
                "checkpoint or load the adapter separately via `lora.adapter_path`."
            )

    def _maybe_filter_peft_state_dict(self, state_dict: dict):
        if self.peft_cls is None:
            return state_dict

        from megatron.bridge.training.checkpointing import apply_peft_adapter_filter_to_state_dict

        return apply_peft_adapter_filter_to_state_dict(state_dict, self.peft_cls)

    def _load_megatron_fsdp_checkpoint(self, local_path: str, del_local_after_load=False):
        dist_checkpoint_path = get_dist_checkpoint_path(local_path)
        if not os.path.isfile(os.path.join(dist_checkpoint_path, ".metadata")):
            raise FileNotFoundError(f"Megatron-FSDP checkpoint metadata not found at {dist_checkpoint_path}/.metadata.")

        sharded_state_dict = self.generate_state_dict(
            generate_model=True,
            generate_optimizer=True,
            generate_extra=True,
            is_loading=True,
            metadata=self._build_sharded_state_dict_metadata(),
        )

        from megatron.bridge.training.checkpointing import load_fsdp_dtensor_checkpoint

        checkpoint_model = getattr(self.model[0], "module", self.model[0])
        sharded_state_dict["_model"] = [checkpoint_model]
        state_dict, _, _, _ = load_fsdp_dtensor_checkpoint(
            load_dir=dist_checkpoint_path,
            ckpt_cfg=self.checkpoint_config,
            rank0=False,
            sharded_state_dict=sharded_state_dict,
            iteration=None,
            release=False,
            checkpoint_path_override=dist_checkpoint_path,
            cfg=self.transformer_config,
        )

        if self.should_load_model:
            self.model[0].load_state_dict(state_dict["model"], strict=True)
            log_with_rank(f"Loaded sharded model checkpoint from {local_path}", rank=self.rank, logger=logger)
        if self.should_load_optimizer:
            self.optimizer.load_state_dict(state_dict["optimizer"])
            log_with_rank(f"Loaded optimizer checkpoint from {local_path}", rank=self.rank, logger=logger)
            if self.use_checkpoint_opt_param_scheduler:
                assert "lr_scheduler" in state_dict, (
                    f"LR scheduler state dict not found in {state_dict.keys()}. Please check the checkpoint file "
                    f"{local_path}."
                )
                if self.lr_scheduler is not None:
                    self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
                    log_with_rank(f"Loaded LR scheduler checkpoint from {local_path}", rank=self.rank, logger=logger)
        if self.should_load_extra:
            self.load_rng_states(state_dict["rng_state"])
            log_with_rank(f"Loaded RNG states from {local_path}", rank=self.rank, logger=logger)
        log_with_rank(f"Loaded Megatron-FSDP checkpoint from {local_path}", rank=self.rank, logger=logger)

        if del_local_after_load:
            try:
                os.remove(local_path) if is_non_local(local_path) else None
            except Exception as e:
                log_with_rank(
                    f"remove local resume ckpt file after loading failed, exception {e} will be ignored",
                    rank=self.rank,
                    logger=logger,
                )

    def _save_megatron_fsdp_checkpoint(self, dist_checkpoint_path: str):
        state_dict = self.generate_state_dict(
            generate_model=self.should_save_model,
            generate_optimizer=self.should_save_optimizer,
            generate_extra=self.should_save_extra,
            metadata=self._build_sharded_state_dict_metadata(),
        )

        from megatron.bridge.training.checkpointing import save_fsdp_dtensor_checkpoint

        checkpoint_model = getattr(self.model[0], "module", self.model[0])
        save_fsdp_dtensor_checkpoint(
            dist_checkpoint_path,
            state_dict,
            cfg=self.transformer_config,
            model=checkpoint_model,
        )
        return None

    def load_rng_states(self, rng_states, data_parallel_random_init=False, use_dist_ckpt=True):
        if self.use_megatron_fsdp:
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            tp_rank = mpu.get_tensor_model_parallel_rank()
            key = f"({pp_rank}, {tp_rank})"
            if key in rng_states:
                rng_states = rng_states[key]
            else:
                log_with_rank(
                    f"RNG state for PP/TP key {key} not found; falling back to the first saved RNG state.",
                    rank=self.rank,
                    logger=logger,
                    log_only_rank_0=True,
                )
                rng_states = next(iter(rng_states.values()))

        # access rng_state for data parallel rank
        if data_parallel_random_init:
            rng_states = rng_states[mpu.get_data_parallel_rank()]
        else:
            rng_states = rng_states[0]
        random.setstate(rng_states["random_rng_state"])
        np.random.set_state(rng_states["np_rng_state"])
        torch.set_rng_state(rng_states["torch_rng_state"])

        if get_device_name() != "cpu":
            get_torch_device().set_rng_state(rng_states[f"{get_device_name()}_rng_state"])

        # Check for empty states array
        if not rng_states["rng_tracker_states"]:
            raise KeyError
        tensor_parallel.get_cuda_rng_tracker().set_states(rng_states["rng_tracker_states"])

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load=False):
        if local_path is not None:
            assert os.path.exists(local_path), f"Checkpoint path {local_path} does not exist."

        # For load optimizer dist_ckpt
        try:
            import transformer_engine

            torch.serialization.add_safe_globals([torch.optim.AdamW])
            torch.serialization.add_safe_globals([transformer_engine.pytorch.optimizers.fused_adam.FusedAdam])
        except Exception:
            pass

        if self.use_megatron_fsdp:
            self._load_megatron_fsdp_checkpoint(local_path, del_local_after_load=del_local_after_load)
            return

        dist_checkpoint_path = get_dist_checkpoint_path(local_path)

        self._raise_for_unsupported_peft_checkpoint_layout(local_path, dist_checkpoint_path)

        load_content_metadata = getattr(dist_checkpointing, "load_content_metadata", None)
        if load_content_metadata is None:
            # For backward compatibility
            sharded_sd_metadata = None
        else:
            sharded_sd_metadata = load_content_metadata(checkpoint_dir=dist_checkpoint_path)
        if sharded_sd_metadata is None:
            if self.use_distributed_optimizer:
                # Backward-compatibility with old checkpoints which don't have content versioning
                # Can be removed after ending support for MLM optimizer checkpoints with MCore < v0.13
                # (for MCore v0.13+ checkpoints `sharded_sd_metadata is not None`)
                sharded_sd_metadata = {
                    "distrib_optim_sharding_type": "fully_sharded_model_space",
                }
            else:
                sharded_sd_metadata = self._build_sharded_state_dict_metadata()

        # Get State Dict for loading
        should_load_dist_model = self.should_load_model and (self.use_dist_checkpointing or self.peft_cls is not None)

        # PRE-GENERATE snapshot of (max_lr, min_lr) per (HDO, param_group).
        # Captures the values BEFORE `generate_state_dict(..., is_loading=True)`
        # which itself triggers `DistributedOptimizer.sharded_state_dict(is_loading=True)`
        # -> `self.load_state_dict(self.state_dict())` and corrupts max_lr via
        # the `_filter_and_reorder_param_groups` identifier-key collision
        # described in `_restore_bias_predictor_from_optimizer_master`'s docstring.
        # We restore from this snapshot after the explicit `optimizer.load_state_dict` below.
        _pre_generate_lr_snapshot = []
        if self.should_load_optimizer:
            for _hdo in _iter_optimizer_chained_subs(self.optimizer):
                _pre_generate_lr_snapshot.append(
                    [
                        (g.get("max_lr"), g.get("min_lr"))
                        for g in (getattr(_hdo, "param_groups", []) or [])
                    ]
                )
            if _optim_resume_debug_enabled() and self.rank == 0:
                log_with_rank(
                    f"[OptimResumeDebug] rank=0 PRE-GENERATE snapshot of (max_lr, min_lr) per (HDO, group): "
                    f"{_pre_generate_lr_snapshot}",
                    rank=self.rank,
                    logger=logger,
                )

        sharded_state_dict = self.generate_state_dict(
            should_load_dist_model,
            self.should_load_optimizer,
            self.should_load_extra,
            is_loading=True,
            metadata=sharded_sd_metadata,
        )
        sharded_state_dict = self._maybe_filter_peft_state_dict(sharded_state_dict)
        log_with_rank(f"Generated state dict for loading: {sharded_state_dict.keys()}", rank=self.rank, logger=logger)

        # Load Dist Checkpointing
        state_dict = load_dist_checkpointing(
            sharded_state_dict=sharded_state_dict,
            ckpt_dir=dist_checkpoint_path,
        )

        if should_load_dist_model:
            assert "model" in state_dict or any(
                f"model{vpp_rank}" in state_dict for vpp_rank in range(len(self.model))
            ), f"Model state dict not found in {state_dict.keys()}. Please check the checkpoint file {local_path}."
            for vpp_rank, model in enumerate(self.model):
                if len(self.model) == 1:
                    model_state_dict = state_dict["model"]
                else:
                    assert f"model{vpp_rank}" in state_dict, f"model{vpp_rank} not found in state_dict"
                    model_state_dict = state_dict[f"model{vpp_rank}"]
                mpu.set_virtual_pipeline_model_parallel_rank(vpp_rank)
                self.model[vpp_rank].load_state_dict(model_state_dict, strict=self.peft_cls is None)
            if self.peft_cls is not None:
                log_with_rank(
                    f"Loaded PEFT adapter checkpoint from {dist_checkpoint_path}", rank=self.rank, logger=logger
                )
            else:
                log_with_rank(f"Loaded sharded model checkpoint from {local_path}", rank=self.rank, logger=logger)

        # Skip HF checkpoint loading if PEFT is used
        elif self.should_load_model and self.use_hf_checkpoint and self.peft_cls is None:
            hf_model_path = get_hf_model_checkpoint_path(local_path)
            if self.vanilla_bridge:
                self.bridge.load_weights(self.model, hf_model_path)
            else:
                self.bridge.load_hf_weights(self.model, hf_model_path)
            log_with_rank(f"Loaded HF model checkpoint from {hf_model_path} with bridge", rank=self.rank, logger=logger)

        if self.should_load_optimizer:
            assert "optimizer" in state_dict, (
                f"Optimizer state dict not found in {state_dict.keys()}. Please check the checkpoint file {local_path}."
            )
            optimizer_state_dict = state_dict["optimizer"]
            self.optimizer.load_state_dict(optimizer_state_dict)
            log_with_rank(f"Loaded optimizer checkpoint from {local_path}", rank=self.rank, logger=logger)

            # Bug 2 fix: restore max_lr / min_lr that may have been corrupted by
            # the `_filter_and_reorder_param_groups` identifier-key collision
            # during the (implicit) sharded_state_dict(is_loading=True) call.
            # With klpost1e2 (bias_predictor_lr_mult=100) the main_wd1 group
            # silently inherited the predictor's max_lr → actor diverged within
            # 4 steps post-resume.  Snapshot taken pre-generate (above).
            n_restored_lr_groups = 0
            if _pre_generate_lr_snapshot:
                for _hdo, _snap in zip(
                    _iter_optimizer_chained_subs(self.optimizer), _pre_generate_lr_snapshot
                ):
                    _groups = getattr(_hdo, "param_groups", []) or []
                    for _g, (snap_max_lr, snap_min_lr) in zip(_groups, _snap):
                        if snap_max_lr is not None and _g.get("max_lr") != snap_max_lr:
                            _g["max_lr"] = snap_max_lr
                            n_restored_lr_groups += 1
                        if snap_min_lr is not None:
                            _g["min_lr"] = snap_min_lr
                if n_restored_lr_groups > 0:
                    log_with_rank(
                        f"Restored max_lr on {n_restored_lr_groups} param group(s) after "
                        f"identifier-key collision during optimizer load_state_dict",
                        rank=self.rank,
                        logger=logger,
                    )

            # Bug 1 fix: with use_hf_checkpoint=True the bridge.load_hf_weights
            # call above leaves the model-side bias_predictor at zero-init (HF
            # safetensors don't carry the param).  The fp32 master in the
            # precision-aware optimizer holds the trained value but only flows
            # back into the bf16 model param inside optimizer.step().  Force a
            # one-shot master -> model copy here so the *next* forward (i.e. the
            # first rollout / training step after resume) sees the trained
            # predictor instead of zero.
            n_restored = _restore_bias_predictor_from_optimizer_master(self.model, self.optimizer)
            if n_restored > 0:
                log_with_rank(
                    f"Restored {n_restored} bias_predictor params from optimizer master copy",
                    rank=self.rank,
                    logger=logger,
                )

            if _optim_resume_debug_enabled():
                _dump_optimizer_state_for_debug(self.model, self.optimizer, self.rank)

            if self.use_checkpoint_opt_param_scheduler:
                assert "lr_scheduler" in state_dict, (
                    f"LR scheduler state dict not found in {state_dict.keys()}. Please check the checkpoint file "
                    f"{local_path}."
                )
                lr_scheduler_state_dict = state_dict["lr_scheduler"]
                if self.lr_scheduler is not None:
                    self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
                    log_with_rank(f"Loaded LR scheduler checkpoint from {local_path}", rank=self.rank, logger=logger)

        if self.should_load_extra:
            assert "rng_state" in state_dict, (
                f"RNG state dict not found in {state_dict.keys()}. Please check the checkpoint file {local_path}."
            )
            rng_state = state_dict["rng_state"]
            self.load_rng_states(rng_state)
            log_with_rank(f"Loaded RNG states from {local_path}", rank=self.rank, logger=logger)

        if del_local_after_load:
            try:
                os.remove(local_path) if is_non_local(local_path) else None
            except Exception as e:
                log_with_rank(
                    f"remove local resume ckpt file after loading failed, exception {e} will be ignored",
                    rank=self.rank,
                    logger=logger,
                )

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):
        # record the previous global step
        self.previous_global_step = global_step

        if not self.checkpoint_config.async_save:
            self.ensure_checkpoint_capacity(max_ckpt_to_keep)

        local_path = local_mkdir_safe(local_path)
        dist_checkpoint_path = get_dist_checkpoint_path(local_path)
        hf_config_tokenizer_path = get_hf_model_checkpoint_path(local_path)

        # Note that model weights, optimizer states, and extra states are generated
        # together in a state dict, we save them in one time
        if self.use_megatron_fsdp:
            async_save_request = self._save_megatron_fsdp_checkpoint(dist_checkpoint_path)
        elif self.use_dist_checkpointing:
            # Generate state dict for saving
            sharded_sd_metadata = self._build_sharded_state_dict_metadata()
            state_dict = self.generate_state_dict(
                self.should_save_model,
                self.should_save_optimizer,
                self.should_save_extra,
                metadata=sharded_sd_metadata,
            )
            state_dict = self._maybe_filter_peft_state_dict(state_dict)
            log_with_rank(f"Generated state dict for saving: {state_dict.keys()}", rank=self.rank, logger=logger)

            for vpp_rank, model in enumerate(self.model):
                if len(self.model) > 1:
                    model_i_keys = state_dict[f"model{vpp_rank}"].keys()
                    log_with_rank(f"Generated state dict for saving: {model_i_keys}", rank=self.rank, logger=logger)
                else:
                    log_with_rank(
                        f"Generated state dict for saving: {state_dict['model'].keys()}",
                        rank=self.rank,
                        logger=logger,
                    )
            # Start Async save if enabled
            async_save_request = save_dist_checkpointing(
                sharded_state_dict=state_dict,
                ckpt_path=dist_checkpoint_path,
                async_save=self.checkpoint_config.async_save,
                content_metadata=sharded_sd_metadata,
            )

            # Synchronize all async save requests
            if not self.checkpoint_config.async_save:
                assert async_save_request is None, "Async save request should be None when not using async save."
                torch.distributed.barrier()
        else:
            assert self.use_hf_checkpoint, "When not using distributed checkpointing, use_hf_checkpoint should be True."
            # Generate optimizer and exra state dicts
            sharded_sd_metadata = self._build_sharded_state_dict_metadata()
            state_dict = self.generate_state_dict(
                generate_model=self.should_save_model and self.peft_cls is not None,
                generate_optimizer=self.should_save_optimizer,
                generate_extra=self.should_save_extra,
                metadata=sharded_sd_metadata,
            )
            state_dict = self._maybe_filter_peft_state_dict(state_dict)
            # Save optimizer and extra states to local path
            # Start Async save if enabled
            async_save_request = save_dist_checkpointing(
                sharded_state_dict=state_dict,
                ckpt_path=dist_checkpoint_path,
                async_save=self.checkpoint_config.async_save,
                content_metadata=sharded_sd_metadata,
            )

            # Synchronize all async save requests
            if not self.checkpoint_config.async_save:
                assert async_save_request is None, "Async save request should be None when not using async save."
                torch.distributed.barrier()

        if self.should_save_model:
            if self.use_hf_checkpoint:
                # Use mbridge to save HF model checkpoint
                log_with_rank(f"Saving HF model checkpoint to {local_path} with bridge", rank=self.rank, logger=logger)
                hf_ckpt_path = get_hf_model_checkpoint_path(local_path)
                if self.vanilla_bridge:
                    extended_args = {}
                    mbridge_config = getattr(self.checkpoint_config, "mbridge_config", None) or {}
                    for sig in inspect.signature(self.bridge.save_weights).parameters:
                        if sig == "weights_path" or sig == "models":
                            continue
                        if sig in mbridge_config:
                            extended_args[sig] = mbridge_config[sig]
                    self.bridge.save_weights(self.model, hf_ckpt_path, **extended_args)
                else:
                    if self.peft_cls is not None:
                        hf_adapter_ckpt_path = os.path.join(hf_ckpt_path, "adapter")
                        self.bridge.save_hf_adapter(self.model, hf_adapter_ckpt_path, self.peft_cls)
                        log_with_rank(
                            f"Saved HF PEFT adapter checkpoint to {hf_adapter_ckpt_path}",
                            rank=self.rank,
                            logger=logger,
                            log_only_rank_0=True,
                        )
                    else:
                        self.bridge.save_hf_weights(self.model, hf_ckpt_path, strict=self.checkpoint_config.strict)

                log_with_rank(f"Saved bridge checkpoint to {hf_ckpt_path}", rank=self.rank, logger=logger)

            # Only rank 0 saves the hf config and tokenizer to huggingface path
            # No matter whether we save hf model or not
            if self.rank == 0:
                # Save tokenizer
                if self.processing_class is not None:
                    self.processing_class.save_pretrained(hf_config_tokenizer_path)
                # Save huggingface config
                self.hf_config.save_pretrained(hf_config_tokenizer_path)
                if hasattr(self.hf_config, "name_or_path") and self.hf_config.name_or_path:
                    try:
                        generation_config = GenerationConfig.from_pretrained(self.hf_config.name_or_path)
                        generation_config.save_pretrained(hf_config_tokenizer_path)
                    except Exception:
                        # if the generation config isn't available, we don't save it
                        pass
                log_with_rank(
                    f"Saved Huggingface config and tokenizer to {hf_config_tokenizer_path}",
                    rank=self.rank,
                    logger=logger,
                    log_only_rank_0=True,
                )

        if self.should_save_extra:
            if self.rank == 0:
                # Save transformer config
                print(self.transformer_config)
                transformer_config_dict = _to_json_safe_config_dict(_config_to_shallow_dict(self.transformer_config))
                transformer_config_path = get_transformer_config_checkpoint_path(local_path)
                # NOTE: With Megatron-Bridge backend, a circular import issue occurs when transformers version >= 5.4.0.
                with open(transformer_config_path, "w") as f:
                    json.dump(
                        transformer_config_dict,
                        f,
                        indent=2,
                        default=lambda o: o.to_dict() if hasattr(o, "to_dict") else o,
                    )

        if self.should_save_hf_model and not self.use_hf_checkpoint:
            # wait for everyone to dump to local
            if self.bridge is not None:
                hf_model_ckpt_path = get_hf_model_checkpoint_path(local_path)
                if self.vanilla_bridge:
                    extended_args = {}
                    mbridge_config = getattr(self.checkpoint_config, "mbridge_config", None) or {}
                    for sig in inspect.signature(self.bridge.save_weights).parameters:
                        if sig == "weights_path" or sig == "models":
                            continue
                        if sig in mbridge_config:
                            extended_args[sig] = mbridge_config[sig]
                    self.bridge.save_weights(self.model, hf_model_ckpt_path, **extended_args)
                else:
                    self.bridge.save_hf_weights(self.model, hf_model_ckpt_path)
            else:
                state_dict = self.weight_saver(
                    self.model,
                    self.hf_config,
                    dtype=self.param_dtype,
                    is_value_model=self.is_value_model,
                    tie_word_embeddings=self.share_embeddings_and_output_weights,
                )

                torch.distributed.barrier()
                if self.rank == 0:
                    hf_model_ckpt_path = get_hf_model_checkpoint_path(local_path)
                    import warnings

                    from accelerate import init_empty_weights

                    with init_empty_weights(), warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if "mistral7b-rm" in self.config.model.path:
                            from transformers import MistralForSequenceClassification

                            model = MistralForSequenceClassification.from_pretrained(
                                self.config.model.path
                            )  # use score head instead of lm_head
                            state_dict["score.weight"] = state_dict["score.weight"]
                        else:
                            from transformers import AutoModelForCausalLM

                            model = AutoModelForCausalLM.from_pretrained(self.config.model.path, torch_dtype="auto")
                    model.save_pretrained(hf_model_ckpt_path, state_dict=state_dict)
                    log_with_rank(
                        f"Saved Huggingface config and tokenizer to {hf_model_ckpt_path}",
                        rank=self.rank,
                        logger=logger,
                        log_only_rank_0=True,
                    )

                    if hdfs_path is not None:
                        log_with_rank(
                            f"Uploading checkpoint to {hdfs_path}", rank=self.rank, logger=logger, log_only_rank_0=True
                        )
                        from verl.utils import hdfs_io

                        hdfs_io.makedirs(hdfs_path, exist_ok=True)
                        hdfs_io.copy(src=hf_model_ckpt_path, dst=hdfs_path, dirs_exist_ok=True)
                        log_with_rank(
                            f"HDFS checkpoint uploaded to {hdfs_path}",
                            rank=self.rank,
                            logger=logger,
                            log_only_rank_0=True,
                        )

        def finalize_save_fn():
            # Rank 0 uploads checkpoint to HDFS if hdfs_path is provided
            log_with_rank(
                f"Dist checkpointing save completed for {dist_checkpoint_path}", rank=self.rank, logger=logger
            )
            if self.rank == 0:
                if hdfs_path is not None:
                    log_with_rank(f"Uploading checkpoint to {hdfs_path}", rank=self.rank, logger=logger)
                    from verl.utils import hdfs_io

                    hdfs_io.makedirs(hdfs_path, exist_ok=True)
                    hdfs_io.copy(src=dist_checkpoint_path, dst=hdfs_path, dirs_exist_ok=True)
                    hdfs_io.copy(src=hf_config_tokenizer_path, dst=hdfs_path, dirs_exist_ok=True)

            # update latest_checkpointed_iteration.txt when async_save is True
            if self.checkpoint_config.async_save and self.rank == 0:
                log_with_rank(
                    f"Update latest_checkpointed_iteration.txt to step {global_step}",
                    rank=self.rank,
                    logger=logger,
                )
                local_latest_checkpointed_iteration = os.path.join(
                    os.path.dirname(os.path.dirname(local_path)), "latest_checkpointed_iteration.txt"
                )
                with open(local_latest_checkpointed_iteration, "w") as f:
                    f.write(str(global_step))

            self.register_checkpoint(local_path, max_ckpt_to_keep)

        if self.checkpoint_config.async_save:
            assert async_save_request is not None, "Async save request should not be None when using async save."
            async_save_request.add_finalize_fn(finalize_save_fn)
            try:
                from megatron.core.dist_checkpointing.strategies.async_utils import AsyncCallsQueue

                AsyncCallsQueue(persistent=False).schedule_async_request(async_save_request)
            except ImportError:
                from megatron.core.dist_checkpointing.strategies.base import async_calls

                async_calls.schedule_async_request(async_save_request)
        else:
            finalize_save_fn()
