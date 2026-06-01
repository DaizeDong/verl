"""Tests for the bias_predictor master-copy backfill helper.

Background: with `use_dist_checkpointing=False` + mbridge HF save, model
weights live in HF safetensors which doesn't carry router.bias_predictor.weight.
After resume, `bridge.load_hf_weights` leaves the actor-side predictor at
zero-init. The dist_ckpt-side optimizer state IS restored, so the fp32 master
already holds the correct value. `_restore_bias_predictor_from_optimizer_master`
propagates master -> model so the very first forward after load sees the
trained predictor (otherwise router/predictive_bias_to_logits_ratio is 0 for
one step).

In real Megatron the fp32 master is keyed inside HybridDeviceOptimizer's
`param_to_fp32_param` by the *shard view* of the model parameter (i.e.
`model_param.view(-1)[start:end]`), and Megatron tags those shards with
`_is_bp_shard = True` (`distrib_optimizer.py:413`). With
`use_precision_aware_optimizer=True` `model_param.main_param` is None
(`distrib_optimizer.py:405-409`), so the helper has to walk from the
optimizer side rather than from `model.named_parameters()`. These tests build
fake optimizers around that contract.
"""

import pytest
import torch

pytest.importorskip("megatron.core")

from verl.utils.checkpoint.megatron_checkpoint_manager import (
    _all_gather_after_backfill,
    _iter_bias_predictor_params,
    _iter_bias_predictor_shards,
    _iter_optimizer_chained_subs,
    _iter_optimizer_model_chunks,
    _restore_bias_predictor_from_optimizer_master,
)


class _FakeRouter(torch.nn.Module):
    def __init__(self, hidden, n_experts):
        super().__init__()
        self.bias_predictor = torch.nn.Linear(hidden, n_experts, bias=False)
        # Mirror the Megatron-side attribute that distinguishes predictor params.
        setattr(self.bias_predictor.weight, "is_bias_predictor", True)


class _FakeMlp(torch.nn.Module):
    def __init__(self, hidden, n_experts):
        super().__init__()
        self.router = _FakeRouter(hidden, n_experts)


class _FakeLayer(torch.nn.Module):
    def __init__(self, hidden, n_experts):
        super().__init__()
        self.mlp = _FakeMlp(hidden, n_experts)


class _FakeModel(torch.nn.Module):
    def __init__(self, n_layers=3, hidden=8, n_experts=4):
        super().__init__()

        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [_FakeLayer(hidden, n_experts) for _ in range(n_layers)]
                )

        self.decoder = Decoder()


class _FakeHybridDeviceOptimizer:
    """Minimal stand-in for Megatron's HybridDeviceOptimizer.

    We mirror only what the helper introspects: `param_to_fp32_param` plus
    `param_groups` (so the helper can iterate the optimizer's shard params).
    """

    def __init__(self, param_to_fp32_param=None, param_groups=None):
        self.param_to_fp32_param = dict(param_to_fp32_param or {})
        self.param_groups = list(param_groups or [])


class _FakeChainedOptimizer:
    """Stand-in for Megatron's ChainedOptimizer."""

    def __init__(self, chained_optimizers):
        self.chained_optimizers = chained_optimizers


class _FakeDistributedOptimizer:
    """Stand-in for Megatron's DistributedOptimizer.

    Crucially, `param_to_fp32_param` lives on the *inner* HDO (`self.optimizer`),
    not on the DO itself — this is what production looks like and what the
    backfill helper has to walk through. We also expose `param_groups` as a
    property that delegates to the HDO (matching `MegatronOptimizer.param_groups`
    in `optimizer.py:287-296`).

    `model_chunks` mirrors Megatron's `DistributedOptimizer.model_chunks` so the
    backfill all-gather helper has somewhere to find the DDP-wrapped chunks.
    """

    def __init__(self, hdo, model_chunks=None):
        self.optimizer = hdo
        self.model_chunks = list(model_chunks or [])

    @property
    def param_groups(self):
        return self.optimizer.param_groups


class _FakeModelChunk:
    """Tiny stand-in for a DDP-wrapped model chunk. We only care about
    `start_param_sync(force_sync=True)` getting called once per chunk."""

    def __init__(self):
        self.sync_calls = []

    def start_param_sync(self, force_sync=False):
        self.sync_calls.append(force_sync)


def _build_predictor_shards_for_model(model, fill: float, *, dtype=None):
    """For every bias_predictor param on `model`, take a flat view of the
    parameter (this is what Megatron's `DistributedOptimizer` slices into
    `shard_model_param`), tag it `_is_bp_shard=True`, and pair it with an
    fp32 master tensor of the same numel filled with `fill`.

    The shard is a *view*, so `shard.data.copy_(...)` writes through to the
    underlying model parameter — which is exactly the property the production
    code relies on.
    """
    shards = []
    masters = []
    for _vpp, _name, param in _iter_bias_predictor_params([model]):
        if dtype is not None:
            # Replace the parameter with a same-shape tensor of the requested
            # dtype so we can exercise dtype-conversion paths. We have to
            # rebind because torch.nn.Linear's weight is fp32 by default.
            param.data = param.data.to(dtype)
        shard = param.data.view(-1)
        setattr(shard, "_is_bp_shard", True)
        master = torch.full_like(shard.detach().float(), fill, dtype=torch.float32)
        shards.append(shard)
        masters.append(master)
    return shards, masters


def _make_optimizer_with_predictor_shards(model, fill: float, *, dtype=None):
    shards, masters = _build_predictor_shards_for_model(model, fill, dtype=dtype)
    param_to_fp32 = {shard: master for shard, master in zip(shards, masters)}
    param_groups = [{"params": shards}]
    return _FakeHybridDeviceOptimizer(param_to_fp32, param_groups), shards, masters


def _zero_predictor_params(model):
    for _vpp, _name, param in _iter_bias_predictor_params([model]):
        with torch.no_grad():
            param.zero_()


def test_iter_optimizer_chained_subs_flattens_chained():
    leaf_a = _FakeHybridDeviceOptimizer()
    leaf_b = _FakeHybridDeviceOptimizer()
    chained = _FakeChainedOptimizer([leaf_a, leaf_b])
    assert list(_iter_optimizer_chained_subs(chained)) == [leaf_a, leaf_b]


def test_iter_optimizer_chained_subs_handles_nested_chains():
    leaf = _FakeHybridDeviceOptimizer()
    inner = _FakeChainedOptimizer([leaf])
    outer = _FakeChainedOptimizer([inner, leaf])
    assert list(_iter_optimizer_chained_subs(outer)) == [leaf, leaf]


def test_iter_optimizer_chained_subs_handles_none():
    assert list(_iter_optimizer_chained_subs(None)) == []


def test_iter_bias_predictor_shards_returns_only_tagged_shards():
    """The iterator must skip `param_groups` entries that aren't tagged
    `_is_bp_shard=True` (e.g. ordinary attention params sharing the same
    optimizer)."""
    model = _FakeModel(n_layers=2, hidden=4, n_experts=2)
    optimizer, predictor_shards, masters = _make_optimizer_with_predictor_shards(model, fill=0.0)
    n_predictor = len(predictor_shards)  # snapshot before mutating shared list
    # Add an ordinary (untagged) param into the same param_group. NB:
    # `optimizer.param_groups[0]["params"]` aliases the same list as
    # `predictor_shards`, so the append below also bumps len(predictor_shards).
    untagged = torch.zeros(4)
    optimizer.param_groups[0]["params"].append(untagged)
    optimizer.param_to_fp32_param[untagged] = torch.full_like(untagged.float(), 9.9)

    yielded = list(_iter_bias_predictor_shards(optimizer))
    assert len(yielded) == n_predictor
    for shard, _fp32 in yielded:
        assert getattr(shard, "_is_bp_shard", False)


def test_restore_copies_master_into_model():
    """Happy path: fp32 master flows into the shard view, which writes
    through to the model parameter's underlying storage."""
    model = _FakeModel(n_layers=4, hidden=8, n_experts=4)
    _zero_predictor_params(model)

    optimizer, shards, masters = _make_optimizer_with_predictor_shards(model, fill=0.42)

    n = _restore_bias_predictor_from_optimizer_master([model], optimizer)
    assert n == sum(1 for _ in _iter_bias_predictor_params([model]))

    for _vpp, _name, param in _iter_bias_predictor_params([model]):
        torch.testing.assert_close(param.data, torch.full_like(param.data, 0.42))


def test_restore_via_chained_optimizer():
    """Real verl R3+predictor runs use a ChainedOptimizer because the predictor
    has its own param group via ParamKey(attr='is_bias_predictor'). Make sure
    the helper walks chained_optimizers and finds the master."""
    model = _FakeModel(n_layers=2, hidden=4, n_experts=2)
    _zero_predictor_params(model)

    inner_optimizer, _, _ = _make_optimizer_with_predictor_shards(model, fill=-0.7)
    chained = _FakeChainedOptimizer([_FakeHybridDeviceOptimizer(), inner_optimizer])

    n = _restore_bias_predictor_from_optimizer_master([model], chained)
    assert n == sum(1 for _ in _iter_bias_predictor_params([model]))

    for _vpp, _name, param in _iter_bias_predictor_params([model]):
        torch.testing.assert_close(param.data, torch.full_like(param.data, -0.7))


def test_restore_via_chain_of_distributed_optimizers_wrapping_hdo():
    """Regression for the real production hierarchy:

        ChainedOptimizer
          -> [DistributedOptimizer, DistributedOptimizer]
                .optimizer = HybridDeviceOptimizer (holds param_to_fp32_param)

    Earlier `_iter_optimizer_chained_subs` only descended through
    `chained_optimizers`, stopping at the DistributedOptimizer — which has
    neither `chained_optimizers` nor `param_to_fp32_param` — so the backfill
    silently restored 0 shards on resume. This test fails on that older
    version.
    """
    model = _FakeModel(n_layers=2, hidden=4, n_experts=2)
    _zero_predictor_params(model)

    hdo, _, _ = _make_optimizer_with_predictor_shards(model, fill=0.31)
    # First chain entry has no predictor shards (mirrors the dense param group).
    empty_hdo = _FakeHybridDeviceOptimizer()
    chained = _FakeChainedOptimizer(
        [_FakeDistributedOptimizer(empty_hdo), _FakeDistributedOptimizer(hdo)]
    )

    n = _restore_bias_predictor_from_optimizer_master([model], chained)
    assert n == sum(1 for _ in _iter_bias_predictor_params([model]))

    for _vpp, _name, param in _iter_bias_predictor_params([model]):
        torch.testing.assert_close(param.data, torch.full_like(param.data, 0.31))


def test_iter_optimizer_chained_subs_descends_into_distributed_optimizer():
    """Direct contract for `_iter_optimizer_chained_subs`: when it sees a
    DistributedOptimizer-shaped leaf (no `chained_optimizers`, no
    `param_to_fp32_param`, but `.optimizer` -> HDO), it must yield the inner
    HDO so downstream lookups find the fp32 master map."""
    hdo = _FakeHybridDeviceOptimizer(param_to_fp32_param={}, param_groups=[])
    distributed = _FakeDistributedOptimizer(hdo)
    chained = _FakeChainedOptimizer([distributed])
    assert list(_iter_optimizer_chained_subs(chained)) == [hdo]


def test_all_gather_after_backfill_calls_force_sync_on_each_chunk():
    """The restore is only correct cross-rank if every DDP model_chunk gets a
    forced all-gather afterwards. Without it the rank-local shard write only
    populates 1/DP of the predictor weight buffer and the metric collapses
    proportionally (we observed 1.03e-5 vs. trained 7.3e-4 at DP=64, a ~64x
    gap matching exactly 1/DP)."""
    chunk_a, chunk_b = _FakeModelChunk(), _FakeModelChunk()
    distributed_a = _FakeDistributedOptimizer(_FakeHybridDeviceOptimizer(), [chunk_a])
    distributed_b = _FakeDistributedOptimizer(_FakeHybridDeviceOptimizer(), [chunk_b])
    chained = _FakeChainedOptimizer([distributed_a, distributed_b])

    n = _all_gather_after_backfill(chained)
    assert n == 2
    # Both chunks must have been synced exactly once with force_sync=True.
    assert chunk_a.sync_calls == [True]
    assert chunk_b.sync_calls == [True]


def test_all_gather_after_backfill_dedupes_chunks_seen_via_chain_and_sub_optimizer():
    """Real ChainedOptimizer aggregates `model_chunks` from its sub-optimizers,
    so the same chunk is reachable both directly (via Chained.model_chunks) and
    via Chained.chained_optimizers[i].model_chunks. Make sure we call
    start_param_sync once per chunk, not twice."""
    shared_chunk = _FakeModelChunk()
    distributed = _FakeDistributedOptimizer(_FakeHybridDeviceOptimizer(), [shared_chunk])
    chained = _FakeChainedOptimizer([distributed])
    # Mirror ChainedOptimizer.__init__ aggregating chunks from sub-optimizers.
    chained.model_chunks = [shared_chunk]

    n = _all_gather_after_backfill(chained)
    assert n == 1
    assert shared_chunk.sync_calls == [True]


def test_all_gather_after_backfill_skips_chunks_without_start_param_sync():
    """If we encounter something stored in `model_chunks` that doesn't expose
    the DDP `start_param_sync` interface (custom training harness, mocked
    object, etc.), skip it instead of crashing."""

    class _NoSyncChunk:
        pass

    chunk_real = _FakeModelChunk()
    distributed = _FakeDistributedOptimizer(
        _FakeHybridDeviceOptimizer(), [_NoSyncChunk(), chunk_real]
    )

    n = _all_gather_after_backfill(distributed)
    assert n == 1
    assert chunk_real.sync_calls == [True]


def test_iter_optimizer_model_chunks_walks_chained_distributed_to_chunks():
    """Direct contract: walk Chained -> DistributedOptimizer -> model_chunks."""
    chunk_a, chunk_b = _FakeModelChunk(), _FakeModelChunk()
    chained = _FakeChainedOptimizer(
        [
            _FakeDistributedOptimizer(_FakeHybridDeviceOptimizer(), [chunk_a]),
            _FakeDistributedOptimizer(_FakeHybridDeviceOptimizer(), [chunk_b]),
        ]
    )
    chunks = list(_iter_optimizer_model_chunks(chained))
    assert chunks == [chunk_a, chunk_b]


def test_max_lr_snapshot_restore_logic():
    """Pin the load-time max_lr / min_lr snapshot-restore that compensates for
    Megatron's `_filter_and_reorder_param_groups` collapsing groups whose
    identifier-tuple `(wd_mult, lr_mult, is_expert_parallel, is_decoupled_lr)`
    happens to be identical.

    In real verl runs with `config_overrides={ParamKey(attr='is_bias_predictor'):
    override}`, `_get_param_groups` hardcodes lr_mult=1.0 and
    is_decoupled_lr=False on every group, so the dense main_wd1 group
    (1.0, 1.0, F, F) collides with the bias_predictor override group
    (1.0, 1.0, F, F). On resume the predictor's max_lr=1e-4 overwrites the
    main_wd1 entry in `loaded_groups_map`, so the main params end up driven at
    100x the intended lr_mult. We capture max_lr/min_lr from the freshly
    constructed optimizer (where `_get_param_groups` populated them correctly)
    and restore them after load_state_dict. This test simulates load_state_dict
    corruption and verifies the restore.
    """
    # Build a chain that mirrors the real production layout
    # (ChainedOptimizer -> DistributedOptimizer -> HybridDeviceOptimizer) so
    # the snapshot walk reaches each HDO and matches groups by position.
    hdo_main = _FakeHybridDeviceOptimizer(
        param_to_fp32_param={},
        param_groups=[
            {"params": [], "max_lr": 1e-6, "min_lr": 0.0, "wd_mult": 1.0},  # main_wd1
            {"params": [], "max_lr": 1e-6, "min_lr": 0.0, "wd_mult": 0.0},  # main_wd0
            {"params": [], "max_lr": 1e-4, "min_lr": 0.0, "wd_mult": 1.0},  # predictor
        ],
    )
    hdo_expert = _FakeHybridDeviceOptimizer(
        param_to_fp32_param={},
        param_groups=[{"params": [], "max_lr": 1e-6, "min_lr": 0.0, "wd_mult": 1.0}],
    )
    distributed_main = _FakeDistributedOptimizer(hdo_main, [_FakeModelChunk()])
    distributed_expert = _FakeDistributedOptimizer(hdo_expert, [_FakeModelChunk()])
    chained = _FakeChainedOptimizer([distributed_main, distributed_expert])

    # Snapshot the init-time max_lr / min_lr per (HDO, group) — mirrors the
    # snapshot block inserted just before optimizer.load_state_dict.
    snapshot = [
        [(g.get("max_lr"), g.get("min_lr")) for g in (getattr(hdo, "param_groups", []) or [])]
        for hdo in _iter_optimizer_chained_subs(chained)
    ]

    # Simulate the collision damage: Megatron's _filter_and_reorder collapses
    # main_wd1 + predictor and assigns predictor's max_lr (1e-4) to BOTH.
    hdo_main.param_groups[0]["max_lr"] = 1e-4  # corrupted
    hdo_main.param_groups[0]["lr"] = 1e-4  # scheduler would later pick this up
    # main_wd0 (wd_mult=0) does NOT collide, stays correct.
    # predictor stays at 1e-4 (correct).

    # Restore — mirrors the post-load restoration block.
    n_restored = 0
    for hdo, snap in zip(_iter_optimizer_chained_subs(chained), snapshot):
        groups = getattr(hdo, "param_groups", []) or []
        for g, (snap_max_lr, snap_min_lr) in zip(groups, snap):
            if snap_max_lr is not None and g.get("max_lr") != snap_max_lr:
                g["max_lr"] = snap_max_lr
                n_restored += 1
            if snap_min_lr is not None:
                g["min_lr"] = snap_min_lr

    assert n_restored == 1, "exactly the main_wd1 group should have been corrupted+restored"
    assert hdo_main.param_groups[0]["max_lr"] == 1e-6, "main_wd1 max_lr restored to init value"
    assert hdo_main.param_groups[1]["max_lr"] == 1e-6, "main_wd0 max_lr untouched"
    assert hdo_main.param_groups[2]["max_lr"] == 1e-4, "predictor max_lr correct, untouched"
    assert hdo_expert.param_groups[0]["max_lr"] == 1e-6, "expert HDO untouched"


def test_iter_optimizer_model_chunks_does_not_touch_chained_optimizer_dot_optimizer():
    """Regression: real `ChainedOptimizer.optimizer` is a property that asserts
    `len(chained_optimizers) == 1` (`optimizer.py:1104-1112`). Production verl
    chains have at least two sub-optimizers (dense + expert, plus
    predictor-override groups), so reading `.optimizer` raises AssertionError
    and crashes the resume path right after the bias_predictor backfill. This
    test mirrors that by having `.optimizer` blow up; the walker must reach
    every chunk without triggering it."""

    class _AssertingChainedOptimizer:
        def __init__(self, chained_optimizers, model_chunks):
            self.chained_optimizers = chained_optimizers
            self.model_chunks = list(model_chunks)

        @property
        def optimizer(self):
            raise AssertionError(
                "ChainedOptimizer has more than one optimizer when accessing self.optimizer"
            )

    chunk_a, chunk_b = _FakeModelChunk(), _FakeModelChunk()
    sub_a = _FakeDistributedOptimizer(_FakeHybridDeviceOptimizer(), [chunk_a])
    sub_b = _FakeDistributedOptimizer(_FakeHybridDeviceOptimizer(), [chunk_b])
    chained = _AssertingChainedOptimizer([sub_a, sub_b], [chunk_a, chunk_b])

    # Must not raise; must reach both chunks.
    chunks = list(_iter_optimizer_model_chunks(chained))
    assert {id(c) for c in chunks} == {id(chunk_a), id(chunk_b)}

    n = _all_gather_after_backfill(chained)
    assert n == 2
    assert chunk_a.sync_calls == [True]
    assert chunk_b.sync_calls == [True]


def test_restore_handles_dtype_conversion_on_copy():
    """Master is always fp32, model param may be bf16. copy_ must downcast."""
    model = _FakeModel(n_layers=1, hidden=4, n_experts=2)
    optimizer, _, _ = _make_optimizer_with_predictor_shards(
        model, fill=1.25, dtype=torch.bfloat16
    )
    _zero_predictor_params(model)

    n = _restore_bias_predictor_from_optimizer_master([model], optimizer)
    assert n == 1

    for _vpp, _name, param in _iter_bias_predictor_params([model]):
        assert param.dtype == torch.bfloat16
        # bf16 may quantize the magnitude slightly, so just check it's roughly right.
        assert (param.data - 1.25).abs().max().item() < 0.02


def test_restore_is_noop_when_optimizer_lacks_master_map():
    """Optimizers we don't know how to introspect must not crash; the helper
    returns 0 and leaves params untouched."""
    model = _FakeModel(n_layers=1)
    snapshot = {
        name: param.detach().clone()
        for _vpp, name, param in _iter_bias_predictor_params([model])
    }

    class _UnknownOptimizer:
        pass

    n = _restore_bias_predictor_from_optimizer_master([model], _UnknownOptimizer())
    assert n == 0
    for _vpp, name, param in _iter_bias_predictor_params([model]):
        torch.testing.assert_close(param.data, snapshot[name])


def test_restore_is_noop_when_no_predictor_shards_tagged():
    """If the optimizer holds an HDO but no shard is tagged `_is_bp_shard`
    (e.g. non-R3 runs, or a Megatron build without the predictor tag), the
    helper returns 0 silently.

    Specifically a model with NO bias_predictor at all (non-R3 run): the
    function must return 0 immediately *without* triggering an all-gather —
    we don't want to perturb DP communication on plain runs."""

    # Model with no bias_predictor module at all (non-R3 case).
    plain_model = torch.nn.Linear(4, 2)
    untagged = torch.zeros(4)
    chunk = _FakeModelChunk()
    optimizer = _FakeDistributedOptimizer(
        _FakeHybridDeviceOptimizer(
            param_to_fp32_param={untagged: torch.full_like(untagged.float(), 1.0)},
            param_groups=[{"params": [untagged]}],
        ),
        [chunk],
    )
    n = _restore_bias_predictor_from_optimizer_master([plain_model], optimizer)
    assert n == 0
    # No predictor in model -> we must NOT issue the all-gather collective.
    assert chunk.sync_calls == []


def test_restore_runs_force_sync_even_when_local_rank_has_no_shards():
    """Regression for the NCCL timeout in job 1612473.

    In production not every rank owns a predictor shard locally (shard
    ownership depends on bucket packing — we observed ~39/64 ranks with
    shards). `_all_gather_after_backfill` issues a DP collective via
    `start_param_sync(force_sync=True)`, so the ~25 ranks without local
    shards MUST still participate or the collective hangs until NCCL
    times out (1800s). This test pins that contract: the helper must
    still call `start_param_sync` on every model_chunk whenever the model
    itself has a bias_predictor, even if this rank's optimizer doesn't
    expose a tagged shard."""

    # Model HAS predictor params (R3+predictor run), but the optimizer
    # exposes no `_is_bp_shard`-tagged shards on *this* rank.
    model = _FakeModel(n_layers=2, hidden=4, n_experts=2)
    chunk = _FakeModelChunk()
    optimizer = _FakeDistributedOptimizer(
        _FakeHybridDeviceOptimizer(param_to_fp32_param={}, param_groups=[]),
        [chunk],
    )

    n = _restore_bias_predictor_from_optimizer_master([model], optimizer)
    assert n == 0  # no local shards copied
    assert chunk.sync_calls == [True], (
        "force_sync must run on this rank to keep the DP collective in lockstep "
        "with peer ranks that do own predictor shards"
    )


def test_restore_skips_shards_missing_from_master_map():
    """If a tagged shard exists but isn't keyed in `param_to_fp32_param`
    (corrupt/incomplete optimizer state), the helper skips it instead of
    crashing — those shards stay at the loaded model value."""
    model = _FakeModel(n_layers=3, hidden=4, n_experts=2)
    optimizer, shards, masters = _make_optimizer_with_predictor_shards(model, fill=0.9)
    _zero_predictor_params(model)

    # Drop one shard from the master map (still tagged + still in param_groups).
    dropped_shard = shards[0]
    optimizer.param_to_fp32_param.pop(dropped_shard)

    n = _restore_bias_predictor_from_optimizer_master([model], optimizer)
    assert n == len(shards) - 1

    # The dropped shard's underlying model param should remain at zero;
    # the rest should have been overwritten with 0.9.
    iters = list(_iter_bias_predictor_params([model]))
    # ordering of `_iter_bias_predictor_params` matches construction order,
    # which matches `_make_optimizer_with_predictor_shards`'s shard list.
    _, _, dropped_param = iters[0]
    assert dropped_param.abs().sum().item() == 0.0, "dropped shard's param must remain zero"
    for _vpp, _name, param in iters[1:]:
        torch.testing.assert_close(param.data, torch.full_like(param.data, 0.9))
