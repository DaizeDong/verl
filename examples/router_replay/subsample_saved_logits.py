#!/usr/bin/env python
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Offline subsample already-saved router logits files to shrink disk usage for analysis.

For each step directory under router_logits/, the saver writes:
  - log_prob_{step}_tp{tp}_pp{pp}.pt      (one per tp/pp rank)
  - training_{step}_mini{N}_tp{tp}_pp{pp}.pt  (N = 0..K; K varies per run)

Each .pt file is a dict with fields:
  compute_log_prob: {layer_idx: [num_tokens, num_experts]}  (log_prob file only)
  training:         {layer_idx: [num_tokens, num_experts]}  (training_mini* file only)
  predictive_bias:  {layer_idx: [num_tokens, 1, num_experts]} (optional; log_prob file)
  router_weights:   {layer_idx: Tensor}
  global_token_ids: [num_tokens] (int64)
  step, tp_rank, pp_rank, dp_world_size

Subsampling strategy:
  1. For each log_prob file: uniformly (linspace) sample `rate` fraction of tokens.
     Record the sampled global_token_ids in a shared set.
  2. For each training_mini{N} file (in order): keep only rows whose global_token_id
     is in the sampled set. This preserves the id↔row correspondence used by
     analyze_saved_logits.py.

The number of mini-step files per step is auto-detected from the filenames.

Output layout mirrors the input but under --out_dir.

Example:
  python subsample_saved_logits.py \\
      --in_dir  /path/to/router_logits \\
      --out_dir /path/to/router_logits_sampled \\
      --rate    0.1 \\
      --steps   10 20 30
"""

import argparse
import os
import re
import sys
from pathlib import Path

import torch


LOG_PROB_RE = re.compile(r"^log_prob_(\d+)_tp(\d+)_pp(\d+)\.pt$")
TRAINING_RE = re.compile(r"^training_(\d+)_mini(\d+)_tp(\d+)_pp(\d+)\.pt$")


def discover_steps(in_dir: Path) -> list[int]:
    return sorted(int(p.name) for p in in_dir.iterdir() if p.is_dir() and p.name.isdigit())


def discover_tp_pp(step_dir: Path) -> list[tuple[int, int]]:
    """Find unique (tp, pp) pairs that have a log_prob file."""
    pairs = set()
    for fname in os.listdir(step_dir):
        m = LOG_PROB_RE.match(fname)
        if m:
            _, tp, pp = m.groups()
            pairs.add((int(tp), int(pp)))
    return sorted(pairs)


def discover_minis(step_dir: Path, tp: int, pp: int) -> list[int]:
    """Find all mini indices for this (step, tp, pp)."""
    minis = []
    for fname in os.listdir(step_dir):
        m = TRAINING_RE.match(fname)
        if m:
            _, mini, f_tp, f_pp = m.groups()
            if int(f_tp) == tp and int(f_pp) == pp:
                minis.append(int(mini))
    return sorted(minis)


def subsample_logits_dict(d: dict, indices: torch.Tensor, expected_n: int | None = None) -> dict:
    """Index into dim-0 of every tensor value. Non-tensor/empty entries kept as-is.

    If expected_n is provided, only subsample tensors whose dim-0 matches it — leave
    mismatched tensors unchanged (R3 predictive_bias can have a different token count
    than compute_log_prob because it's populated from rollout/log_prob in a separate
    codepath).
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] > 0:
            if expected_n is not None and v.shape[0] != expected_n:
                # Keep unchanged; row count doesn't match the sampling indices domain.
                out[k] = v
            else:
                out[k] = v.index_select(0, indices)
        else:
            out[k] = v
    return out


def _atomic_replace(tmp_path: Path, dst_path: Path):
    """torch.save to tmp then atomic rename. Both paths must be on the same filesystem."""
    os.replace(tmp_path, dst_path)


def subsample_log_prob_file(in_path: Path, out_path: Path, rate: float) -> tuple[set[int], int, int]:
    """Uniformly sample `rate` fraction of tokens. Returns (sampled_id_set, n_before, n_after)."""
    data = torch.load(in_path, map_location="cpu", weights_only=False)
    gtids = data.get("global_token_ids")
    if not isinstance(gtids, torch.Tensor) or gtids.numel() == 0:
        print(f"[WARN] {in_path.name}: global_token_ids missing/empty — copying unchanged")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, out_path)
        return set(), 0, 0

    n = int(gtids.shape[0])
    if 0.0 < rate < 1.0 and n > 0:
        num_keep = max(1, int(round(n * rate)))
        if num_keep < n:
            idx = torch.round(torch.linspace(0, n - 1, num_keep)).long().clamp(0, n - 1)
        else:
            idx = torch.arange(n, dtype=torch.long)
    else:
        idx = torch.arange(n, dtype=torch.long)

    new_data = dict(data)
    new_data["global_token_ids"] = gtids.index_select(0, idx)
    if isinstance(data.get("compute_log_prob"), dict):
        new_data["compute_log_prob"] = subsample_logits_dict(data["compute_log_prob"], idx, expected_n=n)
    if isinstance(data.get("predictive_bias"), dict):
        new_data["predictive_bias"] = subsample_logits_dict(data["predictive_bias"], idx, expected_n=n)
    # training/router_weights left untouched (empty in log_prob file)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(new_data, tmp_path)
    _atomic_replace(tmp_path, out_path)
    sampled_ids = set(new_data["global_token_ids"].tolist())
    return sampled_ids, n, int(new_data["global_token_ids"].shape[0])


def filter_training_file(in_path: Path, out_path: Path, sampled_ids: set[int]) -> tuple[int, int]:
    """Keep only rows whose global_token_id is in sampled_ids. Returns (n_before, n_after)."""
    data = torch.load(in_path, map_location="cpu", weights_only=False)
    gtids = data.get("global_token_ids")
    if not isinstance(gtids, torch.Tensor) or gtids.numel() == 0:
        print(f"[WARN] {in_path.name}: global_token_ids missing/empty — copying unchanged")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, out_path)
        return 0, 0

    n = int(gtids.shape[0])
    id_list = gtids.tolist()
    mask = torch.tensor([tid in sampled_ids for tid in id_list], dtype=torch.bool)
    idx = mask.nonzero(as_tuple=False).squeeze(-1)

    new_data = dict(data)
    new_data["global_token_ids"] = gtids.index_select(0, idx)
    if isinstance(data.get("training"), dict):
        new_data["training"] = subsample_logits_dict(data["training"], idx, expected_n=n)
    if isinstance(data.get("predictive_bias"), dict):
        new_data["predictive_bias"] = subsample_logits_dict(data["predictive_bias"], idx, expected_n=n)
    # compute_log_prob/router_weights left untouched (empty in training file)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(new_data, tmp_path)
    _atomic_replace(tmp_path, out_path)
    return n, int(new_data["global_token_ids"].shape[0])


SAMPLED_MARKER = ".sampled"


def process_step(step: int, in_dir: Path, out_dir: Path, rate: float, skip_if_marked: bool = True):
    src = in_dir / str(step)
    dst = out_dir / str(step)
    pairs = discover_tp_pp(src)
    if not pairs:
        print(f"[step {step}] no log_prob files found, skipping")
        return

    marker = dst / SAMPLED_MARKER
    if skip_if_marked and marker.exists():
        print(f"[step {step}] already sampled (marker {SAMPLED_MARKER} present), skipping")
        return

    print(f"\n===== step {step} =====")
    for tp, pp in pairs:
        # 1. Subsample log_prob, record sampled IDs
        log_name = f"log_prob_{step}_tp{tp}_pp{pp}.pt"
        sampled_ids, n_lp_before, n_lp_after = subsample_log_prob_file(
            src / log_name, dst / log_name, rate
        )
        print(f"  [tp{tp}pp{pp}] log_prob: {n_lp_before} -> {n_lp_after} tokens")

        # 2. Filter each training mini file using the sampled set
        minis = discover_minis(src, tp, pp)
        for mini in minis:
            tr_name = f"training_{step}_mini{mini}_tp{tp}_pp{pp}.pt"
            n_before, n_after = filter_training_file(src / tr_name, dst / tr_name, sampled_ids)
            print(f"  [tp{tp}pp{pp}] training_mini{mini}: {n_before} -> {n_after} tokens")

    # Mark completion so a future batch run skips this step.
    dst.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"rate={rate}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_dir", required=True, help="Source router_logits directory")
    ap.add_argument("--out_dir", default=None,
                    help="Destination directory for subsampled files. Default: same as in_dir (in-place overwrite)")
    ap.add_argument("--in_place", action="store_true",
                    help="Overwrite in_dir (equivalent to --out_dir=in_dir). Each file is written atomically via .tmp + rename.")
    ap.add_argument("--rate", type=float, default=0.1, help="Fraction of log_prob tokens to keep (default 0.1)")
    ap.add_argument("--steps", type=int, nargs="*", default=None,
                    help="Specific step numbers to process; default = all steps under in_dir")
    ap.add_argument("--no_skip", action="store_true",
                    help="Re-process steps even if a .sampled marker exists")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    if args.in_place or args.out_dir is None:
        out_dir = in_dir
    else:
        out_dir = Path(args.out_dir)
    if not in_dir.is_dir():
        sys.exit(f"in_dir not a directory: {in_dir}")

    steps = args.steps if args.steps else discover_steps(in_dir)
    if not steps:
        print(f"[WARN] no numeric step subdirectories found in {in_dir} — nothing to do")
        return

    print(f"in_dir:  {in_dir}")
    print(f"out_dir: {out_dir}{' (in-place)' if out_dir == in_dir else ''}")
    print(f"subsampling rate: {args.rate}")
    print(f"steps: {steps}")

    # Experiment-level marker: skip the whole directory if we've already processed it.
    dir_marker = out_dir / SAMPLED_MARKER
    if not args.no_skip and dir_marker.exists():
        print(f"[SKIP] {dir_marker} exists — directory already subsampled.")
        return

    for step in steps:
        process_step(step, in_dir, out_dir, args.rate, skip_if_marked=not args.no_skip)

    out_dir.mkdir(parents=True, exist_ok=True)
    dir_marker.write_text(f"rate={args.rate}\nsteps={steps}\n")
    print(f"\nDone. Marker: {dir_marker}")


if __name__ == "__main__":
    main()
