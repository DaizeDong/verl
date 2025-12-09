#!/usr/bin/env python
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
重写的router logits分析脚本。

功能:
1. 读取目录下的log_prob_{N}_tp{M}_pp{K}.pt和training_{N}_tp{M}_pp{K}.pt文件对
2. 对每对文件进行分析:
   - 解析router_logits tensor
   - 验证token数量一致性
   - 计算整体logits的entropy和topK logits的entropy
   - 获取topK experts的索引
   - 计算log_prob和training之间topK experts索引的差异
   - 分析差异与entropy的相关性并绘图
   - 绘制其他MoE相关的图

使用方法:
    python analyze_saved_logits.py --save_dir /path/to/saved/logits --topk 2 --output_dir /path/to/output
"""

import argparse
import glob
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use('Agg')  # 使用非交互式backend，适合多线程环境
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.multiprocessing as mp
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

# 自动检测GPU数量
NUM_GPUS = torch.cuda.device_count()


def print_gpu_info():
    """打印GPU信息。"""
    print("\n" + "=" * 80)
    print("GPU 信息")
    print("=" * 80)
    if NUM_GPUS == 0:
        print("未检测到可用GPU，将使用CPU计算")
    else:
        print(f"检测到 {NUM_GPUS} 个GPU:")
        for i in range(NUM_GPUS):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024 ** 3
            print(f"  GPU {i}: {props.name}")
            print(f"         内存: {memory_gb:.1f} GB")
            print(f"         计算能力: {props.major}.{props.minor}")
    print("=" * 80)


def get_gpu_memory_usage():
    """获取GPU内存使用情况。"""
    if NUM_GPUS == 0:
        return []

    usage = []
    for i in range(NUM_GPUS):
        allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
        usage.append({
            'gpu_id': i,
            'allocated_gb': allocated,
            'reserved_gb': reserved,
        })
    return usage


def estimate_layers_per_gpu(sample_layer_shape: tuple, num_experts: int,
                            available_memory_gb: float = None) -> int:
    """
    估算每张GPU可以处理的层数。
    
    Args:
        sample_layer_shape: 单层logits的形状 (num_tokens, num_experts)
        num_experts: 专家数量
        available_memory_gb: 可用显存（GB），如果None则自动检测
        
    Returns:
        每张GPU建议处理的层数
    """
    if NUM_GPUS == 0:
        return 1

    num_tokens, _ = sample_layer_shape

    # 单层数据量估算（float32）
    # 输入: log_prob_logits + training_logits = 2 * num_tokens * num_experts * 4 bytes
    # 中间结果: log_softmax, probs, topk_ids, topk_logits等 ≈ 3倍输入
    # 输出: entropy, topk_entropy, topk_diff ≈ 0.5倍输入
    # 总计约 5.5倍输入数据
    layer_size_gb = (num_tokens * num_experts * 4 * 5.5) / (1024 ** 3)

    # 获取可用显存
    if available_memory_gb is None:
        # 使用第一张GPU的显存作为参考
        props = torch.cuda.get_device_properties(0)
        total_memory_gb = props.total_memory / (1024 ** 3)
        # 保留30%显存用于其他操作和缓存
        available_memory_gb = total_memory_gb * 0.7

    # 计算能处理的层数
    layers_per_gpu = int(available_memory_gb / layer_size_gb)

    # 至少处理1层，最多不超过总层数，但也要考虑性能
    # 对于H200 (139GB)，单层1.2GB，理论上可放100+层，但批量太大可能影响性能
    # 限制在合理范围内：最少1层，最多50层（或总层数）
    return max(1, min(layers_per_gpu, 50))


def process_layers_batch(log_prob_batch: torch.Tensor, training_batch: torch.Tensor,
                         k: int, gpu_id: int) -> Dict[str, torch.Tensor]:
    """
    批量处理多层（合并计算，然后split结果）。
    
    Args:
        log_prob_batch: 合并后的log_prob logits [total_tokens, num_experts]
        training_batch: 合并后的training logits [total_tokens, num_experts]
        k: topK的K值
        gpu_id: GPU ID
        
    Returns:
        包含所有结果的字典（所有tensor在CPU上）
    """
    device = f'cuda:{gpu_id}'
    if not log_prob_batch.is_cuda:
        log_prob_batch = log_prob_batch.to(device, non_blocking=True)
        training_batch = training_batch.to(device, non_blocking=True)

    t0 = time.time()
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        # 1. 批量计算整体entropy
        log_prob_log_softmax = torch.log_softmax(log_prob_batch, dim=-1)
        training_log_softmax = torch.log_softmax(training_batch, dim=-1)
        log_prob_probs = torch.exp(log_prob_log_softmax)
        training_probs = torch.exp(training_log_softmax)

        log_prob_entropy = -(log_prob_probs * log_prob_log_softmax).sum(dim=-1)
        training_entropy = -(training_probs * training_log_softmax).sum(dim=-1)

        # 2. 批量计算topK
        _, log_prob_topk_ids = torch.topk(log_prob_batch, k, dim=-1)
        _, training_topk_ids = torch.topk(training_batch, k, dim=-1)
        log_prob_topk_logits = torch.gather(log_prob_batch, -1, log_prob_topk_ids)
        training_topk_logits = torch.gather(training_batch, -1, training_topk_ids)

        # 3. 批量计算topK entropy
        log_prob_topk_log_softmax = torch.log_softmax(log_prob_topk_logits, dim=-1)
        training_topk_log_softmax = torch.log_softmax(training_topk_logits, dim=-1)
        log_prob_topk_probs = torch.exp(log_prob_topk_log_softmax)
        training_topk_probs = torch.exp(training_topk_log_softmax)

        log_prob_topk_entropy = -(log_prob_topk_probs * log_prob_topk_log_softmax).sum(dim=-1)
        training_topk_entropy = -(training_topk_probs * training_topk_log_softmax).sum(dim=-1)

        # 4. 批量计算topK差异（完全向量化）
        log_prob_expanded = log_prob_topk_ids.unsqueeze(-1)  # [total_tokens, k, 1]
        training_expanded = training_topk_ids.unsqueeze(1)  # [total_tokens, 1, k]
        matches = (log_prob_expanded == training_expanded)
        log_prob_match_count = (matches.sum(dim=-1) > 0).sum(dim=-1)
        training_match_count = (matches.sum(dim=-2) > 0).sum(dim=-1)
        # 除以2，因为对称差中每个差异的专家被计算了两次
        topk_diff = ((k - log_prob_match_count) + (k - training_match_count)) / 2.0

        stream.synchronize()

    total_time = time.time() - t0

    # 转回CPU
    results = {
        'log_prob_entropy': log_prob_entropy.cpu(),
        'training_entropy': training_entropy.cpu(),
        'log_prob_topk_entropy': log_prob_topk_entropy.cpu(),
        'training_topk_entropy': training_topk_entropy.cpu(),
        'topk_diff': topk_diff.cpu(),
        'batch_time': total_time,
    }

    return results


def print_gpu_memory_stats():
    """打印GPU内存统计。"""
    usage = get_gpu_memory_usage()
    if not usage:
        return

    print("\n当前GPU内存使用:")
    for info in usage:
        print(f"  GPU {info['gpu_id']}: "
              f"已分配 {info['allocated_gb']:.2f} GB, "
              f"已保留 {info['reserved_gb']:.2f} GB")


print_gpu_info()

# 设置multiprocessing启动方法
if NUM_GPUS > 0:
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

# 性能统计
PERF_STATS = {
    'entropy_time': [],
    'topk_entropy_time': [],
    'topk_diff_time': [],
}


def print_performance_stats():
    """打印性能统计信息。"""
    if not any(PERF_STATS.values()):
        return

    print("\n" + "=" * 80)
    print("性能统计")
    print("=" * 80)

    total_all = 0
    for name, times in PERF_STATS.items():
        if times:
            total_time = sum(times)
            avg_time = total_time / len(times)
            total_all += total_time
            print(f"{name:25s}: 总耗时 {total_time:6.2f}s, 平均 {avg_time:.4f}s/层, {len(times)}层")

    if total_all > 0:
        print(f"\n{'总计':25s}: {total_all:.2f}s")

        # 打印各部分占比
        print("\n各模块耗时占比:")
        for name, times in PERF_STATS.items():
            if times:
                total_time = sum(times)
                percentage = (total_time / total_all) * 100
                print(f"  {name:23s}: {percentage:5.1f}%")

    print("=" * 80)

    # 打印最终GPU内存统计
    if NUM_GPUS > 0:
        print_gpu_memory_stats()


def split_data_for_gpus(data: torch.Tensor, num_gpus: int) -> List[torch.Tensor]:
    """
    将数据按第一维（通常是token维度）分割到多个GPU。
    
    Args:
        data: 待分割的数据 [num_tokens, ...]
        num_gpus: GPU数量
        
    Returns:
        分割后的数据列表
    """
    if num_gpus <= 1:
        return [data]

    num_tokens = data.shape[0]
    chunk_size = (num_tokens + num_gpus - 1) // num_gpus

    chunks = []
    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_tokens)
        if start_idx < end_idx:
            chunks.append(data[start_idx:end_idx])

    return chunks


def entropy_on_gpu(logits: torch.Tensor, gpu_id: int, dim: int = -1,
                   keep_on_gpu: bool = False) -> torch.Tensor:
    """
    在指定GPU上计算熵（优化版：使用log_softmax，更高效）。
    
    Args:
        logits: logits张量（已在GPU上）
        gpu_id: GPU ID
        dim: 计算熵的维度
        keep_on_gpu: 是否保持在GPU上（默认False，转回CPU）
        
    Returns:
        熵值（在CPU或GPU上）
    """
    device = f'cuda:{gpu_id}'
    if not logits.is_cuda:
        logits = logits.to(device)

    # 使用log_softmax更高效：避免先softmax再log
    log_probs = torch.log_softmax(logits, dim=dim)
    probs = torch.exp(log_probs)

    # 计算熵：-sum(p * log(p))
    entropy_result = -(probs * log_probs).sum(dim=dim)

    return entropy_result if keep_on_gpu else entropy_result.cpu()


def entropy(probs: torch.Tensor, dim: int = -1, use_gpu: bool = True) -> torch.Tensor:
    """
    计算概率分布的熵（支持多GPU加速）。
    
    Args:
        probs: 概率分布张量或logits
        dim: 计算熵的维度
        use_gpu: 是否使用GPU加速
        
    Returns:
        熵值
    """
    if not use_gpu or NUM_GPUS == 0:
        # CPU计算
        if probs.max() > 1.0 or probs.min() < 0.0:
            probs = torch.softmax(probs, dim=dim)
        probs = torch.clamp(probs, min=1e-10)
        return -(probs * torch.log(probs)).sum(dim=dim)

    # 多GPU并行计算
    chunks = split_data_for_gpus(probs, NUM_GPUS)
    results = []

    # 使用torch.cuda.Stream并行处理
    streams = [torch.cuda.Stream(device=f'cuda:{i}') for i in range(len(chunks))]

    for i, chunk in enumerate(chunks):
        gpu_id = i % NUM_GPUS
        with torch.cuda.stream(streams[i]):
            result = entropy_on_gpu(chunk, gpu_id, dim)
            results.append(result)

    # 等待所有流完成
    for stream in streams:
        stream.synchronize()

    return torch.cat(results, dim=0)


def topk_entropy_on_gpu(logits: torch.Tensor, k: int, gpu_id: int,
                        keep_on_gpu: bool = False) -> torch.Tensor:
    """
    在指定GPU上计算topK熵（优化版：使用log_softmax，更高效）。
    
    Args:
        logits: 原始logits（已在GPU上）
        k: topK的K值
        gpu_id: GPU ID
        keep_on_gpu: 是否保持在GPU上（默认False，转回CPU）
        
    Returns:
        topK的熵（在CPU或GPU上）
    """
    device = f'cuda:{gpu_id}'
    if not logits.is_cuda:
        logits = logits.to(device)

    # 获取topK的logits
    topk_logits, _ = torch.topk(logits, k, dim=-1)

    # 使用log_softmax更高效
    topk_log_probs = torch.log_softmax(topk_logits, dim=-1)
    topk_probs = torch.exp(topk_log_probs)

    # 计算熵
    result = -(topk_probs * topk_log_probs).sum(dim=-1)

    return result if keep_on_gpu else result.cpu()


def topk_entropy(logits: torch.Tensor, k: int, use_gpu: bool = True) -> torch.Tensor:
    """
    计算topK logits的熵（支持多GPU加速）。
    
    Args:
        logits: 原始logits [num_tokens, num_experts]
        k: topK的K值
        use_gpu: 是否使用GPU加速
        
    Returns:
        topK的熵 [num_tokens]
    """
    if not use_gpu or NUM_GPUS == 0:
        # CPU计算
        topk_logits, _ = torch.topk(logits, k, dim=-1)
        topk_probs = torch.softmax(topk_logits, dim=-1)
        topk_probs = torch.clamp(topk_probs, min=1e-10)
        return -(topk_probs * torch.log(topk_probs)).sum(dim=-1)

    # 多GPU并行计算
    chunks = split_data_for_gpus(logits, NUM_GPUS)
    results = []

    # 使用torch.cuda.Stream并行处理
    streams = [torch.cuda.Stream(device=f'cuda:{i}') for i in range(len(chunks))]

    for i, chunk in enumerate(chunks):
        gpu_id = i % NUM_GPUS
        with torch.cuda.stream(streams[i]):
            result = topk_entropy_on_gpu(chunk, k, gpu_id)
            results.append(result)

    # 等待所有流完成
    for stream in streams:
        stream.synchronize()

    return torch.cat(results, dim=0)


def find_file_pairs(save_dir: str) -> List[Tuple[str, str, str, str, str]]:
    """
    查找log_prob和training文件对。
    
    文件名格式:
    - log_prob_{N}_tp{M}_pp{K}.pt
    - training_{N}_tp{M}_pp{K}.pt
    
    Returns:
        List of (step_N, tp_M, pp_K, log_prob_file, training_file)
    """
    all_files = glob.glob(os.path.join(save_dir, "*.pt"))

    # 解析文件名
    log_prob_pattern = re.compile(r"log_prob_(\d+)_tp(\d+)_pp(\d+)\.pt")
    training_pattern = re.compile(r"training_(\d+)_tp(\d+)_pp(\d+)\.pt")

    log_prob_files = {}
    training_files = {}

    for filepath in all_files:
        filename = os.path.basename(filepath)

        match = log_prob_pattern.match(filename)
        if match:
            step, tp, pp = match.groups()
            key = (step, tp, pp)
            log_prob_files[key] = filepath
            continue

        match = training_pattern.match(filename)
        if match:
            step, tp, pp = match.groups()
            key = (step, tp, pp)
            training_files[key] = filepath

    # 找到匹配的文件对
    pairs = []
    for key in log_prob_files:
        if key in training_files:
            step, tp, pp = key
            pairs.append((step, tp, pp, log_prob_files[key], training_files[key]))

    # 按照step, tp, pp的整数值排序
    return sorted(pairs, key=lambda x: (int(x[0]), int(x[1]), int(x[2])))


def load_logits_data(filepath: str, use_weights_only: bool = False,
                     target_device: str = 'cpu') -> Dict:
    """
    加载保存的logits数据（优化版，支持直接加载到GPU）。
    
    Args:
        filepath: 文件路径
        use_weights_only: 是否只加载权重（更快更安全，需要PyTorch >= 2.0）
        target_device: 目标设备 ('cpu', 'cuda', 'cuda:0'等)
        
    Returns:
        包含logits数据的字典
    """
    try:
        # 获取文件大小
        file_size_mb = os.path.getsize(filepath) / (1024 ** 2)

        # PyTorch 2.0+ 支持 weights_only 参数，更快更安全
        # 但如果文件包含非tensor数据，可能会失败
        if use_weights_only:
            try:
                data = torch.load(filepath, map_location=target_device, weights_only=True)
            except:
                # 如果weights_only失败，回退到普通加载
                data = torch.load(filepath, map_location=target_device)
        else:
            data = torch.load(filepath, map_location=target_device)

        return data
    except Exception as e:
        print(f"  ❌ 加载文件 {os.path.basename(filepath)} 出错: {e}")
        return None


def load_file_pair_parallel(log_prob_file: str, training_file: str, use_parallel: bool = True) -> Tuple[Dict, Dict]:
    """
    并行加载log_prob和training文件对（使用多线程加速）。
    
    Args:
        log_prob_file: log_prob文件路径
        training_file: training文件路径
        use_parallel: 是否使用并行加载（默认True）
        
    Returns:
        (log_prob_data, training_data) 元组
    """
    if use_parallel:
        # 并行加载（2个线程）
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_log_prob = executor.submit(load_logits_data, log_prob_file)
            future_training = executor.submit(load_logits_data, training_file)

            log_prob_data = future_log_prob.result()
            training_data = future_training.result()
    else:
        # 串行加载
        log_prob_data = load_logits_data(log_prob_file)
        training_data = load_logits_data(training_file)

    return log_prob_data, training_data


def extract_logits_from_data(data: Dict) -> Dict[int, torch.Tensor]:
    """
    从保存的数据中提取router logits。
    
    根据router_replay_saver.py的格式:
    data = {
        "step": step,
        "tp_rank": tp_rank,
        "pp_rank": pp_rank,
        "dp_world_size": dp_world_size,
        "compute_log_prob": {layer_idx: logits_tensor, ...},
        "training": {layer_idx: logits_tensor, ...},
    }
    
    Args:
        data: 加载的数据字典
        
    Returns:
        {layer_idx: logits_tensor} 字典
    """
    # 尝试多个可能的键名
    print(f"All available keys in data: {list(data.keys())}")

    for phase_key in ["compute_log_prob", "training"]:
        if phase_key in data and len(data[phase_key]) > 0:
            return data[phase_key]

    print(f"警告: 未找到logits数据，可用的键: {data.keys()}")
    return {}


def calculate_topk_expert_diff_on_gpu(log_prob_logits: torch.Tensor,
                                      training_logits: torch.Tensor,
                                      k: int, gpu_id: int) -> torch.Tensor:
    """
    在指定GPU上计算topK专家索引的差异（完全向量化，无Python循环）。
    
    Args:
        log_prob_logits: log_prob阶段的logits（已在GPU上）
        training_logits: training阶段的logits（已在GPU上）
        k: topK的K值
        gpu_id: GPU ID
        
    Returns:
        差异数量（在CPU上）
    """
    device = f'cuda:{gpu_id}'
    if not log_prob_logits.is_cuda:
        log_prob_logits = log_prob_logits.to(device)
    if not training_logits.is_cuda:
        training_logits = training_logits.to(device)

    # 获取topK专家索引 [num_tokens, k]
    _, log_prob_topk_ids = torch.topk(log_prob_logits, k, dim=-1)
    _, training_topk_ids = torch.topk(training_logits, k, dim=-1)

    num_tokens = log_prob_logits.shape[0]

    # 完全向量化的方法：计算两个集合的对称差
    # 对称差 = (A - B) ∪ (B - A) = A ∪ B - A ∩ B
    # 对于每个token，我们需要计算两个topK集合的对称差大小

    # 方法：对每个token，统计两个集合中不重复的元素数量
    # 使用广播和比较来找到匹配的元素

    # 扩展维度以便广播比较: [num_tokens, k, 1] vs [num_tokens, 1, k]
    log_prob_expanded = log_prob_topk_ids.unsqueeze(-1)  # [num_tokens, k, 1]
    training_expanded = training_topk_ids.unsqueeze(1)  # [num_tokens, 1, k]

    # 找到所有匹配的元素: [num_tokens, k, k]
    matches = (log_prob_expanded == training_expanded)

    # 对每个token，统计log_prob中有多少元素在training中（交集）
    # matches.sum(dim=-1) 得到每个log_prob元素在training中出现的次数 [num_tokens, k]
    log_prob_in_training = matches.sum(dim=-1) > 0  # [num_tokens, k]
    log_prob_match_count = log_prob_in_training.sum(dim=-1)  # [num_tokens]

    # 对每个token，统计training中有多少元素在log_prob中（交集）
    training_in_log_prob = matches.sum(dim=-2) > 0  # [num_tokens, k]
    training_match_count = training_in_log_prob.sum(dim=-1)  # [num_tokens]

    # 并集大小 = k + k - 交集大小
    # 交集大小 = min(log_prob_match_count, training_match_count) 实际上应该是相同的
    # 但为了安全，我们使用更精确的计算
    intersection_size = log_prob_match_count  # 或 training_match_count，应该相同

    # 对称差大小 = 并集大小 - 交集大小 = (k + k) - 2 * 交集大小
    # 但更准确的是：对称差 = (k - log_prob_match_count) + (k - training_match_count)
    # 因为不在交集中的元素就是差异
    # 注意：除以2，因为对称差中每个差异的专家被计算了两次（在log_prob和training中各一次）
    diff_counts = ((k - log_prob_match_count) + (k - training_match_count)) / 2.0

    return diff_counts.cpu()


def calculate_topk_expert_diff(log_prob_logits: torch.Tensor,
                               training_logits: torch.Tensor,
                               k: int, use_gpu: bool = True) -> torch.Tensor:
    """
    计算log_prob和training的topK专家索引的差异数量（支持多GPU加速）。
    
    Args:
        log_prob_logits: log_prob阶段的logits [num_tokens, num_experts]
        training_logits: training阶段的logits [num_tokens, num_experts]
        k: topK的K值
        use_gpu: 是否使用GPU加速
        
    Returns:
        每个token的差异数量 [num_tokens]
    """
    if not use_gpu or NUM_GPUS == 0:
        # CPU计算
        _, log_prob_topk_ids = torch.topk(log_prob_logits, k, dim=-1)
        _, training_topk_ids = torch.topk(training_logits, k, dim=-1)

        num_tokens = log_prob_logits.shape[0]
        diff_counts = torch.zeros(num_tokens)

        for i in range(num_tokens):
            log_prob_set = set(log_prob_topk_ids[i].tolist())
            training_set = set(training_topk_ids[i].tolist())
            diff_counts[i] = len(log_prob_set.symmetric_difference(training_set))

        return diff_counts

    # 多GPU并行计算
    log_prob_chunks = split_data_for_gpus(log_prob_logits, NUM_GPUS)
    training_chunks = split_data_for_gpus(training_logits, NUM_GPUS)
    results = []

    # 使用torch.cuda.Stream并行处理
    streams = [torch.cuda.Stream(device=f'cuda:{i}') for i in range(len(log_prob_chunks))]

    for i, (lp_chunk, tr_chunk) in enumerate(zip(log_prob_chunks, training_chunks)):
        gpu_id = i % NUM_GPUS
        with torch.cuda.stream(streams[i]):
            result = calculate_topk_expert_diff_on_gpu(lp_chunk, tr_chunk, k, gpu_id)
            results.append(result)

    # 等待所有流完成
    for stream in streams:
        stream.synchronize()

    return torch.cat(results, dim=0)


def analyze_file_pair(log_prob_file: str, training_file: str, k: int,
                      output_dir: str, step: str, tp: str, pp: str,
                      use_parallel_load: bool = True, use_parallel_plot: bool = True):
    """
    分析一对log_prob和training文件。
    
    Args:
        log_prob_file: log_prob文件路径
        training_file: training文件路径
        k: topK的K值
        output_dir: 输出目录
        step: 步数
        tp: tensor parallel rank
        pp: pipeline parallel rank
        use_parallel_load: 是否使用并行加载（默认True）
        use_parallel_plot: 是否使用并行绘图（默认True）
    """
    print(f"\n{'=' * 80}")
    print(f"分析文件对: step={step}, tp={tp}, pp={pp}")
    print(f"{'=' * 80}")

    # 1. 加载数据（并行加载两个文件以加速）
    print("\n[1/4] 加载数据...")
    t_load_start = time.time()

    # 获取文件大小信息
    log_prob_size_mb = os.path.getsize(log_prob_file) / (1024 ** 2)
    training_size_mb = os.path.getsize(training_file) / (1024 ** 2)
    total_size_mb = log_prob_size_mb + training_size_mb

    print(f"  文件大小: log_prob {log_prob_size_mb:.1f} MB + training {training_size_mb:.1f} MB = {total_size_mb:.1f} MB")
    print(f"  加载模式: {'并行加载（2线程）' if use_parallel_load else '串行加载'}")

    # 加载文件（可选并行）
    log_prob_data, training_data = load_file_pair_parallel(log_prob_file, training_file, use_parallel_load)

    if log_prob_data is None or training_data is None:
        print("❌ 数据加载失败，跳过该文件对")
        return

    load_time = time.time() - t_load_start
    load_speed_mbps = total_size_mb / load_time if load_time > 0 else 0

    print(f"  ✓ log_prob文件: {os.path.basename(log_prob_file)}")
    print(f"  ✓ training文件: {os.path.basename(training_file)}")
    print(f"  ✓ 加载完成: 耗时 {load_time:.2f}s, 速度 {load_speed_mbps:.1f} MB/s")

    # 2. 提取logits
    print("\n[2/4] 提取logits...")
    log_prob_logits_dict = extract_logits_from_data(log_prob_data)
    print(f"  ✓ 提取log_prob logits: {len(log_prob_logits_dict)} 层，形状: {list(log_prob_logits_dict.values())[0].shape if log_prob_logits_dict else 'N/A'}")
    training_logits_dict = extract_logits_from_data(training_data)
    print(f"  ✓ 提取training logits: {len(training_logits_dict)} 层，形状: {list(training_logits_dict.values())[0].shape if training_logits_dict else 'N/A'}")

    if not log_prob_logits_dict or not training_logits_dict:
        print("❌ 未能提取logits，跳过该文件对")
        return

    # 找到共同的层
    log_prob_layers = set(log_prob_logits_dict.keys())
    training_layers = set(training_logits_dict.keys())
    common_layers = sorted(log_prob_layers & training_layers)

    if not common_layers:
        print("❌ 没有共同的层，跳过该文件对")
        return

    print(f"  ✓ 找到共同的层: {len(common_layers)} 层")
    print(f"  层索引: {common_layers}")

    # 统计总token数
    total_tokens = sum(log_prob_logits_dict[layer].shape[0] for layer in common_layers)
    first_layer_experts = log_prob_logits_dict[common_layers[0]].shape[1]
    print(f"  总数据量: {total_tokens:,} tokens, {first_layer_experts} experts")

    # 3. 对每一层进行分析（批量处理策略：每张卡处理多层）
    print(f"\n[3/4] 分析各层 (共{len(common_layers)}层)...")
    results = {}

    use_gpu = NUM_GPUS > 0
    analysis_start_time = time.time()

    if use_gpu:
        # 估算每张卡能处理的层数
        sample_shape = log_prob_logits_dict[common_layers[0]].shape
        num_experts = sample_shape[1]
        layers_per_gpu = estimate_layers_per_gpu(sample_shape, num_experts)

        print(f"  计算模式: 批量并行（{NUM_GPUS} GPU，每GPU处理{layers_per_gpu}层）")
        print(f"  策略: 多层合并计算，充分利用显存，减少kernel启动")

        # 将层分配到GPU（轮询分配）
        layer_groups = {}  # {gpu_id: [layer_indices]}
        for idx, layer_idx in enumerate(common_layers):
            gpu_id = idx % NUM_GPUS
            if gpu_id not in layer_groups:
                layer_groups[gpu_id] = []
            layer_groups[gpu_id].append(layer_idx)

        print(f"  GPU分配: {[(gpu_id, len(layers)) for gpu_id, layers in sorted(layer_groups.items())]}")
        print_gpu_memory_stats()

        # 预加载数据到GPU
        print(f"  预加载数据到GPU...")
        for layer_idx in common_layers:
            gpu_id = layer_idx % NUM_GPUS
            device = f'cuda:{gpu_id}'
            log_prob_logits_dict[layer_idx] = log_prob_logits_dict[layer_idx].to(device, non_blocking=True)
            training_logits_dict[layer_idx] = training_logits_dict[layer_idx].to(device, non_blocking=True)
        torch.cuda.synchronize()
        print(f"  ✓ 数据已预加载到GPU")
        print_gpu_memory_stats()
    else:
        print("  计算模式: CPU")
        layers_per_gpu = 1
        layer_groups = {0: common_layers}

    def process_layers_batch_on_gpu(layer_indices: List[int], gpu_id: int):
        """
        在指定GPU上批量处理多层（合并计算，然后split结果）。
        
        Args:
            layer_indices: 要处理的层索引列表
            gpu_id: GPU ID
            
        Returns:
            {layer_idx: result_dict} 字典
        """
        device = f'cuda:{gpu_id}'

        # 收集这些层的数据
        log_prob_layers = []
        training_layers = []
        layer_token_counts = []

        for layer_idx in layer_indices:
            log_prob_logits = log_prob_logits_dict[layer_idx]
            training_logits = training_logits_dict[layer_idx]

            # 确保在正确的GPU上
            if not log_prob_logits.is_cuda or log_prob_logits.device.index != gpu_id:
                log_prob_logits = log_prob_logits.to(device, non_blocking=True)
                training_logits = training_logits.to(device, non_blocking=True)

            num_tokens = log_prob_logits.shape[0]
            log_prob_layers.append(log_prob_logits)
            training_layers.append(training_logits)
            layer_token_counts.append(num_tokens)

        # 合并所有层的数据 [total_tokens, num_experts]
        log_prob_batch = torch.cat(log_prob_layers, dim=0)
        training_batch = torch.cat(training_layers, dim=0)

        # 批量计算
        batch_results = process_layers_batch(log_prob_batch, training_batch, k, gpu_id)

        # Split结果回各层
        batch_time = batch_results['batch_time']
        num_layers = len(layer_indices)
        layer_results = {}
        start_idx = 0

        for i, layer_idx in enumerate(layer_indices):
            num_tokens = layer_token_counts[i]
            end_idx = start_idx + num_tokens

            layer_results[layer_idx] = {
                'layer_idx': layer_idx,
                'num_tokens': num_tokens,
                'num_experts': log_prob_layers[i].shape[1],
                'log_prob_entropy': batch_results['log_prob_entropy'][start_idx:end_idx].numpy(),
                'training_entropy': batch_results['training_entropy'][start_idx:end_idx].numpy(),
                'log_prob_topk_entropy': batch_results['log_prob_topk_entropy'][start_idx:end_idx].numpy(),
                'training_topk_entropy': batch_results['training_topk_entropy'][start_idx:end_idx].numpy(),
                'topk_diff': batch_results['topk_diff'][start_idx:end_idx].numpy(),
                'log_prob_logits': log_prob_layers[i].cpu(),
                'training_logits': training_layers[i].cpu(),
                'entropy_time': batch_time * 0.3 / num_layers,
                'topk_entropy_time': batch_time * 0.3 / num_layers,
                'topk_diff_time': batch_time * 0.4 / num_layers,
            }

            start_idx = end_idx

        return layer_results

    def process_single_layer(layer_idx, gpu_id):
        """处理单层（在指定GPU上，优化版：合并计算，减少重复）"""
        log_prob_logits = log_prob_logits_dict[layer_idx]
        training_logits = training_logits_dict[layer_idx]

        # 验证形状一致性
        if log_prob_logits.shape != training_logits.shape:
            return None, f"形状不匹配: {log_prob_logits.shape} vs {training_logits.shape}"

        num_tokens, num_experts = log_prob_logits.shape
        device = f'cuda:{gpu_id}' if use_gpu else 'cpu'

        # 如果数据在CPU，转移到指定GPU
        if use_gpu and not log_prob_logits.is_cuda:
            log_prob_logits = log_prob_logits.to(device, non_blocking=True)
            training_logits = training_logits.to(device, non_blocking=True)

        # 使用torch.cuda.Stream加速（如果使用GPU）
        if use_gpu:
            stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(stream):
                # 合并计算：一次性计算所有需要的值
                t0 = time.time()

                # 1. 计算整体entropy（使用log_softmax更高效）
                log_prob_log_softmax = torch.log_softmax(log_prob_logits, dim=-1)
                training_log_softmax = torch.log_softmax(training_logits, dim=-1)
                log_prob_probs = torch.exp(log_prob_log_softmax)
                training_probs = torch.exp(training_log_softmax)

                log_prob_entropy = -(log_prob_probs * log_prob_log_softmax).sum(dim=-1)
                training_entropy = -(training_probs * training_log_softmax).sum(dim=-1)

                # 2. 计算topK（只计算一次，复用结果）
                _, log_prob_topk_ids = torch.topk(log_prob_logits, k, dim=-1)
                _, training_topk_ids = torch.topk(training_logits, k, dim=-1)
                log_prob_topk_logits = torch.gather(log_prob_logits, -1, log_prob_topk_ids)
                training_topk_logits = torch.gather(training_logits, -1, training_topk_ids)

                # 3. 计算topK entropy
                log_prob_topk_log_softmax = torch.log_softmax(log_prob_topk_logits, dim=-1)
                training_topk_log_softmax = torch.log_softmax(training_topk_logits, dim=-1)
                log_prob_topk_probs = torch.exp(log_prob_topk_log_softmax)
                training_topk_probs = torch.exp(training_topk_log_softmax)

                log_prob_topk_entropy = -(log_prob_topk_probs * log_prob_topk_log_softmax).sum(dim=-1)
                training_topk_entropy = -(training_topk_probs * training_topk_log_softmax).sum(dim=-1)

                # 4. 计算topK差异（完全向量化）
                log_prob_expanded = log_prob_topk_ids.unsqueeze(-1)  # [num_tokens, k, 1]
                training_expanded = training_topk_ids.unsqueeze(1)  # [num_tokens, 1, k]
                matches = (log_prob_expanded == training_expanded)
                log_prob_match_count = (matches.sum(dim=-1) > 0).sum(dim=-1)
                training_match_count = (matches.sum(dim=-2) > 0).sum(dim=-1)
                # 除以2，因为对称差中每个差异的专家被计算了两次
                topk_diff = ((k - log_prob_match_count) + (k - training_match_count)) / 2.0

                # 同步stream
                stream.synchronize()

                total_time = time.time() - t0
                entropy_time = total_time * 0.3  # 估算
                topk_entropy_time = total_time * 0.3
                topk_diff_time = total_time * 0.4

                # 转回CPU（批量转换，更高效）
                log_prob_entropy = log_prob_entropy.cpu()
                training_entropy = training_entropy.cpu()
                log_prob_topk_entropy = log_prob_topk_entropy.cpu()
                training_topk_entropy = training_topk_entropy.cpu()
                topk_diff = topk_diff.cpu()
                log_prob_logits_cpu = log_prob_logits.cpu()
                training_logits_cpu = training_logits.cpu()
        else:
            # CPU计算
            t0 = time.time()
            log_prob_entropy = entropy(log_prob_logits, dim=-1, use_gpu=False)
            training_entropy = entropy(training_logits, dim=-1, use_gpu=False)
            entropy_time = time.time() - t0

            t0 = time.time()
            log_prob_topk_entropy = topk_entropy(log_prob_logits, k, use_gpu=False)
            training_topk_entropy = topk_entropy(training_logits, k, use_gpu=False)
            topk_entropy_time = time.time() - t0

            t0 = time.time()
            topk_diff = calculate_topk_expert_diff(log_prob_logits, training_logits, k, use_gpu=False)
            topk_diff_time = time.time() - t0

            log_prob_logits_cpu = log_prob_logits
            training_logits_cpu = training_logits

        return {
            'layer_idx': layer_idx,
            'num_tokens': num_tokens,
            'num_experts': num_experts,
            'log_prob_entropy': log_prob_entropy.numpy(),
            'training_entropy': training_entropy.numpy(),
            'log_prob_topk_entropy': log_prob_topk_entropy.numpy(),
            'training_topk_entropy': training_topk_entropy.numpy(),
            'topk_diff': topk_diff.numpy(),
            'log_prob_logits': log_prob_logits_cpu,
            'training_logits': training_logits_cpu,
            'entropy_time': entropy_time,
            'topk_entropy_time': topk_entropy_time,
            'topk_diff_time': topk_diff_time,
        }, None

    # 使用批量处理策略
    if use_gpu and len(common_layers) > 1:
        # 将层分组（每GPU处理layers_per_gpu层）
        batches = []  # [(gpu_id, [layer_indices])]
        for gpu_id, layer_list in sorted(layer_groups.items()):
            # 将层列表按layers_per_gpu分组
            for i in range(0, len(layer_list), layers_per_gpu):
                batch_layers = layer_list[i:i + layers_per_gpu]
                batches.append((gpu_id, batch_layers))

        print(f"\n  批量处理配置:")
        print(f"    总批次数: {len(batches)}")
        print(f"    每批平均层数: {sum(len(layers) for _, layers in batches) / len(batches):.1f}")

        # 打印首层详细信息
        num_tokens, num_experts = log_prob_logits_dict[common_layers[0]].shape
        data_size_mb = (num_tokens * num_experts * 4) / (1024 ** 2)  # float32
        print(f"\n  首层信息 (Layer {common_layers[0]}):")
        print(f"    Tokens: {num_tokens:,}")
        print(f"    Experts: {num_experts}")
        print(f"    单层数据量: {data_size_mb:.2f} MB")
        print(f"    策略: 每GPU批量处理{layers_per_gpu}层，合并计算后split结果")

        # 使用ThreadPoolExecutor并行处理批次
        with ThreadPoolExecutor(max_workers=NUM_GPUS) as executor:
            futures = {}
            for gpu_id, batch_layers in batches:
                future = executor.submit(process_layers_batch_on_gpu, batch_layers, gpu_id)
                futures[future] = (gpu_id, batch_layers)

            # 使用tqdm显示进度
            for future in tqdm(futures, desc="  进度", ncols=100,
                               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
                gpu_id, batch_layers = futures[future]
                try:
                    batch_results = future.result()
                    for layer_idx, result in batch_results.items():
                        results[layer_idx] = result
                        # 使用批量处理中计算的时间
                        PERF_STATS['entropy_time'].append(result['entropy_time'])
                        PERF_STATS['topk_entropy_time'].append(result['topk_entropy_time'])
                        PERF_STATS['topk_diff_time'].append(result['topk_diff_time'])
                except Exception as e:
                    print(f"\n  ❌ GPU {gpu_id} 批次 {batch_layers} 处理出错: {e}")
                    import traceback
                    traceback.print_exc()
    else:
        # 串行处理（CPU或单GPU，使用单层处理）
        for layer_idx in tqdm(common_layers, desc="  进度", ncols=100,
                              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
            gpu_id = (layer_idx % NUM_GPUS) if use_gpu else 0
            result, error = process_single_layer(layer_idx, gpu_id)
            if error:
                print(f"\n  ⚠ 层 {layer_idx}: {error}")
                continue
            if result:
                results[result['layer_idx']] = result
                PERF_STATS['entropy_time'].append(result['entropy_time'])
                PERF_STATS['topk_entropy_time'].append(result['topk_entropy_time'])
                PERF_STATS['topk_diff_time'].append(result['topk_diff_time'])

    analysis_time = time.time() - analysis_start_time
    print(f"\n  ✓ 所有层分析完成，耗时: {analysis_time:.2f}s")
    print(f"  平均每层: {analysis_time / len(common_layers):.3f}s")

    # 打印分析阶段的GPU内存使用
    if use_gpu:
        print_gpu_memory_stats()

    # 4. 绘制相关性图和MoE可视化图（并行执行）
    print(f"\n[4/4] 生成可视化图表...")
    viz_start_time = time.time()

    # 相关性分析和MoE可视化（根据use_parallel_plot决定是否并行）
    if use_parallel_plot:
        # 并行执行两个主要绘图任务
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_corr = executor.submit(plot_correlation_analysis, results, k, output_dir, step, tp, pp, use_parallel_plot)
            future_moe = executor.submit(plot_moe_visualizations, results, k, output_dir, step, tp, pp, use_parallel_plot)

            # 等待两个主要绘图任务完成
            try:
                future_corr.result()
            except Exception as e:
                print(f"  ⚠ 相关性分析图绘制出错: {e}")
                import traceback
                traceback.print_exc()

            try:
                future_moe.result()
            except Exception as e:
                print(f"  ⚠ MoE可视化图绘制出错: {e}")
                import traceback
                traceback.print_exc()
    else:
        # 串行执行
        try:
            plot_correlation_analysis(results, k, output_dir, step, tp, pp, use_parallel_plot)
        except Exception as e:
            print(f"  ⚠ 相关性分析图绘制出错: {e}")
            import traceback
            traceback.print_exc()

        try:
            plot_moe_visualizations(results, k, output_dir, step, tp, pp, use_parallel_plot)
        except Exception as e:
            print(f"  ⚠ MoE可视化图绘制出错: {e}")
            import traceback
            traceback.print_exc()

    viz_time = time.time() - viz_start_time
    print(f"  ✓ 所有可视化完成，总耗时: {viz_time:.2f}s")

    total_time = time.time() - t_load_start
    print(f"\n{'=' * 80}")
    print(f"✓ 文件对分析完成: step={step}, tp={tp}, pp={pp}")
    print(f"  总耗时: {total_time:.2f}s")
    print(f"  数据加载: {time.time() - t_load_start - analysis_time - viz_time:.2f}s")
    print(f"  层分析: {analysis_time:.2f}s")
    print(f"  可视化: {viz_time:.2f}s")
    print(f"{'=' * 80}")


def plot_correlation_analysis(results: Dict, k: int, output_dir: str,
                              step: str, tp: str, pp: str, use_parallel: bool = True):
    """
    绘制topK差异与entropy的相关性分析图。
    
    Args:
        results: 分析结果字典
        k: topK的K值
        output_dir: 输出目录
        step: 步数
        tp: tensor parallel rank
        pp: pipeline parallel rank
        use_parallel: 是否使用并行绘图（默认True）
    """
    base_corr_dir = os.path.join(output_dir, f"step{step}_tp{tp}_pp{pp}", "correlation")
    corr_entropy_dir = os.path.join(base_corr_dir, "entropy")
    corr_entropy_exp_dir = os.path.join(base_corr_dir, "entropy_exp")
    os.makedirs(corr_entropy_dir, exist_ok=True)
    os.makedirs(corr_entropy_exp_dir, exist_ok=True)

    # 合并所有层的数据进行全局相关性分析
    all_topk_diff = []
    all_log_prob_entropy = []
    all_training_entropy = []
    all_log_prob_topk_entropy = []
    all_training_topk_entropy = []

    for layer_idx in sorted(results.keys()):
        r = results[layer_idx]
        all_topk_diff.append(r['topk_diff'])
        all_log_prob_entropy.append(r['log_prob_entropy'])
        all_training_entropy.append(r['training_entropy'])
        all_log_prob_topk_entropy.append(r['log_prob_topk_entropy'])
        all_training_topk_entropy.append(r['training_topk_entropy'])

    all_topk_diff = np.concatenate(all_topk_diff)
    all_log_prob_entropy = np.concatenate(all_log_prob_entropy)
    all_training_entropy = np.concatenate(all_training_entropy)
    all_log_prob_topk_entropy = np.concatenate(all_log_prob_topk_entropy)
    all_training_topk_entropy = np.concatenate(all_training_topk_entropy)

    # 打印数据统计信息
    print(f"  数据统计:")
    print(f"    TopK Diff: min={all_topk_diff.min():.2f}, max={all_topk_diff.max():.2f}, mean={all_topk_diff.mean():.2f}")
    print(f"    Log_Prob Entropy: min={all_log_prob_entropy.min():.4f}, max={all_log_prob_entropy.max():.4f}, mean={all_log_prob_entropy.mean():.4f}")
    print(f"    Training Entropy: min={all_training_entropy.min():.4f}, max={all_training_entropy.max():.4f}, mean={all_training_entropy.mean():.4f}")

    # 计算exp(entropy)，防止溢出
    # 限制entropy的最大值，避免exp溢出（exp(700)会溢出float32）
    max_exp_arg = 50  # 安全范围
    all_log_prob_exp_entropy = np.exp(np.clip(all_log_prob_entropy, None, max_exp_arg))
    all_training_exp_entropy = np.exp(np.clip(all_training_entropy, None, max_exp_arg))
    all_log_prob_exp_topk_entropy = np.exp(np.clip(all_log_prob_topk_entropy, None, max_exp_arg))
    all_training_exp_topk_entropy = np.exp(np.clip(all_training_topk_entropy, None, max_exp_arg))

    # 检查是否有溢出
    clipped_count = np.sum(all_log_prob_entropy > max_exp_arg) + np.sum(all_training_entropy > max_exp_arg) + \
                    np.sum(all_log_prob_topk_entropy > max_exp_arg) + np.sum(all_training_topk_entropy > max_exp_arg)
    if clipped_count > 0:
        print(f"  ⚠ 警告: {clipped_count} 个entropy值被裁剪以避免exp溢出")

    # 并行绘制所有相关性散点图（分别保存到entropy和entropy_exp子目录）
    scatter_plots = [
        # 原始entropy的散点图 -> correlation/entropy
        ("Log_Prob Entropy", all_log_prob_entropy, all_topk_diff,
         "Log_Prob Entropy (All Experts)", f"TopK({k}) Expert Difference Count",
         f"TopK Difference vs Log_Prob Entropy (Step {step})",
         os.path.join(corr_entropy_dir, "topk_diff_vs_logprob_entropy.png")),
        ("Training Entropy", all_training_entropy, all_topk_diff,
         "Training Entropy (All Experts)", f"TopK({k}) Expert Difference Count",
         f"TopK Difference vs Training Entropy (Step {step})",
         os.path.join(corr_entropy_dir, "topk_diff_vs_training_entropy.png")),
        ("Log_Prob TopK Entropy", all_log_prob_topk_entropy, all_topk_diff,
         f"Log_Prob TopK({k}) Entropy", f"TopK({k}) Expert Difference Count",
         f"TopK Difference vs Log_Prob TopK Entropy (Step {step})",
         os.path.join(corr_entropy_dir, "topk_diff_vs_logprob_topk_entropy.png")),
        ("Training TopK Entropy", all_training_topk_entropy, all_topk_diff,
         f"Training TopK({k}) Entropy", f"TopK({k}) Expert Difference Count",
         f"TopK Difference vs Training TopK Entropy (Step {step})",
         os.path.join(corr_entropy_dir, "topk_diff_vs_training_topk_entropy.png")),
        # exp(entropy)的散点图 -> correlation/entropy_exp
        ("Log_Prob exp(Entropy)", all_log_prob_exp_entropy, all_topk_diff,
         "Log_Prob exp(Entropy) (All Experts)", f"TopK({k}) Expert Difference Count",
         f"TopK Difference vs Log_Prob exp(Entropy) (Step {step})",
         os.path.join(corr_entropy_exp_dir, "topk_diff_vs_logprob_exp_entropy.png")),
        ("Training exp(Entropy)", all_training_exp_entropy, all_topk_diff,
         "Training exp(Entropy) (All Experts)", f"TopK({k}) Expert Difference Count",
         f"TopK Difference vs Training exp(Entropy) (Step {step})",
         os.path.join(corr_entropy_exp_dir, "topk_diff_vs_training_exp_entropy.png")),
        ("Log_Prob exp(TopK Entropy)", all_log_prob_exp_topk_entropy, all_topk_diff,
         f"Log_Prob exp(TopK({k}) Entropy)", f"TopK({k}) Expert Difference Count",
         f"TopK Difference vs Log_Prob exp(TopK Entropy) (Step {step})",
         os.path.join(corr_entropy_exp_dir, "topk_diff_vs_logprob_exp_topk_entropy.png")),
        ("Training exp(TopK Entropy)", all_training_exp_topk_entropy, all_topk_diff,
         f"Training exp(TopK({k}) Entropy)", f"TopK({k}) Expert Difference Count",
         f"TopK Difference vs Training exp(TopK Entropy) (Step {step})",
         os.path.join(corr_entropy_exp_dir, "topk_diff_vs_training_exp_topk_entropy.png")),
    ]

    # 绘制散点图（并行或串行）
    if use_parallel:
        print(f"  → 并行绘制 {len(scatter_plots)} 个散点图...")
        with ThreadPoolExecutor(max_workers=min(8, len(scatter_plots))) as executor:
            futures = {}
            for task_name, x, y, xlabel, ylabel, title, save_path in scatter_plots:
                future = executor.submit(plot_scatter_with_correlation, x, y, xlabel, ylabel, title, save_path)
                futures[future] = task_name

            # 等待所有散点图完成，使用as_completed确保所有任务都完成
            from concurrent.futures import as_completed
            completed_count = 0
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="  绘制散点图", ncols=100,
                               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
                task_name = futures[future]
                try:
                    future.result()  # 获取结果，如果有异常会在这里抛出
                    completed_count += 1
                except Exception as e:
                    print(f"\n  ⚠ {task_name} 散点图绘制出错: {e}")
                    import traceback
                    traceback.print_exc()

            print(f"  ✓ 完成 {completed_count}/{len(scatter_plots)} 个散点图")
    else:
        print(f"  → 串行绘制 {len(scatter_plots)} 个散点图...")
        for task_name, x, y, xlabel, ylabel, title, save_path in tqdm(scatter_plots,
                                                                      desc="  绘制散点图", ncols=100,
                                                                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
            try:
                plot_scatter_with_correlation(x, y, xlabel, ylabel, title, save_path)
            except Exception as e:
                print(f"\n  ⚠ {task_name} 散点图绘制出错: {e}")
                import traceback
                traceback.print_exc()

    # 绘制每层的相关性系数（单独执行，因为需要所有散点图的数据）
    plot_layerwise_correlation(results, k, corr_entropy_dir, corr_entropy_exp_dir, step)

    print(f"相关性分析图已保存到:")
    print(f"  - Entropy: {corr_entropy_dir}")
    print(f"  - Entropy Exp: {corr_entropy_exp_dir}")


def downsample_data(data: np.ndarray, max_samples: int = 100000, random_seed: int = 42) -> np.ndarray:
    """
    对数据进行下采样以加速绘图。
    
    Args:
        data: 输入数据数组
        max_samples: 最大采样数量（默认1M）
        random_seed: 随机种子（默认42）
        
    Returns:
        下采样后的数据
    """
    if len(data) <= max_samples:
        return data

    np.random.seed(random_seed)
    indices = np.random.choice(len(data), max_samples, replace=False)
    return data[indices]


def safe_tight_layout():
    """安全地调用tight_layout，抑制警告。"""
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, message='.*Tight layout.*')
        try:
            plt.tight_layout()
        except:
            pass


def plot_scatter_with_correlation(x: np.ndarray, y: np.ndarray,
                                  xlabel: str, ylabel: str,
                                  title: str, save_path: str):
    """
    绘制散点图并计算相关性（线程安全版本）。
    
    Args:
        x: x轴数据
        y: y轴数据
        xlabel: x轴标签
        ylabel: y轴标签
        title: 图标题
        save_path: 保存路径
    """
    # 检查数据有效性
    if len(x) == 0 or len(y) == 0 or len(x) != len(y):
        print(f"  ⚠ 数据无效，跳过绘图: {title} (x_len={len(x)}, y_len={len(y)})")
        return

    # 移除NaN和Inf
    valid_mask = np.isfinite(x) & np.isfinite(y)
    invalid_count = len(x) - valid_mask.sum()
    if invalid_count > 0:
        print(f"  ⚠ {title}: 移除了 {invalid_count}/{len(x)} 个无效数据点")

    if valid_mask.sum() == 0:
        print(f"  ⚠ 没有有效数据，跳过绘图: {title}")
        return

    x_clean = x[valid_mask]
    y_clean = y[valid_mask]

    # 检查数据范围
    x_min, x_max = x_clean.min(), x_clean.max()
    y_min, y_max = y_clean.min(), y_clean.max()

    # 检查数据是否全为常数
    if x_max - x_min < 1e-10:
        print(f"  ⚠ {title}: x数据为常数 ({x_min:.6f})，跳过绘图")
        return
    if y_max - y_min < 1e-10:
        print(f"  ⚠ {title}: y数据为常数 ({y_min:.6f})，跳过绘图")
        return

    # 下采样以加速绘图和相关性计算
    max_samples_plot = 100000  # tokens for plotting
    max_samples_corr = 500000  # tokens for correlation calculation (larger for accuracy)

    # 下采样用于绘图
    if len(x_clean) > max_samples_plot:
        indices_plot = np.random.choice(len(x_clean), max_samples_plot, replace=False)
        x_sample = x_clean[indices_plot]
        y_sample = y_clean[indices_plot]
    else:
        x_sample = x_clean
        y_sample = y_clean

    # 下采样用于相关性计算（使用更多数据以保持准确性）
    if len(x_clean) > max_samples_corr:
        indices_corr = np.random.choice(len(x_clean), max_samples_corr, replace=False)
        x_corr = x_clean[indices_corr]
        y_corr = y_clean[indices_corr]
        print(f"  → 相关性计算下采样到 {max_samples_corr:,} tokens (原始: {len(x_clean):,})")
    else:
        x_corr = x_clean
        y_corr = y_clean

    # 计算相关性（使用下采样数据以加速，特别是Spearman需要排序）
    try:
        # Pearson相关性（快速）
        pearson_corr, pearson_p = pearsonr(x_corr, y_corr)
        # Spearman相关性（需要排序，可能较慢）
        spearman_corr, spearman_p = spearmanr(x_corr, y_corr)
    except Exception as e:
        print(f"  ⚠ 计算相关性失败: {title}, 错误: {e}")
        return

    # 绘图（使用独立的figure，确保线程安全）
    fig, ax = plt.subplots(figsize=(10, 8))

    # 检查数据点数量
    if len(x_sample) == 0:
        print(f"  ⚠ {title}: 采样后数据为空，跳过绘图")
        plt.close(fig)
        return

    # 绘制散点图，使用更大的点如果数据点较少
    point_size = max(1, min(10, 100000 / max(len(x_sample), 1)))
    ax.scatter(x_sample, y_sample, alpha=0.3, s=point_size)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # 设置合理的坐标轴范围
    x_range = x_max - x_min
    y_range = y_max - y_min
    if x_range > 0:
        ax.set_xlim(x_min - 0.05 * x_range, x_max + 0.05 * x_range)
    if y_range > 0:
        ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

    # 添加相关性信息
    text = f"Pearson r={pearson_corr:.4f} (p={pearson_p:.2e})\n"
    text += f"Spearman ρ={spearman_corr:.4f} (p={spearman_p:.2e})"
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    safe_tight_layout()
    plt.savefig(save_path, dpi=320, bbox_inches='tight')
    plt.close(fig)  # 明确关闭figure


def compute_layer_correlations(layer_idx: int, layer_data: Dict) -> tuple:
    """
    计算单层的所有相关性系数（用于并行计算）。
    
    Args:
        layer_idx: 层索引
        layer_data: 层数据字典
        
    Returns:
        (layer_idx, correlations_dict) 元组
    """
    r = layer_data
    topk_diff = r['topk_diff']

    correlations = {}

    # Log_prob entropy
    corr, _ = pearsonr(r['log_prob_entropy'], topk_diff)
    correlations['logprob_entropy'] = corr

    # Training entropy
    corr, _ = pearsonr(r['training_entropy'], topk_diff)
    correlations['training_entropy'] = corr

    # Log_prob topK entropy
    corr, _ = pearsonr(r['log_prob_topk_entropy'], topk_diff)
    correlations['logprob_topk_entropy'] = corr

    # Training topK entropy
    corr, _ = pearsonr(r['training_topk_entropy'], topk_diff)
    correlations['training_topk_entropy'] = corr

    # Log_prob exp(entropy)
    exp_entropy = np.exp(r['log_prob_entropy'])
    corr, _ = pearsonr(exp_entropy, topk_diff)
    correlations['logprob_exp_entropy'] = corr

    # Training exp(entropy)
    exp_entropy = np.exp(r['training_entropy'])
    corr, _ = pearsonr(exp_entropy, topk_diff)
    correlations['training_exp_entropy'] = corr

    # Log_prob exp(topK entropy)
    exp_topk_entropy = np.exp(r['log_prob_topk_entropy'])
    corr, _ = pearsonr(exp_topk_entropy, topk_diff)
    correlations['logprob_exp_topk_entropy'] = corr

    # Training exp(topK entropy)
    exp_topk_entropy = np.exp(r['training_topk_entropy'])
    corr, _ = pearsonr(exp_topk_entropy, topk_diff)
    correlations['training_exp_topk_entropy'] = corr

    return (layer_idx, correlations)


def plot_layerwise_correlation(results: Dict, k: int, corr_entropy_dir: str, corr_entropy_exp_dir: str, step: str):
    """
    绘制每层的相关性系数（并行计算版本）。
    
    Args:
        results: 分析结果字典
        k: topK的K值
        corr_entropy_dir: 原始entropy相关性图输出目录
        corr_entropy_exp_dir: exp(entropy)相关性图输出目录
        step: 步数
    """
    layers = sorted(results.keys())

    # 并行计算每层的相关性系数
    print(f"  → 并行计算 {len(layers)} 层的相关性系数...")
    corr_start_time = time.time()

    with ThreadPoolExecutor(max_workers=min(16, len(layers))) as executor:
        futures = {}
        for layer_idx in layers:
            future = executor.submit(compute_layer_correlations, layer_idx, results[layer_idx])
            futures[future] = layer_idx

        # 收集结果（使用tqdm显示进度）
        layer_correlations = {}
        from concurrent.futures import as_completed
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="  计算相关性", ncols=100,
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
            try:
                layer_idx, correlations = future.result()
                layer_correlations[layer_idx] = correlations
            except Exception as e:
                layer_idx = futures[future]
                print(f"\n  ⚠ 层 {layer_idx} 相关性计算失败: {e}")
                import traceback
                traceback.print_exc()

    corr_time = time.time() - corr_start_time
    print(f"  ✓ 相关性计算完成，耗时: {corr_time:.2f}s")

    # 按层顺序整理结果
    pearson_corrs = {
        'logprob_entropy': [],
        'training_entropy': [],
        'logprob_topk_entropy': [],
        'training_topk_entropy': [],
        'logprob_exp_entropy': [],
        'training_exp_entropy': [],
        'logprob_exp_topk_entropy': [],
        'training_exp_topk_entropy': [],
    }

    for layer_idx in layers:
        if layer_idx in layer_correlations:
            corr = layer_correlations[layer_idx]
            pearson_corrs['logprob_entropy'].append(corr['logprob_entropy'])
            pearson_corrs['training_entropy'].append(corr['training_entropy'])
            pearson_corrs['logprob_topk_entropy'].append(corr['logprob_topk_entropy'])
            pearson_corrs['training_topk_entropy'].append(corr['training_topk_entropy'])
            pearson_corrs['logprob_exp_entropy'].append(corr['logprob_exp_entropy'])
            pearson_corrs['training_exp_entropy'].append(corr['training_exp_entropy'])
            pearson_corrs['logprob_exp_topk_entropy'].append(corr['logprob_exp_topk_entropy'])
            pearson_corrs['training_exp_topk_entropy'].append(corr['training_exp_topk_entropy'])

    # 绘制原始entropy的线图
    plt.figure(figsize=(12, 6))
    plt.plot(layers, pearson_corrs['logprob_entropy'], 'o-', label='Log_Prob Entropy (All Experts)')
    plt.plot(layers, pearson_corrs['training_entropy'], 's-', label='Training Entropy (All Experts)')
    plt.plot(layers, pearson_corrs['logprob_topk_entropy'], '^-', label=f'Log_Prob TopK({k}) Entropy')
    plt.plot(layers, pearson_corrs['training_topk_entropy'], 'v-', label=f'Training TopK({k}) Entropy')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel('Layer ID')
    plt.ylabel('Pearson Correlation Coefficient')
    plt.title(f'Layer-wise TopK Difference vs Entropy Correlation (Step {step})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    try:
        plt.tight_layout()
    except:
        pass  # 如果tight_layout失败，继续执行
    plt.savefig(os.path.join(corr_entropy_dir, 'layerwise_correlation.png'), dpi=320, bbox_inches='tight')
    plt.close()

    # 绘制exp(entropy)的线图
    plt.figure(figsize=(12, 6))
    plt.plot(layers, pearson_corrs['logprob_exp_entropy'], 'o', label='Log_Prob exp(Entropy) (All Experts)', linestyle='--', markersize=6)
    plt.plot(layers, pearson_corrs['training_exp_entropy'], 's', label='Training exp(Entropy) (All Experts)', linestyle='--', markersize=6)
    plt.plot(layers, pearson_corrs['logprob_exp_topk_entropy'], '^', label=f'Log_Prob exp(TopK({k}) Entropy)', linestyle='--', markersize=6)
    plt.plot(layers, pearson_corrs['training_exp_topk_entropy'], 'v', label=f'Training exp(TopK({k}) Entropy)', linestyle='--', markersize=6)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel('Layer ID')
    plt.ylabel('Pearson Correlation Coefficient')
    plt.title(f'Layer-wise TopK Difference vs exp(Entropy) Correlation (Step {step})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    safe_tight_layout()
    plt.savefig(os.path.join(corr_entropy_exp_dir, 'layerwise_correlation.png'), dpi=320, bbox_inches='tight')
    plt.close()


def plot_moe_visualizations(results: Dict, k: int, output_dir: str,
                            step: str, tp: str, pp: str, use_parallel: bool = True):
    """
    绘制MoE相关的可视化图（重新组织目录结构）。
    
    参考run_routing.py中的绘图代码。
    
    Args:
        results: 分析结果字典
        k: topK的K值
        output_dir: 输出目录
        step: 步数
        tp: tensor parallel rank
        pp: pipeline parallel rank
        use_parallel: 是否使用并行绘图（默认True）
    """
    base_dir = os.path.join(output_dir, f"step{step}_tp{tp}_pp{pp}")

    # 创建新的目录结构
    entropy_dir = os.path.join(base_dir, "entropy")
    entropy_by_diff_dir = os.path.join(entropy_dir, "by_diff")
    topk_diff_dir = os.path.join(base_dir, "topk_diff")
    scores_load_dir = os.path.join(base_dir, "scores_load")

    os.makedirs(entropy_dir, exist_ok=True)
    os.makedirs(entropy_by_diff_dir, exist_ok=True)
    os.makedirs(topk_diff_dir, exist_ok=True)
    os.makedirs(scores_load_dir, exist_ok=True)

    sns.set_theme(style="darkgrid")

    # 并行绘制所有MoE可视化图
    plot_tasks = [
        # Entropy相关的图
        ("Entropy Distribution", plot_entropy_distribution, (results, k, entropy_dir, step), {}),
        ("Entropy by TopK Diff", plot_entropy_by_topk_diff, (results, k, entropy_by_diff_dir, step), {}),
        # TopK差异相关的图
        ("TopK Diff Distribution", plot_topk_diff_distribution, (results, k, topk_diff_dir, step), {}),
        ("TopK Diff Heatmap", plot_topk_diff_heatmap, (results, k, topk_diff_dir, step), {}),
        # Scores和Load相关的热力图
        ("Avg Scores (log_prob)", plot_average_scores_heatmap, (results, scores_load_dir, step, "log_prob"), {}),
        ("Avg Scores (training)", plot_average_scores_heatmap, (results, scores_load_dir, step, "training"), {}),
        ("Expert Load (log_prob)", plot_expert_load_heatmap, (results, k, scores_load_dir, step, "log_prob"), {}),
        ("Expert Load (training)", plot_expert_load_heatmap, (results, k, scores_load_dir, step, "training"), {}),
    ]

    plot_start_time = time.time()

    if use_parallel:
        print(f"  → 并行绘制所有MoE可视化图 ({len(plot_tasks)}个任务)...")
        from concurrent.futures import as_completed
        with ThreadPoolExecutor(max_workers=min(8, len(plot_tasks))) as executor:
            futures = {}
            for task_name, func, args, kwargs in plot_tasks:
                future = executor.submit(func, *args, **kwargs)
                futures[future] = task_name

            # 等待所有绘图完成，使用as_completed确保所有任务都完成
            completed = 0
            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    future.result()  # 获取结果，如果有异常会在这里抛出
                    completed += 1
                except Exception as e:
                    print(f"  ⚠ {task_name} 绘图出错: {e}")
                    import traceback
                    traceback.print_exc()
    else:
        print(f"  → 串行绘制所有MoE可视化图 ({len(plot_tasks)}个任务)...")
        completed = 0
        for task_name, func, args, kwargs in plot_tasks:
            try:
                func(*args, **kwargs)
                completed += 1
            except Exception as e:
                print(f"  ⚠ {task_name} 绘图出错: {e}")
                import traceback
                traceback.print_exc()

    plot_time = time.time() - plot_start_time
    print(f"  ✓ 所有绘图完成 ({completed}/{len(plot_tasks)}), 耗时: {plot_time:.2f}s")

    print(f"  ✓ MoE可视化图已保存:")
    print(f"    - Entropy: {entropy_dir}")
    print(f"    - TopK Difference: {topk_diff_dir}")
    print(f"    - Scores & Load: {scores_load_dir}")


def plot_entropy_distribution(results: Dict, k: int, viz_dir: str, step: str):
    """绘制entropy分布图。"""
    # 合并所有层的数据
    all_log_prob_entropy = []
    all_training_entropy = []
    all_log_prob_topk_entropy = []
    all_training_topk_entropy = []

    for layer_idx in sorted(results.keys()):
        r = results[layer_idx]
        all_log_prob_entropy.append(r['log_prob_entropy'])
        all_training_entropy.append(r['training_entropy'])
        all_log_prob_topk_entropy.append(r['log_prob_topk_entropy'])
        all_training_topk_entropy.append(r['training_topk_entropy'])

    if len(all_log_prob_entropy) == 0:
        print(f"  ⚠ plot_entropy_distribution: 没有数据")
        return

    all_log_prob_entropy = np.concatenate(all_log_prob_entropy)
    all_training_entropy = np.concatenate(all_training_entropy)
    all_log_prob_topk_entropy = np.concatenate(all_log_prob_topk_entropy)
    all_training_topk_entropy = np.concatenate(all_training_topk_entropy)

    # 检查数据有效性
    if len(all_log_prob_entropy) == 0:
        print(f"  ⚠ plot_entropy_distribution: 合并后数据为空")
        return

    # 下采样以加速绘图
    original_count = len(all_log_prob_entropy)
    max_samples = 100000  # 1M tokens
    if original_count > max_samples:
        indices = np.random.choice(original_count, max_samples, replace=False)
        all_log_prob_entropy = all_log_prob_entropy[indices]
        all_training_entropy = all_training_entropy[indices]
        all_log_prob_topk_entropy = all_log_prob_topk_entropy[indices]
        all_training_topk_entropy = all_training_topk_entropy[indices]
        print(f"  → 下采样到 {max_samples:,} tokens (原始: {original_count:,})")

    # 1. 整体entropy分布
    if len(all_log_prob_entropy) > 0 and len(all_training_entropy) > 0:
        data = pd.DataFrame({
            'Entropy': np.concatenate([all_log_prob_entropy, all_training_entropy]),
            'Phase': ['Log_Prob'] * len(all_log_prob_entropy) + ['Training'] * len(all_training_entropy)
        })

        # 移除NaN和Inf
        data = data[np.isfinite(data['Entropy'])]

        if len(data) > 0:
            plt.figure(figsize=(10, 6))
            try:
                sns.kdeplot(data=data, x='Entropy', hue='Phase', fill=True, common_norm=False, alpha=0.5)
                plt.title(f'Overall Entropy Distribution (Step {step})')
                plt.xlabel('Entropy')
                plt.ylabel('Density')
                try:
                    plt.tight_layout()
                except:
                    pass
                plt.savefig(os.path.join(viz_dir, 'entropy_distribution_all.png'), dpi=320, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"  ⚠ 绘制entropy分布图失败: {e}")
                plt.close()

    # 2. TopK entropy分布
    if len(all_log_prob_topk_entropy) > 0 and len(all_training_topk_entropy) > 0:
        data = pd.DataFrame({
            'Entropy': np.concatenate([all_log_prob_topk_entropy, all_training_topk_entropy]),
            'Phase': ['Log_Prob'] * len(all_log_prob_topk_entropy) + ['Training'] * len(all_training_topk_entropy)
        })

        # 移除NaN和Inf
        data = data[np.isfinite(data['Entropy'])]

        if len(data) > 0:
            plt.figure(figsize=(10, 6))
            try:
                sns.kdeplot(data=data, x='Entropy', hue='Phase', fill=True, common_norm=False, alpha=0.5)
                plt.title(f'TopK({k}) Entropy Distribution (Step {step})')
                plt.xlabel('Entropy')
                plt.ylabel('Density')
                try:
                    plt.tight_layout()
                except:
                    pass
                plt.savefig(os.path.join(viz_dir, f'entropy_distribution_topk{k}.png'), dpi=320, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"  ⚠ 绘制TopK entropy分布图失败: {e}")
                plt.close()


def plot_topk_diff_distribution(results: Dict, k: int, viz_dir: str, step: str):
    """绘制TopK差异分布图。"""
    all_topk_diff = []
    for layer_idx in sorted(results.keys()):
        all_topk_diff.append(results[layer_idx]['topk_diff'])
    all_topk_diff = np.concatenate(all_topk_diff)

    if len(all_topk_diff) == 0:
        print(f"  ⚠ plot_topk_diff_distribution: 没有数据")
        return

    # 下采样以加速统计（如果数据量太大）
    original_count = len(all_topk_diff)
    max_samples = 100000  # 1M tokens
    if original_count > max_samples:
        all_topk_diff = downsample_data(all_topk_diff, max_samples)
        print(f"  → 下采样到 {len(all_topk_diff):,} tokens (原始: {original_count:,})")

    plt.figure(figsize=(12, 6))
    unique, counts = np.unique(all_topk_diff, return_counts=True)
    total_tokens = len(all_topk_diff)  # 使用下采样后的数量
    percentages = (counts / total_tokens) * 100
    mean_diff = all_topk_diff.mean()

    # 绘制柱状图
    bars = plt.bar(unique, counts)

    # 在每个柱子上方添加百分比标注
    for i, (val, count, pct) in enumerate(zip(unique, counts, percentages)):
        plt.text(val, count, f'{pct:.1f}%',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.xlabel(f'TopK({k}) Expert Difference Count')
    plt.ylabel('Frequency')
    plt.title(f'TopK Expert Difference Distribution (Step {step}, Mean: {mean_diff:.2f})')
    plt.xticks(range(int(unique.max()) + 1))

    # 在图上添加总token数信息
    plt.text(0.02, 0.98, f'Total Tokens: {total_tokens:,}',
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    safe_tight_layout()
    plt.savefig(os.path.join(viz_dir, f'topk_diff_distribution.png'), dpi=320, bbox_inches='tight')
    plt.close()


def plot_entropy_by_topk_diff(results: Dict, k: int, viz_dir: str, step: str):
    """
    按topk_diff分组绘制entropy分布图。
    
    Args:
        results: 分析结果字典
        k: topK的K值
        viz_dir: 可视化图输出目录
        step: 步数
    """
    # 收集所有层的数据
    all_topk_diff = []
    all_log_prob_entropy = []
    all_training_entropy = []
    all_log_prob_topk_entropy = []
    all_training_topk_entropy = []

    for layer_idx in sorted(results.keys()):
        r = results[layer_idx]
        all_topk_diff.append(r['topk_diff'])
        all_log_prob_entropy.append(r['log_prob_entropy'])
        all_training_entropy.append(r['training_entropy'])
        all_log_prob_topk_entropy.append(r['log_prob_topk_entropy'])
        all_training_topk_entropy.append(r['training_topk_entropy'])

    all_topk_diff = np.concatenate(all_topk_diff)
    all_log_prob_entropy = np.concatenate(all_log_prob_entropy)
    all_training_entropy = np.concatenate(all_training_entropy)
    all_log_prob_topk_entropy = np.concatenate(all_log_prob_topk_entropy)
    all_training_topk_entropy = np.concatenate(all_training_topk_entropy)

    # 获取唯一的topk_diff值（四舍五入到整数）
    unique_diffs = np.unique(np.round(all_topk_diff).astype(int))
    unique_diffs = unique_diffs[unique_diffs <= k]  # 限制范围（topk_diff已经除以2，最大值是k）

    # 计算每个diff value的占比和平均差异数量（在下采样前计算，保持原始比例）
    total_tokens = len(all_topk_diff)
    mean_diff = all_topk_diff.mean()
    diff_percentages = {}
    for diff_val in unique_diffs:
        mask = np.round(all_topk_diff).astype(int) == diff_val
        count = mask.sum()
        diff_percentages[diff_val] = (count / total_tokens) * 100

    # 下采样以加速绘图（如果数据量太大）
    max_samples = 100000  # 1M tokens
    if len(all_topk_diff) > max_samples:
        # 按diff值分层下采样，保持每个diff组的比例
        np.random.seed(42)
        sampled_indices = []
        for diff_val in unique_diffs:
            mask = np.round(all_topk_diff).astype(int) == diff_val
            diff_indices = np.where(mask)[0]
            if len(diff_indices) > 0:
                # 计算该组应该采样的数量（保持比例）
                target_count = int(len(diff_indices) * max_samples / total_tokens)
                target_count = max(1, min(target_count, len(diff_indices)))  # 至少1个，不超过总数
                sampled = np.random.choice(diff_indices, target_count, replace=False)
                sampled_indices.append(sampled)

        if sampled_indices:
            sampled_indices = np.concatenate(sampled_indices)
            all_topk_diff = all_topk_diff[sampled_indices]
            all_log_prob_entropy = all_log_prob_entropy[sampled_indices]
            all_training_entropy = all_training_entropy[sampled_indices]
            all_log_prob_topk_entropy = all_log_prob_topk_entropy[sampled_indices]
            all_training_topk_entropy = all_training_topk_entropy[sampled_indices]
            print(f"  → 下采样到 {len(all_topk_diff):,} tokens (原始: {total_tokens:,})")

    # 1. 整体entropy按topk_diff分组
    data_list = []
    for diff_val in unique_diffs:
        mask = np.round(all_topk_diff).astype(int) == diff_val
        if mask.sum() > 0:
            data_list.append(pd.DataFrame({
                'Entropy': all_log_prob_entropy[mask],
                'Phase': 'Log_Prob',
                'TopK_Diff': diff_val
            }))
            data_list.append(pd.DataFrame({
                'Entropy': all_training_entropy[mask],
                'Phase': 'Training',
                'TopK_Diff': diff_val
            }))

    if data_list:
        data = pd.concat(data_list, ignore_index=True)

        plt.figure(figsize=(14, 8))
        # 使用violin plot或box plot展示分布
        sns.violinplot(data=data, x='TopK_Diff', y='Entropy', hue='Phase',
                       inner='box', palette='Set2')

        # 修改x轴标签，添加占比信息
        ax = plt.gca()
        labels = []
        for diff_val in unique_diffs:
            pct = diff_percentages.get(diff_val, 0)
            labels.append(f'{diff_val}\n({pct:.1f}%)')
        ax.set_xticklabels(labels)

        plt.xlabel(f'TopK({k}) Expert Difference Count (Percentage)')
        plt.ylabel('Entropy (All Experts)')
        plt.title(f'Entropy Distribution by TopK Difference (Step {step}, Mean Diff: {mean_diff:.2f})')
        plt.legend(title='Phase')

        # 添加总token数信息
        plt.text(0.02, 0.98, f'Total Tokens: {total_tokens:,}',
                 transform=ax.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        safe_tight_layout()
        plt.savefig(os.path.join(viz_dir, 'entropy_by_topk_diff_all.png'), dpi=320, bbox_inches='tight')
        plt.close()

    # 2. TopK entropy按topk_diff分组
    data_list = []
    for diff_val in unique_diffs:
        mask = np.round(all_topk_diff).astype(int) == diff_val
        if mask.sum() > 0:
            data_list.append(pd.DataFrame({
                'Entropy': all_log_prob_topk_entropy[mask],
                'Phase': 'Log_Prob',
                'TopK_Diff': diff_val
            }))
            data_list.append(pd.DataFrame({
                'Entropy': all_training_topk_entropy[mask],
                'Phase': 'Training',
                'TopK_Diff': diff_val
            }))

    if data_list:
        data = pd.concat(data_list, ignore_index=True)

        plt.figure(figsize=(14, 8))
        sns.violinplot(data=data, x='TopK_Diff', y='Entropy', hue='Phase',
                       inner='box', palette='Set2')

        # 修改x轴标签，添加占比信息
        ax = plt.gca()
        labels = []
        for diff_val in unique_diffs:
            pct = diff_percentages.get(diff_val, 0)
            labels.append(f'{diff_val}\n({pct:.1f}%)')
        ax.set_xticklabels(labels)

        plt.xlabel(f'TopK({k}) Expert Difference Count (Percentage)')
        plt.ylabel(f'Entropy (TopK({k}))')
        plt.title(f'TopK Entropy Distribution by TopK Difference (Step {step}, Mean Diff: {mean_diff:.2f})')
        plt.legend(title='Phase')

        # 添加总token数信息
        plt.text(0.02, 0.98, f'Total Tokens: {total_tokens:,}',
                 transform=ax.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        safe_tight_layout()
        plt.savefig(os.path.join(viz_dir, f'entropy_by_topk_diff_topk{k}.png'), dpi=320, bbox_inches='tight')
        plt.close()


def plot_average_scores_heatmap(results: Dict, viz_dir: str, step: str, phase: str):
    """绘制平均scores热力图。"""
    layers = sorted(results.keys())

    # 收集每层的平均scores
    avg_scores_list = []
    for layer_idx in layers:
        logits = results[layer_idx][f'{phase}_logits']
        # 转换为softmax scores
        scores = torch.softmax(logits, dim=-1)
        avg_scores = scores.mean(dim=0).numpy()
        avg_scores_list.append(avg_scores)

    # 创建DataFrame
    df = pd.DataFrame(avg_scores_list, index=layers)

    # 绘制热力图
    plt.figure(figsize=(max(10, len(df.columns) * 0.3), max(6, len(df) * 0.3)))
    sns.heatmap(df, cmap='YlOrRd', cbar=True, xticklabels=True, yticklabels=True)
    plt.title(f'Average Scores Heatmap - {phase} (Step {step})')
    plt.xlabel('Expert ID')
    plt.ylabel('Layer ID')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'avg_scores_heatmap_{phase}.png'), dpi=320, bbox_inches='tight')
    plt.close()


def plot_expert_load_heatmap(results: Dict, k: int, viz_dir: str, step: str, phase: str):
    """绘制Expert负载热力图。"""
    layers = sorted(results.keys())

    # 收集每层的expert负载
    load_list = []
    use_gpu = NUM_GPUS > 0

    for layer_idx in layers:
        logits = results[layer_idx][f'{phase}_logits']
        num_tokens, num_experts = logits.shape

        # 对于大数据使用GPU加速
        if use_gpu and num_tokens > 1000:
            device = 'cuda:0'
            logits_gpu = logits.to(device)
            _, topk_ids = torch.topk(logits_gpu, k, dim=-1)
            topk_ids = topk_ids.cpu()
        else:
            _, topk_ids = torch.topk(logits, k, dim=-1)

        # 统计每个专家被选中的次数
        tokens_per_expert = torch.bincount(topk_ids.flatten(), minlength=num_experts)
        load = (tokens_per_expert / (num_tokens * k)).numpy()
        load_list.append(load)

    # 清理GPU内存
    if use_gpu:
        torch.cuda.empty_cache()

    # 创建DataFrame
    df = pd.DataFrame(load_list, index=layers)

    # 绘制热力图
    plt.figure(figsize=(max(10, len(df.columns) * 0.3), max(6, len(df) * 0.3)))
    sns.heatmap(df, cmap='YlOrRd', cbar=True, xticklabels=True, yticklabels=True)
    plt.title(f'Expert Load Heatmap - {phase} (Step {step})')
    plt.xlabel('Expert ID')
    plt.ylabel('Layer ID')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'expert_load_heatmap_{phase}.png'), dpi=320, bbox_inches='tight')
    plt.close()


def plot_topk_diff_heatmap(results: Dict, k: int, viz_dir: str, step: str):
    """绘制TopK差异的统计热力图。"""
    layers = sorted(results.keys())

    # 收集每层的TopK差异统计，并计算平均差异数量
    diff_stats = []
    all_diffs = []
    for layer_idx in layers:
        topk_diff = results[layer_idx]['topk_diff']
        all_diffs.append(topk_diff)
        # 统计每个差异值的比例
        unique, counts = np.unique(topk_diff, return_counts=True)
        stats = np.zeros(k + 1)  # 最大差异为k（topk_diff已经除以2）
        for val, count in zip(unique, counts):
            val_int = int(val)
            if val_int <= k:  # 确保不超出范围
                stats[val_int] = count / len(topk_diff)
        diff_stats.append(stats)

    # 计算所有层的平均差异数量
    all_topk_diff = np.concatenate(all_diffs)
    mean_diff = all_topk_diff.mean()

    # 创建DataFrame
    df = pd.DataFrame(diff_stats, index=layers, columns=[str(i) for i in range(k + 1)])

    # 绘制热力图
    plt.figure(figsize=(max(10, len(df.columns) * 0.5), max(6, len(df) * 0.3)))
    sns.heatmap(df, cmap='YlOrRd', cbar=True, xticklabels=True, yticklabels=True, annot=False)
    plt.title(f'TopK({k}) Difference Distribution Heatmap (Step {step}, Mean Diff: {mean_diff:.2f})')
    plt.xlabel('Difference Count')
    plt.ylabel('Layer ID')
    safe_tight_layout()
    plt.savefig(os.path.join(viz_dir, 'topk_diff_heatmap.png'), dpi=320, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="分析保存的router logits (支持多GPU加速和并行加载)")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="包含保存的logits的目录")
    parser.add_argument("--topk", type=int, default=2,
                        help="TopK的K值 (默认: 2)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录 (默认: save_dir/analysis_output)")
    parser.add_argument("--no_gpu", action='store_true',
                        help="禁用GPU加速，强制使用CPU")
    parser.add_argument("--serial_load", action='store_true',
                        help="串行加载文件（禁用并行加载）")
    parser.add_argument("--serial_plot", action='store_true',
                        help="串行绘制图像（禁用并行绘图）")

    args = parser.parse_args()

    # 如果用户指定不使用GPU，则覆盖全局设置
    if args.no_gpu:
        global NUM_GPUS
        NUM_GPUS = 0
        print("已禁用GPU加速，将使用CPU计算")

    if not os.path.exists(args.save_dir):
        print(f"错误: 目录 {args.save_dir} 不存在")
        return

    if args.output_dir is None:
        args.output_dir = os.path.join(args.save_dir, "analysis_output")

    os.makedirs(args.output_dir, exist_ok=True)

    # 打印配置信息
    print("\n" + "=" * 80)
    print("配置信息")
    print("=" * 80)
    print(f"数据目录: {args.save_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"TopK值: {args.topk}")
    print(f"GPU加速: {'禁用 (强制CPU)' if args.no_gpu else f'启用 ({NUM_GPUS} GPUs)'}")
    print(f"数据加载: {'串行加载' if args.serial_load else '并行加载（2线程，推荐）'}")
    print("=" * 80)

    # 1. 查找文件对
    print("\n" + "=" * 80)
    print("步骤 1: 扫描数据文件")
    print("=" * 80)
    file_pairs = find_file_pairs(args.save_dir)

    if not file_pairs:
        print("❌ 未找到匹配的文件对")
        print("\n文件命名要求:")
        print("  - log_prob_{N}_tp{M}_pp{K}.pt")
        print("  - training_{N}_tp{M}_pp{K}.pt")
        return

    print(f"✓ 找到 {len(file_pairs)} 对匹配的文件:\n")

    # 统计总文件大小
    total_data_size = 0
    for idx, (step, tp, pp, log_prob_file, training_file) in enumerate(file_pairs, 1):
        log_prob_size = os.path.getsize(log_prob_file) / (1024 ** 2)
        training_size = os.path.getsize(training_file) / (1024 ** 2)
        pair_size = log_prob_size + training_size
        total_data_size += pair_size

        print(f"  [{idx}] Step {step:>3}, TP {tp}, PP {pp} (大小: {pair_size:.1f} MB)")
        print(f"      log_prob: {os.path.basename(log_prob_file)}")
        print(f"      training: {os.path.basename(training_file)}")

    print(f"\n总数据量: {total_data_size:.1f} MB ({total_data_size / 1024:.2f} GB)")
    if len(file_pairs) > 1:
        print(f"平均每对: {total_data_size / len(file_pairs):.1f} MB")

    print("\n" + "=" * 80)
    print(f"步骤 2: 分析文件对 (共 {len(file_pairs)} 对)")
    print("=" * 80)

    # 2. 分析每对文件
    total_start_time = time.time()
    success_count = 0

    for idx, (step, tp, pp, log_prob_file, training_file) in enumerate(file_pairs, 1):
        print(f"\n处理进度: [{idx}/{len(file_pairs)}]")
        try:
            analyze_file_pair(log_prob_file, training_file, args.topk,
                              args.output_dir, step, tp, pp,
                              use_parallel_load=not args.serial_load,
                              use_parallel_plot=not args.serial_plot)
            success_count += 1
        except Exception as e:
            print(f"\n❌ 分析 step={step}, tp={tp}, pp={pp} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    total_time = time.time() - total_start_time

    # 打印性能统计
    print_performance_stats()

    # 打印最终总结
    print("\n" + "=" * 80)
    print("分析完成总结")
    print("=" * 80)
    print(f"处理文件对: {success_count}/{len(file_pairs)} 成功")
    print(f"总耗时: {total_time:.2f}s ({total_time / 60:.1f} 分钟)")
    if success_count > 0:
        print(f"平均每对: {total_time / success_count:.2f}s")
    print(f"\n结果保存位置: {args.output_dir}")

    # 列出生成的目录
    output_dirs = [d for d in os.listdir(args.output_dir)
                   if os.path.isdir(os.path.join(args.output_dir, d))]
    if output_dirs:
        print(f"\n生成的分析目录 ({len(output_dirs)} 个):")
        for d in sorted(output_dirs)[:10]:  # 只显示前10个
            print(f"  - {d}/")
        if len(output_dirs) > 10:
            print(f"  ... 以及其他 {len(output_dirs) - 10} 个目录")

    print("=" * 80)

    if success_count == len(file_pairs):
        print("✓ 所有文件对分析成功!")
    elif success_count > 0:
        print(f"⚠ 部分文件对分析失败 ({len(file_pairs) - success_count} 个)")
    else:
        print("❌ 所有文件对分析失败")

    print("=" * 80)


if __name__ == "__main__":
    main()
