#!/usr/bin/env python3
"""
从 wandb 拉取同一 group 的曲线数据，对于重复的 step 取最高值，输出到 JSON
"""

import argparse
import json
import os
from collections import defaultdict
from typing import List, Optional

import wandb


def fetch_wandb_data(
    entity: str,
    project: str,
    groups: List[str],
    metric_keys: List[str],
    output_file: str = "wandb_data.json",
    api_key_file: Optional[str] = None
):
    """
    从 wandb 拉取指定 groups 的数据
    
    Args:
        entity: wandb entity 名称
        project: wandb project 名称
        groups: wandb group 名称列表
        metric_keys: 要拉取的 metric key 列表
        output_file: 输出的 JSON 文件路径
        api_key_file: API key 文件路径（可选）
    """
    # 从文件读取 API key（如果提供）
    api_key = None
    if api_key_file:
        if not os.path.exists(api_key_file):
            raise FileNotFoundError(f"API key 文件不存在: {api_key_file}")

        with open(api_key_file, 'r', encoding='utf-8') as f:
            api_key = f.read().strip()

        print(f"从文件读取 API key: {api_key_file}")

    # 初始化 wandb API
    api = wandb.Api(api_key=api_key) if api_key else wandb.Api()

    # 用于存储所有数据: {group: {metric_key: {step: (value, timestamp)}}}
    data_dict = {}

    total_runs = 0

    # 遍历所有 groups
    for group in groups:
        print(f"\n{'='*60}")
        print(f"处理 group: {group}")
        print(f"{'='*60}")

        # 初始化该 group 的数据结构
        if group not in data_dict:
            data_dict[group] = {key: {} for key in metric_keys}

        # 获取该 project 下指定 group 的所有 runs
        runs = api.runs(f"{entity}/{project}", filters={"group": group})

        print(f"找到 {len(runs)} 个 runs 在 group '{group}'")

        if len(runs) == 0:
            print(f"警告: group '{group}' 中没有找到任何 runs")
            continue

        total_runs += len(runs)

        # 遍历所有 runs
        for run in runs:
            print(f"  处理 run: {run.name} (id: {run.id})")

            # 获取 run 的历史数据
            history = run.history()

            # 对每个指定的 metric key 进行处理
            for key in metric_keys:
                if key not in history.columns:
                    print(f"    警告: key '{key}' 不存在于 run {run.name}")
                    continue

                # 确保 _timestamp 列存在
                if '_timestamp' not in history.columns:
                    print(f"    警告: 缺少 _timestamp 列，使用默认时间戳")
                    history['_timestamp'] = 0

                # 过滤出有效数据（非 NaN）
                valid_data = history[['_step', key, '_timestamp']].dropna(subset=['_step', key])

                # 对于每个 step，保留 timestamp 最旧的值
                for _, row in valid_data.iterrows():
                    step = int(row['_step'])
                    value = float(row[key])
                    timestamp = row['_timestamp']

                    # 如果该 step 不存在，或者当前 timestamp 更旧，则更新
                    if step not in data_dict[group][key]:
                        data_dict[group][key][step] = (value, timestamp)
                    else:
                        existing_timestamp = data_dict[group][key][step][1]
                        if timestamp < existing_timestamp:
                            data_dict[group][key][step] = (value, timestamp)

                print(f"    已处理 key '{key}': {len(valid_data)} 个数据点")

    # 将数据转换为可序列化的格式: {group: {key: {step: [...], value: [...]}}}
    output_data = {}
    total_steps_count = 0

    for group in groups:
        output_data[group] = {}
        
        for key in metric_keys:
            if not data_dict.get(group, {}).get(key):
                print(f"警告: group '{group}' 的 key '{key}' 没有有效数据")
                output_data[group][key] = {"step": [], "value": []}
                continue

            # 按 step 排序
            sorted_steps = sorted(data_dict[group][key].keys())
            steps = []
            values = []
            
            for step in sorted_steps:
                steps.append(step)
                values.append(data_dict[group][key][step][0])  # 只取 value，不取 timestamp
            
            output_data[group][key] = {
                "step": steps,
                "value": values
            }
            
            total_steps_count += len(steps)
            print(f"  Group '{group}', key '{key}': {len(steps)} unique steps")

    # 写入 JSON 文件（数组不换行）
    import re
    
    def format_json_compact_arrays(data, indent=2):
        """格式化 JSON，使数组元素不换行"""
        # 先生成格式化的 JSON
        json_str = json.dumps(data, indent=indent, ensure_ascii=False)
        
        # 使用正则表达式将数组压缩到一行
        # 匹配 [ 开始，后面跟着换行和缩进的元素，直到 ] 结束
        pattern = r'\[\s*\n\s*((?:[^\[\]{}]|\s)*?)\s*\n\s*\]'
        
        def compress_array(match):
            # 获取数组内容
            content = match.group(1)
            # 移除所有换行和多余空格，保留逗号分隔
            items = re.split(r',\s*\n\s*', content)
            items = [item.strip() for item in items if item.strip()]
            # 重新组合成一行
            return '[' + ', '.join(items) + ']'
        
        # 多次替换，直到没有匹配为止（处理嵌套数组）
        prev_str = None
        while prev_str != json_str:
            prev_str = json_str
            json_str = re.sub(pattern, compress_array, json_str)
        
        return json_str
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(format_json_compact_arrays(output_data))

    print(f"\n数据已保存到: {output_file}")

    # 打印摘要信息
    print("\n" + "="*60)
    print("摘要:")
    print(f"  Groups: {', '.join(groups)}")
    print(f"  总 groups 数: {len(groups)}")
    print(f"  总 runs 数: {total_runs}")
    print(f"  总数据点数: {total_steps_count}")
    print(f"  拉取的 keys: {', '.join(metric_keys)}")
    print(f"\n  各 group 详情:")
    for group in groups:
        print(f"    {group}:")
        for key in metric_keys:
            num_points = len(output_data.get(group, {}).get(key, {}).get("step", []))
            if num_points > 0:
                print(f"      - {key}: {num_points} steps")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="从 wandb 拉取指定 groups 的曲线数据，对重复 step 取最高值"
    )
    parser.add_argument(
        "--entity",
        type=str,
        required=True,
        help="wandb entity 名称"
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="wandb project 名称"
    )
    parser.add_argument(
        "--groups",
        type=str,
        nargs="+",
        required=True,
        help="wandb group 名称 (可以指定多个)"
    )
    parser.add_argument(
        "--keys",
        type=str,
        nargs="+",
        required=True,
        help="要拉取的 metric keys (可以指定多个)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="wandb_data.json",
        help="输出的 JSON 文件路径 (默认: wandb_data.json)"
    )
    parser.add_argument(
        "--api_key_file",
        type=str,
        default=None,
        help="API key 文件路径 (可选，不指定则使用 wandb login 的认证)"
    )

    args = parser.parse_args()

    fetch_wandb_data(
        entity=args.entity,
        project=args.project,
        groups=args.groups,
        metric_keys=args.keys,
        output_file=args.output,
        api_key_file=args.api_key_file
    )


if __name__ == "__main__":
    main()
