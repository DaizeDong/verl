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

"""Local reward compatibility hook for open-math-style offline evaluation."""


def reward_func(data_source, solution_str, ground_truth, extra_info=None):
    if not isinstance(solution_str, str) or not solution_str:
        return 0.0

    source = str(data_source)

    if source in {"math_dapo", "math", "math_dapo_reasoning"} or source.startswith("aime"):
        from verl.utils.reward_score import math_dapo

        return math_dapo.compute_score(solution_str, ground_truth)["acc"]

    if source in {"openai/gsm8k", "gsm8k"}:
        from verl.utils.reward_score import gsm8k

        return gsm8k.compute_score(solution_str, ground_truth)

    if source in {"Maxwell-Jia/AIME_2024", "opencompass/cnmo2024_en", "opencompass/cnmo2024_zh"}:
        from verl.utils.reward_score import math_reward

        return math_reward.compute_score(solution_str, ground_truth)

    if source == "math500_moonlight":
        # GSM8K-trained models output "#### <answer>". Extract everything after the
        # last "####" and compare with ground_truth using LaTeX equivalence.
        from verl.utils.reward_score import math_reward

        idx = solution_str.rfind("####")
        if idx < 0:
            return 0.0
        candidate = solution_str[idx + 4:].strip()
        # Trim surrounding $ signs and trailing punctuation that often follows.
        candidate = candidate.strip("$").strip().rstrip(".")
        if not candidate:
            return 0.0
        try:
            return 1.0 if math_reward.is_equiv(candidate, ground_truth) else 0.0
        except Exception:
            return 0.0

    if source == "Idavidrein/gpqa":
        from recipe.r1.tasks import gpqa

        return gpqa.compute_score(solution_str, ground_truth)

    if source in {"livecodebench/code_generation_lite", "livecodebench/code_generation"}:
        from recipe.r1.tasks import livecodebench

        return livecodebench.compute_score(solution_str, ground_truth)

    raise NotImplementedError(f"Unsupported data_source={source!r}")
