#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
DATA_FILE="${DATA_FILE:?DATA_FILE is required}"
OUTPUT_FILE="${OUTPUT_FILE:-${DATA_FILE%.parquet}.gen.parquet}"

PROMPT_KEY="${PROMPT_KEY:-prompt}"
RESPONSE_KEY="${RESPONSE_KEY:-responses}"
REWARD_FN_PATH="${REWARD_FN_PATH:-$PROJECT_DIR/examples/open_math_reasoning/reward_score_compat.py}"

ROLLOUT_NAME="${ROLLOUT_NAME:-sglang}"
LOAD_FORMAT="${LOAD_FORMAT:-hf}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
TP_SIZE="${TP_SIZE:-1}"
N_SAMPLES="${N_SAMPLES:-8}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:--1}"
PROMPT_LENGTH="${PROMPT_LENGTH:-1024}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
SKIP_GENERATION="${SKIP_GENERATION:-0}"

if [ ! -f "$PROJECT_DIR/recipe/r1/tasks/math_reward.py" ]; then
    git -C "$PROJECT_DIR" submodule update --init --recursive recipe
fi

if [ "$SKIP_GENERATION" != "1" ]; then
    python3 -m verl.trainer.main_generation_server \
        trainer.nnodes="$NNODES" \
        trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
        data.train_files="$DATA_FILE" \
        data.prompt_key="$PROMPT_KEY" \
        data.output_path="$OUTPUT_FILE" \
        actor_rollout_ref.model.path="$MODEL_PATH" \
        actor_rollout_ref.model.trust_remote_code="$TRUST_REMOTE_CODE" \
        actor_rollout_ref.rollout.name="$ROLLOUT_NAME" \
        actor_rollout_ref.rollout.load_format="$LOAD_FORMAT" \
        actor_rollout_ref.rollout.n="$N_SAMPLES" \
        actor_rollout_ref.rollout.temperature="$TEMPERATURE" \
        actor_rollout_ref.rollout.top_p="$TOP_P" \
        actor_rollout_ref.rollout.top_k="$TOP_K" \
        actor_rollout_ref.rollout.prompt_length="$PROMPT_LENGTH" \
        actor_rollout_ref.rollout.response_length="$RESPONSE_LENGTH" \
        actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE" \
        actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
        actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS"
fi

python3 -m verl.trainer.main_eval \
    data.path="$OUTPUT_FILE" \
    data.prompt_key="$PROMPT_KEY" \
    data.response_key="$RESPONSE_KEY" \
    custom_reward_function.path="$REWARD_FN_PATH" \
    custom_reward_function.name=reward_func
