#!/usr/bin/env bash
set -euo pipefail
set -x

############################################
# Basic (match reference style)
############################################
NNODES=${NNODES:-1}
NGPUS_PER_NODES=${NGPUS_PER_NODES:-6}

# In Slurm, prefer Slurm envs if present
SLURM_NNODES=${SLURM_NNODES:-$NNODES}
SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-$NGPUS_PER_NODES}

############################################
# Data paths
############################################
export TRAIN_FILE=${TRAIN_FILE:-"/common/users/jc3585/olmoe/data/gsm8k/train.parquet"}
export TEST_FILE=${TEST_FILE:-"/common/users/jc3585/olmoe/data/gsm8k/test.parquet"}
# export TRAIN_FILE=${TRAIN_FILE:-"/common/users/jc3585/olmoe/data/DAPO-Math-17k/train.parquet"}
# export TEST_FILE=${TEST_FILE:-"/common/users/jc3585/olmoe/data/aime-2024-full/train.parquet"}

############################################
# Safer NCCL defaults (single-node PCIe)
############################################
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29602}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

export WANDB_DIR=/common/users/jc3585/olmoe/wandb
export WANDB_CACHE_DIR=/common/users/jc3585/olmoe/wandb/cache
export WANDB_ARTIFACTS_DIR=/common/users/jc3585/olmoe/wandb/artifacts


############################################
# Python path (Ray workers import verl.*)
############################################
export PYTHONPATH="/common/users/jc3585/olmoe/verl${PYTHONPATH:+:$PYTHONPATH}"

############################################
# W&B (do NOT hardcode key here)
############################################
export WANDB_API_KEY=5c8208adc889d4ac9073a9badf17b719482e81e3
export WANDB_GROUP=${WANDB_GROUP:-"olmoe"}
TIME=$(date +%Y%m%d-%H%M%S)
USE_PPO=${USE_PPO:-0}
if [ "${USE_PPO}" -eq 1 ]; then
  export WANDB_PROJECT=${WANDB_PROJECT:-"verl-PREEXP-OLMoE-1B-7B-0125-Instruct-GSM8K-PPO"}
  export WANDB_NAME=${WANDB_NAME:-"ppo-local-${TIME}"}
else
  export WANDB_PROJECT=${WANDB_PROJECT:-"verl-PREEXP-OLMoE-1B-7B-0125-Instruct-GSM8K-GRPO"}
  export WANDB_NAME=${WANDB_NAME:-"grpo-local-${TIME}"}
fi
export WANDB_MODE=${WANDB_MODE:-"online"}   # or offline

############################################
# Chat template
# Leave empty to use the tokenizer's built-in chat template (recommended).
############################################
export CHAT_TEMPLATE=${CHAT_TEMPLATE:-""}

############################################
# Model path (prefer NVMe cache if exists)
############################################
CACHE_ROOT=${HF_CACHE_ROOT:-${HF_NVME_CACHE:-/tmp/hf_cache}}

DEFAULT_MODEL_PATH="/common/users/jc3585/hf_cache/hub/models--allenai--OLMoE-1B-7B-0125-Instruct/snapshots/b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e"

MODEL_PATH=${MODEL_PATH:-$DEFAULT_MODEL_PATH}

# If not set, use a simple chat template for non-Instruct models.
if [ -z "${CHAT_TEMPLATE}" ]; then
  if [[ "${MODEL_PATH}" != *Instruct* ]]; then
    CHAT_TEMPLATE=$(cat <<'EOF'
{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}
{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}
EOF
)
  fi
fi

############################################
# Dist checkpoint (TP1)
############################################
CKPT_TP1=${CKPT_TP1:-"/common/users/jc3585/olmoe/megatron_ckpt/olmoe_1b7b_0125_Instruct_tp1_pp1_mcore_fullqk"}

############################################
# Parallel (keep your minimal-memory TP=1)
############################################
COMMON_PP=${COMMON_PP:-1}
COMMON_VPP=${COMMON_VPP:-null}
COMMON_CP=${COMMON_CP:-1}
COMMON_TP=${COMMON_TP:-1}
COMMON_EP=${COMMON_EP:-1}
COMMON_ETP=${COMMON_ETP:-1}

ACTOR_PP=${ACTOR_PP:-$COMMON_PP}
ACTOR_VPP=${ACTOR_VPP:-$COMMON_VPP}
ACTOR_CP=${ACTOR_CP:-$COMMON_CP}
ACTOR_TP=${ACTOR_TP:-$COMMON_TP}
ACTOR_EP=${ACTOR_EP:-$COMMON_EP}
ACTOR_ETP=${ACTOR_ETP:-$COMMON_ETP}

REF_PP=${REF_PP:-$COMMON_PP}
REF_VPP=${REF_VPP:-$COMMON_VPP}
REF_CP=${REF_CP:-$COMMON_CP}
REF_TP=${REF_TP:-$COMMON_TP}
REF_EP=${REF_EP:-$COMMON_EP}
REF_ETP=${REF_ETP:-$COMMON_ETP}

CRITIC_PP=${CRITIC_PP:-$COMMON_PP}
CRITIC_VPP=${CRITIC_VPP:-$COMMON_VPP}
CRITIC_CP=${CRITIC_CP:-$COMMON_CP}
CRITIC_TP=${CRITIC_TP:-$COMMON_TP}
CRITIC_EP=${CRITIC_EP:-$COMMON_EP}
CRITIC_ETP=${CRITIC_ETP:-$COMMON_ETP}

# Rollout TP (in reference they separate TRAIN_TP vs INFER_TP)
INFER_TP=${INFER_TP:-1}

############################################
# Algo / lengths / buffers (match reference knobs)
############################################
if [ "${USE_PPO}" -eq 1 ]; then
  adv_estimator=${adv_estimator:-gae}
else
  adv_estimator=${adv_estimator:-grpo}
fi

use_kl_in_reward=${use_kl_in_reward:-False}
kl_coef=${kl_coef:-0.0}
use_kl_loss=${use_kl_loss:-False}
kl_loss_coef=${kl_loss_coef:-0.0}

clip_ratio_low=${clip_ratio_low:-0.2}
clip_ratio_high=${clip_ratio_high:-0.4}
clip_ratio_c=${clip_ratio_c:-10.0}
# You currently had max_prompt_length=1024 and no max_response_length.
# Reference uses much longer; keep conservative defaults, but expose knobs:
max_prompt_length=${max_prompt_length:-1024}
max_response_length=${max_response_length:-3072}
max_ctx_len=${max_ctx_len:-4096}
if [ $((max_prompt_length + max_response_length)) -gt "${max_ctx_len}" ]; then
  max_response_length=$((max_ctx_len - max_prompt_length))
fi

enable_overlong_buffer=${enable_overlong_buffer:-True}
overlong_buffer_len=${overlong_buffer_len:-128}
overlong_penalty_factor=${overlong_penalty_factor:-1.0}

loss_agg_mode=${loss_agg_mode:-"token-mean"}

############################################
# Batch sizes (align names to reference style)
############################################
train_prompt_bsz=${train_prompt_bsz:-24}  # your data.train_batch_size=24
if [ "${USE_PPO}" -eq 1 ]; then
  n_resp_per_prompt=${n_resp_per_prompt:-4}
else
  n_resp_per_prompt=${n_resp_per_prompt:-8}  # your rollout.n=8
fi
train_prompt_mini_bsz=${train_prompt_mini_bsz:-24}  # your ppo_mini_batch_size=24
train_ppo_micro_batch_size_per_gpu=${train_ppo_micro_batch_size_per_gpu:-1}  # your micro=1
infer_ppo_micro_batch_size_per_gpu=${infer_ppo_micro_batch_size_per_gpu:-1}  # your logprob micro=1
critic_ppo_micro_batch_size_per_gpu=${critic_ppo_micro_batch_size_per_gpu:-$train_ppo_micro_batch_size_per_gpu}

# Token limits per GPU (reference uses these to avoid OOM / enable dynamic bsz)
use_dynamic_bsz=${use_dynamic_bsz:-True}
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))

############################################
# Rollout sampling 
############################################
rollout_temperature=${rollout_temperature:-1}
val_temperature=${val_temperature:-0.2}
top_p=${top_p:-0.9}
top_k=${top_k:--1}
val_top_p=${val_top_p:-0.7}

over_sample_rate=${over_sample_rate:-0.1}

############################################
# Offload 
############################################
offload=${offload:-True}
if [ "${USE_PPO}" -eq 1 ]; then
  optimizer_offload_fraction=${optimizer_offload_fraction:-1.0}
else
  optimizer_offload_fraction=${optimizer_offload_fraction:-0.0}
fi

############################################
# Ref model toggle
# Your current script effectively disables ref. Keep default disabled,
# but allow turning it on by: ENABLE_REF=1
############################################
ENABLE_REF=${ENABLE_REF:-0}

############################################
# Optional: start local Ray head (same behavior as your script)
############################################
if [ -z "${SKIP_RAY_START:-}" ]; then
  export RAY_memory_monitor_refresh_ms=0
  export RAY_ADDRESS=${RAY_ADDRESS:-127.0.0.1:9339}
  ray stop -f >/dev/null 2>&1 || true
  ray start \
    --head \
    --port=9339 \
    --num-cpus=128 \
    --num-gpus=6 \
    --include-dashboard=false \
    --disable-usage-stats
fi

############################################
# Config path/name (reference uses --config-path)
# If you don’t have ./config, leave CONFIG_PATH empty and use -cn only.
############################################
CONFIG_PATH=${CONFIG_PATH:-""}
CONFIG_NAME=${CONFIG_NAME:-"ppo_megatron_trainer"}

CFG_ARGS=()
if [ -n "${CONFIG_PATH}" ]; then
  CFG_ARGS+=(--config-path="${CONFIG_PATH}" --config-name="${CONFIG_NAME}")
else
  # your original style
  CFG_ARGS+=(-cn "${CONFIG_NAME}")
fi

CHAT_TEMPLATE_ARG=()
if [ -n "${CHAT_TEMPLATE}" ]; then
  CHAT_TEMPLATE_ARG=('+data.apply_chat_template_kwargs.chat_template=${oc.env:CHAT_TEMPLATE}')
fi

CRITIC_ARGS=()
if [ "${USE_PPO}" -eq 1 ]; then
  CRITIC_ARGS+=(
    "critic.model.path=${MODEL_PATH}"
    "critic.model.tokenizer_path=${MODEL_PATH}"
    "critic.model.trust_remote_code=True"
    "+critic.model.override_config.model_config.architectures=['OlmoeForCausalLM']"
    "+critic.model.override_config.model_config.max_position_embeddings=$((max_prompt_length + max_response_length))"
    "critic.ppo_micro_batch_size_per_gpu=${critic_ppo_micro_batch_size_per_gpu}"
    "critic.megatron.use_dist_checkpointing=True"
    "critic.megatron.dist_checkpointing_path=${CKPT_TP1}"
    "critic.megatron.tensor_model_parallel_size=${CRITIC_TP}"
    "critic.megatron.pipeline_model_parallel_size=${CRITIC_PP}"
    "critic.megatron.virtual_pipeline_model_parallel_size=${CRITIC_VPP}"
    "critic.megatron.context_parallel_size=${CRITIC_CP}"
    "critic.megatron.expert_model_parallel_size=${CRITIC_EP}"
    "critic.megatron.expert_tensor_parallel_size=${CRITIC_ETP}"
  )
fi

############################################
# Launch (reference-style param list)
############################################
python3 -m verl.trainer.main_ppo "${CFG_ARGS[@]}" \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TEST_FILE}" \
  data.prompt_key=prompt \
  data.truncation="left" \
  data.max_prompt_length=${max_prompt_length} \
  data.max_response_length=${max_response_length} \
  data.filter_overlong_prompts=True \
  data.filter_overlong_prompts_workers=1 \
  data.train_batch_size=${train_prompt_bsz} \
  "${CHAT_TEMPLATE_ARG[@]}" \
  \
  algorithm.adv_estimator=${adv_estimator} \
  algorithm.use_kl_in_reward=${use_kl_in_reward} \
  algorithm.kl_ctrl.kl_coef=${kl_coef} \
  \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.tokenizer_path="${MODEL_PATH}" \
  actor_rollout_ref.model.trust_remote_code=True \
  "+actor_rollout_ref.model.override_config.model_config.architectures=['OlmoeForCausalLM']" \
  "+actor_rollout_ref.model.override_config.model_config.max_position_embeddings=$((max_prompt_length + max_response_length))" \
  \
  "${CRITIC_ARGS[@]}" \
  \
  actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
  actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
  actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
  actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
  actor_rollout_ref.actor.clip_ratio_c=${clip_ratio_c} \
  actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
  actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_ppo_micro_batch_size_per_gpu} \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
  actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
  \
  actor_rollout_ref.actor.megatron.use_mbridge=False \
  actor_rollout_ref.actor.megatron.sequence_parallel=False \
  actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
  actor_rollout_ref.actor.megatron.use_distributed_optimizer=True \
  actor_rollout_ref.actor.megatron.dist_checkpointing_path="${CKPT_TP1}" \
  actor_rollout_ref.actor.megatron.param_offload=${offload} \
  actor_rollout_ref.actor.megatron.grad_offload=${offload} \
  actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP} \
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP} \
  actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=${ACTOR_VPP} \
  actor_rollout_ref.actor.megatron.context_parallel_size=${ACTOR_CP} \
  actor_rollout_ref.actor.megatron.expert_model_parallel_size=${ACTOR_EP} \
  actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ACTOR_ETP} \
  "+actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${optimizer_offload_fraction}" \
  \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.skip_rollout=False \
  actor_rollout_ref.rollout.'n'=${n_resp_per_prompt} \
  actor_rollout_ref.rollout.over_sample_rate=${over_sample_rate} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${INFER_TP} \
  "+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${infer_ppo_micro_batch_size_per_gpu} \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
  actor_rollout_ref.rollout.temperature=${rollout_temperature} \
  actor_rollout_ref.rollout.top_p=${top_p} \
  actor_rollout_ref.rollout.top_k=${top_k} \
  actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
  actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
  actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  \
  reward_model.reward_manager=dapo \
  "+reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer}" \
  "+reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len}" \
  "+reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor}" \
  "+reward_model.reward_kwargs.overlong_buffer_cfg.log=False" \
  "+reward_model.reward_kwargs.max_resp_len=${max_response_length}" \
  \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${WANDB_PROJECT}" \
  trainer.experiment_name="${WANDB_NAME}" \
  trainer.n_gpus_per_node=6 \
  trainer.nnodes=1 \
  trainer.test_freq=10 \
  trainer.total_epochs=1 \
  +trainer.rollout_data_dir="/common/users/jc3585/olmoe/outputs/${TIME}/rollout_logs" \
  +trainer.validation_data_dir="/common/users/jc3585/olmoe/outputs/${TIME}/val_logs" \
  trainer.log_val_generations=10 \
  trainer.save_freq=200 \
  +trainer.save_freq_final=1 \
  \
  $( [ "${ENABLE_REF}" -eq 1 ] && echo "+actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${infer_ppo_micro_batch_size_per_gpu}" || true ) \
  $( [ "${ENABLE_REF}" -eq 1 ] && echo "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}" || true ) \
  $( [ "${ENABLE_REF}" -eq 1 ] && echo "actor_rollout_ref.ref.megatron.use_dist_checkpointing=True" || true ) \
  $( [ "${ENABLE_REF}" -eq 1 ] && echo "actor_rollout_ref.ref.megatron.dist_checkpointing_path=${CKPT_TP1}" || true )
