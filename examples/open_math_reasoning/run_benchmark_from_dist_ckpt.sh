#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

ACTOR_CKPT_DIR="${ACTOR_CKPT_DIR:?ACTOR_CKPT_DIR is required}"
DATA_FILE="${DATA_FILE:?DATA_FILE is required}"

MERGED_HF_DIR="${MERGED_HF_DIR:-${ACTOR_CKPT_DIR%/}/merged_hf}"
BACKEND="${BACKEND:-megatron}"
TIE_WORD_EMBEDDING="${TIE_WORD_EMBEDDING:-1}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"

if [ "$SKIP_EXPORT" != "1" ]; then
    rm -rf "$MERGED_HF_DIR"
    mkdir -p "$MERGED_HF_DIR"

    MERGE_ARGS=(
        python3 scripts/legacy_model_merger.py merge
        --backend "$BACKEND"
        --local_dir "$ACTOR_CKPT_DIR"
        --target_dir "$MERGED_HF_DIR"
    )
    if [ "$TIE_WORD_EMBEDDING" = "1" ]; then
        MERGE_ARGS+=(--tie-word-embedding)
    fi
    "${MERGE_ARGS[@]}"
fi

MODEL_PATH="$MERGED_HF_DIR" \
DATA_FILE="$DATA_FILE" \
OUTPUT_FILE="${OUTPUT_FILE:-${DATA_FILE%.parquet}.gen.parquet}" \
REWARD_FN_PATH="${REWARD_FN_PATH:-$PROJECT_DIR/examples/open_math_reasoning/reward_score_compat.py}" \
bash "$PROJECT_DIR/examples/open_math_reasoning/run_benchmark_from_ckpt.sh"
