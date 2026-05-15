#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

JUDGE_CUDA_VISIBLE_DEVICES="${JUDGE_CUDA_VISIBLE_DEVICES:-3}"
JUDGE_HOST="${JUDGE_HOST:-127.0.0.1}"
JUDGE_PORT="${JUDGE_PORT:-8888}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen2.5-32B-Instruct}"
JUDGE_SERVED_NAME="${JUDGE_SERVED_NAME:-Qwen2.5-32B-Instruct}"
JUDGE_TP="${JUDGE_TP:-1}"
JUDGE_MEM_UTIL="${JUDGE_MEM_UTIL:-0.9}"
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-6000}"
JUDGE_PYTHON="${JUDGE_PYTHON:-python}"

export CUDA_VISIBLE_DEVICES="${JUDGE_CUDA_VISIBLE_DEVICES}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/aca:${REPO_ROOT}:${PYTHONPATH:-}"

"${JUDGE_PYTHON}" "${REPO_ROOT}/data_pipeline/start_vllm_server.py" \
  --model "${JUDGE_MODEL}" \
  --host "${JUDGE_HOST}" \
  --port "${JUDGE_PORT}" \
  --served-model-name "${JUDGE_SERVED_NAME}" \
  --tensor-parallel-size "${JUDGE_TP}" \
  --gpu-memory-utilization "${JUDGE_MEM_UTIL}" \
  --max-model-len "${JUDGE_MAX_MODEL_LEN}"
