#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

GEN_CUDA_VISIBLE_DEVICES="${GEN_CUDA_VISIBLE_DEVICES:-0,1}"
GEN_HOST="${GEN_HOST:-127.0.0.1}"
GEN_PORT="${GEN_PORT:-18080}"
GEN_MODEL="${GEN_MODEL:-OpenGVLab/InternVL2_5-8B}"
GEN_TP="${GEN_TP:-2}"
GEN_IMAGE_LIMIT="${GEN_IMAGE_LIMIT:-8}"
GEN_PYTHON="${GEN_PYTHON:-python}"

export CUDA_VISIBLE_DEVICES="${GEN_CUDA_VISIBLE_DEVICES}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/aca:${REPO_ROOT}:${PYTHONPATH:-}"

"${GEN_PYTHON}" "${REPO_ROOT}/aca/gen_server_internvl25.py" \
  --model "${GEN_MODEL}" \
  --tensor-parallel-size "${GEN_TP}" \
  --host "${GEN_HOST}" \
  --port "${GEN_PORT}" \
  --image-limit "${GEN_IMAGE_LIMIT}"
