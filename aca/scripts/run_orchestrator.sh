#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PRM_CUDA_VISIBLE_DEVICES="${PRM_CUDA_VISIBLE_DEVICES:-2}"
PRM_CKPT="${PRM_CKPT:-}"
PRM_BACKEND="${PRM_BACKEND:-internvl}"
PRM_BASE_MODEL="${PRM_BASE_MODEL:-}"
DATASET_NAME="${DATASET_NAME:-}"
ORCH_PYTHON="${ORCH_PYTHON:-python}"

GEN_URL="${GEN_URL:-http://127.0.0.1:18080}"
INPUT_JSON="${INPUT_JSON:-}"
IMAGE_ROOT="${IMAGE_ROOT:-}"
OUTPUT_JSON="${OUTPUT_JSON:-}"
SEED="${SEED:--1}"
SAVE_DIR="${SAVE_DIR:-}"
SAVE_RAW_OVERSAMPLE="${SAVE_RAW_OVERSAMPLE:-0}"
RESUME="${RESUME:-0}"

JUDGE_URL="${JUDGE_URL:-}"
JUDGE_SERVED_NAME="${JUDGE_SERVED_NAME:-}"

N0="${N0:-4}"
N_TOTAL="${N_TOTAL:-16}"
M="${M:-4}"
OVERSAMPLE="${OVERSAMPLE:-2.0}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-4}"
RISK_LAMBDA="${RISK_LAMBDA:-0.5}"
C_STOP="${C_STOP:-0.3}"
EXPAND_POLICY="${EXPAND_POLICY:-ucb_runnerup}"
C_CUT="${C_CUT:-1.0}"
P_BAD="${P_BAD:-0.3}"
DISABLE_EARLY_STOP="${DISABLE_EARLY_STOP:-0}"

BASELINE_A_JSON="${BASELINE_A_JSON:-}"
BASELINE_TOKENIZER="${BASELINE_TOKENIZER:-}"

TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-30}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.05}"

INPUT_SIZE="${INPUT_SIZE:-0}"
DYNAMIC="${DYNAMIC:-1}"
USE_THUMBNAIL="${USE_THUMBNAIL:-}"
MAX_NUM="${MAX_NUM:-6}"

PRM_PROCESSOR_USE_FAST="${PRM_PROCESSOR_USE_FAST:-0}"
PRM_MAX_SEQ_LENGTH="${PRM_MAX_SEQ_LENGTH:-32768}"
PRM_MIN_PIXELS="${PRM_MIN_PIXELS:-200704}"
PRM_MAX_PIXELS="${PRM_MAX_PIXELS:-1003520}"
PRM_VIDEO_MIN_PIXELS="${PRM_VIDEO_MIN_PIXELS:-100352}"
PRM_VIDEO_MAX_PIXELS="${PRM_VIDEO_MAX_PIXELS:-602112}"
PRM_VIDEO_MIN_FRAMES="${PRM_VIDEO_MIN_FRAMES:-4}"
PRM_VIDEO_MAX_FRAMES="${PRM_VIDEO_MAX_FRAMES:-128}"
PRM_VIDEO_FPS="${PRM_VIDEO_FPS:-2.0}"
PRM_GRID_MAX_COLS="${PRM_GRID_MAX_COLS:-3}"
PRM_BF16="${PRM_BF16:-}"
PRM_LOAD_WEIGHTS_TO_GPU="${PRM_LOAD_WEIGHTS_TO_GPU:-0}"

if [[ -z "${PRM_CKPT}" || -z "${INPUT_JSON}" || -z "${OUTPUT_JSON}" ]]; then
  echo "PRM_CKPT, INPUT_JSON, and OUTPUT_JSON are required." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${PRM_CUDA_VISIBLE_DEVICES}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/aca:${REPO_ROOT}:${PYTHONPATH:-}"

cmd=("${ORCH_PYTHON}" "${REPO_ROOT}/aca/orchestrator_adaptive_bon.py"
  --input "${INPUT_JSON}"
  --output "${OUTPUT_JSON}"
  --prm-ckpt "${PRM_CKPT}"
  --prm-backend "${PRM_BACKEND}"
  --gen-url "${GEN_URL}"
  --seed "${SEED}"
  --n0 "${N0}" --n-total "${N_TOTAL}" --m "${M}"
  --oversample "${OVERSAMPLE}"
  --mini-batch-size "${MINI_BATCH_SIZE}"
  --risk-lambda "${RISK_LAMBDA}"
  --c-stop "${C_STOP}"
  --expand-policy "${EXPAND_POLICY}"
  --c-cut "${C_CUT}"
  --p-bad "${P_BAD}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --top-k "${TOP_K}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --repetition-penalty "${REPETITION_PENALTY}"
  --input-size "${INPUT_SIZE}"
  --max-num "${MAX_NUM}"
  --prm-max-seq-length "${PRM_MAX_SEQ_LENGTH}"
  --prm-min-pixels "${PRM_MIN_PIXELS}"
  --prm-max-pixels "${PRM_MAX_PIXELS}"
  --prm-video-min-pixels "${PRM_VIDEO_MIN_PIXELS}"
  --prm-video-max-pixels "${PRM_VIDEO_MAX_PIXELS}"
  --prm-video-min-frames "${PRM_VIDEO_MIN_FRAMES}"
  --prm-video-max-frames "${PRM_VIDEO_MAX_FRAMES}"
  --prm-video-fps "${PRM_VIDEO_FPS}"
  --prm-grid-max-cols "${PRM_GRID_MAX_COLS}"
)

if [[ -n "${IMAGE_ROOT}" ]]; then
  cmd+=(--image-root "${IMAGE_ROOT}")
fi

if [[ -n "${PRM_BASE_MODEL}" ]]; then
  cmd+=(--prm-base-model "${PRM_BASE_MODEL}")
fi

if [[ -n "${DATASET_NAME}" ]]; then
  cmd+=(--dataset-name "${DATASET_NAME}")
fi

if [[ "${DYNAMIC}" == "0" || "${DYNAMIC}" == "false" ]]; then
  cmd+=(--no-dynamic)
fi

if [[ "${USE_THUMBNAIL}" == "1" || "${USE_THUMBNAIL}" == "true" ]]; then
  cmd+=(--use-thumbnail)
elif [[ "${USE_THUMBNAIL}" == "0" || "${USE_THUMBNAIL}" == "false" ]]; then
  cmd+=(--no-thumbnail)
fi

if [[ "${PRM_PROCESSOR_USE_FAST}" == "1" || "${PRM_PROCESSOR_USE_FAST}" == "true" ]]; then
  cmd+=(--prm-processor-use-fast)
fi

if [[ "${PRM_BF16}" == "1" || "${PRM_BF16}" == "true" ]]; then
  cmd+=(--prm-bf16)
elif [[ "${PRM_BF16}" == "0" || "${PRM_BF16}" == "false" ]]; then
  cmd+=(--no-prm-bf16)
fi

if [[ "${PRM_LOAD_WEIGHTS_TO_GPU}" == "1" || "${PRM_LOAD_WEIGHTS_TO_GPU}" == "true" ]]; then
  cmd+=(--prm-load-weights-to-gpu)
fi

if [[ -n "${BASELINE_A_JSON}" ]]; then
  cmd+=(--baseline-a-json "${BASELINE_A_JSON}")
  if [[ -n "${BASELINE_TOKENIZER}" ]]; then
    cmd+=(--baseline-tokenizer "${BASELINE_TOKENIZER}")
  fi
fi

if [[ "${DISABLE_EARLY_STOP}" == "1" || "${DISABLE_EARLY_STOP}" == "true" ]]; then
  cmd+=(--disable-early-stop)
fi

if [[ -n "${SAVE_DIR}" ]]; then
  cmd+=(--save-intermediate-dir "${SAVE_DIR}")
  if [[ "${SAVE_RAW_OVERSAMPLE}" == "1" || "${SAVE_RAW_OVERSAMPLE}" == "true" ]]; then
    cmd+=(--save-raw-oversample)
  fi
fi

if [[ "${RESUME}" == "1" || "${RESUME}" == "true" ]]; then
  cmd+=(--resume)
fi

if [[ -n "${JUDGE_URL}" && -n "${JUDGE_SERVED_NAME}" ]]; then
  cmd+=(--judge-url "${JUDGE_URL}" --judge-model "${JUDGE_SERVED_NAME}")
fi

"${cmd[@]}"
