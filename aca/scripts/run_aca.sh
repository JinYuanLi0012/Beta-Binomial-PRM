#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage:
  bash aca/scripts/run_aca.sh --prm-ckpt PATH [options]

Options:
  --mode aca|aca-noearlystop|purebon
  --input PATH
  --image-root PATH
  --output PATH
  --gen-model PATH_OR_HF_ID
  --judge-model PATH_OR_HF_ID
  --no-judge
  --gen-gpus IDS
  --prm-gpu ID
  --judge-gpu IDS
  --gen-port PORT
  --judge-port PORT
  --gen-tp N
  --judge-tp N
  --seed INT
  --cache-root PATH
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODE="${MODE:-aca}"
GEN_MODEL="${GEN_MODEL:-OpenGVLab/InternVL2_5-8B}"
PRM_CKPT="${PRM_CKPT:-}"
INPUT_JSON="${INPUT_JSON:-${REPO_ROOT}/datasets/MathVista/seed_dataset.json}"
IMAGE_ROOT="${IMAGE_ROOT:-${REPO_ROOT}/datasets/MathVista}"
OUTPUT_JSON="${OUTPUT_JSON:-${REPO_ROOT}/work_dirs/aca_mathvista/out_aca.json}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen2.5-32B-Instruct}"

GEN_CUDA_VISIBLE_DEVICES="${GEN_CUDA_VISIBLE_DEVICES:-0,1}"
PRM_CUDA_VISIBLE_DEVICES="${PRM_CUDA_VISIBLE_DEVICES:-2}"
JUDGE_CUDA_VISIBLE_DEVICES="${JUDGE_CUDA_VISIBLE_DEVICES:-3}"
GEN_HOST="${GEN_HOST:-127.0.0.1}"
GEN_PORT="${GEN_PORT:-18080}"
GEN_TP="${GEN_TP:-2}"
JUDGE_HOST="${JUDGE_HOST:-127.0.0.1}"
JUDGE_PORT="${JUDGE_PORT:-8888}"
JUDGE_TP="${JUDGE_TP:-1}"
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-6000}"
GEN_HEALTHZ_TIMEOUT="${GEN_HEALTHZ_TIMEOUT:-1800}"
JUDGE_HEALTHZ_TIMEOUT="${JUDGE_HEALTHZ_TIMEOUT:-240}"
SEED="${SEED:--1}"
NO_JUDGE="${NO_JUDGE:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2;;
    --gen-model) GEN_MODEL="$2"; shift 2;;
    --prm-ckpt) PRM_CKPT="$2"; shift 2;;
    --input) INPUT_JSON="$2"; shift 2;;
    --image-root) IMAGE_ROOT="$2"; shift 2;;
    --output) OUTPUT_JSON="$2"; shift 2;;
    --judge-model) JUDGE_MODEL="$2"; shift 2;;
    --no-judge) NO_JUDGE="1"; shift 1;;
    --gen-gpus) GEN_CUDA_VISIBLE_DEVICES="$2"; shift 2;;
    --prm-gpu) PRM_CUDA_VISIBLE_DEVICES="$2"; shift 2;;
    --judge-gpu) JUDGE_CUDA_VISIBLE_DEVICES="$2"; shift 2;;
    --gen-port) GEN_PORT="$2"; shift 2;;
    --judge-port) JUDGE_PORT="$2"; shift 2;;
    --gen-tp) GEN_TP="$2"; shift 2;;
    --judge-tp) JUDGE_TP="$2"; shift 2;;
    --judge-max-model-len) JUDGE_MAX_MODEL_LEN="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --cache-root) CACHE_ROOT="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1;;
  esac
done

if [[ -z "${PRM_CKPT}" ]]; then
  echo "--prm-ckpt is required." >&2
  exit 1
fi

case "${MODE}" in
  aca)
    N0="${N0:-4}"
    N_TOTAL="${N_TOTAL:-16}"
    M="${M:-4}"
    DISABLE_EARLY_STOP="${DISABLE_EARLY_STOP:-0}"
    ;;
  aca-noearlystop)
    N0="${N0:-4}"
    N_TOTAL="${N_TOTAL:-16}"
    M="${M:-4}"
    DISABLE_EARLY_STOP="${DISABLE_EARLY_STOP:-1}"
    ;;
  purebon)
    N0="${N0:-16}"
    N_TOTAL="${N_TOTAL:-16}"
    M="${M:-0}"
    DISABLE_EARLY_STOP="${DISABLE_EARLY_STOP:-1}"
    ;;
  *)
    echo "Unsupported --mode: ${MODE}" >&2
    exit 1
    ;;
esac

OUT_DIR="$(dirname "${OUTPUT_JSON}")"
LOG_DIR="${OUT_DIR}/logs"
SAVE_DIR="${SAVE_DIR:-${OUT_DIR}/intermediate}"
mkdir -p "${LOG_DIR}" "${SAVE_DIR}"

if [[ -n "${CACHE_ROOT:-}" ]]; then
  export HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
  export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${CACHE_ROOT}/vllm}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/torchinductor}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
  mkdir -p "${HF_HOME}" "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
fi

cleanup() {
  set +e
  if [[ -n "${GEN_PID:-}" ]]; then kill "${GEN_PID}" 2>/dev/null || true; fi
  if [[ -n "${JUDGE_PID:-}" ]]; then kill "${JUDGE_PID}" 2>/dev/null || true; fi
}
trap cleanup EXIT

export GEN_MODEL GEN_CUDA_VISIBLE_DEVICES GEN_HOST GEN_PORT GEN_TP
echo "[run_aca] starting policy generator..."
bash "${SCRIPT_DIR}/start_gen.sh" > "${LOG_DIR}/gen.log" 2>&1 &
GEN_PID=$!

GEN_URL="http://${GEN_HOST}:${GEN_PORT}"
echo "[run_aca] waiting for ${GEN_URL}/healthz ..."
for _ in $(seq 1 "${GEN_HEALTHZ_TIMEOUT}"); do
  if curl -sf "${GEN_URL}/healthz" >/dev/null; then
    break
  fi
  sleep 1
done
curl -sf "${GEN_URL}/healthz" >/dev/null || {
  echo "Generator service is not ready. See ${LOG_DIR}/gen.log" >&2
  exit 1
}

JUDGE_URL=""
JUDGE_SERVED_NAME=""
if [[ "${NO_JUDGE}" != "1" && -n "${JUDGE_MODEL}" ]]; then
  export JUDGE_MODEL JUDGE_CUDA_VISIBLE_DEVICES JUDGE_HOST JUDGE_PORT JUDGE_TP JUDGE_MAX_MODEL_LEN
  echo "[run_aca] starting judge server..."
  bash "${SCRIPT_DIR}/start_judge.sh" > "${LOG_DIR}/judge.log" 2>&1 &
  JUDGE_PID=$!
  JUDGE_URL="http://${JUDGE_HOST}:${JUDGE_PORT}/v1"
  JUDGE_SERVED_NAME="${JUDGE_SERVED_NAME:-Qwen2.5-32B-Instruct}"
  echo "[run_aca] waiting for ${JUDGE_URL}/models ..."
  for _ in $(seq 1 "${JUDGE_HEALTHZ_TIMEOUT}"); do
    if curl -sf "${JUDGE_URL}/models" >/dev/null; then
      break
    fi
    sleep 1
  done
  curl -sf "${JUDGE_URL}/models" >/dev/null || {
    echo "Judge server is not ready. See ${LOG_DIR}/judge.log" >&2
    exit 1
  }
fi

export PRM_CKPT INPUT_JSON IMAGE_ROOT OUTPUT_JSON GEN_URL JUDGE_URL JUDGE_SERVED_NAME
export PRM_CUDA_VISIBLE_DEVICES SEED SAVE_DIR RESUME="${RESUME:-1}"
export N0 N_TOTAL M DISABLE_EARLY_STOP

echo "[run_aca] running orchestrator mode=${MODE} output=${OUTPUT_JSON}"
bash "${SCRIPT_DIR}/run_orchestrator.sh" | tee "${LOG_DIR}/orchestrator.log"

echo "[run_aca] summarizing..."
python "${REPO_ROOT}/aca/summarize_results.py" --input "${OUTPUT_JSON}" | tee "${LOG_DIR}/summary.log"
