set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

#export CUDA_VISIBLE_DEVICES="0,1,2,3"
GPUS=${GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-512}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


META_PATH=${META_PATH:-"${REPO_ROOT}/shell/data/meta_visualprm400k_beta_binom.json"}
MODEL_PATH=${MODEL_PATH:-"OpenGVLab/InternVL2_5-8B"}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"${REPO_ROOT}/configs/zero_stage3_config.json"}

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
NPROC_PER_NODE=${NPROC_PER_NODE:-$GPUS}

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export MASTER_PORT=4322
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi
if [ -n "${CUDA_HOME:-}" ]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi

CUDA_TARGETS_INCLUDE="${CUDA_HOME}/targets/x86_64-linux/include"
CUDA_CCCL_INCLUDE="${CUDA_HOME}/targets/x86_64-linux/include/cccl"
if [ -d "${CUDA_TARGETS_INCLUDE}" ]; then
  export CPATH="${CUDA_TARGETS_INCLUDE}:${CPATH:-}"
  export CPLUS_INCLUDE_PATH="${CUDA_TARGETS_INCLUDE}:${CPLUS_INCLUDE_PATH:-}"
fi
if [ -d "${CUDA_CCCL_INCLUDE}" ]; then
  export CPATH="${CUDA_CCCL_INCLUDE}:${CPATH:-}"
  export CPLUS_INCLUDE_PATH="${CUDA_CCCL_INCLUDE}:${CPLUS_INCLUDE_PATH:-}"
fi

export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/ephemeral/cache/torch_extensions}
mkdir -p "${TORCH_EXTENSIONS_DIR}"
#module load gcc-13.3.0
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/work_dirs/internvl_chat_v2_5/visualprm400K-Beta-Binomial"}

# Beta-Binom PRM hyperparams (can be overridden by environment variables)
BETA_BINOM_KAPPA_MIN=${BETA_BINOM_KAPPA_MIN:-1e-3}
BETA_BINOM_KAPPA_INIT=${BETA_BINOM_KAPPA_INIT:-4.0}
BETA_BINOM_EVI_REG=${BETA_BINOM_EVI_REG:-5e-2}
BETA_BINOM_KAPPA_HEAD_LR_MULT=${BETA_BINOM_KAPPA_HEAD_LR_MULT:-10.0}

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

python -m torch.distributed.run \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --nproc_per_node=${NPROC_PER_NODE} \
  --master_port=${MASTER_PORT} \
  "${REPO_ROOT}/src/internvl/train/internvl_chat_finetune_beta_binom.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "${META_PATH}" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.4 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --save_strategy "steps" \
  --save_only_model True \
  --save_steps 150 \
  --save_total_limit 18 \
  --learning_rate 1e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.05 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 8192 \
  --beta_binom_kappa_min ${BETA_BINOM_KAPPA_MIN} \
  --beta_binom_kappa_init ${BETA_BINOM_KAPPA_INIT} \
  --beta_binom_evi_reg ${BETA_BINOM_EVI_REG} \
  --beta_binom_kappa_head_lr_mult ${BETA_BINOM_KAPPA_HEAD_LR_MULT} \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
