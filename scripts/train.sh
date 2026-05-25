#!/bin/bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# ConGra QLoRA 학습 스크립트
# GPU 인스턴스(L4 24GB)에서 실행
#
# 단일 모드:   bash scripts/train.sh --mode baseline
# Qwen 1.5B:   bash scripts/train.sh --mode baseline --qwen-1.5b
# 전체 ablation: bash scripts/train.sh --ablation
# ═══════════════════════════════════════════════════════════════

# ── 실험 설정 ─────────────────────────────────────────────────
MODE="baseline"
CONTEXT_LINES=20

MODEL_NAME="deepseek-ai/deepseek-coder-1.3b-base"
MODEL_TAG="deepseek-coder-1.3b-base"

BATCH_SIZE=2
EVAL_BATCH_SIZE=1
GRAD_ACCUM=2             # effective batch = BATCH_SIZE * GRAD_ACCUM
NUM_EPOCHS=10            # early stopping이 실제 제어
LR=2e-4
MAX_SEQ_LEN=2048

LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

EVAL_STEPS=200
PATIENCE=3

SEED=42
USE_WANDB=false
ABLATION=false

# ── 인자 파싱 ─────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)       MODE="$2";          shift 2 ;;
        --ctx)        CONTEXT_LINES="$2"; shift 2 ;;
        --model)      MODEL_NAME="$2"; MODEL_TAG="$(basename "$2" | tr '[:upper:]' '[:lower:]')"; shift 2 ;;
        --batch)      BATCH_SIZE="$2";    shift 2 ;;
        --eval-batch) EVAL_BATCH_SIZE="$2"; shift 2 ;;
        --lr)         LR="$2";            shift 2 ;;
        --wandb)      USE_WANDB=true;     shift   ;;
        --resume)     RESUME="$2";        shift 2 ;;
        --ablation)   ABLATION=true;      shift   ;;
        # 6.7B 프리셋
        --6.7b)       MODEL_NAME="deepseek-ai/deepseek-coder-6.7b-base"; MODEL_TAG="deepseek-coder-6.7b-base"; BATCH_SIZE=1; GRAD_ACCUM=8; shift ;;
        # qwen 1.5b
        --qwen-1.5b)  MODEL_NAME="/opt/dlami/nvme/models/qwen2.5-coder-1.5b"; MODEL_TAG="qwen2.5-coder-1.5b"; BATCH_SIZE=1; EVAL_BATCH_SIZE=1; GRAD_ACCUM=4; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

CKPT_NAME_SUFFIX=""
if [ "${MODEL_TAG}" != "deepseek-coder-1.3b-base" ]; then
    CKPT_NAME_SUFFIX="_${MODEL_TAG}"
fi


# ── Python 환경 선택 ──────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! ${PYTHON_BIN} -c "import torch" >/dev/null 2>&1; then
    if command -v conda >/dev/null 2>&1 && conda run -n final_project python -c "import torch" >/dev/null 2>&1; then
        PYTHON_BIN="conda run --no-capture-output -n final_project python"
        echo "[INFO] Using conda env: final_project"
    else
        echo "[ERROR] torch not found in ${PYTHON_BIN}. Activate/install the training env first."
        echo "        Try: conda activate final_project"
        exit 1
    fi
fi

# ── 로그 설정 ─────────────────────────────────────────────────
LOG_DIR="./logs/${MODEL_TAG}"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)



# ── NVMe 모델 복구/다운로드 ──────────────────────────────────
ensure_model_available() {
    if [[ "${MODEL_NAME}" != /* ]]; then
        return 0
    fi

    if [ -f "${MODEL_NAME}/config.json" ]; then
        return 0
    fi

    case "${MODEL_TAG}" in
        qwen2.5-coder-1.5b)
            echo ""
            echo "[Model] Local model missing: ${MODEL_NAME}"
            echo "[Model] Restoring/downloading Qwen 1.5B to NVMe"
            bash scripts/download_model.sh --qwen-1.5b
            ;;
        deepseek-coder-6.7b-base)
            echo ""
            echo "[Model] Local model missing: ${MODEL_NAME}"
            echo "[Model] Restoring/downloading DeepSeek Coder 6.7B to NVMe"
            bash scripts/download_model.sh --6.7b
            ;;
        deepseek-coder-1.3b-base)
            echo ""
            echo "[Model] Local model missing: ${MODEL_NAME}"
            echo "[Model] Restoring/downloading DeepSeek Coder 1.3B to NVMe"
            bash scripts/download_model.sh --1.3b
            ;;
        *)
            echo "[ERROR] Local model path missing and no restore preset is known: ${MODEL_NAME}"
            exit 1
            ;;
    esac

    if [ ! -f "${MODEL_NAME}/config.json" ]; then
        echo "[ERROR] Model restore failed: ${MODEL_NAME}"
        exit 1
    fi
}

# ── 로컬 복사본에서 NVMe 체크포인트 복구 ─────────────────────
restore_ckpt_if_needed() {
    local m="$1"
    local ckpt_name="${m}_ctx${CONTEXT_LINES}${CKPT_NAME_SUFFIX}"
    local nvme_dir="/opt/dlami/nvme/ckpts/${ckpt_name}"
    local local_dir="./ckpts/${ckpt_name}"

    if [ -d "${nvme_dir}" ]; then
        return 0
    fi

    if [ ! -d "${local_dir}" ]; then
        return 0
    fi

    echo ""
    echo "[Restore] NVMe checkpoint missing; restoring local copy"
    echo "[Restore] ${local_dir} → ${nvme_dir}"
    mkdir -p "/opt/dlami/nvme/ckpts"
    cp -r "${local_dir}" "${nvme_dir}"
    echo "[Restore] Done: ${nvme_dir}"
}

# ── 체크포인트 복사 ───────────────────────────────────────────
copy_ckpt() {
    local m="$1"
    local ckpt_name="${m}_ctx${CONTEXT_LINES}${CKPT_NAME_SUFFIX}"
    local src="/opt/dlami/nvme/ckpts/${ckpt_name}"
    local dst="./ckpts/${ckpt_name}"

    if [ ! -d "${src}" ]; then
        echo "[WARN] Checkpoint not found, skipping copy: ${src}"
        return 0
    fi

    echo ""
    echo "[Copy] ${src} → ${dst}"
    mkdir -p "./ckpts"
    [ -d "${dst}" ] && rm -rf "${dst}"
    cp -r "${src}" "${dst}"
    echo "[Copy] Done: ${dst}"
}


# ── 완료된 모드 확인 ─────────────────────────────────────────
is_mode_complete() {
    local m="$1"
    local ckpt_name="${m}_ctx${CONTEXT_LINES}${CKPT_NAME_SUFFIX}"
    local nvme_final="/opt/dlami/nvme/ckpts/${ckpt_name}/final"
    local local_final="./ckpts/${ckpt_name}/final"

    [ -f "${local_final}/adapter_model.safetensors" ] || [ -f "${nvme_final}/adapter_model.safetensors" ]
}

# ── 단일 모드 학습 ────────────────────────────────────────────
run_train() {
    local m="$1"
    local dataset_dir="./data/processed/dataset_${m}_ctx${CONTEXT_LINES}"
    local ckpt_name="${m}_ctx${CONTEXT_LINES}${CKPT_NAME_SUFFIX}"
    local output_dir="/opt/dlami/nvme/ckpts/${ckpt_name}"

    echo "========================================"
    echo " ConGra QLoRA Training"
    echo "========================================"
    echo " Mode:          ${m}"
    echo " Model:         ${MODEL_NAME}"
    echo " Model tag:     ${MODEL_TAG}"
    echo " Dataset:       ${dataset_dir}"
    echo " Output:        ${output_dir}"
    echo " Batch:         ${BATCH_SIZE} x ${GRAD_ACCUM} = $((BATCH_SIZE * GRAD_ACCUM))"
    echo " Eval batch:    ${EVAL_BATCH_SIZE}"
    echo " LR:            ${LR}"
    echo " Epochs:        ${NUM_EPOCHS} (patience=${PATIENCE})"
    echo " Max seq len:   ${MAX_SEQ_LEN}"
    echo " LoRA:          r=${LORA_R}, alpha=${LORA_ALPHA}"
    echo " WandB:         ${USE_WANDB}"
    echo "========================================"
    echo ""

    if [ ! -d "${dataset_dir}" ]; then
        echo "[ERROR] Dataset not found: ${dataset_dir}"
        echo "  Run preprocess.sh --mode ${m} --ctx ${CONTEXT_LINES} first."
        return 1
    fi

    ensure_model_available
    restore_ckpt_if_needed "${m}"

    CMD="${PYTHON_BIN} train.py \
        --dataset_dir ${dataset_dir} \
        --model_name ${MODEL_NAME} \
        --output_dir ${output_dir} \
        --batch_size ${BATCH_SIZE} \
        --eval_batch_size ${EVAL_BATCH_SIZE} \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        --num_epochs ${NUM_EPOCHS} \
        --learning_rate ${LR} \
        --max_seq_length ${MAX_SEQ_LEN} \
        --lora_r ${LORA_R} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_dropout ${LORA_DROPOUT} \
        --eval_steps ${EVAL_STEPS} \
        --early_stopping_patience ${PATIENCE} \
        --seed ${SEED}"

    if [ "${USE_WANDB}" = true ]; then
        CMD="${CMD} --use_wandb --wandb_project congra-merge-conflict"
    fi

    # --resume는 단일 모드에서만 적용
    if [ "${ABLATION}" = false ] && [ -n "${RESUME:-}" ]; then
        CMD="${CMD} --resume_from_checkpoint ${RESUME}"
    fi

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True eval ${CMD}

    copy_ckpt "${m}"
}

# ── 모드 결정 ─────────────────────────────────────────────────
if [ "${ABLATION}" = true ]; then
    MODES=("baseline" "type" "ast" "ast+type")
    echo "========================================"
    echo " [Ablation] 전체 모드 순차 학습"
    echo " Modes: ${MODES[*]}"
    echo " ctx:   ${CONTEXT_LINES}"
    echo "========================================"
else
    MODES=("${MODE}")
fi

# ── 모드별 학습 실행 ──────────────────────────────────────────
for m in "${MODES[@]}"; do
    if [ "${ABLATION}" = true ] && is_mode_complete "${m}"; then
        echo "[Skip] ${m}: final checkpoint already exists"
        continue
    fi

    LOG_FILE="${LOG_DIR}/train_${m}_ctx${CONTEXT_LINES}_${TIMESTAMP}.log"
    echo "[Log] ${LOG_FILE}"
    (
        exec > >(tee -a "${LOG_FILE}") 2>&1
        run_train "${m}"
    )
done

# ── Ablation 완료 요약 ────────────────────────────────────────
if [ "${ABLATION}" = true ]; then
    echo ""
    echo "========================================"
    echo " [Ablation] 완료 요약"
    echo "========================================"
    for m in "${MODES[@]}"; do
        dst="./ckpts/${m}_ctx${CONTEXT_LINES}${CKPT_NAME_SUFFIX}"
        if [ -d "${dst}" ]; then
            echo "  [OK]   ${dst}"
        else
            echo "  [MISS] ${dst}"
        fi
    done
    echo "========================================"
fi
