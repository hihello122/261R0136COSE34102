#!/bin/bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# ConGra QLoRA 학습 스크립트
# GPU 인스턴스(L4 24GB)에서 실행
#
# 단일 모드:   bash scripts/train.sh --mode baseline
# 전체 ablation: bash scripts/train.sh --ablation
# ═══════════════════════════════════════════════════════════════

# ── 실험 설정 ─────────────────────────────────────────────────
MODE="baseline"
CONTEXT_LINES=20

MODEL_NAME="deepseek-ai/deepseek-coder-1.3b-base"

BATCH_SIZE=2
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
        --model)      MODEL_NAME="$2";    shift 2 ;;
        --batch)      BATCH_SIZE="$2";    shift 2 ;;
        --lr)         LR="$2";            shift 2 ;;
        --wandb)      USE_WANDB=true;     shift   ;;
        --resume)     RESUME="$2";        shift 2 ;;
        --ablation)   ABLATION=true;      shift   ;;
        # 6.7B 프리셋
        --6.7b)       MODEL_NAME="deepseek-ai/deepseek-coder-6.7b-base"; BATCH_SIZE=1; GRAD_ACCUM=8; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── 로그 설정 ─────────────────────────────────────────────────
LOG_DIR="./logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── 체크포인트 복사 ───────────────────────────────────────────
copy_ckpt() {
    local m="$1"
    local src="/opt/dlami/nvme/ckpts/${m}_ctx${CONTEXT_LINES}"
    local dst="./ckpts/${m}_ctx${CONTEXT_LINES}"

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

# ── 단일 모드 학습 ────────────────────────────────────────────
run_train() {
    local m="$1"
    local dataset_dir="./data/processed/dataset_${m}_ctx${CONTEXT_LINES}"
    local output_dir="/opt/dlami/nvme/ckpts/${m}_ctx${CONTEXT_LINES}"

    echo "========================================"
    echo " ConGra QLoRA Training"
    echo "========================================"
    echo " Mode:          ${m}"
    echo " Model:         ${MODEL_NAME}"
    echo " Dataset:       ${dataset_dir}"
    echo " Output:        ${output_dir}"
    echo " Batch:         ${BATCH_SIZE} x ${GRAD_ACCUM} = $((BATCH_SIZE * GRAD_ACCUM))"
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

    CMD="python3 train.py \
        --dataset_dir ${dataset_dir} \
        --model_name ${MODEL_NAME} \
        --output_dir ${output_dir} \
        --batch_size ${BATCH_SIZE} \
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
        dst="./ckpts/${m}_ctx${CONTEXT_LINES}"
        if [ -d "${dst}" ]; then
            echo "  [OK]   ${dst}"
        else
            echo "  [MISS] ${dst}"
        fi
    done
    echo "========================================"
fi
