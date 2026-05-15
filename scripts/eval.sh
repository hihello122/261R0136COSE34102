#!/bin/bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# ConGra 평가 스크립트
# GPU 인스턴스에서 실행
#
# 단일 모드:           bash scripts/eval.sh --mode baseline
# 전체 ablation:       bash scripts/eval.sh --ablation
# zero-shot만:         bash scripts/eval.sh --zeroshot
# ablation + zeroshot: bash scripts/eval.sh --ablation --zeroshot
# ═══════════════════════════════════════════════════════════════

# ── 실험 설정 ─────────────────────────────────────────────────
MODE="baseline"
CONTEXT_LINES=20
SPLIT="test"
MAX_NEW_TOKENS=512
MAX_SAMPLES=""           # 빈 문자열이면 전체 평가
RUN_BASELINE_COMPARE=false
ABLATION=false
ZEROSHOT=false

# base 모델명: ckpts가 없을 때를 위해 기본값 설정
BASE_MODEL_NAME="deepseek-ai/deepseek-coder-1.3b-base"

# ── 인자 파싱 ─────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)        MODE="$2";             shift 2 ;;
        --ctx)         CONTEXT_LINES="$2";    shift 2 ;;
        --split)       SPLIT="$2";            shift 2 ;;
        --max-samples) MAX_SAMPLES="$2";      shift 2 ;;
        --compare)     RUN_BASELINE_COMPARE=true; shift ;;
        --ablation)    ABLATION=true;         shift ;;
        --zeroshot)    ZEROSHOT=true;         shift ;;
        --base-model)  BASE_MODEL_NAME="$2";  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── 로그 ──────────────────────────────────────────────────────
LOG_DIR="./logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── base 모델명 자동 탐지 ─────────────────────────────────────
# ckpts 중 하나라도 있으면 adapter_config.json에서 읽어옴
_detect_base_model() {
    for m in baseline type ast "ast+type"; do
        local cfg="./ckpts/${m}_ctx${CONTEXT_LINES}/final/adapter_config.json"
        if [ -f "${cfg}" ]; then
            python3 -c "import json; print(json.load(open('${cfg}'))['base_model_name_or_path'])"
            return
        fi
    done
    echo "${BASE_MODEL_NAME}"
}

# ── zero-shot 평가 함수 ───────────────────────────────────────
run_zeroshot() {
    local m="$1"
    local DATASET_DIR="./data/processed/dataset_${m}_ctx${CONTEXT_LINES}"
    local EVAL_OUTPUT_DIR="./eval_results/${m}_ctx${CONTEXT_LINES}"

    if [ ! -d "${DATASET_DIR}" ]; then
        echo "[ERROR] Dataset not found: ${DATASET_DIR}"
        return 1
    fi

    mkdir -p "${EVAL_OUTPUT_DIR}"

    local base_model
    base_model=$(_detect_base_model)

    echo "========================================"
    echo " Zero-shot Evaluation"
    echo "========================================"
    echo " Mode:        ${m}"
    echo " Base model:  ${base_model}"
    echo " Dataset:     ${DATASET_DIR}"
    echo " Split:       ${SPLIT}"
    echo " Output:      ${EVAL_OUTPUT_DIR}"
    echo "========================================"
    echo ""

    CMD="python3 evaluate.py \
        --model_name ${base_model} \
        --dataset_dir ${DATASET_DIR} \
        --split ${SPLIT} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --output_dir ${EVAL_OUTPUT_DIR}"

    [ -n "${MAX_SAMPLES}" ] && CMD="${CMD} --max_samples ${MAX_SAMPLES}"
    eval ${CMD}

    echo ""
    echo "Zero-shot results saved to: ${EVAL_OUTPUT_DIR}/metrics_base.json"
}

# ── 파인튜닝 모델 평가 함수 ───────────────────────────────────
run_eval() {
    local m="$1"
    local DATASET_DIR="./data/processed/dataset_${m}_ctx${CONTEXT_LINES}"
    local MODEL_DIR="./ckpts/${m}_ctx${CONTEXT_LINES}/final"
    local EVAL_OUTPUT_DIR="./eval_results/${m}_ctx${CONTEXT_LINES}"

    if [ ! -d "${DATASET_DIR}" ]; then
        echo "[ERROR] Dataset not found: ${DATASET_DIR}"
        return 1
    fi
    if [ ! -d "${MODEL_DIR}" ]; then
        echo "[ERROR] Model not found: ${MODEL_DIR}"
        echo "  Run train.sh --mode ${m} first."
        return 1
    fi

    mkdir -p "${EVAL_OUTPUT_DIR}"

    echo "========================================"
    echo " ConGra Evaluation"
    echo "========================================"
    echo " Mode:        ${m}"
    echo " Model:       ${MODEL_DIR}"
    echo " Dataset:     ${DATASET_DIR}"
    echo " Split:       ${SPLIT}"
    echo " Max tokens:  ${MAX_NEW_TOKENS}"
    echo " Output:      ${EVAL_OUTPUT_DIR}"
    echo "========================================"
    echo ""

    echo ">>> Evaluating fine-tuned model..."

    CMD="python3 evaluate.py \
        --model_dir ${MODEL_DIR} \
        --dataset_dir ${DATASET_DIR} \
        --split ${SPLIT} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --output_dir ${EVAL_OUTPUT_DIR}"

    [ -n "${MAX_SAMPLES}" ] && CMD="${CMD} --max_samples ${MAX_SAMPLES}"
    eval ${CMD}

    # --compare: 베이스 모델도 함께 평가
    if [ "${RUN_BASELINE_COMPARE}" = true ]; then
        echo ""
        run_zeroshot "${m}"

        echo ""
        echo "========================================"
        echo " Comparison Summary (${m})"
        echo "========================================"
        python3 -c "
import json
with open('${EVAL_OUTPUT_DIR}/metrics_finetuned.json') as f:
    ft = json.load(f)['overall']
with open('${EVAL_OUTPUT_DIR}/metrics_base.json') as f:
    base = json.load(f)['overall']

print(f\"{'Metric':<20} {'Zero-shot':>10} {'Fine-tuned':>10} {'Delta':>10}\")
print('-' * 52)
for key in ['exact_match_rate', 'avg_bleu', 'avg_codebleu', 'avg_token_f1', 'avg_chrf', 'avg_edit_distance']:
    if key not in base or key not in ft:
        continue
    b, f_ = base[key], ft[key]
    d = f_ - b
    sign = '+' if d > 0 else ''
    better = '↑' if (d > 0 and key != 'avg_edit_distance') or (d < 0 and key == 'avg_edit_distance') else '↓' if d != 0 else ''
    print(f'{key:<22} {b:>10.4f} {f_:>10.4f} {sign}{d:>9.4f} {better}')
"
    fi

    echo ""
    echo "Results saved to: ${EVAL_OUTPUT_DIR}/"
}

# ── 모드 결정 ─────────────────────────────────────────────────
if [ "${ABLATION}" = true ]; then
    MODES=("baseline" "type" "ast" "ast+type")
    echo "========================================"
    if [ "${ZEROSHOT}" = true ]; then
        echo " [Ablation] 전체 모드 순차 평가 (zero-shot 포함)"
    else
        echo " [Ablation] 전체 모드 순차 평가"
    fi
    echo " Modes: ${MODES[*]}"
    echo " ctx:   ${CONTEXT_LINES}"
    echo "========================================"
else
    MODES=("${MODE}")
fi

# ── 모드별 실행 ───────────────────────────────────────────────
for m in "${MODES[@]}"; do
    FINETUNED_RESULT="./eval_results/${m}_ctx${CONTEXT_LINES}/metrics_finetuned.json"
    BASE_RESULT="./eval_results/${m}_ctx${CONTEXT_LINES}/metrics_base.json"

    # zero-shot만 요청한 경우
    if [ "${ZEROSHOT}" = true ] && [ "${ABLATION}" = false ]; then
        LOG_FILE="${LOG_DIR}/eval_zeroshot_${m}_ctx${CONTEXT_LINES}_${SPLIT}_${TIMESTAMP}.log"
        echo "[Log] ${LOG_FILE}"
        (
            exec > >(tee -a "${LOG_FILE}") 2>&1
            run_zeroshot "${m}"
        )
    # zero-shot + 파인튜닝 모두
    elif [ "${ZEROSHOT}" = true ]; then
        if [ "${ABLATION}" = true ] && [ -f "${BASE_RESULT}" ]; then
            echo "[Skip] ${m}: zero-shot 결과 이미 존재 → ${BASE_RESULT}"
        else
            LOG_FILE="${LOG_DIR}/eval_zeroshot_${m}_ctx${CONTEXT_LINES}_${SPLIT}_${TIMESTAMP}.log"
            echo "[Log] ${LOG_FILE}"
            (
                exec > >(tee -a "${LOG_FILE}") 2>&1
                run_zeroshot "${m}"
            )
        fi
        if [ "${ABLATION}" = true ] && [ -f "${FINETUNED_RESULT}" ]; then
            echo "[Skip] ${m}: 파인튜닝 결과 이미 존재 → ${FINETUNED_RESULT}"
        else
            LOG_FILE="${LOG_DIR}/eval_${m}_ctx${CONTEXT_LINES}_${SPLIT}_${TIMESTAMP}.log"
            echo "[Log] ${LOG_FILE}"
            (
                exec > >(tee -a "${LOG_FILE}") 2>&1
                run_eval "${m}"
            )
        fi
    # 파인튜닝만
    else
        if [ "${ABLATION}" = true ] && [ -f "${FINETUNED_RESULT}" ]; then
            echo "[Skip] ${m}: 파인튜닝 결과 이미 존재 → ${FINETUNED_RESULT}"
        else
            LOG_FILE="${LOG_DIR}/eval_${m}_ctx${CONTEXT_LINES}_${SPLIT}_${TIMESTAMP}.log"
            echo "[Log] ${LOG_FILE}"
            (
                exec > >(tee -a "${LOG_FILE}") 2>&1
                run_eval "${m}"
            )
        fi
    fi
done

# ── Ablation 완료 요약 ────────────────────────────────────────
if [ "${ABLATION}" = true ]; then
    echo ""
    echo "========================================"
    echo " [Ablation] 결과 요약"
    echo "========================================"
    python3 -c "
import json, os

modes = ['baseline', 'type', 'ast', 'ast+type']
ctx = ${CONTEXT_LINES}
keys = ['exact_match_rate', 'avg_bleu', 'avg_codebleu', 'avg_token_f1', 'avg_chrf', 'avg_edit_distance']
labels = ['EM', 'BLEU', 'CodeBLEU', 'Token-F1', 'chrF', 'EditDist']
has_zeroshot = ${ZEROSHOT}

# 헤더
print(f\"{'Model':<18}\" + ''.join(f'{l:>12}' for l in labels))
print('-' * (18 + 12 * len(keys)))

# zero-shot 행 (모드별 프롬프트 포맷이 다르므로 각각 출력)
if has_zeroshot:
    for m in modes:
        path = f'./eval_results/{m}_ctx{ctx}/metrics_base.json'
        if not os.path.exists(path):
            continue
        with open(path) as f:
            overall = json.load(f)['overall']
        row = f'zero-shot/{m:<8}' + ''.join(f'{overall[k]:>12.4f}' for k in keys)
        print(row)
    print('-' * (18 + 12 * len(keys)))

# 파인튜닝 모델 행
for m in modes:
    path = f'./eval_results/{m}_ctx{ctx}/metrics_finetuned.json'
    if not os.path.exists(path):
        print(f'{m:<18}  (결과 없음)')
        continue
    with open(path) as f:
        overall = json.load(f)['overall']
    row = f'{m:<18}' + ''.join(f'{overall[k]:>12.4f}' for k in keys)
    print(row)
"
    echo "========================================"
fi
