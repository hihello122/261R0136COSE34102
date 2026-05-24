#!/bin/bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# ConGra 평가 스크립트
# GPU 인스턴스에서 실행
#
# 단일 모드:              bash scripts/eval.sh --mode baseline
# Qwen 1.5B:              bash scripts/eval.sh --ablation --qwen-1.5b
# 전체 ablation:          bash scripts/eval.sh --ablation
# zero-shot만:            bash scripts/eval.sh --zeroshot
# ablation + zeroshot:    bash scripts/eval.sh --ablation --zeroshot
#
# HumanEval-X Java:       bash scripts/eval.sh --mode baseline --humanevalx
# zero-shot + HumanEval:  bash scripts/eval.sh --zeroshot --humanevalx
# Qwen + HumanEval:       bash scripts/eval.sh --ablation --zeroshot --humanevalx --qwen-1.5b
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
HUMANEVALX=false

BASE_MODEL_NAME="deepseek-ai/deepseek-coder-1.3b-base"
MODEL_TAG="deepseek-coder-1.3b-base"
CKPT_NAME_SUFFIX=""

# ── 인자 파싱 ─────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)        MODE="$2";             shift 2 ;;
        --ctx)         CONTEXT_LINES="$2";    shift 2 ;;
        --split)       SPLIT="$2";            shift 2 ;;
        --max-samples) MAX_SAMPLES="$2";      shift 2 ;;
        --compare)       RUN_BASELINE_COMPARE=true; shift ;;
        --ablation)      ABLATION=true;         shift ;;
        --zeroshot)      ZEROSHOT=true;         shift ;;
        --base-model)    BASE_MODEL_NAME="$2"; MODEL_TAG="$(basename "$2" | tr '[:upper:]' '[:lower:]')"; shift 2 ;;
        --qwen-1.5b)     BASE_MODEL_NAME="/opt/dlami/nvme/models/qwen2.5-coder-1.5b"; MODEL_TAG="qwen2.5-coder-1.5b"; shift ;;
        --humanevalx)    HUMANEVALX=true;       shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ "${MODEL_TAG}" != "deepseek-coder-1.3b-base" ]; then
    CKPT_NAME_SUFFIX="_${MODEL_TAG}"
fi
RESULTS_ROOT="./eval_results/${MODEL_TAG}"

# ── Python 환경 선택 ──────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! ${PYTHON_BIN} -c "import torch" >/dev/null 2>&1; then
    if command -v conda >/dev/null 2>&1 && conda run -n final_project python -c "import torch" >/dev/null 2>&1; then
        PYTHON_BIN="conda run --no-capture-output -n final_project python"
        echo "[INFO] Using conda env: final_project"
    else
        echo "[ERROR] torch not found in ${PYTHON_BIN}. Activate/install the eval env first."
        echo "        Try: conda activate final_project"
        exit 1
    fi
fi

# ── 로그 ──────────────────────────────────────────────────────
LOG_DIR="./logs/${MODEL_TAG}"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── NVMe 모델 복구/다운로드 ──────────────────────────────────
ensure_model_available() {
    local model_path="$1"
    if [[ "${model_path}" != /* ]]; then
        return 0
    fi
    if [ -f "${model_path}/config.json" ]; then
        return 0
    fi

    case "${MODEL_TAG}" in
        qwen2.5-coder-1.5b)
            echo ""
            echo "[Model] Local model missing: ${model_path}"
            echo "[Model] Restoring/downloading Qwen 1.5B to NVMe"
            bash scripts/download_model.sh --qwen-1.5b
            ;;
        deepseek-coder-6.7b-base)
            echo ""
            echo "[Model] Local model missing: ${model_path}"
            echo "[Model] Restoring/downloading DeepSeek Coder 6.7B to NVMe"
            bash scripts/download_model.sh --6.7b
            ;;
        deepseek-coder-1.3b-base)
            echo ""
            echo "[Model] Local model missing: ${model_path}"
            echo "[Model] Restoring/downloading DeepSeek Coder 1.3B to NVMe"
            bash scripts/download_model.sh --1.3b
            ;;
        *)
            echo "[ERROR] Local model path missing and no restore preset is known: ${model_path}"
            exit 1
            ;;
    esac

    if [ ! -f "${model_path}/config.json" ]; then
        echo "[ERROR] Model restore failed: ${model_path}"
        exit 1
    fi
}

_detect_base_model() {
    for m in baseline type ast "ast+type"; do
        local cfg="./ckpts/${m}_ctx${CONTEXT_LINES}${CKPT_NAME_SUFFIX}/final/adapter_config.json"
        if [ -f "${cfg}" ]; then
            python3 -c "import json; print(json.load(open('${cfg}'))['base_model_name_or_path'])"
            return
        fi
    done
    echo "${BASE_MODEL_NAME}"
}

run_zeroshot() {
    local m="$1"
    local DATASET_DIR="./data/processed/dataset_${m}_ctx${CONTEXT_LINES}"
    local EVAL_OUTPUT_DIR="${RESULTS_ROOT}/${m}_ctx${CONTEXT_LINES}"

    if [ ! -d "${DATASET_DIR}" ]; then
        echo "[ERROR] Dataset not found: ${DATASET_DIR}"
        return 1
    fi

    mkdir -p "${EVAL_OUTPUT_DIR}"

    local base_model
    base_model=$(_detect_base_model)
    ensure_model_available "${base_model}"

    echo "========================================"
    echo " Zero-shot Evaluation"
    echo "========================================"
    echo " Mode:        ${m}"
    echo " Model tag:   ${MODEL_TAG}"
    echo " Base model:  ${base_model}"
    echo " Dataset:     ${DATASET_DIR}"
    echo " Split:       ${SPLIT}"
    echo " Output:      ${EVAL_OUTPUT_DIR}"
    echo "========================================"
    echo ""

    CMD="${PYTHON_BIN} evaluate.py \
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

run_eval() {
    local m="$1"
    local DATASET_DIR="./data/processed/dataset_${m}_ctx${CONTEXT_LINES}"
    local MODEL_DIR="./ckpts/${m}_ctx${CONTEXT_LINES}${CKPT_NAME_SUFFIX}/final"
    local EVAL_OUTPUT_DIR="${RESULTS_ROOT}/${m}_ctx${CONTEXT_LINES}"

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
    ensure_model_available "${BASE_MODEL_NAME}"

    echo "========================================"
    echo " ConGra Evaluation"
    echo "========================================"
    echo " Mode:        ${m}"
    echo " Model tag:   ${MODEL_TAG}"
    echo " Model:       ${MODEL_DIR}"
    echo " Base model:  ${BASE_MODEL_NAME}"
    echo " Dataset:     ${DATASET_DIR}"
    echo " Split:       ${SPLIT}"
    echo " Max tokens:  ${MAX_NEW_TOKENS}"
    echo " Output:      ${EVAL_OUTPUT_DIR}"
    echo "========================================"
    echo ""

    CMD="${PYTHON_BIN} evaluate.py \
        --model_dir ${MODEL_DIR} \
        --base_model_name ${BASE_MODEL_NAME} \
        --dataset_dir ${DATASET_DIR} \
        --split ${SPLIT} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --output_dir ${EVAL_OUTPUT_DIR}"

    [ -n "${MAX_SAMPLES}" ] && CMD="${CMD} --max_samples ${MAX_SAMPLES}"
    eval ${CMD}

    if [ "${RUN_BASELINE_COMPARE}" = true ]; then
        echo ""
        run_zeroshot "${m}"

        echo ""
        echo "========================================"
        echo " Comparison Summary (${m})"
        echo "========================================"
        ${PYTHON_BIN} -c "
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
    echo " model: ${MODEL_TAG}"
    echo " out:   ${RESULTS_ROOT}"
    echo "========================================"
else
    MODES=("${MODE}")
fi

for m in "${MODES[@]}"; do
    FINETUNED_RESULT="${RESULTS_ROOT}/${m}_ctx${CONTEXT_LINES}/metrics_finetuned.json"
    BASE_RESULT="${RESULTS_ROOT}/${m}_ctx${CONTEXT_LINES}/metrics_base.json"

    if [ "${ZEROSHOT}" = true ] && [ "${ABLATION}" = false ]; then
        LOG_FILE="${LOG_DIR}/eval_zeroshot_${MODEL_TAG}_${m}_ctx${CONTEXT_LINES}_${SPLIT}_${TIMESTAMP}.log"
        echo "[Log] ${LOG_FILE}"
        (
            exec > >(tee -a "${LOG_FILE}") 2>&1
            run_zeroshot "${m}"
        )
    elif [ "${ZEROSHOT}" = true ]; then
        if [ "${ABLATION}" = true ] && [ -f "${BASE_RESULT}" ]; then
            echo "[Skip] ${m}: zero-shot 결과 이미 존재 → ${BASE_RESULT}"
        else
            LOG_FILE="${LOG_DIR}/eval_zeroshot_${MODEL_TAG}_${m}_ctx${CONTEXT_LINES}_${SPLIT}_${TIMESTAMP}.log"
            echo "[Log] ${LOG_FILE}"
            (
                exec > >(tee -a "${LOG_FILE}") 2>&1
                run_zeroshot "${m}"
            )
        fi
        if [ "${ABLATION}" = true ] && [ -f "${FINETUNED_RESULT}" ]; then
            echo "[Skip] ${m}: 파인튜닝 결과 이미 존재 → ${FINETUNED_RESULT}"
        else
            LOG_FILE="${LOG_DIR}/eval_${MODEL_TAG}_${m}_ctx${CONTEXT_LINES}_${SPLIT}_${TIMESTAMP}.log"
            echo "[Log] ${LOG_FILE}"
            (
                exec > >(tee -a "${LOG_FILE}") 2>&1
                run_eval "${m}"
            )
        fi
    else
        if [ "${ABLATION}" = true ] && [ -f "${FINETUNED_RESULT}" ]; then
            echo "[Skip] ${m}: 파인튜닝 결과 이미 존재 → ${FINETUNED_RESULT}"
        else
            LOG_FILE="${LOG_DIR}/eval_${MODEL_TAG}_${m}_ctx${CONTEXT_LINES}_${SPLIT}_${TIMESTAMP}.log"
            echo "[Log] ${LOG_FILE}"
            (
                exec > >(tee -a "${LOG_FILE}") 2>&1
                run_eval "${m}"
            )
        fi
    fi
done

# ── Code Generation Benchmark 평가 ───────────────────────────
if [ "${HUMANEVALX}" = true ]; then
    CODEGEN_BENCHMARKS=("humanevalx")

    echo ""
    echo "========================================"
    echo " Code Generation Benchmark Evaluation"
    echo " Benchmarks: ${CODEGEN_BENCHMARKS[*]}"
    echo " Model:      ${MODEL_TAG}"
    echo " Out:        ${RESULTS_ROOT}"
    echo "========================================"

    for bm in "${CODEGEN_BENCHMARKS[@]}"; do

        # ── zero-shot (base model) ──────────────────────────────
        if [ "${ZEROSHOT}" = true ]; then
            base_model=$(_detect_base_model)
            ensure_model_available "${base_model}"
            CG_OUT="${RESULTS_ROOT}/codegen_${bm}/base"
            mkdir -p "${CG_OUT}"
            CG_BASE_RESULT="${CG_OUT}/metrics_${bm}_base.json"
            if [ "${ABLATION}" = true ] && [ -f "${CG_BASE_RESULT}" ]; then
                echo "[Skip] codegen ${bm}/base: zero-shot 결과 이미 존재 → ${CG_BASE_RESULT}"
            else
                LOG_FILE="${LOG_DIR}/eval_codegen_zeroshot_${MODEL_TAG}_${bm}_${TIMESTAMP}.log"
                echo "[Log] ${LOG_FILE}"
                (
                    exec > >(tee -a "${LOG_FILE}") 2>&1
                    echo "========================================"
                    echo " Zero-shot | ${bm} | model: ${base_model}"
                    echo " Output: ${CG_OUT}"
                    echo "========================================"
                    CMD="${PYTHON_BIN} evaluate.py \
                        --model_name ${base_model} \
                        --benchmark ${bm} \
                        --max_new_tokens ${MAX_NEW_TOKENS} \
                        --output_dir ${CG_OUT}"
                    [ -n "${MAX_SAMPLES}" ] && CMD="${CMD} --max_samples ${MAX_SAMPLES}"
                    eval ${CMD}
                )
            fi
        fi

        # ── fine-tuned (모드별) ─────────────────────────────────
        if [ "${ZEROSHOT}" = false ] || [ "${ABLATION}" = true ]; then
            for m in "${MODES[@]}"; do
                MODEL_DIR="./ckpts/${m}_ctx${CONTEXT_LINES}${CKPT_NAME_SUFFIX}/final"
                CG_OUT="${RESULTS_ROOT}/codegen_${bm}/${m}_ctx${CONTEXT_LINES}"
                mkdir -p "${CG_OUT}"
                CG_FINETUNED_RESULT="${CG_OUT}/metrics_${bm}_finetuned.json"

                if [ ! -d "${MODEL_DIR}" ]; then
                    echo "[Skip] codegen ${bm}/${m}: 모델 없음 → ${MODEL_DIR}"
                    continue
                fi

                if [ "${ABLATION}" = true ] && [ -f "${CG_FINETUNED_RESULT}" ]; then
                    echo "[Skip] codegen ${bm}/${m}: 파인튜닝 결과 이미 존재 → ${CG_FINETUNED_RESULT}"
                    continue
                fi

                ensure_model_available "${BASE_MODEL_NAME}"
                LOG_FILE="${LOG_DIR}/eval_codegen_${MODEL_TAG}_${bm}_${m}_ctx${CONTEXT_LINES}_${TIMESTAMP}.log"
                echo "[Log] ${LOG_FILE}"
                (
                    exec > >(tee -a "${LOG_FILE}") 2>&1
                    echo "========================================"
                    echo " Fine-tuned | ${bm} | mode: ${m} | ctx: ${CONTEXT_LINES}"
                    echo " Model:  ${MODEL_DIR}"
                    echo " Base:   ${BASE_MODEL_NAME}"
                    echo " Output: ${CG_OUT}"
                    echo "========================================"
                    CMD="${PYTHON_BIN} evaluate.py \
                        --model_dir ${MODEL_DIR} \
                        --base_model_name ${BASE_MODEL_NAME} \
                        --benchmark ${bm} \
                        --max_new_tokens ${MAX_NEW_TOKENS} \
                        --output_dir ${CG_OUT}"
                    [ -n "${MAX_SAMPLES}" ] && CMD="${CMD} --max_samples ${MAX_SAMPLES}"
                    eval ${CMD}
                )
            done
        fi

    done

    # ── Codegen 요약 ─────────────────────────────────────────────
    echo ""
    echo "========================================"
    echo " Code Generation Results 요약"
    echo "========================================"
    ${PYTHON_BIN} -c "
import json, os, glob

benchmarks = ['humanevalx']
root = '${RESULTS_ROOT}'

for bm in benchmarks:
    print(f'\n[{bm}]')
    print(f"  {'Tag':<30} {'pass@1':>8} {'BLEU':>8} {'CodeBLEU':>10} {'chrF':>8}")
    print('  ' + '-' * 68)
    pattern = f'{root}/codegen_{bm}/**/metrics_*.json'
    for path in sorted(glob.glob(pattern, recursive=True)):
        try:
            with open(path) as f:
                data = json.load(f)
            tag = os.path.basename(os.path.dirname(path))
            p1   = data.get('pass@1', 0.0)
            bleu = data.get('avg_bleu', 0.0)
            cb   = data.get('avg_codebleu', 0.0)
            chrf = data.get('avg_chrf', 0.0)
            print(f"  {tag:<30} {p1:>8.4f} {bleu:>8.4f} {cb:>10.4f} {chrf:>8.4f}")
        except Exception:
            pass
"
    echo "========================================"
fi

# ── Ablation 완료 요약 ────────────────────────────────────────
if [ "${ABLATION}" = true ]; then
    echo ""
    echo "========================================"
    echo " [Ablation] 결과 요약"
    echo "========================================"
    ${PYTHON_BIN} -c "
import json, os
modes = ['baseline', 'type', 'ast', 'ast+type']
ctx = ${CONTEXT_LINES}
root = '${RESULTS_ROOT}'
has_zeroshot = '${ZEROSHOT}' == 'true'
keys = ['exact_match_rate', 'avg_bleu', 'avg_codebleu', 'avg_token_f1', 'avg_chrf', 'avg_edit_distance']
labels = ['EM', 'BLEU', 'CodeBLEU', 'Token-F1', 'chrF', 'EditDist']

# 헤더
print(f\"{'Model':<18}\" + ''.join(f'{l:>12}' for l in labels))
print('-' * (18 + 12 * len(keys)))
if has_zeroshot:
    for m in modes:
        path = f'{root}/{m}_ctx{ctx}/metrics_base.json'
        if not os.path.exists(path):
            continue
        with open(path) as f:
            overall = json.load(f)['overall']
        print(f'zero-shot/{m:<8}' + ''.join(f'{overall[k]:>12.4f}' for k in keys))
    print('-' * (18 + 12 * len(keys)))
for m in modes:
    path = f'{root}/{m}_ctx{ctx}/metrics_finetuned.json'
    if not os.path.exists(path):
        print(f'{m:<18}  (결과 없음)')
        continue
    with open(path) as f:
        overall = json.load(f)['overall']
    print(f'{m:<18}' + ''.join(f'{overall[k]:>12.4f}' for k in keys))
"
    echo "========================================"
fi
