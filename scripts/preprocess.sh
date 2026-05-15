#!/bin/bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# ConGra 전처리 스크립트 (체크포인트 / 청크 지원)
#
# 기본 동작
#   - 프로젝트 단위로 extract → 캐시 (data/processed/.cache/<mode>_ctx<N>/<project>.jsonl)
#   - 모든 프로젝트 추출 끝나면 finalize 로 Dataset 빌드
#   - 중간에 끊겨도 다시 실행하면 끝난 프로젝트는 자동으로 skip
#
# 옵션
#   --mode {baseline|ast|type|ast+type|all}
#   --ctx <int>
#   --skip-stats
#   --projects "p1 p2"     특정 프로젝트만 (extract 단계 한정)
#   --phase {extract|finalize|all}
#                          extract: 캐시 생성만
#                          finalize: 캐시 → Dataset
#                          all: 둘 다 (기본값)
#   --force                기존 캐시 무시하고 재추출
#   --per-project          extract 시 프로젝트마다 python 프로세스를 새로 띄움
#                          (메모리/오류 격리, eclipse 같이 무거운 케이스 권장)
#
# 사용 예
#   # 풀 파이프라인 (기본 — 끊기면 재실행으로 resume)
#   bash scripts/preprocess.sh --mode ast+type
#
#   # eclipse 만 따로 (다른 셀에서 병렬 실행 가능)
#   bash scripts/preprocess.sh --mode ast+type --phase extract --projects "eclipse"
#
#   # 캐시가 다 차면 마지막에 한 번만 finalize
#   bash scripts/preprocess.sh --mode ast+type --phase finalize
# ═══════════════════════════════════════════════════════════════

# ── 설정 ──────────────────────────────────────────────────────
DATA_DIR="./data/raw_datasets/Java"
OUTPUT_DIR="./data/processed"
CONTEXT_LINES=20
MODE="baseline"          # baseline | ast | type | ast+type | all
MAX_SEQ_LEN=2048
TOKENIZER="deepseek-ai/deepseek-coder-1.3b-base"
GUMTREE_BIN="./gumtree-3.0.0/bin/gumtree"
SKIP_STATS=false
PROJECTS=""              # 공백 구분, 미지정시 전체
PHASE="all"              # extract | finalize | all
FORCE=false
PER_PROJECT=false

# ── 인자 파싱 ─────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)        MODE="$2";          shift 2 ;;
        --ctx)         CONTEXT_LINES="$2"; shift 2 ;;
        --skip-stats)  SKIP_STATS=true;    shift   ;;
        --projects)    PROJECTS="$2";      shift 2 ;;
        --phase)       PHASE="$2";         shift 2 ;;
        --force)       FORCE=true;         shift   ;;
        --per-project) PER_PROJECT=true;   shift   ;;
        -h|--help)
            sed -n '4,33p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

case "${PHASE}" in
    extract|finalize|all) ;;
    *) echo "[ERROR] --phase must be one of: extract|finalize|all"; exit 1 ;;
esac

# ── 모드 목록 결정 ────────────────────────────────────────────
if [ "${MODE}" = "all" ]; then
    MODES=("baseline" "type")
    if command -v "${GUMTREE_BIN}" &> /dev/null; then
        MODES+=("ast" "ast+type")
    else
        echo "[WARN] GumTree not found. Skipping ast / ast+type modes."
    fi
else
    MODES=("${MODE}")
fi

# ── 헬퍼 ──────────────────────────────────────────────────────
csv_join() {
    # 공백 구분 → 콤마 구분 (preprocess.py --projects 형식)
    local IFS=','
    echo "$*"
}

build_common_args() {
    local m="$1"
    local args="--data_dir ${DATA_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --context_lines ${CONTEXT_LINES} \
        --mode ${m} \
        --max_seq_len ${MAX_SEQ_LEN} \
        --tokenizer ${TOKENIZER} \
        --gumtree_bin ${GUMTREE_BIN}"
    if [ "${SKIP_STATS}" = true ]; then
        args="${args} --skip_stats"
    fi
    if [ "${FORCE}" = true ]; then
        args="${args} --force"
    fi
    echo "${args}"
}

list_projects_to_extract() {
    # PROJECTS 가 비어있으면 DATA_DIR 의 전체 프로젝트 목록 반환
    if [ -n "${PROJECTS}" ]; then
        echo "${PROJECTS}"
    else
        ls -1 "${DATA_DIR}" | while read -r p; do
            [ -d "${DATA_DIR}/${p}" ] && echo -n "${p} "
        done
        echo
    fi
}

run_extract() {
    local m="$1"
    local cache_dir="${OUTPUT_DIR}/.cache/${m}_ctx${CONTEXT_LINES}"
    local common
    common="$(build_common_args "${m}")"

    echo ""
    echo "----------------------------------------"
    echo " [extract] mode=${m}  cache=${cache_dir}"
    if [ -n "${PROJECTS}" ]; then
        echo " projects: ${PROJECTS}"
    fi
    echo "----------------------------------------"

    if [ "${PER_PROJECT}" = true ]; then
        # 프로젝트별로 python 프로세스를 새로 띄움 → 격리
        local plist
        plist="$(list_projects_to_extract)"
        for p in ${plist}; do
            # 이미 캐시 된 프로젝트는 python 띄우기 전에 스킵
            if [ "${FORCE}" != true ] && [ -f "${cache_dir}/${p}.jsonl" ]; then
                echo ">>> [SKIP] ${m} :: ${p} (cached: ${cache_dir}/${p}.jsonl)"
                continue
            fi
            echo ""
            echo ">>> [extract] ${m} :: ${p}"
            # shellcheck disable=SC2086
            python3 preprocess.py ${common} --phase extract --projects "${p}"
        done
    else
        # 처리 대상 프로젝트 목록을 만든 뒤, 캐시 된 것은 사전에 걸러낸다
        local plist
        plist="$(list_projects_to_extract)"
        local todo=()
        for p in ${plist}; do
            if [ "${FORCE}" != true ] && [ -f "${cache_dir}/${p}.jsonl" ]; then
                echo ">>> [SKIP] ${m} :: ${p} (cached: ${cache_dir}/${p}.jsonl)"
                continue
            fi
            todo+=("${p}")
        done

        if [ "${#todo[@]}" -eq 0 ]; then
            echo ">>> [extract] ${m}: nothing to do (all projects cached)"
            return
        fi

        local proj_arg="--projects $(csv_join ${todo[@]})"
        # shellcheck disable=SC2086
        python3 preprocess.py ${common} --phase extract ${proj_arg}
    fi
}

run_finalize() {
    local m="$1"
    local common
    common="$(build_common_args "${m}")"

    echo ""
    echo "----------------------------------------"
    echo " [finalize] mode=${m}  → ${OUTPUT_DIR}/dataset_${m}_ctx${CONTEXT_LINES}"
    echo "----------------------------------------"

    # shellcheck disable=SC2086
    python3 preprocess.py ${common} --phase finalize
}

run_one_mode() {
    local m="$1"
    echo ""
    echo "========================================"
    echo " ConGra Preprocess: ${m} (phase=${PHASE})"
    echo "========================================"
    echo " Context lines: ${CONTEXT_LINES}"
    echo " Output:        ${OUTPUT_DIR}/dataset_${m}_ctx${CONTEXT_LINES}"
    echo " Cache:         ${OUTPUT_DIR}/.cache/${m}_ctx${CONTEXT_LINES}"
    echo "========================================"

    case "${PHASE}" in
        extract)
            run_extract "${m}"
            ;;
        finalize)
            run_finalize "${m}"
            ;;
        all)
            run_extract "${m}"
            run_finalize "${m}"
            ;;
    esac
}

# ── 실행 ──────────────────────────────────────────────────────
for m in "${MODES[@]}"; do
    # mode=all 일 때 이미 finalize 된 dataset 이 있으면 다음 모드로 스킵
    # (--force 는 이 스킵을 무시)
    if [ "${MODE}" = "all" ] && [ "${FORCE}" != true ]; then
        dataset_dir="${OUTPUT_DIR}/dataset_${m}_ctx${CONTEXT_LINES}"
        if [ -f "${dataset_dir}/dataset_dict.json" ]; then
            echo ""
            echo "========================================"
            echo " [SKIP] mode=${m}: dataset already exists"
            echo "        ${dataset_dir}"
            echo "        (use --force to re-run)"
            echo "========================================"
            continue
        fi
    fi
    run_one_mode "${m}"
done

# ── 결과 요약 ─────────────────────────────────────────────────
echo ""
echo "========================================"
echo " Summary"
echo "========================================"
for m in "${MODES[@]}"; do
    cache_dir="${OUTPUT_DIR}/.cache/${m}_ctx${CONTEXT_LINES}"
    if [ -d "${cache_dir}" ]; then
        n=$(ls "${cache_dir}" 2>/dev/null | grep -c '\.jsonl$' || true)
        echo "  cache  ${cache_dir}: ${n} project(s)"
    fi
    dir="${OUTPUT_DIR}/dataset_${m}_ctx${CONTEXT_LINES}"
    if [ -d "${dir}" ]; then
        echo "  [OK]   ${dir}"
    else
        echo "  [MISS] ${dir} (run with --phase finalize)"
    fi
done
echo ""
