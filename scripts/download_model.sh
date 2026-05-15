#!/bin/bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# Coder 모델 체크포인트 다운로드 스크립트
# /opt/dlami/nvme (217GB 여유)에 저장합니다.
# ═══════════════════════════════════════════════════════════════

# ── 기본 설정 ─────────────────────────────────────────────────
MODEL_SIZE="1.3b"
# /dev/root 는 100% 꽉 참 → nvme 마운트 포인트 사용
MODELS_DIR="/opt/dlami/nvme/models"

# ── 인자 파싱 ─────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --6.7b)       MODEL_SIZE="6.7b"; shift ;;
        --1.3b)       MODEL_SIZE="1.3b"; shift ;;
        --qwen-1.5b)  MODEL_SIZE="qwen-1.5b"; shift ;;
        --dir)        MODELS_DIR="$2";   shift 2 ;;
        *) echo "Unknown arg: $1"; echo "Usage: $0 [--1.3b|--6.7b|--qwen-1.5b] [--dir <path>]"; exit 1 ;;
    esac
done

# ── 모델 ID 결정 ──────────────────────────────────────────────
case "$MODEL_SIZE" in
    "1.3b")
        MODEL_ID="deepseek-ai/deepseek-coder-1.3b-base"
        LOCAL_DIR="${MODELS_DIR}/deepseek-coder-1.3b-base"
        REQUIRED_GB=3
        ;;
    "6.7b")
        MODEL_ID="deepseek-ai/deepseek-coder-6.7b-base"
        LOCAL_DIR="${MODELS_DIR}/deepseek-coder-6.7b-base"
        REQUIRED_GB=15
        ;;
    "qwen-1.5b")
        MODEL_ID="Qwen/Qwen2.5-Coder-1.5B"
        LOCAL_DIR="${MODELS_DIR}/qwen2.5-coder-1.5b"
        REQUIRED_GB=4
        ;;
esac

echo "========================================"
echo " Coder Model Download"
echo "========================================"
echo " Model:     ${MODEL_ID}"
echo " Save to:   ${LOCAL_DIR}"
echo "========================================"
echo ""

# ── 디스크 여유 공간 확인 ────────────────────────────────────
mkdir -p "${MODELS_DIR}"
AVAIL_GB=$(df -BG "${MODELS_DIR}" | awk 'NR==2 {gsub("G",""); print $4}')
echo "[INFO] ${MODELS_DIR} 여유 공간: ${AVAIL_GB}GB (필요: ${REQUIRED_GB}GB)"
if (( AVAIL_GB < REQUIRED_GB )); then
    echo "[ERROR] 디스크 공간 부족. 최소 ${REQUIRED_GB}GB 필요, ${AVAIL_GB}GB 가용."
    exit 1
fi

# ── huggingface_hub 설치 확인 ────────────────────────────────
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "[INFO] huggingface_hub 설치 중..."
    pip install -q huggingface_hub
fi

# ── 이미 다운로드된 경우 스킵 ───────────────────────────────
if [ -d "${LOCAL_DIR}" ] && [ -f "${LOCAL_DIR}/config.json" ]; then
    echo "[INFO] 이미 다운로드된 모델이 존재합니다: ${LOCAL_DIR}"
    echo ""
    echo "학습 시 아래 경로를 사용하세요:"
    echo "  bash scripts/train.sh --model ${LOCAL_DIR}"
    exit 0
fi

# ── 다운로드 ─────────────────────────────────────────────────
# HF_HUB_DISABLE_XET=1 : xet(CAS) 전송 방식 비활성화 → segfault 우회
# max_workers=1         : 병렬 다운로드 비활성화 → 안정성 향상
echo "[INFO] 다운로드 시작... (xet 비활성화, 순차 다운로드)"
HF_HUB_DISABLE_XET=1 python3 - <<PYEOF
from huggingface_hub import snapshot_download

model_id = "${MODEL_ID}"
local_dir = "${LOCAL_DIR}"

print(f"Downloading {model_id} → {local_dir}")
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    max_workers=1,
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
)
print("Download complete.")
PYEOF

# ── 완료 메시지 ───────────────────────────────────────────────
echo ""
echo "[OK] 모델 다운로드 완료: ${LOCAL_DIR}"
echo ""
echo "학습 시 아래와 같이 로컬 경로를 지정하세요:"
echo ""
echo "  # train.sh 사용 시"
echo "  bash scripts/train.sh --model ${LOCAL_DIR}"
echo ""
echo "  # 직접 실행 시"
echo "  python3 train.py --model_name ${LOCAL_DIR} ..."
