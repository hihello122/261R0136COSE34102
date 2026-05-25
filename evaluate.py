"""
ConGra Java merge conflict resolution - 평가 스크립트.

학습된 모델을 로드하여 test set에 대해 생성 및 평가 수행.

지표:
  1. Exact Match Rate - resolution 문자열 완전 일치율
  2. BLEU - 토큰 단위 BLEU score
  3. CodeBLEU - 코드 특화 BLEU (AST + data-flow 반영)
  4. Edit Distance (Levenshtein) - 정규화된 편집 거리

Usage:
    python evaluate.py \
        --model_dir ./output/final \
        --dataset_dir ./data/processed/dataset \
        --split test \
        --max_new_tokens 512 \
        --batch_size 4

    # base 모델 (파인튜닝 전) 평가
    python evaluate.py \
        --model_name deepseek-ai/deepseek-coder-1.3b-base \
        --dataset_dir ./data/processed/dataset \
        --split test
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate merge conflict resolution model")

    # Model - 둘 중 하나만 지정
    parser.add_argument("--model_dir", type=str, default=None,
                        help="파인튜닝된 LoRA adapter 디렉토리")
    parser.add_argument("--model_name", type=str, default=None,
                        help="베이스 모델 이름 (파인튜닝 전 평가용)")
    parser.add_argument("--base_model_name", type=str, default=None,
                        help="LoRA adapter 평가 시 adapter_config의 base_model_name_or_path를 덮어쓰기")

    # Data
    parser.add_argument("--dataset_dir", type=str, default="./data/processed/dataset")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="평가할 최대 샘플 수 (디버깅용)")

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0이면 greedy decoding")
    parser.add_argument("--batch_size", type=int, default=1)

    # Output
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--no_flash_attn", action="store_true")

    # Benchmark
    parser.add_argument(
        "--benchmark", type=str, default="congra",
        choices=["congra", "humanevalx"],
        help="평가 벤치마크 (congra: ConGra 병합충돌 해결, humanevalx: HumanEval-X Java)",
    )

    return parser.parse_args()


# deepseek-coder tokenizer.decode() 가 일으키는 두 가지 텍스트 손실을 보정한다.
#   A) 공백 손실: 'import org.X' → 'importorg.X'
#      - ';importX' → ';\nimportX'  (개행 복원)
#      - 줄 시작 'importX' → 'import X'  (공백 복원)
#   B) 개행 손실: '\n' 이 Ċ (U+010A, GPT-2 byte-level 표현) 로 남거나 완전히 사라짐
#      - Ċ → '\n' 으로 치환
_IMPORT_SPLIT_RE = re.compile(r';(?=(?:import|package)[a-zA-Z])')
_IMPORT_SPACE_RE = re.compile(r'^(\s*)(import|package)(?=[a-zA-Z])', re.MULTILINE)


def normalize_code(code: str) -> str:
    """비교를 위한 코드 정규화: 공백/줄바꿈 차이 무시."""
    # B) Ċ (U+010A) → 실제 개행
    code = code.replace('Ċ', '\n')
    # A) import/package 문 공백·개행 복원
    code = _IMPORT_SPLIT_RE.sub(';\n', code)
    code = _IMPORT_SPACE_RE.sub(r'\1\2 ', code)
    lines = code.strip().splitlines()
    lines = [line.rstrip() for line in lines]
    # 빈 줄 제거
    lines = [line for line in lines if line.strip()]
    return "\n".join(lines)


def compute_exact_match(pred: str, gold: str) -> bool:
    return normalize_code(pred) == normalize_code(gold)


def compute_edit_distance(pred: str, gold: str) -> float:
    """정규화된 Levenshtein 편집 거리 (0=동일, 1=완전히 다름)."""
    pred_norm = normalize_code(pred)
    gold_norm = normalize_code(gold)
    if pred_norm == gold_norm:
        return 0.0

    m, n = len(pred_norm), len(gold_norm)
    if m == 0 or n == 0:
        return 1.0

    # 메모리 효율적 DP
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if pred_norm[i - 1] == gold_norm[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev

    return prev[n] / max(m, n)


def compute_codebleu_score(pred: str, gold: str) -> float:
    """CodeBLEU: n-gram + AST + data-flow 기반 코드 유사도."""
    try:
        from codebleu import calc_codebleu
        result = calc_codebleu([gold], [pred], lang="java", weights=(0.25, 0.25, 0.25, 0.25))
        return result["codebleu"]
    except Exception:
        return 0.0


def compute_bleu_score(pred: str, gold: str, tokenizer=None) -> float:
    """토큰 레벨 BLEU-4.

    tokenizer가 주어지면 모델 토크나이저로 인코딩한 token ID 시퀀스로 n-gram을
    비교한다. deepseek-coder 계열 토크나이저는 decode() 시 공백을 복원하지 않으므로
    문자열 split() 대신 token ID 비교가 정확하다.
    tokenizer가 없으면 기존 방식(공백 split)으로 폴백.
    """
    import math
    from collections import Counter

    def ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    if tokenizer is not None:
        pred_tokens = tokenizer.encode(pred.strip(), add_special_tokens=False)
        gold_tokens = tokenizer.encode(gold.strip(), add_special_tokens=False)
    else:
        pred_tokens = normalize_code(pred).split()
        gold_tokens = normalize_code(gold).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    bp = min(1.0, len(pred_tokens) / len(gold_tokens))

    scores = []
    for n in range(1, 5):
        pred_ng = Counter(ngrams(pred_tokens, n))
        gold_ng = Counter(ngrams(gold_tokens, n))
        total = sum(pred_ng.values())
        if total == 0:
            scores.append(0.0)
            continue
        clipped = sum(min(cnt, gold_ng.get(ng, 0)) for ng, cnt in pred_ng.items())
        scores.append(clipped / total)

    if any(s == 0 for s in scores):
        return 0.0

    log_avg = sum(math.log(s) for s in scores) / 4
    return bp * math.exp(log_avg)


def compute_token_f1(pred: str, gold: str, tokenizer=None) -> float:
    """토큰 레벨 F1.

    multiset 교집합 기반 precision/recall → F1.
    tokenizer가 주어지면 token ID 시퀀스로 비교 (공백 손실 무관).
    """
    from collections import Counter

    if tokenizer is not None:
        pred_tokens = tokenizer.encode(pred.strip(), add_special_tokens=False)
        gold_tokens = tokenizer.encode(gold.strip(), add_special_tokens=False)
    else:
        pred_tokens = normalize_code(pred).split()
        gold_tokens = normalize_code(gold).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_cnt = Counter(pred_tokens)
    gold_cnt = Counter(gold_tokens)
    common = sum(min(pred_cnt[t], gold_cnt[t]) for t in pred_cnt)

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_chrf(pred: str, gold: str, max_n: int = 6, beta: float = 1.0) -> float:
    """chrF: 문자 n-gram F-score (order 1~max_n 평균).

    beta=1 → precision/recall 동등 가중치.
    공백 포함 문자 수준 비교이므로 토크나이저 공백 손실에 무관.
    """
    from collections import Counter

    def char_ngrams(text, n):
        return Counter(text[i:i+n] for i in range(len(text) - n + 1))

    pred_str = pred.strip()
    gold_str = gold.strip()

    if not pred_str or not gold_str:
        return 0.0

    f_scores = []
    for n in range(1, max_n + 1):
        pred_ng = char_ngrams(pred_str, n)
        gold_ng = char_ngrams(gold_str, n)
        common = sum(min(pred_ng[g], gold_ng.get(g, 0)) for g in pred_ng)
        total_pred = sum(pred_ng.values())
        total_gold = sum(gold_ng.values())

        if total_pred == 0 or total_gold == 0:
            f_scores.append(0.0)
            continue

        p = common / total_pred
        r = common / total_gold
        if p + r == 0:
            f_scores.append(0.0)
        else:
            f_scores.append((1 + beta ** 2) * p * r / (beta ** 2 * p + r))

    return sum(f_scores) / len(f_scores) if f_scores else 0.0


def load_model_and_tokenizer(args):
    """모델과 토크나이저를 로드."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    if args.no_flash_attn:
        attn_impl = "eager"
    else:
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "eager"

    if args.model_dir:
        # LoRA adapter 로드
        adapter_config_path = os.path.join(args.model_dir, "adapter_config.json")
        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)
        base_model_name = args.base_model_name or adapter_config.get(
            "base_model_name_or_path",
            "deepseek-ai/deepseek-coder-1.3b-base"
        )

        print(f"Loading base model: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            attn_implementation=attn_impl,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        print(f"Loading LoRA adapter: {args.model_dir}")
        model = PeftModel.from_pretrained(base_model, args.model_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    elif args.model_name:
        # 베이스 모델 직접 로드
        print(f"Loading base model: {args.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            attn_implementation=attn_impl,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    else:
        raise ValueError("--model_dir 또는 --model_name 중 하나를 지정하세요.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_resolution(model, tokenizer, prompt: str, args) -> str:
    """하나의 프롬프트에 대해 resolution을 생성."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature == 0.0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = 0.95

    outputs = model.generate(**inputs, **gen_kwargs)
    # 입력 부분 제거하고 생성된 토큰만 디코딩
    input_len = inputs["input_ids"].shape[1]
    generated = outputs[0][input_len:]
    result = tokenizer.decode(generated, skip_special_tokens=True)
    return result


# ── HumanEval-X Java 벤치마크 지원 ──────────────────────────────────────────

def check_java_available() -> bool:
    """javac/java 실행 가능 여부 확인."""
    try:
        r = subprocess.run(["javac", "-version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _compile_and_run_java(files: dict, main_class: str, timeout: int = 10) -> bool:
    """여러 .java 파일을 컴파일하고 main_class를 실행. exit 0이면 True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for fname, content in files.items():
            with open(os.path.join(tmpdir, fname), "w", encoding="utf-8") as f:
                f.write(content)
        try:
            r = subprocess.run(
                ["javac"] + list(files.keys()),
                cwd=tmpdir, capture_output=True, timeout=timeout,
            )
            if r.returncode != 0:
                return False
            r = subprocess.run(
                ["java", "-cp", ".", main_class],
                cwd=tmpdir, capture_output=True, timeout=timeout,
            )
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def _extract_test_class_name(test_code: str, fallback: str) -> str:
    """테스트 코드에서 public class 이름을 추출."""
    m = re.search(r'\bpublic\s+class\s+(\w+)', test_code)
    return m.group(1) if m else fallback


def truncate_at_stop(text: str, stop_sequences: list) -> str:
    """생성 텍스트를 첫 번째 stop sequence 직후에서 자름."""
    for stop in stop_sequences:
        idx = text.find(stop)
        if idx != -1:
            return text[:idx + len(stop)]
    return text


_HUMANEVALX_TRAILING_PATTERNS = [
    re.compile(r"\n\s*/\*\*"),
    re.compile(r"\n\s*(?:public|private|protected)\s+(?:static\s+)?[\w<>\[\], ?]+\s+\w+\s*\("),
    re.compile(r"\n\s*(?:public\s+)?class\s+\w+"),
    re.compile(r"\n\s*(?:import|package)\s+"),
]


def _strip_humanevalx_trailing_prompt(text: str) -> str:
    """HumanEval-X completion 뒤에 붙은 다음 문제 프롬프트/선언을 제거."""
    cut = len(text)
    for pattern in _HUMANEVALX_TRAILING_PATTERNS:
        match = pattern.search(text)
        if match:
            cut = min(cut, match.start())
    return text[:cut].rstrip()


def _brace_delta(code: str) -> int:
    """문자열/주석을 대략 무시하고 Java brace balance를 계산."""
    delta = 0
    i = 0
    state = "code"
    while i < len(code):
        ch = code[i]
        nxt = code[i + 1] if i + 1 < len(code) else ""
        if state == "code":
            if ch == '"':
                state = "double"
            elif ch == "'":
                state = "single"
            elif ch == "/" and nxt == "/":
                state = "line_comment"
                i += 1
            elif ch == "/" and nxt == "*":
                state = "block_comment"
                i += 1
            elif ch == "{":
                delta += 1
            elif ch == "}":
                delta -= 1
        elif state == "double":
            if ch == "\\":
                i += 1
            elif ch == '"':
                state = "code"
        elif state == "single":
            if ch == "\\":
                i += 1
            elif ch == "'":
                state = "code"
        elif state == "line_comment":
            if ch == "\n":
                state = "code"
        elif state == "block_comment":
            if ch == "*" and nxt == "/":
                state = "code"
                i += 1
        i += 1
    return delta


def postprocess_humanevalx_prediction(prompt: str, raw_pred: str) -> str:
    """HumanEval-X Java completion을 평가 가능한 method/class completion으로 정리."""
    # 먼저 정상적인 method+class 닫힘이 생성된 경우를 보존한다.
    pred = truncate_at_stop(raw_pred, ["\n}\n}", "\n    }\n}"])
    pred = _strip_humanevalx_trailing_prompt(pred)

    # 모델이 method만 닫고 다음 프롬프트를 이어 쓰는 경우 class 닫힘을 보정한다.
    missing = _brace_delta(prompt + pred)
    if 0 < missing <= 2:
        pred = pred.rstrip() + "\n" + "\n".join("}" for _ in range(missing))

    return pred


def load_humanevalx_java(max_samples=None):
    """HumanEval-X Java test split 로드 (THUDM/humaneval-x).

    datasets >= 2.14 에서 loading script가 지원되지 않으므로
    parquet/jsonl 직접 로드 → hub 파일 탐색 순으로 fallback.
    """
    from datasets import load_dataset, Dataset

    # ── 1) jsonl 직접 로드 (loading script 우회, 실제 파일 경로 우선) ──
    _JSONL_PATTERNS = [
        "hf://datasets/THUDM/humaneval-x/data/java/data/humaneval.jsonl",   # 실제 경로
        "hf://datasets/THUDM/humaneval-x/data/java/data/test*.jsonl.gz",
        "hf://datasets/THUDM/humaneval-x/data/java/data/test*.jsonl",
        "hf://datasets/THUDM/humaneval-x/data/java/test*.jsonl",
    ]
    for pat in _JSONL_PATTERNS:
        try:
            ds = load_dataset("json", data_files={"test": pat}, split="test")
            if max_samples:
                ds = ds.select(range(min(max_samples, len(ds))))
            return ds
        except Exception:
            pass

    # ── 2) parquet 직접 로드 ─────────────────────────────────────
    _PARQUET_PATTERNS = [
        "hf://datasets/THUDM/humaneval-x/data/java/data/test-*.parquet",
        "hf://datasets/THUDM/humaneval-x/data/java/test-*.parquet",
    ]
    for pat in _PARQUET_PATTERNS:
        try:
            ds = load_dataset("parquet", data_files={"test": pat}, split="test")
            if max_samples:
                ds = ds.select(range(min(max_samples, len(ds))))
            return ds
        except Exception:
            pass

    # ── 3) Hub API로 파일 탐색 후 다운로드 ──────────────────────
    try:
        import gzip as _gzip
        import pandas as _pd
        from huggingface_hub import HfApi, hf_hub_download

        all_files = list(HfApi().list_repo_files("THUDM/humaneval-x", repo_type="dataset"))
        java_files = [
            f for f in all_files
            if re.search(r"java.*test.*\.(parquet|jsonl(\.gz)?)$", f, re.IGNORECASE)
        ]
        if not java_files:
            raise FileNotFoundError("No Java test files found in THUDM/humaneval-x repo")

        local = hf_hub_download("THUDM/humaneval-x", java_files[0], repo_type="dataset")

        if local.endswith(".parquet"):
            ds = Dataset.from_pandas(_pd.read_parquet(local))
        else:
            opener = _gzip.open if local.endswith(".gz") else open
            rows = []
            with opener(local, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            ds = Dataset.from_list(rows)

        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        return ds

    except Exception as e:
        raise RuntimeError(
            f"HumanEval-X Java 로드 실패: {e}\n"
            "datasets >= 2.14 에서 loading script가 지원되지 않습니다. "
            "pip install 'datasets<2.14' 로 다운그레이드하세요."
        ) from e


def run_humanevalx_eval(model, tokenizer, args) -> tuple:
    """HumanEval-X Java 벤치마크 평가."""
    print("\nLoading HumanEval-X Java (THUDM/humaneval-x)...")
    ds = load_humanevalx_java(args.max_samples)
    print(f"  {len(ds)} problems")

    java_ok = check_java_available()
    if not java_ok:
        print("  [Warning] javac/java를 찾을 수 없음 – 실행 테스트 건너뜀, 텍스트 지표만 보고")

    # HumanEval-X Java: 메서드 + 클래스 닫는 괄호에서 중단
    stop_seqs = ["\n}\n}", "\n    }\n}"]
    results = []

    for sample in tqdm(ds, desc="HumanEval-X Java"):
        prompt = sample["prompt"]          # imports + class Solution { + method sig
        gold = sample["canonical_solution"]
        test_code = sample["test"]         # class Main { public static void main... }

        raw_pred = generate_resolution(model, tokenizer, prompt, args)
        pred = postprocess_humanevalx_prediction(prompt, raw_pred)

        passed = False
        if java_ok:
            test_class = _extract_test_class_name(test_code, "Main")
            passed = _compile_and_run_java(
                {"Solution.java": prompt + pred, f"{test_class}.java": test_code},
                main_class=test_class,
            )

        bleu = compute_bleu_score(pred, gold, tokenizer=tokenizer)
        codebleu = compute_codebleu_score(pred, gold)
        chrf = compute_chrf(pred, gold)
        edit_dist = compute_edit_distance(pred, gold)

        results.append({
            "task_id": sample["task_id"],
            "passed": passed,
            "bleu": bleu,
            "codebleu": codebleu,
            "chrf": chrf,
            "edit_distance": edit_dist,
            "prediction": pred,
            "gold": gold,
        })

    total = len(results)
    def avg(k): return sum(r[k] for r in results) / total if total else 0
    overall = {
        "benchmark": "humanevalx-java",
        "total_problems": total,
        "pass@1": avg("passed"),
        "execution_available": java_ok,
        "avg_bleu": avg("bleu"),
        "avg_codebleu": avg("codebleu"),
        "avg_chrf": avg("chrf"),
        "avg_edit_distance": avg("edit_distance"),
    }

    suffix = " (실행 기반)" if java_ok else " (N/A – Java 런타임 없음)"
    print("\n" + "=" * 60)
    print("HumanEval-X Java Results")
    print("=" * 60)
    print(f"  Problems:    {overall['total_problems']}")
    print(f"  pass@1:      {overall['pass@1']:.4f}{suffix}")
    print(f"  BLEU:        {overall['avg_bleu']:.4f}")
    print(f"  CodeBLEU:    {overall['avg_codebleu']:.4f}")
    print(f"  chrF:        {overall['avg_chrf']:.4f}")
    print(f"  Edit Dist:   {overall['avg_edit_distance']:.4f}")

    return overall, results


def _run_congra_eval(model, tokenizer, args, output_dir: Path):
    """ConGra 병합 충돌 해결 벤치마크 평가."""
    # ── 1. 데이터 로드 ───────────────────────────────────────────
    print("Loading dataset...")
    dataset = load_from_disk(args.dataset_dir)
    test_ds = dataset[args.split]
    if args.max_samples:
        test_ds = test_ds.select(range(min(args.max_samples, len(test_ds))))
    print(f"  {args.split}: {len(test_ds)} samples")

    # ── 2. 생성 및 평가 ──────────────────────────────────────────
    print(f"\nGenerating resolutions (max_new_tokens={args.max_new_tokens})...")

    results = []
    project_metrics = defaultdict(lambda: {
        "exact_match": [], "bleu": [], "codebleu": [],
        "token_f1": [], "chrf": [], "edit_dist": [],
    })

    for i, sample in enumerate(tqdm(test_ds, desc="Evaluating")):
        prompt = sample["prompt"]
        gold = sample["resolution"]
        pred = generate_resolution(model, tokenizer, prompt, args)

        em = compute_exact_match(pred, gold)
        bleu = compute_bleu_score(pred, gold, tokenizer=tokenizer)
        codebleu = compute_codebleu_score(pred, gold)
        token_f1 = compute_token_f1(pred, gold, tokenizer=tokenizer)
        chrf = compute_chrf(pred, gold)
        edit_dist = compute_edit_distance(pred, gold)

        result = {
            "idx": i,
            "project": sample["project"],
            "file": sample["file"],
            "exact_match": em,
            "bleu": bleu,
            "codebleu": codebleu,
            "token_f1": token_f1,
            "chrf": chrf,
            "edit_distance": edit_dist,
            "prediction": pred,
            "gold": gold,
        }
        results.append(result)

        project = sample["project"]
        project_metrics[project]["exact_match"].append(int(em))
        project_metrics[project]["bleu"].append(bleu)
        project_metrics[project]["codebleu"].append(codebleu)
        project_metrics[project]["token_f1"].append(token_f1)
        project_metrics[project]["chrf"].append(chrf)
        project_metrics[project]["edit_dist"].append(edit_dist)

    # ── 3. 집계 ──────────────────────────────────────────────────
    total = len(results)
    def avg(key): return sum(r[key] for r in results) / total if total else 0
    overall = {
        "total_samples": total,
        "exact_match_rate": avg("exact_match"),
        "avg_bleu": avg("bleu"),
        "avg_codebleu": avg("codebleu"),
        "avg_token_f1": avg("token_f1"),
        "avg_chrf": avg("chrf"),
        "avg_edit_distance": avg("edit_distance"),
    }

    per_project = {}
    for proj, metrics in project_metrics.items():
        n = len(metrics["exact_match"])
        def pavg(k): return sum(metrics[k]) / n
        per_project[proj] = {
            "count": n,
            "exact_match_rate": pavg("exact_match"),
            "avg_bleu": pavg("bleu"),
            "avg_codebleu": pavg("codebleu"),
            "avg_token_f1": pavg("token_f1"),
            "avg_chrf": pavg("chrf"),
            "avg_edit_distance": pavg("edit_dist"),
        }

    # ── 4. 출력 ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Overall Results")
    print("=" * 60)
    print(f"  Samples:        {overall['total_samples']}")
    print(f"  Exact Match:    {overall['exact_match_rate']:.4f}")
    print(f"  BLEU:           {overall['avg_bleu']:.4f}")
    print(f"  CodeBLEU:       {overall['avg_codebleu']:.4f}")
    print(f"  Token-F1:       {overall['avg_token_f1']:.4f}")
    print(f"  chrF:           {overall['avg_chrf']:.4f}")
    print(f"  Edit Distance:  {overall['avg_edit_distance']:.4f}")

    print("\nPer-project Results:")
    for proj, m in sorted(per_project.items()):
        print(f"  {proj:20s}  n={m['count']:4d}  EM={m['exact_match_rate']:.3f}  "
              f"BLEU={m['avg_bleu']:.3f}  CodeBLEU={m['avg_codebleu']:.3f}  "
              f"F1={m['avg_token_f1']:.3f}  chrF={m['avg_chrf']:.3f}  ED={m['avg_edit_distance']:.3f}")

    # ── 5. 저장 ──────────────────────────────────────────────────
    model_tag = "finetuned" if args.model_dir else "base"
    summary = {
        "model": args.model_dir or args.model_name,
        "split": args.split,
        "overall": overall,
        "per_project": per_project,
    }
    summary_path = output_dir / f"metrics_{model_tag}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved to {summary_path}")

    preds_path = output_dir / f"predictions_{model_tag}.jsonl"
    with open(preds_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Predictions saved to {preds_path}")

    failures = [r for r in results if not r["exact_match"]]
    failures.sort(key=lambda x: x["edit_distance"])
    print(f"\nClosest failures (lowest edit distance, top 5):")
    for r in failures[:5]:
        print(f"  [{r['project']}/{r['file']}] ED={r['edit_distance']:.3f} "
              f"BLEU={r['bleu']:.3f} F1={r['token_f1']:.3f} chrF={r['chrf']:.3f}")
        print(f"    Gold: {normalize_code(r['gold'])[:100]}...")
        print(f"    Pred: {normalize_code(r['prediction'])[:100]}...")
        print()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 모델 로드 ─────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(args)

    # ── 벤치마크 분기 ─────────────────────────────────────────────
    if args.benchmark == "congra":
        _run_congra_eval(model, tokenizer, args, output_dir)

    elif args.benchmark == "humanevalx":
        overall, results = run_humanevalx_eval(model, tokenizer, args)
        model_tag = "finetuned" if args.model_dir else "base"
        summary = {"model": args.model_dir or args.model_name, **overall}
        summary_path = output_dir / f"metrics_humanevalx_{model_tag}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nMetrics saved to {summary_path}")

        preds_path = output_dir / f"predictions_humanevalx_{model_tag}.jsonl"
        with open(preds_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Predictions saved to {preds_path}")


if __name__ == "__main__":
    main()
