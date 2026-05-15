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
        base_model_name = adapter_config.get(
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


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 데이터 로드 ───────────────────────────────────────────
    print("Loading dataset...")
    dataset = load_from_disk(args.dataset_dir)
    test_ds = dataset[args.split]
    if args.max_samples:
        test_ds = test_ds.select(range(min(args.max_samples, len(test_ds))))
    print(f"  {args.split}: {len(test_ds)} samples")

    # ── 2. 모델 로드 ─────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(args)

    # ── 3. 생성 및 평가 ──────────────────────────────────────────
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

    # ── 4. 집계 ──────────────────────────────────────────────────
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

    # ── 5. 출력 ──────────────────────────────────────────────────
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

    # ── 6. 저장 ──────────────────────────────────────────────────
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

    # 전체 prediction 저장
    preds_path = output_dir / f"predictions_{model_tag}.jsonl"
    with open(preds_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Predictions saved to {preds_path}")

    # 오답 분석용 - EM 실패 케이스 상위 N개
    failures = [r for r in results if not r["exact_match"]]
    failures.sort(key=lambda x: x["edit_distance"])
    print(f"\nClosest failures (lowest edit distance, top 5):")
    for r in failures[:5]:
        print(f"  [{r['project']}/{r['file']}] ED={r['edit_distance']:.3f} "
              f"BLEU={r['bleu']:.3f} F1={r['token_f1']:.3f} chrF={r['chrf']:.3f}")
        print(f"    Gold: {normalize_code(r['gold'])[:100]}...")
        print(f"    Pred: {normalize_code(r['prediction'])[:100]}...")
        print()


if __name__ == "__main__":
    main()
