"""
ConGra Java merge conflict resolution - QLoRA 파인튜닝 스크립트.

SSH GPU 인스턴스(L4 24GB)에서 실행.
전처리된 HuggingFace Dataset을 로드하여 DeepSeek Coder를 QLoRA로 학습.

Usage:
    python train.py \
        --dataset_dir ./data/processed/dataset \
        --model_name deepseek-ai/deepseek-coder-1.3b-base \
        --output_dir ./output \
        --num_epochs 10 \
        --batch_size 4 \
        --max_seq_length 1024

    # 6.7B 스케일업 시
    python train.py \
        --model_name deepseek-ai/deepseek-coder-6.7b-base \
        --batch_size 1 \
        --gradient_accumulation_steps 8
"""

import argparse
import os

import torch
from datasets import load_from_disk
from peft import LoraConfig, prepare_model_for_kbit_training
from dataclasses import dataclass
from typing import Any

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    PreTrainedTokenizerBase,
)
from transformers.trainer_utils import get_last_checkpoint


@dataclass
class DataCollatorForCompletionOnlyLM:
    response_template: list[int]
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 1024

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_ids = [torch.tensor(f["input_ids"][:self.max_length], dtype=torch.long) for f in features]
        max_len = max(len(ids) for ids in batch_ids)
        pad_id = self.tokenizer.pad_token_id

        all_input_ids, all_attn, all_labels = [], [], []
        for ids in batch_ids:
            # find first occurrence of the response template
            start_idx = None
            for j in range(len(ids) - len(self.response_template) + 1):
                if ids[j : j + len(self.response_template)].tolist() == self.response_template:
                    start_idx = j + len(self.response_template)
                    break

            lbl = ids.clone()
            lbl[: start_idx if start_idx is not None else len(lbl)] = -100

            pad = max_len - len(ids)
            all_input_ids.append(torch.cat([ids, ids.new_full((pad,), pad_id)]))
            all_attn.append(torch.cat([torch.ones(len(ids), dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
            all_labels.append(torch.cat([lbl, lbl.new_full((pad,), -100)]))

        return {
            "input_ids": torch.stack(all_input_ids),
            "attention_mask": torch.stack(all_attn),
            "labels": torch.stack(all_labels),
        }
from trl import SFTConfig, SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for merge conflict resolution")

    # Data
    parser.add_argument("--dataset_dir", type=str, default="./data/processed/dataset")

    # Model
    parser.add_argument("--model_name", type=str, default="deepseek-ai/deepseek-coder-1.3b-base")
    parser.add_argument("--output_dir", type=str, default="/opt/dlami/nvme/ckpts")

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Training
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--early_stopping_patience", type=int, default=3)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="congra-merge-conflict")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--no_flash_attn", action="store_true",
                        help="Flash Attention 2 비활성화")

    return parser.parse_args()


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = parse_args()

    # wandb 설정
    if args.use_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        report_to = "wandb"
    else:
        report_to = "none"

    print(f"Model:          {args.model_name}")
    print(f"Dataset:        {args.dataset_dir}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Max seq length: {args.max_seq_length}")
    print()

    # ── 1. Dataset 로드 ───────────────────────────────────────────
    print("Loading dataset...")
    dataset = load_from_disk(args.dataset_dir)
    train_ds = dataset["train"].select_columns(["text"])
    val_ds = dataset["val"].select_columns(["text"])
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val:   {len(val_ds)} samples")

    # ── 2. Tokenizer + 데이터셋 사전 토크나이징 ──────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizing dataset (truncation applied here)...")
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
            add_special_tokens=True,
        )
    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"],
                            num_proc=4, desc="  Train")
    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"],
                        num_proc=4, desc="  Val")
    print(f"  Tokenized. Train max tokens: {max(len(x) for x in train_ds['input_ids'])}")

    # ── 3. Quantization config (4-bit) ───────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ── 4. Model 로드 ────────────────────────────────────────────
    print("Loading model...")
    if args.no_flash_attn:
        attn_impl = "eager"
    else:
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    print(f"  Attention: {attn_impl}")
    print(f"  Parameters: {model.num_parameters():,}")

    # ── 5. LoRA config ───────────────────────────────────────────
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── 6. Training arguments ────────────────────────────────────
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=report_to,
        seed=args.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        max_length=args.max_seq_length,
        packing=False,
        dataset_text_field=None,
    )

    # ── 7. Completion-only loss masking ────────────────────────────
    # "// Resolution\n" 이후 토큰만 loss 계산, 프롬프트 부분은 마스킹 (-100)
    # packing=True일 때는 collator 사용 불가 → packing 끄고 collator 사용
    RESPONSE_TEMPLATE = "\n// Resolution\n"
    response_token_ids = tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_token_ids,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
    )

    # ── 8. SFTTrainer ────────────────────────────────────────────
    print("Initializing trainer...")
    print(f"  Response template: {repr(RESPONSE_TEMPLATE)}")
    print(f"  Response token ids: {response_token_ids}")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        ],
    )

    # ── 9. 체크포인트 감지 및 학습 ───────────────────────────────
    # --resume_from_checkpoint 미지정 시 output_dir 내 최신 체크포인트 자동 탐색
    resume_ckpt = args.resume_from_checkpoint
    if resume_ckpt is None:
        last_ckpt = get_last_checkpoint(args.output_dir)
        if last_ckpt is not None:
            resume_ckpt = last_ckpt
            print(f"[Checkpoint] 기존 체크포인트 발견 → 자동 resume: {resume_ckpt}")
        else:
            print("[Checkpoint] 기존 체크포인트 없음 → 처음부터 학습")
    else:
        print(f"[Checkpoint] 지정된 체크포인트에서 resume: {resume_ckpt}")
    print()

    print("Starting training...")
    print(f"  Epochs: {args.num_epochs} (with early stopping, patience={args.early_stopping_patience})")
    print(f"  Batch: {args.batch_size} x {args.gradient_accumulation_steps} = {args.batch_size * args.gradient_accumulation_steps}")
    print()

    trainer.train(resume_from_checkpoint=resume_ckpt)

    # ── 10. 저장 ──────────────────────────────────────────────────
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    print(f"Saving final model to {final_dir}...")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # eval loss 기록
    metrics = trainer.evaluate()
    print(f"\nFinal eval metrics: {metrics}")

    trainer.save_state()
    print("Training complete!")


if __name__ == "__main__":
    main()
