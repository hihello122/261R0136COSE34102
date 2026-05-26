# 2026 Natural Language Processing (COSE461) Final Project

**Team 22**

고려대학교 정보대학 컴퓨터학과 2021320147 박주영

고려대학교 보건과학대학 바이오의공학부 2021250031 정예준

## STAR-Merge: Structural and Type-Aware Merge Conflict Resolution

STAR-Merge is an NLP-based framework for resolving Java merge conflicts by augmenting token-level neural merge models with structural and type-aware code information.

**This README provides a brief overview of our project; for more detailed explanations, please refer to our project report.**

## 1. Dataset

**Congra Benchmark**

We use the Congra benchmark as the main dataset for training and evaluation.
Congra provides real-world Java merge conflict samples consisting of `base`, `ours`, `theirs`, and the corresponding human-resolved code.

This benchmark is used for both:

- fine-tuning language models
- evaluating merge conflict resolution performance

**HumanEval-X Benchmark**

In addition to Congra, we use HumanEval-X as an external validation benchmark.
HumanEval-X is not used during training and serves as an unseen external benchmark to evaluate whether the trained models can generalize beyond the main training dataset.

We use HumanEval-X only for evaluation.

## 2. Method

**Abstract Syntax Tree**

An Abstract Syntax Tree represents the syntactic structure of source code.
For merge conflict resolution, AST information helps the model understand where a change occurs in the program structure, such as inside a method declaration, return statement, variable declaration, or method invocation.

**Type Information**

Type information provides lightweight semantic context from Java code, such as class names, method signatures, return types, parameter types, variable declarations, and imports.
This information can help the model generate resolutions that are more consistent with the surrounding code, especially when conflicts involve variable usage, method calls, or type-dependent edits.

**QLoRA Fine-tuning**

We fine-tune pretrained code language models using QLoRA.
QLoRA enables efficient fine-tuning by updating a small number of low-rank adapter parameters while keeping the base model quantized.

## 3. Expreiments

**Models**

- DeepSeek-Coder
- Qwen2.5-Coder-1.5B

DeepSeek-Coder is used as the main baseline model, and Qwen2.5-Coder-1.5B is additionally used to compare whether the proposed method is effective across different code language models.

**Train**

| Setting | Description |
| --- | --- |
| Zero-shot | The pretrained model directly generates the resolved code without fine-tuning |
| Baseline | Fine-tuned with only the raw conflict input: `base`, `ours`, and `theirs` |
| Type | Fine-tuned with raw conflict input plus type-aware context |
| AST | Fine-tuned with raw conflict input plus AST-based structural information |
| AST + Type | Fine-tuned with both AST information and type-aware context |

The main purpose of this ablation study is to analyze the contribution of each additional information source.

**Evaluate**

For the `Congra benchmark`, we evaluate the following four input configurations:

- Baseline
- Type
- AST
- AST + Type

For each configuration, we compare:

- Zero-shot performance
- Fine-tuned performance

We use the following metrics:

- BLEU
- CodeBLEU
- Token-F1
- chrF
- Edit Distance

We also evaluate the same ablation settings on `HumanEval-X benchmark` as an external validation benchmark.

The evaluated configurations are:

- base (Zero-shot)
- Baseline
- Type
- AST
- AST + Type

We use the following metrics:

- BLEU
- CodeBLEU
- chrF

## 4. Results

For detailed evaluation results and visualizations, please refer to the `eval_results/` and `figures/` directories.
