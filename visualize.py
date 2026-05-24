"""
eval_results/ 디렉토리의 JSON 파일을 읽어 ablation 및 코드젠 결과를 시각화.

Usage:
    python visualize.py --model-tag qwen2.5-coder-1.5b --ctx 20
    python visualize.py --results_dir ./eval_results/qwen2.5-coder-1.5b --output_dir ./figures/qwen2.5-coder-1.5b
    python visualize.py --output_dir ./figures --no_codegen
    python visualize.py --model-tag qwen2.5-coder-1.5b --no_codegen
"""

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── 설정 ──────────────────────────────────────────────────────
METRICS = ["exact_match_rate", "avg_bleu", "avg_codebleu", "avg_token_f1", "avg_chrf", "avg_edit_distance"]
METRIC_LABELS = ["Exact Match", "BLEU", "CodeBLEU", "Token-F1", "chrF", "Edit Dist↓"]
MODE_ORDER = ["baseline", "type", "ast", "ast+type"]
PALETTE_FT = "#4C72B0"   # fine-tuned
PALETTE_ZS = "#DD8452"   # zero-shot

CODEGEN_BENCHMARKS = ["humanevalx", "multiple_java"]
CODEGEN_BM_LABELS  = {"humanevalx": "HumanEval-X Java", "multiple_java": "MultiPL-E Java"}
# humanevalx: pass@1 + text metrics / multiple_java: pass@1만
CODEGEN_BM_METRICS = {
    "humanevalx":    ["pass@1", "avg_bleu", "avg_codebleu", "avg_chrf"],
    "multiple_java": ["pass@1"],
}
CODEGEN_BM_METRIC_LABELS = {
    "humanevalx":    ["pass@1", "BLEU", "CodeBLEU", "chrF"],
    "multiple_java": ["pass@1"],
}
_CONGRA_CODEGEN_TAG_ORDER = ["base"] + [
    f"{m}_ctx{c}" for c in [20, 10, 5] for m in MODE_ORDER
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


# ── 데이터 로드 ───────────────────────────────────────────────

def load_results(results_dir: Path, ctx: int):
    """eval_results/ 에서 모든 모드의 결과를 로드."""
    data = {}  # {mode: {"finetuned": {...}, "base": {...}}}
    pattern = re.compile(rf"^(.+)_ctx{ctx}$")

    for entry in sorted(results_dir.iterdir()):
        if not entry.is_dir():
            continue
        m = pattern.match(entry.name)
        if not m:
            continue
        mode = m.group(1)
        data[mode] = {}
        for tag in ("finetuned", "base"):
            path = entry / f"metrics_{tag}.json"
            if path.exists():
                with open(path) as f:
                    data[mode][tag] = json.load(f)

    return data


def load_codegen_results(results_dir: Path) -> dict:
    """codegen_* 디렉토리에서 HumanEval-X / MultiPL-E 결과를 로드.

    Returns:
        {benchmark: {tag: metrics_dict}}
        tag 예: "base", "baseline_ctx20", "type_ctx20", ...
    """
    codegen = {}
    for bm in CODEGEN_BENCHMARKS:
        bm_dir = results_dir / f"codegen_{bm}"
        if not bm_dir.exists():
            continue
        codegen[bm] = {}
        for tag_dir in sorted(bm_dir.iterdir()):
            if not tag_dir.is_dir():
                continue
            tag = tag_dir.name
            for suffix in ("finetuned", "base"):
                path = tag_dir / f"metrics_{bm}_{suffix}.json"
                if path.exists():
                    with open(path) as f:
                        codegen[bm][tag] = json.load(f)
                    break
    return codegen


def ordered_codegen_tags(tags: list) -> list:
    """codegen tag를 base → ConGra 모드 순으로 정렬."""
    ordered = [t for t in _CONGRA_CODEGEN_TAG_ORDER if t in tags]
    rest    = [t for t in tags if t not in ordered]
    return ordered + sorted(rest)


def detect_ctx(results_dir: Path):
    """결과 디렉토리 이름에서 ctx 값을 자동 탐지."""
    ctxs = set()
    for entry in results_dir.iterdir():
        m = re.search(r"_ctx(\d+)$", entry.name)
        if m:
            ctxs.add(int(m.group(1)))
    return sorted(ctxs)


# ── 유틸 ──────────────────────────────────────────────────────

def ordered_modes(data: dict):
    return [m for m in MODE_ORDER if m in data] + [m for m in sorted(data) if m not in MODE_ORDER]


def get_overall(data, mode, tag):
    try:
        return data[mode][tag]["overall"]
    except KeyError:
        return None


def metric_val(overall, key):
    if overall is None:
        return float("nan")
    return overall.get(key, float("nan"))


# ── 그래프 1: Ablation 전체 지표 비교 (grouped bar) ──────────

def plot_ablation_overview(data, modes, output_dir, ctx, title_suffix=""):
    n_metrics = len(METRICS)
    n_modes = len(modes)
    has_base = any("base" in data[m] for m in modes)

    fig, axes = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics, 4.5))
    fig.suptitle(f"Ablation Overview  (ctx={ctx}){title_suffix}", fontsize=14, fontweight="bold", y=1.02)

    x = np.arange(n_modes)
    width = 0.35 if has_base else 0.55

    for ax, key, label in zip(axes, METRICS, METRIC_LABELS):
        ft_vals = [metric_val(get_overall(data, m, "finetuned"), key) for m in modes]
        bars_ft = ax.bar(x, ft_vals, width, label="Fine-tuned", color=PALETTE_FT, zorder=3)

        if has_base:
            zs_vals = [metric_val(get_overall(data, m, "base"), key) for m in modes]
            ax.bar(x + width, zs_vals, width, label="Zero-shot", color=PALETTE_ZS, zorder=3)

        # 값 레이블
        for bar in bars_ft:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.3f}",
                        ha="center", va="bottom", fontsize=6.5, rotation=45)

        ax.set_title(label, fontsize=10, pad=4)
        ax.set_xticks(x + (width / 2 if has_base else 0))
        ax.set_xticklabels([m.replace("+", "+\n") for m in modes], fontsize=8)
        ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.25))
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

    if has_base:
        handles = [
            mpatches.Patch(color=PALETTE_FT, label="Fine-tuned"),
            mpatches.Patch(color=PALETTE_ZS, label="Zero-shot"),
        ]
        fig.legend(handles=handles, loc="upper right", fontsize=9, frameon=False)

    fig.tight_layout()
    path = output_dir / f"ablation_overview_ctx{ctx}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 그래프 2: Fine-tuned vs Zero-shot 델타 heatmap ──────────

def plot_delta_heatmap(data, modes, output_dir, ctx, title_suffix=""):
    has_base = any("base" in data[m] for m in modes)
    if not has_base:
        return

    delta_matrix = []
    valid_modes = []
    for m in modes:
        ft = get_overall(data, m, "finetuned")
        zs = get_overall(data, m, "base")
        if ft is None or zs is None:
            continue
        row = []
        for key in METRICS:
            delta = metric_val(ft, key) - metric_val(zs, key)
            # Edit Distance는 낮을수록 좋으므로 부호 반전
            if key == "avg_edit_distance":
                delta = -delta
            row.append(delta)
        delta_matrix.append(row)
        valid_modes.append(m)

    if not delta_matrix:
        return

    matrix = np.array(delta_matrix)
    vmax = np.nanmax(np.abs(matrix))

    fig, ax = plt.subplots(figsize=(len(METRICS) * 1.6, len(valid_modes) * 0.9 + 1.2))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels(METRIC_LABELS, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(valid_modes)))
    ax.set_yticklabels(valid_modes, fontsize=10)
    ax.set_title(f"Fine-tuned Δ over Zero-shot  (ctx={ctx}){title_suffix}\n(green = fine-tuned better)", fontsize=11, pad=10)

    for i in range(len(valid_modes)):
        for j in range(len(METRICS)):
            val = matrix[i, j]
            sign = "+" if val > 0 else ""
            ax.text(j, i, f"{sign}{val:.3f}", ha="center", va="center", fontsize=8,
                    color="black" if abs(val) < vmax * 0.6 else "white")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Δ (fine-tuned − zero-shot)")
    fig.tight_layout()
    path = output_dir / f"delta_heatmap_ctx{ctx}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 그래프 3: Per-project 지표 비교 ─────────────────────────

def plot_per_project(data, modes, output_dir, ctx, title_suffix=""):
    # 전체 프로젝트 수집
    projects = set()
    for m in modes:
        for tag in ("finetuned", "base"):
            try:
                projects.update(data[m][tag]["per_project"].keys())
            except KeyError:
                pass
    projects = sorted(projects)
    if not projects:
        return

    show_metrics = ["avg_bleu", "avg_codebleu", "avg_token_f1", "avg_chrf"]
    show_labels = ["BLEU", "CodeBLEU", "Token-F1", "chrF"]

    fig, axes = plt.subplots(len(show_metrics), 1, figsize=(max(8, len(modes) * 1.8), 3.5 * len(show_metrics)))
    fig.suptitle(f"Per-project Metrics by Mode  (ctx={ctx}){title_suffix}", fontsize=13, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 0.8, len(projects)))
    x = np.arange(len(modes))
    proj_width = 0.7 / len(projects)

    for ax, key, label in zip(axes, show_metrics, show_labels):
        for pi, (proj, color) in enumerate(zip(projects, colors)):
            vals = []
            for m in modes:
                try:
                    vals.append(data[m]["finetuned"]["per_project"][proj][key])
                except KeyError:
                    vals.append(float("nan"))
            offset = (pi - len(projects) / 2 + 0.5) * proj_width
            ax.bar(x + offset, vals, proj_width, label=proj, color=color, zorder=3)

        ax.set_title(f"{label} (fine-tuned)", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(modes, fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    path = output_dir / f"per_project_ctx{ctx}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 그래프 4: Radar chart (ablation 모드별 종합 프로파일) ────

def plot_radar(data, modes, output_dir, ctx, title_suffix=""):
    # edit_distance는 반전(낮을수록 좋으므로 1 - val)
    radar_keys = ["exact_match_rate", "avg_bleu", "avg_codebleu", "avg_token_f1", "avg_chrf"]
    radar_labels = ["Exact Match", "BLEU", "CodeBLEU", "Token-F1", "chrF"]
    N = len(radar_keys)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(modes)))

    for m, color in zip(modes, colors):
        overall = get_overall(data, m, "finetuned")
        if overall is None:
            continue
        vals = [metric_val(overall, k) for k in radar_keys]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=1.8, color=color, label=m)
        ax.fill(angles, vals, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(f"Ablation Radar  (ctx={ctx}, fine-tuned){title_suffix}", fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9, frameon=False)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    path = output_dir / f"radar_ctx{ctx}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 그래프 5: Codegen 벤치마크 bar chart ─────────────────────

def plot_codegen_bars(codegen: dict, output_dir: Path):
    """HumanEval-X / MultiPL-E Java 벤치마크별 bar chart."""
    available = [bm for bm in CODEGEN_BENCHMARKS if bm in codegen]
    if not available:
        return

    for bm in available:
        tags = ordered_codegen_tags(list(codegen[bm].keys()))
        metrics = CODEGEN_BM_METRICS[bm]
        m_labels = CODEGEN_BM_METRIC_LABELS[bm]
        n_m = len(metrics)

        fig, axes = plt.subplots(1, n_m, figsize=(3.8 * n_m, 4.5))
        fig.suptitle(CODEGEN_BM_LABELS[bm], fontsize=13, fontweight="bold", y=1.02)
        if n_m == 1:
            axes = [axes]

        x = np.arange(len(tags))
        bar_colors = [PALETTE_ZS if t == "base" else PALETTE_FT for t in tags]
        tick_labels = [t.replace("_ctx", "\nctx") for t in tags]

        for ax, key, label in zip(axes, metrics, m_labels):
            vals = [codegen[bm][t].get(key, float("nan")) for t in tags]
            bars = ax.bar(x, vals, 0.6, color=bar_colors, zorder=3)

            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}",
                        ha="center", va="bottom", fontsize=7, rotation=45,
                    )

            finite = [v for v in vals if not np.isnan(v)]
            ylim_top = min(1.0, (max(finite) if finite else 0.2) * 1.3 + 0.05)
            ax.set_title(label, fontsize=10, pad=4)
            ax.set_xticks(x)
            ax.set_xticklabels(tick_labels, fontsize=7.5, rotation=30, ha="right")
            ax.set_ylim(0, ylim_top)
            ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
            ax.set_axisbelow(True)

        handles = [
            mpatches.Patch(color=PALETTE_ZS, label="Zero-shot (base)"),
            mpatches.Patch(color=PALETTE_FT, label="Fine-tuned"),
        ]
        fig.legend(handles=handles, loc="upper right", fontsize=9, frameon=False)
        fig.tight_layout()
        path = output_dir / f"codegen_{bm}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


# ── 그래프 6: Codegen pass@1 delta heatmap ───────────────────

def plot_codegen_delta_heatmap(codegen: dict, output_dir: Path):
    """fine-tuned vs zero-shot pass@1 delta를 벤치마크 × 모드 heatmap으로 표시."""
    available = [bm for bm in CODEGEN_BENCHMARKS if bm in codegen]
    if not available:
        return

    # 공통 fine-tuned tag 목록 (base 제외)
    all_ft_tags: set = set()
    for bm in available:
        all_ft_tags.update(t for t in codegen[bm] if t != "base")
    ft_tags = ordered_codegen_tags([t for t in _CONGRA_CODEGEN_TAG_ORDER if t in all_ft_tags]
                                    + [t for t in sorted(all_ft_tags) if t not in _CONGRA_CODEGEN_TAG_ORDER])
    if not ft_tags:
        return

    # delta matrix: rows = ft_tags, cols = benchmarks
    matrix = np.full((len(ft_tags), len(available)), float("nan"))
    for bi, bm in enumerate(available):
        base_p1 = codegen[bm].get("base", {}).get("pass@1", float("nan"))
        for ti, tag in enumerate(ft_tags):
            ft_p1 = codegen[bm].get(tag, {}).get("pass@1", float("nan"))
            if not (np.isnan(base_p1) or np.isnan(ft_p1)):
                matrix[ti, bi] = ft_p1 - base_p1

    if np.all(np.isnan(matrix)):
        return

    vmax = np.nanmax(np.abs(matrix))
    fig, ax = plt.subplots(figsize=(max(4, len(available) * 2.2), max(3, len(ft_tags) * 0.85) + 1.2))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(available)))
    ax.set_xticklabels([CODEGEN_BM_LABELS[bm] for bm in available], fontsize=10)
    ax.set_yticks(range(len(ft_tags)))
    ax.set_yticklabels(ft_tags, fontsize=9)
    ax.set_title("Code Gen pass@1 Δ (fine-tuned − zero-shot)\n(green = fine-tuning helps)", fontsize=11, pad=10)

    for ti in range(len(ft_tags)):
        for bi in range(len(available)):
            val = matrix[ti, bi]
            if not np.isnan(val):
                sign = "+" if val > 0 else ""
                ax.text(bi, ti, f"{sign}{val:.3f}", ha="center", va="center", fontsize=9,
                        color="black" if abs(val) < vmax * 0.6 else "white")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Δ pass@1 (fine-tuned − zero-shot)")
    fig.tight_layout()
    path = output_dir / "codegen_delta_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 그래프 7: ConGra vs Codegen scatter (forgetting 분석) ─────

def plot_codegen_vs_congra(data: dict, codegen: dict, modes: list, output_dir: Path, ctx: int):
    """ConGra CodeBLEU vs 코드젠 pass@1 scatter – catastrophic forgetting 분석."""
    available = [bm for bm in CODEGEN_BENCHMARKS if bm in codegen]
    if not available or not modes:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5.5 * len(available), 4.5))
    if len(available) == 1:
        axes = [axes]

    mode_colors = dict(zip(modes, plt.cm.tab10(np.linspace(0, 0.8, len(modes)))))

    for ax, bm in zip(axes, available):
        base_p1 = codegen[bm].get("base", {}).get("pass@1", None)

        for m in modes:
            congra_overall = get_overall(data, m, "finetuned")
            if congra_overall is None:
                continue
            congra_codebleu = metric_val(congra_overall, "avg_codebleu")

            tag_key = f"{m}_ctx{ctx}"
            cg_p1 = codegen[bm].get(tag_key, {}).get("pass@1", float("nan"))
            if np.isnan(cg_p1) or np.isnan(congra_codebleu):
                continue

            ax.scatter(congra_codebleu, cg_p1, s=90, color=mode_colors[m], zorder=3, label=m)
            ax.annotate(m, (congra_codebleu, cg_p1),
                        textcoords="offset points", xytext=(6, 3), fontsize=8)

        if base_p1 is not None:
            ax.axhline(base_p1, color=PALETTE_ZS, linestyle="--", linewidth=1.2,
                       label=f"Base zero-shot  {base_p1:.3f}")

        ax.set_xlabel("ConGra CodeBLEU (fine-tuned)", fontsize=9)
        ax.set_ylabel(f"{CODEGEN_BM_LABELS[bm]}  pass@1", fontsize=9)
        ax.set_title(f"ConGra vs {CODEGEN_BM_LABELS[bm]}\n(ctx={ctx})", fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    path = output_dir / f"codegen_vs_congra_ctx{ctx}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 텍스트 요약 테이블 ────────────────────────────────────────

def print_summary_table(data, modes, ctx):
    keys = METRICS
    labels = METRIC_LABELS
    col_w = 11

    print(f"\n{'='*80}")
    print(f"  Summary Table  (ctx={ctx})")
    print(f"{'='*80}")
    header = f"{'Model':<22}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    print("-" * len(header))

    for m in modes:
        for tag, prefix in [("base", "zs/"), ("finetuned", "ft/")]:
            overall = get_overall(data, m, tag)
            if overall is None:
                continue
            name = f"{prefix}{m}"
            row = f"{name:<22}" + "".join(f"{metric_val(overall, k):>{col_w}.4f}" for k in keys)
            print(row)
        print()

    print("=" * 80)


def print_codegen_summary(codegen: dict):
    """코드젠 벤치마크 결과 텍스트 요약."""
    available = [bm for bm in CODEGEN_BENCHMARKS if bm in codegen]
    if not available:
        return

    print(f"\n{'='*80}")
    print("  Code Generation Benchmark Summary")
    print(f"{'='*80}")

    for bm in available:
        tags = ordered_codegen_tags(list(codegen[bm].keys()))
        metrics = CODEGEN_BM_METRICS[bm]
        m_labels = CODEGEN_BM_METRIC_LABELS[bm]
        col_w = 11

        print(f"\n  [{CODEGEN_BM_LABELS[bm]}]")
        header = f"  {'Tag':<28}" + "".join(f"{l:>{col_w}}" for l in m_labels)
        print(header)
        print("  " + "-" * (len(header) - 2))

        base_p1 = codegen[bm].get("base", {}).get("pass@1", None)
        for tag in tags:
            vals = [codegen[bm][tag].get(k, float("nan")) for k in metrics]
            row = f"  {tag:<28}" + "".join(f"{v:>{col_w}.4f}" for v in vals)
            # pass@1 delta vs base 표시
            if tag != "base" and base_p1 is not None:
                delta = vals[0] - base_p1
                sign = "+" if delta >= 0 else ""
                row += f"   (Δpass@1 {sign}{delta:.4f})"
            print(row)

    print(f"\n{'='*80}")


# ── main ──────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="./eval_results")
    p.add_argument("--output_dir", default="./figures")
    p.add_argument("--model-tag", default=None, help="eval_results/<model-tag> 하위 결과를 사용")
    p.add_argument("--ctx", type=int, default=None, help="컨텍스트 라인 수 (미지정 시 자동 탐지)")
    p.add_argument("--no_codegen", action="store_true", help="코드젠 벤치마크 플롯 건너뜀")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    if args.model_tag:
        results_dir = results_dir / args.model_tag
        if args.output_dir == "./figures":
            output_dir = output_dir / args.model_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"[ERROR] results_dir not found: {results_dir}")
        return

    # ── ConGra ablation ──────────────────────────────────────────
    ctxs = [args.ctx] if args.ctx else detect_ctx(results_dir)
    if not ctxs:
        print(f"[Warning] No ConGra results found in {results_dir}")
    else:
        for ctx in ctxs:
            print(f"\n[ctx={ctx}] Loading ConGra results...")
            data = load_results(results_dir, ctx)
            if not data:
                print(f"  No data found for ctx={ctx}")
                continue

            modes = ordered_modes(data)
            print(f"  Modes: {modes}")
            print(f"  Generating ConGra plots...")

            title_suffix = f"  [{args.model_tag}]" if args.model_tag else ""
            plot_ablation_overview(data, modes, output_dir, ctx, title_suffix)
            plot_delta_heatmap(data, modes, output_dir, ctx, title_suffix)
            plot_per_project(data, modes, output_dir, ctx, title_suffix)
            plot_radar(data, modes, output_dir, ctx, title_suffix)
            print_summary_table(data, modes, ctx)

    # ── Code Generation benchmarks ───────────────────────────────
    if not args.no_codegen:
        print(f"\nLoading codegen results...")
        codegen = load_codegen_results(results_dir)
        if not codegen:
            print("  No codegen results found (run eval.sh --humanevalx first)")
        else:
            print(f"  Benchmarks: {list(codegen.keys())}")
            print(f"  Generating codegen plots...")
            plot_codegen_bars(codegen, output_dir)
            plot_codegen_delta_heatmap(codegen, output_dir)

            # ConGra vs codegen scatter: ctx별로 생성
            if ctxs:
                for ctx in ctxs:
                    data = load_results(results_dir, ctx)
                    if data:
                        modes = ordered_modes(data)
                        plot_codegen_vs_congra(data, codegen, modes, output_dir, ctx)

            print_codegen_summary(codegen)

    print(f"\nAll figures saved to: {output_dir}/")


if __name__ == "__main__":
    main()
