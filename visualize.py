"""
eval_results/ 디렉토리의 JSON 파일을 읽어 ablation 결과를 시각화.

Usage:
    python visualize.py                          # eval_results/ 자동 탐색
    python visualize.py --results_dir ./eval_results --ctx 20
    python visualize.py --output_dir ./figures
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

def plot_ablation_overview(data, modes, output_dir, ctx):
    n_metrics = len(METRICS)
    n_modes = len(modes)
    has_base = any("base" in data[m] for m in modes)

    fig, axes = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics, 4.5))
    fig.suptitle(f"Ablation Overview  (ctx={ctx})", fontsize=14, fontweight="bold", y=1.02)

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

def plot_delta_heatmap(data, modes, output_dir, ctx):
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
    ax.set_title(f"Fine-tuned Δ over Zero-shot  (ctx={ctx})\n(green = fine-tuned better)", fontsize=11, pad=10)

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

def plot_per_project(data, modes, output_dir, ctx):
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
    fig.suptitle(f"Per-project Metrics by Mode  (ctx={ctx})", fontsize=13, fontweight="bold")

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

def plot_radar(data, modes, output_dir, ctx):
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
    ax.set_title(f"Ablation Radar  (ctx={ctx}, fine-tuned)", fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9, frameon=False)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    path = output_dir / f"radar_ctx{ctx}.png"
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


# ── main ──────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="./eval_results")
    p.add_argument("--output_dir", default="./figures")
    p.add_argument("--ctx", type=int, default=None, help="컨텍스트 라인 수 (미지정 시 자동 탐지)")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"[ERROR] results_dir not found: {results_dir}")
        return

    ctxs = [args.ctx] if args.ctx else detect_ctx(results_dir)
    if not ctxs:
        print(f"[ERROR] No results found in {results_dir}")
        return

    for ctx in ctxs:
        print(f"\n[ctx={ctx}] Loading results...")
        data = load_results(results_dir, ctx)
        if not data:
            print(f"  No data found for ctx={ctx}")
            continue

        modes = ordered_modes(data)
        print(f"  Modes: {modes}")
        print(f"  Generating plots...")

        plot_ablation_overview(data, modes, output_dir, ctx)
        plot_delta_heatmap(data, modes, output_dir, ctx)
        plot_per_project(data, modes, output_dir, ctx)
        plot_radar(data, modes, output_dir, ctx)
        print_summary_table(data, modes, ctx)

    print(f"\nAll figures saved to: {output_dir}/")


if __name__ == "__main__":
    main()
