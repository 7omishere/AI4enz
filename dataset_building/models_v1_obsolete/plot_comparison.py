#!/usr/bin/env python3
"""可视化：Trenzition vs Baselines 模型对比"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_PATH = Path(__file__).resolve().parent / "checkpoints" / "evaluation_results.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "checkpoints" / "model_comparison.png"

def main():
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Trenzition vs Baselines — Model Comparison on Test Set",
                 fontsize=15, fontweight="bold")

    colors = {"Trenzition": "#d62728", "RandomForest": "#1f77b4",
              "MLP": "#ff7f0e", "Ridge": "#2ca02c", "Mean": "#7f7f7f"}

    for task_idx, task in enumerate(["pkd", "kcat"]):
        t = task.upper()
        col = task_idx

        # Gather data
        trenz = data["Trenzition"].get(task, {})
        baselines = data["Baselines"].get(task, {})

        all_models = {"Trenzition": trenz, **baselines}

        metrics = ["R2", "Spearman_rho", "MSE", "MAE"]
        metric_labels = ["R²", "Spearman ρ", "MSE", "MAE"]
        metric_better_higher = [True, True, False, False]

        for mi, (metric, label, higher_better) in enumerate(
            zip(metrics, metric_labels, metric_better_higher)
        ):
            ax = axes[mi // 2, mi % 2]
            names = list(all_models.keys())
            values = [all_models[n].get(metric, 0) for n in names]

            # Handle NaN
            for i, v in enumerate(values):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    values[i] = 0

            # Sort
            if higher_better:
                order = np.argsort(values)[::-1]
            else:
                order = np.argsort(values)

            names_sorted = [names[i] for i in order]
            values_sorted = [values[i] for i in order]
            bar_colors = [colors.get(n, "#999999") for n in names_sorted]

            bars = ax.bar(names_sorted, values_sorted, color=bar_colors, edgecolor="white", linewidth=0.5)
            ax.set_title(f"{t} — {label}")
            ax.set_ylabel(label)

            # Value labels
            for bar, val in zip(bars, values_sorted):
                fmt = f"{val:.4f}" if abs(val) < 10 else f"{val:.2f}"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f" {fmt}", ha="center", va="bottom" if val >= 0 else "top",
                        fontsize=7, rotation=90, color="black")

            ax.tick_params(axis="x", rotation=45, labelsize=8)
            ax.grid(axis="y", alpha=0.3)

    # ── Combined scatter: Trenzition R² across tasks ──
    ax = axes[1, 1]
    tasks_labels = ["pKd (Binding Affinity)", "kcat (Catalytic Efficiency)"]
    r2_values = [
        data["Trenzition"].get("pkd", {}).get("R2", 0),
        data["Trenzition"].get("kcat", {}).get("R2", 0),
    ]
    baseline_r2 = [
        data["Baselines"].get("pkd", {}).get("RandomForest", {}).get("R2", 0),
        data["Baselines"].get("kcat", {}).get("RandomForest", {}).get("R2", 0),
    ]

    x = np.arange(len(tasks_labels))
    width = 0.35
    ax.bar(x - width / 2, baseline_r2, width, label="RandomForest (best baseline)",
           color="#1f77b4", edgecolor="white")
    ax.bar(x + width / 2, r2_values, width, label="Trenzition",
           color="#d62728", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks_labels)
    ax.set_ylabel("R²")
    ax.set_title("R²: Trenzition vs Best Baseline")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Value labels
    for i, (br, tr) in enumerate(zip(baseline_r2, r2_values)):
        ax.text(i - width / 2, br, f" {br:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, tr, f" {tr:.3f}", ha="center", va="bottom", fontsize=8,
                fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUTPUT_PATH}")

    # Print summary
    print(f"\nTrenzition vs Baselines — Summary")
    print(f"{'='*60}")
    for task in ["pkd", "kcat"]:
        t = data["Trenzition"].get(task, {})
        b = data["Baselines"].get(task, {})
        best_base = max(b.items(), key=lambda x: x[1].get("R2", -999))
        print(f"\n{task.upper()}:")
        print(f"  Trenzition R²:     {t.get('R2', 0):.4f}")
        print(f"  Best baseline:     {best_base[0]} (R²={best_base[1].get('R2', 0):.4f})")
        print(f"  Trenzition uplift: {t.get('R2', 0) / best_base[1].get('R2', 1):.1f}×")

if __name__ == "__main__":
    main()
