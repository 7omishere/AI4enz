#!/usr/bin/env python3
"""可视化消融实验结果"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_PATH = Path(__file__).resolve().parent / "checkpoints" / "ablation_results.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "checkpoints" / "ablation_study.png"

def main():
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    # Simplify names for display
    name_map = {
        "Full Model": "Full Model",
        "No Gate": "No Gate",
        "No Cofactor": "No Cofactor",
        "No Ligand GNN": "No Ligand",
        "No Ligand + Cofactor": "No Lig+Cof",
        "ODE Steps = 1": "ODE Steps=1",
        "No BINN (skip ODE)": "No BINN",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Ablation Study — Component Contributions to Trenzition Performance",
                 fontsize=15, fontweight="bold")

    variants_full = list(data.keys())
    variants_short = [name_map.get(v, v) for v in variants_full]

    # Color map
    colors = {
        "Full Model": "#2ca02c",
        "No Gate": "#1f77b4",
        "No Cofactor": "#ff7f0e",
        "No Ligand GNN": "#d62728",
        "No Ligand + Cofactor": "#9467bd",
        "ODE Steps = 1": "#8c564b",
        "No BINN (skip ODE)": "#7f7f7f",
    }

    for task_idx, task in enumerate(["pkd", "kcat"]):
        # ── R² bar chart ──
        ax = axes[task_idx, 0]
        values = [data[v][task]["R2"] for v in variants_full]

        # Clip extreme negative values for readability
        display_values = [max(v, -5.0) for v in values]

        bar_colors = [colors.get(v, "#999") for v in variants_full]
        bars = ax.barh(variants_short, display_values, color=bar_colors, edgecolor="white")

        # Add actual value labels
        for bar, v_display, v_actual in zip(bars, display_values, values):
            label = f"{v_actual:.4f}" if v_actual > -1 else f"{v_actual:.3f}"
            x = max(bar.get_width(), 0.02)
            ax.text(x, bar.get_y() + bar.get_height()/2, f" {label}",
                   va="center", fontsize=8)

        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.set_title(f"{task.upper()} — R² by Component Ablation")
        ax.set_xlabel("R²")

        # ── Δ R² (drop from full) ──
        ax = axes[task_idx, 1]
        base_r2 = data["Full Model"][task]["R2"]
        deltas = [data[v][task]["R2"] - base_r2 for v in variants_full]
        # Clip extreme negatives
        deltas_display = [max(d, -2.0) for d in deltas]

        bar_colors = [colors.get(v, "#999") for v in variants_full]
        bars = ax.barh(variants_short, deltas_display, color=bar_colors, edgecolor="white")

        for bar, d_display, d_actual in zip(bars, deltas_display, deltas):
            label = f"{d_actual:+.4f}" if abs(d_actual) < 1 else f"{d_actual:+.3f}"
            x = bar.get_width()
            x_pos = x + 0.01 if x >= 0 else x - 0.01
            ha = "left" if x >= 0 else "right"
            ax.text(x_pos, bar.get_y() + bar.get_height()/2, f" {label}",
                   va="center", ha=ha, fontsize=8)

        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.set_title(f"{task.upper()} — Δ R² from Full Model")
        ax.set_xlabel("Δ R² (lower = worse)")

    plt.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUTPUT_PATH}")

    # ── Print key findings ──
    print(f"\n{'='*60}")
    print("Key Ablation Findings")
    print(f"{'='*60}")

    base_pkd = data["Full Model"]["pkd"]["R2"]
    base_kcat = data["Full Model"]["kcat"]["R2"]

    components = [
        ("No Ligand GNN", "Ligand GNN"),
        ("No Cofactor", "Cofactor Encoder"),
        ("No Gate", "Gate Mechanism"),
        ("ODE Steps = 1", "Multi-step ODE"),
        ("No BINN (skip ODE)", "BINN / ODE"),
    ]

    for variant, component in components:
        p_impact = (base_pkd - data[variant]["pkd"]["R2"]) / base_pkd * 100
        k_impact = (base_kcat - data[variant]["kcat"]["R2"]) / base_kcat * 100
        print(f"\n{component}:")
        print(f"  pKd R²: {data[variant]['pkd']['R2']:.4f} (drop: {p_impact:+.0f}%)")
        print(f"  kcat R²: {data[variant]['kcat']['R2']:.4f} (drop: {k_impact:+.0f}%)")

if __name__ == "__main__":
    main()
