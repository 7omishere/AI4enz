#!/usr/bin/env python3
"""Plot training loss curves from train log file."""

import re
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def parse_log(log_path: str) -> dict:
    """Parse training log to extract per-epoch metrics."""
    pattern = re.compile(
        r'Epoch\s+(\d+)/\d+\s+\|\s+train:\s+([\d.]+)\s+val:\s+([\d.]+)\s+'
        r'\(best:\s+([\d.]+)\)\s+\|\s+L_ts:\s+([\d.]+)\s+L_cat:\s+([\d.]+)\s+'
        r'L_barrier:\s+([\d.]+)'
    )

    epochs, train_loss, val_loss, best_vals = [], [], [], []
    l_ts_list, l_cat_list = [], []

    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
                val_loss.append(float(m.group(3)))
                best_vals.append(float(m.group(4)))
                l_ts_list.append(float(m.group(5)))
                l_cat_list.append(float(m.group(6)))

    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_vals": best_vals,
        "l_ts": l_ts_list,
        "l_cat": l_cat_list,
    }


def plot_all(data: dict, output_path: str):
    """Create multi-panel training diagnostic plot."""
    epochs = np.array(data["epochs"])
    train_loss = np.array(data["train_loss"])
    val_loss = np.array(data["val_loss"])
    l_ts = np.array(data["l_ts"])
    l_cat = np.array(data["l_cat"])

    best_idx = np.argmin(val_loss)
    best_epoch = epochs[best_idx]
    best_val = val_loss[best_idx]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Trenzition Training — TransitionBINN on trenzition V5",
                 fontsize=14, fontweight="bold")

    # ── Panel 1: Train + Val Loss ──
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, color="#1f77b4", alpha=0.6, linewidth=0.8, label="Train Loss")
    ax.plot(epochs, val_loss, color="#d62728", linewidth=1.5, label="Val Loss")
    ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.5,
               label=f"Best: E{best_epoch} (val={best_val:.4f})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("Train & Val Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Val Loss (log scale) ──
    ax = axes[0, 1]
    ax.semilogy(epochs, val_loss, color="#d62728", linewidth=1.5)
    ax.scatter(best_epoch, best_val, color="green", s=80, zorder=5,
               label=f"Best: E{best_epoch} ({best_val:.4f})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Loss (log scale)")
    ax.set_title("Val Loss (Log Scale)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Smoothed Val Loss ──
    ax = axes[1, 0]
    window = max(3, len(val_loss) // 20)
    smoothed = np.convolve(val_loss, np.ones(window)/window, mode="valid")
    smooth_epochs = epochs[window-1:]
    ax.plot(smooth_epochs, smoothed, color="#9467bd", linewidth=1.5)
    ax.scatter(best_epoch, best_val, color="green", s=80, zorder=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Smoothed Val Loss (w={window})")
    ax.set_title("Val Loss Trend (Smoothed)")
    ax.grid(True, alpha=0.3)

    # ── Panel 4: Loss Components ──
    ax = axes[1, 1]
    ax.plot(epochs, l_ts, color="#ff7f0e", linewidth=1.2, label="L_ts (pKd)")
    ax.plot(epochs, l_cat, color="#2ca02c", linewidth=1.2, label="L_cat (kcat)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Component Loss")
    ax.set_title("Loss Components: pKd vs kcat")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {output_path}")

    # ── Summary stats ──
    print(f"\n{'='*50}")
    print(f"Training Summary — Trenzition on trenzition V5")
    print(f"{'='*50}")
    print(f"Total epochs:      {len(epochs)}")
    print(f"Best val loss:     {best_val:.4f} (epoch {best_epoch})")
    print(f"Final train loss:  {train_loss[-1]:.4f}")
    print(f"Final val loss:    {val_loss[-1]:.4f}")
    print(f"Final L_ts:        {l_ts[-1]:.4f}")
    print(f"Final L_cat:       {l_cat[-1]:.4f}")
    print(f"Val improvement:   {val_loss[0]:.4f} → {best_val:.4f} "
          f"({val_loss[0]/best_val:.1f}×)")

    # Phase analysis
    first_third = val_loss[:len(val_loss)//3].mean()
    last_third = val_loss[-len(val_loss)//3:].mean()
    print(f"\nPhase analysis:")
    print(f"  First 1/3 avg val: {first_third:.4f}")
    print(f"  Last 1/3 avg val:  {last_third:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = str(Path(args.log).parent.parent / "checkpoints" / "loss_curves.png")

    data = parse_log(args.log)
    print(f"Parsed {len(data['epochs'])} epochs from {args.log}")
    plot_all(data, args.output)


if __name__ == "__main__":
    main()
