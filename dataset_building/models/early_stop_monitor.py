#!/usr/bin/env python3
"""早停监控脚本：持续读取训练日志，val loss 在 patience 个 epoch 内未改善则 kill."""

import re
import os
import time
import signal
import argparse

def parse_epochs(log_path: str) -> list[tuple[int, float]]:
    """解析日志中的 epoch 和 val loss，返回 [(epoch, val_loss), ...]"""
    results = []
    pattern = re.compile(
        r'Epoch\s+(\d+)/\d+\s+\|\s+train:\s+[\d.]+\s+val:\s+([\d.]+)'
    )
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                results.append((int(m.group(1)), float(m.group(2))))
    return results

def main():
    parser = argparse.ArgumentParser(description="Early stopping monitor")
    parser.add_argument("--log", required=True, help="Training log path")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--pid", type=int, help="Kill this PID on stop")
    args = parser.parse_args()

    print(f"[monitor] log={args.log}, patience={args.patience}, min_delta={args.min_delta}, pid={args.pid}")

    while True:
        epochs = parse_epochs(args.log)
        if not epochs:
            print("[monitor] No epochs found, waiting...")
            time.sleep(60)
            continue

        best_val = min(v for _, v in epochs)
        best_epoch = next(e for e, v in epochs if v == best_val)
        current_epoch = epochs[-1][0]
        epochs_since_best = current_epoch - best_epoch

        recent = epochs[-5:]
        recent_str = "  ".join(f"E{e}:{v:.5f}" for e, v in recent)
        print(f"[monitor] Epoch {current_epoch:3d} | best: {best_val:.5f} (E{best_epoch}) | "
              f"no improvement for {epochs_since_best} epochs | recent: {recent_str}")

        if epochs_since_best >= args.patience:
            print(f"[monitor] STOP! Val loss hasn't improved for {epochs_since_best} ≥ {args.patience} epochs.")
            if args.pid:
                os.kill(args.pid, signal.SIGTERM)
                print(f"[monitor] Sent SIGTERM to PID {args.pid}")
            break

        if current_epoch >= 100:
            print("[monitor] Training finished (100 epochs).")
            break

        time.sleep(60)

if __name__ == "__main__":
    main()
