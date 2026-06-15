#!/usr/bin/env python3
"""清理临时文件 + 检查 ESM-2 进度 + 重启编码任务（如需要）"""
import subprocess, os, sys, time

# 1. 清理
print("=== 1. 清理 /tmp/claude-1000 ===")
result = subprocess.run(["rm", "-rf", "/tmp/claude-1000"], capture_output=True, text=True)
print(f"exit={result.returncode}")

# 2. 检查 proteins.h5
print("\n=== 2. proteins.h5 状态 ===")
import h5py
h5_path = "/home/domi/AI4enz/dataset_building/processed/proteins.h5"
with h5py.File(h5_path, 'r') as h5:
    keys = list(h5.keys())
    has_esm = sum(1 for k in keys if 'esm2_embed' in h5[k])
    print(f"Total: {len(keys):,}, with ESM-2: {has_esm:,}")

# 3. 检查进程
print("\n=== 3. ESM-2 进程 ===")
result = subprocess.run(["pgrep", "-af", "encode_trenzition"], capture_output=True, text=True)
if result.stdout.strip():
    print(result.stdout)
else:
    print("没有 encode_trenzition 进程在运行")

# 4. metadata
print("\n=== 4. metadata.parquet ===")
import pandas as pd
meta = pd.read_parquet("/home/domi/AI4enz/dataset_building/processed/metadata.parquet")
print(f"Rows: {len(meta):,}")
print(f"Splits: {meta['split'].value_counts().to_dict()}")

# 5. 重启 ESM-2 (如需要)
print("\n=== 5. 建议 ===")
esm2_target = 19280
if has_esm < esm2_target:
    missing = esm2_target - has_esm
    print(f"还缺 {missing} 个蛋白的 ESM-2 ({100*has_esm/esm2_target:.1f}% 完成)")
    print("重新启动: cd /home/domi/AI4enz/dataset_building && source /home/domi/BINN/.venv/bin/activate && python scripts/encode_trenzition_v5.py --device cpu --esm-batch-size 1")
else:
    print("✓ ESM-2 全部完成!")
