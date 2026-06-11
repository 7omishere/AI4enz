#!/bin/bash
# 彻底清理临时文件系统
echo "=== 检查 /tmp/claude-1000 ==="
df -h /tmp/claude-1000 2>&1
mount | grep claude 2>&1
echo ""
echo "=== 清理 ==="
rm -rf /tmp/claude-1000 2>&1
echo "exit: $?"
echo ""
echo "=== 检查 ESM-2 进度 ==="
source /home/domi/BINN/.venv/bin/activate
python3 -c "
import h5py
with h5py.File('/home/domi/AI4enz/dataset_building/processed/proteins.h5','r') as h:
    keys=list(h.keys())
    esm=sum(1 for k in keys if 'esm2_embed' in h[k])
print(f'proteins.h5: {len(keys)} total, {esm} with ESM-2 ({100*esm/19280:.1f}%)')
"
echo ""
echo "=== ESM-2 进程 ==="
ps aux | grep encode_trenzition | grep -v grep
