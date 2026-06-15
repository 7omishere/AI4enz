#!/bin/bash
# 诊断脚本：检查 ESM-2 编码任务状态
# 运行方式: bash /home/domi/AI4enz/dataset_building/scripts/check_esm2_status.sh

echo "=== 1. 清理临时文件 ==="
rm -rf /tmp/claude-1000
echo "Done."

echo ""
echo "=== 2. proteins.h5 状态 ==="
source /home/domi/BINN/.venv/bin/activate
python3 -c "
import h5py
h5 = h5py.File('/home/domi/AI4enz/dataset_building/processed/proteins.h5', 'r')
keys = list(h5.keys())
has_esm = sum(1 for k in keys if 'esm2_embed' in h5[k])
has_seq = sum(1 for k in keys if 'sequence' in h5[k])
print(f'Total proteins: {len(keys):,}')
print(f'With sequence: {has_seq:,}')
print(f'With ESM-2: {has_esm:,}')
h5.close()
"

echo ""
echo "=== 3. ESM-2 进程检查 ==="
ps aux | grep -E "encode_trenzition|esm" | grep -v grep

echo ""
echo "=== 4. metadata.parquet 状态 ==="
python3 -c "
import pandas as pd
m = pd.read_parquet('/home/domi/AI4enz/dataset_building/processed/metadata.parquet')
print(f'Rows: {len(m):,}')
print(f'Splits: {m[\"split\"].value_counts().to_dict()}')
h = set(m['protein_seq_hash'])
print(f'Unique proteins: {len(h):,}')
"

echo ""
echo "=== 5. 如果 ESM-2 未完成，重新启动 ==="
echo "运行: cd /home/domi/AI4enz/dataset_building && source /home/domi/BINN/.venv/bin/activate && python scripts/encode_trenzition_v5.py --device cpu --esm-batch-size 1"
