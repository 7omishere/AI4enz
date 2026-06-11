#!/usr/bin/env python3
"""
解析 BRENDA/SKiD kcat_archive 数据
从 metadata.txt 文件中提取 kcat、Km、EC号、UniProt等信息
"""

import os
import re
import json
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

ARCHIVE_DIR = Path(__file__).parent.parent / "external_data" / "kcat_archive"
OUTPUT_FILE = Path(__file__).parent.parent / "processed" / "oxidoreductase" / "brenda_kcat_data.parquet"

def parse_metadata_file(filepath: Path) -> dict:
    """解析单个 metadata.txt 文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return None
    
    record = {'entry_id': filepath.stem}
    
    # 解析各个字段
    patterns = {
        'ec_number': r'EC number:\s*(.+?)(?:\n|$)',
        'uniprot_id': r'UniProtKB ID:\s*(.+?)(?:\n|$)',
        'organism': r'Source organism:\s*(.+?)(?:\n|$)',
        'substrate_name': r'Substrate name:\s*(.+?)(?:\n|$)',
        'substrate_smiles': r'Substrate SMILES:\s*(.+?)(?:\n|$)',
        'ph': r'Reaction pH:\s*([\d.]+)',
        'temperature': r'Reaction temperature:\s*([\d.]+)',
        'mutation': r'Mutation \(if present\):\s*(.+?)(?:\n|$)',
        'kcat_value': r'kcat \(pkcat\) value:\s*([\d.e+-]+)\s*\(([\d.e+-]+)\)',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            if key == 'kcat_value':
                record['kcat'] = float(match.group(1))
                record['pkcat'] = float(match.group(2))
            else:
                record[key] = match.group(1).strip()
    
    return record if 'ec_number' in record else None

def process_archives():
    """批量处理所有 archive 目录"""
    print("=== 解析 kcat_archive 数据 ===\n")
    
    # 查找所有 metadata.txt 文件 (格式: SKiD_kcat_N_metadata.txt)
    metadata_files = list(ARCHIVE_DIR.glob("*/SKiD_kcat_*_metadata.txt"))
    print(f"找到 {len(metadata_files)} 个元数据文件\n")
    
    # 并行解析
    records = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(parse_metadata_file, f): f for f in metadata_files}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="解析进度"):
            result = future.result()
            if result:
                records.append(result)
    
    print(f"\n成功解析: {len(records)} 条记录")
    
    # 转换为 DataFrame
    df = pd.DataFrame(records)
    
    # 只保留氧化还原酶 (EC 1.x.x.x)
    df_ox = df[df['ec_number'].str.startswith('1.', na=False)]
    print(f"氧化还原酶 (1.x.x.x): {len(df_ox)} 条")
    
    # 统计
    print(f"\n=== 数据统计 ===")
    print(f"有 kcat 值: {df_ox['kcat'].notna().sum()}")
    print(f"有 UniProt ID: {df_ox['uniprot_id'].notna().sum()}")
    
    # EC号分布
    print(f"\n=== EC号分布 (前10) ===")
    ec_dist = df_ox['ec_number'].value_counts().head(10)
    for ec, cnt in ec_dist.items():
        print(f"  {ec}: {cnt}")
    
    # 保存
    df_ox.to_parquet(OUTPUT_FILE, index=False)
    print(f"\n已保存到: {OUTPUT_FILE}")
    
    return df_ox

if __name__ == "__main__":
    process_archives()