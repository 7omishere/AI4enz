#!/usr/bin/env python3
"""
通过 UniProt API 获取 EC 号到 UniProt ID 的映射
用于补充 BRENDA 数据与主数据集的匹配
"""

import requests
import pandas as pd
import json
import time
from collections import defaultdict

def fetch_uniprot_by_ec(ec_number: str, max_retries=3) -> list:
    """通过 EC 号查询 UniProt"""
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f"ec:{ec_number}",
        "format": "json",
        "size": 500,
        "fields": "accession,gene_names,organism_name"
    }
    
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                return data.get('results', [])
            elif resp.status_code == 429:
                # Rate limited, wait and retry
                time.sleep(5 * (attempt + 1))
            else:
                print(f"  Error {resp.status_code}: {resp.text[:100]}")
                return []
        except Exception as e:
            print(f"  Exception: {e}")
            time.sleep(2)
    
    return []

def main():
    # 加载 BRENDA 蛋白级数据
    print("加载 BRENDA 蛋白级数据...")
    brenda_pl = pd.read_parquet('processed/oxidoreductase/brenda_2026_protein_level.parquet')
    
    # 获取所有唯一的 EC 号
    all_ecs = set(brenda_pl['ec_number'].unique())
    print(f"BRENDA 中唯一 EC 号: {len(all_ecs)}")
    
    # 加载主数据集的 EC→UniProt 映射 (已匹配部分)
    main_ds = pd.read_parquet('processed/oxidoreductase/recommended_training_set_clean.parquet')
    existing_mapping = defaultdict(set)
    for _, row in main_ds[['ec_numbers', 'uniprot_id']].drop_duplicates().iterrows():
        if pd.notna(row['ec_numbers']) and pd.notna(row['uniprot_id']):
            ec = str(row['ec_numbers']).strip()
            uniprot = str(row['uniprot_id']).strip()
            if uniprot != 'nan':
                existing_mapping[ec].add(uniprot)
    
    print(f"主数据集已有映射: {len(existing_mapping)} EC号")
    
    # 找出需要查询的 EC 号
    ecs_to_query = all_ecs - set(existing_mapping.keys())
    print(f"需要查询的 EC 号: {len(ecs_to_query)}")
    
    # 创建完整的映射
    full_mapping = dict(existing_mapping)
    
    # 查询 UniProt (只查询前100个以节省时间)
    ecs_list = sorted(list(ecs_to_query))[:100]
    print(f"\n开始查询 UniProt (共 {len(ecs_list)} 个 EC号)...")
    
    results = {}
    for i, ec in enumerate(ecs_list):
        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{len(ecs_list)}")
        
        uniprot_ids = fetch_uniprot_by_ec(ec)
        if uniprot_ids:
            ids = [r['primaryAccession'] for r in uniprot_ids]
            full_mapping[ec] = ids
            results[ec] = ids
        
        # 避免频率限制
        if (i + 1) % 10 == 0:
            time.sleep(0.5)
    
    # 保存映射结果
    output_mapping = {ec: list(ids) for ec, ids in full_mapping.items()}
    with open('processed/oxidoreductase/ec_to_uniprot_mapping.json', 'w') as f:
        json.dump(output_mapping, f, indent=2)
    
    # 保存新查询到的映射
    with open('processed/oxidoreductase/new_ec_uniprot_mapping.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ 完成!")
    print(f"总映射数: {len(full_mapping)} EC号")
    print(f"新增映射: {len(results)} EC号")
    print(f"结果保存在: processed/oxidoreductase/ec_to_uniprot_mapping.json")

if __name__ == "__main__":
    main()