#!/usr/bin/env python3
"""
从 UniProt API 批量获取 EC 号
使用 UniProt Search API 进行批量查询
"""

import requests
import time
import json
import pandas as pd
from pathlib import Path
import sys

# 配置
BATCH_SIZE = 100  # 批量大小
REQUEST_DELAY = 0.5  # 请求间隔（秒）
CACHE_FILE = Path(__file__).parent / "uniprot_ec_cache.json"
LOG_FILE = Path(__file__).parent / "uniprot_ec_fetch.log"

def log(msg):
    """日志输出"""
    print(msg, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

def load_cache():
    """加载缓存"""
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    """保存缓存"""
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def extract_ec_numbers(data: dict) -> list:
    """从 UniProt JSON 中提取 EC 号"""
    ec_numbers = []
    
    # 方式1: proteinDescription.recommendedName.ecNumbers
    if 'proteinDescription' in data:
        pd = data['proteinDescription']
        if 'recommendedName' in pd:
            rn = pd['recommendedName']
            if 'ecNumbers' in rn:
                for ec in rn['ecNumbers']:
                    if 'value' in ec:
                        ec_numbers.append(ec['value'])
    
    # 方式2: alternativeNames (可选名称)
    if 'proteinDescription' in data:
        pd = data['proteinDescription']
        if 'alternativeNames' in pd:
            for alt in pd['alternativeNames']:
                if 'ecNumbers' in alt:
                    for ec in alt['ecNumbers']:
                        if 'value' in ec:
                            ec_numbers.append(ec['value'])
    
    return ec_numbers

def fetch_single_ec(accession: str) -> str:
    """获取单个蛋白质的 EC 号"""
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            ec_numbers = extract_ec_numbers(data)
            
            # 只保留氧化还原酶 (1.x.x.x)
            for ec in ec_numbers:
                if ec.startswith('1.'):
                    return ec
    except:
        pass
    
    return None

def fetch_all_ec(uniprot_ids: list) -> dict:
    """
    获取所有 EC 号
    """
    cache = load_cache()
    results = cache.copy()
    
    # 过滤出未缓存的 ID
    uncached = [uid for uid in uniprot_ids if uid not in cache]
    total = len(uncached)
    
    log(f"总需查询: {total} 个 UniProt ID")
    
    for i, accession in enumerate(uncached):
        ec = fetch_single_ec(accession)
        results[accession] = ec
        
        if (i + 1) % 50 == 0:
            saved_ec = sum(1 for v in results.values() if v and v.startswith('1.'))
            log(f"  进度: {i+1}/{total} | 已获取EC号: {saved_ec}")
            save_cache(results)
        
        time.sleep(REQUEST_DELAY)
    
    save_cache(results)
    return results

def main():
    log("=== 开始获取 UniProt EC 号 ===")
    
    # 加载数据集
    df = pd.read_parquet('dataset_building/processed/oxidoreductase/recommended_training_set_clean.parquet')
    
    # 找出缺失 EC 号的记录
    missing_ec = df[df['ec_numbers'].isna() | (df['ec_numbers'] == '')]
    missing_uniprots = missing_ec['uniprot_id'].dropna().unique().tolist()
    
    log(f"需要查询的 UniProt ID 数: {len(missing_uniprots)}")
    
    # 获取 EC 号
    all_ec = fetch_all_ec(missing_uniprots)
    
    # 统计结果
    valid_ec = sum(1 for ec in all_ec.values() if ec and ec.startswith('1.'))
    
    log(f"\n=== 查询结果 ===")
    log(f"成功获取氧化还原酶 EC 号: {valid_ec}")
    
    # 创建 Uniprot -> EC 号映射
    uniprot_ec_map = {uid: ec for uid, ec in all_ec.items() if ec}
    
    # 更新数据集
    log("更新数据集...")
    df_updated = df.copy()
    
    def get_ec(row):
        if pd.notna(row['ec_numbers']) and row['ec_numbers'] != '':
            return row['ec_numbers']
        return uniprot_ec_map.get(row['uniprot_id'], None)
    
    df_updated['ec_numbers'] = df_updated.apply(get_ec, axis=1)
    
    # 统计更新后
    has_ec = df_updated['ec_numbers'].notna() & (df_updated['ec_numbers'] != '')
    log(f"=== 更新后统计 ===")
    log(f"有 EC 号的记录: {has_ec.sum()} / {len(df_updated)}")
    log(f"EC 号覆盖率: {100*has_ec.sum()/len(df_updated):.1f}%")
    
    # EC号分布
    log(f"\n=== EC号分布 (前10) ===")
    ec_dist = df_updated[df_updated['ec_numbers'].notna()]['ec_numbers'].value_counts().head(10)
    for ec, cnt in ec_dist.items():
        log(f"  {ec}: {cnt}")
    
    # 保存结果
    output_path = 'dataset_building/processed/oxidoreductase/recommended_training_set_with_ec.parquet'
    df_updated.to_parquet(output_path, index=False)
    log(f"已保存到: {output_path}")

if __name__ == "__main__":
    main()