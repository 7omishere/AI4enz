#!/usr/bin/env python3
"""
从 SABIO-RK Export API 下载氧化还原酶 (EC 1.x.x.x) 的 kcat 数据
API: https://sabiork.h-its.org/export-api/sabio/kinlaw-entry/json
限制: 60 requests/min
"""

import requests
import json
import time
import pandas as pd
import os
from datetime import datetime

BASE_URL = "https://sabiork.h-its.org/export-api/sabio/kinlaw-entry/json"
OUTPUT_DIR = "/home/domi/AI4enz/dataset_building/processed/oxidoreductase"

def download_sabio_data(ec_pattern="1.*", max_pages=1000, rows_per_page=100):
    """
    下载 SABIO-RK 数据
    
    Args:
        ec_pattern: EC 编号模式，如 "1.*" 表示氧化还原酶
        max_pages: 最大页数
        rows_per_page: 每页记录数（API限制）
    
    Returns:
        DataFrame with kcat data
    """
    all_kcat = []
    
    for page in range(1, max_pages + 1):
        start = (page - 1) * rows_per_page
        
        # 构建查询
        query = f"EcNumber:{ec_pattern}"
        url = f"{BASE_URL}?q={query}&rows={rows_per_page}&start={start}"
        
        try:
            resp = requests.get(url, timeout=30)
            
            if resp.status_code == 429:
                print(f"  ⚠️ 速率限制，等待 60 秒...")
                time.sleep(60)
                continue
            
            if resp.status_code != 200:
                print(f"  ❌ 错误 {resp.status_code}: {resp.text[:100]}")
                break
            
            data = resp.json()
            entries = data.get('data', [])
            total = data.get('meta', {}).get('total_count', 0)
            
            if page == 1:
                print(f"📊 总共 {total:,} 条氧化还原酶记录")
            
            if not entries:
                print(f"  ✅ 完成 (无更多数据)")
                break
            
            # 提取 kcat 数据
            for entry in entries:
                enzyme = entry.get('enzyme_description', {})
                ec = enzyme.get('ec_number', '')
                
                if not ec or not ec.startswith('1.'):
                    continue
                
                # 获取 proteins 中的 UniProt ID
                proteins = enzyme.get('proteins', [])
                uniprot_ids = []
                for protein in proteins:
                    uid = protein.get('uniprot_id')
                    if uid:
                        # 处理复合体格式如 "(P20932)*4"
                        import re
                        matches = re.findall(r'[A-Z]\d{5}', str(uid))
                        uniprot_ids.extend(matches)
                
                if not uniprot_ids:
                    continue
                
                # 获取 kcat 参数
                kineticlaw = entry.get('kineticlaw', {})
                params = kineticlaw.get('parameter', [])
                
                for param in params:
                    param_type = param.get('parameter_type', {})
                    if param_type.get('name') != 'kcat':
                        continue
                    
                    kcat_value = param.get('start_value')
                    kcat_unit = param.get('unit', {}).get('name', 's^(-1)')
                    
                    if kcat_value is None:
                        continue
                    
                    # 获取实验条件
                    conditions = entry.get('experimental_conditions', {})
                    
                    for uniprot_id in set(uniprot_ids):  # 去重
                        all_kcat.append({
                            'ec_number': ec,
                            'uniprot_id': uniprot_id,
                            'kcat_value': kcat_value,
                            'kcat_unit': kcat_unit,
                            'organism': entry.get('general', {}).get('organism', {}).get('name'),
                            'enzyme_name': enzyme.get('enzyme_name'),
                            'ph': conditions.get('envvar_ph', {}).get('start_value'),
                            'temperature': conditions.get('envvar_temperature', {}).get('start_value'),
                            'publication_id': entry.get('publication', {}).get('pubmed_id'),
                            'sabio_id': entry.get('id'),
                        })
            
            # 进度
            if page % 10 == 0:
                print(f"  已处理 {(page-1)*rows_per_page + len(entries)}/{min(total, max_pages*rows_per_page)} 条...")
            
            # 速率限制：60 req/min，保守用 50 req/min
            time.sleep(1.2)
            
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            time.sleep(5)
    
    return pd.DataFrame(all_kcat)


if __name__ == "__main__":
    print("=" * 60)
    print("SABIO-RK kcat 数据下载")
    print("=" * 60)
    
    # 下载氧化还原酶数据
    print("\n📥 下载氧化还原酶 (EC 1.x.x.x) kcat 数据...")
    df = download_sabio_data(ec_pattern="1.*", max_pages=500)
    
    print(f"\n✅ 下载完成!")
    print(f"   提取到 {len(df):,} 条 kcat 记录")
    
    if len(df) > 0:
        # 统计
        print(f"\n📊 统计:")
        print(f"   唯一 UniProt ID: {df['uniprot_id'].nunique():,}")
        print(f"   唯一 EC 号: {df['ec_number'].nunique():,}")
        print(f"   kcat 范围: [{df['kcat_value'].min():.3f}, {df['kcat_value'].max():.3f}]")
        
        # 保存
        output_path = os.path.join(OUTPUT_DIR, "sabio_rk_kcat.parquet")
        df.to_parquet(output_path, index=False)
        print(f"\n💾 保存到: {output_path}")
        
        # 保存 JSON 格式（便于查看）
        json_path = os.path.join(OUTPUT_DIR, "sabio_rk_kcat.json")
        df.to_json(json_path, orient='records', lines=True)
        print(f"💾 JSON 格式: {json_path}")