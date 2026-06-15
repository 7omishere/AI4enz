#!/usr/bin/env python3
"""
通过 UniProt API 批量查询 EC 号
用于补充 BindingDB 数据中缺失的 EC 号标注
"""

import requests
import json
import time
from collections import defaultdict

def query_uniprot_ec(uniprot_id: str, max_retries=3) -> list:
    """查询单个 UniProt ID 的 EC 号"""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}"
    params = {"format": "json", "fields": "ecntoz"}
    
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                ec_numbers = data.get('sequences', {}).get('recommendedName', {}).get('ecNumber', [])
                if not ec_numbers:
                    # 尝试 alternativeNames
                    ec_numbers = data.get('sequences', {}).get('recommendedName', {}).get('alternativeNames', [])
                    ec_numbers = [e.get('ecNumber', []) for e in ec_numbers if e.get('ecNumber')]
                    ec_numbers = [item for sublist in ec_numbers for item in sublist]
                return ec_numbers
            elif resp.status_code == 404:
                # UniProt ID 不存在，尝试通过基因名或其他方式查找
                return []
            elif resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
            else:
                return []
        except Exception as e:
            time.sleep(2)
    return []

def main():
    # 加载需要查询的 UniProt ID 列表
    with open('dataset_building/external_data/uniprot_ids_need_ec.json', 'r') as f:
        uniprot_ids = json.load(f)
    
    print(f"需要查询的 UniProt ID 数量: {len(uniprot_ids)}")
    
    # 批量查询
    results = {}
    batch_size = 100
    
    for i, up_id in enumerate(uniprot_ids):
        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{len(uniprot_ids)}")
        
        ec_numbers = query_uniprot_ec(up_id)
        if ec_numbers:
            # 取第一个 EC 号作为主要分类
            results[up_id] = ec_numbers[0]
        
        # 避免 API 限流
        if (i + 1) % 10 == 0:
            time.sleep(0.2)
    
    print(f"\n✅ 查询完成!")
    print(f"成功获取 EC 号: {len(results)} / {len(uniprot_ids)}")
    
    # 保存查询结果
    with open('dataset_building/external_data/uniprot_ec_mapping.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 结果已保存到: external_data/uniprot_ec_mapping.json")
    
    # 显示示例
    if results:
        print("\n=== 查询结果示例 ===")
        for up_id, ec in list(results.items())[:5]:
            print(f"  {up_id} → {ec}")

if __name__ == "__main__":
    main()