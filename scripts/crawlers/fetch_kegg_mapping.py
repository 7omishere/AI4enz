#!/usr/bin/env python3
"""
通过 KEGG REST API 获取 EC→KO→UniProt 映射
"""
import requests
import json
import time
import pandas as pd
from collections import defaultdict

def query_kegg_ec_to_ko(ec_number, max_retries=3):
    """查询 EC 号对应的 KO 列表"""
    # KEGG EC 号格式：1.1.1.1 → 1+1+1+1
    kegg_ec = ec_number.replace('.', '+')
    url = f"https://rest.kegg.jp/link/ko/{kegg_ec}"
    
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                lines = resp.text.strip().split('\n')
                kos = []
                for line in lines:
                    if line.startswith('ko:'):
                        ko = line.split('\t')[0].replace('ko:', '')
                        kos.append(ko)
                return kos
            elif resp.status_code == 504:  # Gateway Timeout
                time.sleep(2 * (attempt + 1))
            else:
                return []
        except Exception as e:
            time.sleep(1)
    return []

def query_ko_to_uniprot(ko_list, max_retries=3):
    """查询 KO 对应的 UniProt 蛋白"""
    uniprots = set()
    for ko in ko_list:
        url = f"https://rest.kegg.jp/link/uniprot/{ko}"
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    lines = resp.text.strip().split('\n')
                    for line in lines:
                        if 'up:' in line:
                            parts = line.split('\t')
                            if len(parts) == 2:
                                uniprot = parts[1].replace('up:', '')
                                uniprots.add(uniprot)
                elif resp.status_code == 504:
                    time.sleep(2)
                time.sleep(0.1)  # 避免过快
            except Exception as e:
                time.sleep(1)
        time.sleep(0.2)  # KO之间延迟
    return list(uniprots)

def main():
    print("=== KEGG EC→UniProt 映射获取 ===\n")
    
    # 1. 获取需要查询的 EC 号
    brenda = pd.read_parquet('processed/oxidoreductase/brenda_kcat_full.parquet', 
                             columns=['ec_number'])
    all_ecs = set(brenda['ec_number'].str.strip().unique())
    print(f"BRENDA 独特 EC 号: {len(all_ecs)}")
    
    # 2. 加载现有的 SwissProt 映射
    with open('processed/oxidoreductase/swissprot_complete_mapping.json', 'r') as f:
        sprot = json.load(f)
    sprot_ec_to_uniprot = sprot['ec_to_uniprot']
    existing_ecs = set(sprot_ec_to_uniprot.keys())
    print(f"SwissProt 已有的 EC: {len(existing_ecs)}")
    
    # 找出需要 KEGG 补充的 EC
    ecs_to_query = [ec for ec in all_ecs if ec not in existing_ecs and ec.startswith('1.')]
    print(f"需要 KEGG 查询的 EC: {len(ecs_to_query)}")
    
    # 保存已有映射
    full_ec_to_uniprot = dict(sprot_ec_to_uniprot)
    
    # 3. 批量查询 KEGG
    print(f"\n开始 KEGG 查询 (共 {len(ecs_to_query)} 个 EC)...")
    new_mappings = {}
    
    for i, ec in enumerate(ecs_to_query):
        if (i+1) % 10 == 0:
            print(f"  进度: {i+1}/{len(ecs_to_query)} (已获取 {len(new_mappings)} 个新映射)")
        
        # 查询 EC → KO
        kos = query_kegg_ec_to_ko(ec)
        if not kos:
            time.sleep(0.3)
            continue
        
        # 查询 KO → UniProt
        uniprots = query_ko_to_uniprot(kos[:10])  # 只查询前10个KO避免过多请求
        
        if uniprots:
            new_mappings[ec] = uniprots
            full_ec_to_uniprot[ec] = uniprots
        
        # 每5个EC后长时间延迟，避免被限制
        if (i+1) % 5 == 0:
            time.sleep(1)
    
    print(f"\n=== 查询完成 ===")
    print(f"新增 KEGG 映射: {len(new_mappings)} 个 EC")
    
    # 4. 保存结果
    output = {
        'ec_to_uniprot': full_ec_to_uniprot,
        'kegg_only_mappings': new_mappings
    }
    
    with open('processed/oxidoreductase/kegg_ec_mapping.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✅ 保存到: processed/oxidoreductase/kegg_ec_mapping.json")
    
    # 显示样本
    print(f"\n新映射样本:")
    for ec, uniprots in list(new_mappings.items())[:5]:
        print(f"  {ec}: {uniprots[:3]}...")

if __name__ == "__main__":
    main()