#!/usr/bin/env python3
"""
UniProt EC号批量查询脚本 v3
直接正确的API调用方式
"""

import requests
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os

def query_uniprot_single(uniprot_id: str) -> tuple[str, str | None]:
    """查询单个UniProt ID的EC号"""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    
    try:
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            # 获取所有comments
            comments = data.get('comments', [])
            
            for comment in comments:
                if comment.get('commentType') == 'CATALYTIC ACTIVITY':
                    reaction = comment.get('reaction', {})
                    ec_number = reaction.get('ecNumber')
                    
                    if ec_number and isinstance(ec_number, str):
                        # 验证格式 x.x.x.x
                        parts = ec_number.split('.')
                        if len(parts) == 4 and all(p.isdigit() for p in parts):
                            return uniprot_id, ec_number
            
        elif response.status_code == 404:
            return uniprot_id, None
        elif response.status_code == 429:
            time.sleep(3)
            return query_uniprot_single(uniprot_id)
            
    except Exception as e:
        return uniprot_id, None
    
    return uniprot_id, None


def query_batch(uniprot_ids: list, max_workers: int = 5) -> dict:
    """并行查询多个UniProt ID"""
    results = {}
    
    print(f"📡 开始查询 {len(uniprot_ids)} 个UniProt ID...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(query_uniprot_single, uid): uid for uid in uniprot_ids}
        
        completed = 0
        for future in as_completed(futures):
            uid, ec = future.result()
            if ec:
                results[uid] = ec
            completed += 1
            
            if completed % 100 == 0:
                print(f"   进度: {completed}/{len(uniprot_ids)} ({100*completed/len(uniprot_ids):.1f}%)")
            
            time.sleep(0.1)  # 避免过快
    
    success_rate = len(results) / len(uniprot_ids) * 100 if uniprot_ids else 0
    print(f"✅ 查询完成: {len(results)}/{len(uniprot_ids)} 成功 ({success_rate:.1f}%)")
    
    return results


def main():
    input_file = "/home/domi/AI4enz/dataset_building/uniprots_to_query.txt"
    output_file = "/home/domi/AI4enz/dataset_building/uniprotprot/uniprot_ec_mapping.json"
    
    if not os.path.exists(input_file):
        print(f"❌ 文件不存在: {input_file}")
        sys.exit(1)
    
    with open(input_file, 'r') as f:
        uniprot_ids = [line.strip() for line in f if line.strip()]
    
    print(f"📋 加载了 {len(uniprot_ids)} 个UniProt ID")
    
    # 查询EC号
    ec_mapping = query_batch(uniprot_ids, max_workers=5)
    
    # 保存结果
    with open(output_file, 'w') as f:
        json.dump(ec_mapping, f, indent=2)
    
    print(f"💾 结果已保存到: {output_file}")
    
    # 打印统计
    with_ec = sum(1 for v in ec_mapping.values() if v)
    print(f"📊 统计: {with_ec} 个ID有EC号 ({100*with_ec/len(uniprot_ids):.1f}%)")


if __name__ == "__main__":
    main()