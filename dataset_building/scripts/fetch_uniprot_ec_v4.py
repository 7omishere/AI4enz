#!/usr/bin/env python3
"""
UniProt EC号批量查询脚本 v4
修复字符数组问题，并覆盖所有无EC号的样本
"""

import requests
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os

def query_uniprot_single(uniprot_id: str) -> tuple[str, str | None]:
    """查询单个UniProt ID的EC号 - 增强版"""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    
    try:
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            # 遍历所有comments
            comments = data.get('comments', [])
            
            for comment in comments:
                if comment.get('commentType') == 'CATALYTIC ACTIVITY':
                    reaction = comment.get('reaction', {})
                    ec_raw = reaction.get('ecNumber')
                    
                    if ec_raw:
                        # 处理字符串或字符数组
                        if isinstance(ec_raw, list):
                            ec_number = ''.join(ec_raw)
                        elif isinstance(ec_raw, str):
                            ec_number = ec_raw
                        else:
                            ec_number = str(ec_raw)
                        
                        # 验证格式 x.x.x.x
                        parts = ec_number.split('.')
                        if len(parts) == 4 and all(p.isdigit() for p in parts):
                            return uniprot_id, ec_number
                            
            # 备选：从 ecReviewEvidence 获取
            ec_evidence = data.get('ecReviewEvidence', [])
            for ec in ec_evidence:
                if ec.get('value') and isinstance(ec.get('value'), str):
                    ec_val = ec.get('value')
                    if ec_val.count('.') == 3:
                        return uniprot_id, ec_val
            
            # 备选：从 features 获取
            features = data.get('features', [])
            for feat in features:
                if feat.get('type') == 'Catalytic activity':
                    desc = feat.get('description', '')
                    if desc and desc.count('.') == 3:
                        return uniprot_id, desc
            
        elif response.status_code == 404:
            return uniprot_id, None
        elif response.status_code == 429:
            time.sleep(3)
            return query_uniprot_single(uniprot_id)
            
    except Exception as e:
        pass
    
    return uniprot_id, None


def query_batch(uniprot_ids: list, max_workers: int = 5) -> dict:
    """并行查询"""
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
                success = len(results)
                rate = 100 * success / completed if completed > 0 else 0
                print(f"   进度: {completed}/{len(uniprot_ids)} | 成功: {success} ({rate:.1f}%)")
            
            time.sleep(0.05)  # 稍微降低速率避免限流
    
    success_rate = len(results) / len(uniprot_ids) * 100 if uniprot_ids else 0
    print(f"✅ 查询完成: {len(results)}/{len(uniprot_ids)} 成功 ({success_rate:.1f}%)")
    
    return results


def main():
    input_file = "/home/domi/AI4enz/dataset_building/uniprots_to_query.txt"
    output_file = "/home/domi/AI4enz/dataset_building/uniprotprot/uniprot_ec_mapping_complete.json"
    
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
    
    # 打印前10个结果样例
    print(f"\n📝 结果样例:")
    for i, (uid, ec) in enumerate(list(ec_mapping.items())[:10]):
        print(f"   {uid}: {ec}")


if __name__ == "__main__":
    main()