#!/usr/bin/env python3
"""
UniProt EC号批量查询脚本
通过UniProt REST API获取蛋白质EC号注释
"""

import requests
import time
import json
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os

def query_uniprot_batch(uniprot_ids: list, batch_size: int = 15) -> dict:
    """
    批量查询UniProt ID的EC号
    
    Args:
        uniprot_ids: UniProt ID列表
        batch_size: 每批查询数量 (UniProt建议15-20)
    
    Returns:
        dict: {uniprot_id: ec_numbers}
    """
    results = {}
    
    for i in range(0, len(uniprot_ids), batch_size):
        batch = uniprot_ids[i:i+batch_size]
        ids_str = ','.join(batch)
        
        url = f"https://rest.uniprot.org/uniprotkb/stream?query= accession:{ids_str}&fields=accession,ec_length&format=json"
        
        try:
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                data = response.json()
                for entry in data.get('results', []):
                    accession = entry.get('primaryAccession')
                    ec_numbers = entry.get('ecReviewEvidence', [])
                    
                    if ec_numbers:
                        # 取第一个EC号 (主要催化活性)
                        ec_list = []
                        for ec in ec_numbers:
                            if ec.get('evidence'):
                                for ev in ec['evidence']:
                                    ec_val = ev.get('value')
                                    if ec_val and ec_val != 'N/A':
                                        ec_list.append(ec_val)
                                        break
                        
                        if ec_list:
                            # 取最具体的EC号 (最后一个点后的数字)
                            valid_ec = [e for e in ec_list if e.count('.') == 3]
                            if valid_ec:
                                results[accession] = valid_ec[0]
                            elif ec_list:
                                results[accession] = ec_list[0]
            elif response.status_code == 429:
                # Rate limiting，等待后重试
                time.sleep(5)
                response = requests.get(url, timeout=60)
                if response.status_code == 200:
                    data = response.json()
                    # 处理数据...
                    
        except Exception as e:
            print(f"Error querying batch {i//batch_size}: {e}")
        
        # UniProt API 建议每秒不超过5个请求
        time.sleep(0.25)
    
    return results


def query_uniprot_single(uniprot_id: str) -> str | None:
    """查询单个UniProt ID的EC号"""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    
    try:
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            # 方法1: 从 comments -> CATALYTIC ACTIVITY -> reaction.ecNumber 获取
            comments = data.get('comments', [])
            for comment in comments:
                if comment.get('commentType') == 'CATALYTIC ACTIVITY':
                    reaction = comment.get('reaction', {})
                    ec_numbers_raw = reaction.get('ecNumber', [])
                    
                    for ec_raw in ec_numbers_raw:
                        # EC号可能是字符串或字符数组
                        if isinstance(ec_raw, list):
                            ec_val = ''.join(ec_raw)
                        else:
                            ec_val = str(ec_raw)
                        
                        # 验证EC号格式 x.x.x.x
                        parts = ec_val.split('.')
                        if len(parts) == 4 and all(p.isdigit() or p == '' for p in parts):
                            return ec_val
            
            # 方法2: 从 ecReviewEvidence 获取 (部分蛋白)
            ec_evidence = data.get('ecReviewEvidence', [])
            for ec in ec_evidence:
                if ec.get('evidence'):
                    for ev in ec['evidence']:
                        ec_val = ev.get('value')
                        if ec_val and ec_val.count('.') == 3:
                            return ec_val
            
        elif response.status_code == 404:
            return None
        elif response.status_code == 429:
            time.sleep(2)
            return query_uniprot_single(uniprot_id)  # 重试
            
    except Exception as e:
        print(f"Error querying {uniprot_id}: {e}")
    
    return None


def query_with_retry(uniprot_ids: list, max_workers: int = 5, max_retries: int = 3) -> dict:
    """
    并行查询多个UniProt ID
    
    Args:
        uniprot_ids: UniProt ID列表
        max_workers: 并行线程数
        max_retries: 最大重试次数
    
    Returns:
        dict: {uniprot_id: ec_number}
    """
    results = {}
    failed_ids = []
    
    def worker(uid):
        for attempt in range(max_retries):
            try:
                ec = query_uniprot_single(uid)
                return uid, ec
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return uid, None
        return uid, None
    
    print(f"📡 开始查询 {len(uniprot_ids)} 个UniProt ID...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, uid): uid for uid in uniprot_ids}
        
        completed = 0
        for future in as_completed(futures):
            uid, ec = future.result()
            if ec:
                results[uid] = ec
            completed += 1
            
            if completed % 100 == 0:
                print(f"   进度: {completed}/{len(uniprot_ids)} ({100*completed/len(uniprot_ids):.1f}%)")
    
    success_rate = len(results) / len(uniprot_ids) * 100
    print(f"✅ 查询完成: {len(results)}/{len(uniprot_ids)} 成功 ({success_rate:.1f}%)")
    
    return results


def main():
    # 读取需要查询的UniProt ID
    input_file = "/home/domi/AI4enz/dataset_building/uniprots_to_query.txt"
    output_file = "/home/domi/AI4enz/dataset_building/uniprotprot/uniprot_ec_mapping.json"
    
    if not os.path.exists(input_file):
        print(f"❌ 文件不存在: {input_file}")
        print("需要先运行分析脚本生成ID列表")
        sys.exit(1)
    
    with open(input_file, 'r') as f:
        uniprot_ids = [line.strip() for line in f if line.strip()]
    
    print(f"📋 加载了 {len(uniprot_ids)} 个UniProt ID")
    
    # 查询EC号
    ec_mapping = query_with_retry(uniprot_ids, max_workers=3)
    
    # 保存结果
    with open(output_file, 'w') as f:
        json.dump(ec_mapping, f, indent=2)
    
    print(f"💾 结果已保存到: {output_file}")
    
    # 打印统计
    with_ec = sum(1 for v in ec_mapping.values() if v)
    print(f"📊 统计: {with_ec} 个ID有EC号 ({100*with_ec/len(uniprot_ids):.1f}%)")


if __name__ == "__main__":
    main()