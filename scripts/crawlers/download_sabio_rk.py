#!/usr/bin/env python3
"""
分批下载 SABIO-RK 数据并增量保存
支持断点续传，避免超时丢失数据
"""

import requests
import json
import time
import pandas as pd
import os
import re
from datetime import datetime

OUTPUT_DIR = "/home/domi/AI4enz/dataset_building/processed/sabio_rk"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_URL = "https://sabiork.h-its.org/export-api/sabio/kinlaw-entry/json"

# EC类别配置
EC_CLASSES = [
    ("1.*", "oxidoreductases"),
    ("2.*", "transferases"),
    ("3.*", "hydrolases"),
    ("4.*", "lyases"),
    ("5.*", "isomerases"),
    ("6.*", "ligases"),
]

def extract_parameters(entries):
    """从SABIO条目中提取参数数据"""
    records = []
    
    for entry in entries:
        enzyme = entry.get('enzyme_description', {})
        ec = enzyme.get('ec_number', '')
        
        proteins = enzyme.get('proteins', [])
        uniprot_ids = []
        for protein in proteins:
            uid = protein.get('uniprot_id', '')
            if uid:
                # 匹配 UniProt ID 格式 (单个大写字母 + 5位数字)
                matches = re.findall(r'([A-Z]\d{5})', str(uid))
                uniprot_ids.extend(matches)
        
        if not uniprot_ids:
            continue
        
        # 去重 uniprot_ids
        uniprot_ids = list(set(uniprot_ids))
        
        general = entry.get('general', {})
        conditions = entry.get('experimental_conditions', {})
        kineticlaw = entry.get('kineticlaw', {})
        
        base_record = {
            'ec_number': ec,
            'organism': general.get('organism', {}).get('name') if isinstance(general.get('organism'), dict) else str(general.get('organism', '')),
            'enzyme_name': enzyme.get('enzyme_name'),
            'ph': conditions.get('envvar_ph', {}).get('start_value') if isinstance(conditions.get('envvar_ph'), dict) else conditions.get('envvar_ph'),
            'temperature': conditions.get('envvar_temperature', {}).get('start_value') if isinstance(conditions.get('envvar_temperature'), dict) else conditions.get('envvar_temperature'),
            'publication_id': entry.get('publication', {}).get('pubmed_id'),
            'sabio_id': str(entry.get('id', '')),
        }
        
        params = kineticlaw.get('parameter', [])
        for param in params:
            param_type = param.get('parameter_type', {})
            if isinstance(param_type, dict):
                pname = param_type.get('name', '')
            else:
                pname = str(param_type)
            
            value = param.get('start_value')
            
            unit_data = param.get('unit', {})
            if isinstance(unit_data, dict):
                unit = unit_data.get('name', '')
            else:
                unit = str(unit_data)
            
            if value is None:
                continue
            
            # 每个uniprot创建一个记录
            for uid in uniprot_ids:
                record = base_record.copy()
                record['uniprot_id'] = uid
                record['parameter_type'] = pname
                record['value'] = float(value)
                record['unit'] = unit
                records.append(record)
    
    return records

def download_ec_class(ec_pattern, name, rows=100, max_batches=300, checkpoint_interval=50):
    """下载单个EC类别的数据，支持检查点"""
    checkpoint_file = os.path.join(OUTPUT_DIR, f"checkpoint_{name}.json")
    
    # 检查是否有检查点
    start_batch = 0
    all_records = []
    
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
            all_records = data.get('records', [])
            start_batch = data.get('last_batch', 0) + 1
        print(f"  从检查点恢复: 批次 {start_batch}, 已有 {len(all_records)} 条记录")
    
    total = 0
    
    for batch in range(start_batch, max_batches):
        url = f"{BASE_URL}?q=EcNumber:{ec_pattern}&rows={rows}&start={batch * rows}"
        
        try:
            resp = requests.get(url, timeout=60)
            
            if resp.status_code == 429:
                print(f"  批次 {batch}: 速率限制，等待 60s...")
                time.sleep(60)
                batch -= 1
                continue
            
            if resp.status_code != 200:
                print(f"  批次 {batch}: 错误 {resp.status_code}")
                break
            
            data = resp.json()
            entries = data.get('data', [])
            
            if batch == 0:
                total = data.get('meta', {}).get('total_count', 0)
                print(f"\n{name}: 预计 {total:,} 条记录")
            
            if not entries:
                print(f"  批次 {batch}: 完成")
                break
            
            # 提取参数
            records = extract_parameters(entries)
            all_records.extend(records)
            
            # 进度显示
            if (batch + 1) % 10 == 0:
                print(f"  批次 {batch+1}/{max_batches}: +{len(records)} 条 (累计 {len(all_records)})")
            
            # 保存检查点
            if (batch + 1) % checkpoint_interval == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump({'last_batch': batch, 'records': all_records}, f)
                print(f"  💾 检查点已保存")
            
            time.sleep(1.2)
            
        except Exception as e:
            print(f"  批次 {batch}: 错误 - {e}")
            time.sleep(5)
    
    return all_records

def main():
    print("=" * 70)
    print("SABIO-RK 数据下载")
    print(f"开始时间: {datetime.now()}")
    print("=" * 70)
    
    all_data = []
    
    for ec_pattern, name in EC_CLASSES:
        print(f"\n>>> 下载 {name} (EC {ec_pattern})...")
        records = download_ec_class(ec_pattern, name)
        
        if records:
            df = pd.DataFrame(records)
            print(f"    获取 {len(df):,} 条参数记录")
            print(f"    参数类型分布:")
            print(df['parameter_type'].value_counts().to_string().replace('\n', '\n    '))
            all_data.append(df)
    
    # 合并保存
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        output_path = os.path.join(OUTPUT_DIR, "sabio_rk_all.parquet")
        combined.to_parquet(output_path, index=False)
        
        # 分别保存
        for ptype in ['kcat', 'Km', 'kcat/Km']:
            subset = combined[combined['parameter_type'] == ptype]
            if len(subset) > 0:
                subset.to_parquet(os.path.join(OUTPUT_DIR, f"sabio_rk_{ptype.replace('/', '_')}.parquet"), index=False)
        
        # 清理检查点
        for _, name in EC_CLASSES:
            cp = os.path.join(OUTPUT_DIR, f"checkpoint_{name}.json")
            if os.path.exists(cp):
                os.remove(cp)
        
        print(f"\n✅ 完成!")
        print(f"   总记录: {len(combined):,}")
        print(f"   保存到: {output_path}")
    else:
        print("未获取到数据")

if __name__ == "__main__":
    main()