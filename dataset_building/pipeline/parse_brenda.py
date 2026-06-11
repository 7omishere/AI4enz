#!/usr/bin/env python3
"""
解析 BRENDA 数据库，提取 kcat 和 Km 数据
BRENDA 格式:
  ID     - EC号
  PR     - 蛋白/生物体信息
  TN     - Turnover Number (kcat)
  KM     - Km value
"""

import re
import pandas as pd
from pathlib import Path
from collections import defaultdict

BRENDA_FILE = Path(__file__).parent.parent / "brenda_2026_1.txt"
OUTPUT_DIR = Path(__file__).parent.parent / "processed" / "oxidoreductase"

def parse_brenda():
    """解析 BRENDA 文件，提取 kcat 数据"""
    print("=== 解析 BRENDA 数据库 ===\n")
    
    with open(BRENDA_FILE, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # 按酶条目分割
    entries = re.split(r'\n///\n', content)
    print(f"总酶条目: {len(entries)}")
    
    records = []
    current_ec = None
    current_organisms = {}  # 蛋白质编号 -> 生物体
    
    for entry in entries:
        lines = entry.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            # 获取 EC 号
            if line.startswith('ID\t'):
                current_ec = line[3:].strip()
            
            # 解析蛋白质/生物体信息
            elif line.startswith('PR\t'):
                # 格式: PR	#编号# 生物体名 <条件>
                match = re.match(r'PR\s*#(\d+)#\s*(.+?)(?:\s*<|$)', line)
                if match:
                    protein_num = match.group(1)
                    organism = match.group(2).strip()
                    # 清理生物体名
                    organism = re.sub(r'\s*\(.*?\)', '', organism)  # 移除括号内容
                    current_organisms[protein_num] = organism
            
            # 解析 Turnover Number (kcat)
            elif line.startswith('TN\t'):
                # 格式: TN	#蛋白编号# 数值 {底物} (pH X.X) <文献>
                # 数值可能是: "1", "1.5", "1e-3", "2-8" (范围)
                match = re.match(r'TN\s*#(\d+)#\s*([\d.e+-]+)\s*(?:\{([^}]*)\})?\s*(?:\(([^)]*)\))?', line)
                if match and current_ec:
                    protein_num = match.group(1)
                    raw_value = match.group(2)
                    
                    # 处理范围值（如 "2-8"），取第一个值
                    if '-' in raw_value:
                        raw_value = raw_value.split('-')[0]
                    
                    # 处理科学计数法
                    try:
                        kcat_value = float(raw_value)
                    except ValueError:
                        # 跳过无法解析的值
                        continue
                    
                    substrate = match.group(3) or ''
                    conditions = match.group(4) or ''
                    organism = current_organisms.get(protein_num, '')
                    
                    # 提取 pH (robust parsing)
                    ph = None
                    ph_match = re.search(r'pH\s*([\d.]+)', conditions)
                    if ph_match:
                        try:
                            ph = float(ph_match.group(1))
                        except:
                            pass
                    
                    # 提取温度 (robust parsing)
                    temp = None
                    temp_match = re.search(r'([\d.]+)\s*°?[Cc]', conditions)
                    if temp_match:
                        try:
                            temp = float(temp_match.group(1))
                        except:
                            pass
                    
                    records.append({
                        'ec_number': current_ec,
                        'protein_num': protein_num,
                        'organism': organism,
                        'kcat_value': kcat_value,
                        'pkcat': kcat_value,
                        'substrate': substrate,
                        'conditions': conditions,
                        'ph': ph,
                        'temperature': temp,
                        'source': 'brenda_2026'
                    })
    
    df = pd.DataFrame(records)
    print(f"提取到 kcat 记录: {len(df)}")
    
    return df

def filter_oxidoreductases(df):
    """过滤氧化还原酶 (EC 1.x.x.x)"""
    df_ox = df[df['ec_number'].str.startswith('1.', na=False)].copy()
    print(f"氧化还原酶 (1.x.x.x): {len(df_ox)}")
    return df_ox

def save_results(df):
    """保存结果"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 完整数据
    df.to_parquet(OUTPUT_DIR / 'brenda_kcat_full.parquet', index=False)
    print(f"已保存: brenda_kcat_full.parquet")
    
    # 统计分析
    print(f"\n【EC号分布 (前10)】")
    print(df['ec_number'].value_counts().head(10).to_string())
    
    print(f"\n【kcat 值统计】")
    print(df['kcat_value'].describe())

def main():
    df = parse_brenda()
    df_ox = filter_oxidoreductases(df)
    save_results(df_ox)
    
    return df_ox

if __name__ == "__main__":
    main()