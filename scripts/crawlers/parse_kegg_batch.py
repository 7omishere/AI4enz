#!/usr/bin/env python3
"""
分批获取 KEGG EC→UniProt 映射
保存进度，支持断点续传
"""
import requests
import json
import time
import sys

def main():
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    
    # 加载数据
    with open('processed/oxidoreductase/ecs_need_kegg.json', 'r') as f:
        ec_list = json.load(f)['ec_list']
    
    with open('processed/oxidoreductase/swissprot_complete_mapping.json', 'r') as f:
        sprot = json.load(f)
    full_mapping = dict(sprot['ec_to_uniprot'])
    
    # 分批: 每批10个EC
    batch_size = 10
    start_idx = batch * batch_size
    end_idx = min(start_idx + batch_size, len(ec_list))
    
    if start_idx >= len(ec_list):
        print(f"全部完成! 总计 {len(full_mapping)} 个 EC 映射")
        return
    
    batch_ecs = ec_list[start_idx:end_idx]
    print(f"批次 {batch}: 处理 EC {start_idx+1}-{end_idx}/{len(ec_list)}")
    
    new_mappings = {}
    failed = []
    
    for i, ec in enumerate(batch_ecs):
        kegg_ec = ec.replace('.', '+')
        url = f"https://rest.kegg.jp/link/ko/{kegg_ec}"
        
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code != 200:
                failed.append(ec)
                continue
                
            lines = resp.text.strip().split('\n')
            kos = [l.split('\t')[0].replace('ko:', '') for l in lines if l.startswith('ko:')]
            
            if not kos:
                failed.append(ec)
                continue
            
            uniprots = set()
            for ko in kos[:15]:
                try:
                    url2 = f"https://rest.kegg.jp/link/uniprot/{ko}"
                    resp2 = requests.get(url2, timeout=30)
                    if resp2.status_code == 200:
                        for line in resp2.text.strip().split('\n'):
                            if 'up:' in line:
                                uniprots.add(line.split('\t')[1].replace('up:', ''))
                except:
                    pass
                time.sleep(0.1)
            
            if uniprots:
                new_mappings[ec] = list(uniprots)
                full_mapping[ec] = list(uniprots)
            
            time.sleep(0.2)
            
        except Exception as e:
            failed.append(ec)
            time.sleep(0.5)
    
    # 保存进度
    with open('processed/oxidoreductase/kegg_partial.json', 'w') as f:
        json.dump({'full_mapping': full_mapping, 'new_mappings': new_mappings, 'failed': failed, 'batch': batch}, f, indent=2)
    
    print(f"批次 {batch} 完成: 新增 {len(new_mappings)}, 失败 {len(failed)}")
    print(f"总映射: {len(full_mapping)} 个 EC")
    
    # 合并到主映射文件
    with open('processed/oxidoreductase/ec_uniprot_combined.json', 'w') as f:
        json.dump(full_mapping, f, indent=2)
    
    print(f"可继续: python parse_kegg_batch.py {batch+1}")

if __name__ == "__main__":
    main()