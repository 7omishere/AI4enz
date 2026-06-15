import gzip
import json
import re
from collections import defaultdict
import time

print("=== 解析 Swiss-Prot (修正格式) ===\n")
start_time = time.time()

sprot_file = 'uniprot_sprot.dat.gz'
ec_to_uniprot = defaultdict(set)
organism_to_uniprot = defaultdict(set)
ec_organism_to_uniprot = defaultdict(set)

current_entry = {'accession': None, 'organism': None, 'ec_numbers': []}
entries_parsed = 0
ec_entries = 0

print("解析中... (需要几分钟)")

with gzip.open(sprot_file, 'rt', encoding='utf-8') as f:
    for line in f:
        line = line.rstrip()
        
        if line.startswith('AC '):
            parts = line[3:].strip().rstrip(';').split(';')
            current_entry['accession'] = parts[0].strip()
            
        elif line.startswith('OS '):
            org = line[3:].strip().rstrip('.')
            if org:
                current_entry['organism'] = org
                
        elif line.startswith('DE ') and 'EC=' in line:
            # EC 号在 DE 行中: DE   RecName: EC=1.1.1.1;
            ec_matches = re.findall(r'EC=(\d+\.\d+\.\d+\.\d+)', line)
            for ec in ec_matches:
                if ec not in current_entry['ec_numbers']:
                    current_entry['ec_numbers'].append(ec)
                        
        elif line.startswith('//'):
            if current_entry['accession']:
                accession = current_entry['accession']
                
                if current_entry['ec_numbers']:
                    ec_entries += 1
                    for ec in current_entry['ec_numbers']:
                        ec_to_uniprot[ec].add(accession)
                        if current_entry['organism']:
                            ec_organism_to_uniprot[(ec, current_entry['organism'])].add(accession)
                
                if current_entry['organism']:
                    organism_to_uniprot[current_entry['organism']].add(accession)
                
                entries_parsed += 1
                
                if entries_parsed % 100000 == 0:
                    print(f"  已解析: {entries_parsed} 条目, {ec_entries} 条含EC")
            
            current_entry = {'accession': None, 'organism': None, 'ec_numbers': []}

elapsed = time.time() - start_time
print(f"\n=== 解析完成 (耗时 {elapsed:.1f}s) ===")
print(f"总条目: {entries_parsed}")
print(f"含 EC 号条目: {ec_entries}")
print(f"EC→UniProt 映射: {len(ec_to_uniprot)} 个 EC")

# 筛选氧化还原酶
oxidored_ec = {ec: ids for ec, ids in ec_to_uniprot.items() if ec.startswith('1.')}
print(f"氧化还原酶 (1.x.x.x): {len(oxidored_ec)} 个 EC")
print(f"氧化还原酶 UniProt 蛋白: {sum(len(ids) for ids in oxidored_ec.values())} 个")

# 保存映射
mapping = {
    'ec_to_uniprot': {ec: list(ids) for ec, ids in oxidored_ec.items()},
    'ec_organism_to_uniprot': {f"{k[0]}|{k[1]}": list(v) for k, v in ec_organism_to_uniprot.items() if k[0].startswith('1.')}
}

with open('processed/oxidoreductase/swissprot_complete_mapping.json', 'w') as f:
    json.dump(mapping, f, indent=2)

print(f"\n✅ 保存到: processed/oxidoreductase/swissprot_complete_mapping.json")

# 样本
print(f"\n样本 EC→UniProt (前10):")
for i, ec in enumerate(sorted(oxidored_ec.keys())[:10]):
    uniprots = list(oxidored_ec[ec])[:3]
    print(f"  {ec}: {uniprots}...")