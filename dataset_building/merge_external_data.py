"""
merge_external_data.py — 将 CatPred-DB、OED、SKiD 数据合并到现有数据集

官方数据源:
- CatPred-DB: maranasgroup AWS S3 (Nature Communications 2025) — kcat + Ki
- OED: openenzymedb-api.platform.moleculemaker.org (NAR 2025) — kcat + Km
- SKiD: Zenodo DOI 10.5281/zenodo.15355031 (Scientific Data 2025) — kcat + Km + 3D
"""
import pandas as pd, numpy as np, hashlib, json, os, warnings
from pathlib import Path

warnings.filterwarnings('ignore')
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey

EXTERNAL_DIR = Path(__file__).parent / 'external_data'
OUTPUT_DIR = Path(__file__).parent / 'processed' / 'oxidoreductase'
EXISTING_PARQUET = OUTPUT_DIR / 'unified_metadata.parquet'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def s2inchikey(smiles):
    if pd.isna(smiles) or not str(smiles).strip(): return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        return MolToInchiKey(mol) if mol else None
    except: return None

def hash_seq(seq):
    return hashlib.sha256(str(seq).upper().encode()).hexdigest()[:16]

def load_existing():
    if EXISTING_PARQUET.exists():
        df = pd.read_parquet(EXISTING_PARQUET)
        print(f"现有数据集: {len(df)} records, {df['uniprot_id'].nunique()} UniProts")
        return df
    return None

def base_record(sid, inchikey, uniprot, ec, source_db, measurement_type,
                has_kcat=False, log_kcat=None, kcat_val=None,
                pkd=None, quality_weight=0.0, w_mult=0.5, pdb_id=None,
                has_structure=False, temperature=None, ph_val=None,
                extra=None):
    r = {
        'sample_id': sid,
        'protein_seq_hash': None, 'ligand_inchikey': inchikey,
        'uniprot_id': uniprot, 'pdb_id': pdb_id, 'source_db': source_db,
        'ec_numbers': str(ec) if ec else '', 'cofactors': None,
        'protein_name': None, 'reviewed': True,
        'pkd_aligned': pkd, 'pkd_raw': pkd,
        'measurement_type': measurement_type,
        'quality_weight': quality_weight, 'w_multiplier': w_mult,
        'is_censored': False, 'n_measurements': 1, 'pkd_std': np.nan,
        'has_kcat': has_kcat, 'kcat_source': source_db if has_kcat else None,
        'kcat_median_s': kcat_val, 'log_kcat_median': log_kcat,
        'kcat_outlier': False,
        'has_structure': has_structure, 'has_binding_site': False,
        'has_domain_annotation': False, 'n_domains': 0,
        'cofactor_domain_types': None, 'domains_json': None,
        'temperature': temperature, 'ph_val': ph_val,
        'bdb_n_km': 0, 'bdb_n_kcat': 1 if has_kcat else 0, 'bdb_n_kcatkm': 0,
        'bdb_km_median_uM': np.nan, 'bdb_kcat_median_s': kcat_val,
        'bdb_kcatkm_median_M1s1': np.nan,
        'n_km_sabio': 0, 'n_kcat_sabio': 0, 'n_kcatkm_sabio': 0,
        'km_median_uM_sabio': np.nan, 'kcat_median_s_sabio': np.nan,
        'kcatkm_median_M1s1_sabio': np.nan,
        'up_has_kinetics': has_kcat, 'up_n_kcat': 1 if has_kcat else 0,
        'pkd_corrected': pkd, 'correction_source': source_db if pkd else None,
        'split': None,
    }
    if extra: r.update(extra)
    return r

# ─── CatPred ───
def process_catpred():
    records = []
    kcat_df = pd.read_csv(EXTERNAL_DIR/'data/Baseline/kcat/kcat-random_trainval.csv')
    ki_df = pd.read_csv(EXTERNAL_DIR/'data/Baseline/ki/ki-random_trainval.csv')

    skipped = 0
    for _, row in kcat_df.iterrows():
        ik = s2inchikey(row['reactant_smiles'])
        if not ik: skipped += 1; continue
        records.append(base_record(
            f"cp_kcat_{_}", ik, row['uniprot'], row.get('ec'), 'catpred', 'kcat_only',
            has_kcat=True, log_kcat=row['log10_value'], kcat_val=row['value'],
            has_structure=bool(pd.notna(row.get('pdbpath')) and str(row.get('pdbpath','')).strip()),
            temperature=row.get('temperature'), ph_val=row.get('ph'),
        ))
    print(f"CatPred kcat: {len(records)-skipped} valid (+{skipped} bad SMILES)")

    n = len(records)
    for _, row in ki_df.iterrows():
        ik = s2inchikey(row['substrate_smiles'])
        if not ik: skipped += 1; continue
        pkd = -row['log10_value']  # pKi
        records.append(base_record(
            f"cp_ki_{_}", ik, row['uniprot'], row.get('ec'), 'catpred', 'Ki',
            pkd=pkd, quality_weight=0.7, w_mult=1.0,
            has_structure=bool(pd.notna(row.get('pdbpath')) and str(row.get('pdbpath','')).strip()),
            temperature=row.get('temperature'), ph_val=row.get('ph'),
        ))
    ki_count = len(records) - n
    print(f"CatPred Ki: {ki_count} valid (+{skipped-n} bad SMILES)")
    return pd.DataFrame(records)

# ─── OED ───
def process_oed():
    with open(EXTERNAL_DIR/'oed_kinetics.json') as f:
        raw = json.load(f)
    records, skipped = [], 0
    for row in raw:
        ik = s2inchikey(row.get('smiles',''))
        if not ik: skipped += 1; continue
        kv = row.get('kcat_value')
        has_kcat = kv is not None
        log_kcat = np.log10(float(kv)) if has_kcat and float(kv) > 0 else np.nan
        records.append(base_record(
            f"oed_{hashlib.md5((row.get('uniprot','')+row.get('smiles','')).encode()).hexdigest()[:10]}",
            ik, row.get('uniprot',''), row.get('ec'), 'oed', 'kinetics',
            has_kcat=has_kcat, log_kcat=log_kcat, kcat_val=float(kv) if has_kcat else np.nan,
            temperature=row.get('temperature'), ph_val=row.get('ph'),
            extra={'organism_name': row.get('organism'), 'substrate_name': row.get('substrate'),
                   'enzymetype': row.get('enzymetype')}
        ))
    print(f"OED: {len(records)} valid (+{skipped} bad SMILES)")
    return pd.DataFrame(records)

# ─── SKiD ───
def process_skid():
    xf = EXTERNAL_DIR/'SKiD_Main_dataset_v1.xlsx'
    kcat_df = pd.read_excel(xf, sheet_name='kcat_dataset')
    km_df = pd.read_excel(xf, sheet_name='Km_dataset')
    records, skipped = [], 0

    for _, row in kcat_df.iterrows():
        ik = s2inchikey(row['Substrate_SMILES'])
        if not ik: skipped += 1; continue
        pdb = str(row.get('Protein_file','')).replace('.pdb','') if pd.notna(row.get('Protein_file')) else None
        records.append(base_record(
            f"skid_{row['Entry_ID']}", ik, row['UniProt_ID'], row.get('EC_number'),
            'skid', 'kcat_only', has_kcat=True, log_kcat=row['pkcat_value'],
            kcat_val=row['kcat_value'], has_structure=pd.notna(row.get('Protein_file')),
            temperature=row.get('Temperature'), ph_val=row.get('pH'), pdb_id=pdb,
            extra={'organism_name': row.get('Organism_name'),
                   'is_mutant': str(row.get('Mutant','no')).lower()!='no',
                   'mutation_info': str(row.get('Mutation','')) if pd.notna(row.get('Mutation')) else None}
        ))
    print(f"SKiD kcat: {len(records)} valid (+{skipped} bad SMILES)")

    n = len(records)
    for _, row in km_df.iterrows():
        ik = s2inchikey(row['Substrate_SMILES'])
        if not ik: skipped += 1; continue
        pdb = str(row.get('Protein_file','')).replace('.pdb','') if pd.notna(row.get('Protein_file')) else None
        records.append(base_record(
            f"skid_km_{row['Entry_ID']}", ik, row['UniProt_ID'], row.get('EC_number'),
            'skid', 'km_only', w_mult=0.3, has_structure=pd.notna(row.get('Protein_file')),
            temperature=row.get('Temperature'), ph_val=row.get('pH'), pdb_id=pdb,
            extra={'organism_name': row.get('Organism_name'),
                   'is_mutant': str(row.get('Mutant','no')).lower()!='no',
                   'mutation_info': str(row.get('Mutation','')) if pd.notna(row.get('Mutation')) else None}
        ))
    print(f"SKiD Km: {len(records)-n} valid")
    return pd.DataFrame(records)

# ─── Merge & Split ───
def assign_split(df, seed=42):
    np.random.seed(seed)
    # 对已有 uniprot_id 的分配 split (按蛋白层级，避免数据泄漏)
    uniprots = df['uniprot_id'].dropna().unique()
    np.random.shuffle(uniprots)
    n_train = int(len(uniprots) * 0.8)
    n_val = int(len(uniprots) * 0.1)
    split_map = {}
    for u in uniprots[:n_train]: split_map[u] = 'train'
    for u in uniprots[n_train:n_train+n_val]: split_map[u] = 'val'
    for u in uniprots[n_train+n_val:]: split_map[u] = 'test'
    df['split'] = df['uniprot_id'].map(split_map).fillna('train')
    return df

def main():
    print("="*60 + "\n合并外部数据集: CatPred + OED + SKiD\n" + "="*60)
    existing = load_existing()

    catpred = process_catpred()
    oed = process_oed()
    skid = process_skid()
    all_new = pd.concat([catpred, oed, skid], ignore_index=True)

    print(f"\n新数据总计: {len(all_new)} records")
    print(f"  kcat: {all_new['has_kcat'].sum()}")
    print(f"  Ki: {(all_new['measurement_type']=='Ki').sum()}")
    print(f"  UniProts: {all_new['uniprot_id'].nunique()}")
    print(f"  InChIKeys: {all_new['ligand_inchikey'].nunique()}")

    # 去重
    all_new['_key'] = all_new['uniprot_id'].fillna('')+'|'+all_new['ligand_inchikey'].fillna('')+'|'+all_new['measurement_type']
    if existing is not None:
        existing['_key'] = existing['uniprot_id'].fillna('')+'|'+existing['ligand_inchikey'].fillna('')+'|'+existing['measurement_type']
        dup = all_new['_key'].isin(set(existing['_key'])).sum()
        new_unique = all_new[~all_new['_key'].isin(set(existing['_key']))].copy()
        print(f"  与现有重复: {dup}, 真正新增: {len(new_unique)}")
        merged = pd.concat([existing, new_unique.drop(columns=['_key'])], ignore_index=True)
        merged = merged.drop(columns=['_key'], errors='ignore')
    else:
        merged = all_new.drop(columns=['_key'])
        new_unique = all_new.copy()

    merged = assign_split(merged)

    # 清洗 temperature 和 ph_val
    def clean_temp(v):
        if pd.isna(v): return np.nan
        if isinstance(v, (int, float)): return float(v)
        s = str(v).replace('°C','').replace('℃','').strip()
        if '-' in s.replace('.',''):
            parts = [float(x) for x in s.replace('−','-').split('-') if x.strip()]
            return sum(parts)/len(parts) if parts else np.nan
        try: return float(s)
        except: return np.nan
    merged['temperature'] = merged['temperature'].apply(clean_temp).astype(float)
    merged['ph_val'] = pd.to_numeric(merged['ph_val'], errors='coerce')

    print(f"\n=== 合并后 ===")
    print(f"  总记录: {len(merged):,}")
    print(f"  UniProts: {merged['uniprot_id'].nunique():,}")
    print(f"  InChIKeys: {merged['ligand_inchikey'].nunique():,}")
    print(f"  Split: {merged['split'].value_counts().to_dict()}")
    print(f"  Ki/Kd: {(merged['measurement_type'].isin(['Ki','Kd'])).sum():,}")
    print(f"  kcat: {merged['has_kcat'].sum():,} ({100*merged['has_kcat'].sum()/len(merged):.1f}%)")

    # 保存
    p1 = OUTPUT_DIR/'unified_metadata_v2.parquet'
    merged.to_parquet(p1, index=False)
    print(f"\n✅ {p1}")

    hq = merged[merged['measurement_type'].isin(['Ki','Kd'])]
    p2 = OUTPUT_DIR/'high_quality_kd_ki_v2.parquet'
    hq.to_parquet(p2, index=False)
    print(f"✅ {p2} ({len(hq)} Ki/Kd)")

    p3 = OUTPUT_DIR/'new_external_records.parquet'
    new_unique['temperature'] = new_unique['temperature'].apply(clean_temp).astype(float)
    new_unique['ph_val'] = pd.to_numeric(new_unique['ph_val'], errors='coerce')
    new_unique.to_parquet(p3, index=False)
    print(f"✅ {p3} ({len(new_unique)} new)")

    return merged

if __name__ == '__main__':
    main()
