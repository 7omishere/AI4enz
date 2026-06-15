"""
merge_bindingdb.py — 从 BindingDB 提取高可信度 Kd/Ki 合并到 v2 数据集
"""
import pandas as pd, numpy as np, hashlib, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey

EXTERNAL_DIR = Path(__file__).parent / 'external_data'
OUTPUT_DIR = Path(__file__).parent / 'processed' / 'oxidoreductase'
V2_PARQUET = OUTPUT_DIR / 'unified_metadata_v2.parquet'

def s2inchikey(smiles):
    if pd.isna(smiles) or not str(smiles).strip(): return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        return MolToInchiKey(mol) if mol else None
    except: return None

def hash_seq(seq):
    return hashlib.sha256(str(seq).upper().encode()).hexdigest()[:16]

def clean_temp(v):
    if pd.isna(v): return np.nan
    if isinstance(v, (int, float)): return float(v)
    s = str(v).replace('°C','').replace('℃','').replace(' C','').strip()
    try: return float(s)
    except: return np.nan

print("="*60)
print("BindingDB Kd/Ki 提取合并")
print("="*60)

# Load BindingDB
cols = [
    'Ligand SMILES', 'Ki (nM)', 'Kd (nM)',
    'BindingDB Target Chain Sequence 1',
    'UniProt (SwissProt) Primary ID of Target Chain 1',
    'pH', 'Temp (C)', 'Target Name',
    'Target Source Organism According to Curator or DataSource',
    'PDB ID(s) of Target Chain 1',
]
print("Loading BindingDB TSV...")
bdb = pd.read_csv(EXTERNAL_DIR/'BindingDB_All_202605_tsv.zip', sep='\t',
                  usecols=cols, low_memory=False)
print(f"  {len(bdb):,} total rows")

# Filter: Ki or Kd + UniProt + SMILES + Sequence
mask = (
    (bdb['Ki (nM)'].notna() | bdb['Kd (nM)'].notna()) &
    bdb['UniProt (SwissProt) Primary ID of Target Chain 1'].notna() &
    bdb['Ligand SMILES'].notna() &
    bdb['BindingDB Target Chain Sequence 1'].notna()
)
bdb = bdb[mask].copy()
print(f"  {len(bdb):,} with Ki/Kd + UniProt + Sequence + SMILES")

# Convert SMILES → InChIKey
print("Converting SMILES → InChIKey...")
bdb['inchikey'] = bdb['Ligand SMILES'].apply(s2inchikey)
valid_ik = bdb['inchikey'].notna()
print(f"  Valid InChIKeys: {valid_ik.sum():,} / {len(bdb):,}")
bdb = bdb[valid_ik].copy()

# Convert Ki/Kd to pKd
# Ki in nM → pKi = -log10(Ki_M) = -log10(Ki_nM * 1e-9) = 9 - log10(Ki_nM)
def parse_numeric(v):
    """Handle '>100000', '<10', '~50' etc"""
    if pd.isna(v): return np.nan
    if isinstance(v, (int, float)): return float(v)
    s = str(v).strip().replace(',','').replace('~','')
    # Handle censored: >, >=, <, <=
    for prefix in ['>=', '<=', '>', '<']:
        if s.startswith(prefix):
            return float(s[len(prefix):].strip())
    try: return float(s)
    except: return np.nan

bdb['pKd'] = np.nan
has_ki = bdb['Ki (nM)'].notna()
has_kd = bdb['Kd (nM)'].notna()

# Apply parse and compute pKd for all rows
bdb['ki_num'] = bdb['Ki (nM)'].apply(parse_numeric)
bdb['kd_num'] = bdb['Kd (nM)'].apply(parse_numeric)

has_ki = bdb['ki_num'].notna()
has_kd = bdb['kd_num'].notna()
both = has_ki & has_kd

# pKd = 9 - log10(nM), clip to avoid log(0) or negative
bdb['pKd'] = np.nan
bdb.loc[has_ki, 'pKd'] = 9 - np.log10(bdb.loc[has_ki, 'ki_num'].clip(lower=1e-6))
bdb.loc[has_kd & ~has_ki, 'pKd'] = 9 - np.log10(bdb.loc[has_kd & ~has_ki, 'kd_num'].clip(lower=1e-6))
bdb.loc[both, 'pKd'] = 9 - np.log10(bdb.loc[both, 'kd_num'].clip(lower=1e-6))  # prefer Kd

print(f"  pKd range: [{bdb['pKd'].min():.1f}, {bdb['pKd'].max():.1f}]")

# Determine measurement type
bdb['measurement_type'] = 'Ki'
bdb.loc[has_kd & ~has_ki, 'measurement_type'] = 'Kd'
bdb.loc[both, 'measurement_type'] = 'Kd'  # prefer Kd label when both exist

# Build records
print("Building records...")
records = []
for _, row in bdb.iterrows():
    seq_hash = hash_seq(row['BindingDB Target Chain Sequence 1'])

    # Quality weight: Kd > Ki
    qw = 1.0 if row['measurement_type'] == 'Kd' else 0.7

    # PDB
    pdb = None
    if pd.notna(row['PDB ID(s) of Target Chain 1']):
        pdbs = str(row['PDB ID(s) of Target Chain 1']).split(',')[0].strip()
        if len(pdbs) == 4:
            pdb = pdbs

    r = {
        'sample_id': f"bdb_{hashlib.md5((row['UniProt (SwissProt) Primary ID of Target Chain 1']+row['inchikey']).encode()).hexdigest()[:12]}",
        'protein_seq_hash': seq_hash,
        'ligand_inchikey': row['inchikey'],
        'uniprot_id': row['UniProt (SwissProt) Primary ID of Target Chain 1'],
        'pdb_id': pdb,
        'source_db': 'bindingdb',
        'ec_numbers': '',
        'cofactors': None,
        'protein_name': str(row.get('Target Name', '')) if pd.notna(row.get('Target Name')) else None,
        'reviewed': True,
        'pkd_aligned': row['pKd'],
        'pkd_raw': row['pKd'],
        'measurement_type': row['measurement_type'],
        'quality_weight': qw,
        'w_multiplier': 1.0,
        'is_censored': False,
        'n_measurements': 1,
        'pkd_std': np.nan,
        'has_kcat': False,
        'kcat_source': None,
        'kcat_median_s': np.nan,
        'log_kcat_median': np.nan,
        'kcat_outlier': False,
        'has_structure': pdb is not None,
        'has_binding_site': False,
        'has_domain_annotation': False,
        'n_domains': 0,
        'cofactor_domain_types': None,
        'domains_json': None,
        'temperature': clean_temp(row.get('Temp (C)')),
        'ph_val': pd.to_numeric(row.get('pH'), errors='coerce'),
        'bdb_n_km': 0, 'bdb_n_kcat': 0, 'bdb_n_kcatkm': 0,
        'bdb_km_median_uM': np.nan, 'bdb_kcat_median_s': np.nan, 'bdb_kcatkm_median_M1s1': np.nan,
        'n_km_sabio': 0, 'n_kcat_sabio': 0, 'n_kcatkm_sabio': 0,
        'km_median_uM_sabio': np.nan, 'kcat_median_s_sabio': np.nan, 'kcatkm_median_M1s1_sabio': np.nan,
        'up_has_kinetics': False, 'up_n_kcat': 0,
        'pkd_corrected': row['pKd'],
        'correction_source': 'bindingdb',
        'split': None,
        'organism_name': str(row.get('Target Source Organism According to Curator or DataSource', ''))[:200] if pd.notna(row.get('Target Source Organism According to Curator or DataSource')) else None,
    }
    records.append(r)

new_df = pd.DataFrame(records)
print(f"  Records: {len(new_df):,}")
print(f"  UniProts: {new_df['uniprot_id'].nunique():,}")
print(f"  InChIKeys: {new_df['ligand_inchikey'].nunique():,}")
print(f"  Kd: {(new_df['measurement_type']=='Kd').sum():,}")
print(f"  Ki: {(new_df['measurement_type']=='Ki').sum():,}")

# Load existing v2 and merge
existing = pd.read_parquet(V2_PARQUET)
print(f"\n现有 v2 数据集: {len(existing):,} records")

# Dedup
existing['_key'] = existing['uniprot_id'].fillna('')+'|'+existing['ligand_inchikey'].fillna('')+'|pKd'
new_df['_key'] = new_df['uniprot_id'].fillna('')+'|'+new_df['ligand_inchikey'].fillna('')+'|pKd'

existing_keys = set(existing['_key'])
new_unique = new_df[~new_df['_key'].isin(existing_keys)].copy()
print(f"  重复: {len(new_df)-len(new_unique):,}, 新增: {len(new_unique):,}")

# Assign split for new records
np.random.seed(42)
new_ups = new_unique['uniprot_id'].dropna().unique()
np.random.shuffle(new_ups)
n_train = int(len(new_ups) * 0.8)
n_val = int(len(new_ups) * 0.1)
split_map = {}
for u in new_ups[:n_train]: split_map[u] = 'train'
for u in new_ups[n_train:n_train+n_val]: split_map[u] = 'val'
for u in new_ups[n_train+n_val:]: split_map[u] = 'test'
new_unique['split'] = new_unique['uniprot_id'].map(split_map).fillna('train')

# Merge
merged = pd.concat([existing, new_unique.drop(columns=['_key'])], ignore_index=True)
merged = merged.drop(columns=['_key'], errors='ignore')

print(f"\n=== 最终数据集 ===")
print(f"  总记录: {len(merged):,}")
print(f"  UniProts: {merged['uniprot_id'].nunique():,}")
print(f"  InChIKeys: {merged['ligand_inchikey'].nunique():,}")
print(f"  Kd: {(merged['measurement_type']=='Kd').sum():,}")
print(f"  Ki: {(merged['measurement_type']=='Ki').sum():,}")
print(f"  kcat: {merged['has_kcat'].sum():,} ({100*merged['has_kcat'].sum()/len(merged):.1f}%)")

# Save
merged.to_parquet(V2_PARQUET, index=False)
print(f"\n✅ 已更新: {V2_PARQUET}")

# High quality subset
hq = merged[merged['measurement_type'].isin(['Ki','Kd'])]
hq_path = OUTPUT_DIR/'high_quality_kd_ki_v2.parquet'
hq.to_parquet(hq_path, index=False)
print(f"✅ 高质量子集: {hq_path} ({len(hq):,} Ki/Kd)")

# Also save new records separately
new_path = OUTPUT_DIR/'bindingdb_new_records.parquet'
new_unique.to_parquet(new_path, index=False)
print(f"✅ 新增记录: {new_path} ({len(new_unique):,})")
