"""
Query ChEMBL for oxidoreductase protein activities, build IC50→Ki correction
model, and supplement Kd measurements.

References:
    - supplement_kinetics.py (UniProt API pattern)
    - supplement_sabio.py (SABIO-RK API pattern)

Outputs:
    - processed/oxidoreductase/cache/chembl_activities.json   (raw cache)
    - processed/oxidoreductase/chembl_paired_data.parquet     (IC50-Ki pairs)
    - processed/oxidoreductase/ic50_ki_correction.json       (model params)
    - processed/oxidoreductase/chembl_kd_supplement.parquet  (new Kd records)

Usage:
    python datepre/supplement_chembl_activities.py                     # Full run
    python datepre/supplement_chembl_activities.py --max-proteins 5    # Test
    python datepre/supplement_chembl_activities.py --no-fetch          # Cache only
    python datepre/supplement_chembl_activities.py --train-only         # Skip Kd
"""

import argparse
import json
import logging
import pickle
import sys
import time
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent / "dataset_building"
PROCESSED_DIR = PROJECT_DIR / "processed"
OXI_DIR = PROCESSED_DIR / "oxidoreductase"
CACHE_DIR = OXI_DIR / "cache"
UNIFIED_META = OXI_DIR / "unified_metadata.parquet"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

ACTIVITIES_CACHE = CACHE_DIR / "chembl_activities.json"
MOLECULE_CACHE = CACHE_DIR / "chembl_molecules.json"
TARGET_CACHE = CACHE_DIR / "chembl_targets.json"

CHUNK_SIZE = 100

# ─────────────────────────────────────────────────────────
# ChEMBL client
# ─────────────────────────────────────────────────────────

def get_client():
    from chembl_webresource_client.new_client import new_client
    return new_client


def fetch_target_by_uniprot(uniprot_id: str, client=None) -> str | None:
    """Get ChEMBL target ID from UniProt accession. Returns target_chembl_id or None."""
    if client is None:
        client = get_client()

    try:
        result = client.target.filter(
            target_components__accession=uniprot_id
        ).only('target_chembl_id')
        targets = list(result)
        if targets:
            return targets[0]['target_chembl_id']
    except Exception as e:
        log.debug(f"Target query error {uniprot_id}: {e}")
    return None


def fetch_activities(target_chembl_id: str, client=None) -> list[dict]:
    """Fetch all IC50, Ki, Kd activities for a ChEMBL target."""
    if client is None:
        client = get_client()

    activities = []
    try:
        result = client.activity.filter(
            target_chembl_id=target_chembl_id,
            standard_type__in=['IC50', 'Ki', 'Kd'],
            standard_relation='=',
            standard_units='nM',
        ).only(
            'activity_id', 'standard_type', 'standard_value',
            'standard_relation', 'molecule_chembl_id',
            'target_chembl_id', 'assay_type', 'pchembl_value',
        )

        for act in result:
            activities.append(dict(act))
    except Exception as e:
        log.debug(f"Activity query error {target_chembl_id}: {e}")

    return activities


def fetch_molecules_inchikey_batch(molecule_chembl_ids: list[str], client=None) -> dict[str, str]:
    """Get InChIKey for multiple ChEMBL molecules in one API call.

    InChIKey is stored in molecule_structures.standard_inchi_key, NOT molecule_properties.
    """
    if client is None:
        client = get_client()

    result = {}
    try:
        mols = client.molecule.filter(
            molecule_chembl_id__in=molecule_chembl_ids
        ).only('molecule_structures', 'molecule_chembl_id')
        for mol in mols:
            mid = mol.get('molecule_chembl_id')
            structures = mol.get('molecule_structures')
            if mid and structures:
                ik = structures.get('standard_inchi_key')
                if ik:
                    result[mid] = ik
    except Exception as e:
        log.debug(f"Batch molecule query error: {e}")
    return result


# ─────────────────────────────────────────────────────────
# Cache management
# ─────────────────────────────────────────────────────────

def load_json_cache(path: Path) -> dict:
    if path.exists():
        with open(path) as fh:
            return json.load(fh)
    return {}


def save_json_cache(path: Path, data: dict):
    with open(path, 'w') as fh:
        json.dump(data, fh, indent=2)


# ─────────────────────────────────────────────────────────
# Main logic
# ─────────────────────────────────────────────────────────

def query_all_activities(
    uniprot_ids: list[str],
    max_proteins: int | None = None,
) -> tuple[dict, dict, dict]:
    """Query ChEMBL for all oxidoreductase proteins.

    Returns:
        activities_cache: {uniprot_id: [activity_dict, ...]}
        molecule_cache: {molecule_chembl_id: inchikey}
        target_cache: {uniprot_id: target_chembl_id}
    """
    activities_cache = load_json_cache(ACTIVITIES_CACHE)
    molecule_cache = load_json_cache(MOLECULE_CACHE)
    target_cache = load_json_cache(TARGET_CACHE)

    if max_proteins:
        uniprot_ids = uniprot_ids[:max_proteins]

    client = get_client()

    # Step 1: UniProt → ChEMBL Target
    to_fetch_targets = [uid for uid in uniprot_ids if uid not in target_cache]
    log.info(f"Fetching ChEMBL targets: {len(to_fetch_targets)} UniProt IDs")

    for uid in tqdm(to_fetch_targets, desc="Target lookup"):
        tid = fetch_target_by_uniprot(uid, client)
        target_cache[uid] = tid
        if len(target_cache) % 10 == 0:
            save_json_cache(TARGET_CACHE, target_cache)
        time.sleep(0.05)  # Rate limiting

    if to_fetch_targets:
        save_json_cache(TARGET_CACHE, target_cache)

    # stats
    n_with_target = sum(1 for v in target_cache.values() if v)
    log.info(f"  UniProt→ChEMBL mapped: {n_with_target}/{len(target_cache)}")

    # Step 2: Fetch activities per target (parallel)
    to_fetch_activities = {
        uid: tid for uid, tid in target_cache.items()
        if tid and uid not in activities_cache
    }
    log.info(f"Fetching activities: {len(to_fetch_activities)} targets (parallel, workers=4)")

    n_saved = 0
    n_activities = 0
    client2 = get_client()

    def _fetch_one(uid_tid):
        uid, tid = uid_tid
        try:
            acts = fetch_activities(tid, client2)
            return uid, acts
        except Exception as e:
            log.debug(f"Activity fetch error for {uid}: {e}")
            return uid, []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_fetch_one, item): item for item in to_fetch_activities.items()}
        with tqdm(total=len(to_fetch_activities), desc="Activity fetch") as pbar:
            for future in as_completed(futures):
                uid, acts = future.result()
                activities_cache[uid] = acts
                n_activities += len(acts)
                n_saved += 1
                if n_saved % 5 == 0:
                    save_json_cache(ACTIVITIES_CACHE, activities_cache)
                pbar.update(1)
                pbar.set_postfix({"acts": n_activities})

    if to_fetch_activities:
        save_json_cache(ACTIVITIES_CACHE, activities_cache)

    total_acts = sum(len(v) for v in activities_cache.values())
    log.info(f"  Total activities: {total_acts}")

    # Step 3: Resolve molecule InChIKeys (batch, 100 per request)
    all_mol_ids = set()
    for acts in activities_cache.values():
        for a in acts:
            mid = a.get('molecule_chembl_id')
            if mid:
                all_mol_ids.add(mid)

    missing_mols = [mid for mid in all_mol_ids if mid not in molecule_cache]
    log.info(f"Fetching InChIKeys: {len(missing_mols)} molecules (batch size=100)")

    BATCH_SIZE = 100
    n_batches = (len(missing_mols) + BATCH_SIZE - 1) // BATCH_SIZE
    client3 = get_client()

    for i in tqdm(range(0, len(missing_mols), BATCH_SIZE), total=n_batches, desc="Molecule batch"):
        batch = missing_mols[i:i + BATCH_SIZE]
        batch_results = fetch_molecules_inchikey_batch(batch, client3)
        molecule_cache.update(batch_results)
        if (i // BATCH_SIZE) % 5 == 0:
            save_json_cache(MOLECULE_CACHE, molecule_cache)
        time.sleep(0.1)  # Rate limiting between batches

    save_json_cache(MOLECULE_CACHE, molecule_cache)
    log.info(f"  Molecule cache: {len(molecule_cache)} entries")

    return activities_cache, molecule_cache, target_cache


def build_paired_data(
    activities_cache: dict,
    molecule_cache: dict,
) -> pd.DataFrame:
    """Build IC50-Ki paired dataset from ChEMBL activities.

    Groups by (target_chembl_id, molecule_chembl_id) and finds pairs
    where both IC50 and Ki are measured.
    """
    # Collect activities with InChIKeys
    records = []
    for uniprot_id, acts in activities_cache.items():
        for a in acts:
            mid = a.get('molecule_chembl_id')
            ik = molecule_cache.get(mid)
            stype = a.get('standard_type')
            value = a.get('standard_value')
            pchembl = a.get('pchembl_value')

            if not mid or not stype or value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if value <= 0:
                continue

            records.append({
                'uniprot_id': uniprot_id,
                'target_chembl_id': a.get('target_chembl_id'),
                'molecule_chembl_id': mid,
                'inchikey': ik,
                'standard_type': stype,
                'standard_value_nM': value,
                'pchembl_value': pchembl,
            })

    df = pd.DataFrame(records)
    log.info(f"Total ChEMBL activity records: {len(df)}")

    if len(df) == 0:
        log.warning("No ChEMBL activities found. Cannot build paired data.")
        return pd.DataFrame()

    # Stats by type
    type_counts = df['standard_type'].value_counts()
    log.info(f"  IC50: {type_counts.get('IC50', 0)}, Ki: {type_counts.get('Ki', 0)}, "
             f"Kd: {type_counts.get('Kd', 0)}")

    # Group by (target, molecule) to find pairs
    paired_rows = []
    for (tid, mid), grp in df.groupby(['target_chembl_id', 'molecule_chembl_id']):
        ic50_vals = grp[grp['standard_type'] == 'IC50']['standard_value_nM'].values
        ki_vals = grp[grp['standard_type'] == 'Ki']['standard_value_nM'].values

        if len(ic50_vals) > 0 and len(ki_vals) > 0:
            paired_rows.append({
                'target_chembl_id': tid,
                'molecule_chembl_id': mid,
                'inchikey': grp['inchikey'].iloc[0],
                'IC50_nM': float(np.median(ic50_vals)),
                'Ki_nM': float(np.median(ki_vals)),
                'n_ic50': len(ic50_vals),
                'n_ki': len(ki_vals),
            })

    paired_df = pd.DataFrame(paired_rows)
    log.info(f"IC50-Ki paired records: {len(paired_df)}")

    if len(paired_df) > 0:
        # Convert to p-values (-log10 M)
        paired_df['pIC50'] = -np.log10(paired_df['IC50_nM'] * 1e-9)
        paired_df['pKi'] = -np.log10(paired_df['Ki_nM'] * 1e-9)

        # Filter outliers
        valid = (
            (paired_df['pIC50'] >= 2) & (paired_df['pIC50'] <= 14)
            & (paired_df['pKi'] >= 2) & (paired_df['pKi'] <= 14)
        )
        paired_df = paired_df[valid].copy()
        log.info(f"  After outlier filter: {len(paired_df)}")

    return paired_df


def train_correction_model(paired_df: pd.DataFrame) -> dict:
    """Train IC50→Ki correction model from paired data."""
    from sklearn.linear_model import LinearRegression

    if len(paired_df) < 10:
        log.warning(
            f"Only {len(paired_df)} paired records (<10). "
            "Using theoretical correction: Ki = IC50 / 2"
        )
        return {
            'model_type': 'theoretical',
            'a': 1.0,
            'b': np.log10(2),  # pKi = pIC50 + log10(2) ≈ pIC50 + 0.301
            'r2': None,
            'n_pairs': len(paired_df),
            'rmse': None,
        }

    X = paired_df[['pIC50']].values
    y = paired_df['pKi'].values

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    r2 = model.score(X, y)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    a, b = float(model.coef_[0]), float(model.intercept_)

    result = {
        'model_type': 'ols_loglog',
        'a': a,
        'b': b,
        'r2': float(r2),
        'n_pairs': len(paired_df),
        'rmse': float(rmse),
        'mean_pIC50': float(np.mean(X)),
        'mean_pKi': float(np.mean(y)),
        'formula': f'pKi = {a:.4f} * pIC50 + {b:.4f}',
    }

    if r2 < 0.3:
        log.warning(
            f"Correction model R²={r2:.3f} < 0.3. "
            "Falling back to theoretical correction: Ki = IC50 / 2"
        )
        result = {
            'model_type': 'theoretical_fallback',
            'a': 1.0,
            'b': 0.301,
            'r2': float(r2),
            'n_pairs': len(paired_df),
            'rmse': float(rmse),
            'note': f'OLS R²={r2:.3f} too low, using Cheng-Prusoff [S]=KM assumption',
        }

    log.info(f"Correction model: {result['formula'] if 'formula' in result else 'Ki=IC50/2'}")
    log.info(f"  R²={r2:.3f}, RMSE={rmse:.3f} log10 units, n_pairs={len(paired_df)}")

    return result


def extract_kd_supplements(
    activities_cache: dict,
    molecule_cache: dict,
    target_cache: dict,
    df_existing: pd.DataFrame,
) -> pd.DataFrame:
    """Extract Kd measurements from ChEMBL to supplement existing dataset."""
    existing_iks = set(df_existing['ligand_inchikey'].unique())

    kd_records = []
    for uniprot_id, acts in activities_cache.items():
        for a in acts:
            if a.get('standard_type') != 'Kd':
                continue

            mid = a.get('molecule_chembl_id')
            ik = molecule_cache.get(mid)
            value = a.get('standard_value')
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if value <= 0:
                continue

            # Only include if ligand is already in our dataset (known chemistry)
            if ik and ik in existing_iks:
                kd_records.append({
                    'uniprot_id': uniprot_id,
                    'ligand_inchikey': ik,
                    'measurement_type': 'Kd',
                    'pkd_raw': float(-np.log10(value * 1e-9)),
                    'n_measurements': 1,
                    'pkd_std': 0.0,
                    'source': 'chembl_kd',
                    'molecule_chembl_id': mid,
                })

    log.info(f"ChEMBL Kd supplements: {len(kd_records)} records")
    return pd.DataFrame(kd_records)


def main():
    parser = argparse.ArgumentParser(
        description="Query ChEMBL for IC50→Ki correction + Kd supplementation"
    )
    parser.add_argument("--max-proteins", type=int, default=None)
    parser.add_argument("--no-fetch", action="store_true",
                        help="Use cache only, no API calls")
    parser.add_argument("--train-only", action="store_true",
                        help="Train correction model only, skip Kd")
    parser.add_argument("--unified-meta", default=str(UNIFIED_META))
    args = parser.parse_args()

    # Load existing data
    log.info(f"Loading: {args.unified_meta}")
    df = pd.read_parquet(args.unified_meta)
    uniprot_ids = sorted(df['uniprot_id'].dropna().unique())
    log.info(f"  {len(uniprot_ids)} unique UniProt IDs")

    # ── Query ChEMBL ──────────────────────────────────
    if not args.no_fetch:
        activities_cache, molecule_cache, target_cache = query_all_activities(
            uniprot_ids, args.max_proteins
        )
    else:
        activities_cache = load_json_cache(ACTIVITIES_CACHE)
        molecule_cache = load_json_cache(MOLECULE_CACHE)
        target_cache = load_json_cache(TARGET_CACHE)
        log.info(f"Loaded cache: {len(activities_cache)} activities, "
                 f"{len(molecule_cache)} molecules, {len(target_cache)} targets")

    # ── Build paired IC50-Ki data ─────────────────────
    log.info("\n" + "=" * 50)
    log.info("Building IC50-Ki paired dataset...")
    paired_df = build_paired_data(activities_cache, molecule_cache)

    if len(paired_df) > 0:
        paired_path = OXI_DIR / "chembl_paired_data.parquet"
        paired_df.to_parquet(paired_path)
        log.info(f"Saved: {paired_path} ({len(paired_df)} pairs)")

    # ── Train correction model ────────────────────────
    log.info("\n" + "=" * 50)
    log.info("Training IC50→Ki correction model...")
    correction = train_correction_model(paired_df)

    correction_path = OXI_DIR / "ic50_ki_correction.json"
    with open(correction_path, 'w') as fh:
        json.dump(correction, fh, indent=2)
    log.info(f"Saved: {correction_path}")

    # ── Kd supplementation ────────────────────────────
    if not args.train_only:
        log.info("\n" + "=" * 50)
        log.info("Extracting ChEMBL Kd supplements...")
        kd_df = extract_kd_supplements(
            activities_cache, molecule_cache, target_cache, df
        )

        if len(kd_df) > 0:
            kd_path = OXI_DIR / "chembl_kd_supplement.parquet"
            kd_df.to_parquet(kd_path)
            log.info(f"Saved: {kd_path} ({len(kd_df)} records)")

            # Stats
            log.info(f"  Unique UniProt: {kd_df['uniprot_id'].nunique()}")
            log.info(f"  Unique ligands: {kd_df['ligand_inchikey'].nunique()}")
            log.info(f"  pKd range: [{kd_df['pkd_raw'].min():.1f}, {kd_df['pkd_raw'].max():.1f}]")

    log.info("\n" + "=" * 50)
    log.info("Done! Next step: python datepre/apply_ic50_correction.py")


if __name__ == "__main__":
    main()
