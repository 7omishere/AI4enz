"""
Targeted ChEMBL supplement for cold cofactor oxidoreductase enzymes.

Searches ChEMBL by EC number (broader than UniProt-only), fetches
IC50/Ki/Kd activities, applies existing IC50→Ki correction, and outputs
new records ready to merge into unified_metadata.parquet.

Target cofactors:
  - TPP: EC 1.2.4.1, 1.2.4.2, 1.2.4.4 (pyruvate/oxoglutarate dehydrogenase)
  - COQ: EC 1.4.3.13, 1.4.3.21, 1.4.3.22 (amine oxidases)
  - CU:  EC 1.14.18.1, 1.14.17.1 (tyrosinase, dopamine beta-hydroxylase)
  - PLP: EC 1.14.16.4 (tryptophan hydroxylase)

Usage:
  source /home/domi/BINN/.venv/bin/activate
  cd /home/domi/AI4enz
  python datepre/supplement_cold_cofactors.py              # Full run with API calls
  python datepre/supplement_cold_cofactors.py --no-fetch   # Use cache only
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

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
CACHE_DIR.mkdir(parents=True, exist_ok=True)

UNIFIED_META = OXI_DIR / "unified_metadata.parquet"
EC_CACHE = CACHE_DIR / "cold_cofactor_ec_search.json"
ACTIVITIES_CACHE = CACHE_DIR / "cold_cofactor_activities.json"
OUTPUT = OXI_DIR / "cold_cofactor_supplement.parquet"

# ── Cold cofactor EC classes ──
COLD_EC_NUMBERS = {
    "TPP": ["1.2.4.1", "1.2.4.2", "1.2.4.4"],
    "COQ": ["1.4.3.13", "1.4.3.21", "1.4.3.22"],
    "CU":  ["1.14.18.1", "1.14.17.1"],
    "PLP": ["1.14.16.4"],
}

# IC50→Ki correction (from ic50_ki_correction.json)
CORRECTION_A = 0.7172
CORRECTION_B = 2.2939


def get_client():
    from chembl_webresource_client.new_client import new_client
    return new_client


# ═══════════════════════════════════════════════════════════════
# ChEMBL API helpers
# ═══════════════════════════════════════════════════════════════

def search_targets_by_ec(ec_number: str, client=None) -> list[dict]:
    """Search ChEMBL targets by EC number. Returns list of target dicts."""
    if client is None:
        client = get_client()
    targets = []
    try:
        results = client.target.search(ec_number)
        for r in results:
            targets.append({
                "target_chembl_id": r.get("target_chembl_id"),
                "pref_name": r.get("pref_name"),
                "organism": r.get("organism"),
                "target_type": r.get("target_type"),
                "uniprot_ids": [
                    c["accession"] for c in r.get("target_components", [])
                    if c.get("accession")
                ],
            })
    except Exception as e:
        log.warning(f"  EC search error for {ec_number}: {e}")
    return targets


def fetch_activities(target_chembl_id: str, client=None) -> list[dict]:
    """Fetch IC50, Ki, Kd activities for a ChEMBL target. Exact matches, nM units only."""
    if client is None:
        client = get_client()
    activities = []
    try:
        result = client.activity.filter(
            target_chembl_id=target_chembl_id,
            standard_type__in=["IC50", "Ki", "Kd"],
            standard_relation="=",
            standard_units="nM",
        ).only(
            "activity_id", "standard_type", "standard_value",
            "standard_relation", "molecule_chembl_id",
            "target_chembl_id", "assay_type", "pchembl_value",
        )
        for act in result:
            activities.append(dict(act))
    except Exception as e:
        log.debug(f"  Activity query error {target_chembl_id}: {e}")
    return activities


def fetch_molecules_inchikey_batch(molecule_chembl_ids: list[str], client=None) -> dict[str, str]:
    """Get InChIKey for multiple ChEMBL molecules."""
    if client is None:
        client = get_client()

    molecule_chembl_ids = list(set(molecule_chembl_ids))
    result = {}
    # Batch by 100
    for i in range(0, len(molecule_chembl_ids), 100):
        batch = molecule_chembl_ids[i:i + 100]
        try:
            mols = client.molecule.filter(
                molecule_chembl_id__in=batch
            ).only("molecule_structures", "molecule_chembl_id")
            for mol in mols:
                mid = mol.get("molecule_chembl_id")
                structures = mol.get("molecule_structures")
                if mid and structures:
                    ik = structures.get("standard_inchi_key")
                    if ik:
                        result[mid] = ik
        except Exception as e:
            log.debug(f"  Molecule batch error: {e}")
        if i + 100 < len(molecule_chembl_ids):
            time.sleep(0.1)
    return result


# ═══════════════════════════════════════════════════════════════
# Cache helpers
# ═══════════════════════════════════════════════════════════════

def load_json_cache(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_json_cache(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ═══════════════════════════════════════════════════════════════
# Main logic
# ═══════════════════════════════════════════════════════════════

def search_and_fetch(no_fetch: bool = False) -> tuple[dict, dict]:
    """Search ChEMBL by EC number for cold cofactor targets, fetch activities.

    Returns (targets_by_ec, activities_by_target).
    """
    ec_cache = load_json_cache(EC_CACHE)
    act_cache = load_json_cache(ACTIVITIES_CACHE)

    if no_fetch and not ec_cache:
        log.error("--no-fetch specified but no cache exists. Run without --no-fetch first.")
        return ec_cache, act_cache

    client = get_client()

    # Step 1: Search by EC number
    for cf_type, ec_list in COLD_EC_NUMBERS.items():
        for ec_number in ec_list:
            key = f"{cf_type}|{ec_number}"
            if key in ec_cache:
                log.info(f"  [cache] {key}: {len(ec_cache[key])} targets")
                continue

            log.info(f"  Searching: {key}")
            targets = search_targets_by_ec(ec_number, client)
            # Filter to single-protein targets (avoid complexes)
            targets = [t for t in targets if t["target_type"] == "SINGLE PROTEIN" and t["uniprot_ids"]]
            ec_cache[key] = targets
            save_json_cache(EC_CACHE, ec_cache)
            log.info(f"    Found {len(targets)} single-protein targets")
            time.sleep(0.15)

    # Step 2: Collect unique UniProt IDs, filter to oxidoreductase (EC 1.x.x.x)
    all_targets = []
    for key, targets in ec_cache.items():
        all_targets.extend(targets)

    # De-duplicate by target_chembl_id
    seen = set()
    unique_targets = []
    for t in all_targets:
        tid = t["target_chembl_id"]
        if tid not in seen:
            seen.add(tid)
            unique_targets.append(t)

    log.info(f"  Total unique targets: {len(unique_targets)}")

    # Step 3: Fetch activities per target
    all_uniprots = set()
    for t in unique_targets:
        all_uniprots.update(t["uniprot_ids"])

    if no_fetch:
        return ec_cache, act_cache

    # Load existing dataset for Uniprot→protein mapping
    df_existing = pd.read_parquet(UNIFIED_META)
    existing_uniprots = set(u for u in df_existing["uniprot_id"].unique() if isinstance(u, str))

    # Fetch activities for each target
    for t in unique_targets:
        tid = t["target_chembl_id"]
        if tid in act_cache:
            continue

        # Check if at least one Uniprot matches existing dataset
        matches_existing = any(up in existing_uniprots for up in t["uniprot_ids"])
        log.info(f"  Fetching activities for {tid} ({t['pref_name']}): "
                 f"in_dataset={matches_existing}, uniprots={t['uniprot_ids'][:3]}")

        activities = fetch_activities(tid, client)
        act_cache[tid] = activities
        if len(act_cache) % 5 == 0:
            save_json_cache(ACTIVITIES_CACHE, act_cache)
        log.info(f"    → {len(activities)} activities")
        time.sleep(0.1)

    save_json_cache(ACTIVITIES_CACHE, act_cache)
    return ec_cache, act_cache


def build_supplement(ec_cache: dict, act_cache: dict) -> pd.DataFrame:
    """Build supplement records from fetched activities.

    Maps targets to existing proteins in unified_metadata, converts nM values
    to pKd, applies IC50→Ki correction.
    """
    df_existing = pd.read_parquet(UNIFIED_META)
    # Uniprot → protein_seq_hash, cofactors mapping
    uniprot_info = df_existing.groupby("uniprot_id").agg(
        protein_seq_hash=("protein_seq_hash", "first"),
        protein_name=("protein_name", "first"),
        cofactors=("cofactors", "first"),
        ec_numbers=("ec_numbers", "first"),
    ).to_dict("index")

    existing_uniprots = set(df_existing["uniprot_id"].unique())
    existing_pairs = set(zip(df_existing["protein_seq_hash"], df_existing["ligand_inchikey"]))

    # Target → Uniprot mapping from EC cache
    target_uniprot_map = {}
    for key, targets in ec_cache.items():
        cf_type = key.split("|")[0]
        for t in targets:
            tid = t["target_chembl_id"]
            if tid not in target_uniprot_map:
                target_uniprot_map[tid] = (cf_type, t["uniprot_ids"])

    # Collect molecule IDs that need InChIKey resolution
    all_mol_ids = set()
    for tid, acts in act_cache.items():
        for act in acts:
            mid = act.get("molecule_chembl_id")
            if mid:
                all_mol_ids.add(mid)

    log.info(f"  Resolving InChIKey for {len(all_mol_ids)} molecules...")
    mol_inchikey = fetch_molecules_inchikey_batch(list(all_mol_ids))
    log.info(f"    Resolved {len(mol_inchikey)} InChIKeys")

    # Build records
    records = []
    skipped_new_protein = 0
    skipped_duplicate = 0
    skipped_no_inchikey = 0

    for tid, acts in act_cache.items():
        if tid not in target_uniprot_map:
            continue

        cf_type, uniprots = target_uniprot_map[tid]

        # Find matching uniprot in existing dataset
        matched_uniprot = None
        for up in uniprots:
            if up in existing_uniprots:
                matched_uniprot = up
                break

        if matched_uniprot is None:
            skipped_new_protein += len(acts)
            continue

        info = uniprot_info[matched_uniprot]
        seq_hash = info["protein_seq_hash"]
        cofactors = info["cofactors"] or ""
        ec = info["ec_numbers"] or ""
        pname = info["protein_name"] or ""

        for act in acts:
            mid = act.get("molecule_chembl_id", "")
            ik = mol_inchikey.get(mid)
            if not ik:
                skipped_no_inchikey += 1
                continue

            # Skip if already in dataset
            if (seq_hash, ik) in existing_pairs:
                skipped_duplicate += 1
                continue

            stype = act.get("standard_type", "")
            value_nm = act.get("standard_value")
            if value_nm is None or value_nm <= 0:
                continue

            # Convert to pKd
            pkd_raw = -np.log10(float(value_nm) * 1e-9)

            # Apply IC50→Ki correction
            if stype == "IC50":
                pkd_aligned = CORRECTION_A * pkd_raw + CORRECTION_B
                correction_source = "chembl_ic50_ki_model"
            else:
                pkd_aligned = pkd_raw
                correction_source = "chembl_direct"

            # Quality weight
            if stype == "Kd":
                thermo_weight = 1.0
            elif stype == "Ki":
                thermo_weight = 0.7
            else:  # IC50
                thermo_weight = 0.15

            records.append({
                "sample_id": f"chembl_cold_{act.get('activity_id', '')}",
                "protein_seq_hash": seq_hash,
                "ligand_inchikey": ik,
                "uniprot_id": matched_uniprot,
                "pdb_id": "",
                "source_db": "ChEMBL",
                "ec_numbers": ec,
                "cofactors": cofactors,
                "protein_name": pname,
                "reviewed": False,
                "pkd_aligned": round(pkd_aligned, 4),
                "pkd_raw": round(pkd_raw, 4),
                "measurement_type": stype,
                "quality_weight": thermo_weight,
                "w_multiplier": 1.0,
                "is_censored": False,
                "n_measurements": 1,
                "pkd_std": 0.0,
                "has_structure": info.get("has_structure", True),
                "has_binding_site": info.get("has_binding_site", True),
                "has_domain_annotation": info.get("has_domain_annotation", False),
                "n_domains": 0,
                "cofactor_domain_types": "",
                "domains_json": "",
                "bdb_n_km": 0, "bdb_n_kcat": 0, "bdb_n_kcatkm": 0,
                "bdb_km_median_uM": np.nan, "bdb_kcat_median_s": np.nan,
                "bdb_kcatkm_median_M1s1": np.nan,
                "n_km_sabio": 0, "n_kcat_sabio": 0, "n_kcatkm_sabio": 0,
                "km_median_uM_sabio": np.nan, "kcat_median_s_sabio": np.nan,
                "kcatkm_median_M1s1_sabio": np.nan,
                "up_has_kinetics": False, "up_n_kcat": 0,
                "has_kcat": False, "kcat_source": "",
                "kcat_median_s": np.nan, "log_kcat_median": np.nan,
                "kcat_outlier": False,
                "split": "train",  # placeholder, will be re-split
                "structure_source": "AlphaFold DB",
                "pkd_corrected": round(pkd_aligned, 4),
                "correction_source": correction_source,
            })

    log.info(f"  Built {len(records)} new records")
    log.info(f"  Skipped: {skipped_new_protein} new-protein, "
             f"{skipped_duplicate} duplicate, {skipped_no_inchikey} no-InChIKey")

    df_supp = pd.DataFrame(records)
    return df_supp


def main():
    parser = argparse.ArgumentParser(description="Supplement cold cofactor data from ChEMBL")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Skip API calls, use cache only")
    args = parser.parse_args()

    # Step 1: Search + Fetch
    log.info("=" * 60)
    log.info("Step 1: Search ChEMBL by EC number + fetch activities")
    log.info("=" * 60)

    ec_cache, act_cache = search_and_fetch(no_fetch=args.no_fetch)

    if not act_cache:
        log.warning("No activity data. Run without --no-fetch first.")
        return

    total_acts = sum(len(v) for v in act_cache.values())
    log.info(f"  Total activities in cache: {total_acts}")

    # Step 2: Build supplement
    log.info("=" * 60)
    log.info("Step 2: Building supplement records")
    log.info("=" * 60)

    df_supp = build_supplement(ec_cache, act_cache)

    if len(df_supp) == 0:
        log.warning("No new records to add!")
        return

    # Step 3: Save
    log.info("=" * 60)
    log.info("Step 3: Saving supplement")
    log.info("=" * 60)

    # Align columns with unified_metadata
    df_existing = pd.read_parquet(UNIFIED_META)
    for col in df_existing.columns:
        if col not in df_supp.columns:
            df_supp[col] = "" if df_existing[col].dtype == object else (
                False if df_existing[col].dtype == bool else 0.0
            )

    # Ensure matching column order
    df_supp = df_supp[df_existing.columns]

    df_supp.to_parquet(OUTPUT, index=False)
    log.info(f"  Saved {len(df_supp)} records → {OUTPUT}")

    # Breakdown by cofactor type
    df_supp["_cf"] = df_supp["cofactors"].apply(
        lambda x: str(x).split("|")[0].strip() if x else "none"
    )
    log.info("\n  New records by cofactor type:")
    for cf, count in df_supp["_cf"].value_counts().items():
        log.info(f"    {cf}: {count}")

    log.info("\n  New records by measurement type:")
    for mtype, count in df_supp["measurement_type"].value_counts().items():
        log.info(f"    {mtype}: {count}")

    log.info("Done.")


if __name__ == "__main__":
    main()
