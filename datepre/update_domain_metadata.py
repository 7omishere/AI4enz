"""
Update unified_metadata.parquet domain columns from proteins.h5 domain_masks.

Run after Pfam domain scanning (enrich_domains_pfam.py) to sync the summary
columns (has_domain_annotation, n_domains, cofactor_domain_types, domains_json)
with the actual domain_masks arrays in proteins.h5.
"""

import json
import logging
from collections import Counter
from pathlib import Path

import h5py
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
PROTEINS_H5 = PROCESSED_DIR / "proteins.h5"
UNIFIED_META = PROCESSED_DIR / "oxidoreductase" / "unified_metadata.parquet"

COFACTOR_INDEX: dict[str, int] = {
    "NAD": 0, "NADP": 1, "FAD": 2, "FMN": 3, "HEME": 4,
    "FES": 5, "CU": 6, "MPT": 7, "COQ": 8, "PQQ": 9,
    "TPP": 10, "PLP": 11, "COA": 12, "B12": 13, "THF": 14,
}
IDX_TO_CT = {v: k for k, v in COFACTOR_INDEX.items()}


def extract_domain_info(h5, seq_hash: str) -> tuple[bool, int, str, str]:
    """从 proteins.h5 domain_masks 提取域摘要信息。"""
    if seq_hash not in h5 or "domain_masks" not in h5[seq_hash]:
        return False, 0, "", "[]"

    dm = h5[seq_hash]["domain_masks"][:]
    cofactor_types = set()
    positions = []

    for i in range(len(COFACTOR_INDEX)):
        if dm[i].sum() > 0:
            ct = IDX_TO_CT.get(i, f"UNK{i}")
            cofactor_types.add(ct)
            # 找连续区域
            mask = dm[i] > 0
            edges = np.where(np.diff(np.concatenate([[False], mask, [False]])))[0]
            for j in range(0, len(edges), 2):
                positions.append({
                    "cofactor_type": ct,
                    "start": int(edges[j]),
                    "end": int(edges[j + 1]),
                })

    has_annotation = len(cofactor_types) > 0
    n_domains = len(positions)
    ct_str = ",".join(sorted(cofactor_types)) if cofactor_types else ""
    domains_json = json.dumps(positions) if positions else "[]"

    return has_annotation, n_domains, ct_str, domains_json


def main():
    import argparse
    import shutil

    parser = argparse.ArgumentParser(description="Update domain metadata from proteins.h5")
    parser.add_argument("--proteins-h5", default=str(PROTEINS_H5))
    parser.add_argument("--unified-meta", default=str(UNIFIED_META))
    args = parser.parse_args()

    meta_path = Path(args.unified_meta)
    h5_path = Path(args.proteins_h5)

    bak_path = meta_path.with_suffix(".parquet.pre_domain_update")
    shutil.copy(meta_path, bak_path)
    log.info(f"Backup: {bak_path}")

    # Load unified metadata
    df = pd.read_parquet(meta_path)
    log.info(f"Loaded: {len(df):,} rows")

    # Extract domain info from proteins.h5
    h5 = h5py.File(h5_path, "r")
    unique_hashes = sorted(df["protein_seq_hash"].unique())
    log.info(f"Unique seq_hashes: {len(unique_hashes)}")

    domain_info = {}
    for sh in unique_hashes:
        domain_info[sh] = extract_domain_info(h5, sh)
    h5.close()

    # Apply updates
    df["has_domain_annotation"] = df["protein_seq_hash"].map(lambda sh: domain_info[sh][0])
    df["n_domains"] = df["protein_seq_hash"].map(lambda sh: domain_info[sh][1])
    df["cofactor_domain_types"] = df["protein_seq_hash"].map(lambda sh: domain_info[sh][2])
    df["domains_json"] = df["protein_seq_hash"].map(lambda sh: domain_info[sh][3])

    # Stats
    n_with = df["has_domain_annotation"].sum()
    n_proteins = df["protein_seq_hash"].nunique()
    log.info(f"Proteins with domains: {n_with:,} / {n_proteins}")

    all_cts = []
    for cts in df.loc[df["has_domain_annotation"], "cofactor_domain_types"]:
        if cts:
            all_cts.extend(cts.split(","))
    ct_counts = Counter(all_cts)
    log.info("Cofactor type coverage:")
    for ct, count in ct_counts.most_common():
        log.info(f"  {ct}: {count}")

    # Verify split coverage
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        split_cts = set()
        for cts in split_df.loc[split_df["has_domain_annotation"], "cofactor_domain_types"]:
            if cts:
                split_cts.update(cts.split(","))
        log.info(f"  {split_name}: {len(split_df):,} samples, {len(split_cts)} cofactor types")

    df.to_parquet(meta_path, index=False)
    log.info(f"Saved: {meta_path} ({len(df):,} rows x {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
