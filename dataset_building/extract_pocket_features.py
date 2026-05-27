"""
从 AlphaFold 结构数据（已有 contact_map + binding_site_mask）提取口袋残基特征。

输出写入 proteins.h5 的新字段：
  - pocket_contact_number: (K,) 口袋残基的接触数
  - pocket_protrusion_index: (K,) 口袋残基的突起指数
  - pocket_ca_distances: (K, K) 口袋残基间的近似Cα距离（从contact_map估算）
  - pocket_residue_ids: (K,) 口袋残基序号（相对于全长序列）

用法：
  source /home/domi/BINN/.venv/bin/activate
  cd /home/domi/BINN/AI4enz/dataset_building
  python extract_pocket_features.py
"""

import json
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from io import BytesIO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent
PROTEIN_H5 = BASE / "processed" / "proteins.h5"
META_PATH = BASE / "processed" / "oxidoreductase" / "unified_metadata.parquet"

CONTACT_CUTOFF = 8.0  # Å


def load_contact_map(group: h5py.Group) -> np.ndarray:
    """Load contact map from HDF5 group, handling sparse format."""
    cm_data = group["contact_map"][:]
    if group.attrs.get("contact_map_sparse", False):
        csr = load_npz(BytesIO(cm_data.tobytes()))
        return csr.toarray().astype(bool)
    return cm_data.astype(bool)


def estimate_distance_matrix(contact_map: np.ndarray, residue_indices: np.ndarray) -> np.ndarray:
    """Estimate pairwise Cα distances for pocket residues from binary contact map.

    Strategy:
    - If contact_map[i,j] = True:  distance ≈ 4Å (typical Cα contact midpoint)
    - If contact_map[i,j] = False: distance ≈ 12Å (outside contact range)
    - Self-distance (i=j): 0

    This is an approximation; exact distances require re-downloading AF PDB files.
    """
    K = len(residue_indices)
    sub_cm = contact_map[np.ix_(residue_indices, residue_indices)]  # (K, K)

    D = np.full((K, K), 12.0, dtype=np.float32)  # default: non-contact
    D[sub_cm] = 4.0  # contact
    np.fill_diagonal(D, 0.0)  # self
    return D


def extract_pocket_features_for_protein(
    group: h5py.Group,
    seq_hash: str,
) -> dict | None:
    """Extract pocket features for one protein. Returns None if no binding site."""
    bs_mask = group["binding_site_mask"][:]
    if len(bs_mask) == 0 or bs_mask.sum() == 0:
        return None

    # binding_site_mask contains residue indices
    residue_indices = np.asarray(bs_mask, dtype=np.int64)

    cn = group["contact_number"][:]
    pi = group["protrusion_index"][:]

    # Per-residue features for pocket residues
    pocket_cn = cn[residue_indices].astype(np.float32)
    pocket_pi = pi[residue_indices].astype(np.float32)

    # Approximate distance matrix from contact_map
    contact_map = load_contact_map(group)
    pocket_dist = estimate_distance_matrix(contact_map, residue_indices)

    return {
        "pocket_contact_number": pocket_cn,
        "pocket_protrusion_index": pocket_pi,
        "pocket_ca_distances": pocket_dist,
        "pocket_residue_ids": residue_indices.astype(np.int32),
        "n_pocket_residues": len(residue_indices),
    }


def main():
    # Load oxidoreductase protein hashes
    meta = pd.read_parquet(META_PATH)
    ox_hashes = set(meta["protein_seq_hash"].unique())
    log.info(f"Oxidoreductase proteins: {len(ox_hashes)}")

    processed = 0
    skipped = 0
    no_binding = 0

    with h5py.File(PROTEIN_H5, "r+") as h5:
        for seq_hash in sorted(ox_hashes):
            if seq_hash not in h5:
                log.warning(f"  {seq_hash} not in proteins.h5")
                skipped += 1
                continue

            group = h5[seq_hash]

            # Skip if already has pocket features
            if "pocket_ca_distances" in group:
                skipped += 1
                continue

            features = extract_pocket_features_for_protein(group, seq_hash)
            if features is None:
                no_binding += 1
                continue

            # Write to HDF5
            for key, value in features.items():
                if key == "n_pocket_residues":
                    group.attrs["n_pocket_residues"] = value
                else:
                    if key in group:
                        del group[key]
                    group.create_dataset(
                        key, data=value,
                        compression="gzip", compression_opts=4,
                    )

            processed += 1
            if processed % 50 == 0:
                log.info(f"  Processed {processed}...")

    log.info(f"Done: {processed} processed, {skipped} skipped, {no_binding} no binding site")


if __name__ == "__main__":
    main()
