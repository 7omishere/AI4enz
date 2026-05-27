"""
08_make_splits.py
================
Create train/val/test splits by protein sequence clustering to prevent leakage.

Strategy:
  1. Group records by protein_seq_hash
  2. Randomly assign hash groups to train/val/test (80/10/10)
  3. Write new metadata.parquet with updated 'split' column

Usage:
  python 08_make_splits.py [--metadata processed/metadata.parquet] [--seed 42]
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


def make_splits(df: pd.DataFrame, val_frac: float = 0.10, test_frac: float = 0.10,
                seed: int = 42) -> pd.DataFrame:
    """Assign split labels by protein sequence group."""
    rng = np.random.default_rng(seed)

    # Unique protein hashes
    hashes = df['protein_seq_hash'].dropna().unique()
    n = len(hashes)
    log.info(f"Unique protein sequences: {n:,}")

    # Shuffle and split
    perm = rng.permutation(hashes)
    n_test = max(1, int(n * test_frac))
    n_val  = max(1, int(n * val_frac))

    test_hashes = set(perm[:n_test])
    val_hashes  = set(perm[n_test:n_test + n_val])

    # Assign splits
    split_map = {}
    for h in hashes:
        if h in test_hashes:
            split_map[h] = 'test'
        elif h in val_hashes:
            split_map[h] = 'val'
        else:
            split_map[h] = 'train'

    df['split'] = df['protein_seq_hash'].map(split_map).fillna('train')

    # Count
    for s in ['train', 'val', 'test']:
        sub = df[df['split'] == s]
        log.info(f"  {s}: {len(sub):,} records, {sub['protein_seq_hash'].nunique():,} proteins")

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create sequence-based data splits')
    parser.add_argument('--metadata', default=None,
                        help='Path to metadata.parquet (default: ../processed/metadata.parquet)')
    parser.add_argument('--output', default=None,
                        help='Output path (default: ../processed/metadata.parquet, overwrites)')
    parser.add_argument('--val-frac',  type=float, default=0.10)
    parser.add_argument('--test-frac', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    default_meta = script_dir.parent / 'processed' / 'metadata.parquet'

    meta_path = args.metadata or str(default_meta)
    out_path  = args.output   or str(default_meta)

    df = pd.read_parquet(meta_path)
    log.info(f"Loaded {len(df):,} records")

    df = make_splits(df, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)

    # Also update proteins.h5 with split info (we'll just overwrite parquet)
    df.to_parquet(out_path, index=False, compression='snappy')
    log.info(f"Saved → {out_path}")

    # Summary stats per split
    print()
    for s in ['train', 'val', 'test']:
        sub = df[df['split'] == s]
        print(f"{s}: {len(sub):,} records, pkd_aligned μ={sub['pkd_aligned'].mean():.2f} σ={sub['pkd_aligned'].std():.2f}")
