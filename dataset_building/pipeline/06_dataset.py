"""
06_dataset.py
===========
PyTorch Dataset and DataLoader for the enzyme-substrate binding model.

EnzymeBindingDataset:
  - Loads from HDF5 (protein structural data) + .pt (ligand graphs) + Parquet (metadata)
  - Returns raw sequence strings (ESM-2 tokenization happens in collate_fn)
  - Handles missing structural features gracefully (BindingDB proteins)

collate_fn:
  - ESM-2 tokenization with dynamic padding to max length in batch
  - PyG Batch.from_data_list for ligand graphs
  - Returns a BatchedSample namedtuple

Usage:
  from dataset import EnzymeBindingDataset, make_dataloader

  train_loader = make_dataloader(
      metadata_parquet = 'processed/metadata.parquet',
      proteins_h5      = 'processed/proteins.h5',
      ligands_dir      = 'processed/ligands',
      split            = 'train',
      batch_size       = 32,
      esm_model_name   = 'esm2_t33_650M_UR50D',
      num_workers      = 4,
  )

  for batch in train_loader:
      # batch.token_ids:       (B, L_max)  — ESM-2 input IDs
      # batch.attention_mask:  (B, L_max)  — 1 for real tokens, 0 for padding
      # batch.binding_site_mask: (B, L_max) — 1 at binding site positions
      # batch.ligand_graph:    PyG Batch   — batched molecular graphs
      # batch.pkd:             (B,)        — target pKd values
      # batch.quality_weight:  (B,)        — per-sample loss weights
      # batch.has_structure:   (B,)        — bool, True if structural features available
      # batch.contact_number:  list[Tensor|None]  — per-sample, None if unavailable
      # batch.protrusion_index: list[Tensor|None] — per-sample, None if unavailable
"""

import io
import logging
from pathlib import Path
from typing import Optional, List
from collections import namedtuple

import numpy as np
import pandas as pd
import h5py
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Output container
# ─────────────────────────────────────────────────────────────

BatchedSample = namedtuple('BatchedSample', [
    'token_ids',          # (B, L_max) int64
    'attention_mask',     # (B, L_max) int64
    'binding_site_mask',  # (B, L_max) float32  (0/1, padded positions = 0)
    'ligand_graph',       # PyG Batch
    'pkd',                # (B,) float32
    'quality_weight',     # (B,) float32
    'has_structure',      # (B,) bool
    'contact_number',     # list[Tensor|None], length B
    'protrusion_index',   # list[Tensor|None], length B
    'sample_ids',         # list[str], length B
])


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

class EnzymeBindingDataset(Dataset):
    """
    Dataset for enzyme-substrate binding affinity prediction.

    Parameters
    ----------
    metadata_parquet : path to metadata.parquet
    proteins_h5      : path to proteins.h5
    ligands_dir      : directory containing {inchikey}.pt files
    split            : 'train', 'val', or 'test' (None = all)
    min_quality_weight : filter out samples below this quality weight
    require_binding_site : if True, exclude samples without binding site annotation
    """

    def __init__(self,
                 metadata_parquet: str,
                 proteins_h5: str,
                 ligands_dir: str,
                 split: Optional[str] = 'train',
                 min_quality_weight: float = 0.0,
                 require_binding_site: bool = False):

        self.ligands_dir = Path(ligands_dir)

        # Load metadata
        meta = pd.read_parquet(metadata_parquet)
        if split is not None:
            meta = meta[meta['split'] == split]
        if min_quality_weight > 0:
            meta = meta[meta['quality_weight'] >= min_quality_weight]
        if require_binding_site:
            meta = meta[meta['has_binding_site']]

        # Drop rows with missing ligand graphs
        def _has_graph(ik):
            return ik is not None and (self.ligands_dir / f'{ik}.pt').exists()
        meta = meta[meta['ligand_inchikey'].apply(_has_graph)]

        self.meta = meta.reset_index(drop=True)
        log.info(f"Dataset [{split}]: {len(self.meta):,} samples")

        # Open HDF5 (keep file handle open for efficiency)
        self.h5 = h5py.File(proteins_h5, 'r')

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> dict:
        row = self.meta.iloc[idx]

        # ── Protein ──────────────────────────────────────────
        seq_hash = row['protein_seq_hash']
        grp      = self.h5[seq_hash]

        # Sequence
        seq_bytes = grp['sequence'][()]
        if isinstance(seq_bytes, bytes):
            sequence = seq_bytes.decode('utf-8')
        else:
            sequence = str(seq_bytes)

        # Binding site mask (index array → dense bool, then to float)
        site_indices = grp['binding_site_mask'][()]   # int32 array
        L = len(sequence)
        site_mask = np.zeros(L, dtype=np.float32)
        if len(site_indices) > 0:
            valid_idx = site_indices[site_indices < L]
            site_mask[valid_idx] = 1.0

        # Structural features (may be absent for BindingDB proteins)
        has_structure = row.get('has_structure', False)
        contact_number   = None
        protrusion_index = None

        if has_structure and 'contact_number' in grp:
            contact_number   = torch.tensor(grp['contact_number'][()],   dtype=torch.float32)
            protrusion_index = torch.tensor(grp['protrusion_index'][()], dtype=torch.float32)
            # Truncate to sequence length (safety check)
            if len(contact_number) > L:
                contact_number   = contact_number[:L]
                protrusion_index = protrusion_index[:L]

        # ── Ligand ───────────────────────────────────────────
        ik         = row['ligand_inchikey']
        lig_path   = self.ligands_dir / f'{ik}.pt'
        lig_graph  = torch.load(str(lig_path), weights_only=False)

        # ── Labels ───────────────────────────────────────────
        pkd            = float(row['pkd_aligned'])
        quality_weight = float(row['quality_weight']) * float(row.get('w_multiplier', 1.0))
        sample_id      = str(row['sample_id'])

        return {
            'sequence':         sequence,
            'site_mask':        torch.tensor(site_mask, dtype=torch.float32),
            'lig_graph':        lig_graph,
            'pkd':              torch.tensor(pkd,            dtype=torch.float32),
            'quality_weight':   torch.tensor(quality_weight, dtype=torch.float32),
            'has_structure':    bool(has_structure),
            'contact_number':   contact_number,    # Tensor or None
            'protrusion_index': protrusion_index,  # Tensor or None
            'sample_id':        sample_id,
        }

    def close(self):
        """Close the HDF5 file handle."""
        self.h5.close()

    def __del__(self):
        try:
            self.h5.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────
# ESM-2 tokenizer (lazy-loaded singleton)
# ─────────────────────────────────────────────────────────────

_esm_tokenizer = None
_esm_model_name = None


def get_esm_tokenizer(model_name: str = 'esm2_t33_650M_UR50D'):
    """
    Lazy-load ESM-2 tokenizer. Cached after first call.

    Uses the ESM library's AutoTokenizer-compatible interface.
    Falls back to fair-esm if transformers ESM is unavailable.
    """
    global _esm_tokenizer, _esm_model_name

    if _esm_tokenizer is not None and _esm_model_name == model_name:
        return _esm_tokenizer

    try:
        from transformers import AutoTokenizer
        _esm_tokenizer = AutoTokenizer.from_pretrained(f'facebook/{model_name}')
        log.info(f"Loaded ESM-2 tokenizer via transformers: {model_name}")
    except Exception:
        try:
            import esm
            _, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
            _esm_tokenizer = alphabet.get_batch_converter()
            log.info(f"Loaded ESM-2 tokenizer via fair-esm: {model_name}")
        except Exception as e:
            raise RuntimeError(
                f"Could not load ESM-2 tokenizer for {model_name}. "
                f"Install: pip install transformers or pip install fair-esm\n{e}"
            )

    _esm_model_name = model_name
    return _esm_tokenizer


# ─────────────────────────────────────────────────────────────
# collate_fn
# ─────────────────────────────────────────────────────────────

def make_collate_fn(esm_model_name: str = 'esm2_t33_650M_UR50D',
                    max_seq_len: int = 1020):
    """
    Factory that returns a collate_fn with the ESM-2 tokenizer baked in.

    The returned collate_fn:
      1. Tokenizes sequences with dynamic padding to max length in batch
      2. Pads site_mask to match token length (accounting for <cls>/<eos>)
      3. Batches ligand graphs with PyG Batch.from_data_list
      4. Stacks scalar tensors
    """
    tokenizer = get_esm_tokenizer(esm_model_name)

    def collate_fn(samples: List[dict]) -> BatchedSample:
        sequences        = [s['sequence'][:max_seq_len] for s in samples]
        site_masks       = [s['site_mask'][:max_seq_len] for s in samples]
        lig_graphs       = [s['lig_graph']               for s in samples]
        pkds             = torch.stack([s['pkd']            for s in samples])
        quality_weights  = torch.stack([s['quality_weight'] for s in samples])
        has_structures   = torch.tensor([s['has_structure'] for s in samples], dtype=torch.bool)
        contact_numbers  = [s['contact_number']   for s in samples]
        protrusion_idxs  = [s['protrusion_index'] for s in samples]
        sample_ids       = [s['sample_id']        for s in samples]

        # ESM-2 tokenization
        # transformers interface: tokenizer(sequences, return_tensors='pt', padding=True)
        encoding = tokenizer(
            sequences,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_seq_len + 2,   # +2 for <cls> and <eos>
        )
        token_ids      = encoding['input_ids']       # (B, L_tok)
        attention_mask = encoding['attention_mask']  # (B, L_tok)

        # Pad site_mask to match token length
        # ESM-2 adds <cls> at position 0 and <eos> at the end
        # site_mask is aligned to the sequence (no special tokens)
        # → prepend 0 for <cls>, append 0 for <eos>, then pad to L_tok
        L_tok = token_ids.shape[1]
        padded_site_masks = []
        for sm in site_masks:
            # sm: (L_seq,) float32
            # After tokenization: [<cls>, aa_1, ..., aa_L, <eos>, <pad>, ...]
            sm_with_special = torch.cat([
                torch.zeros(1, dtype=torch.float32),   # <cls>
                sm,
                torch.zeros(1, dtype=torch.float32),   # <eos>
            ])
            # Pad to L_tok
            pad_len = L_tok - len(sm_with_special)
            if pad_len > 0:
                sm_with_special = torch.cat([
                    sm_with_special,
                    torch.zeros(pad_len, dtype=torch.float32)
                ])
            else:
                sm_with_special = sm_with_special[:L_tok]
            padded_site_masks.append(sm_with_special)

        binding_site_mask = torch.stack(padded_site_masks)   # (B, L_tok)

        # Batch ligand graphs
        ligand_graph = Batch.from_data_list(lig_graphs)

        return BatchedSample(
            token_ids         = token_ids,
            attention_mask    = attention_mask,
            binding_site_mask = binding_site_mask,
            ligand_graph      = ligand_graph,
            pkd               = pkds,
            quality_weight    = quality_weights,
            has_structure     = has_structures,
            contact_number    = contact_numbers,
            protrusion_index  = protrusion_idxs,
            sample_ids        = sample_ids,
        )

    return collate_fn


# ─────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────

def make_dataloader(metadata_parquet: str,
                    proteins_h5: str,
                    ligands_dir: str,
                    split: str = 'train',
                    batch_size: int = 32,
                    esm_model_name: str = 'esm2_t33_650M_UR50D',
                    num_workers: int = 4,
                    shuffle: Optional[bool] = None,
                    min_quality_weight: float = 0.0,
                    require_binding_site: bool = False,
                    pin_memory: bool = True) -> DataLoader:
    """
    Create a DataLoader for the specified split.

    Parameters
    ----------
    metadata_parquet     : path to metadata.parquet
    proteins_h5          : path to proteins.h5
    ligands_dir          : directory with {inchikey}.pt files
    split                : 'train', 'val', or 'test'
    batch_size           : samples per batch
    esm_model_name       : ESM-2 model name for tokenizer
    num_workers          : DataLoader worker processes
    shuffle              : default True for train, False otherwise
    min_quality_weight   : filter low-quality samples
    require_binding_site : exclude samples without binding site annotation
    pin_memory           : pin memory for faster GPU transfer

    Returns
    -------
    DataLoader yielding BatchedSample namedtuples
    """
    if shuffle is None:
        shuffle = (split == 'train')

    dataset = EnzymeBindingDataset(
        metadata_parquet     = metadata_parquet,
        proteins_h5          = proteins_h5,
        ligands_dir          = ligands_dir,
        split                = split,
        min_quality_weight   = min_quality_weight,
        require_binding_site = require_binding_site,
    )

    collate_fn = make_collate_fn(esm_model_name=esm_model_name)

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        collate_fn  = collate_fn,
        pin_memory  = pin_memory and torch.cuda.is_available(),
        drop_last   = (split == 'train'),
        persistent_workers = (num_workers > 0),
    )

    return loader


# ─────────────────────────────────────────────────────────────
# Quick sanity check (run directly)
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test dataset loading')
    parser.add_argument('--metadata',  required=True)
    parser.add_argument('--proteins',  required=True)
    parser.add_argument('--ligands',   required=True)
    parser.add_argument('--split',     default='train')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--n-batches',  type=int, default=3)
    args = parser.parse_args()

    loader = make_dataloader(
        metadata_parquet = args.metadata,
        proteins_h5      = args.proteins,
        ligands_dir      = args.ligands,
        split            = args.split,
        batch_size       = args.batch_size,
        num_workers      = 0,   # single-process for debugging
    )

    for i, batch in enumerate(loader):
        if i >= args.n_batches:
            break
        print(f"\nBatch {i+1}:")
        print(f"  token_ids:         {batch.token_ids.shape}")
        print(f"  attention_mask:    {batch.attention_mask.shape}")
        print(f"  binding_site_mask: {batch.binding_site_mask.shape}")
        print(f"  ligand_graph:      {batch.ligand_graph}")
        print(f"  pkd:               {batch.pkd}")
        print(f"  quality_weight:    {batch.quality_weight}")
        print(f"  has_structure:     {batch.has_structure}")
        print(f"  sample_ids:        {batch.sample_ids}")

    print("\nDataset sanity check passed.")
