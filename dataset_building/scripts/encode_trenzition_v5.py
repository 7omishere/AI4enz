#!/usr/bin/env python3
"""
encode_trenzition_v5.py
=======================
将 trenzition V5 数据集转换为模型训练所需格式：

1. 构建 metadata.parquet (V5 列映射 + 必需列 + 全平权)
2. 计算 ESM-2 嵌入 (esm2_t33_650M_UR50D, 1280-dim mean-pooled)
3. 写入 proteins.h5 (追加新蛋白)
4. 按 protein_seq_hash 做 70/15/15 蛋白层级划分

输出：
  - processed/metadata.parquet
  - processed/proteins.h5 (更新)

用法：
  source /home/domi/BINN/.venv/bin/activate
  python scripts/encode_trenzition_v5.py [--skip-esm] [--device cuda]
"""

import hashlib
import argparse
import logging
import sys
import time
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import h5py
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "processed"
RELEASE_DIR = BASE_DIR / "release"

LOG_DIR = PROCESSED_DIR
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "encode_esm2.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

V5_INPUT = RELEASE_DIR / "trenzition_full_v5.parquet"
H5_PATH = PROCESSED_DIR / "proteins.h5"
METADATA_OUT = PROCESSED_DIR / "metadata.parquet"
STATS_OUT = RELEASE_DIR / "trenzition_encode_stats_v5.json"

# ─────────────────────────────────────────────────────────────
# Split config
# ─────────────────────────────────────────────────────────────
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────
# kcat_source mapping
# ─────────────────────────────────────────────────────────────
KCAT_SOURCE_MAP = {
    "catapro":               "bdb",       # BRENDA-derived
    "homology_transfer":     "uniprot",   # transfer
    "oed":                   "oed",
    "skid":                  "skid",
    "sabio":                 "sabio",
    "rts":                   "rts",
    "BindingDB_protein_level": "bdb",
}

def compute_seq_hash(seq: str) -> str:
    return hashlib.sha256(str(seq).upper().encode()).hexdigest()[:16]


def build_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """从 V5 构建训练所需 metadata.parquet"""
    log.info("Building metadata from V5...")

    meta = pd.DataFrame()

    # ── 直接映射 ──
    meta["protein_seq_hash"] = df["protein_seq_hash"]
    meta["ligand_inchikey"]  = df["ligand_inchikey"]
    meta["uniprot_id"]       = df["uniprot_id"]
    meta["ec_numbers"]       = df["ec_number"]
    meta["sequence"]         = df["sequence"]

    # ── pKd 字段 ──
    meta["pkd_raw"]     = df["pkd_value"]
    meta["pkd_aligned"] = df["pkd_value"]  # 无 calibration step，直接用

    # ── measurement_type ──
    # 注意: improve_dataset.py 已清理 measurement_type_pkd:
    #   - 有 pKd 但 type 缺失 → "Kd"
    #   - 无 pKd → "" (不再错误标记为 Kd)
    meta["measurement_type"] = df["measurement_type_pkd"].fillna("")
    # 旧版的 fillna("Kd") 将无 pKd 的行也错标为 Kd，已在 improve_dataset.py 修复

    # ── kcat 字段 ──
    kcat_vals = df["kcat_per_s"].copy()
    kcat_pos = kcat_vals.fillna(0).clip(lower=1e-10)  # 避免 log(0)
    meta["log_kcat_median"] = np.log10(kcat_pos)
    meta["has_kcat"] = df["kcat_per_s"].notna()

    # ── kcat_source ──
    meta["kcat_source"] = df["kcat_per_s_source"].map(KCAT_SOURCE_MAP).fillna("unknown")

    # ── Km 字段（三头 KmHead 使用）──
    km_raw = df["km_M"].copy()
    km_pos = km_raw[km_raw > 0]  # Km > 0 才有效
    meta["km_M"] = km_raw
    meta["has_km"] = km_raw.notna() & (km_raw > 0)
    meta["log_km"] = np.nan
    meta.loc[meta["has_km"], "log_km"] = np.log10(km_pos)

    # ── 测量类型编码（三头 BindingHead 使用）──
    # Kd=0, Ki=1, IC50_approx=2, ""=3
    mtype_map = {"Kd": 0, "Ki": 1, "IC50_approx": 2, "": 3, "IC50": 2}
    meta["measurement_type_encoded"] = meta["measurement_type"].map(mtype_map).fillna(3).astype(int)

    # ── 全平权（用户要求） ──
    meta["quality_weight"] = 1.0
    meta["w_multiplier"]   = 1.0

    # ── 缺失的默认值 ──
    meta["has_structure"]        = False
    meta["has_binding_site"]     = False
    meta["has_domain_annotation"] = False
    meta["n_domains"]            = 0
    meta["cofactors"]            = ""
    meta["cofactor_domain_types"] = ""
    meta["domains_json"]         = "[]"
    meta["is_censored"]          = False
    meta["n_measurements"]       = df["n_pkd_measurements"].fillna(1)
    meta["pkd_std"]              = df["pkd_std"].fillna(0.0)
    meta["source_db"]            = df["data_source"]
    meta["pdb_id"]               = ""
    meta["sample_id"]            = [f"v5_{i}" for i in range(len(meta))]

    # ── 筛选有效标签（至少有一个） ──
    n_before = len(meta)
    has_any_label = meta["pkd_raw"].notna() | meta["has_kcat"]
    meta = meta[has_any_label].reset_index(drop=True)
    log.info(f"  Rows with any label: {len(meta):,} (filtered {n_before - len(meta):,} no-label)")

    # ── 统计 ──
    log.info(f"  Metadata shape: {len(meta):,} × {len(meta.columns)}")
    log.info(f"  With pKd:  {meta['pkd_raw'].notna().sum():,}")
    log.info(f"  With kcat: {meta['has_kcat'].sum():,}")
    log.info(f"  With both: {(meta['pkd_raw'].notna() & meta['has_kcat']).sum():,}")

    return meta


def make_splits(meta: pd.DataFrame,
                val_frac: float = 0.15,
                test_frac: float = 0.15,
                seed: int = 42) -> pd.DataFrame:
    """按 protein_seq_hash 做蛋白层级划分（零泄漏）"""
    log.info(f"Creating protein-level splits (val={val_frac}, test={test_frac})...")

    rng = np.random.default_rng(seed)
    hashes = meta["protein_seq_hash"].dropna().unique()
    n = len(hashes)
    log.info(f"  Unique proteins: {n:,}")

    perm = rng.permutation(hashes)
    n_test = max(1, int(n * test_frac))
    n_val  = max(1, int(n * val_frac))

    test_hashes = set(perm[:n_test])
    val_hashes  = set(perm[n_test:n_test + n_val])

    split_map = {}
    for h in hashes:
        if h in test_hashes:
            split_map[h] = "test"
        elif h in val_hashes:
            split_map[h] = "val"
        else:
            split_map[h] = "train"

    meta["split"] = meta["protein_seq_hash"].map(split_map).fillna("train")

    # 验证零泄漏
    train_prots = set(meta[meta["split"] == "train"]["protein_seq_hash"])
    val_prots   = set(meta[meta["split"] == "val"]["protein_seq_hash"])
    test_prots  = set(meta[meta["split"] == "test"]["protein_seq_hash"])

    assert not (train_prots & val_prots), "Train-Val protein leakage!"
    assert not (train_prots & test_prots), "Train-Test protein leakage!"
    assert not (val_prots & test_prots), "Val-Test protein leakage!"
    log.info("  ✓ Zero protein leakage verified")

    for s in ["train", "val", "test"]:
        sub = meta[meta["split"] == s]
        n_prot = sub["protein_seq_hash"].nunique()
        n_pkd  = sub["pkd_raw"].notna().sum()
        n_kcat = sub["has_kcat"].sum()
        log.info(f"  {s:>5}: {len(sub):>7,} samples, {n_prot:>5,} proteins, "
                 f"pKd={n_pkd:,}, kcat={n_kcat:,}")

    return meta


def compute_esm2_embeddings(sequences: list[str],
                            seq_hashes: list[str],
                            device: str = "cuda",
                            batch_size: int = 4) -> dict[str, np.ndarray]:
    """批量计算 ESM-2 mean-pooled embeddings。

    Returns: {seq_hash: np.ndarray(1280,)}
    """
    log.info(f"Computing ESM-2 embeddings for {len(sequences):,} sequences...")

    import esm

    # Load model
    log.info("  Loading esm2_t33_650M_UR50D...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()

    batch_converter = alphabet.get_batch_converter()
    results = {}

    n_batches = (len(sequences) + batch_size - 1) // batch_size
    start_time = time.time()

    for batch_idx in range(n_batches):
        lo = batch_idx * batch_size
        hi = min(lo + batch_size, len(sequences))
        batch_seqs = sequences[lo:hi]
        batch_hashes = seq_hashes[lo:hi]

        # Prepare batch (id, seq) pairs
        batch_data = [(f"seq_{i}", s) for i, s in enumerate(batch_seqs)]

        try:
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                results_dict = model(batch_tokens, repr_layers=[33])
                token_repr = results_dict["representations"][33]  # (B, L, 1280)

            # Mean pooling over sequence positions (exclude BOS/EOS)
            for i in range(len(batch_seqs)):
                # Token 0 = BOS, token L-1 = EOS. Pool tokens 1..L-2
                seq_len = len(batch_seqs[i])
                if seq_len > 0:
                    rep = token_repr[i, 1:1+seq_len, :].mean(dim=0).cpu().numpy()
                else:
                    rep = token_repr[i, 1:-1, :].mean(dim=0).cpu().numpy()

                results[batch_hashes[i]] = rep.astype(np.float32)

        except Exception as e:
            log.warning(f"  Batch {batch_idx} failed: {e}, computing individually...")
            # Fallback: one by one
            for j, (seq, h) in enumerate(zip(batch_seqs, batch_hashes)):
                try:
                    single_data = [(f"seq_0", seq)]
                    _, _, tokens = batch_converter(single_data)
                    tokens = tokens.to(device)
                    with torch.no_grad():
                        r = model(tokens, repr_layers=[33])
                        rep = r["representations"][33][0, 1:1+len(seq), :].mean(dim=0).cpu().numpy()
                    results[h] = rep.astype(np.float32)
                except Exception as e2:
                    log.error(f"  Failed seq {h[:12]}: {e2}")
                    results[h] = np.zeros(1280, dtype=np.float32)

        # Progress
        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (batch_idx + 1) * batch_size / elapsed
            eta = (len(sequences) - hi) / rate
            log.info(f"  [{hi:,}/{len(sequences):,}] {rate:.1f} seq/s, ETA {eta/60:.0f} min")
            # Write progress file (for external monitoring)
            with open(LOG_DIR / "encode_esm2_progress.txt", "w") as pf:
                pf.write(f"done={hi} total={len(sequences)} pct={100*hi/len(sequences):.1f} "
                        f"rate={rate:.1f} eta_min={eta/60:.0f} elapsed_min={elapsed/60:.0f}\n")

    elapsed = time.time() - start_time
    log.info(f"  Done: {len(results):,} embeddings in {elapsed/60:.1f} min "
             f"({len(results)/elapsed:.1f} seq/s)")

    # Final progress
    with open(LOG_DIR / "encode_esm2_progress.txt", "w") as pf:
        pf.write(f"DONE total={len(results)} elapsed_min={elapsed/60:.0f}\n")

    return results


def write_proteins_h5(h5_path: Path,
                      sequences: dict[str, str],
                      esm2_embeds: dict[str, np.ndarray]):
    """将新蛋白序列 + ESM-2 嵌入写入 proteins.h5（追加模式）"""
    log.info(f"Writing to {h5_path}...")

    n_new = 0
    n_esm2 = 0

    with h5py.File(h5_path, "a") as h5:
        for seq_hash, seq in sequences.items():
            if seq_hash in h5:
                # 已存在 — 仅补充缺失的 ESM-2
                grp = h5[seq_hash]
                if "esm2_embed" not in grp and seq_hash in esm2_embeds:
                    grp.create_dataset("esm2_embed", data=esm2_embeds[seq_hash])
                    n_esm2 += 1
                if "sequence" not in grp:
                    grp.create_dataset("sequence", data=seq.encode("utf-8"))
                continue

            # 新建
            grp = h5.create_group(seq_hash)
            grp.create_dataset("sequence", data=seq.encode("utf-8"))
            grp.create_dataset("binding_site_mask", data=np.array([], dtype=np.int32))

            if seq_hash in esm2_embeds:
                grp.create_dataset("esm2_embed", data=esm2_embeds[seq_hash])
                n_esm2 += 1

            n_new += 1

    log.info(f"  New proteins: {n_new:,}, ESM-2 added: {n_esm2:,}")


def main():
    parser = argparse.ArgumentParser(description="Encode trenzition V5 for training")
    parser.add_argument("--skip-esm", action="store_true",
                        help="Skip ESM-2 computation (use AA-prop fallback)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--esm-batch-size", type=int, default=4,
                        help="Batch size for ESM-2 (default: 4 for T33 650M on <16GB VRAM)")
    args = parser.parse_args()

    log.info(f"Device: {args.device}")
    log.info(f"Skip ESM: {args.skip_esm}")

    # ── Step 1: Load V5 ──
    df = pd.read_parquet(V5_INPUT)
    log.info(f"Loaded V5: {len(df):,} rows")

    # ── Step 2: Build metadata ──
    meta = build_metadata(df)

    # ── Step 3: ESM-2 embeddings ──
    if not args.skip_esm:
        # 需要计算 ESM-2 的新蛋白
        with h5py.File(H5_PATH, "r") as h5:
            existing_hashes = set(h5.keys())
            existing_esm2 = {k for k in existing_hashes if "esm2_embed" in h5[k]}

        all_hashes = set(meta["protein_seq_hash"].dropna().unique())
        new_hashes = all_hashes - existing_hashes
        need_esm2 = all_hashes - existing_esm2

        log.info(f"  Total unique proteins: {len(all_hashes):,}")
        log.info(f"  In h5: {len(all_hashes & existing_hashes):,}")
        log.info(f"  New (not in h5): {len(new_hashes):,}")
        log.info(f"  Need ESM-2: {len(need_esm2):,}")

        # 收集所有需要 ESM-2 的序列（dedup by seq_hash）
        hash_to_seq = {}
        for h in need_esm2:
            seq_rows = meta[meta["protein_seq_hash"] == h]
            if len(seq_rows) > 0:
                seq = seq_rows["sequence"].iloc[0]
                if isinstance(seq, str) and seq:
                    hash_to_seq[h] = seq

        # 对已存在但缺 ESM-2 的蛋白也收集序列
        for h in existing_hashes - existing_esm2:
            if h in all_hashes:
                seq_rows = meta[meta["protein_seq_hash"] == h]
                if len(seq_rows) > 0:
                    seq = seq_rows["sequence"].iloc[0]
                    if isinstance(seq, str) and seq:
                        hash_to_seq[h] = seq

        log.info(f"  Sequences to embed: {len(hash_to_seq):,}")

        if hash_to_seq:
            seq_hashes = list(hash_to_seq.keys())
            seq_list   = [hash_to_seq[h] for h in seq_hashes]

            esm2_embeds = compute_esm2_embeddings(
                seq_list, seq_hashes,
                device=args.device,
                batch_size=args.esm_batch_size,
            )

            # Write to h5
            write_proteins_h5(H5_PATH, hash_to_seq, esm2_embeds)
    else:
        log.info("  Skipping ESM-2 (--skip-esm)")

    # ── Step 4: Create splits ──
    meta = make_splits(meta, val_frac=VAL_FRAC, test_frac=TEST_FRAC, seed=RANDOM_SEED)

    # ── Step 5: Save metadata ──
    # Drop sequence from metadata (stored in h5)
    meta_out = meta.drop(columns=["sequence"])
    meta_out.to_parquet(METADATA_OUT, index=False, compression="snappy")
    log.info(f"Metadata saved → {METADATA_OUT} ({len(meta_out):,} rows × {len(meta_out.columns)} cols)")

    # ── Step 6: Stats ──
    with h5py.File(H5_PATH, "r") as h5:
        v5_hashes = set(meta["protein_seq_hash"].unique())
        in_h5 = v5_hashes & set(h5.keys())
        in_esm2 = sum(1 for k in in_h5 if "esm2_embed" in h5[k])

    stats = {
        "total_samples": len(meta),
        "total_proteins": len(v5_hashes),
        "proteins_in_h5": len(in_h5),
        "proteins_with_esm2": in_esm2,
        "esm2_coverage": f"{100*in_esm2/len(v5_hashes):.1f}%",
        "splits": {
            s: {
                "samples": int((meta["split"] == s).sum()),
                "proteins": int(meta[meta["split"] == s]["protein_seq_hash"].nunique()),
                "with_pkd": int(meta[meta["split"] == s]["pkd_raw"].notna().sum()),
                "with_kcat": int(meta[meta["split"] == s]["has_kcat"].sum()),
            }
            for s in ["train", "val", "test"]
        },
    }
    with open(STATS_OUT, "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Stats saved → {STATS_OUT}")

    # ── Final summary ──
    log.info("\n" + "=" * 60)
    log.info("ENCODING COMPLETE")
    log.info("=" * 60)
    log.info(f"  metadata.parquet: {len(meta):,} samples")
    log.info(f"  proteins.h5: {in_esm2}/{len(v5_hashes)} proteins with ESM-2")
    log.info(f"  Splits: train={stats['splits']['train']['samples']:,}, "
             f"val={stats['splits']['val']['samples']:,}, "
             f"test={stats['splits']['test']['samples']:,}")


if __name__ == "__main__":
    main()
