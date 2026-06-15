"""
compute_esm2_embeddings.py
=========================
预计算 ESM-2 (650M) 蛋白质嵌入，存储到 proteins.h5。

对每条蛋白序列执行 ESM-2 前向传播，取 CLS token (position 0) 的 1280-dim
嵌入作为整个蛋白质的向量表征。

支持批量推理（CPU/GPU），大幅提升吞吐量。

用法：
  python scripts/processors/compute_esm2_embeddings.py
  python scripts/processors/compute_esm2_embeddings.py --device cuda --batch-size 16
  python scripts/processors/compute_esm2_embeddings.py --max-proteins 20   # 测试
  python scripts/processors/compute_esm2_embeddings.py --resume             # 断点续传
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent / "dataset_building"
PROCESSED_DIR = PROJECT_DIR / "processed"
PROTEINS_H5 = PROCESSED_DIR / "proteins.h5"

ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"


def load_esm2(device: str = "cpu"):
    """加载 ESM-2 模型和 tokenizer"""
    from transformers import AutoTokenizer, EsmModel

    log.info(f"Loading ESM-2: {ESM2_MODEL}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL)
    model = EsmModel.from_pretrained(ESM2_MODEL)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    log.info(f"  Loaded on {device} in {time.time()-t0:.1f}s "
             f"({sum(p.numel() for p in model.parameters()):,} params)")
    return tokenizer, model


@torch.no_grad()
def embed_batch(
    sequences: list[str],
    tokenizer,
    model,
    device: str = "cpu",
    max_len: int = 1022,
) -> np.ndarray:
    """批量计算 CLS token 的 1280-dim 嵌入 → (B, 1280) float32"""
    tokens = tokenizer(
        [s[:max_len - 2] for s in sequences],
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        padding=True,
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}
    outputs = model(**tokens)
    cls_embeds = outputs.last_hidden_state[:, 0, :]  # (B, 1280)
    return cls_embeds.cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Precompute ESM-2 embeddings (batch)")
    parser.add_argument("--proteins-h5", default=str(PROTEINS_H5))
    parser.add_argument("--max-proteins", type=int, default=None,
                        help="Limit number of proteins (for testing)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for inference (default: 4 for CPU, 16+ for GPU)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip proteins that already have esm2_embed")
    parser.add_argument("--save-interval", type=int, default=500,
                        help="Flush H5 every N proteins")
    args = parser.parse_args()

    # 设置 HF 镜像（国内加速）
    if "HF_ENDPOINT" not in os.environ:
        # 尝试自动检测
        pass

    device = args.device
    log.info(f"Device: {device}, batch_size: {args.batch_size}")

    tokenizer, model = load_esm2(device)

    h5_path = Path(args.proteins_h5)
    log.info(f"Opening proteins.h5: {h5_path}")

    # ── 收集需要处理的蛋白 ──
    with h5py.File(h5_path, "r") as h5_read:
        all_keys = sorted(h5_read.keys())
        if args.resume:
            todo = [k for k in all_keys if "esm2_embed" not in h5_read[k]]
        else:
            todo = list(all_keys)

    n_total = len(all_keys)
    n_done = n_total - len(todo)
    if args.max_proteins:
        todo = todo[:args.max_proteins]

    log.info(f"  Total proteins: {n_total:,}, already have ESM-2: {n_done:,}, "
             f"to process: {len(todo):,}")

    if not todo:
        log.info("All proteins already have ESM-2 embeddings. Nothing to do.")
        return

    # 预估时间
    est_sec_per_seq = 2.0 if device == "cpu" else 0.05
    est_hours = len(todo) * est_sec_per_seq / args.batch_size / 3600
    log.info(f"  Estimated time: ~{est_hours:.1f} hours")

    # ── 批量处理 ──
    h5 = h5py.File(h5_path, "r+")
    bs = args.batch_size
    n_computed = 0
    n_errors = 0
    t_start = time.time()

    # 预加载所有待处理序列到内存（更快）
    log.info("Preloading sequences...")
    seq_map = {}  # seq_hash → sequence string
    for h in tqdm(todo, desc="Reading sequences"):
        try:
            seq_bytes = h5[h]["sequence"][()]
            if isinstance(seq_bytes, bytes):
                seq = seq_bytes.decode("utf-8")
            else:
                seq = str(seq_bytes)
            if seq:
                seq_map[h] = seq
        except Exception:
            n_errors += 1

    todo_hashes = list(seq_map.keys())
    log.info(f"  {len(todo_hashes):,} valid sequences loaded ({n_errors} errors)")

    pbar = tqdm(total=len(todo_hashes), desc="Embedding", unit="seq")

    for i in range(0, len(todo_hashes), bs):
        batch_hashes = todo_hashes[i:i + bs]
        batch_seqs = [seq_map[h] for h in batch_hashes]

        try:
            embeds = embed_batch(batch_seqs, tokenizer, model, device)  # (B, 1280)

            for j, h in enumerate(batch_hashes):
                grp = h5[h]
                if "esm2_embed" in grp:
                    del grp["esm2_embed"]
                grp.create_dataset("esm2_embed", data=embeds[j])
                n_computed += 1

        except Exception as e:
            log.warning(f"  Batch {i//bs} error: {e}")
            # 逐个处理失败的 batch
            for h, seq in zip(batch_hashes, batch_seqs):
                try:
                    tokens = tokenizer(
                        seq[:1020], return_tensors="pt", truncation=True,
                        max_length=1022, padding=False,
                    )
                    tokens = {k: v.to(device) for k, v in tokens.items()}
                    outputs = model(**tokens)
                    embed = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy().astype(np.float32)
                    grp = h5[h]
                    if "esm2_embed" in grp:
                        del grp["esm2_embed"]
                    grp.create_dataset("esm2_embed", data=embed)
                    n_computed += 1
                except Exception as e2:
                    log.warning(f"    {h}: {e2}")
                    n_errors += 1

        pbar.update(len(batch_hashes))

        # 定期刷新
        if n_computed % args.save_interval == 0 and n_computed > 0:
            h5.flush()
            elapsed = time.time() - t_start
            rate = n_computed / elapsed
            eta = (len(todo_hashes) - n_computed) / rate / 3600
            pbar.set_postfix({
                "rate": f"{rate:.2f} seq/s",
                "eta_h": f"{eta:.1f}",
            })

    pbar.close()
    h5.flush()
    h5.close()

    elapsed = time.time() - t_start
    rate = n_computed / elapsed if elapsed > 0 else 0
    log.info(f"Done: {n_computed:,} computed, {n_errors} errors "
             f"in {elapsed/3600:.1f}h ({rate:.2f} seq/s)")


if __name__ == "__main__":
    main()
