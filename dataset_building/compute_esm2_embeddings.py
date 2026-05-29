"""
compute_esm2_embeddings.py
=========================
预计算 ESM-2 (650M) 蛋白质嵌入，存储到 proteins.h5。

对每条蛋白序列执行 ESM-2 前向传播，取 CLS token (position 0) 的 1280-dim
嵌入作为整个蛋白质的向量表征，替换零填充的 AA properties。

用法：
  python datepre/compute_esm2_embeddings.py
  python datepre/compute_esm2_embeddings.py --max-proteins 20   # 测试
  python datepre/compute_esm2_embeddings.py --device cuda       # GPU 加速
"""

import argparse
import logging
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
PROJECT_DIR = SCRIPT_DIR.parent / "dataset_building"
PROCESSED_DIR = PROJECT_DIR / "processed"
PROTEINS_H5 = PROCESSED_DIR / "proteins.h5"

ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"


def load_esm2(device: str = "cpu"):
    """加载 ESM-2 模型和 tokenizer"""
    from transformers import AutoTokenizer, EsmModel

    log.info(f"Loading ESM-2: {ESM2_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL)
    model = EsmModel.from_pretrained(ESM2_MODEL)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    log.info(f"  ESM-2 loaded on {device}")
    return tokenizer, model


@torch.no_grad()
def embed_sequence(
    seq: str,
    tokenizer,
    model,
    device: str = "cpu",
    max_len: int = 1022,
) -> np.ndarray:
    """返回 CLS token 的 1280-dim 嵌入 (float32)"""
    tokens = tokenizer(
        seq[:max_len - 2],
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        padding=True,
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}
    outputs = model(**tokens)
    cls_embed = outputs.last_hidden_state[:, 0, :].squeeze(0)
    return cls_embed.cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Precompute ESM-2 embeddings")
    parser.add_argument("--proteins-h5", default=str(PROTEINS_H5))
    parser.add_argument("--max-proteins", type=int, default=None,
                        help="Limit number of proteins (for testing)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    device = args.device
    tokenizer, model = load_esm2(device)

    h5_path = Path(args.proteins_h5)
    log.info(f"Opening proteins.h5: {h5_path}")
    h5 = h5py.File(h5_path, "r+")

    seq_hashes = sorted(h5.keys())
    n_total = len(seq_hashes)
    if args.max_proteins:
        seq_hashes = seq_hashes[:args.max_proteins]
    log.info(f"  {n_total} total proteins, processing {len(seq_hashes)}")

    n_computed = 0
    n_skipped = 0
    for seq_hash in tqdm(seq_hashes, desc="Embedding proteins"):
        group = h5[seq_hash]

        # 跳过已有嵌入的
        if "esm2_embed" in group:
            n_skipped += 1
            continue

        seq_bytes = group["sequence"][()]
        if isinstance(seq_bytes, bytes):
            seq = seq_bytes.decode("utf-8")
        else:
            seq = str(seq_bytes)

        if not seq:
            zero_embed = np.zeros(1280, dtype=np.float32)
            if "esm2_embed" in group:
                del group["esm2_embed"]
            group.create_dataset("esm2_embed", data=zero_embed)
            continue

        embed = embed_sequence(seq, tokenizer, model, device)
        if "esm2_embed" in group:
            del group["esm2_embed"]
        group.create_dataset("esm2_embed", data=embed)
        n_computed += 1

    h5.close()
    log.info(f"Done: {n_computed} computed, {n_skipped} skipped, "
             f"{len(seq_hashes)} total")


if __name__ == "__main__":
    main()
