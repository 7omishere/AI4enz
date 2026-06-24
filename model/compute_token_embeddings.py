#!/usr/bin/env python3
"""
compute_token_embeddings.py
===========================
为训练集全部 18,929 个蛋白计算 ESM-2 token-level embeddings，
存入 dataset_building/processed/proteins_token.h5。

输出格式（每个蛋白一个 group）：
  {seq_hash: {
      "tokens": (L, 1280) float32,   ← 不含 BOS/EOS 的 token embedding
      "sequence": bytes,              ← 原始序列（方便调试）
  }}

用法：
  python compute_token_embeddings.py                          # CPU, 慢
  python compute_token_embeddings.py --device cuda --batch-size 4   # GPU

依赖：
  pip install esm  (已安装 esm 2.0.0)
"""

import argparse, gc, logging, sys, time, math
from pathlib import Path
import numpy as np, pandas as pd, h5py, torch

BASE = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE / "dataset_building"
PROCESSED = DATASET_DIR / "processed"
METADATA = PROCESSED / "metadata.parquet"
OLD_H5 = PROCESSED / "proteins.h5"
OUT_H5 = PROCESSED / "proteins_token.h5"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(str(PROCESSED / "token_esm2.log"), mode="a")],
)
log = logging.getLogger(__name__)

FLUSH_EVERY = 50          # 每 50 条序列写一次 H5
ESM2_MAX_TOKENS = 1022    # 实际可用残基数（模型上下文 1024 − BOS − EOS）


def get_unique_proteins() -> dict[str, str]:
    """从 metadata 和旧 H5 获取 {hash: sequence} 映射"""
    meta = pd.read_parquet(METADATA)
    hashes = meta["protein_seq_hash"].dropna().unique()
    log.info(f"Metadata: {len(meta):,} rows, {len(hashes):,} unique seq hashes")

    # 从旧 H5 中读取序列
    hash_to_seq = {}
    missing = []
    with h5py.File(OLD_H5, "r") as h5:
        for h in hashes:
            if h in h5 and "sequence" in h5[h]:
                seq_bytes = h5[h]["sequence"][()]
                if isinstance(seq_bytes, bytes):
                    seq = seq_bytes.decode("utf-8")
                else:
                    seq = str(seq_bytes)
                hash_to_seq[h] = seq

    log.info(f"Loaded {len(hash_to_seq):,} sequences from {OLD_H5.name}")
    return hash_to_seq


def compute_token_embeddings(
    hash_to_seq: dict[str, str],
    out_path: str,
    device: str = "cpu",
    batch_size: int = 1,
):
    """计算所有蛋白的 token-level ESM-2 embedding，用已有 ESM-2 模型。"""
    from transformers import AutoModel, AutoTokenizer

    esm_path = os.path.expanduser("~/run/jiaodm/AI4enz/model/ESM2")
    log.info(f"Loading ESM-2 from {esm_path}...")
    sys.stdout.flush()
    t0 = time.time()
    model = AutoModel.from_pretrained(esm_path, trust_remote_code=True)
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(esm_path)
    log.info(f"Model loaded in {time.time()-t0:.1f}s, "
             f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # checkpoint 跟踪已完成
    done_path = PROCESSED / "token_esm2_done.txt"
    done_set = set()
    if done_path.exists():
        with open(done_path) as f:
            done_set = set(line.strip() for line in f if line.strip())
        log.info(f"Checkpoint: {len(done_set):,} already done, resuming")

    entries = [(h, s) for h, s in hash_to_seq.items() if h not in done_set]
    log.info(f"Remaining: {len(entries):,} to compute")
    if not entries:
        log.info("All done!")
        return

    n_total = len(entries)
    start = time.time()
    written = 0

    for i in range(0, n_total, batch_size):
        batch_entries = entries[i:i + batch_size]
        batch_hashes = [h for h, _ in batch_entries]
        batch_seqs = [s for _, s in batch_entries]
        batch_data = [(f"s{j}", s) for j, s in enumerate(batch_seqs)]

        try:
            encoded = tokenizer(batch_seqs, padding=True, truncation=True,
                                max_length=ESM2_MAX_TOKENS, return_tensors="pt")
            tokens = encoded["input_ids"].to(device)
            mask = encoded["attention_mask"].to(device)
            with torch.no_grad():
                out = model(tokens, attention_mask=mask, output_hidden_states=False)
                reps = out.last_hidden_state  # (B, L, 1280)

            batch_results: dict[str, np.ndarray] = {}
            for j, (seq, h) in enumerate(zip(batch_seqs, batch_hashes)):
                slen = min(len(seq), ESM2_MAX_TOKENS)
                # 取实际 token（去掉 BOS=pos 0 和 EOS=pos slen+1）
                n_avail = min(slen, reps.shape[1]) - 2
                if n_avail < 1:
                    n_avail = 1

                # 取 [1:-1] 去掉 BOS/EOS
                emb = reps[j, 1:n_avail+1].cpu().numpy()
                if n_avail <= 0:
                    log.warning(f"Empty/trivial sequence for {h}, using zeros")
                    batch_results[h] = np.zeros((1, 1280), dtype=np.float32)
                else:
                    emb = reps[j, 1:1 + n_avail, :].cpu().numpy().astype(np.float32)
                    batch_results[h] = emb

            del out, reps, tokens
        except Exception as e:
            log.error(f"Batch {i} error: {e}")
            for h in batch_hashes:
                batch_results[h] = np.zeros((1, 1280), dtype=np.float32)

        # 增量写入 H5
        with h5py.File(out_path, "a") as h5:
            for h in batch_hashes:
                if h not in h5:
                    grp = h5.create_group(h)
                else:
                    grp = h5[h]
                    # 覆写
                    if "tokens" in grp:
                        del grp["tokens"]
                grp.create_dataset("tokens", data=batch_results[h],
                                   compression="gzip", compression_opts=4)
                if "sequence" not in grp:
                    idx = batch_hashes.index(h)
                    grp.create_dataset("sequence",
                                       data=batch_seqs[idx].encode("utf-8"))
                # checkpoint
                with open(done_path, "a") as f:
                    f.write(h + "\n")
                written += 1

        del batch_results
        gc.collect()

        done = i + len(batch_entries)
        if done % 10 == 0 or done >= n_total:
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 0
            eta_remaining = (n_total - done) / rate if rate > 0 else 0
            log.info(
                f"[{done:,}/{n_total:,}] {rate:.1f} seq/s, "
                f"ETA {eta_remaining/60:.0f}min | written={written:,}"
            )
            sys.stdout.flush()

    elapsed = time.time() - start
    log.info(
        f"Done: {written:,} token embeddings in {elapsed/60:.1f}min"
        f" ({written/elapsed:.1f} seq/s)"
    )

    # 清理
    del model, tokenizer
    gc.collect()
    if done_path.exists():
        done_path.unlink()


def verify():
    """验证输出文件完整性"""
    with h5py.File(OUT_H5, "r") as h5:
        keys = list(h5.keys())
        log.info(f"Verifying {len(keys):,} entries in {OUT_H5.name}")
        lens = []
        for k in keys[:100]:
            g = h5[k]
            if "tokens" in g:
                lens.append(g["tokens"].shape[0])
        log.info(f"  Token lengths (first 100): mean={np.mean(lens):.0f}, "
                 f"max={max(lens)}, min={min(lens)}")
        total_size = sum(h5[k]["tokens"].nbytes for k in keys)
        log.info(f"  Total uncompressed size: {total_size/1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-token ESM-2 embeddings for all proteins"
    )
    parser.add_argument("--device", default="cpu",
                        help="Device: 'cpu' or 'cuda'")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for ESM-2 (GPU: 4-8, CPU: 1)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing output file without recomputing")
    parser.add_argument("--torch-threads", type=int, default=8,
                        help="Torch intra-op threads (CPU only)")
    args = parser.parse_args()

    if args.verify:
        verify()
        return

    if args.device == "cpu":
        torch.set_num_threads(args.torch_threads)
        log.info(f"Torch threads: {args.torch_threads}")

    hash_to_seq = get_unique_proteins()
    compute_token_embeddings(
        hash_to_seq, str(OUT_H5),
        device=args.device, batch_size=args.batch_size,
    )
    verify()


if __name__ == "__main__":
    main()
