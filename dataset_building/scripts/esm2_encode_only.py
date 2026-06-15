#!/usr/bin/env python3
"""
ESM-2 only: 仅为缺失的蛋白计算 embeddings 并写入 proteins.h5。
增量写入 + 断点续传，防止 OOM。

日志写入 processed/esm2_only.log，进度写入 processed/esm2_progress.txt
"""
import argparse, gc, logging, sys, time
from pathlib import Path
import numpy as np, pandas as pd, h5py, torch

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "processed"
RELEASE = BASE / "release"
METADATA = PROCESSED / "metadata.parquet"
H5_PATH = PROCESSED / "proteins.h5"
CHECKPOINT_PATH = PROCESSED / "esm2_checkpoint.txt"  # 断点续传：已完成的 hash 列表

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(PROCESSED / "esm2_only.log", mode="a")])
log = logging.getLogger(__name__)

# 增量写入间隔（每处理这么多条序列就写一次 H5）
FLUSH_EVERY = 50


def load_checkpoint():
    """读取已完成的 protein hash，用于断点续传。"""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            done = set(line.strip() for line in f if line.strip())
        log.info(f"Checkpoint loaded: {len(done):,} already done")
        return done
    return set()


def save_checkpoint(hash_val):
    """追加一个 hash 到断点文件。"""
    with open(CHECKPOINT_PATH, "a") as f:
        f.write(hash_val + "\n")


def compute_and_write_esm2(sequences, seq_hashes, h5_path, device="cpu", batch_size=1):
    """
    计算 ESM-2 embeddings，增量写入 H5，不累积全部结果。
    返回已写入的数量。
    """
    import esm
    log.info("Loading esm2_t33_650M_UR50D...")
    sys.stdout.flush()
    t0 = time.time()
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    log.info(f"Model loaded in {time.time()-t0:.1f}s")
    sys.stdout.flush()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    n = len(sequences)
    start = time.time()
    written = 0

    for i in range(0, n, batch_size):
        batch_seqs = sequences[i:i+batch_size]
        batch_hashes = seq_hashes[i:i+batch_size]
        batch_data = [(f"s{j}", s) for j, s in enumerate(batch_seqs)]

        # 计算一批 embeddings
        batch_results = {}
        try:
            _, _, tokens = batch_converter(batch_data)
            tokens = tokens.to(device)
            with torch.no_grad():
                r = model(tokens, repr_layers=[33])
                reps = r["representations"][33]
            for j, (seq, h) in enumerate(zip(batch_seqs, batch_hashes)):
                slen = len(seq)
                if slen > 0 and slen < reps.shape[1] - 1:
                    emb = reps[j, 1:1+slen, :].mean(dim=0).cpu().numpy().astype(np.float32)
                else:
                    emb = reps[j, 1:-1, :].mean(dim=0).cpu().numpy().astype(np.float32)
                batch_results[h] = emb
            del reps, r, tokens
        except Exception as e:
            log.error(f"Batch {i} error: {e}")
            for seq, h in zip(batch_seqs, batch_hashes):
                batch_results[h] = np.zeros(1280, dtype=np.float32)

        # 立即增量写入 H5（避免累积内存）
        with h5py.File(h5_path, "a") as h5:
            for h, emb in batch_results.items():
                if h not in h5:
                    grp = h5.create_group(h)
                    # 找到对应的序列
                    idx = batch_hashes.index(h)
                    grp.create_dataset("sequence", data=batch_seqs[idx].encode("utf-8"))
                    grp.create_dataset("binding_site_mask", data=np.array([], dtype=np.int32))
                if "esm2_embed" in h5[h]:
                    del h5[h]["esm2_embed"]
                h5[h].create_dataset("esm2_embed", data=emb)
                save_checkpoint(h)
                written += 1

        del batch_results
        gc.collect()

        done = i + len(batch_seqs)
        if done % 10 == 0 or done >= n:
            elapsed = time.time() - start
            rate = done / elapsed
            eta = (n - done) / rate if rate > 0 else 0
            log.info(f"[{done:,}/{n:,}] {rate:.1f} seq/s, ETA {eta/60:.0f}min | written={written:,}")
            sys.stdout.flush()
            with open(PROCESSED / "esm2_progress.txt", "w") as f:
                f.write(f"{done}/{n} {100*done/n:.1f}% rate={rate:.1f}/s eta={eta/60:.0f}min written={written}\n")

    # 清理模型释放内存
    del model, alphabet, batch_converter
    gc.collect()

    elapsed = time.time() - start
    log.info(f"Done: {written:,} embeddings in {elapsed/60:.1f}min ({written/elapsed:.1f} seq/s)")
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.torch_threads)
    log.info(f"Torch threads: {args.torch_threads}")

    # Load metadata to find proteins needing ESM-2
    meta = pd.read_parquet(METADATA)
    all_hashes = set(meta["protein_seq_hash"].dropna().unique())
    log.info(f"Metadata: {len(meta):,} rows, {len(all_hashes):,} unique proteins")

    # 断点续传：已有 ESM-2 + checkpoint 中已写入的
    with h5py.File(H5_PATH, "r") as h5:
        in_h5 = all_hashes & set(h5.keys())
        has_esm = {k for k in in_h5 if "esm2_embed" in h5[k]}
    checkpoint_done = load_checkpoint()
    has_esm = has_esm | checkpoint_done
    log.info(f"In h5: {len(in_h5):,}, with ESM-2 (incl checkpoint): {len(has_esm):,}")

    need_esm = all_hashes - has_esm
    log.info(f"Need ESM-2: {len(need_esm):,}")

    if not need_esm:
        log.info("All proteins already have ESM-2. Done!")
        CHECKPOINT_PATH.unlink(missing_ok=True)
        with open(PROCESSED / "esm2_progress.txt", "w") as f:
            f.write("DONE\n")
        return

    # Sequences from V5 (metadata doesn't store sequences)
    v5 = pd.read_parquet(RELEASE / "trenzition_full_v5.parquet")

    # Collect sequences (dedup by hash) — 只加载需要的
    hash_to_seq = {}
    for h in need_esm:
        rows = v5[v5["protein_seq_hash"] == h]
        if len(rows) > 0:
            s = rows["sequence"].iloc[0]
            if isinstance(s, str) and s:
                hash_to_seq[h] = s
    del v5
    gc.collect()

    log.info(f"Sequences collected: {len(hash_to_seq):,}")

    # 计算并增量写入
    hashes = list(hash_to_seq.keys())
    seqs = [hash_to_seq[h] for h in hashes]
    del hash_to_seq
    gc.collect()

    n_written = compute_and_write_esm2(seqs, hashes, str(H5_PATH),
                                       device=args.device, batch_size=args.batch_size)

    log.info(f"Total written this run: {n_written:,}")

    # 清理断点文件
    CHECKPOINT_PATH.unlink(missing_ok=True)

    # Verify
    with h5py.File(H5_PATH, "r") as h5:
        final_esm = sum(1 for k in all_hashes if k in h5 and "esm2_embed" in h5[k])
    log.info(f"Final ESM-2 coverage: {final_esm}/{len(all_hashes)} ({100*final_esm/len(all_hashes):.1f}%)")

    with open(PROCESSED / "esm2_progress.txt", "w") as f:
        f.write(f"DONE {final_esm}/{len(all_hashes)}\n")


if __name__ == "__main__":
    main()
