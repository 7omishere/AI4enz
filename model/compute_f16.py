#!/usr/bin/env python3
"""
compute_f16.py — 在计算节点上直接输出 float16 token embeddings.
复用 compute_token_embeddings.py 的逻辑，但：
  - 输出 float16 (省一半空间)
  - 路径适配服务器结构 (metadata.parquet + proteins.h5 在 ~/run/jdm/)
  - 输出到 ~/run/jdm/proteins_token_f16.h5
"""
import argparse, gc, logging, sys, time, os
from pathlib import Path
import numpy as np, pandas as pd, h5py, torch

BASE = Path(__file__).resolve().parent.parent
METADATA = BASE / "metadata.parquet"
OLD_H5 = BASE / "proteins.h5"
OUT_H5 = BASE / "proteins_token_f16.h5"
ESM2_PATH = os.path.expanduser("~/run/lp/ESM2")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

FLUSH_EVERY = 50
ESM2_MAX_TOKENS = 1022


def get_unique_proteins():
    meta = pd.read_parquet(METADATA)
    hashes = meta["protein_seq_hash"].dropna().unique()
    log.info(f"Metadata: {len(meta):,} rows, {len(hashes):,} unique seq hashes")

    hash_to_seq = {}
    with h5py.File(OLD_H5, "r") as h5:
        for h in hashes:
            if h in h5 and "sequence" in h5[h]:
                seq_bytes = h5[h]["sequence"][()]
                seq = seq_bytes.decode("utf-8") if isinstance(seq_bytes, bytes) else str(seq_bytes)
                hash_to_seq[h] = seq

    log.info(f"Loaded {len(hash_to_seq):,} sequences from {OLD_H5.name}")
    return hash_to_seq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    from transformers import AutoModel, AutoTokenizer

    log.info(f"Loading ESM-2 from {ESM2_PATH}...")
    sys.stdout.flush()
    t0 = time.time()
    model = AutoModel.from_pretrained(ESM2_PATH, trust_remote_code=True)
    model = model.to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(ESM2_PATH)
    log.info(f"Model loaded in {time.time()-t0:.1f}s")

    hash_to_seq = get_unique_proteins()
    entries = list(hash_to_seq.items())
    n_total = len(entries)
    start = time.time()

    # Open H5 for writing
    h5 = h5py.File(OUT_H5, "w")
    written = 0

    for i in range(0, n_total, args.batch_size):
        batch_entries = entries[i:i + args.batch_size]
        batch_hashes = [h for h, _ in batch_entries]
        batch_seqs = [s for _, s in batch_entries]

        try:
            encoded = tokenizer(batch_seqs, padding=True, truncation=True,
                                max_length=ESM2_MAX_TOKENS, return_tensors="pt")
            tokens = encoded["input_ids"].to(args.device)
            mask = encoded["attention_mask"].to(args.device)

            with torch.no_grad():
                out = model(tokens, attention_mask=mask, output_hidden_states=False)
                reps = out.last_hidden_state

            for j, (seq, h) in enumerate(zip(batch_seqs, batch_hashes)):
                slen = min(len(seq), ESM2_MAX_TOKENS)
                n_avail = min(slen, reps.shape[1]) - 2
                if n_avail < 1:
                    n_avail = 1
                    log.warning(f"Empty sequence for {h}")
                emb = reps[j, 1:1 + n_avail, :].cpu().numpy().astype(np.float16)
                grp = h5.create_group(h)
                grp.create_dataset("tokens", data=emb, compression="gzip", compression_opts=4)
                grp.create_dataset("sequence", data=seq.encode("utf-8"))
                written += 1

            del out, reps, tokens
        except Exception as e:
            log.error(f"Batch {i} error: {e}")
            for h in batch_hashes:
                grp = h5.create_group(h)
                grp.create_dataset("tokens", data=np.zeros((1, 1280), dtype=np.float16),
                                   compression="gzip", compression_opts=4)
                grp.create_dataset("sequence", data=batch_seqs[0].encode("utf-8"))
                written += 1

        gc.collect()

        done = i + len(batch_entries)
        if done % 10 == 0 or done >= n_total:
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n_total - done) / rate if rate > 0 else 0
            log.info(f"[{done:,}/{n_total:,}] {rate:.1f} seq/s, ETA {eta/60:.0f}min")
            sys.stdout.flush()

    h5.close()
    elapsed = time.time() - start
    log.info(f"Done: {written:,} embeddings in {elapsed/60:.1f}min")

    size_gb = OUT_H5.stat().st_size / 1e9
    log.info(f"Output: {OUT_H5} ({size_gb:.1f} GB, float16)")


if __name__ == "__main__":
    main()
