"""
enrich_domains_fast.py — 优化版 Pfam 域标注（仅辅因子 HMM，速度快 200x）
"""
import argparse, logging, time, gzip
from pathlib import Path
import h5py, numpy as np
import pyhmmer
from pyhmmer.plan7 import HMMFile
from pyhmmer.easel import Alphabet, TextSequence
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent / "dataset_building"
PROCESSED_DIR = PROJECT_DIR / "processed"
PROTEINS_H5 = PROCESSED_DIR / "proteins.h5"
PFAM_FULL = PROCESSED_DIR / "Pfam-A.hmm.gz"
PFAM_COFACTOR = PROCESSED_DIR / "Pfam-cofactor.hmm"

COFACTOR_INDEX = {"NAD":0,"NADP":1,"FAD":2,"FMN":3,"HEME":4,"FES":5,"CU":6,"MPT":7,"COQ":8,"PQQ":9,"TPP":10,"PLP":11,"COA":12,"B12":13,"THF":14}
N_TYPES = len(COFACTOR_INDEX)

PFAM_COFACTOR_MAP = {
    "PF13460":"NAD","PF00175":"NAD","PF03446":"NAD","PF07992":"NAD","PF14833":"NAD","PF03447":"NAD","PF07993":"NAD","PF07994":"NAD","PF08030":"NAD","PF13241":"NAD","PF13450":"NAD","PF13454":"NAD","PF10712":"NAD","PF18127":"NAD","PF10727":"NAD","PF10079":"NAD","PF22953":"NAD","PF23114":"NAD",
    "PF00667":"FAD","PF00890":"FAD","PF01494":"FAD","PF00941":"FAD","PF01266":"FAD","PF01565":"FAD","PF08021":"FAD","PF10531":"FAD","PF00970":"FAD","PF03441":"FAD","PF08022":"FAD","PF12318":"FAD","PF21688":"FAD","PF02913":"FAD","PF22607":"FAD","PF01613":"FAD","PF00441":"FAD","PF08028":"FAD","PF12806":"FAD","PF02770":"FAD","PF02771":"FAD","PF14749":"FAD","PF21263":"FAD",
    "PF00258":"FMN","PF03350":"FMN","PF02441":"FMN","PF01070":"FMN","PF02525":"FMN","PF12641":"FMN","PF12682":"FMN","PF12724":"FMN","PF07972":"FMN",
    "PF00067":"HEME","PF04187":"HEME","PF14376":"HEME","PF14518":"HEME","PF09065":"HEME","PF14830":"HEME","PF05594":"HEME","PF13332":"HEME","PF05692":"HEME","PF13447":"HEME","PF00426":"HEME","PF21329":"HEME","PF06646":"HEME","PF05865":"HEME","PF23018":"HEME","PF00033":"HEME","PF13435":"HEME","PF14522":"HEME",
    "PF00111":"FES","PF13510":"FES","PF12801":"FES","PF12797":"FES","PF13237":"FES","PF13187":"FES","PF00037":"FES","PF12838":"FES","PF13183":"FES","PF12798":"FES",
    "PF00116":"CU","PF07732":"CU","PF07731":"CU","PF00379":"CU","PF02798":"MPT","PF00384":"MPT","PF01568":"MPT","PF01077":"COQ","PF00994":"PQQ",
    "PF00205":"TPP","PF02775":"TPP","PF02776":"TPP",
    "PF00456":"PLP","PF00155":"PLP","PF00202":"PLP","PF00266":"PLP","PF01041":"PLP","PF01053":"PLP","PF01212":"PLP","PF01276":"PLP","PF02261":"PLP","PF02347":"PLP","PF02826":"PLP","PF02955":"PLP","PF03841":"PLP","PF04895":"PLP","PF06243":"PLP","PF14031":"PLP","PF14542":"PLP","PF22608":"PLP",
    "PF00550":"COA","PF01144":"COA","PF01553":"COA","PF02515":"COA","PF02629":"COA","PF13380":"COA","PF13607":"COA","PF13735":"COA","PF16197":"COA",
    "PF00561":"B12","PF02310":"B12","PF02572":"B12","PF02607":"B12","PF02965":"B12","PF03186":"B12","PF03324":"B12","PF16861":"B12",
    "PF00384":"THF","PF01243":"THF","PF02142":"THF","PF02649":"THF",
}


def prepare_cofactor_hmms() -> str:
    """从 Pfam-A.hmm.gz 提取辅因子 HMM → 返回未压缩路径。"""
    if PFAM_COFACTOR.exists():
        log.info(f"Cofactor HMM exists: {PFAM_COFACTOR}")
        return str(PFAM_COFACTOR)

    log.info("Extracting cofactor HMMs from Pfam-A.hmm.gz...")
    target = set(PFAM_COFACTOR_MAP.keys())
    entries = []
    with gzip.open(PFAM_FULL, 'rt') as f:
        buf, acc = [], None
        for line in f:
            if line.startswith('HMMER3/'):
                if buf and acc and acc.split('.')[0] in target:
                    entries.append(''.join(buf))
                buf, acc = [line], None
            else:
                buf.append(line)
                if line.startswith('ACC   '): acc = line.strip().split()[1]
        if buf and acc and acc.split('.')[0] in target:
            entries.append(''.join(buf))
    with open(PFAM_COFACTOR, 'w') as f:
        f.write(''.join(entries))
    log.info(f"  {len(entries)} HMMs → {PFAM_COFACTOR} ({PFAM_COFACTOR.stat().st_size//1024} KB)")
    return str(PFAM_COFACTOR)


def scan_proteins(proteins: dict[str, str], hmm_path: str) -> dict[str, dict[str, list]]:
    """pyhmmer.hmmscan — 一个序列 vs 辅因子 HMM。"""
    alpha = Alphabet.amino()
    target = set(PFAM_COFACTOR_MAP.keys())

    seq_hashes = list(proteins.keys())
    pyseqs = []
    for sh in seq_hashes:
        try:
            ts = TextSequence(sequence=proteins[sh].encode("ascii", errors="replace"))
            ds = ts.digitize(alpha); ds.name = sh.encode()
            pyseqs.append(ds)
        except Exception:
            pyseqs.append(None)

    results = {sh: {} for sh in seq_hashes}
    with HMMFile(hmm_path) as hf:
        for i, ds in enumerate(tqdm(pyseqs, desc="hmmscan (cofactor)")):
            sh = seq_hashes[i]
            if ds is None: continue
            hf.rewind()
            try:
                for hits in pyhmmer.hmmscan([ds], hf, cpus=0):
                    for hit in hits:
                        acc = hit.accession.decode() if isinstance(hit.accession, bytes) else hit.accession
                        cf = PFAM_COFACTOR_MAP.get(acc.split('.')[0])
                        if cf is None: continue
                        best_d, best_s = None, -float("inf")
                        for d in hit.domains:
                            if d.score > best_s: best_s, best_d = d.score, d
                        if best_d and best_s > 20:
                            results[sh].setdefault(cf, []).append((best_d.env_from, best_d.env_to))
            except Exception:
                pass
    return results


def build_masks(proteins: dict[str, str], hits: dict) -> dict[str, np.ndarray]:
    masks = {}
    for h, seq in proteins.items():
        m = np.zeros((N_TYPES, len(seq)), dtype=np.float32)
        for cf, pos in hits.get(h, {}).items():
            if cf in COFACTOR_INDEX:
                ci = COFACTOR_INDEX[cf]
                for s, e in pos:
                    m[ci, max(0, int(s)-1):min(len(seq), int(e))] = 1.0
        masks[h] = m
    return masks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-proteins", type=int, default=None)
    p.add_argument("--proteins-h5", default=str(PROTEINS_H5))
    args = p.parse_args()

    hmm_path = prepare_cofactor_hmms()
    h5_path = Path(args.proteins_h5)

    with h5py.File(h5_path, 'r') as h5:
        all_keys = sorted(h5.keys())
        todo = [k for k in all_keys if 'domain_masks' not in h5[k]]
    if args.max_proteins: todo = todo[:args.max_proteins]
    log.info(f"Total: {len(all_keys):,}, need domains: {len(todo):,}")

    proteins = {}
    with h5py.File(h5_path, 'r') as h5:
        for k in todo:
            try:
                s = h5[k]['sequence'][()]; seq = s.decode() if isinstance(s, bytes) else str(s)
                if seq: proteins[k] = seq
            except: pass
    log.info(f"Valid sequences: {len(proteins):,}")

    t0 = time.time()
    hits = scan_proteins(proteins, hmm_path)
    dt = time.time() - t0
    n_cf = sum(1 for v in hits.values() if v)
    log.info(f"Scan: {dt:.1f}s ({len(proteins)/max(1,dt):.1f} seq/s), cofactor hits: {n_cf}")

    masks = build_masks(proteins, hits)
    n_nz = sum(1 for m in masks.values() if m.any())
    log.info(f"Non-zero domain_masks: {n_nz}/{len(masks)}")

    with h5py.File(h5_path, 'r+') as h5:
        for sh, m in tqdm(masks.items(), desc="Writing"):
            g = h5[sh]
            if 'domain_masks' in g: del g['domain_masks']
            g.create_dataset('domain_masks', data=m)

    hf = h5py.File(h5_path, 'r')
    n_dm = sum(1 for k in hf.keys() if 'domain_masks' in hf[k] and hf[k]['domain_masks'][:].any())
    hf.close()
    log.info(f"H5 non-zero domain_masks: {n_dm:,} / {len(all_keys):,}")
    est = len(todo) / max(1, len(proteins)) * dt
    log.info(f"Full run est: ~{est/3600:.1f}h")


if __name__ == "__main__":
    main()
