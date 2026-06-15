"""
enrich_domains_pfam.py
======================
使用 pyhmmer + Pfam-A HMM 数据库本地扫描 541 个氧化还原酶蛋白序列，
替代 UniProt 预存注释中的域信息。

工作流程：
  1. 解压 Pfam-A.hmm.gz（如未解压）
  2. 加载 HMM 数据库
  3. 对每个蛋白序列运行 hmmscan
  4. 将 Pfam accession 映射为 cofactor type (通过 DOMAIN_COFACTOR_MAP)
  5. 构建 domain_masks (15, L) 和 domain_positions
  6. 更新 proteins.h5 中每个蛋白的 domain_masks/domain_positions

用法：
  python datepre/enrich_domains_pfam.py
  python datepre/enrich_domains_pfam.py --max-proteins 20     # 测试
"""

import argparse
import gzip
import logging
import shutil
from pathlib import Path

import h5py
import numpy as np
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
PFAM_GZ = PROCESSED_DIR / "Pfam-A.hmm.gz"
PFAM_HMM = PROCESSED_DIR / "Pfam-A.hmm"

# 与 extract_domains.py 保持一致的辅因子类型索引
COFACTOR_INDEX: dict[str, int] = {
    "NAD": 0, "NADP": 1, "FAD": 2, "FMN": 3, "HEME": 4,
    "FES": 5, "CU": 6, "MPT": 7, "COQ": 8, "PQQ": 9,
    "TPP": 10, "PLP": 11, "COA": 12, "B12": 13, "THF": 14,
}
N_COFACTOR_TYPES = len(COFACTOR_INDEX)

# Pfam accession → cofactor type 映射
PFAM_COFACTOR_MAP: dict[str, str] = {
    # NAD(P)-binding Rossmann fold
    "PF13460": "NAD", "PF00175": "NAD", "PF03446": "NAD",
    "PF07992": "NAD", "PF14833": "NAD", "PF03447": "NAD",
    "PF07993": "NAD", "PF07994": "NAD", "PF08030": "NAD",
    "PF13241": "NAD", "PF13450": "NAD", "PF13454": "NAD",
    "PF10712": "NAD", "PF18127": "NAD",
    "PF10727": "NAD", "PF10079": "NAD", "PF22953": "NAD",
    "PF23114": "NAD",
    # FAD-binding
    "PF00667": "FAD", "PF00890": "FAD", "PF01494": "FAD",
    "PF00941": "FAD", "PF01266": "FAD", "PF01565": "FAD",
    "PF08021": "FAD", "PF10531": "FAD", "PF00970": "FAD",
    "PF03441": "FAD", "PF08022": "FAD", "PF12318": "FAD",
    "PF21688": "FAD", "PF02913": "FAD", "PF22607": "FAD",
    "PF01613": "FAD", "PF00441": "FAD", "PF08028": "FAD",
    "PF12806": "FAD", "PF02770": "FAD", "PF02771": "FAD",
    "PF14749": "FAD", "PF21263": "FAD",
    # FMN-binding
    "PF00258": "FMN", "PF03350": "FMN", "PF02441": "FMN",
    "PF01070": "FMN", "PF02525": "FMN", "PF12641": "FMN",
    "PF12682": "FMN", "PF12724": "FMN", "PF07972": "FMN",
    # Heme-binding
    "PF00067": "HEME", "PF04187": "HEME", "PF14376": "HEME",
    "PF14518": "HEME", "PF09065": "HEME", "PF14830": "HEME",
    "PF05594": "HEME", "PF13332": "HEME", "PF05692": "HEME",
    "PF13447": "HEME", "PF00426": "HEME", "PF21329": "HEME",
    "PF06646": "HEME", "PF05865": "HEME", "PF23018": "HEME",
    "PF00033": "HEME", "PF13435": "HEME", "PF14522": "HEME",
    "PF13442": "HEME", "PF16694": "HEME", "PF11783": "HEME",
    "PF10643": "HEME", "PF22085": "HEME",
    # Iron-Sulfur
    "PF00111": "FES", "PF13510": "FES", "PF00037": "FES",
    "PF13247": "FES", "PF00355": "FES", "PF13183": "FES",
    "PF12801": "FES", "PF12837": "FES", "PF12798": "FES",
    "PF13085": "FES", "PF12797": "FES", "PF01799": "FES",
    "PF04324": "FES", "PF13237": "FES", "PF13353": "FES",
    "PF13370": "FES", "PF13459": "FES", "PF13484": "FES",
    "PF13534": "FES", "PF13746": "FES", "PF06902": "FES",
    "PF14691": "FES", "PF14697": "FES", "PF17179": "FES",
    "PF18009": "FES", "PF18109": "FES", "PF12800": "FES",
    "PF12838": "FES", "PF13187": "FES", "PF05187": "FES",
    "PF22151": "FES", "PF00142": "FES", "PF22117": "FES",
    "PF04068": "FES", "PF16947": "FES", "PF08609": "FES",
    "PF11591": "FES", "PF01257": "FES", "PF23544": "FES",
    "PF10418": "FES", "PF04384": "FES", "PF01521": "FES",
    "PF04060": "FES", "PF01883": "FES", "PF22609": "FES",
    "PF00301": "FES", "PF18267": "FES", "PF00374": "FES",
    "PF10399": "FES", "PF20410": "FES", "PF10588": "FES",
    "PF10589": "FES", "PF05484": "FES", "PF25160": "FES",
    "PF13806": "FES", "PF18465": "FES", "PF22543": "FES",
    # Copper
    "PF00394": "CU", "PF07731": "CU", "PF07732": "CU",
    "PF00127": "CU", "PF01032": "CU", "PF00649": "CU",
    "PF11617": "CU", "PF13473": "CU",
    # Molybdopterin
    "PF00994": "MPT", "PF00174": "MPT", "PF00384": "MPT",
    "PF01568": "MPT", "PF01315": "MPT", "PF20294": "MPT",
    "PF18818": "MPT", "PF02738": "MPT", "PF20256": "MPT",
    "PF18364": "MPT", "PF26314": "MPT", "PF27552": "MPT",
    "PF27547": "MPT", "PF01973": "MPT", "PF09176": "MPT",
    "PF15638": "MPT", "PF15639": "MPT", "PF15640": "MPT",
    "PF15641": "MPT", "PF27529": "MPT",
    # Coenzyme Q / Quinone
    "PF00507": "COQ", "PF00346": "COQ", "PF21162": "COQ",
    # PQQ
    "PF01011": "PQQ", "PF13360": "PQQ", "PF10527": "PQQ",
    "PF10535": "PQQ",
    # TPP
    "PF02776": "TPP", "PF00205": "TPP", "PF02775": "TPP",
    "PF02780": "TPP", "PF22613": "TPP", "PF00456": "TPP",
    "PF01910": "TPP", "PF22141": "TPP", "PF22156": "TPP",
    # PLP
    "PF00155": "PLP", "PF00202": "PLP", "PF00266": "PLP",
    "PF01276": "PLP", "PF01842": "PLP", "PF01063": "PLP",
    "PF12897": "PLP", "PF00282": "PLP",
    # CoA
    "PF02629": "COA", "PF13380": "COA", "PF00549": "COA",
    "PF01144": "COA", "PF01553": "COA", "PF01636": "COA",
    "PF02515": "COA", "PF13607": "COA", "PF13147": "COA",
    "PF14542": "COA", "PF00583": "COA", "PF13673": "COA",
    "PF13720": "COA", "PF13880": "COA", "PF03421": "COA",
    "PF17013": "COA", "PF05301": "COA", "PF17668": "COA",
    "PF18014": "COA", "PF18015": "COA", "PF00797": "COA",
    "PF13302": "COA", "PF13420": "COA", "PF13444": "COA",
    "PF13480": "COA", "PF13508": "COA", "PF13523": "COA",
    "PF13527": "COA", "PF02551": "COA",
    # B12
    "PF02310": "B12", "PF02572": "B12", "PF02607": "B12",
    "PF01122": "B12",
    # THF
    "PF01268": "THF", "PF01463": "THF", "PF01597": "THF",
    "PF01770": "THF", "PF03024": "THF",
}
# Regex keyword patterns for fallback classification
_PFAM_NAME_COFACTOR_PATTERNS: list[tuple[str, str]] = [
    ("NAD", r"\bNAD\b|NADP|nicotinamide|Rossmann"),
    ("FAD", r"\bFAD\b|flavin|acyl-CoA.*de?h"),
    ("FMN", r"\bFMN\b|flavodoxin|flavoprotein"),
    ("HEME", r"\bheme\b|\bhaem\b|cytochrome|P450|globin|\bCYP\b"),
    ("FES", r"\bFe[-. ]S\b|iron[-. ]sulfur|ferredoxin|rubredoxin|Rieske|[234]Fe|hydrogenase"),
    ("CU", r"\bcopper\b|cupredoxin|plastocyanin|azurin|multicopper"),
    ("MPT", r"molybdopterin|molybdenum|\bMopterin\b|\bMoCo\b|\bMpt\b|MPTase"),
    ("COQ", r"ubiquinone|quinone|quinol"),
    ("PQQ", r"\bPQQ\b|pyrroloquinoline"),
    ("TPP", r"\bTPP\b|thiamin|transketolase|pyruvate decarboxylase"),
    ("PLP", r"\bPLP\b|aminotransf|pyridoxal|decarboxylase"),
    ("COA", r"\bCoA\b|coenzyme A|acetyltransf|acyltransf|ketoacyl"),
    ("B12", r"\bB12\b|cobalamin"),
    ("THF", r"\bTHF\b|tetrahydrofolate|folate"),
]


def classify_cofactor(pfam_accession: str, pfam_name: str) -> str | None:
    """Classify a Pfam hit to a cofactor type, using hardcoded map + keyword fallback."""
    # Strip version suffix (e.g. PF00067.23 → PF00067)
    base_acc = pfam_accession.split(".")[0]
    if base_acc in PFAM_COFACTOR_MAP:
        return PFAM_COFACTOR_MAP[base_acc]
    # Keyword fallback
    import re
    for ct, pattern in _PFAM_NAME_COFACTOR_PATTERNS:
        if re.search(pattern, pfam_name, re.IGNORECASE):
            return ct
    return None


def load_sequences(h5_path: Path, max_proteins: int | None = None):
    """从 proteins.h5 提取所有蛋白序列。"""
    h5 = h5py.File(h5_path, "r")
    seq_hashes = sorted(h5.keys())
    if max_proteins:
        seq_hashes = seq_hashes[:max_proteins]

    sequences = {}
    for sh in seq_hashes:
        seq_bytes = h5[sh]["sequence"][()]
        if isinstance(seq_bytes, bytes):
            seq = seq_bytes.decode("utf-8")
        else:
            seq = str(seq_bytes)
        sequences[sh] = seq
    h5.close()
    return sequences


def scan_with_pyhmmer(
    sequences: dict[str, str],
    hmm_path: Path,
    hmmer_cpus: int = 4,
):
    """使用 pyhmmer 对蛋白质序列运行 hmmscan。"""
    from pyhmmer.plan7 import HMMFile
    from pyhmmer.easel import Alphabet, TextSequence

    alphabet = Alphabet.amino()
    results = {}

    # 将序列转换为 pyhmmer 格式
    seq_hashes = list(sequences.keys())
    pyhmmer_seqs = []
    for sh in seq_hashes:
        seq = sequences[sh]
        try:
            ts = TextSequence(sequence=seq.encode("ascii", errors="replace"))
            ds = ts.digitize(alphabet)
            ds.name = sh.encode()
            pyhmmer_seqs.append(ds)
        except Exception:
            # 序列包含非标准氨基酸，跳过扫描但创建空结果
            pyhmmer_seqs.append(None)
            results[sh] = []

    # 扫描
    with HMMFile(str(hmm_path)) as hmm_file:
        for i, ds in enumerate(tqdm(pyhmmer_seqs, desc="hmmscan")):
            sh = seq_hashes[i]
            if ds is None:
                results[sh] = []
                continue

            hits_list = []
            hmm_file.rewind()
            try:
                # 使用 pyhmmer 的 pipeline
                import pyhmmer
                for hits in pyhmmer.hmmscan([ds], hmm_file, cpus=0):
                    for hit in hits:
                        best_domain = None
                        best_score = -float("inf")
                        for dom in hit.domains:
                            if dom.score > best_score:
                                best_score = dom.score
                                best_domain = dom

                        if best_domain is not None and best_score > 20:  # 阈值
                            pfam_acc = hit.accession.decode() if isinstance(hit.accession, bytes) else hit.accession
                            pfam_name = hit.name.decode() if isinstance(hit.name, bytes) else hit.name
                            hits_list.append({
                                "accession": pfam_acc,
                                "name": pfam_name,
                                "score": best_score,
                                "start": best_domain.env_from,
                                "end": best_domain.env_to,
                            })
            except Exception as e:
                log.debug(f"hmmscan error for {sh}: {e}")

            results[sh] = hits_list

    return results


def build_domain_masks(
    sequences: dict[str, str],
    hmmscan_results: dict[str, list[dict]],
):
    """根据 hmmscan 结果构建 domain_masks 和 domain_positions。"""
    domain_masks_all = {}
    domain_positions_all = {}

    for sh, seq in sequences.items():
        L = len(seq)
        masks = np.zeros((N_COFACTOR_TYPES, L), dtype=np.float32)
        positions = []

        hits = hmmscan_results.get(sh, [])
        for hit in hits:
            ct = classify_cofactor(hit["accession"], hit["name"])
            if ct is not None and ct in COFACTOR_INDEX:
                idx = COFACTOR_INDEX[ct]
                start = max(0, hit["start"] - 1)  # 1-based → 0-based
                end = min(L, hit["end"])
                if start < end:
                    masks[idx, start:end] = 1.0
                    positions.append({
                        "cofactor_type": ct,
                        "start": start,
                        "end": end,
                        "pfam_acc": hit["accession"],
                        "pfam_name": hit["name"],
                        "score": float(hit["score"]),
                    })

        domain_masks_all[sh] = masks
        domain_positions_all[sh] = positions

    return domain_masks_all, domain_positions_all


def update_proteins_h5(
    h5_path: Path,
    domain_masks_all: dict[str, np.ndarray],
    domain_positions_all: dict[str, list[dict]],
):
    """更新 proteins.h5 中的 domain_masks 和 domain_positions。"""
    import json
    h5 = h5py.File(h5_path, "r+")

    n_updated = 0
    for sh, masks in domain_masks_all.items():
        if sh not in h5:
            continue
        group = h5[sh]

        # 更新 domain_masks
        if "domain_masks" in group:
            del group["domain_masks"]
        group.create_dataset("domain_masks", data=masks, dtype=np.float32)

        # 更新 domain_positions
        if "domain_positions" in group:
            del group["domain_positions"]
        pos_list = domain_positions_all.get(sh, [])
        if pos_list:
            pos_json = json.dumps(pos_list)
            group.create_dataset("domain_positions", data=pos_json.encode())

        n_updated += 1

    h5.close()
    log.info(f"Updated {n_updated} proteins in {h5_path}")
    return n_updated


def main():
    parser = argparse.ArgumentParser(description="Pfam domain scanning for oxidoreductases")
    parser.add_argument("--proteins-h5", default=str(PROTEINS_H5))
    parser.add_argument("--pfam-hmm", default=str(PFAM_HMM))
    parser.add_argument("--pfam-cofactor-hmm", default=str(
        PROCESSED_DIR / "Pfam_cofactor.hmm"))
    parser.add_argument("--pfam-gz", default=str(PFAM_GZ))
    parser.add_argument("--use-filtered", action="store_true", default=True,
                        help="Use filtered cofactor-only HMM (faster)")
    parser.add_argument("--max-proteins", type=int, default=None)
    parser.add_argument("--cpus", type=int, default=4)
    args = parser.parse_args()

    # 1. 选择 HMM 文件
    cofactor_hmm = Path(args.pfam_cofactor_hmm)
    if args.use_filtered and cofactor_hmm.exists():
        hmm_path = cofactor_hmm
        log.info(f"Using filtered cofactor HMM: {hmm_path}")
    else:
        hmm_path = Path(args.pfam_hmm)
        gz_path = Path(args.pfam_gz)
        if not hmm_path.exists() and gz_path.exists():
            log.info(f"Decompressing {gz_path} → {hmm_path}")
            with gzip.open(gz_path, "rb") as f_in:
                with open(hmm_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            log.info(f"  {hmm_path.stat().st_size / 1e9:.1f} GB")

    if not hmm_path.exists():
        log.error(f"HMM file not found: {hmm_path}")
        return

    # 2. 加载序列
    log.info("Loading sequences from proteins.h5")
    sequences = load_sequences(Path(args.proteins_h5), args.max_proteins)
    log.info(f"  {len(sequences)} sequences loaded")

    # 3. hmmscan
    log.info("Running hmmscan against Pfam-A...")
    hmmscan_results = scan_with_pyhmmer(sequences, hmm_path)

    # 4. 统计
    n_with_hits = sum(1 for hits in hmmscan_results.values() if hits)
    n_hits_total = sum(len(hits) for hits in hmmscan_results.values())
    cofactor_hits = 0
    for hits in hmmscan_results.values():
        for h in hits:
            if classify_cofactor(h["accession"], h["name"]):
                cofactor_hits += 1
    log.info(f"  Proteins with Pfam hits: {n_with_hits}/{len(sequences)}")
    log.info(f"  Total Pfam hits: {n_hits_total}")
    log.info(f"  Cofactor-relevant hits: {cofactor_hits}")

    # 5. 构建 domain_masks
    log.info("Building domain_masks...")
    masks_all, pos_all = build_domain_masks(sequences, hmmscan_results)

    nz_count = sum(1 for m in masks_all.values() if m.sum() > 0)
    log.info(f"  Proteins with non-zero domain_masks: {nz_count}")
    for ct, idx in COFACTOR_INDEX.items():
        count = sum(1 for m in masks_all.values() if m[idx].sum() > 0)
        if count > 0:
            log.info(f"    {ct}: {count} proteins")

    # 6. 更新 proteins.h5
    log.info("Updating proteins.h5...")
    update_proteins_h5(Path(args.proteins_h5), masks_all, pos_all)

    log.info("Done")


if __name__ == "__main__":
    main()
