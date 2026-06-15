"""
compute_pocket_features.py
==========================
从 AlphaFold 结构的 contact_map + protrusion_index 检测结合口袋，
计算 pocket_contact_number, pocket_protrusion_index, pocket_ca_distances。

算法：低突起指数残基的空间聚类 → 最大的凹面簇 → 口袋。
无需外部工具，纯 NumPy。

用法：
  python scripts/processors/compute_pocket_features.py --max-proteins 50
  python scripts/processors/compute_pocket_features.py --workers 4
"""

import argparse, logging, time, os, io, requests, pickle, gzip
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent / "dataset_building"
PROCESSED_DIR = PROJECT_DIR / "processed"
PROTEINS_H5 = PROCESSED_DIR / "proteins.h5"
AF_CACHE = PROCESSED_DIR / "alphafold_cache"

# AlphaFold API
AF_API = "https://alphafold.ebi.ac.uk/api/prediction"
AF_FILE = "https://alphafold.ebi.ac.uk/files"


def parse_pdb_ca(pdb_text: str) -> np.ndarray | None:
    """提取 Cα 坐标 → (N, 3) 或 None。"""
    coords = []
    for line in pdb_text.split('\n'):
        if line.startswith('ATOM') and line[12:16].strip() == 'CA':
            try:
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                pass
    return np.array(coords, dtype=np.float32) if coords else None


def get_alphafold_structure(uniprot_id: str) -> str | None:
    """获取 AlphaFold 预测 PDB 结构。"""
    try:
        resp = requests.get(f"{AF_API}/{uniprot_id}", timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not isinstance(data, list) or not data:
            return None
        entry = data[0]
        mid, ver = entry['modelEntityId'], entry['latestVersion']
        url = f"{AF_FILE}/{mid}-model_v{ver}.pdb"
        resp2 = requests.get(url, timeout=30)
        return resp2.text if resp2.status_code == 200 else None
    except Exception:
        return None


def detect_pocket_residues(
    coords: np.ndarray,
    contact_map: np.ndarray,
    protrusion: np.ndarray,
    contact_number: np.ndarray,
    top_fraction: float = 0.25,
    dist_cutoff: float = 12.0,
    min_pocket_size: int = 5,
) -> np.ndarray:
    """
    基于几何特征检测口袋残基。

    算法：
    1. 选突起指数最低的 top_fraction 残基（凹面区域）
    2. 用距离矩阵在这些残基上做单链接聚类
    3. 取最大簇作为口袋
    4. 扩展到簇周围 dist_cutoff 范围内的残基

    Returns: bool array (N,)  True = 口袋残基
    """
    n = len(coords)
    if n < min_pocket_size:
        return np.zeros(n, dtype=bool)

    # Step 1: 候选残基 = 低突起指数（凹面） + 中等接触数（不在核心）
    protrusion_thresh = np.percentile(protrusion, top_fraction * 100)
    cn_median = np.median(contact_number)
    candidates = (protrusion <= protrusion_thresh) & (contact_number < cn_median * 1.5)

    n_candidates = candidates.sum()
    if n_candidates < min_pocket_size:
        # 放宽条件：只用突起指数
        candidates = protrusion <= np.percentile(protrusion, 50)
        n_candidates = candidates.sum()
        if n_candidates < min_pocket_size:
            # 选突起指数最低的 min_pocket_size 个残基
            idx = np.argsort(protrusion)[:min_pocket_size]
            pocket_mask = np.zeros(n, dtype=bool)
            pocket_mask[idx] = True
            return pocket_mask

    # Step 2: 候选残基的空间聚类（使用 contact_map 引导）
    candidate_indices = np.where(candidates)[0]

    # 计算候选残基间的距离矩阵
    cand_coords = coords[candidate_indices]
    diff = cand_coords[:, None, :] - cand_coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))

    # 单链接聚类：距离 < 8Å 的残基在同一簇
    visited = np.zeros(n_candidates, dtype=bool)
    clusters = []
    for i in range(n_candidates):
        if visited[i]:
            continue
        # BFS
        cluster = [i]
        visited[i] = True
        queue = [i]
        while queue:
            u = queue.pop(0)
            neighbors = np.where((dist[u] < 8.0) & ~visited)[0]
            for v in neighbors:
                visited[v] = True
                cluster.append(v)
                queue.append(v)
        clusters.append(cluster)

    if not clusters:
        return np.zeros(n, dtype=bool)

    # Step 3: 最大簇 = 口袋核心
    largest = max(clusters, key=len)
    pocket_core = set(candidate_indices[largest])

    # Step 4: 扩展到核心周围 12Å 的残基
    pocket_residues = set(pocket_core)
    for ci in pocket_core:
        nearby = np.where(contact_map[ci] > 0)[0]  # 接触的残基
        pocket_residues.update(nearby)
        # 也加 8-12Å 范围内的
        for j in range(n):
            if j not in pocket_residues:
                d = np.linalg.norm(coords[ci] - coords[j])
                if d < dist_cutoff:
                    pocket_residues.add(j)

    pocket_mask = np.zeros(n, dtype=bool)
    pocket_mask[list(pocket_residues)] = True
    return pocket_mask


def compute_pocket_features(coords: np.ndarray, pocket_mask: np.ndarray) -> dict:
    """
    计算口袋特征：
      - pocket_contact_number: (K,) 口袋残基接触数
      - pocket_protrusion_index: (K,) 口袋残基突起指数
      - pocket_ca_distances: (K, K) 口袋残基 Cα 距离矩阵
      - pocket_residue_ids: (K,) 口袋残基在序列中的索引
    """
    p_idx = np.where(pocket_mask)[0]
    K = len(p_idx)
    if K == 0:
        return {}

    p_coords = coords[p_idx]

    # 距离矩阵
    diff = p_coords[:, None, :] - p_coords[None, :, :]
    p_dist = np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)

    # Contact number (8Å cutoff)
    p_contact_number = (p_dist < 8.0).sum(axis=-1).astype(np.float32)
    np.fill_diagonal(p_dist, 0)

    # Protrusion index
    max_cn = p_contact_number.max()
    if max_cn > 0:
        p_protrusion = 1.0 - (p_contact_number / max_cn)
    else:
        p_protrusion = np.zeros(K, dtype=np.float32)

    return {
        'pocket_ca_distances': p_dist,
        'pocket_contact_number': p_contact_number,
        'pocket_protrusion_index': p_protrusion.astype(np.float32),
        'pocket_residue_ids': p_idx.astype(np.int32),
    }


def process_one(seq_hash: str, uniprot_id: str, h5_path: str) -> bool:
    """处理单个蛋白：获取结构 → 检测口袋 → 计算特征 → 写入 H5。"""
    try:
        pdb_text = get_alphafold_structure(uniprot_id)
        if pdb_text is None:
            return False

        coords = parse_pdb_ca(pdb_text)
        if coords is None or len(coords) < 10:
            return False

        # 计算基础结构特征
        n = len(coords)
        diff = coords[:, None, :] - coords[None, :, :]
        dist_mat = np.sqrt((diff ** 2).sum(axis=-1))
        contact_map = (dist_mat < 8.0).astype(np.float32)
        np.fill_diagonal(contact_map, 0)
        contact_number = contact_map.sum(axis=-1)
        max_cn = contact_number.max()
        protrusion = 1.0 - (contact_number / max_cn) if max_cn > 0 else np.zeros(n)

        # 检测口袋
        pocket_mask = detect_pocket_residues(coords, contact_map, protrusion, contact_number)
        if pocket_mask.sum() < 3:
            return False

        # 计算口袋特征
        pf = compute_pocket_features(coords, pocket_mask)
        if not pf:
            return False

        # 写入 H5
        with h5py.File(h5_path, 'r+') as h5:
            if seq_hash not in h5:
                return False
            grp = h5[seq_hash]
            for k in ['pocket_ca_distances', 'pocket_contact_number',
                       'pocket_protrusion_index', 'pocket_residue_ids']:
                if k in grp:
                    del grp[k]
                grp.create_dataset(k, data=pf[k])

        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-proteins", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    h5_path = str(PROTEINS_H5)
    import pandas as pd
    df = pd.read_parquet(PROJECT_DIR / 'release' / 'recommended_training_set_v3.parquet')

    # 需要口袋特征的蛋白：有 AlphaFold 结构但没有口袋特征
    with h5py.File(h5_path, 'r') as h5:
        all_keys = set(h5.keys())
        has_structure = {k for k in all_keys if 'contact_number' in h5[k]}
        has_pocket = {k for k in all_keys if 'pocket_ca_distances' in h5[k]}
        todo_struct = has_structure - has_pocket

    # 映射到 UniProt ID
    valid = df[df['protein_seq_hash'].notna() & df['protein_seq_hash'].isin(todo_struct)]
    hash_to_uid = {}
    for h, uid in valid.groupby('protein_seq_hash')['uniprot_id'].first().items():
        if pd.notna(uid):
            hash_to_uid[h] = str(uid)

    todo = list(hash_to_uid.items())
    if args.max_proteins:
        todo = todo[:args.max_proteins]

    log.info(f"需口袋特征: {len(todo):,} (total no-pocket: {len(todo_struct):,})")
    est = len(todo) * 1.5 / args.workers / 3600
    log.info(f"预估: ~{est:.1f}h with {args.workers} workers")

    n_ok = n_fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, h, uid, h5_path): (h, uid) for h, uid in todo}
        pbar = tqdm(as_completed(futures), total=len(futures), desc="Pocket features")
        for f in pbar:
            h, uid = futures[f]
            try:
                if f.result(): n_ok += 1
                else: n_fail += 1
            except: n_fail += 1
            if (n_ok + n_fail) % 200 == 0:
                pbar.set_postfix(ok=n_ok, fail=n_fail)

    h5 = h5py.File(h5_path, 'r')
    n_pocket = sum(1 for k in h5.keys() if 'pocket_ca_distances' in h5[k])
    h5.close()
    log.info(f"完成: {n_ok:,} OK, {n_fail:,} fail")
    log.info(f"H5 有口袋特征: {n_pocket:,} / {len(all_keys):,}")


if __name__ == "__main__":
    main()
