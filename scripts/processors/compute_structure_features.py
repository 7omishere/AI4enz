"""
compute_structure_features.py
=============================
从 AlphaFold DB 下载预测结构，计算 contact_number, protrusion_index, contact_map
并存储到 proteins.h5。

用法：
  python scripts/processors/compute_structure_features.py --max-proteins 50  # 测试
  python scripts/processors/compute_structure_features.py --workers 8        # 完整运行
"""

import argparse, logging, time, os
from pathlib import Path
import numpy as np
import h5py
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent / "dataset_building"
PROCESSED_DIR = PROJECT_DIR / "processed"
PROTEINS_H5 = PROCESSED_DIR / "proteins.h5"

AF_API = "https://alphafold.ebi.ac.uk/api/prediction"
AF_FILE = "https://alphafold.ebi.ac.uk/files"

# 需要结构特征的蛋白缺少这些 dataset
REQUIRED_DATASETS = ['contact_number', 'protrusion_index', 'contact_map']


def compute_structural_features(coords: np.ndarray, ca_indices: np.ndarray | None = None):
    """
    从 3D 坐标计算结构特征。

    Args:
        coords: (N, 3) 所有原子的 XYZ 坐标
        ca_indices: Cα 原子的索引，如果为 None 则取所有原子

    Returns:
        contact_number: (N_ca,) 每个残基的 Cα 接触数（8Å 内其他 Cα 数量）
        protrusion_index: (N_ca,) 突起指数
        contact_map: (N_ca, N_ca) Cα-Cα 距离矩阵
    """
    if ca_indices is not None:
        ca_coords = coords[ca_indices]
    else:
        ca_coords = coords

    n = len(ca_coords)
    if n < 3:
        return np.zeros(n), np.zeros(n), np.zeros((n, n))

    # 距离矩阵
    diff = ca_coords[:, None, :] - ca_coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))

    # Contact map: Cα within 8Å
    contact_map = (dist < 8.0).astype(np.float32)
    np.fill_diagonal(contact_map, 0.0)

    # Contact number: number of neighbors within 8Å
    contact_number = contact_map.sum(axis=-1)

    # Protrusion index: how much a residue protrudes from the surface
    # Defined as: 1 - (contact_number / max_contact_number_of_neighbors)
    # Higher = more protruding
    max_contact = np.max(contact_number)
    if max_contact > 0:
        protrusion = 1.0 - (contact_number / max_contact)
    else:
        protrusion = np.zeros(n)

    return contact_number.astype(np.float32), protrusion.astype(np.float32), contact_map


def parse_pdb_coords(pdb_text: str):
    """解析 PDB 文本，提取 Cα 原子坐标"""
    ca_coords = []
    for line in pdb_text.split('\n'):
        if line.startswith('ATOM') and (line[12:16].strip() == 'CA' or 'CA' in line[12:16]):
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                ca_coords.append([x, y, z])
            except ValueError:
                continue
    return np.array(ca_coords, dtype=np.float32) if ca_coords else None


def fetch_alphafold_version(uniprot_id: str) -> tuple[str, int] | None:
    """获取 AlphaFold 模型的最新版本号"""
    try:
        resp = requests.get(f"{AF_API}/{uniprot_id}", timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and data:
                entry = data[0]
                return entry['modelEntityId'], entry['latestVersion']
    except Exception:
        pass
    return None


def download_alphafold_structure(model_id: str, version: int) -> str | None:
    """下载 AlphaFold PDB 结构"""
    url = f"{AF_FILE}/{model_id}-model_v{version}.pdb"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.text
    except Exception:
        pass
    return None


def process_one_protein(seq_hash: str, uniprot_id: str, h5_path: str) -> bool:
    """处理单个蛋白：下载 AlphaFold 结构 → 计算特征 → 写入 H5"""
    try:
        # 1. 获取 AlphaFold 版本
        af_info = fetch_alphafold_version(uniprot_id)
        if af_info is None:
            return False
        model_id, version = af_info

        # 2. 下载 PDB
        pdb_text = download_alphafold_structure(model_id, version)
        if pdb_text is None:
            return False

        # 3. 提取 Cα 坐标
        coords = parse_pdb_coords(pdb_text)
        if coords is None or len(coords) < 3:
            return False

        # 4. 计算特征
        cn, pi, cm = compute_structural_features(coords)

        # 5. 写入 H5
        with h5py.File(h5_path, 'r+') as h5:
            if seq_hash not in h5:
                return False
            grp = h5[seq_hash]

            # 删除旧数据（如果有）
            for ds_name in ['contact_number', 'protrusion_index', 'contact_map']:
                if ds_name in grp:
                    del grp[ds_name]

            grp.create_dataset('contact_number', data=cn)
            grp.create_dataset('protrusion_index', data=pi)
            grp.create_dataset('contact_map', data=cm)

            # 更新 binding_site_mask（如果只是占位符）
            if 'binding_site_mask' in grp:
                bsm = grp['binding_site_mask'][:]
                if np.all(bsm == 0):
                    del grp['binding_site_mask']
                    grp.create_dataset('binding_site_mask', data=np.zeros(len(cn), dtype=np.float32))

        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="从 AlphaFold 计算结构特征")
    parser.add_argument("--proteins-h5", default=str(PROTEINS_H5))
    parser.add_argument("--max-proteins", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--delay", type=float, default=0.1,
                        help="AlphaFold API 请求间延迟")
    args = parser.parse_args()

    h5_path = Path(args.proteins_h5)
    log.info(f"Opening {h5_path}")

    # 找到需要处理的蛋白
    import pandas as pd
    df = pd.read_parquet(PROJECT_DIR / 'release' / 'recommended_training_set_v3.parquet')

    with h5py.File(h5_path, 'r') as h5:
        all_keys = set(h5.keys())
        # 缺结构特征的蛋白
        no_struct = {k for k in all_keys if 'contact_number' not in h5[k]}

    # 映射 seq_hash → UniProt ID
    valid = df[df['protein_seq_hash'].notna() & df['protein_seq_hash'].isin(no_struct)]
    hash_to_uid = {}
    for h, uid in valid.groupby('protein_seq_hash')['uniprot_id'].first().items():
        if pd.notna(uid):
            hash_to_uid[h] = str(uid)

    todo = list(hash_to_uid.items())
    if args.max_proteins:
        todo = todo[:args.max_proteins]

    log.info(f"需要处理: {len(todo):,} 个蛋白 (total no-struct: {len(no_struct):,})")

    n_ok, n_fail = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for seq_hash, uid in todo:
            f = executor.submit(process_one_protein, seq_hash, uid, str(h5_path))
            futures[f] = (seq_hash, uid)

        pbar = tqdm(as_completed(futures), total=len(futures), desc="Structure features")
        for f in pbar:
            seq_hash, uid = futures[f]
            try:
                ok = f.result()
                if ok:
                    n_ok += 1
                else:
                    n_fail += 1
            except Exception:
                n_fail += 1

            if (n_ok + n_fail) % 100 == 0:
                pbar.set_postfix(ok=n_ok, fail=n_fail)

    # 最终统计
    h5_final = h5py.File(h5_path, 'r')
    n_struct = sum(1 for k in h5_final.keys() if 'contact_number' in h5_final[k])
    n_total = len(h5_final.keys())
    h5_final.close()
    log.info(f"完成: {n_ok:,} OK, {n_fail:,} fail")
    log.info(f"H5 有结构特征: {n_struct:,} / {n_total:,}")


if __name__ == "__main__":
    main()
