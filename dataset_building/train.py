"""
train.py
=======
酶挖掘排序模型训练脚本。

训练策略：
  1. 阶段 1 (warmup): 仅 pKd 损失，让底物排序能力先收敛
  2. 阶段 2 (联合): pKd + kcat 联合训练（kcat 为蛋白级辅助任务）

用法：
  # 小规模测试
  python train.py --epochs 5 --batch-size 32 --max-samples 5000

  # 完整训练
  python train.py --epochs 100 --batch-size 128

  # 断点续训
  python train.py --resume checkpoints/last.ckpt
"""

import os
import sys
import json
import pickle
import argparse
import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.data import Data as PyGData
from tqdm import tqdm

# 添加 datepre 目录到路径
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "datepre"))
from ranking_model import (
    TransitionBINN, create_bin_optimizer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# 路径
PROJECT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_DIR / "processed"
OXIDOREDUCTASE_DIR = PROCESSED_DIR / "oxidoreductase"
LIGAND_DIR = PROCESSED_DIR / "ligands"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"

# ─────────────────────────────────────────────────────────────
# 氨基酸物化性质（轻量级蛋白质编码，作为 ESM-2 的备选）
# ─────────────────────────────────────────────────────────────

AA_PROPERTIES: dict[str, list[float]] = {
    #          hydroph  volume  charge  polarity  flexibility  SA
    "A": [1.8, 31.0, 0.0, 0.0, 1.0, 1.0],
    "R": [-4.5, 124.0, 1.0, 1.0, 1.0, 6.13],
    "N": [-3.5, 56.0, 0.0, 1.0, 1.0, 2.95],
    "D": [-3.5, 54.0, -1.0, 1.0, 1.0, 2.78],
    "C": [2.5, 55.0, 0.0, 0.0, 0.0, 1.0],
    "Q": [-3.5, 85.0, 0.0, 1.0, 1.0, 3.0],
    "E": [-3.5, 83.0, -1.0, 1.0, 1.0, 3.0],
    "G": [-0.4, 3.0, 0.0, 0.0, 2.0, 0.0],
    "H": [-3.2, 96.0, 0.1, 1.0, 1.0, 2.98],
    "I": [4.5, 111.0, 0.0, 0.0, 1.0, 4.0],
    "L": [3.8, 111.0, 0.0, 0.0, 1.0, 4.0],
    "K": [-3.9, 119.0, 1.0, 1.0, 1.0, 5.0],
    "M": [1.9, 105.0, 0.0, 0.0, 1.0, 3.8],
    "F": [2.8, 132.0, 0.0, 0.0, 0.0, 5.89],
    "P": [-1.6, 32.0, 0.0, 0.0, 2.0, 2.5],
    "S": [-0.8, 32.0, 0.0, 1.0, 1.0, 1.5],
    "T": [-0.7, 61.0, 0.0, 1.0, 1.0, 2.6],
    "W": [-0.9, 170.0, 0.0, 1.0, 0.0, 8.08],
    "Y": [-1.3, 136.0, 0.0, 1.0, 0.0, 6.47],
    "V": [4.2, 84.0, 0.0, 0.0, 1.0, 3.0],
}
AA_PROP_DIM = len(next(iter(AA_PROPERTIES.values())))

# 热力学数据分层权重：反映不同测量类型对热力学平衡常数的可信度
THERMO_WEIGHT = {
    'Kd':   1.00,   # 直接热力学平衡常数，最可信
    'Ki':   0.70,   # 竞争性抑制常数，需假设机制
    'IC50': 0.15,   # 非热力学量，经ChEMBL校正后仍有系统误差
}

# ─────────────────────────────────────────────────────────────
# Min-Max 归一化参数（基于数据集统计）
# 用于平衡 pKd 和 log_kcat 的损失量级
# ─────────────────────────────────────────────────────────────
NORM_PARAMS = {
    # pKd: 范围 [0, 14] → [0, 1]
    'pkd_min': 0.0,
    'pkd_max': 14.0,
    # log10(kcat): 范围 [-6, 7] → [0, 1]
    'kcat_min': -6.0,
    'kcat_max': 7.0,
}


def min_max_normalize(value, min_val, max_val):
    """Min-Max 归一化到 [0, 1]"""
    return (value - min_val) / (max_val - min_val)


def min_max_denormalize(value, min_val, max_val):
    """反归一化"""
    return value * (max_val - min_val) + min_val


def sequence_to_embedding(seq: str, max_len: int = 1020) -> torch.Tensor:
    """轻量级：氨基酸物化性质编码 → mean pooling"""
    feats = []
    for aa in seq[:max_len].upper():
        if aa in AA_PROPERTIES:
            feats.append(AA_PROPERTIES[aa])
        else:
            feats.append([0.0] * AA_PROP_DIM)
    if not feats:
        return torch.zeros(AA_PROP_DIM)
    return torch.tensor(feats, dtype=torch.float32).mean(dim=0)


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

class OxidoreductaseDataset(Dataset):
    """氧化还原酶数据集：直接加载 unified_metadata.parquet + proteins.h5。

    修复了三个关键 bug：
      1. 序列从 proteins.h5 加载（而非 metadata 中不存在的 "sequence" 列）
      2. struct_feat 从 proteins.h5 计算（而非始终为零）
      3. 使用统一数据集，消除 record_idx 合并错误
    """

    def __init__(self,
                 unified_metadata_path: str,
                 proteins_h5_path: str,
                 ligand_dir: str,
                 split: str = "train",
                 max_samples: Optional[int] = None,
                 use_esm2: bool = True,
                 ):
        self.ligand_dir = Path(ligand_dir)
        self.split = split
        self.use_esm2 = use_esm2
        self.proteins_h5_path = proteins_h5_path
        self._h5 = None  # 每个 worker 进程惰性打开

        # 直接加载统一元数据（已包含所有 join 和标签）
        log.info(f"Loading unified metadata from {unified_metadata_path}")
        self.df = pd.read_parquet(unified_metadata_path)

        # Split 筛选
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        log.info(f"  {split} split: {len(self.df):,} samples")

        if max_samples:
            self.df = self.df.head(max_samples)
            log.info(f"  Limited to {max_samples} samples")

        # Log statistics
        n_with_pkd = self.df["pkd_raw"].notna().sum()
        n_with_kcat = self.df["has_kcat"].sum()
        n_with_cofactor = (self.df["cofactors"].notna() & (self.df["cofactors"] != "")).sum()
        n_with_structure = self.df["has_structure"].sum()
        n_with_domain = self.df["has_domain_annotation"].sum()
        log.info(f"  With pKd: {n_with_pkd}, With kcat: {n_with_kcat}, "
                 f"With cofactor: {n_with_cofactor}")
        log.info(f"  With structure: {n_with_structure}, With domains: {n_with_domain}")
        log.info(f"  Protein encoding: {'ESM-2 (precomputed)' if use_esm2 else 'AA properties'}")

    @property
    def h5(self):
        """惰性打开 h5 文件（每个 DataLoader worker 独立打开）"""
        if self._h5 is None:
            self._h5 = h5py.File(self.proteins_h5_path, "r")
        return self._h5

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        seq_hash = row["protein_seq_hash"]
        group = self.h5[seq_hash]

        # ── 蛋白序列（从 proteins.h5 加载） ──
        seq_bytes = group["sequence"][()]
        if isinstance(seq_bytes, bytes):
            seq = seq_bytes.decode("utf-8")
        elif isinstance(seq_bytes, np.ndarray):
            seq = str(seq_bytes)
        else:
            seq = str(seq_bytes)

        # ── 序列嵌入 ──
        if self.use_esm2 and "esm2_embed" in group:
            # ESM-2 预计算嵌入（1280-dim，从 proteins.h5 直接加载）
            seq_embed = torch.from_numpy(group["esm2_embed"][:]).float()
        else:
            # AA属性路径: 6维物化特征 → ProteinEncoder.aa_proj 学习投影
            seq_embed = sequence_to_embedding(seq)  # (AA_PROP_DIM,)

        # ── 结构特征（从 proteins.h5 计算） ──
        has_structure = bool(row["has_structure"])
        if has_structure and "contact_number" in group:
            cn_mean = float(np.mean(group["contact_number"][:]))
            pi_mean = float(np.mean(group["protrusion_index"][:]))
            bs_flag = 1.0 if row["has_binding_site"] else 0.0
            struct_feat = torch.tensor([cn_mean, pi_mean, bs_flag], dtype=torch.float32)
        else:
            struct_feat = torch.zeros(3)

        # ── 口袋几何特征（从 proteins.h5 加载） ──
        if "pocket_ca_distances" in group:
            pocket_cn = torch.from_numpy(group["pocket_contact_number"][:]).float()
            pocket_pi = torch.from_numpy(group["pocket_protrusion_index"][:]).float()
            pocket_dist = torch.from_numpy(group["pocket_ca_distances"][:]).float()
            pocket_mask = torch.ones(len(pocket_cn), dtype=torch.bool)
        else:
            pocket_cn = torch.zeros(0)
            pocket_pi = torch.zeros(0)
            pocket_dist = torch.zeros(0, 0)
            pocket_mask = torch.zeros(0, dtype=torch.bool)

        # ── 域掩码（从 proteins.h5 加载） ──
        if "domain_masks" in group:
            domain_masks = torch.from_numpy(group["domain_masks"][:]).float()
        else:
            domain_masks = torch.zeros(15, max(len(seq), 1))

        # ── 配体分子图 ──
        inchikey = row["ligand_inchikey"]
        ligand_path = self.ligand_dir / f"{inchikey}.pt"
        if ligand_path.exists():
            ligand_data = torch.load(ligand_path, weights_only=False)
        else:
            ligand_data = PyGData(
                x=torch.zeros(1, 79),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                edge_attr=torch.zeros(0, 10),
            )

        # ── 辅因子 ──
        cofactor_str = row.get("cofactors", "") or ""

        # ── 标签（已归一化到 [0, 1]） ──
        pkd_val = row["pkd_aligned"] if pd.notna(row["pkd_aligned"]) else row["pkd_raw"]
        has_pkd = pd.notna(pkd_val)

        has_kcat = bool(row["has_kcat"])
        log_kcat_label = float(row["log_kcat_median"]) if has_kcat else 0.0

        # ✅ Min-Max 归一化目标值（统一到 [0, 1] 范围）
        pkd_normalized = min_max_normalize(
            pkd_val, NORM_PARAMS['pkd_min'], NORM_PARAMS['pkd_max']
        ) if has_pkd else 0.0
        kcat_normalized = min_max_normalize(
            log_kcat_label, NORM_PARAMS['kcat_min'], NORM_PARAMS['kcat_max']
        ) if has_kcat else 0.0

        # kcat 来源权重（BRENDA=1.0, SABIO-RK=0.9, UniProt=0.7）
        kcat_source = str(row.get("kcat_source", ""))
        if "bdb" in kcat_source:
            kcat_weight = 1.0
        elif "sabio" in kcat_source:
            kcat_weight = 0.9
        else:
            kcat_weight = 0.7

        # 最终质量权重 = 热力学分层权重 × 原有 quality_weight（censored惩罚等）
        measurement_type = str(row.get("measurement_type", "Kd"))
        thermo_w = THERMO_WEIGHT.get(measurement_type, 0.5)
        base_qw = float(row.get("quality_weight", 1.0))
        quality_weight = thermo_w * base_qw

        return {
            "ligand_data": ligand_data,
            "seq_embed": seq_embed,
            "struct_feat": struct_feat,
            "domain_masks": domain_masks,
            "has_structure": torch.tensor(has_structure, dtype=torch.bool),
            "cofactor_str": cofactor_str,
            # ✅ 归一化后的目标值 [0, 1]，用于损失计算
            "pkd_target": torch.tensor(pkd_normalized, dtype=torch.float32),
            "has_pkd": torch.tensor(has_pkd, dtype=torch.bool),
            "log_kcat_target": torch.tensor(kcat_normalized, dtype=torch.float32),
            "has_kcat": torch.tensor(has_kcat, dtype=torch.bool),
            "kcat_weight": torch.tensor(kcat_weight, dtype=torch.float32),
            "quality_weight": torch.tensor(quality_weight, dtype=torch.float32),
            "pocket_cn": pocket_cn,
            "pocket_pi": pocket_pi,
            "pocket_dist": pocket_dist,
            "pocket_mask": pocket_mask,
            # 原始值（用于评估和反归一化）
            "pkd_raw": torch.tensor(pkd_val if has_pkd else 0.0, dtype=torch.float32),
            "log_kcat_raw": torch.tensor(log_kcat_label if has_kcat else 0.0, dtype=torch.float32),
        }


def collate_fn(batch: list[dict]) -> dict:
    """自定义 collate：处理 PyG 图、变长 domain_masks"""
    from torch_geometric.data import Batch as PyGBatch

    ligand_batch = PyGBatch.from_data_list([item["ligand_data"] for item in batch])

    seq_embed = torch.stack([item["seq_embed"] for item in batch])
    struct_feat = torch.stack([item["struct_feat"] for item in batch])
    has_structure = torch.stack([item["has_structure"] for item in batch])

    cofactor_strs = [item["cofactor_str"] for item in batch]
    pkd_target = torch.stack([item["pkd_target"] for item in batch])
    has_pkd = torch.stack([item["has_pkd"] for item in batch])
    log_kcat_target = torch.stack([item["log_kcat_target"] for item in batch])
    has_kcat = torch.stack([item["has_kcat"] for item in batch])
    kcat_weight = torch.stack([item["kcat_weight"] for item in batch])
    quality_weight = torch.stack([item["quality_weight"] for item in batch])

    # ── pocket 特征变长 padding ──
    pocket_cn_list = [item["pocket_cn"] for item in batch]
    pocket_pi_list = [item["pocket_pi"] for item in batch]
    pocket_dist_list = [item["pocket_dist"] for item in batch]
    pocket_mask_list = [item["pocket_mask"] for item in batch]
    max_K = max(cn.size(0) for cn in pocket_cn_list)
    B = len(batch)
    if max_K > 0:
        pocket_cn_padded = torch.zeros(B, max_K)
        pocket_pi_padded = torch.zeros(B, max_K)
        pocket_dist_padded = torch.zeros(B, max_K, max_K)
        pocket_mask_padded = torch.zeros(B, max_K, dtype=torch.bool)
        for i in range(B):
            K = pocket_cn_list[i].size(0)
            if K > 0:
                pocket_cn_padded[i, :K] = pocket_cn_list[i]
                pocket_pi_padded[i, :K] = pocket_pi_list[i]
                pocket_dist_padded[i, :K, :K] = pocket_dist_list[i]
                pocket_mask_padded[i, :K] = pocket_mask_list[i]
    else:
        pocket_cn_padded = torch.zeros(B, 1)
        pocket_pi_padded = torch.zeros(B, 1)
        pocket_dist_padded = torch.zeros(B, 1, 1)
        pocket_mask_padded = torch.zeros(B, 1, dtype=torch.bool)

    # ── domain_masks 变长 padding ──
    domain_masks_list = [item["domain_masks"] for item in batch]  # each (15, L_i)
    max_len = max(dm.size(-1) for dm in domain_masks_list)
    n_types = domain_masks_list[0].size(0)  # 15
    padded = torch.zeros(len(batch), n_types, max_len)
    for i, dm in enumerate(domain_masks_list):
        padded[i, :, :dm.size(-1)] = dm
    # padding mask: True where the position is valid (not padded)
    domain_padding_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, dm in enumerate(domain_masks_list):
        domain_padding_mask[i, :dm.size(-1)] = True

    return {
        "ligand_data": ligand_batch,
        "seq_embed": seq_embed,
        "struct_feat": struct_feat,
        "domain_masks": padded,
        "domain_padding_mask": domain_padding_mask,
        "has_structure": has_structure,
        "cofactor_strs": cofactor_strs,
        "pkd_target": pkd_target,
        "pkd_target_mask": has_pkd,
        "log_kcat_target": log_kcat_target,
        "kcat_target_mask": has_kcat,
        "kcat_weights": kcat_weight,
        "quality_weight": quality_weight,
        "pocket_cn": pocket_cn_padded,
        "pocket_pi": pocket_pi_padded,
        "pocket_dist": pocket_dist_padded,
        "pocket_mask": pocket_mask_padded,
    }


# ─────────────────────────────────────────────────────────────
# 训练器
# ─────────────────────────────────────────────────────────────

class Trainer:
    """训练循环封装"""

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: Optional[DataLoader] = None,
                 lr: float = 1e-4,
                 weight_decay: float = 1e-5,
                 device: str = "cuda",
                 checkpoint_dir: str = "checkpoints",
                 optimizer_fn=None,
                 model_type: str = "bin",
                 ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model_type = model_type

        self.global_step = 0

        if optimizer_fn is not None:
            self.optimizer = optimizer_fn(model, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = create_bin_optimizer(model, lr=lr, weight_decay=weight_decay)
        
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5000, T_mult=2,
        )

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float("inf")
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        epoch_losses = {}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            self.global_step += 1

            # 移动到设备
            batch_gpu = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            # 特殊处理 PyG Batch 对象
            batch_gpu["ligand_data"] = batch_gpu["ligand_data"].to(self.device)

            # 前向
            outputs = self.model(
                batch_gpu["ligand_data"],
                batch_gpu["seq_embed"],
                batch_gpu["cofactor_strs"],
                batch_gpu["struct_feat"],
                batch_gpu["has_structure"],
                domain_masks=batch_gpu.get("domain_masks"),
                domain_padding_mask=batch_gpu.get("domain_padding_mask"),
                pocket_cn=batch_gpu.get("pocket_cn"),
                pocket_pi=batch_gpu.get("pocket_pi"),
                pocket_dist=batch_gpu.get("pocket_dist"),
                pocket_mask=batch_gpu.get("pocket_mask"),
            )

            # 损失
            total_loss, losses = self.model.compute_loss(
                outputs,
                {
                    "pkd_target": batch_gpu["pkd_target"],
                    "pkd_target_mask": batch_gpu["pkd_target_mask"],
                    "log_kcat_target": batch_gpu["log_kcat_target"],
                    "kcat_target_mask": batch_gpu["kcat_target_mask"],
                    "kcat_weights": batch_gpu["kcat_weights"],
                    "quality_weight": batch_gpu["quality_weight"],
                },
            )

            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            # 累加（跳过 weights 字典）
            for k, v in losses.items():
                if k == 'weights':
                    continue
                epoch_losses.setdefault(k, 0.0)
                epoch_losses[k] += v.item()

            # 进度条
            pbar.set_postfix({
                "loss": f"{total_loss.item():.3f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

        # 平均
        n_batches = len(self.train_loader)
        return {k: v / n_batches for k, v in epoch_losses.items()}

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        val_losses = {}

        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            batch_gpu = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            # 特殊处理 PyG Batch 对象
            batch_gpu["ligand_data"] = batch_gpu["ligand_data"].to(self.device)

            outputs = self.model(
                batch_gpu["ligand_data"],
                batch_gpu["seq_embed"],
                batch_gpu["cofactor_strs"],
                batch_gpu["struct_feat"],
                batch_gpu["has_structure"],
                domain_masks=batch_gpu.get("domain_masks"),
                domain_padding_mask=batch_gpu.get("domain_padding_mask"),
                pocket_cn=batch_gpu.get("pocket_cn"),
                pocket_pi=batch_gpu.get("pocket_pi"),
                pocket_dist=batch_gpu.get("pocket_dist"),
                pocket_mask=batch_gpu.get("pocket_mask"),
            )

            total_loss, losses = self.model.compute_loss(
                outputs,
                {
                    "pkd_target": batch_gpu["pkd_target"],
                    "pkd_target_mask": batch_gpu["pkd_target_mask"],
                    "log_kcat_target": batch_gpu["log_kcat_target"],
                    "kcat_target_mask": batch_gpu["kcat_target_mask"],
                    "kcat_weights": batch_gpu["kcat_weights"],
                    "quality_weight": batch_gpu["quality_weight"],
                },
            )

            # 累加（跳过 weights 字典）
            for k, v in losses.items():
                if k == 'weights':
                    continue
                val_losses.setdefault(k, 0.0)
                val_losses[k] += v.item()

        n_batches = len(self.val_loader)
        return {k: v / n_batches for k, v in val_losses.items()}

    def save_checkpoint(self, filename: str, extra: dict | None = None):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }
        if extra:
            checkpoint.update(extra)

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        log.info(f"Checkpoint saved → {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        log.info(f"Checkpoint loaded from {path} (step {self.global_step})")

    def fit(self, epochs: int, save_every: int = 10):
        log.info("=" * 50)
        log.info(f"Starting training: {epochs} epochs")
        log.info(f"Train batches: {len(self.train_loader)}, "
                 f"Val batches: {len(self.val_loader)}")
        log.info("=" * 50)

        for epoch in range(1, epochs + 1):
            train_losses = self.train_epoch(epoch)
            val_losses = self.validate()

            # 记录
            self.train_losses.append(train_losses.get("total", 0))
            self.val_losses.append(val_losses.get("total", 0))

            # 日志
            loss_info = (
                f"Epoch {epoch:3d}/{epochs} | "
                f"train: {train_losses['total']:.4f}  "
                f"val: {val_losses['total']:.4f}  "
                f"(best: {self.best_val_loss:.4f})  |  "
                f"L_ts: {train_losses.get('L_ts', 0):.4f}  "
                f"L_cat: {train_losses.get('L_catalysis', 0):.4f}  "
                f"L_barrier: {train_losses.get('L_barrier', 0):.4f}"
            )
            log.info(loss_info)

            # 保存最佳
            val_total = val_losses["total"]
            if val_total < self.best_val_loss:
                self.best_val_loss = val_total
                self.save_checkpoint("best.ckpt", {
                    "epoch": epoch,
                    "train_losses": self.train_losses,
                    "val_losses": self.val_losses,
                })

            # 定期保存
            if epoch % save_every == 0:
                self.save_checkpoint(f"epoch_{epoch:04d}.ckpt", {"epoch": epoch})

            # 保存最新
            self.save_checkpoint("last.ckpt", {"epoch": epoch})

        # 最终评估
        if self.test_loader:
            test_losses = self.validate_with_loader(self.test_loader, "Testing")
            log.info(f"Test loss: {test_losses.get('total', 0):.4f}")
            return self.train_losses, self.val_losses, test_losses

        return self.train_losses, self.val_losses

    @torch.no_grad()
    def validate_with_loader(self, loader, desc="Eval") -> dict:
        self.model.eval()
        total_losses = {}
        for batch in tqdm(loader, desc=desc, leave=False):
            batch_gpu = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            # 特殊处理 PyG Batch 对象
            batch_gpu["ligand_data"] = batch_gpu["ligand_data"].to(self.device)
            outputs = self.model(
                batch_gpu["ligand_data"],
                batch_gpu["seq_embed"],
                batch_gpu["cofactor_strs"],
                batch_gpu["struct_feat"],
                batch_gpu["has_structure"],
                domain_masks=batch_gpu.get("domain_masks"),
                domain_padding_mask=batch_gpu.get("domain_padding_mask"),
                pocket_cn=batch_gpu.get("pocket_cn"),
                pocket_pi=batch_gpu.get("pocket_pi"),
                pocket_dist=batch_gpu.get("pocket_dist"),
                pocket_mask=batch_gpu.get("pocket_mask"),
            )
            _, losses = self.model.compute_loss(
                outputs,
                {
                    "pkd_target": batch_gpu["pkd_target"],
                    "pkd_target_mask": batch_gpu["pkd_target_mask"],
                    "log_kcat_target": batch_gpu["log_kcat_target"],
                    "kcat_target_mask": batch_gpu["kcat_target_mask"],
                    "kcat_weights": batch_gpu["kcat_weights"],
                    "quality_weight": batch_gpu["quality_weight"],
                },
            )
            # 累加（跳过 weights 字典）
            for k, v in losses.items():
                if k == 'weights':
                    continue
                total_losses.setdefault(k, 0.0)
                total_losses[k] += v.item()

        n = max(len(loader), 1)
        return {k: v / n for k, v in total_losses.items()}


# ─────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train TransitionBINN (基于过渡态理论)")
    # 数据
    parser.add_argument("--unified-metadata", default=str(
        OXIDOREDUCTASE_DIR / "unified_metadata.parquet"))
    parser.add_argument("--proteins-h5", default=str(PROCESSED_DIR / "proteins.h5"))
    parser.add_argument("--ligand-dir", default=str(LIGAND_DIR))
    # 训练
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-samples", type=int, default=None)
    # 模型参数 (TransitionBINN)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--gnn-layers", type=int, default=3)
    parser.add_argument("--n-ode-steps", type=int, default=5,
                        help="BINN的ODE积分步数")
    parser.add_argument("--no-gate", action="store_true",
                        help="禁用BINN的门控机制")
    # 硬件
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-esm2", action="store_true",
                        help="Use AA properties instead of ESM-2 (default: ESM-2)")
    # 检查点
    parser.add_argument("--checkpoint-dir", default=str(CHECKPOINT_DIR))
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()

    log.info(f"Device: {args.device}")
    log.info("Using TransitionBINN model (Transition State Theory)")

    # ── 数据集 ──
    train_dataset = OxidoreductaseDataset(
        args.unified_metadata, args.proteins_h5, args.ligand_dir,
        split="train", max_samples=args.max_samples, use_esm2=not args.no_esm2,
    )
    val_dataset = OxidoreductaseDataset(
        args.unified_metadata, args.proteins_h5, args.ligand_dir,
        split="val", max_samples=args.max_samples // 8 if args.max_samples else None,
        use_esm2=not args.no_esm2,
    )
    test_dataset = OxidoreductaseDataset(
        args.unified_metadata, args.proteins_h5, args.ligand_dir,
        split="test",
        max_samples=args.max_samples // 8 if args.max_samples else None,
        use_esm2=not args.no_esm2,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )

    # ── 模型（TransitionBINN） ──
    model = TransitionBINN(
        hidden_dim=args.hidden_dim,
        gnn_layers=args.gnn_layers,
        n_ode_steps=args.n_ode_steps,
        use_gate=not args.no_gate,
    )
    optimizer_fn = create_bin_optimizer
    log.info("Using TransitionBINN model (Transition State Theory)")

    log.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ── 训练器 ──
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        optimizer_fn=optimizer_fn,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # ── 训练 ──
    train_losses, val_losses, *test = trainer.fit(
        epochs=args.epochs,
        save_every=args.save_every,
    )

    # ── 保存训练历史 ──
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    if test:
        history["test_losses"] = test[0]

    import json as _json
    history_path = Path(args.checkpoint_dir) / "training_history.json"
    with open(history_path, "w") as f:
        _json.dump(history, f, indent=2)
    log.info(f"Training history saved → {history_path}")


if __name__ == "__main__":
    main()
