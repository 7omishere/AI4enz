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
import contextlib
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

# 当前 models/ 目录即包含 ranking_model
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from ranking_model import (
    Trenzition, create_trenzition_optimizer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# 路径
PROJECT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_DIR.parent / "processed"
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
    # pKd: 范围 [0, 12] → [0, 1]（数据 P99.9=10.32，留余量至 12）
    'pkd_min': 0.0,
    'pkd_max': 12.0,
    # log10(kcat): 范围 [-7, 8] → [0, 1]（数据 P0.01=-6.98, P100=7.76）
    'kcat_min': -7.0,
    'kcat_max': 8.0,
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
    """酶数据集：直接加载 unified_metadata.parquet + proteins.h5。

    精简版 (v2)：仅加载序列嵌入 + 配体图 + 辅因子字符串。
    结构/口袋/域特征已从 ProteinEncoder 移除，不再加载。
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

        # ── 过滤无效 seq_hash / 检查 ESM-2 覆盖率 ──
        # 提前打开 H5 获取有效 key 集合（之后每个 worker 会独立打开）
        n_before = len(self.df)
        null_mask = self.df["protein_seq_hash"].isna()
        n_null = null_mask.sum()
        with h5py.File(proteins_h5_path, "r") as _h5:
            _valid_keys = set(_h5.keys())
            # 统计 ESM-2 覆盖率（不影响过滤，仅用于诊断）
            _esm2_keys = {k for k in _valid_keys if "esm2_embed" in _h5[k]}
        valid_mask = self.df["protein_seq_hash"].notna()
        in_h5_mask = valid_mask & self.df["protein_seq_hash"].isin(_valid_keys)
        self.df = self.df[in_h5_mask].reset_index(drop=True)
        n_filtered = n_before - len(self.df)
        if n_filtered > 0:
            log.info(f"  Filtered out {n_filtered:,} samples "
                     f"(null seq_hash: {n_null:,}, not in H5: {n_filtered - n_null:,})")

        n_with_esm2 = self.df["protein_seq_hash"].isin(_esm2_keys).sum()
        if use_esm2 and n_with_esm2 < len(self.df):
            log.info(f"  ESM-2 coverage: {n_with_esm2:,}/{len(self.df):,} "
                     f"({n_with_esm2/len(self.df)*100:.1f}%) — "
                     f"{len(self.df) - n_with_esm2:,} samples will use AA-property fallback")

        if max_samples:
            self.df = self.df.head(max_samples)
            log.info(f"  Limited to {max_samples} samples")

        # Log statistics
        n_with_pkd = self.df["pkd_raw"].notna().sum()
        n_with_kcat = self.df["has_kcat"].sum()
        n_with_cofactor = (self.df["cofactors"].notna() & (self.df["cofactors"] != "")).sum()
        log.info(f"  With pKd: {n_with_pkd}, With kcat: {n_with_kcat}, "
                 f"With cofactor: {n_with_cofactor}")
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
        # Ensure 'smiles' attribute exists for PyG collate compatibility
        if not hasattr(ligand_data, 'smiles') or ligand_data.smiles is None:
            ligand_data.smiles = ""

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

        # 负样本标记 (用于 gate 正则化)
        is_negative = bool(row.get("is_negative", False))

        return {
            "ligand_data": ligand_data,
            "seq_embed": seq_embed,
            "cofactor_str": cofactor_str,
            # ✅ 归一化后的目标值 [0, 1]，用于损失计算
            "pkd_target": torch.tensor(pkd_normalized, dtype=torch.float32),
            "has_pkd": torch.tensor(has_pkd, dtype=torch.bool),
            "log_kcat_target": torch.tensor(kcat_normalized, dtype=torch.float32),
            "has_kcat": torch.tensor(has_kcat, dtype=torch.bool),
            "kcat_weight": torch.tensor(kcat_weight, dtype=torch.float32),
            "quality_weight": torch.tensor(quality_weight, dtype=torch.float32),
            "is_negative": torch.tensor(is_negative, dtype=torch.bool),
            # 原始值（用于评估和反归一化）
            "pkd_raw": torch.tensor(pkd_val if has_pkd else 0.0, dtype=torch.float32),
            "log_kcat_raw": torch.tensor(log_kcat_label if has_kcat else 0.0, dtype=torch.float32),
        }


def collate_fn(batch: list[dict]) -> dict:
    """自定义 collate：处理 PyG 图"""
    from torch_geometric.data import Batch as PyGBatch

    ligand_batch = PyGBatch.from_data_list([item["ligand_data"] for item in batch])

    seq_embed = torch.stack([item["seq_embed"] for item in batch])
    cofactor_strs = [item["cofactor_str"] for item in batch]
    pkd_target = torch.stack([item["pkd_target"] for item in batch])
    has_pkd = torch.stack([item["has_pkd"] for item in batch])
    log_kcat_target = torch.stack([item["log_kcat_target"] for item in batch])
    has_kcat = torch.stack([item["has_kcat"] for item in batch])
    kcat_weight = torch.stack([item["kcat_weight"] for item in batch])
    quality_weight = torch.stack([item["quality_weight"] for item in batch])

    is_negative = torch.stack([item["is_negative"] for item in batch])

    return {
        "ligand_data": ligand_batch,
        "seq_embed": seq_embed,
        "cofactor_strs": cofactor_strs,
        "pkd_target": pkd_target,
        "pkd_target_mask": has_pkd,
        "log_kcat_target": log_kcat_target,
        "kcat_target_mask": has_kcat,
        "kcat_weights": kcat_weight,
        "quality_weight": quality_weight,
        "is_negative": is_negative,
    }


# ─────────────────────────────────────────────────────────────
# 训练器
# ─────────────────────────────────────────────────────────────

class Trainer:
    """训练循环封装。支持 CPU / GPU、AMP 混合精度、torch.compile、负样本 gate 正则化。"""

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
                 use_amp: bool = False,
                 use_compile: bool = False,
                 grad_accum_steps: int = 1,
                 warmup_steps: int = 1000,
                 gate_weight: float = 0.02,
                 ):
        self.device = device
        self.use_amp = use_amp and device.startswith("cuda")
        self.grad_accum_steps = grad_accum_steps
        self.gate_weight = gate_weight

        # ── torch.compile (PyTorch ≥ 2.0) ──
        if use_compile:
            if hasattr(torch, 'compile'):
                log.info("Compiling model with torch.compile (mode='reduce-overhead')...")
                model = torch.compile(model, mode="reduce-overhead")
            else:
                log.warning("torch.compile not available (requires PyTorch ≥ 2.0), skipping.")

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model_type = model_type

        # ── AMP scaler (CUDA only) ──
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        self.global_step = 0
        self.warmup_steps = warmup_steps

        if optimizer_fn is not None:
            self.optimizer = optimizer_fn(model, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = create_trenzition_optimizer(model, lr=lr, weight_decay=weight_decay)

        # ── Scheduler: warmup + CosineAnnealingWarmRestarts ──
        base_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5000, T_mult=2,
        )
        if warmup_steps > 0:
            from torch.optim.lr_scheduler import LinearLR, SequentialLR
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=1e-6,    # 从几乎 0 开始
                end_factor=1.0,        # 线性升至原始 LR
                total_iters=warmup_steps,
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                [warmup_scheduler, base_scheduler],
                milestones=[warmup_steps],
            )
            log.info(f"Warmup: {warmup_steps} steps linear increase → CosineAnnealingWarmRestarts")
        else:
            self.scheduler = base_scheduler

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float("inf")
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

        if self.use_amp:
            log.info("AMP (automatic mixed precision) enabled — bfloat16/float16")
        if self.grad_accum_steps > 1:
            log.info(f"Gradient accumulation: {self.grad_accum_steps} steps "
                     f"(effective batch={self.grad_accum_steps * train_loader.batch_size})")

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        epoch_losses = {}
        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # 移动到设备
            batch_gpu = {
                k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            # 特殊处理 PyG Batch 对象
            batch_gpu["ligand_data"] = batch_gpu["ligand_data"].to(self.device)

            # ── 前向（AMP 可选） ──
            with torch.amp.autocast("cuda") if self.use_amp else contextlib.nullcontext():
                outputs = self.model(
                    batch_gpu["ligand_data"],
                    batch_gpu["seq_embed"],
                    batch_gpu["cofactor_strs"],
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

                # ── Gate 正则化 (负样本学习) ──
                if self.gate_weight > 0 and outputs.get("gate_profile") is not None:
                    gate_profile = outputs["gate_profile"]  # (n_steps, B)
                    gate_mean = gate_profile.mean(dim=0)    # (B,)  每样本平均门控值
                    is_neg = batch_gpu.get("is_negative", torch.zeros_like(gate_mean, dtype=torch.bool))

                    if is_neg.any():
                        # 正样本: gate → 1 | 负样本: gate → 0
                        l_gate_pos = ((1.0 - gate_mean[~is_neg]) ** 2).mean() if (~is_neg).any() else 0.0
                        l_gate_neg = (gate_mean[is_neg] ** 2).mean() if is_neg.any() else 0.0
                        l_gate = self.gate_weight * (l_gate_pos + l_gate_neg)
                        total_loss = total_loss + l_gate
                        losses["L_gate"] = l_gate.item() if isinstance(l_gate, torch.Tensor) else float(l_gate)

                # 梯度累积归一化
                total_loss = total_loss / self.grad_accum_steps

            # ── 反向传播 ──
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            # 梯度累积：每 grad_accum_steps 步更新一次
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

            # 累加（跳过 weights 字典）
            for k, v in losses.items():
                if k == 'weights':
                    continue
                epoch_losses.setdefault(k, 0.0)
                epoch_losses[k] += v.item() if hasattr(v, 'item') else float(v)

            # 进度条
            pbar.set_postfix({
                "loss": f"{total_loss.item() if hasattr(total_loss, 'item') else float(total_loss):.3f}",
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
                k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            # 特殊处理 PyG Batch 对象
            batch_gpu["ligand_data"] = batch_gpu["ligand_data"].to(self.device)

            outputs = self.model(
                batch_gpu["ligand_data"],
                batch_gpu["seq_embed"],
                batch_gpu["cofactor_strs"],
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

            # Gate 正则化 (val)
            if self.gate_weight > 0 and outputs.get("gate_profile") is not None:
                gate_mean = outputs["gate_profile"].mean(dim=0)
                is_neg = batch_gpu.get("is_negative", torch.zeros_like(gate_mean, dtype=torch.bool))
                if is_neg.any():
                    l_gate_pos = ((1.0 - gate_mean[~is_neg]) ** 2).mean() if (~is_neg).any() else 0.0
                    l_gate_neg = (gate_mean[is_neg] ** 2).mean() if is_neg.any() else 0.0
                    l_gate = self.gate_weight * (l_gate_pos + l_gate_neg)
                    losses["L_gate"] = l_gate.item() if isinstance(l_gate, torch.Tensor) else float(l_gate)

            # 累加（跳过 weights 字典）
            for k, v in losses.items():
                if k == 'weights':
                    continue
                val_losses.setdefault(k, 0.0)
                val_losses[k] += v.item() if hasattr(v, 'item') else float(v)

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
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
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
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
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
                f"L_eyr: {train_losses.get('L_eyring', 0):.4f}  "
                f"L_gate: {train_losses.get('L_gate', 0):.4f}"
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
                k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            # 特殊处理 PyG Batch 对象
            batch_gpu["ligand_data"] = batch_gpu["ligand_data"].to(self.device)
            outputs = self.model(
                batch_gpu["ligand_data"],
                batch_gpu["seq_embed"],
                batch_gpu["cofactor_strs"],
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
                total_losses[k] += v.item() if hasattr(v, 'item') else float(v)

        n = max(len(loader), 1)
        return {k: v / n for k, v in total_losses.items()}


# ─────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Trenzition")
    # 数据
    parser.add_argument("--unified-metadata", default=str(
        PROCESSED_DIR / "metadata.parquet"))
    parser.add_argument("--proteins-h5", default=str(PROCESSED_DIR / "proteins.h5"))
    parser.add_argument("--ligand-dir", default=str(LIGAND_DIR))
    # 训练
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-samples", type=int, default=None)
    # 模型参数 (Trenzition)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--gnn-layers", type=int, default=3)
    parser.add_argument("--n-ode-steps", type=int, default=1,
                        help="BINN的ODE积分步数（v3默认1步，消融实验确认5步无额外收益）")
    parser.add_argument("--no-gate", action="store_true",
                        help="Disable Gate (ablation only, gate is essential for negative sample training)")
    # 硬件 / GPU
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers (GPU: 4-8, CPU: 0-2)")
    parser.add_argument("--amp", action="store_true",
                        help="Enable AMP mixed precision (CUDA only, ~2x speed, less VRAM)")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile (PyTorch ≥ 2.0, ~30%% speedup on GPU)")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size × grad_accum)")
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="Linear LR warmup steps (0=disable)")
    parser.add_argument("--no-esm2", action="store_true",
                        help="Use AA properties instead of ESM-2 (default: ESM-2)")
    # 检查点
    parser.add_argument("--checkpoint-dir", default=str(CHECKPOINT_DIR))
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--finetune", default=None, help="Fine-tune from a checkpoint (lower LR on encoders)")
    parser.add_argument("--save-every", type=int, default=10)
    # 负样本 / Gate 正则化
    parser.add_argument("--gate-weight", type=float, default=0.02,
                        help="Gate regularization weight (0=disable). Higher = stronger pos/neg separation")
    args = parser.parse_args()

    log.info(f"Device: {args.device}")
    log.info("Using Trenzition model")
    if args.gate_weight > 0:
        log.info(f"Gate regularization: weight={args.gate_weight}")

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

    use_cuda = args.device.startswith("cuda")
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers,
        pin_memory=use_cuda, persistent_workers=use_cuda and args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=min(args.num_workers, 2),
        pin_memory=use_cuda, persistent_workers=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=min(args.num_workers, 2),
        pin_memory=use_cuda, persistent_workers=False,
    )

    # ── 模型 ──
    if args.no_gate:
        log.warning("--no-gate: disabling gate (ablation only). Gate is needed for negative sample training.")
    model = Trenzition(
        hidden_dim=args.hidden_dim,
        gnn_layers=args.gnn_layers,
        n_ode_steps=args.n_ode_steps,
        use_gate=not args.no_gate,
    )
    optimizer_fn = create_trenzition_optimizer
    log.info(f"Trenzition: hidden={args.hidden_dim}, ODE steps={args.n_ode_steps}, "
             f"gate={'ON' if not args.no_gate else 'OFF'}, gate_weight={args.gate_weight}")

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
        use_amp=args.amp,
        use_compile=args.compile,
        grad_accum_steps=args.grad_accum,
        warmup_steps=args.warmup_steps,
        gate_weight=args.gate_weight,
    )

    # ── Fine-tune: 从已有权重加载，降低 encoder 学习率 ──
    if args.finetune:
        log.info(f"Fine-tuning from {args.finetune}")
        ckpt = torch.load(args.finetune, map_location=args.device, weights_only=False)
        state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
        model.load_state_dict(state, strict=False)
        # 降低 encoder 学习率 (10x smaller)，让 gate/dynamics 更快适应
        for name, param in model.named_parameters():
            if any(x in name for x in ['ligand_encoder', 'protein_encoder', 'cofactor_encoder']):
                param.requires_grad = True  # 仍然可训练，但 LR 更低
        log.info("Encoder LR reduced (10x) — gate/dynamics adapt faster")

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
