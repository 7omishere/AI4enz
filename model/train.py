"""
train.py
========
酶挖掘排序模型训练脚本。

三头联合训练：BindingDualHead(Kd/Ki) + KcatHead(BINN+ODE+Eyring) + KmHead

v4 新特性:
  - 交叉注意力融合（--use-cross-attn）：蛋白残基 × 配体原子
  - 异方差损失（--heteroscedastic）：预测 mean + variance，NLL 损失

用法：
  # v3 模式（向后兼容）
  python train.py --epochs 5 --batch-size 32 --max-samples 5000

  # v4 完全体（交叉注意力 + 异方差损失，GPU 推荐）
  python train.py --epochs 100 --batch-size 128 --use-cross-attn --heteroscedastic

  # v4 仅交叉注意力（SmoothL1 损失）
  python train.py --epochs 100 --batch-size 128 --use-cross-attn

  # 断点续训
  python train.py --resume checkpoints/last.ckpt
"""

import os
import sys
import json
import pickle
import random
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
from torch_geometric.data import Batch as PyGBatch
from tqdm import tqdm

# 当前 models/ 目录即包含 ranking_model
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from ranking_model import (
    Trenzition, create_threehead_optimizer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# 路径（从 model/ 回退到 AI4enz/dataset_building/）
PROJECT_DIR = Path(__file__).resolve().parent
BASE_DIR = PROJECT_DIR.parent
DATASET_DIR = BASE_DIR / "dataset_building"
PROCESSED_DIR = DATASET_DIR / "processed"
LIGAND_DIR = PROCESSED_DIR / "ligands"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"

# ─────────────────────────────────────────────────────────────
# Min-Max 归一化参数
# ─────────────────────────────────────────────────────────────
NORM_PARAMS = {
    'pkd_min': 0.0,
    'pkd_max': 12.0,
    'kcat_min': -7.0,
    'kcat_max': 8.0,
    'km_min': -13.0,
    'km_max': 3.0,
}


def min_max_normalize(value, min_val, max_val):
    """Min-Max 归一化到 [0, 1]"""
    return (value - min_val) / (max_val - min_val)


def min_max_denormalize(value, min_val, max_val):
    """反归一化"""
    return value * (max_val - min_val) + min_val


# ─────────────────────────────────────────────────────────────
# 分层采样器（防止任务偏差）
# ─────────────────────────────────────────────────────────────


class StratifiedBatchSampler:
    """三任务分层采样器：每个 batch 从 binding/kcat/km 三组各采等量样本。

    设计目标：
      - 每个 step 三个头都有足够的梯度信号
      - 共享层（encoder）不会只看到 binding 样本
      - kcat/Km 数据少，通过 oversample 补齐；binding 数据多，每个 epoch 只采子集

    实现：
      - 每个 batch = n_per_group 条 binding + n_per_group 条 kcat + n_per_group 条 Km
      - epoch 长度 = 最小的组的样本数 / n_per_group（~190 batch/epoch for B=128）
      - kcat/Km 样本少，会在一个 epoch 内被重复采样；binding 采一部分
      - collate_fn 和 loss masking 自动处理各组标签有无（已有机制）
    """

    def __init__(self, df, batch_size: int):
        # 三个组（有 overlap：部分样本同时有 kcat+Km）
        pkd_mask = df["pkd_raw"].notna()
        kcat_mask = df["has_kcat"] == True
        km_mask = df["has_km"] == True

        self.binding_idx = df.index[pkd_mask].tolist()
        self.kcat_idx = df.index[kcat_mask].tolist()
        self.km_idx = df.index[km_mask].tolist()

        self.batch_size = batch_size
        self.n_per_group = max(batch_size // 3, 1)

        # epoch 长度由最小组的样本量决定
        min_group = min(len(self.binding_idx), len(self.kcat_idx), len(self.km_idx))
        self.n_batches = max(min_group // self.n_per_group, 1)

        log.info("StratifiedBatchSampler:")
        log.info(f"  binding: {len(self.binding_idx):,}  |  "
                 f"kcat: {len(self.kcat_idx):,}  |  "
                 f"km: {len(self.km_idx):,}")
        log.info(f"  Per batch: {self.n_per_group} from each group "
                 f"(= batch_size={batch_size})")
        log.info(f"  Epoch: {self.n_batches} batches "
                 f"(={self.n_batches * batch_size:,} samples, "
                 f"smallest group limits)")

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = (
                random.choices(self.binding_idx, k=self.n_per_group)
                + random.choices(self.kcat_idx, k=self.n_per_group)
                + random.choices(self.km_idx, k=self.n_per_group)
            )
            yield batch

    def __len__(self) -> int:
        return self.n_batches


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

class OxidoreductaseDataset(Dataset):
    """酶数据集：始终从 proteins_token.h5 加载 token-level ESM-2 嵌入。

    已删除 mean-pooled / AA 属性回退路径，所有样本使用 (L, 1280) token 嵌入。
    """

    def __init__(self,
                 unified_metadata_path: str,
                 proteins_token_h5_path: str,
                 ligand_dir: str,
                 split: str = "train",
                 max_samples: Optional[int] = None,
                 ):  # token-only 模式: 无 pooled 回退
        self.ligand_dir = Path(ligand_dir)
        self.split = split
        self.proteins_token_h5_path = proteins_token_h5_path
        self._h5_token = None

        # 直接加载统一元数据
        log.info(f"Loading unified metadata from {unified_metadata_path}")
        self.df = pd.read_parquet(unified_metadata_path)

        # Split 筛选
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        log.info(f"  {split} split: {len(self.df):,} samples")

        # ── 检查 token H5 覆盖率（始终需要 token 嵌入） ──
        with h5py.File(proteins_token_h5_path, "r") as _th5:
            self._token_available = set(_th5.keys())
        n_with_tokens = self.df["protein_seq_hash"].isin(self._token_available).sum()
        if n_with_tokens < len(self.df):
            raise RuntimeError(
                f"Token H5 coverage: {n_with_tokens:,}/{len(self.df):,} "
                f"({100*n_with_tokens/len(self.df):.1f}%) — not 100%. "
                "Run compute_token_embeddings.py first."
            )
        log.info(f"  Token ESM-2: 100% coverage ({n_with_tokens:,}/{len(self.df):,})")

        if max_samples:
            self.df = self.df.head(max_samples)
            log.info(f"  Limited to {max_samples} samples")

        # Log statistics
        n_with_pkd = self.df["pkd_raw"].notna().sum()
        n_with_kcat = self.df["has_kcat"].sum()
        n_with_cofactor = (self.df["cofactors"].notna() & (self.df["cofactors"] != "")).sum()
        log.info(f"  With pKd: {n_with_pkd}, With kcat: {n_with_kcat}, "
                 f"With cofactor: {n_with_cofactor}")

    @property
    def h5_token(self):
        if self._h5_token is None:
            self._h5_token = h5py.File(self.proteins_token_h5_path, "r")
        return self._h5_token

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        seq_hash = row["protein_seq_hash"]

        # ── 蛋白 token 嵌入: 始终从 proteins_token.h5 加载 ──
        token_group = self.h5_token[seq_hash]
        seq_bytes = token_group["sequence"][()]
        if isinstance(seq_bytes, bytes):
            seq = seq_bytes.decode("utf-8")
        else:
            seq = str(seq_bytes)
        protein_tokens = torch.from_numpy(token_group["tokens"][:]).float()  # (L, 1280)

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
                num_nodes=1,
            )
        if not hasattr(ligand_data, 'smiles') or ligand_data.smiles is None:
            ligand_data.smiles = ""

        # ── 辅因子 ──
        cofactor_str = row.get("cofactors", "") or ""

        # ── 标签 ──
        pkd_val = row["pkd_aligned"] if pd.notna(row["pkd_aligned"]) else row["pkd_raw"]
        has_pkd = pd.notna(pkd_val)

        has_kcat = bool(row["has_kcat"])
        log_kcat_label = float(row["log_kcat_median"]) if has_kcat else 0.0
        # Safety: reject kcat=0 artifacts (clamped to 1e-10 → log10=-10)
        if has_kcat and log_kcat_label <= -9.5:
            has_kcat = False
            log_kcat_label = 0.0

        protein_seq_len = len(seq)

        has_km = bool(row.get("has_km", False))
        log_km_label = float(row["log_km"]) if has_km else 0.0

        mtype_encoded = int(row.get("measurement_type_encoded", 3))
        temperature_K = float(row.get("temperature_K", 298.15))

        # Min-Max 归一化目标值
        pkd_normalized = min_max_normalize(
            pkd_val, NORM_PARAMS['pkd_min'], NORM_PARAMS['pkd_max']
        ) if has_pkd else 0.0
        kcat_normalized = min_max_normalize(
            log_kcat_label, NORM_PARAMS['kcat_min'], NORM_PARAMS['kcat_max']
        ) if has_kcat else 0.0

        result = {
            "ligand_data": ligand_data,
            "protein_tokens": protein_tokens,  # (L, 1280)
            "protein_seq_len": protein_seq_len,
            "cofactor_str": cofactor_str,
            "pkd_target": torch.tensor(pkd_normalized, dtype=torch.float32),
            "has_pkd": torch.tensor(has_pkd, dtype=torch.bool),
            "log_kcat_target": torch.tensor(kcat_normalized, dtype=torch.float32),
            "has_kcat": torch.tensor(has_kcat, dtype=torch.bool),
            "pkd_raw": torch.tensor(pkd_val if has_pkd else 0.0, dtype=torch.float32),
            "log_kcat_raw": torch.tensor(log_kcat_label if has_kcat else 0.0, dtype=torch.float32),
            "log_km_target": torch.tensor(log_km_label, dtype=torch.float32),
            "has_km": torch.tensor(has_km, dtype=torch.bool),
            "measurement_type": torch.tensor(mtype_encoded, dtype=torch.long),
            "temperature_K": torch.tensor(temperature_K, dtype=torch.float32),
        }

        return result


def collate_fn(batch: list[dict]) -> dict:
    """
    自定义 collate：处理 PyG 图 + 变长序列 padding

    v4 新增:
      - protein_tokens: padding 到 batch 内最大长度
      - protein_mask: 有效残基掩码
    """
    # ── 配体 PyG batch ──
    ligand_batch = PyGBatch.from_data_list([item["ligand_data"] for item in batch])

    # ── 蛋白 token embedding ──
    token_list = [item["protein_tokens"] for item in batch]
    max_len = max(t.shape[0] for t in token_list)
    token_dim = token_list[0].shape[-1]
    padded_tokens = torch.zeros(len(batch), max_len, token_dim)
    protein_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, tokens in enumerate(token_list):
        seq_len = tokens.shape[0]
        padded_tokens[i, :seq_len] = tokens
        protein_mask[i, :seq_len] = True

    # ── 其他张量 ──
    cofactor_strs = [item["cofactor_str"] for item in batch]
    pkd_target = torch.stack([item["pkd_target"] for item in batch])
    has_pkd = torch.stack([item["has_pkd"] for item in batch])
    log_kcat_target = torch.stack([item["log_kcat_target"] for item in batch])
    has_kcat = torch.stack([item["has_kcat"] for item in batch])
    log_km_target = torch.stack([item["log_km_target"] for item in batch])
    has_km = torch.stack([item["has_km"] for item in batch])
    measurement_type = torch.stack([item["measurement_type"] for item in batch])
    temperature_K = torch.stack([item["temperature_K"] for item in batch])

    return {
        "ligand_data": ligand_batch,
        "protein_tokens": padded_tokens,
        "protein_mask": protein_mask,
        "cofactor_strs": cofactor_strs,
        "pkd_target": pkd_target,
        "pkd_target_mask": has_pkd,
        "log_kcat_target": log_kcat_target,
        "kcat_target_mask": has_kcat,
        "log_km_target": log_km_target,
        "km_target_mask": has_km,
        "measurement_type": measurement_type,
        "temperature_K": temperature_K,
        # 反归一化版本（三头 Joint Loss 用）
        "log_kcat_target_denorm": min_max_denormalize(
            log_kcat_target, NORM_PARAMS['kcat_min'], NORM_PARAMS['kcat_max']
        ),
    }


# ─────────────────────────────────────────────────────────────
# 训练器
# ─────────────────────────────────────────────────────────────

class Trainer:
    """训练循环封装。支持 CPU / GPU、AMP 混合精度、torch.compile。"""

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
                 use_amp: bool = False,
                 use_compile: bool = False,
                 grad_accum_steps: int = 1,
                 warmup_steps: int = 1000,
                 kcat_weight: float = 1.0,
                 km_weight: float = 1.0,
                 joint_km_weight: float = 0.1,
                 binding_weight: float = 1.0,
                 use_pcgrad: bool = False,
                 ):
        self.device = device
        self.use_amp = use_amp and device.startswith("cuda")
        self.grad_accum_steps = grad_accum_steps
        self.kcat_weight = kcat_weight
        self.km_weight = km_weight
        self.joint_km_weight = joint_km_weight
        self.binding_weight = binding_weight
        self.use_pcgrad = use_pcgrad
        if use_pcgrad and self.use_amp:
            log.warning("PCGrad + AMP not supported; disabling PCGrad.")
            self.use_pcgrad = False
        self.pcgrad_shared_prefixes = [
            'ligand_encoder', 'protein_encoder', 'cofactor_encoder',
            'temperature_encoder', 'fusion', 'cofactor_proj',
            'ligand_atom_proj',
        ] if use_pcgrad else None

        # ── torch.compile ──
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
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        self.global_step = 0
        self.warmup_steps = warmup_steps

        if optimizer_fn is None:
            raise ValueError("optimizer_fn is required (e.g. create_threehead_optimizer)")
        self.optimizer = optimizer_fn(model, lr=lr, weight_decay=weight_decay)

        # ── Scheduler ──
        base_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5000, T_mult=2,
        )
        if warmup_steps > 0:
            from torch.optim.lr_scheduler import LinearLR, SequentialLR
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=1e-6,
                end_factor=1.0,
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
            batch_gpu["ligand_data"] = batch_gpu["ligand_data"].to(self.device)

            # 前向（AMP 可选）
            with torch.amp.autocast("cuda") if self.use_amp else contextlib.nullcontext():
                outputs = self._model_forward(batch_gpu)
                total_loss, losses = self.model.compute_loss(
                    outputs, self._make_loss_kwargs(batch_gpu),
                    binding_weight=self.binding_weight,
                    kcat_weight=self.kcat_weight, km_weight=self.km_weight,
                    joint_km_weight=self.joint_km_weight,
                )
                total_loss = total_loss / self.grad_accum_steps

            # 跳过 NaN batch
            if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                log.warning(f"  Skip NaN/Inf batch {batch_idx}")
                self.optimizer.zero_grad()
                continue

            # 反向传播（标准 vs PCGrad）
            if self.use_pcgrad:
                self._pcgrad_step(outputs, losses, batch_gpu)
            elif self.use_amp:
                self.scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            # 梯度累积更新
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

            # 累加损失
            for k, v in losses.items():
                if k == 'weights':
                    continue
                epoch_losses.setdefault(k, 0.0)
                epoch_losses[k] += v.item() if hasattr(v, 'item') else float(v)

            postfix = {
                "loss": f"{total_loss.item() if hasattr(total_loss, 'item') else float(total_loss):.3f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            }
            # 如果启用了不确定性加权，显示当前 σ 值
            sigma_b = losses.get('task_sigma_binding')
            if sigma_b is not None:
                postfix["s_b"] = f"{sigma_b.item():.3f}"
            sigma_k = losses.get('task_sigma_kcat')
            if sigma_k is not None:
                postfix["s_k"] = f"{sigma_k.item():.3f}"
            sigma_m = losses.get('task_sigma_km')
            if sigma_m is not None:
                postfix["s_m"] = f"{sigma_m.item():.3f}"
            pbar.set_postfix(postfix)

        n_batches = len(self.train_loader)
        return {k: v / n_batches for k, v in epoch_losses.items()}

    def _make_loss_kwargs(self, batch_gpu: dict) -> dict:
        return {
            "pkd_target": batch_gpu["pkd_target"],
            "pkd_target_mask": batch_gpu["pkd_target_mask"],
            "log_kcat_target": batch_gpu["log_kcat_target"],
            "kcat_target_mask": batch_gpu["kcat_target_mask"],
            "measurement_type": batch_gpu["measurement_type"],
            "log_km_target": batch_gpu["log_km_target"],
            "km_target_mask": batch_gpu["km_target_mask"],
            "log_kcat_target_denorm": batch_gpu["log_kcat_target_denorm"],
        }

    def _model_forward(self, batch_gpu: dict) -> dict:
        """模型前向（token-only 模式）"""
        return self.model(
            ligand_data=batch_gpu["ligand_data"],
            protein_tokens=batch_gpu["protein_tokens"],  # (B, L, 1280)
            protein_mask=batch_gpu["protein_mask"],      # (B, L)
            cofactor_strs=batch_gpu["cofactor_strs"],
            measurement_types=batch_gpu.get("measurement_type"),
            temperature_K=batch_gpu.get("temperature_K"),
        )

    # ─────────────────────────────────────────────────────────
    # PCGrad: 梯度投影消除共享层冲突
    # ─────────────────────────────────────────────────────────

    def _pcgrad_step(self, outputs: dict, losses: dict, batch_gpu: dict):
        """PCGrad 梯度投影步骤。

        对 binding/kcat/km 三任务的梯度做冲突消除（余弦相似度 < 0 时投影）：
        - 只对共享层参数做（encoder/fusion），预测头参数不做
        - 不支持 AMP（需 unscale 更复杂），如使用 AMP 会 fallback 到标准训练
        """
        # 三个任务的损失名称和对应 key
        task_keys = {
            'binding': 'L_binding',
            'kcat': 'L_kcat',
            'km': 'L_km',
        }

        # 收集共享层参数
        shared_params = []
        shared_names = []
        for name, param in self.model.named_parameters():
            if any(p in name for p in self.pcgrad_shared_prefixes):
                if param.requires_grad:
                    shared_params.append(param)
                    shared_names.append(name)

        if not shared_params:
            return  # 没有共享参数，无事可做

        # 1) 分别 backward 每个任务，收集共享层梯度
        task_grads = []
        task_names = list(task_keys.keys())
        for i, (task_key, loss_name) in enumerate(task_keys.items()):
            loss_val = losses.get(loss_name, None)
            if loss_val is None or (isinstance(loss_val, torch.Tensor) and loss_val.item() == 0.0):
                task_grads.append(None)
                continue

            self.model.zero_grad()
            # 最后一个是最后一次，不需要 retain_graph
            retain = i < len(task_keys) - 1
            loss_val.backward(retain_graph=True)

            # 收集共享层梯度
            grads = []
            for param in shared_params:
                if param.grad is not None:
                    grads.append(param.grad.flatten())
            if grads:
                task_grads.append(torch.cat(grads))
            else:
                task_grads.append(None)

        # 2) PCGrad: 冲突投影（仅对有效梯度对）
        valid_indices = [i for i, g in enumerate(task_grads) if g is not None]
        task_names = list(task_keys.keys())

        for idx_i in valid_indices:
            for idx_j in valid_indices:
                if idx_i >= idx_j:
                    continue
                gi = task_grads[idx_i]
                gj = task_grads[idx_j]
                gi_norm = gi.norm()
                gj_norm = gj.norm()
                if gi_norm < 1e-8 or gj_norm < 1e-8:
                    continue
                cos_sim = torch.dot(gi, gj) / (gi_norm * gj_norm)
                if cos_sim < 0:  # 梯度冲突 → 投影
                    # 将 gj 投影到 gi 的法平面
                    proj = torch.dot(gj, gi) / (torch.dot(gi, gi) + 1e-8)
                    task_grads[idx_j] = gj - proj * gi

        # 3) 合并梯度：平均所有任务的有效梯度
        valid_grads = [g for g in task_grads if g is not None]
        if not valid_grads:
            return
        combined_grad = torch.stack(valid_grads).mean(dim=0)  # (total_dims,)

        # 4) 写回梯度
        idx = 0
        for param in shared_params:
            n = param.grad.numel()
            param.grad.copy_(combined_grad[idx:idx + n].reshape(param.grad.shape))
            idx += n

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        val_losses = {}

        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            batch_gpu = {
                k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            batch_gpu["ligand_data"] = batch_gpu["ligand_data"].to(self.device)

            outputs = self._model_forward(batch_gpu)
            total_loss, losses = self.model.compute_loss(
                outputs, self._make_loss_kwargs(batch_gpu),
                binding_weight=self.binding_weight,
                kcat_weight=self.kcat_weight, km_weight=self.km_weight,
                joint_km_weight=self.joint_km_weight,
            )

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

            self.train_losses.append(train_losses.get("total", 0))
            self.val_losses.append(val_losses.get("total", 0))

            loss_info = (
                f"Epoch {epoch:3d}/{epochs} | "
                f"train: {train_losses['total']:.4f}  "
                f"val: {val_losses['total']:.4f}  "
                f"(best: {self.best_val_loss:.4f})  |  "
                f"L_bind: {train_losses.get('L_binding', 0):.4f}  "
                f"L_kcat: {train_losses.get('L_kcat', 0):.4f}  "
                f"L_km: {train_losses.get('L_km', 0):.4f}  "
                f"L_jnt: {train_losses.get('L_joint', 0):.4f}"
            )
            log.info(loss_info)

            val_total = val_losses["total"]
            if val_total < self.best_val_loss:
                self.best_val_loss = val_total
                self.save_checkpoint("best.ckpt", {
                    "epoch": epoch,
                    "train_losses": self.train_losses,
                    "val_losses": self.val_losses,
                })

            if epoch % save_every == 0:
                self.save_checkpoint(f"epoch_{epoch:04d}.ckpt", {"epoch": epoch})

            self.save_checkpoint("last.ckpt", {"epoch": epoch})

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
            batch_gpu["ligand_data"] = batch_gpu["ligand_data"].to(self.device)
            outputs = self._model_forward(batch_gpu)
            _, losses = self.model.compute_loss(
                outputs, self._make_loss_kwargs(batch_gpu),
                binding_weight=self.binding_weight,
                kcat_weight=self.kcat_weight, km_weight=self.km_weight,
                joint_km_weight=self.joint_km_weight,
            )
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
    parser.add_argument("--proteins-token-h5", default=str(
        PROCESSED_DIR / "proteins_token.h5"))
    parser.add_argument("--ligand-dir", default=str(LIGAND_DIR))
    # 训练
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-samples", type=int, default=None)
    # 模型参数 (Trenzition)
    parser.add_argument("--hidden-dim", type=int, default=512,
                        help="Hidden dimension (256→512 based on ESM-2 embedding distortion analysis)")
    parser.add_argument("--gnn-layers", type=int, default=3)
    # ── v4 新参数 ──
    parser.add_argument("--use-cross-attn", action="store_true",
                        help="启用蛋白残基 × 配体原子交叉注意力融合")
    parser.add_argument("--cross-attn-heads", type=int, default=4,
                        help="交叉注意力头数 (默认=4)")
    parser.add_argument("--heteroscedastic", action="store_true",
                        help="启用异方差损失 (NLL, 预测 mean+variance)")
    # 硬件 / GPU
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true",
                        help="Enable AMP mixed precision (CUDA only)")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile (PyTorch ≥ 2.0)")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    # 检查点
    parser.add_argument("--checkpoint-dir", default=str(CHECKPOINT_DIR))
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--finetune", default=None, help="Fine-tune from a checkpoint")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--kcat-ode-steps", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--binding-weight", type=float, default=1.0)
    parser.add_argument("--kcat-weight", type=float, default=1.0)
    parser.add_argument("--km-weight", type=float, default=1.0)
    parser.add_argument("--no-threehead", action="store_true",
                        help="Use deprecated single-head mode (not recommended)")
    # ── 多任务平衡策略 ──
    parser.add_argument("--stratified-sampling", action="store_true",
                        help="Step 1: 分层采样 — 每个 batch 三任务等量采样 (1:1:1)")
    parser.add_argument("--uncertainty-weighting", action="store_true",
                        help="Step 2: 不确定性加权 — 可学习 σ 自动调整各头 loss 权重")
    parser.add_argument("--pcgrad", action="store_true",
                        help="Step 3: PCGrad — 梯度投影消除共享层冲突")
    args = parser.parse_args()

    log.info(f"Device: {args.device}")

    # ── 日志模式信息 ──
    mode_parts = ["Trenzition"]
    if args.use_cross_attn:
        mode_parts.append("CrossAttn")
    if args.heteroscedastic:
        mode_parts.append("Heteroscedastic(NLL)")
    log.info(f"Model mode: {'+'.join(mode_parts)}")

    # ── 数据集 ──
    train_dataset = OxidoreductaseDataset(
        args.unified_metadata, args.proteins_token_h5, args.ligand_dir,
        split="train", max_samples=args.max_samples,
    )
    val_dataset = OxidoreductaseDataset(
        args.unified_metadata, args.proteins_token_h5, args.ligand_dir,
        split="val",
        max_samples=args.max_samples // 8 if args.max_samples else None,
    )
    test_dataset = OxidoreductaseDataset(
        args.unified_metadata, args.proteins_token_h5, args.ligand_dir,
        split="test",
        max_samples=args.max_samples // 8 if args.max_samples else None,
    )

    use_cuda = args.device.startswith("cuda")

    # ── 分层采样（可选） ──
    if args.stratified_sampling:
        log.info("Using stratified sampling (1:1:1 binding/kcat/Km per batch)")
        strat_sampler = StratifiedBatchSampler(
            train_dataset.df, args.batch_size,
        )
        train_loader = DataLoader(
            train_dataset, batch_sampler=strat_sampler,
            collate_fn=collate_fn, num_workers=args.num_workers,
            pin_memory=use_cuda,
            persistent_workers=use_cuda and args.num_workers > 0,
        )
    else:
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
    use_three_head = not args.no_threehead
    if use_three_head:
        head_desc = "BindingDualHead + KcatHead(BINN+ODE+Eyring) + KmHead"
        if args.use_cross_attn:
            head_desc += " + CrossAttnFusion"
        if args.heteroscedastic:
            head_desc += " (Heteroscedastic NLL)"
        log.info(f"Using: {head_desc}")

        model = Trenzition(
            hidden_dim=args.hidden_dim,
            gnn_layers=args.gnn_layers,
            three_head=True,
            kcat_ode_steps=args.kcat_ode_steps,
            dropout=args.dropout,
            # v4 参数
            use_cross_attn=args.use_cross_attn,
            cross_attn_heads=args.cross_attn_heads,
            heteroscedastic=args.heteroscedastic,
            # v5+ 多任务平衡
            use_uncertainty_weighting=args.uncertainty_weighting,
        )
        log.info(
            f"Three-Head: hidden={args.hidden_dim}, "
            f"kcat_ODE_steps={args.kcat_ode_steps}, "
            f"cross_attn={'Y' if args.use_cross_attn else 'N'}, "
            f"heteroscedastic={'Y' if args.heteroscedastic else 'N'}"
        )
        if args.uncertainty_weighting:
            log.info("  + Uncertainty Weighting (learnable task sigmas)")
        if args.stratified_sampling:
            log.info("  + Stratified Sampling (1:1:1 per batch)")
        if args.pcgrad:
            log.info("  + PCGrad (gradient projection)")
    else:
        log.warning("Deprecated single-head mode.")
        model = Trenzition(
            hidden_dim=args.hidden_dim,
            gnn_layers=args.gnn_layers,
            three_head=False,
        )
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
        optimizer_fn=create_threehead_optimizer,
        use_amp=args.amp,
        use_compile=args.compile,
        grad_accum_steps=args.grad_accum,
        warmup_steps=args.warmup_steps,
        kcat_weight=args.kcat_weight,
        km_weight=args.km_weight,
        binding_weight=args.binding_weight,
        use_pcgrad=args.pcgrad,
    )

    # ── Fine-tune ──
    if args.finetune:
        log.info(f"Fine-tuning from {args.finetune}")
        ckpt = torch.load(args.finetune, map_location=args.device, weights_only=False)
        state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
        model.load_state_dict(state, strict=False)
        for name, param in model.named_parameters():
            if any(x in name for x in ['ligand_encoder', 'protein_encoder', 'cofactor_encoder']):
                param.requires_grad = True
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

    history_path = Path(args.checkpoint_dir) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    log.info(f"Training history saved → {history_path}")


if __name__ == "__main__":
    main()
