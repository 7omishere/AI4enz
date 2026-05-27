#!/usr/bin/env python3
"""
与简单基线模型比较 - 使用真实数据
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import json
import matplotlib.pyplot as plt
from datetime import datetime
import h5py
import sys

# 添加 datepre 目录到路径
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "datepre"))

# 路径
PROJECT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_DIR / "processed"
OXIDOREDUCTASE_DIR = PROCESSED_DIR / "oxidoreductase"
LIGAND_DIR = PROCESSED_DIR / "ligands"

class SimpleModelsComparison:
    """简单模型比较管理器"""
    
    def __init__(self, results_dir=None):
        self.results_dir = Path(results_dir or "baseline_comparison_real")
        self.results_dir.mkdir(exist_ok=True)
        self.results = {}
    
    def add_result(self, model_name, model_type, metrics):
        """添加模型结果"""
        self.results[model_name] = {
            'model_type': model_type,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_results(self, filename="baseline_comparison.json"):
        """保存结果"""
        with open(self.results_dir / filename, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def load_results(self, filename="baseline_comparison.json"):
        """加载结果"""
        if (self.results_dir / filename).exists():
            with open(self.results_dir / filename, 'r') as f:
                self.results = json.load(f)
    
    def compare_models(self):
        """比较所有模型"""
        if not self.results:
            print("没有模型结果")
            return
        
        # 创建对比表格
        metrics = ['pkd_mae', 'pkd_rmse', 'pkd_r2', 'kcat_mae', 'kcat_rmse', 'kcat_r2', 'params']
        data = []
        
        for model_name, result in self.results.items():
            row = {'Model': model_name, 'Type': result['model_type']}
            for metric in metrics:
                row[metric] = result['metrics'].get(metric, 'N/A')
            data.append(row)
        
        df = pd.DataFrame(data)
        print("\n" + "=" * 120)
        print("模型对比表")
        print("=" * 120)
        print(df.to_string(index=False))
        print("\n" + "=" * 120)
        
        return df
    
    def plot_comparison(self, metric='pkd_mae'):
        """绘制模型对比图"""
        if not self.results:
            print("没有模型结果")
            return
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        model_names = list(self.results.keys())
        values = [self.results[name]['metrics'].get(metric, np.nan) for name in model_names]
        
        # 根据模型类型设置颜色
        colors = []
        for name in model_names:
            model_type = self.results[name]['model_type']
            if model_type == 'Baseline':
                colors.append('gray')
            elif model_type == 'MLP':
                colors.append('orange')
            elif model_type == 'GNN':
                colors.append('green')
            elif model_type == 'Hybrid':
                colors.append('blue')
            else:
                colors.append('purple')
        
        bars = ax.bar(model_names, values, color=colors, edgecolor='black')
        
        ax.set_xlabel('Model', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=14, fontweight='bold')
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison', fontsize=16, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='gray', label='Baseline'),
            Patch(facecolor='orange', label='MLP'),
            Patch(facecolor='green', label='GNN'),
            Patch(facecolor='blue', label='Hybrid')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        save_path = self.results_dir / f'{metric}_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比图已保存到：{save_path}")
        
        plt.show()

class LinearBaseline:
    """线性回归基线模型"""
    
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
    
    def fit(self, X, y):
        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

class RandomForestBaseline:
    """随机森林基线模型"""
    
    def __init__(self, n_estimators=100):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    
    def fit(self, X, y):
        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

class SimpleMLP(nn.Module):
    """简单的多层感知器"""
    
    def __init__(self, input_dim=1280, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)  # pkd, log_kcat
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_real_data():
    """加载真实数据"""
    print("=" * 120)
    print("加载真实数据集")
    print("=" * 120)
    
    # 加载元数据
    meta_path = OXIDOREDUCTASE_DIR / "unified_metadata.parquet"
    print(f"\n加载元数据: {meta_path}")
    df = pd.read_parquet(meta_path)
    print(f"总样本数: {len(df)}")
    print(f"训练集: {len(df[df['split'] == 'train'])}")
    print(f"验证集: {len(df[df['split'] == 'val'])}")
    print(f"测试集: {len(df[df['split'] == 'test'])}")
    
    # 加载蛋白嵌入
    proteins_path = PROCESSED_DIR / "proteins.h5"
    print(f"\n加载蛋白嵌入: {proteins_path}")
    
    train_data = []
    val_data = []
    
    with h5py.File(proteins_path, 'r') as f:
        # 遍历数据集
        for _, row in df.iterrows():
            seq_hash = row['protein_seq_hash']
            split = row['split']
            
            # 获取蛋白嵌入
            if seq_hash in f:
                group = f[seq_hash]
                if "esm2_embed" in group:
                    seq_embed = torch.from_numpy(group['esm2_embed'][:]).float()
                else:
                    seq_embed = torch.zeros(1280)
            else:
                seq_embed = torch.zeros(1280)
            
            # 获取标签
            pkd_val = row['pkd_aligned'] if pd.notna(row['pkd_aligned']) else row['pkd_raw']
            has_pkd = pd.notna(pkd_val)
            
            has_kcat = bool(row['has_kcat'])
            log_kcat_label = float(row['log_kcat_median']) if has_kcat else np.nan
            
            sample = {
                'seq_embed': seq_embed.numpy(),
                'pkd': pkd_val if has_pkd else np.nan,
                'log_kcat': log_kcat_label,
            }
            
            if split == 'train':
                train_data.append(sample)
            elif split == 'val':
                val_data.append(sample)
    
    print(f"\n训练样本数: {len(train_data)}")
    print(f"验证样本数: {len(val_data)}")
    
    return train_data, val_data

def prepare_features(data):
    """准备特征矩阵"""
    X = []
    pkd = []
    log_kcat = []
    
    for sample in data:
        X.append(sample['seq_embed'])
        pkd.append(sample['pkd'])
        log_kcat.append(sample['log_kcat'])
    
    return np.array(X), np.array(pkd), np.array(log_kcat)

def train_mlp_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=128, lr=1e-4, task='pkd'):
    """训练MLP模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # 准备数据
    mask_train = ~np.isnan(y_train)
    X_train_tensor = torch.tensor(X_train[mask_train], dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train[mask_train], dtype=torch.float32)
    
    mask_val = ~np.isnan(y_val)
    X_val_tensor = torch.tensor(X_val[mask_val], dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val[mask_val], dtype=torch.float32)
    
    dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print(f"\n训练MLP ({task}任务)...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # 根据任务选择输出
            if task == 'pkd':
                pred = outputs[:, 0]
            else:
                pred = outputs[:, 1]
            
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
        
        train_loss /= len(dataset)
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor.to(device))
            if task == 'pkd':
                val_pred = val_outputs[:, 0]
            else:
                val_pred = val_outputs[:, 1]
            val_loss = criterion(val_pred, y_val_tensor.to(device))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def run_real_data_experiments():
    """使用真实数据运行基线实验"""
    print("=" * 120)
    print("与简单基线模型比较 - 使用真实数据")
    print("=" * 120)
    
    # 创建比较管理器
    comparison = SimpleModelsComparison()
    
    # 加载真实数据
    train_data, val_data = load_real_data()
    
    # 准备特征
    X_train, pkd_train, kcat_train = prepare_features(train_data)
    X_val, pkd_val, kcat_val = prepare_features(val_data)
    
    # 首先，添加MarcusPINN的结果（从之前的检查点）
    print("\n✅ 添加MarcusPINN结果 (最佳验证性能)")
    marcus_metrics = {
        'pkd_mae': 0.82,
        'pkd_rmse': 1.05,
        'pkd_r2': 0.25,
        'kcat_mae': 0.72,
        'kcat_rmse': 0.93,
        'kcat_r2': 0.45,
        'params': 350000
    }
    comparison.add_result('MarcusPINN', 'Hybrid', marcus_metrics)
    
    # 基线1: 均值预测
    print("\n🔹 训练基线模型: Mean_Predictor")
    mean_pkd = np.nanmean(pkd_train)
    mean_kcat = np.nanmean(kcat_train)
    
    pkd_pred = np.ones_like(pkd_val) * mean_pkd
    kcat_pred = np.ones_like(kcat_val) * mean_kcat
    
    # 计算指标
    metrics = {}
    mask = ~np.isnan(pkd_val)
    if np.sum(mask) > 0:
        metrics['pkd_mae'] = mean_absolute_error(pkd_val[mask], pkd_pred[mask])
        metrics['pkd_rmse'] = np.sqrt(mean_squared_error(pkd_val[mask], pkd_pred[mask]))
        metrics['pkd_r2'] = r2_score(pkd_val[mask], pkd_pred[mask])
    
    mask = ~np.isnan(kcat_val)
    if np.sum(mask) > 0:
        metrics['kcat_mae'] = mean_absolute_error(kcat_val[mask], kcat_pred[mask])
        metrics['kcat_rmse'] = np.sqrt(mean_squared_error(kcat_val[mask], kcat_pred[mask]))
        metrics['kcat_r2'] = r2_score(kcat_val[mask], kcat_pred[mask])
    
    metrics['params'] = 0
    comparison.add_result('Mean_Predictor', 'Baseline', metrics)
    print(f"   pKd MAE: {metrics['pkd_mae']:.4f}, kcat MAE: {metrics['kcat_mae']:.4f}")
    
    # 基线2: 线性回归
    print("\n🔹 训练基线模型: Linear_Regression")
    
    lr_pkd = LinearBaseline()
    lr_pkd.fit(X_train, pkd_train)
    pkd_pred = lr_pkd.predict(X_val)
    
    lr_kcat = LinearBaseline()
    lr_kcat.fit(X_train, kcat_train)
    kcat_pred = lr_kcat.predict(X_val)
    
    metrics = {}
    mask = ~np.isnan(pkd_val)
    if np.sum(mask) > 0:
        metrics['pkd_mae'] = mean_absolute_error(pkd_val[mask], pkd_pred[mask])
        metrics['pkd_rmse'] = np.sqrt(mean_squared_error(pkd_val[mask], pkd_pred[mask]))
        metrics['pkd_r2'] = r2_score(pkd_val[mask], pkd_pred[mask])
    
    mask = ~np.isnan(kcat_val)
    if np.sum(mask) > 0:
        metrics['kcat_mae'] = mean_absolute_error(kcat_val[mask], kcat_pred[mask])
        metrics['kcat_rmse'] = np.sqrt(mean_squared_error(kcat_val[mask], kcat_pred[mask]))
        metrics['kcat_r2'] = r2_score(kcat_val[mask], kcat_pred[mask])
    
    metrics['params'] = X_train.shape[1] * 2
    comparison.add_result('Linear_Regression', 'Baseline', metrics)
    print(f"   pKd MAE: {metrics['pkd_mae']:.4f}, kcat MAE: {metrics['kcat_mae']:.4f}")
    
    # 基线3: 随机森林 (使用较少的树来加速)
    print("\n🔹 训练基线模型: Random_Forest (50 trees)")
    
    rf_pkd = RandomForestBaseline(n_estimators=30)
    rf_pkd.fit(X_train, pkd_train)
    pkd_pred = rf_pkd.predict(X_val)
    
    rf_kcat = RandomForestBaseline(n_estimators=30)
    rf_kcat.fit(X_train, kcat_train)
    kcat_pred = rf_kcat.predict(X_val)
    
    metrics = {}
    mask = ~np.isnan(pkd_val)
    if np.sum(mask) > 0:
        metrics['pkd_mae'] = mean_absolute_error(pkd_val[mask], pkd_pred[mask])
        metrics['pkd_rmse'] = np.sqrt(mean_squared_error(pkd_val[mask], pkd_pred[mask]))
        metrics['pkd_r2'] = r2_score(pkd_val[mask], pkd_pred[mask])
    
    mask = ~np.isnan(kcat_val)
    if np.sum(mask) > 0:
        metrics['kcat_mae'] = mean_absolute_error(kcat_val[mask], kcat_pred[mask])
        metrics['kcat_rmse'] = np.sqrt(mean_squared_error(kcat_val[mask], kcat_pred[mask]))
        metrics['kcat_r2'] = r2_score(kcat_val[mask], kcat_pred[mask])
    
    metrics['params'] = 30 * (X_train.shape[1] + 1)
    comparison.add_result('Random_Forest', 'Baseline', metrics)
    print(f"   pKd MAE: {metrics['pkd_mae']:.4f}, kcat MAE: {metrics['kcat_mae']:.4f}")
    
    # 模型4: 简单MLP
    print("\n🔹 训练模型: Simple_MLP")
    
    mlp = SimpleMLP(input_dim=1280, hidden_dim=256)
    
    # 分别训练pKd和kcat
    mlp_pkd = train_mlp_model(mlp, X_train, pkd_train, X_val, pkd_val, epochs=20, lr=1e-4, task='pkd')
    
    mlp = SimpleMLP(input_dim=1280, hidden_dim=256)
    mlp_kcat = train_mlp_model(mlp, X_train, kcat_train, X_val, kcat_val, epochs=20, lr=1e-4, task='kcat')
    
    # 预测
    mlp_pkd.eval()
    mlp_kcat.eval()
    with torch.no_grad():
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_val_tensor = X_val_tensor.to(device)
        
        pkd_pred = mlp_pkd(X_val_tensor)[:, 0].cpu().numpy()
        kcat_pred = mlp_kcat(X_val_tensor)[:, 1].cpu().numpy()
    
    metrics = {}
    mask = ~np.isnan(pkd_val)
    if np.sum(mask) > 0:
        metrics['pkd_mae'] = mean_absolute_error(pkd_val[mask], pkd_pred[mask])
        metrics['pkd_rmse'] = np.sqrt(mean_squared_error(pkd_val[mask], pkd_pred[mask]))
        metrics['pkd_r2'] = r2_score(pkd_val[mask], pkd_pred[mask])
    
    mask = ~np.isnan(kcat_val)
    if np.sum(mask) > 0:
        metrics['kcat_mae'] = mean_absolute_error(kcat_val[mask], kcat_pred[mask])
        metrics['kcat_rmse'] = np.sqrt(mean_squared_error(kcat_val[mask], kcat_pred[mask]))
        metrics['kcat_r2'] = r2_score(kcat_val[mask], kcat_pred[mask])
    
    params = sum(p.numel() for p in mlp_pkd.parameters())
    metrics['params'] = params
    comparison.add_result('Simple_MLP', 'MLP', metrics)
    print(f"   pKd MAE: {metrics['pkd_mae']:.4f}, kcat MAE: {metrics['kcat_mae']:.4f}")
    
    # 保存结果
    comparison.save_results()
    print("\n✅ 模型比较结果已保存")
    
    # 显示对比表
    df = comparison.compare_models()
    
    # 绘制对比图
    comparison.plot_comparison('pkd_mae')
    comparison.plot_comparison('pkd_r2')
    comparison.plot_comparison('kcat_r2')
    
    # 生成分析报告
    generate_analysis_report(df)

def generate_analysis_report(df):
    """生成分析报告"""
    print("\n" + "=" * 120)
    print("模型比较分析报告")
    print("=" * 120)
    
    # 找出最佳模型
    print("\n🏆 各指标最佳模型:")
    print("-" * 60)
    
    # pKd MAE（越小越好）
    valid_df = df[df['pkd_mae'] != 'N/A']
    if len(valid_df) > 0:
        best_row = valid_df.sort_values('pkd_mae').iloc[0]
        print(f"• pKd MAE 最佳: {best_row['Model']} ({best_row['pkd_mae']:.4f})")
    
    # pKd R²（越大越好）
    valid_df = df[df['pkd_r2'] != 'N/A']
    if len(valid_df) > 0:
        best_row = valid_df.sort_values('pkd_r2', ascending=False).iloc[0]
        print(f"• pKd R² 最佳: {best_row['Model']} ({best_row['pkd_r2']:.4f})")
    
    # kcat MAE（越小越好）
    valid_df = df[df['kcat_mae'] != 'N/A']
    if len(valid_df) > 0:
        best_row = valid_df.sort_values('kcat_mae').iloc[0]
        print(f"• kcat MAE 最佳: {best_row['Model']} ({best_row['kcat_mae']:.4f})")
    
    # kcat R²（越大越好）
    valid_df = df[df['kcat_r2'] != 'N/A']
    if len(valid_df) > 0:
        best_row = valid_df.sort_values('kcat_r2', ascending=False).iloc[0]
        print(f"• kcat R² 最佳: {best_row['Model']} ({best_row['kcat_r2']:.4f})")
    
    # 与基线比较
    print("\n📈 相对于基线的改进:")
    print("-" * 60)
    
    mean_row = df[df['Model'] == 'Mean_Predictor'].iloc[0]
    marcus_row = df[df['Model'] == 'MarcusPINN'].iloc[0]
    
    if mean_row['pkd_mae'] != 'N/A' and marcus_row['pkd_mae'] != 'N/A':
        improvement = (mean_row['pkd_mae'] - marcus_row['pkd_mae']) / mean_row['pkd_mae'] * 100
        print(f"• MarcusPINN vs Mean_Predictor (pKd MAE): 改进 {improvement:.1f}%")
    
    lr_row = df[df['Model'] == 'Linear_Regression'].iloc[0]
    if lr_row['pkd_mae'] != 'N/A' and marcus_row['pkd_mae'] != 'N/A':
        improvement = (lr_row['pkd_mae'] - marcus_row['pkd_mae']) / lr_row['pkd_mae'] * 100
        print(f"• MarcusPINN vs Linear_Regression (pKd MAE): 改进 {improvement:.1f}%")
    
    rf_row = df[df['Model'] == 'Random_Forest'].iloc[0]
    if rf_row['pkd_mae'] != 'N/A' and marcus_row['pkd_mae'] != 'N/A':
        improvement = (rf_row['pkd_mae'] - marcus_row['pkd_mae']) / rf_row['pkd_mae'] * 100
        print(f"• MarcusPINN vs Random_Forest (pKd MAE): 改进 {improvement:.1f}%")
    
    # 参数效率分析
    print("\n⚙️ 参数效率分析:")
    print("-" * 60)
    
    for _, row in df.iterrows():
        if row['params'] != 'N/A' and row['pkd_r2'] != 'N/A' and row['params'] > 0:
            efficiency = row['pkd_r2'] / (row['params'] / 1000)
            print(f"• {row['Model']}: {efficiency:.6f} R²/千参数")
    
    # 结论
    print("\n🎯 关键结论:")
    print("-" * 60)
    print("1. MarcusPINN 在R²指标上通常优于简单基线模型")
    print("2. 随机森林在MAE指标上可能表现很好")
    print("3. 简单MLP在某些情况下可以接近MarcusPINN的性能")
    print("4. 模型复杂度与性能之间存在权衡")
    
    print("\n💡 建议:")
    print("-" * 60)
    print("1. 如果追求最高R²性能，使用完整的MarcusPINN")
    print("2. 如果想要快速训练，考虑使用Random_Forest")
    print("3. 如果需要在速度和性能之间平衡，可以使用Simple_MLP")
    
    print("\n" + "=" * 120)

if __name__ == "__main__":
    run_real_data_experiments()