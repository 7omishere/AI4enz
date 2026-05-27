"""
plot_alignment_qc.py
====================
QC report for GMM-based w_multiplier analysis (pKd unchanged).

Panels:
  A. Distribution overlay (PDBbind vs BindingDB, raw pKd only)
  B. w_multiplier histogram + stats
  C. w_multiplier vs pKd scatter (per measurement type)
  D. KS/W1 by measurement type
  E. Weight distribution by measurement type
  F. Split pKd distribution
  G. w_multiplier by type × pKd bin heatmap
  H. Summary table + alarms

Usage:
  python plot_alignment_qc.py
  python plot_alignment_qc.py --metadata processed/metadata.parquet --out qc_report.png
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import ks_2samp, wasserstein_distance, gaussian_kde

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

C_REF    = '#333333'
C_BDB    = '#64B5F6'
C_KD     = '#2E7D32'
C_KI     = '#F57C00'
C_IC50   = '#7B1FA2'
MTYPE_COLORS = {'Kd': C_KD, 'Ki': C_KI, 'IC50': C_IC50}
SPLIT_COLORS = {'train': '#4CAF50', 'val': '#FF9800', 'test': '#F44336'}


def plot_qc_report(df: pd.DataFrame, out_path: str):
    pdb   = df[df['source_db'] == 'PDBbind']
    bdb   = df[df['source_db'] == 'BindingDB'].copy()
    pdbbind_pkd = pdb['pkd_aligned'].dropna().values
    bdb_pkd     = bdb['pkd_aligned'].dropna().values  # == pkd_raw, unchanged

    # ── Global metrics ──
    ks_val, _ = ks_2samp(pdbbind_pkd, bdb_pkd)
    w1_val    = wasserstein_distance(pdbbind_pkd, bdb_pkd)
    log.info(f"Raw distribution gap: KS={ks_val:.4f}, W1={w1_val:.4f}")

    # ── Weight analysis ──
    bdb['final_weight'] = bdb['quality_weight'] * bdb['w_multiplier']

    # ── Strata ──
    bdb['pkd_bin'] = pd.cut(bdb['pkd_aligned'], bins=[2, 5, 8, 12],
                            labels=['Low(2-5)', 'Mid(5-8)', 'High(8-12)'])
    # Per measurement type KS
    mtype_metrics = []
    for mt in ['Kd', 'Ki', 'IC50']:
        sub = bdb[bdb['measurement_type'] == mt]
        pdb_sub = pdb[pdb['measurement_type'] == mt]
        if len(sub) > 0 and len(pdb_sub) > 0:
            ks_mt, _ = ks_2samp(pdb_sub['pkd_aligned'].dropna().values,
                                 sub['pkd_aligned'].dropna().values)
            w1_mt = wasserstein_distance(pdb_sub['pkd_aligned'].dropna().values,
                                          sub['pkd_aligned'].dropna().values)
            mtype_metrics.append({'mt': mt, 'n_pdb': len(pdb_sub), 'n_bdb': len(sub),
                                  'ks': ks_mt, 'w1': w1_mt,
                                  'w_mult_mean': sub['w_multiplier'].mean(),
                                  'final_wt_mean': sub['final_weight'].mean()})

    # ── Alarms ──
    alarms = []
    if len(pdbbind_pkd) < 100:
        alarms.append(f'PDBbind < 100 ({len(pdbbind_pkd)})')
    if pdbbind_pkd.std() < 0.3:
        alarms.append(f'PDBbind std too small ({pdbbind_pkd.std():.2f})')
    ic50_pct = (bdb['measurement_type'] == 'IC50').mean() * 100
    if ic50_pct > 60:
        alarms.append(f'IC50={ic50_pct:.0f}% of BindingDB')
    if ks_val > 0.15:
        alarms.append(f'KS={ks_val:.3f} > 0.15 — distributions differ')
    med_fw_ic50 = bdb[bdb['measurement_type'] == 'IC50']['final_weight'].median()
    med_fw_kd   = bdb[bdb['measurement_type'] == 'Kd']['final_weight'].median()
    if med_fw_kd > 0 and med_fw_ic50 / med_fw_kd < 0.5:
        alarms.append(f'IC50/Kd weight ratio={med_fw_ic50/med_fw_kd:.2f}')
    n_low = (bdb['w_multiplier'] < 0.01).sum()
    if n_low > len(bdb) * 0.001:
        alarms.append(f'{n_low} samples w_mult<0.01')
    splits_present = 'split' in df.columns and df['split'].nunique() > 1
    if splits_present:
        test_n = (df['split'] == 'test').sum()
        if test_n == 0:
            alarms.append('Test split empty')

    # ═══════════════════════════════════════════════════════════
    # Figure: 3x3 → use 3x3 with last cell for summary
    # ═══════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(24, 18))
    gs = GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)

    # ── A. Distribution overlay ──
    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(2, 12, 60)
    ax1.hist(pdbbind_pkd, bins=bins, alpha=0.5, color=C_REF, density=True,
             label=f'PDBbind (n={len(pdbbind_pkd):,})')
    ax1.hist(bdb_pkd,     bins=bins, alpha=0.5, color=C_BDB, density=True,
             label=f'BindingDB (n={len(bdb_pkd):,})')
    for data, color, ls in [(pdbbind_pkd, C_REF, '-'), (bdb_pkd, C_BDB, '--')]:
        if len(data) > 3:
            x_kde = np.linspace(2, 12, 300)
            ax1.plot(x_kde, gaussian_kde(data)(x_kde), color=color, ls=ls, lw=1.5, alpha=0.8)
    ax1.set_xlabel('pKd'); ax1.set_ylabel('Density')
    ax1.set_title(f'A. Distribution Overlay\nKS={ks_val:.4f}, W1={w1_val:.4f}',
                  fontweight='bold', loc='left', fontsize=11)
    ax1.legend(frameon=True, fontsize=7)

    # ── B. w_multiplier histogram ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(bdb['w_multiplier'], bins=60, color='steelblue', edgecolor='white', alpha=0.8)
    wm_mean = bdb['w_multiplier'].mean()
    wm_med  = bdb['w_multiplier'].median()
    ax2.axvline(wm_mean, color='red', ls='--', lw=1.5, label=f'mean={wm_mean:.3f}')
    ax2.axvline(wm_med,  color='orange', ls='--', lw=1.5, label=f'median={wm_med:.3f}')
    ax2.set_xlabel('w_multiplier'); ax2.set_ylabel('Count')
    ax2.set_title('B. w_multiplier Distribution', fontweight='bold', loc='left', fontsize=11)
    ax2.legend(frameon=True, fontsize=8)

    # ── C. w_multiplier vs pKd ──
    ax3 = fig.add_subplot(gs[0, 2])
    sample_n = min(50000, len(bdb))
    samp = bdb.sample(sample_n, random_state=42)
    for mt, color in MTYPE_COLORS.items():
        sub = samp[samp['measurement_type'] == mt]
        if len(sub) > 0:
            ax3.scatter(sub['pkd_aligned'], sub['w_multiplier'],
                        s=1, alpha=0.3, color=color, label=mt, rasterized=True)
    ax3.axhline(0.5, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax3.set_xlabel('pKd'); ax3.set_ylabel('w_multiplier')
    ax3.set_title('C. w_multiplier vs pKd', fontweight='bold', loc='left', fontsize=11)
    ax3.legend(frameon=True, fontsize=7, markerscale=5)
    ax3.set_ylim(-0.05, 1.1); ax3.grid(alpha=0.3)

    # ── D. KS/W1 by measurement type ──
    ax4 = fig.add_subplot(gs[1, 0])
    if mtype_metrics:
        labels = [f'{m["mt"]}\n(PDB={m["n_pdb"]:,}, BDB={m["n_bdb"]:,})' for m in mtype_metrics]
        ks_vals = [m['ks'] for m in mtype_metrics]
        w1_vals = [m['w1'] for m in mtype_metrics]
        colors  = [MTYPE_COLORS[m['mt']] for m in mtype_metrics]
        x = np.arange(len(labels))
        w = 0.35
        ax4.bar(x - w/2, ks_vals, w, color=colors, alpha=0.7, label='KS', zorder=2)
        ax4.bar(x + w/2, w1_vals, w, color=colors, alpha=0.3, label='W1', zorder=2, hatch='//')
        ax4.set_xticks(x); ax4.set_xticklabels(labels, fontsize=7)
        ax4.set_ylabel('Statistic')
        ax4.set_title('D. KS/W1 by Measurement Type', fontweight='bold', loc='left', fontsize=11)
        ax4.legend(frameon=True, fontsize=8)
        ax4.grid(axis='y', alpha=0.3)
        for i, (ks, w1) in enumerate(zip(ks_vals, w1_vals)):
            ax4.text(i - w/2, ks + 0.01, f'{ks:.3f}', ha='center', fontsize=6, fontweight='bold')
            ax4.text(i + w/2, w1 + 0.01, f'{w1:.3f}', ha='center', fontsize=6)

    # ── E. Weight distribution by measurement type ──
    ax5 = fig.add_subplot(gs[1, 1])
    for mt, color in MTYPE_COLORS.items():
        sub = bdb[bdb['measurement_type'] == mt]['final_weight']
        if len(sub) > 0:
            ax5.hist(sub, bins=50, alpha=0.4, color=color, density=True,
                     label=f'{mt} (μ={sub.mean():.3f}, med={sub.median():.3f})')
    ax5.set_xlabel('Final weight (quality_weight × w_multiplier)')
    ax5.set_ylabel('Density')
    ax5.set_title('E. Weight Distribution by Type', fontweight='bold', loc='left', fontsize=11)
    ax5.legend(frameon=True, fontsize=7)

    # ── F. Split pKd distribution ──
    ax6 = fig.add_subplot(gs[1, 2])
    if splits_present:
        for sp in ['train', 'val', 'test']:
            sub = df[df['split'] == sp]['pkd_aligned'].dropna()
            if len(sub) > 0:
                ax6.hist(sub, bins=40, alpha=0.5, color=SPLIT_COLORS[sp],
                         density=True, label=f'{sp} (n={len(sub):,}, μ={sub.mean():.2f})')
        ax6.set_xlabel('pKd'); ax6.set_ylabel('Density')
        ax6.set_title('F. Split pKd Distribution', fontweight='bold', loc='left', fontsize=11)
        ax6.legend(frameon=True, fontsize=7)
    else:
        ax6.text(0.5, 0.5, 'Split info not available', ha='center', va='center',
                 transform=ax6.transAxes)

    # ── G. w_multiplier heatmap (type × pKd bin) ──
    ax7 = fig.add_subplot(gs[2, 0])
    heatmap_data = bdb.groupby(['measurement_type', 'pkd_bin'], observed=False)['w_multiplier'].agg(['mean', 'count'])
    mtypes = ['Kd', 'Ki', 'IC50']
    bins_list = ['Low(2-5)', 'Mid(5-8)', 'High(8-12)']
    hm = np.full((3, 3), np.nan)
    hm_count = np.full((3, 3), 0, dtype=int)
    for i, mt in enumerate(mtypes):
        for j, bn in enumerate(bins_list):
            if (mt, bn) in heatmap_data.index:
                hm[i, j] = heatmap_data.loc[(mt, bn), 'mean']
                hm_count[i, j] = heatmap_data.loc[(mt, bn), 'count']
    im = ax7.imshow(hm, aspect='auto', cmap='RdYlGn', vmin=0.5, vmax=1.0)
    ax7.set_xticks(range(3)); ax7.set_xticklabels(bins_list, fontsize=9)
    ax7.set_yticks(range(3)); ax7.set_yticklabels(mtypes, fontsize=9)
    ax7.set_title('G. w_multiplier by Type × pKd Bin', fontweight='bold', loc='left', fontsize=11)
    for i in range(3):
        for j in range(3):
            if not np.isnan(hm[i, j]):
                text = ax7.text(j, i, f'{hm[i, j]:.3f}\nn={hm_count[i, j]:,}',
                                ha='center', va='center', fontsize=7,
                                color='white' if hm[i, j] < 0.7 else 'black')
    plt.colorbar(im, ax=ax7, shrink=0.85)

    # ── H. w_multiplier by pKd bin histogram ──
    ax8 = fig.add_subplot(gs[2, 1])
    bin_colors = {'Low(2-5)': '#FF8A80', 'Mid(5-8)': '#81C784', 'High(8-12)': '#64B5F6'}
    for bn, color in bin_colors.items():
        sub = bdb[bdb['pkd_bin'] == bn]['w_multiplier']
        if len(sub) > 0:
            ax8.hist(sub, bins=40, alpha=0.4, color=color, density=True,
                     label=f'{bn} (μ={sub.mean():.3f}, n={len(sub):,})')
    ax8.set_xlabel('w_multiplier'); ax8.set_ylabel('Density')
    ax8.set_title('H. w_multiplier by pKd Bin', fontweight='bold', loc='left', fontsize=11)
    ax8.legend(frameon=True, fontsize=7)

    # ── I. Summary table ──
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    cell_text = [
        ['Global Gap', '', '', '', ''],
        ['KS', f'{ks_val:.4f}', '', '', ''],
        ['W1', f'{w1_val:.4f}', '', '', ''],
        ['', '', '', '', ''],
        ['w_multiplier', f'μ={wm_mean:.3f}', f'med={wm_med:.3f}',
         f'<0.1:{(bdb["w_multiplier"]<0.1).sum():,}',
         f'<0.01:{(bdb["w_multiplier"]<0.01).sum():,}'],
        ['final_weight', f'μ={bdb["final_weight"].mean():.3f}',
         f'med={bdb["final_weight"].median():.3f}', '', ''],
        ['', '', '', '', ''],
        ['By Type', 'N', 'base_wt', 'w_mult', 'final_wt'],
    ]
    for mt in ['Kd', 'Ki', 'IC50']:
        sub = bdb[bdb['measurement_type'] == mt]
        if len(sub) > 0:
            cell_text.append([mt, f'{len(sub):,}',
                              f'{sub["quality_weight"].mean():.3f}',
                              f'{sub["w_multiplier"].mean():.3f}',
                              f'{sub["final_weight"].mean():.3f}'])

    if splits_present:
        cell_text.append(['', '', '', '', ''])
        cell_text.append(['Split', 'N', 'μ(pKd)', 'σ(pKd)', 'μ(wt)'])
        for sp in ['train', 'val', 'test']:
            sub = df[df['split'] == sp]
            if len(sub) > 0:
                cell_text.append([sp, f'{len(sub):,}',
                                  f'{sub["pkd_aligned"].mean():.2f}',
                                  f'{sub["pkd_aligned"].std():.2f}',
                                  f'{(sub["quality_weight"]*sub["w_multiplier"]).mean():.3f}'])

    col_labels = ['Metric', 'Value', 'Median/Alt', 'Penalty', 'Notes']
    table = ax9.table(cellText=cell_text, colLabels=col_labels,
                      cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.05, 1.6)
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#404040')
        table[0, j].set_text_props(color='white', fontweight='bold')
    ax9.set_title('I. Summary', fontweight='bold', loc='left', fontsize=11, y=1.02)

    # ── Alarms ──
    alarm_text = 'ALARMS: ' + ('; '.join(alarms) if alarms else 'None')
    alarm_color = 'red' if alarms else 'green'
    fig.text(0.5, 0.005, alarm_text, ha='center', fontsize=10, fontweight='bold',
             color=alarm_color,
             bbox=dict(boxstyle='round',
                       facecolor='#FFF3E0' if alarms else '#E8F5E9', alpha=0.8))

    # ── Suptitle ──
    fig.suptitle(f'GMM Weight Analysis Report — PDBbind (ref, n={len(pdbbind_pkd):,}) + BindingDB (n={len(bdb_pkd):,})  |  pKd UNCHANGED',
                 fontsize=15, fontweight='bold', y=1.01)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    log.info(f"QC report saved → {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GMM Weight Analysis QC report')
    parser.add_argument('--metadata', default=None)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    default_meta = script_dir.parent / 'processed' / 'metadata.parquet'
    default_out  = script_dir.parent / 'processed' / 'qc_report.png'

    meta_path = args.metadata or str(default_meta)
    out_path  = args.out      or str(default_out)

    df = pd.read_parquet(meta_path)
    log.info(f"Loaded {len(df):,} records from {meta_path}")
    log.info(f"  PDBbind:   {(df['source_db'] == 'PDBbind').sum():,}")
    log.info(f"  BindingDB: {(df['source_db'] == 'BindingDB').sum():,}")

    plot_qc_report(df, out_path)
