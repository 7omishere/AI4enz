"""
03_align_distributions.py
========================
GMM-based distribution analysis: fit GMMs to both databases and compute
per-sample w_multiplier from the mapping derivative. Labels (pKd) are
NOT modified — only the weight signal is produced.

Design:
  - PDBbind pKd values are the thermodynamic reference.
  - BindingDB pKd values are NEVER altered; pkd_aligned == pkd_raw.
  - w_multiplier = min(r, 1/r) where r = dmapping/dx = pdf_src/pfd_tgt.
  - High w_multiplier → distributions naturally overlap → trustworthy.
  - Low w_multiplier  → alignment would stretch/compress heavily → less reliable.
  - The w_multiplier signal feeds into final_loss = base_weight × w_multiplier,
    and can inform downstream physical constraints.

Usage:
  python 03_align_distributions.py \\
      --pdbbind  processed/pdbbind_records.pkl \\
      --bindingdb processed/bindingdb_records.pkl \\
      --out-bindingdb processed/bindingdb_aligned.pkl \\
      --out-mapping   processed/alignment_mapping.joblib \\
      --plot          processed/gmm_analysis.png
"""

import os
import pickle
import argparse
import logging
from typing import Optional

import numpy as np
import joblib
from scipy.interpolate import interp1d
from scipy.stats import ks_2samp
from sklearn.mixture import GaussianMixture

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 1. GMM fitting with BIC-based component selection
# ─────────────────────────────────────────────────────────────

def fit_gmm_bic(values: np.ndarray,
                max_components: int = 10,
                noise_sigma: float = 0.12,
                random_state: int = 42) -> GaussianMixture:
    """
    Fit a GMM to `values` with BIC-selected number of components.

    1. Add small Gaussian noise to smooth quantization artifacts.
    2. Fit GMMs with 1..max_components components.
    3. Select the model with the lowest BIC.
    """
    rng = np.random.default_rng(random_state)
    smoothed = values + rng.normal(0, noise_sigma, size=len(values))
    smoothed = smoothed.reshape(-1, 1)

    n_samples = len(values)
    max_components = min(max_components, n_samples)

    best_bic   = np.inf
    best_model = None

    for n in range(1, max_components + 1):
        gm = GaussianMixture(n_components=n,
                             covariance_type='full',
                             random_state=random_state,
                             max_iter=200,
                             n_init=3)
        gm.fit(smoothed)
        bic = gm.bic(smoothed)
        if bic < best_bic:
            best_bic   = bic
            best_model = gm

    log.info(f"  Best GMM: n_components={best_model.n_components}, BIC={best_bic:.1f}")
    return best_model


# ─────────────────────────────────────────────────────────────
# 2. GMM probability functions
# ─────────────────────────────────────────────────────────────

def gmm_cdf(gmm: GaussianMixture, x: np.ndarray) -> np.ndarray:
    """CDF of a fitted GMM at points x."""
    from scipy.stats import norm

    weights = gmm.weights_
    means   = gmm.means_.flatten()
    stds    = np.sqrt(gmm.covariances_.flatten())

    cdf = np.zeros_like(x, dtype=float)
    for w, mu, sigma in zip(weights, means, stds):
        cdf += w * norm.cdf(x, loc=mu, scale=sigma)
    return cdf


def gmm_pdf(gmm: GaussianMixture, x: np.ndarray) -> np.ndarray:
    """PDF of a fitted GMM at points x."""
    from scipy.stats import norm

    weights = gmm.weights_
    means   = gmm.means_.flatten()
    stds    = np.sqrt(gmm.covariances_.flatten())

    pdf = np.zeros_like(x, dtype=float)
    for w, mu, sigma in zip(weights, means, stds):
        pdf += w * norm.pdf(x, loc=mu, scale=sigma)
    return pdf


# ─────────────────────────────────────────────────────────────
# 3. Quantile mapping (for diagnostics and w_multiplier only)
# ─────────────────────────────────────────────────────────────

def build_quantile_mapping(source_gmm: GaussianMixture,
                            target_gmm: GaussianMixture,
                            n_grid: int = 2000,
                            pkd_range: tuple = (2.0, 15.0)
                            ) -> interp1d:
    """
    Build quantile mapping: source pKd → target pKd.
    Used only for derivative computation, NOT applied to labels.
    """
    grid = np.linspace(pkd_range[0], pkd_range[1], n_grid)

    source_cdf = gmm_cdf(source_gmm, grid)
    target_cdf = gmm_cdf(target_gmm, grid)

    target_cdf_u, idx = np.unique(target_cdf, return_index=True)
    target_grid_u     = grid[idx]
    inv_target_cdf    = interp1d(target_cdf_u, target_grid_u,
                                  kind='linear', bounds_error=False,
                                  fill_value=(target_grid_u[0], target_grid_u[-1]))

    source_cdf_u, idx2 = np.unique(source_cdf, return_index=True)
    source_grid_u      = grid[idx2]
    source_to_quantile = interp1d(source_grid_u, source_cdf_u,
                                   kind='linear', bounds_error=False,
                                   fill_value=(source_cdf_u[0], source_cdf_u[-1]))

    def mapping(x: np.ndarray) -> np.ndarray:
        q = source_to_quantile(x)
        return inv_target_cdf(q)

    mapping.source_gmm = source_gmm
    mapping.target_gmm = target_gmm
    mapping.grid       = grid
    mapping.source_cdf = source_cdf
    mapping.target_cdf = target_cdf
    return mapping


def compute_w_multiplier(mapping, pkd_raw: np.ndarray) -> np.ndarray:
    """
    Compute w_multiplier from mapping derivative without applying the mapping.

    dmapping/dx = pdf_src(x) / pdf_tgt(mapping(x))

    w_multiplier = min(r, 1/r)  where r = dmapping/dx.

    Range [0, 1]. Samples that would require heavy stretching/compression
    to align get lower multipliers. Labels are never altered.
    """
    x = np.asarray(pkd_raw, dtype=float)

    pdf_src = gmm_pdf(mapping.source_gmm, x)
    mapped  = mapping(x)
    pdf_tgt = gmm_pdf(mapping.target_gmm, mapped)

    eps = 1e-10
    deriv = pdf_src / np.maximum(pdf_tgt, eps)

    w = np.minimum(deriv, 1.0 / np.maximum(deriv, eps))
    return np.clip(w, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────
# 4. Diagnostics
# ─────────────────────────────────────────────────────────────

def compute_diagnostics(pdbbind_pkd: np.ndarray,
                         bindingdb_pkd: np.ndarray) -> dict:
    """Compute KS and Wasserstein-1 between the two raw distributions."""
    from scipy.stats import wasserstein_distance

    ks_val, _ = ks_2samp(pdbbind_pkd, bindingdb_pkd)
    w1_val    = wasserstein_distance(pdbbind_pkd, bindingdb_pkd)

    return {'ks': ks_val, 'w1': w1_val}


def plot_analysis(pdbbind_pkd: np.ndarray,
                   bindingdb_pkd: np.ndarray,
                   w_mult: np.ndarray,
                   diag: dict,
                   out_path: str) -> None:
    """Save a 3-panel diagnostic figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    bins = np.linspace(2, 12, 50)

    # Panel 1: distribution overlay
    axes[0].hist(pdbbind_pkd,   bins=bins, alpha=0.6, label='PDBbind',  density=True)
    axes[0].hist(bindingdb_pkd, bins=bins, alpha=0.6, label='BindingDB', density=True)
    axes[0].set_title(f'pKd Distributions\nKS={diag["ks"]:.4f}, W1={diag["w1"]:.4f}')
    axes[0].set_xlabel('pKd')
    axes[0].set_ylabel('Density')
    axes[0].legend()

    # Panel 2: w_multiplier histogram
    axes[1].hist(w_mult, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    axes[1].axvline(w_mult.mean(), color='red', ls='--', lw=1, label=f'mean={w_mult.mean():.3f}')
    axes[1].axvline(np.median(w_mult), color='orange', ls='--', lw=1, label=f'median={np.median(w_mult):.3f}')
    axes[1].set_title('w_multiplier Distribution')
    axes[1].set_xlabel('w_multiplier')
    axes[1].set_ylabel('Count')
    axes[1].legend()

    # Panel 3: w_multiplier vs pKd
    sample_n = min(20000, len(bindingdb_pkd))
    idx = np.random.default_rng(42).choice(len(bindingdb_pkd), sample_n, replace=False)
    axes[2].scatter(bindingdb_pkd[idx], w_mult[idx], s=1, alpha=0.3, c='steelblue', rasterized=True)
    axes[2].set_xlabel('BindingDB pKd')
    axes[2].set_ylabel('w_multiplier')
    axes[2].set_title('w_multiplier vs pKd')
    axes[2].set_ylim(-0.05, 1.1)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Analysis plot saved → {out_path}")


# ─────────────────────────────────────────────────────────────
# 5. Main: fit GMMs, compute w_multiplier, do NOT modify pKd
# ─────────────────────────────────────────────────────────────

def align_distributions(pdbbind_records: list,
                         bindingdb_records: list,
                         out_bindingdb_path: str,
                         out_mapping_path: str,
                         plot_path: Optional[str] = None,
                         max_components: int = 10,
                         noise_sigma: float = 0.12) -> list:
    """
    Fit GMMs to both databases and compute w_multiplier for BindingDB records.

    IMPORTANT: pkd_aligned = pkd_raw (labels are NEVER modified).
    Only the w_multiplier weight signal is produced.

    Returns list of BindingDB records with 'w_multiplier' field added.
    """
    # Extract pKd arrays
    pdbbind_pkd  = np.array([r['pkd_raw'] for r in pdbbind_records
                              if r['pkd_raw'] is not None and
                              2.0 <= r['pkd_raw'] <= 12.0])
    bindingdb_pkd = np.array([r['pkd_raw'] for r in bindingdb_records
                               if r['pkd_raw'] is not None and
                               2.0 <= r['pkd_raw'] <= 12.0])

    log.info(f"PDBbind:  {len(pdbbind_pkd):,} values, "
             f"mean={pdbbind_pkd.mean():.2f}, std={pdbbind_pkd.std():.2f}")
    log.info(f"BindingDB: {len(bindingdb_pkd):,} values, "
             f"mean={bindingdb_pkd.mean():.2f}, std={bindingdb_pkd.std():.2f}")

    # Diagnostic: raw distribution gap
    diag = compute_diagnostics(pdbbind_pkd, bindingdb_pkd)
    log.info(f"Raw distribution gap: KS={diag['ks']:.4f}, W1={diag['w1']:.4f}")

    # Fit GMMs
    log.info("Fitting GMM to PDBbind (reference) ...")
    target_gmm = fit_gmm_bic(pdbbind_pkd,  max_components=max_components,
                              noise_sigma=noise_sigma)

    log.info("Fitting GMM to BindingDB (source) ...")
    source_gmm = fit_gmm_bic(bindingdb_pkd, max_components=max_components,
                              noise_sigma=noise_sigma)

    # Build quantile mapping (for derivative computation only)
    log.info("Building quantile mapping (for w_multiplier computation) ...")
    mapping = build_quantile_mapping(source_gmm, target_gmm)

    # Compute w_multiplier for BindingDB records (no pKd modification)
    log.info("Computing w_multiplier for BindingDB records ...")
    raw_values = np.array([r['pkd_raw'] if r['pkd_raw'] is not None else np.nan
                           for r in bindingdb_records])
    valid_mask = ~np.isnan(raw_values)

    w_mult = np.ones(len(bindingdb_records))
    w_mult[valid_mask] = compute_w_multiplier(mapping, raw_values[valid_mask])

    aligned_records = []
    for i, r in enumerate(bindingdb_records):
        r = r.copy()
        r['pkd_aligned'] = r['pkd_raw']  # NO modification
        r['w_multiplier'] = float(w_mult[i])
        aligned_records.append(r)

    log.info(f"  w_multiplier: mean={w_mult[valid_mask].mean():.3f}, "
             f"median={np.median(w_mult[valid_mask]):.3f}, "
             f"min={w_mult[valid_mask].min():.6f}")
    log.info(f"  pkd_aligned == pkd_raw (labels unchanged)")

    # Save mapping bundle (GMMs + diagnostics)
    mapping_bundle = {
        'source_gmm':  source_gmm,
        'target_gmm':  target_gmm,
        'diagnostics': diag,
        'noise_sigma': noise_sigma,
    }
    os.makedirs(os.path.dirname(out_mapping_path) or '.', exist_ok=True)
    joblib.dump(mapping_bundle, out_mapping_path)
    log.info(f"GMM bundle saved → {out_mapping_path}")

    # Save records
    os.makedirs(os.path.dirname(out_bindingdb_path) or '.', exist_ok=True)
    with open(out_bindingdb_path, 'wb') as f:
        pickle.dump(aligned_records, f)
    log.info(f"Records (with w_multiplier, pKd unchanged) saved → {out_bindingdb_path}")

    # Optional diagnostic plot
    if plot_path:
        plot_analysis(pdbbind_pkd, bindingdb_pkd, w_mult[valid_mask], diag, plot_path)

    return aligned_records


def load_and_apply_mapping(mapping_path: str,
                            pkd_values: np.ndarray) -> np.ndarray:
    """Load saved GMMs and compute w_multiplier for new values."""
    bundle     = joblib.load(mapping_path)
    source_gmm = bundle['source_gmm']
    target_gmm = bundle['target_gmm']
    mapping    = build_quantile_mapping(source_gmm, target_gmm)
    return compute_w_multiplier(mapping, pkd_values)


# ─────────────────────────────────────────────────────────────
# 6. CLI
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fit GMMs and compute w_multiplier (no pKd modification)')
    parser.add_argument('--pdbbind',      required=True)
    parser.add_argument('--bindingdb',    required=True)
    parser.add_argument('--out-bindingdb', required=True)
    parser.add_argument('--out-mapping',   required=True)
    parser.add_argument('--plot',          default=None)
    parser.add_argument('--max-components', type=int, default=10)
    parser.add_argument('--noise-sigma',    type=float, default=0.12)
    args = parser.parse_args()

    with open(args.pdbbind,   'rb') as f:
        pdbbind_records = pickle.load(f)
    with open(args.bindingdb, 'rb') as f:
        bindingdb_records = pickle.load(f)

    align_distributions(
        pdbbind_records=pdbbind_records,
        bindingdb_records=bindingdb_records,
        out_bindingdb_path=args.out_bindingdb,
        out_mapping_path=args.out_mapping,
        plot_path=args.plot,
        max_components=args.max_components,
        noise_sigma=args.noise_sigma,
    )
