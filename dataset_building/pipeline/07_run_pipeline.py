"""
07_run_pipeline.py
================
End-to-end orchestrator for the enzyme-substrate binding data pipeline.

Runs all steps in order:
  1. Parse PDBbind → processed/pdbbind_records.pkl
  2. Parse BindingDB → processed/bindingdb_records.pkl
  3. GMM analysis + w_multiplier → processed/bindingdb_aligned.pkl (pKd unchanged)
  4. Build ligand graphs → processed/ligands/{inchikey}.pt
  5. Write HDF5 + Parquet → processed/proteins.h5, processed/metadata.parquet

Each step checks for existing output and skips if already done (--resume).

Data paths are auto-detected from ../data/ relative to this script.
You can override any path via command-line arguments.

Usage:
  # Full run (auto-detect data/)
  python 07_run_pipeline.py --workers 8

  # Override specific paths
  python 07_run_pipeline.py \\
      --pdbbind-index  data/index/INDEX_general_PL.2020R1.lst \\
      --pdbbind-struct data/P-L \\
      --bindingdb-tsv  data/BindingDB_All.tsv \\
      --out-dir        processed \\
      --workers        8

  # Debug run (small subset)
  python 07_run_pipeline.py --max-pdbbind 100 --max-bindingdb 1000

  # Resume from checkpoint
  python 07_run_pipeline.py --resume
"""

import os
import pickle
import argparse
import importlib
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Auto-detect data directory relative to this script
# ─────────────────────────────────────────────────────────────

def _resolve_data_dir() -> Path:
    """Find the data directory relative to this script, then CWD."""
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / 'data'
    if data_dir.is_dir():
        return data_dir
    cwd_data = Path.cwd() / 'data'
    if cwd_data.is_dir():
        return cwd_data
    raise FileNotFoundError(
        "Cannot find data/ directory. "
        "Expected at ../data/ relative to this script or ./data/ in CWD."
    )

def _find_pdbbind_index(data_dir: Path) -> str:
    """Find PDBbind PL index file in data/index/."""
    candidates = sorted(data_dir.glob('index/INDEX_general_PL*.lst'))
    if not candidates:
        raise FileNotFoundError(f"No PDBbind PL index found in {data_dir}/index/")
    return str(candidates[0])

def _find_pdbbind_struct(data_dir: Path) -> str:
    """Find PDBbind structure directory (P-L/)."""
    pl_dir = data_dir / 'P-L'
    if pl_dir.is_dir():
        return str(pl_dir)
    raise FileNotFoundError(f"No P-L structure directory found in {data_dir}")

def _find_bindingdb_tsv(data_dir: Path) -> str:
    """Find BindingDB TSV file (may be .tsv or .tsv.gz)."""
    candidates = sorted(data_dir.glob('BindingDB_All.tsv*'))
    if not candidates:
        raise FileNotFoundError(f"No BindingDB TSV found in {data_dir}")
    return str(candidates[0])

def _find_coreset(data_dir: Path) -> str | None:
    """Find PDBbind core set file, if available."""
    candidates = sorted(data_dir.glob('index/CoreSet*'))
    if candidates:
        return str(candidates[0])
    return None


def _exists(path: str) -> bool:
    return Path(path).exists() and Path(path).stat().st_size > 0


def run_pipeline(
    pdbbind_index:  str,
    pdbbind_struct: str,
    bindingdb_tsv:  str,
    out_dir:        str,
    core_set:       str  = None,
    workers:        int  = 4,
    max_pdbbind:    int  = None,
    max_bindingdb:  int  = None,
    resume:         bool = True,
    no_uniprot:     bool = False,
    plot_alignment: bool = True,
) -> None:
    """
    Run the full data pipeline.

    Parameters
    ----------
    pdbbind_index  : INDEX_general_PL_data.2020
    pdbbind_struct : path to general-set or refined-set directory
    bindingdb_tsv  : BindingDB_All.tsv.gz
    out_dir        : output directory
    core_set       : PDBbind 2016 CoreSet.dat (for test split)
    workers        : parallel workers for parsing and graph building
    max_pdbbind    : limit PDBbind complexes (None = all)
    max_bindingdb  : limit BindingDB rows (None = all)
    resume         : skip steps whose output already exists
    no_uniprot     : skip UniProt API calls (faster, no binding sites for BindingDB)
    plot_alignment : save alignment diagnostic plot
    """
    os.makedirs(out_dir, exist_ok=True)

    pdbbind_pkl   = str(Path(out_dir) / 'pdbbind_records.pkl')
    bindingdb_pkl = str(Path(out_dir) / 'bindingdb_records.pkl')
    aligned_pkl   = str(Path(out_dir) / 'bindingdb_aligned.pkl')
    mapping_jl    = str(Path(out_dir) / 'alignment_mapping.joblib')
    ligands_dir   = str(Path(out_dir) / 'ligands')
    h5_path       = str(Path(out_dir) / 'proteins.h5')
    parquet_path  = str(Path(out_dir) / 'metadata.parquet')
    align_plot    = str(Path(out_dir) / 'pkd_alignment.png') if plot_alignment else None

    t0 = time.time()

    # ── Step 1: Parse PDBbind ────────────────────────────────
    if resume and _exists(pdbbind_pkl):
        log.info(f"[Step 1] Skipping PDBbind parse (found {pdbbind_pkl})")
        with open(pdbbind_pkl, 'rb') as f:
            pdbbind_records = pickle.load(f)
    else:
        log.info("[Step 1] Parsing PDBbind ...")
        parse_pdbbind = importlib.import_module('01_parse_pdbbind').parse_pdbbind
        pdbbind_records = parse_pdbbind(
            index_path    = pdbbind_index,
            struct_dir    = pdbbind_struct,
            out_path      = pdbbind_pkl,
            n_workers     = workers,
            max_complexes = max_pdbbind,
        )
    log.info(f"  PDBbind: {len(pdbbind_records):,} records")

    # ── Step 2: Parse BindingDB ──────────────────────────────
    if resume and _exists(bindingdb_pkl):
        log.info(f"[Step 2] Skipping BindingDB parse (found {bindingdb_pkl})")
        with open(bindingdb_pkl, 'rb') as f:
            bindingdb_records = pickle.load(f)
    else:
        log.info("[Step 2] Parsing BindingDB ...")
        parse_bindingdb = importlib.import_module('02_parse_bindingdb').parse_bindingdb
        bindingdb_records = parse_bindingdb(
            tsv_path            = bindingdb_tsv,
            out_path            = bindingdb_pkl,
            fetch_binding_sites = not no_uniprot,
            uniprot_workers     = workers,
            max_rows            = max_bindingdb,
        )
    log.info(f"  BindingDB: {len(bindingdb_records):,} records")

    # ── Step 3: GMM analysis + w_multiplier (no pKd change) ──
    if resume and _exists(aligned_pkl):
        log.info(f"[Step 3] Skipping alignment (found {aligned_pkl})")
        with open(aligned_pkl, 'rb') as f:
            aligned_records = pickle.load(f)
    else:
        log.info("[Step 3] GMM analysis + w_multiplier (pKd unchanged) ...")
        align_distributions = importlib.import_module('03_align_distributions').align_distributions
        aligned_records = align_distributions(
            pdbbind_records    = pdbbind_records,
            bindingdb_records  = bindingdb_records,
            out_bindingdb_path = aligned_pkl,
            out_mapping_path   = mapping_jl,
            plot_path          = align_plot,
        )
    log.info(f"  Aligned BindingDB: {len(aligned_records):,} records")

    # PDBbind is the thermodynamic reference — no alignment stretching needed
    for r in pdbbind_records:
        r.setdefault('w_multiplier', 1.0)

    # ── Step 4: Build ligand graphs ──────────────────────────
    all_records = pdbbind_records + aligned_records
    if resume and Path(ligands_dir).exists() and any(Path(ligands_dir).glob('*.pt')):
        n_existing = len(list(Path(ligands_dir).glob('*.pt')))
        log.info(f"[Step 4] Ligand graphs: {n_existing:,} .pt files found, "
                 f"building missing ones ...")
    else:
        log.info("[Step 4] Building ligand molecular graphs ...")

    build_ligand_graphs = importlib.import_module('04_build_ligand_graphs').build_ligand_graphs
    build_ligand_graphs(
        records   = all_records,
        out_dir   = ligands_dir,
        n_workers = workers,
        overwrite = not resume,
    )

    # ── Step 5: Write HDF5 + Parquet ─────────────────────────
    if resume and _exists(h5_path) and _exists(parquet_path):
        log.info(f"[Step 5] Skipping storage write (found {h5_path}, {parquet_path})")
    else:
        log.info("[Step 5] Writing HDF5 + Parquet storage ...")
        write_storage = importlib.import_module('05_write_storage').write_storage
        write_storage(
            pdbbind_records   = pdbbind_records,
            bindingdb_records = aligned_records,
            out_dir           = out_dir,
            core_set_path     = core_set,
        )

    elapsed = time.time() - t0
    log.info(f"\nPipeline complete in {elapsed/3600:.1f} hours")
    log.info(f"  proteins.h5      → {h5_path}")
    log.info(f"  metadata.parquet → {parquet_path}")
    log.info(f"  ligands/         → {ligands_dir}/")
    if align_plot:
        log.info(f"  alignment plot   → {align_plot}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='End-to-end enzyme binding data pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--pdbbind-index',  default=None,
                        help='PDBbind PL index file (auto-detected from data/ if omitted)')
    parser.add_argument('--pdbbind-struct', default=None,
                        help='PDBbind structure directory (auto-detected from data/ if omitted)')
    parser.add_argument('--bindingdb-tsv',  default=None,
                        help='BindingDB TSV file (auto-detected from data/ if omitted)')
    parser.add_argument('--out-dir',        default=None,
                        help='Output directory (default: ../processed)')
    parser.add_argument('--core-set',       default=None,
                        help='PDBbind 2016 CoreSet.dat (auto-detected if available)')
    parser.add_argument('--workers',        type=int, default=4,
                        help='Parallel workers')
    parser.add_argument('--max-pdbbind',    type=int, default=None,
                        help='Limit PDBbind complexes (debug)')
    parser.add_argument('--max-bindingdb',  type=int, default=None,
                        help='Limit BindingDB rows (debug)')
    parser.add_argument('--resume',         action='store_true', default=True,
                        help='Skip steps with existing output')
    parser.add_argument('--no-resume',      dest='resume', action='store_false')
    parser.add_argument('--no-uniprot',     action='store_true',
                        help='Skip UniProt API calls')
    parser.add_argument('--no-plot',        action='store_true',
                        help='Skip alignment diagnostic plot')
    parser.add_argument('--test',           action='store_true',
                        help='Test mode: process 5 PDBbind + 100 BindingDB, skip UniProt/plot')
    args = parser.parse_args()

    # --test is a shortcut for debug-mode settings
    if args.test:
        args.max_pdbbind   = args.max_pdbbind   or 10
        args.max_bindingdb = args.max_bindingdb or 500
        args.no_uniprot    = True
        args.no_plot       = True
        args.resume        = False
        log.info("Test mode: max_pdbbind=10, max_bindingdb=500, no_uniprot, no_plot, no_resume")

    # Auto-detect data paths
    data_dir = _resolve_data_dir()
    log.info(f"Data directory: {data_dir}")

    script_dir = Path(__file__).resolve().parent
    default_out = str(script_dir.parent / 'processed')

    pdbbind_index  = args.pdbbind_index  or _find_pdbbind_index(data_dir)
    pdbbind_struct = args.pdbbind_struct or _find_pdbbind_struct(data_dir)
    bindingdb_tsv  = args.bindingdb_tsv  or _find_bindingdb_tsv(data_dir)
    out_dir        = args.out_dir        or default_out
    core_set       = args.core_set       or _find_coreset(data_dir)

    log.info(f"  PDBbind index : {pdbbind_index}")
    log.info(f"  PDBbind struct: {pdbbind_struct}")
    log.info(f"  BindingDB TSV : {bindingdb_tsv}")
    log.info(f"  Core set      : {core_set or '(not found)'}")
    log.info(f"  Output dir    : {out_dir}")

    run_pipeline(
        pdbbind_index  = pdbbind_index,
        pdbbind_struct = pdbbind_struct,
        bindingdb_tsv  = bindingdb_tsv,
        out_dir        = out_dir,
        core_set       = core_set,
        workers        = args.workers,
        max_pdbbind    = args.max_pdbbind,
        max_bindingdb  = args.max_bindingdb,
        resume         = args.resume,
        no_uniprot     = args.no_uniprot,
        plot_alignment = not args.no_plot,
    )
