"""
parse_brenda.py
===============
Parse the BRENDA flat file (brenda_2026_1.txt) and align kinetic data
(KM, kcat, kcat/KM) to the oxidoreductase dataset by EC number.

BRENDA flat file format:
  - Entries delimited by "///"
  - Each entry starts with "ID\t<ec_number>"
  - PR lines define organisms: #N# Organism name <refs>
  - KM\t#N# value {substrate} (conditions) <refs>
  - TN\t#N# value {substrate} (conditions) <refs>
  - KKM\t#N# value {substrate} (conditions) <refs>

Output:
  - processed/oxidoreductase/brenda_kinetics.parquet  (per-EC kinetics)
  - processed/oxidoreductase/brenda_aligned.parquet    (per-record aligned)
"""

import os
import re
import pickle
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent / "dataset_building"
DATA_DIR = PROJECT_DIR / "data"
OUT_DIR = PROJECT_DIR / "processed" / "oxidoreductase"


# ─────────────────────────────────────────────────────────────
# 1. BRENDA flat file parser
# ─────────────────────────────────────────────────────────────

def parse_brenda_flatfile(filepath: str, ec_filter_prefix: str = "1.") -> dict:
    """
    Parse BRENDA flat file into structured data.

    Returns dict keyed by EC number, each containing:
      - organisms: {org_id: name_string}
      - km_entries: [{organism, substrate, value_mM, pmid}]
      - kcat_entries: [{organism, substrate, value_s, pmid}]
      - kcatkm_entries: [{organism, substrate, value_M1s1, pmid}]
    """
    log.info(f"Parsing BRENDA flat file: {filepath}")

    entries = {}
    current_ec = None
    current_section = None
    current_organisms = {}
    current_km = []
    current_kcat = []
    current_kcatkm = []

    def _flush_entry():
        nonlocal current_ec, current_organisms, current_km, current_kcat, current_kcatkm
        if current_ec and current_ec.startswith(ec_filter_prefix):
            entries[current_ec] = {
                "organisms": current_organisms.copy(),
                "km_entries": current_km.copy(),
                "kcat_entries": current_kcat.copy(),
                "kcatkm_entries": current_kcatkm.copy(),
            }
        current_ec = None
        current_organisms = {}
        current_km = []
        current_kcat = []
        current_kcatkm = []

    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in tqdm(f, desc="Parsing BRENDA", total=5_400_000):
            line = line.rstrip("\n")

            # Entry delimiter
            if line == "///":
                _flush_entry()
                continue

            # Section headers
            if line == "PROTEIN":
                current_section = "PROTEIN"
                continue
            elif line == "KM_VALUE":
                current_section = "KM"
                continue
            elif line == "TURNOVER_NUMBER":
                current_section = "kcat"
                continue
            elif line == "KCAT_KM_VALUE":
                current_section = "kcatkm"
                continue
            elif line in ("KI_VALUE", "IC50_VALUE", "SPECIFIC_ACTIVITY",
                          "PH_OPTIMUM", "PH_RANGE", "PH_STABILITY",
                          "TEMPERATURE_OPTIMUM", "TEMPERATURE_RANGE",
                          "TEMPERATURE_STABILITY", "MOLECULAR_WEIGHT",
                          "SUBUNITS", "POSTTRANSLATIONAL_MODIFICATION",
                          "PURIFICATION", "CLONED", "EXPRESSION",
                          "RENATURED", "APPLICATION", "ENGINEERING",
                          "GENERAL_STABILITY", "ORGANIC_SOLVENT_STABILITY",
                          "OXIDATION_STABILITY", "STORAGE_STABILITY",
                          "COFACTOR", "METALS_IONS", "INHIBITORS",
                          "ACTIVATING_COMPOUND", "NATURAL_SUBSTRATE_PRODUCT",
                          "SUBSTRATE_PRODUCT", "REACTION", "REACTION_TYPE",
                          "RECOMMENDED_NAME", "SYSTEMATIC_NAME",
                          "SYNONYMS", "SOURCE_TISSUE", "LOCALIZATION",
                          "GENERAL_INFORMATION", "REFERENCE",
                          "CRYSTALLIZATION", "EN", "GI", "GS", "AP", "AC",
                          "IN", "CF", "CR", "CL", "LO", "ME", "MW", "NSP",
                          "OS", "OSS", "PHO", "PHR", "PHS", "PM", "PU",
                          "RF", "RN", "RT", "SA", "SN", "SP", "SS", "ST",
                          "SU", "SY", "TR", "TS", "EXP", "REN"):
                current_section = None
                continue

            # ID line
            if line.startswith("ID\t"):
                current_ec = line.split("\t")[1].strip()
                current_section = None
                continue

            # Data lines
            if line.startswith("PR\t") and current_section == "PROTEIN":
                org_id, name = _parse_pr_line(line)
                if org_id:
                    current_organisms[org_id] = name
                continue

            if line.startswith("KM\t") and current_section == "KM":
                entry = _parse_kinetic_line(line, "KM")
                if entry:
                    current_km.append(entry)
                continue

            if line.startswith("TN\t") and current_section == "kcat":
                entry = _parse_kinetic_line(line, "kcat")
                if entry:
                    current_kcat.append(entry)
                continue

            if line.startswith("KKM\t") and current_section == "kcatkm":
                entry = _parse_kinetic_line(line, "kcatkm")
                if entry:
                    current_kcatkm.append(entry)
                continue

    # Don't forget the last entry
    _flush_entry()

    n_entries = len(entries)
    n_km = sum(len(e["km_entries"]) for e in entries.values())
    n_kcat = sum(len(e["kcat_entries"]) for e in entries.values())
    n_kcatkm = sum(len(e["kcatkm_entries"]) for e in entries.values())
    log.info(f"Parsed {n_entries} EC {ec_filter_prefix} entries: "
             f"{n_km} KM, {n_kcat} kcat, {n_kcatkm} kcat/KM")
    return entries


def _parse_pr_line(line: str) -> tuple:
    """Parse a PROTEIN line like: PR\t#1# Gallus gallus <44>"""
    match = re.match(r"PR\t#(\d+)#\s+(.+)$", line)
    if match:
        org_id = int(match.group(1))
        name = match.group(2).strip()
        # Trim trailing references <...>
        name = re.sub(r"\s*<\d+(?:,\d+)*>\s*$", "", name)
        return org_id, name
    return None, None


def _parse_kinetic_line(line: str, param_type: str) -> dict | None:
    """
    Parse KM/TN/KKM data lines.

    Format: KM\t#10# 0.05 {benzyl alcohol}  (#10# isoenzyme ADH-3, pH 10.0 <49>) <49>

    Returns dict with organism_id, substrate, value, unit, comment, pmid
    """
    # Split on tab
    parts = line.split("\t", 1)
    if len(parts) < 2:
        return None
    content = parts[1].strip()

    # Extract organism reference: #N#
    org_match = re.match(r"#(\d+)#\s+", content)
    if not org_match:
        return None
    org_id = int(org_match.group(1))
    rest = content[org_match.end():]

    # Extract value (number, possibly with decimal or scientific notation)
    val_match = re.match(r"([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)\s*", rest)
    if not val_match:
        return None
    value = float(val_match.group(1))
    rest = rest[val_match.end():]

    # Extract substrate: {substrate}
    sub_match = re.match(r"\{([^}]*)\}", rest)
    substrate = sub_match.group(1).strip() if sub_match else ""
    if sub_match:
        rest = rest[sub_match.end():].strip()

    # Extract comment: (comment)
    comment = ""
    if rest.startswith("("):
        depth = 0
        end = 0
        for i, c in enumerate(rest):
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end > 0:
            comment = rest[1:end].strip()
            rest = rest[end + 1:].strip()

    # Extract PMIDs from trailing <N> and <N,N,...>
    pmids = []
    pmid_match = re.findall(r"<(\d+)>", rest)
    pmids = [int(p) for p in pmid_match]

    # Determine unit based on param type
    unit = _infer_unit(param_type, comment)

    return {
        "organism_id": org_id,
        "substrate": substrate,
        "value": value,
        "unit": unit,
        "comment": comment,
        "pmid": ";".join(str(p) for p in pmids) if pmids else None,
    }


def _infer_unit(param_type: str, comment: str) -> str:
    """Infer measurement unit."""
    if param_type == "KM":
        # BRENDA KM values are in mM by default
        return "mM"
    elif param_type == "kcat":
        # BRENDA TN values are in s⁻¹
        return "s⁻¹"
    elif param_type == "kcatkm":
        # BRENDA KKM values are in mM⁻¹s⁻¹
        # But kcat/KM should be M⁻¹s⁻¹, so convert from mM⁻¹s⁻¹
        return "mM⁻¹s⁻¹"
    return ""


# ─────────────────────────────────────────────────────────────
# 2. Build per-EC kinetics summary
# ─────────────────────────────────────────────────────────────

def build_ec_summary(brenda_data: dict) -> pd.DataFrame:
    """Aggregate BRENDA kinetics per EC number."""
    rows = []
    for ec, data in brenda_data.items():
        # KM stats
        if data["km_entries"]:
            km_vals = [e["value"] for e in data["km_entries"]]
            km_uM = [v * 1000 for v in km_vals]  # mM → uM
            n_km = len(km_vals)
            km_median = float(np.median(km_uM))
            km_unique_substrates = len(set(e["substrate"] for e in data["km_entries"]))
        else:
            n_km, km_median, km_unique_substrates = 0, None, 0

        # kcat stats
        if data["kcat_entries"]:
            kcat_vals = [e["value"] for e in data["kcat_entries"]]
            n_kcat = len(kcat_vals)
            kcat_median = float(np.median(kcat_vals))
            kcat_unique_substrates = len(set(e["substrate"] for e in data["kcat_entries"]))
        else:
            n_kcat, kcat_median, kcat_unique_substrates = 0, None, 0

        # kcat/KM stats (convert from mM⁻¹s⁻¹ to M⁻¹s⁻¹)
        if data["kcatkm_entries"]:
            kkm_vals = [e["value"] * 1000 for e in data["kcatkm_entries"]]
            n_kkm = len(kkm_vals)
            kkm_median = float(np.median(kkm_vals))
            kkm_unique_substrates = len(set(e["substrate"] for e in data["kcatkm_entries"]))
        else:
            n_kkm, kkm_median, kkm_unique_substrates = 0, None, 0

        rows.append({
            "ec_number": ec,
            "n_organisms": len(data["organisms"]),
            "n_km": n_km,
            "km_median_uM": km_median,
            "km_unique_substrates": km_unique_substrates,
            "n_kcat": n_kcat,
            "kcat_median_s": kcat_median,
            "kcat_unique_substrates": kcat_unique_substrates,
            "n_kcatkm": n_kkm,
            "kcatkm_median_M1s1": kkm_median,
            "kcatkm_unique_substrates": kkm_unique_substrates,
        })

    df = pd.DataFrame(rows)
    return df


# ─────────────────────────────────────────────────────────────
# 3. Align with oxidoreductase records
# ─────────────────────────────────────────────────────────────

def align_to_records(
    brenda_data: dict,
    ec_summary: pd.DataFrame,
    records: list,
    uniprot_annotations: dict,
) -> pd.DataFrame:
    """
    For each oxidoreductase record, find matching BRENDA EC kinetics.

    The alignment is EC-based: if the record's UniProt entry has EC numbers
    that are in BRENDA, we attach the median KM/kcat/kcatKM for that EC.
    """
    aligned_rows = []

    for r in records:
        uid = r.get("uniprot_id")
        if not uid:
            continue

        ann = uniprot_annotations.get(uid, {})
        ecs = ann.get("ec_numbers", [])
        if not ecs:
            continue

        # Find first matching EC in BRENDA
        for ec in ecs:
            if ec in brenda_data:
                bd = brenda_data[ec]
                summary = ec_summary[ec_summary["ec_number"] == ec].iloc[0] if len(ec_summary[ec_summary["ec_number"] == ec]) > 0 else None

                aligned_rows.append({
                    "uniprot_id": uid,
                    "pdb_id": r.get("pdb_id"),
                    "ec_number": ec,
                    "source_db": r.get("source_db"),
                    "measurement_type": r.get("measurement_type"),
                    "pkd_raw": r.get("pkd_raw"),
                    "n_km_bdb": len(bd["km_entries"]),
                    "n_kcat_bdb": len(bd["kcat_entries"]),
                    "n_kcatkm_bdb": len(bd["kcatkm_entries"]),
                    "km_median_uM": summary["km_median_uM"] if summary is not None else None,
                    "kcat_median_s": summary["kcat_median_s"] if summary is not None else None,
                    "kcatkm_median_M1s1": summary["kcatkm_median_M1s1"] if summary is not None else None,
                })
                break  # Use first matching EC

    df = pd.DataFrame(aligned_rows)
    return df


# ─────────────────────────────────────────────────────────────
# 4. Main entry point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Parse BRENDA flat file")
    parser.add_argument("--brenda", default=str(DATA_DIR / "brenda_2026_1.txt"))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--max-entries", type=int, default=None,
                        help="Limit EC entries for testing")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ── Step 1: Parse BRENDA ──
    log.info("=" * 50)
    log.info("Step 1: Parsing BRENDA flat file")
    log.info("=" * 50)
    brenda_data = parse_brenda_flatfile(args.brenda, ec_filter_prefix="1.")

    # ── Step 2: Build EC summary ──
    log.info("=" * 50)
    log.info("Step 2: Building EC-level kinetics summary")
    log.info("=" * 50)
    ec_summary = build_ec_summary(brenda_data)
    log.info(f"  EC entries with kinetics: {len(ec_summary)}")

    n_with_kcat = sum(1 for _, r in ec_summary.iterrows() if r["n_kcat"] > 0)
    n_with_kkm = sum(1 for _, r in ec_summary.iterrows() if r["n_kcatkm"] > 0)
    log.info(f"  With kcat: {n_with_kcat}")
    log.info(f"  With kcat/KM: {n_with_kkm}")

    # Top ECs by kcat coverage
    if n_with_kcat > 0:
        top_kcat = ec_summary.nlargest(10, "n_kcat")[
            ["ec_number", "n_kcat", "kcat_median_s", "n_km", "km_median_uM"]
        ]
        log.info(f"  Top ECs by kcat entries:\n{top_kcat.to_string()}")

    # ── Step 3: Align with oxidoreductase records ──
    log.info("=" * 50)
    log.info("Step 3: Aligning with oxidoreductase records")
    log.info("=" * 50)

    recs_path = out_dir / "oxidoreductase_records.pkl"
    ann_cache_path = out_dir / "cache" / "uniprot_annotations.json"

    if recs_path.exists() and ann_cache_path.exists():
        import json
        records = pickle.load(open(recs_path, "rb"))
        with open(ann_cache_path) as f:
            uniprot_annotations = json.load(f)

        aligned_df = align_to_records(brenda_data, ec_summary,
                                       records, uniprot_annotations)
        log.info(f"  Aligned records: {len(aligned_df)}")

        if len(aligned_df) > 0:
            n_km = sum(aligned_df["n_km_bdb"] > 0)
            n_kcat = sum(aligned_df["n_kcat_bdb"] > 0)
            n_kkm = sum(aligned_df["n_kcatkm_bdb"] > 0)
            log.info(f"  Records with BRENDA KM: {n_km}")
            log.info(f"  Records with BRENDA kcat: {n_kcat}")
            log.info(f"  Records with BRENDA kcat/KM: {n_kkm}")

            kkm_vals = aligned_df["kcatkm_median_M1s1"].dropna()
            if len(kkm_vals) > 0:
                log.info(f"  kcat/KM range: [{kkm_vals.min():.1f}, {kkm_vals.max():.1f}] M⁻¹s⁻¹")
                log.info(f"  kcat/KM > 10^8 (diffusion limit): "
                         f"{sum(kkm_vals > 1e8)} records")

        aligned_path = out_dir / "brenda_aligned.parquet"
        aligned_df.to_parquet(aligned_path, index=False)
        log.info(f"  Aligned data saved → {aligned_path}")
    else:
        log.warning("  oxidoreductase_records.pkl or uniprot_annotations.json "
                     "not found, skipping alignment")

    # ── Step 4: Save ──
    log.info("=" * 50)
    log.info("Step 4: Saving BRENDA kinetics")
    log.info("=" * 50)

    ec_summary_path = out_dir / "brenda_ec_summary.parquet"
    ec_summary.to_parquet(ec_summary_path, index=False)
    log.info(f"  EC summary → {ec_summary_path}")

    # Also save per-EC detailed kinetics (all KM/kcat/kcatKM entries)
    detailed_rows = []
    for ec, data in brenda_data.items():
        for entry in data["km_entries"]:
            entry["ec_number"] = ec
            entry["param_type"] = "KM"
            detailed_rows.append(entry)
        for entry in data["kcat_entries"]:
            entry["ec_number"] = ec
            entry["param_type"] = "kcat"
            detailed_rows.append(entry)
        for entry in data["kcatkm_entries"]:
            entry["ec_number"] = ec
            entry["param_type"] = "kcatKM"
            detailed_rows.append(entry)

    detailed_df = pd.DataFrame(detailed_rows)
    detailed_path = out_dir / "brenda_kinetics.parquet"
    detailed_df.to_parquet(detailed_path, index=False)
    log.info(f"  Detailed kinetics ({len(detailed_df):,} entries) → {detailed_path}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  BRENDA 动力学数据解析 — 完成报告")
    print("=" * 60)
    print(f"  EC 1.x.x.x 条目数:        {len(brenda_data)}")
    print(f"  KM 总数:                  {sum(len(e['km_entries']) for e in brenda_data.values()):,}")
    print(f"  kcat 总数:                {sum(len(e['kcat_entries']) for e in brenda_data.values()):,}")
    print(f"  kcat/KM 总数:             {sum(len(e['kcatkm_entries']) for e in brenda_data.values()):,}")
    print(f"  有 kcat 的 EC 数:         {n_with_kcat}")
    print(f"  有 kcat/KM 的 EC 数:      {n_with_kkm}")
    if recs_path.exists():
        print(f"  对齐记录数:               {len(aligned_df) if 'aligned_df' in dir() else 'N/A'}")
    print(f"  输出目录:                 {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
