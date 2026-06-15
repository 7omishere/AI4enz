#!/usr/bin/env python3
"""
从 BRENDA + UniProt SwissProt 提取辅因子信息，补充到 metadata.parquet。

数据源:
  - BRENDA: EC → cofactor(s)
  - UniProt: uniprot_id → cofactor(s)

用法：
  source /home/domi/BINN/.venv/bin/activate
  python scripts/supplement_cofactors.py
"""

import gzip
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
BRENDA_FILE = BASE_DIR / "BRENDA" / "brenda_2026_1.txt"
UNIPROT_FILE = BASE_DIR / "uniprotprot" / "uniprot_sprot.dat.gz"
METADATA_PATH = BASE_DIR / "processed" / "metadata.parquet"

# ── 辅因子名称 → 模型标准名 ──
# 映射到 ranking_model.py 中的 COFACTOR_DOMAIN_TYPES + COFACTOR_PRIORS
COFACTOR_NAME_MAP = {
    # NAD family
    "nad+": "NAD", "nadh": "NAD", "nad(h)": "NAD", "nad": "NAD",
    "nadp+": "NADP", "nadph": "NADP", "nadp(h)": "NADP", "nadp": "NADP",
    # Flavin
    "fad": "FAD", "fadh2": "FAD", "fadh": "FAD",
    "fmn": "FMN", "fmnh2": "FMN",
    # Heme
    "heme": "HEME", "heme b": "HEME", "heme c": "HEME",
    "heme a": "HEME", "protoheme": "HEME", "protoheme ix": "HEME",
    "siroheme": "HEME",
    # Iron-sulfur
    "iron-sulfur": "FES", "fes": "FES", "fe-s": "FES",
    "2fe-2s": "FES", "4fe-4s": "FES", "3fe-4s": "FES",
    "[2fe-2s]": "FES", "[4fe-4s]": "FES", "[3fe-4s]": "FES",
    "fe2s2": "FES", "fe4s4": "FES", "fe-s cluster": "FES",
    "iron-sulfur cluster": "FES",
    # Copper
    "copper": "CU", "cu2+": "CU", "cu+": "CU", "cu": "CU",
    "cu(ii)": "CU", "cu(i)": "CU",
    # Molybdopterin
    "molybdopterin": "MPT", "molybdenum": "MPT", "moco": "MPT",
    "molybdenum cofactor": "MPT", "molybdate": "MPT",
    # Coenzyme Q
    "coenzyme q": "COQ", "ubiquinone": "COQ", "coq": "COQ", "q10": "COQ",
    # PQQ
    "pqq": "PQQ", "pyrroloquinoline quinone": "PQQ",
    # TPP
    "thiamine": "TPP", "tpp": "TPP",
    "thiamine diphosphate": "TPP", "thiamine pyrophosphate": "TPP",
    # PLP
    "pyridoxal": "PLP", "plp": "PLP",
    "pyridoxal phosphate": "PLP", "pyridoxal 5'-phosphate": "PLP",
    # CoA
    "coenzyme a": "COA", "coa": "COA", "acetyl-coa": "COA",
    # B12
    "cobalamin": "B12", "adenosylcobalamin": "B12",
    "methylcobalamin": "B12", "b12": "B12", "vitamin b12": "B12",
    "cob(ii)alamin": "B12",
    # THF
    "tetrahydrofolate": "THF", "thf": "THF",
    "5,10-methylenetetrahydrofolate": "THF",
    # Metals (in COFACTOR_PRIORS)
    "zn(2+)": "ZN", "zinc": "ZN", "zn2+": "ZN",
    "mg(2+)": "MG", "magnesium": "MG", "mg2+": "MG",
    "ni(2+)": "NI", "nickel": "NI", "ni2+": "NI",
    "sulfur": "SULFUR", "s(2-)": "SULFUR",
}


def normalize_cofactor_name(name: str) -> str | None:
    """Map cofactor name to canonical type."""
    name = name.strip().lower()
    name = re.sub(r'\([^)]*\)', '', name).strip()
    name = re.sub(r'\s+', ' ', name)
    if name in COFACTOR_NAME_MAP:
        return COFACTOR_NAME_MAP[name]
    for key, val in COFACTOR_NAME_MAP.items():
        if key in name or name in key:
            return val
    return None


def parse_brenda_cofactors() -> dict[str, set[str]]:
    """BRENDA → {EC: {cofactor_type, ...}}"""
    print(f"Parsing BRENDA: {BRENDA_FILE.name}")

    with open(BRENDA_FILE, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    entries = re.split(r"\n///\n", content)
    print(f"  Entries: {len(entries):,}")

    ec_cofactors: dict[str, set[str]] = defaultdict(set)
    n_matched = 0

    for entry in entries:
        ec_match = re.search(r"^ID\s+(.+)", entry, re.MULTILINE)
        if not ec_match:
            continue
        ec = ec_match.group(1).strip()
        if not ec or ec == "spontaneous":
            continue

        cf_match = re.search(r"^COFACTOR\n((?:CF\s+.+\n?)+)", entry, re.MULTILINE)
        if not cf_match:
            continue

        for line in cf_match.group(1).split("\n"):
            line = line.strip()
            if not line.startswith("CF"):
                continue
            text = re.sub(r"#[\d,]+#", " ", line)
            for part in text.split():
                if part.startswith("<") or part.startswith("("):
                    continue
                norm = normalize_cofactor_name(part)
                if norm:
                    ec_cofactors[ec].add(norm)
                    n_matched += 1

    print(f"  ECs with cofactors: {len(ec_cofactors):,} ({n_matched} annotations)")
    return ec_cofactors


def parse_uniprot_cofactors() -> dict[str, set[str]]:
    """UniProt SwissProt → {uniprot_ac: {cofactor_type, ...}}"""
    print(f"\nParsing UniProt: {UNIPROT_FILE.name}")

    uniprot_cofactors: dict[str, set[str]] = defaultdict(set)

    with gzip.open(UNIPROT_FILE, "rt", encoding="utf-8", errors="replace") as f:
        current_ac = None
        in_cofactor = False
        for line in f:
            if line.startswith("AC   "):
                current_ac = line[5:].strip().split(";")[0].strip()
            elif line.startswith("CC   -!- COFACTOR:"):
                in_cofactor = True
            elif in_cofactor and "Name=" in line:
                # Format: CC       Name=FAD; Xref=...
                name_match = re.search(r"Name=([^;]+)", line)
                if name_match and current_ac:
                    norm = normalize_cofactor_name(name_match.group(1))
                    if norm:
                        uniprot_cofactors[current_ac].add(norm)
                in_cofactor = False
            elif not line.startswith("CC       "):
                in_cofactor = False

    print(f"  Proteins with cofactors: {len(uniprot_cofactors):,}")
    return uniprot_cofactors


def main():
    # 1. Parse both sources
    ec_cofactors = parse_brenda_cofactors()
    uniprot_cofactors = parse_uniprot_cofactors()

    # 2. Show distribution
    type_counts = defaultdict(int)
    for cfs in ec_cofactors.values():
        for cf in cfs:
            type_counts[cf] += 1
    print("\nTop cofactor types (BRENDA):")
    for cf, count in sorted(type_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {cf}: {count:,} ECs")

    # 3. Apply to metadata (EC-based + UniProt-based)
    meta = pd.read_parquet(METADATA_PATH)
    print(f"\nMetadata: {len(meta):,} rows")

    def get_ec_list(ec_str):
        if pd.isna(ec_str) or not ec_str:
            return []
        return [e.strip() for e in str(ec_str).split(";") if e.strip()]

    before = (meta["cofactors"].notna() & (meta["cofactors"] != "")).sum()

    new_cofactors = []
    ec_hits = 0
    uniprot_hits = 0
    for _, row in meta.iterrows():
        cfs = set()

        # EC-based (BRENDA)
        for ec in get_ec_list(row["ec_numbers"]):
            if ec in ec_cofactors:
                cfs.update(ec_cofactors[ec])
                ec_hits += 1

        # UniProt-based
        uid = row["uniprot_id"]
        if uid and uid in uniprot_cofactors:
            cfs.update(uniprot_cofactors[uid])
            uniprot_hits += 1

        new_cofactors.append("|".join(sorted(cfs)) if cfs else "")

    meta["cofactors"] = new_cofactors
    after = (meta["cofactors"] != "").sum()
    print(f"  EC-based hits:       {ec_hits:,} rows")
    print(f"  UniProt-based hits:  {uniprot_hits:,} rows")
    print(f"  Before: {before:,} → After: {after:,} ({100*after/len(meta):.1f}%)")

    meta.to_parquet(METADATA_PATH, index=False, compression="snappy")
    print(f"\nSaved → {METADATA_PATH}")

    # 4. Show final cofactor distribution
    final_type_counts = defaultdict(int)
    for cf_str in meta["cofactors"]:
        for cf in cf_str.split("|"):
            if cf:
                final_type_counts[cf] += 1
    print("\nFinal cofactor type distribution in metadata:")
    for cf, count in sorted(final_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {cf}: {count:,} ({100*count/len(meta):.1f}%)")


if __name__ == "__main__":
    main()
