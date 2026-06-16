#!/usr/bin/env python3
"""
extract_brenda_temperature.py
=============================
从 BRENDA (brenda_2026_1.txt) 提取 kcat 条目的温度数据。
构建 EC 级温度映射，补充到 Trenzition V5 未匹配温度的条目。

用法: python extract_brenda_temperature.py
"""

import re
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
BRENDA_PATH = PROJECT_DIR / "BRENDA" / "brenda_2026_1.txt"
V5_METADATA_PATH = PROJECT_DIR / "processed" / "metadata.parquet"
V5_WITH_NEG_PATH = PROJECT_DIR / "processed" / "metadata_with_negatives.parquet"


def parse_brenda_temperature(br_path: str) -> dict:
    """
    解析 BRENDA 全文，提取所有 EC 的 kcat 温度。

    返回:
        {ec_number: list_of_temperatures_C}
    """
    log.info(f"Loading BRENDA from {br_path} ({Path(br_path).stat().st_size / 1024 / 1024:.0f} MB)")
    with open(br_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    entries = re.split(r"\n///\n", content)
    log.info(f"Total enzyme entries: {len(entries)}")

    ec_temps = defaultdict(list)
    n_total_tn = 0
    n_with_temp = 0
    n_uniq_ec = set()

    for idx, entry in enumerate(entries):
        lines = entry.strip().split("\n")
        current_ec = None

        for line in lines:
            line = line.strip()
            if line.startswith("ID\t"):
                current_ec = line[3:].strip()
                n_uniq_ec.add(current_ec)
            elif line.startswith("TN\t") and current_ec:
                n_total_tn += 1
                # BRENDA TN 格式: TN\t#protein# value {substrate} (conditions) <ref>
                # conditions 中可能包含 "30°C", "pH 7.5", "25°C" 等
                match = re.match(
                    r'TN\s*#(\d+)#\s*([\d.eE+-]+)\s*(?:\{([^}]*)\})?\s*(?:\(([^)]*)\))?',
                    line,
                )
                if match:
                    conditions = match.group(4) or ""

                    # 提取温度: 数字 + °C/C 或 "temperature: XX"
                    temp = None
                    # 模式 1: "30°C", "25 C", "37°C"
                    t_match = re.search(
                        r'([\d.]+)\s*°?\s*[Cc](?:elsius)?(?:\s|\)|$|,|;)',
                        conditions,
                    )
                    if t_match:
                        try:
                            temp = float(t_match.group(1))
                        except ValueError:
                            pass

                    if temp is not None and 0 < temp < 300:
                        ec_temps[current_ec].append(temp)
                        n_with_temp += 1

    log.info(f"Unique ECs: {len(n_uniq_ec)}")
    log.info(f"TN entries: {n_total_tn}")
    log.info(f"TN with temperature: {n_with_temp}")
    log.info(f"ECs with temp data: {len(ec_temps)}")

    return ec_temps


def build_brenda_temp_map(ec_temps: dict) -> dict:
    """从 BRENDA 解析结果构建温度映射。

    返回:
        {ec_number: median_temperature_K}
    """
    temp_map = {}
    for ec, temps in ec_temps.items():
        if len(temps) >= 1:
            temp_map[ec] = np.median(temps)

    log.info(f"Built BRENDA temperature map: {len(temp_map)} EC keys")

    # 统计
    temp_vals = [v for v in temp_map.values()]
    log.info(f"  Temperature range: {min(temp_vals):.0f} - {max(temp_vals):.0f}°C")
    log.info(f"  25°C: {sum(1 for v in temp_vals if abs(v - 25) < 0.5)} ECs "
             f"({100 * sum(1 for v in temp_vals if abs(v - 25) < 0.5) / len(temp_vals):.1f}%)")
    log.info(f"  30°C: {sum(1 for v in temp_vals if abs(v - 30) < 0.5)} ECs")
    log.info(f"  37°C: {sum(1 for v in temp_vals if abs(v - 37) < 0.5)} ECs")

    # 转换为 Kelvin
    return {ec: t + 273.15 for ec, t in temp_map.items()}


def apply_brenda_temperature(
    metadata_path: str,
    output_path: str,
    brenda_temp_map: dict,
) -> pd.DataFrame:
    """
    对 metadata 中仍用默认温度的条目，用 BRENDA EC 级温度补充。
    已从 OED 获得温度的条目不覆盖。
    """
    df = pd.read_parquet(metadata_path)
    log.info(f"Metadata entries: {len(df)}")

    # 当前仍用默认温度的
    still_default = df["temperature_K"] == 298.15
    n_before = still_default.sum()
    log.info(f"Still using default 298.15K: {n_before} ({100 * n_before / len(df):.1f}%)")

    # 按 EC 匹配
    ec_keys = df.loc[still_default, "ec_numbers"].fillna("")
    brenda_match = ec_keys.isin(brenda_temp_map.keys())
    n_matched = brenda_match.sum()

    # 应用
    brenda_indices = still_default.index[still_default][brenda_match.values]
    matched_temps = ec_keys[brenda_match.values].map(brenda_temp_map)
    df.loc[brenda_indices, "temperature_K"] = matched_temps.values

    log.info(f"BRENDA matched: {n_matched}/{n_before} ({100 * n_matched / n_before:.1f}% of default)")

    # 最终统计
    n_still_default = (df["temperature_K"] == 298.15).sum()
    n_with_any = (df["temperature_K"] != 298.15).sum()
    log.info(f"After BRENDA supplement:")
    log.info(f"  With temperature: {n_with_any}/{len(df)} ({100 * n_with_any / len(df):.1f}%)")
    log.info(f"  Still default: {n_still_default} ({100 * n_still_default / len(df):.1f}%)")

    # 按 split
    log.info(f"Coverage by split:")
    for s in ["train", "val", "test"]:
        mask = df["split"] == s
        covered = ((df["temperature_K"] != 298.15) & mask).sum()
        total = mask.sum()
        log.info(f"  {s}: {covered}/{total} ({100 * covered / total:.1f}%)")

    df.to_parquet(output_path, index=False)
    log.info(f"Saved to {output_path}")

    return df


def main():
    # 1. 解析 BRENDA 温度
    ec_temps = parse_brenda_temperature(str(BRENDA_PATH))
    brenda_map = build_brenda_temp_map(ec_temps)

    # 2. 补充 V5 core
    log.info("\n" + "=" * 60)
    log.info("Applying to V5 core metadata")
    log.info("=" * 60)
    apply_brenda_temperature(
        str(V5_METADATA_PATH),
        str(V5_METADATA_PATH),
        brenda_map,
    )

    # 3. 补充 V5 with negatives
    log.info("\n" + "=" * 60)
    log.info("Applying to V5 with negatives")
    log.info("=" * 60)
    apply_brenda_temperature(
        str(V5_WITH_NEG_PATH),
        str(V5_WITH_NEG_PATH),
        brenda_map,
    )

    # 4. 看一下 BRENDA 特有 ECs（OED 没有的）
    log.info("\n" + "=" * 60)
    log.info("BRENDA-specific ECs (not in OED)")
    log.info("=" * 60)
    import json as _json
    with open(PROJECT_DIR / "OED" / "oed_kinetics.json") as f:
        oed = _json.load(f)
    oed_ecs = set()
    for entry in oed:
        ec = entry.get("ec", "").strip()
        if ec:
            oed_ecs.add(ec)

    brenda_ecs = set(brenda_map.keys())
    unique_brenda = brenda_ecs - oed_ecs
    log.info(f"OED ECs: {len(oed_ecs)}")
    log.info(f"BRENDA ECs: {len(brenda_ecs)}")
    log.info(f"BRENDA-only ECs: {len(unique_brenda)}")

    log.info("\n✓ BRENDA temperature extraction complete!")


if __name__ == "__main__":
    main()
