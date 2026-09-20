# -*- coding: utf-8 -*-
"""05 合并四处理器 + 同炉差分(M_D/Δρ_M/M_B/hier_size_ratio)→ master_table。"""
import _bootstrap  # noqa
from nfm.config import load_config
from nfm.data_processing.integrate import build_master_table

if __name__ == "__main__":
    cfg = load_config()
    df = build_master_table(cfg)
    print(f"✓ master_table:{df.shape[0]} 行 × {df.shape[1]} 列 "
          f"→ data/processed/master_table.csv")
    for c in ("Theta", "M_D", "M_B", "dRho_M", "hier_size_ratio"):
        present = c in df.columns
        print(f"   {c}: {'有' if present else '缺'}")
