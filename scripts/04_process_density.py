# -*- coding: utf-8 -*-
"""04 压实密度(单点;多压力可选)。"""
import _bootstrap  # noqa
from pathlib import Path
from nfm.config import load_config
from nfm.data_processing import density_processor

if __name__ == "__main__":
    cfg = load_config()
    raw = "data/raw/density/compact_raw.csv"
    if Path(raw).exists():
        out = density_processor.load_single_point(raw, "data/interim/density_features.csv")
        print(f"✓ 压实密度(单点){len(out)} 样本")
    else:
        print(f"⚠ 未找到 {raw};请放入手工录入表(列 sample_id, mass_g, volume_cm3)")
    if cfg.raw["instruments"]["compaction_density"]["multi_pressure"]:
        density_processor.process_multi_pressure(
            "data/raw/density_multiP", "data/interim/density_multiP.csv")
        print("✓ 多压力 Heckel/Kawakita 完成(可选扩展)")
