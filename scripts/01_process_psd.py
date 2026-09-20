# -*- coding: utf-8 -*-
"""01 激光粒度 → D 值/Span + 双峰拟合(B_bimodal)。"""
import _bootstrap  # noqa
from nfm.config import load_config
from nfm.data_processing import psd_processor

if __name__ == "__main__":
    cfg = load_config()
    out = psd_processor.process_all("data/raw/psd/samples",
                                    "data/interim/psd_features.csv", cfg)
    print(f"✓ PSD 处理完成,{len(out)} 样本 → data/interim/psd_features.csv")
    print("  列:", list(out.columns))
