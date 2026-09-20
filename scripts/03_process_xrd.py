# -*- coding: utf-8 -*-
"""03 XRD → D_XRD + Williamson–Hall + 晶格参数。"""
import _bootstrap  # noqa
from nfm.config import load_config
from nfm.data_processing import xrd_processor

if __name__ == "__main__":
    cfg = load_config()
    out = xrd_processor.process_all("data/raw/xrd",
                                    "data/interim/xrd_features.csv", cfg)
    print(f"[done] XRD 处理完成,{len(out)} 样本 -> data/interim/xrd_features.csv")
    print("  列:", list(out.columns))
