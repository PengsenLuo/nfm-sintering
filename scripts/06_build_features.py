# -*- coding: utf-8 -*-
"""06 物理派生特征自检(Θ 已在 integrate 写入;此处验证并打印 Θ 排序与塌缩用列)。"""
import _bootstrap  # noqa
import pandas as pd
from nfm.config import load_config

if __name__ == "__main__":
    cfg = load_config()
    df = pd.read_csv("data/processed/master_table.csv")
    need = ["T_K", "inv_T_K", "Theta", "t_eff"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        print("⚠ 缺物理派生特征:", miss, "(请先跑 05_integrate.py)")
    else:
        print("✓ 物理派生特征齐全。Θ 范围:",
              round(df.Theta.min(), 3), "–", round(df.Theta.max(), 3), "h")
        print("  提示:X.4 塌缩检验用 Θ 作横轴,D_XRD / D_pri_sem 作纵轴")
