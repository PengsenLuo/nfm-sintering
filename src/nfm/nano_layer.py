# -*- coding: utf-8 -*-
"""
nano_layer.py —— 纳米层(D_XRD/晶格参数)唯一合法取数入口
==========================================================
仪器1(SmartLab,51样)与仪器2(MiniFlex,30样)在扣除 lnΘ+precursor 后的 D_XRD
残差中位数分别为 +4.41nm / -8.61nm(Mann-Whitney p=1.1e-4),lattice_a/lattice_c/
c_a_ratio 同样系统性分裂,量级与被研究的工艺效应相当,且与 t_hold/beta 混杂
——跨仪器合并做纳米层分析不可辩护。仪器2 的 30 行数据保留在 xrd_features.csv/
master_table.csv 中供审计,但任何纳米层的分析/作图/建模只能通过本函数取数。
"""
from __future__ import annotations

import warnings

import pandas as pd

NANO_COLS = ("D_XRD", "lattice_a", "lattice_c", "c_a_ratio")


def nano_layer_frame(df: pd.DataFrame, require_reliable: bool = True) -> pd.DataFrame:
    """纳米层唯一合法取数入口:仅仪器1;可选剔除分辨率不可信样本。
    跨仪器合并已被证伪(见本文件顶部模块 docstring)。
    """
    out = df[df["xrd_instrument"] == 1]
    if require_reliable:
        out = out[out["D_XRD_reliable"] == True]  # noqa: E712
    warnings.warn(f"nano_layer_frame: 纳米层取数 n={len(out)}"
                 f"(仅 xrd_instrument==1{' 且 D_XRD_reliable==True' if require_reliable else ''})")
    return out
