# -*- coding: utf-8 -*-
"""
density_processor.py —— 压实密度处理器(模块二·处理器 4,v0.2)
==============================================================
主线:全部样品的**单点压实密度**(手工录入汇总表 → 标准化)。
本研究不测 BET、不测振实密度;堆积层只保留压实密度。

可选扩展(config.instruments.compaction_density.multi_pressure=true):
  对 6–9 个代表样的多压力 ρ(P) 作 Heckel / Kawakita 拟合,
  把压实密度差异细化为填隙/重排/破碎机制(X.9 第 1 项,性价比最高)。
  默认关闭——主线结论不依赖于此。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_single_point(raw_csv, out_csv):
    """读取手工录入表(列:sample_id, mass_g, volume_cm3 或直接 compaction_density)。"""
    df = pd.read_csv(raw_csv)
    if "compaction_density" not in df.columns:
        df["compaction_density"] = df["mass_g"] / df["volume_cm3"]
    out = df[["sample_id", "compaction_density"]].copy()
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


# ---------------------------------------------------------------------
# 可选:多压力机制分解
# ---------------------------------------------------------------------
def heckel_fit(P_MPa, rho_rel):
    """Heckel:ln(1/(1-D)) = k·P + A。返回 k(屈服强度倒数相关)与 A(重排项)。"""
    P = np.asarray(P_MPa, float)
    D = np.asarray(rho_rel, float)
    y = np.log(1.0 / (1.0 - np.clip(D, 1e-6, 1 - 1e-6)))
    A_mat = np.vstack([P, np.ones_like(P)]).T
    (k, A), *_ = np.linalg.lstsq(A_mat, y, rcond=None)
    return dict(heckel_k=float(k), heckel_A=float(A),
                yield_strength_MPa=float(1.0 / (3 * k)) if k > 0 else np.nan)


def kawakita_fit(P_MPa, C):
    """Kawakita:P/C = P/a + 1/(a·b)。C 为体积减小率。返回 a(最大可压缩)与 b。"""
    P = np.asarray(P_MPa, float)
    C = np.asarray(C, float)
    y = P / np.clip(C, 1e-6, None)
    A_mat = np.vstack([P, np.ones_like(P)]).T
    (inv_a, inv_ab), *_ = np.linalg.lstsq(A_mat, y, rcond=None)
    a = 1.0 / inv_a if inv_a > 0 else np.nan
    b = inv_a / inv_ab if inv_ab > 0 else np.nan
    return dict(kawakita_a=float(a), kawakita_b=float(b))


def process_multi_pressure(raw_dir, out_csv):
    """对代表样的多压力曲线作 Heckel/Kawakita;raw_dir 下每样一 csv(P_MPa, rho_rel, C)。"""
    raw_dir = Path(raw_dir)
    rows = []
    for f in sorted(raw_dir.glob("*.csv")):
        df = pd.read_csv(f)
        rec = {"sample_id": f.stem}
        rec.update(heckel_fit(df["P_MPa"], df["rho_rel"]))
        if "C" in df.columns:
            rec.update(kawakita_fit(df["P_MPa"], df["C"]))
        rows.append(rec)
    out = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out
