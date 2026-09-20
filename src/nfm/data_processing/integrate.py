# -*- coding: utf-8 -*-
"""
integrate.py —— 数据整合(模块二·整合,v0.2)
=============================================
把四个处理器的 interim 结果按 sample_id 合并到样本索引上,生成派生量,
校验后落盘 data/processed/master_table.csv(建模层唯一入口)。

v0.2 整合层新增:
  - hier_size_ratio = D50 / D_pri_sem      (层级结构尺度比,替代旧 agglomeration_index)
  - M_D / M_B / dRho_M                      (同炉差分,见 features.memory_indices)
顺序:先并表 → 加热暴露特征(Θ 需在差分前,因记忆指标按 Θ 排序分析) → 差分。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nfm import schema
from nfm.config import build_sample_index
from nfm.features.thermal_exposure import add_thermal_features
from nfm.features.memory_indices import compute_within_furnace_indices


def _merge(base: pd.DataFrame, interim_dir: Path) -> pd.DataFrame:
    out = base.copy()
    for name in ("psd_features", "sem_features", "xrd_features", "density_features"):
        f = interim_dir / f"{name}.csv"
        if f.exists():
            part = pd.read_csv(f)
            out = out.merge(part, on="sample_id", how="left", suffixes=("", f"_{name}"))
    return out


def build_master_table(cfg, interim_dir="data/interim",
                       out_csv="data/processed/master_table.csv") -> pd.DataFrame:
    interim_dir = Path(interim_dir)
    base = build_sample_index(cfg)             # 81 行 + 工艺/前驱体输入
    df = _merge(base, interim_dir)

    # 派生:层级结构尺度比(非一次颗粒数!)
    if {"D50", "D_pri_sem"}.issubset(df.columns):
        df["hier_size_ratio"] = df["D50"] / df["D_pri_sem"].replace(0, np.nan)

    # 热暴露特征(Θ 等)——差分分析按 Θ 排序,故先算
    df = add_thermal_features(df, cfg)

    # 同炉差分记忆指标
    df = compute_within_furnace_indices(df, cfg)

    # 校验(差分量/WH/Θ 允许缺失;核心测量列必须齐)
    report = schema.validate_master_table(df)
    if not report.ok:
        # 数据未齐时给出软提醒而非硬失败(便于流程演练);正式出图前应 raise
        print("⚠ master_table 校验提示:")
        for p in report.problems:
            print("   -", p)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df
