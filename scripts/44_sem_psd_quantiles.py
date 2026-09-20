# -*- coding: utf-8 -*-
"""44 产物粒度改 SEM 口径
====================================================================
从 data/interim/sem_shape_descriptors.csv(83904 行逐颗粒记录,81 样全在)
逐样本算数量加权与体积加权两套 D10/D50/D90/Span,不引入任何新的粒度测量
口径——所有原始数字仍来自 500x SEM 分割结果(sem_processor.py 已产出的
d_um 列),本脚本只做统计口径的重新汇总。

先核对:该文件的 d_um 与 sem_processor.py._aggregate_secondary 计算
D_sec 所用的同一批直径是否为同一批(逐样本 median(d_um) 应与
master_table.csv 的 D_sec 一致,若最大绝对差超过 0.01 um 立即停止——
说明两条路径的颗粒集合不同,产物粒度的"改口径"结论就立不住。
"""
import _bootstrap  # noqa

from pathlib import Path

import numpy as np
import pandas as pd

SHAPE_CSV = Path("data/interim/sem_shape_descriptors.csv")
MASTER_TABLE = Path("data/processed/master_table.csv")
OUT_CSV = Path("data/interim/sem_psd_quantiles.csv")
D_SEC_TOLERANCE = 0.01  # um


def numeric_quantiles(d: np.ndarray) -> dict:
    d = np.asarray(d, dtype=float)
    q10, q50, q90 = np.percentile(d, [10, 50, 90])
    return {
        "n_particles": int(len(d)),
        "D10_num": float(q10), "D50_num": float(q50), "D90_num": float(q90),
        "Span_num": float((q90 - q10) / q50),
    }


def volume_weighted_quantiles(d: np.ndarray) -> dict:
    d = np.sort(np.asarray(d, dtype=float))
    w = d ** 3
    cw = np.cumsum(w) / w.sum()
    q10, q50, q90 = np.interp([0.10, 0.50, 0.90], cw, d)
    return {
        "D10_vol": float(q10), "D50_vol": float(q50), "D90_vol": float(q90),
        "Span_vol": float((q90 - q10) / q50),
    }


def n_carry(d: np.ndarray, frac: float) -> int:
    """Fewest largest-first particles whose cumulative d**3 first reaches
    `frac` of total volume -- a diagnostic for how few particles dominate
    the volume-weighted quantiles above."""
    d = np.sort(np.asarray(d, dtype=float))[::-1]
    w = d ** 3
    cw = np.cumsum(w) / w.sum()
    return int(np.searchsorted(cw, frac) + 1)


def _check_dsec_agreement(shape_df: pd.DataFrame, master_df: pd.DataFrame) -> None:
    med = shape_df.groupby("sample_id")["d_um"].median()
    m = master_df.set_index("sample_id")["D_sec"]
    diff = (med - m.reindex(med.index)).abs()
    max_diff = float(diff.max())
    print(f"[check] median(d_um) vs master_table.D_sec: max abs diff = {max_diff:.6f} um")
    assert max_diff <= D_SEC_TOLERANCE, (
        f"median(d_um) diverges from D_sec by up to {max_diff:.4f} um "
        f"(tolerance {D_SEC_TOLERANCE}) -- the two pipelines are reading "
        f"different particle sets; STOP, do not proceed with the SEM-caliber "
        f"switch until this is resolved."
    )


def main():
    shape_df = pd.read_csv(SHAPE_CSV)
    master_df = pd.read_csv(MASTER_TABLE)
    assert shape_df["sample_id"].nunique() == 81, (
        f"expected 81 samples in {SHAPE_CSV}, got {shape_df['sample_id'].nunique()}"
    )
    _check_dsec_agreement(shape_df, master_df)

    rows = []
    for sid, g in shape_df.groupby("sample_id"):
        d = g["d_um"].to_numpy()
        rec = {"sample_id": sid, "precursor": g["precursor"].iloc[0]}
        rec.update(numeric_quantiles(d))
        rec.update(volume_weighted_quantiles(d))
        rec["n_carry50vol"] = n_carry(d, 0.50)
        rec["n_carry90vol"] = n_carry(d, 0.90)
        rows.append(rec)
    out = pd.DataFrame(rows).merge(
        master_df[["sample_id", "condition_id", "Theta"]], on="sample_id", how="left"
    )
    out = out[[
        "sample_id", "condition_id", "precursor", "Theta", "n_particles",
        "D10_num", "D50_num", "D90_num", "Span_num",
        "D10_vol", "D50_vol", "D90_vol", "Span_vol",
        "n_carry50vol", "n_carry90vol",
    ]].sort_values("sample_id")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"[done] {OUT_CSV} ({len(out)} rows)")


if __name__ == "__main__":
    main()
