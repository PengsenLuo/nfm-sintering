# -*- coding: utf-8 -*-
"""
memory_indices.py —— 形貌记忆指标(同炉差分,X.2.4 / X.5)
=========================================================
这是新叙事相对旧代码最实质的新增之一:**按 condition_id 分组**,在同一炉次内
对 S/M/L 做差分,得到不受炉次热历史混杂的记忆指标。

  M_D(同炉粒径记忆系数,逐条件,产物激光口径/水测,作分散单元佐证,非主指标):
      M_D = (D50_L^prod - D50_S^prod) / (D50_L^prec - D50_S^prec)

  M_D_agg(同炉团聚体记忆系数,逐条件,免水主指标):
      M_D_agg = (D_sec_L^prod - D_sec_S^prod) / (D50_L^prec - D50_S^prec)
      分子为 SEM 500× 团聚体 D_sec(干态免水),分母与 M_D 相同,仍用前驱体激光 D50
      (前驱体不与水反应,可信)。二者分子口径不同,不应互相替代对照。

  M_B(双峰保留指数,仅 M 组):
      M_B = B_bimodal(M 产物) / B_bimodal(M 前驱体)

  dRho_M(双峰压实净增益,仅 M 组,参考值用压实密度):
      dRho_M = ρ_M - (w_S·ρ_S + w_L·ρ_L)

记忆衰减拟合(对 M_D vs Θ):
      M(Θ) = 1 / (1 + exp[a·(lnΘ - lnΘ_c)])
给出衰减陡度 a 与临界热暴露量 Θ_c。
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def compute_within_furnace_indices(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """对每个 condition_id 计算 M_D / M_D_agg / dRho_M / M_B,写回对应行。

    约定:M_D、M_D_agg 标注在该条件的三行上(全条件共享一个值);M_B、dRho_M 仅 M 行有值。
    需要列:condition_id, precursor, D50, D_sec, compaction_density, B_bimodal,
            precursor_D50(前驱体 D50,用于 M_D/M_D_agg 分母与前驱体 B)。
    """
    out = df.copy()
    for col in ("M_D", "M_D_agg", "M_B", "dRho_M"):
        if col not in out.columns:
            out[col] = np.nan
    if "D_sec" not in out.columns:
        out["D_sec"] = np.nan
    if "M_B_note" not in out.columns:
        out["M_B_note"] = None

    desc = cfg.raw["design"]["precursor_desc"]
    w_small = desc["M"]["blend_ratio"]          # 小颗粒质量分数 0.3
    w_large = 1.0 - w_small

    # 前驱体双峰特征量 B_prec(由前驱体激光粒度拟合得到;此处从配置/中间表读取)
    # 若无单独前驱体测量,退化为对 M 组以"产物/初始投料理论双峰"归一,见注释。
    B_prec_M = cfg.raw["design"].get("precursor_B", {}).get("M", np.nan)
    UNIMODAL_NOTES = {"unimodal_by_modecount", "degenerate_unimodal"}

    for cid, g in out.groupby("condition_id"):
        rows = {r.precursor: r for r in g.itertuples()}
        if {"S", "L"}.issubset(rows):
            den = rows["L"].precursor_D50 - rows["S"].precursor_D50

            num = rows["L"].D50 - rows["S"].D50
            m_d = num / den if abs(den) > 1e-9 else np.nan
            out.loc[out.condition_id == cid, "M_D"] = m_d

            d_sec_s, d_sec_l = rows["S"].D_sec, rows["L"].D_sec
            if pd.notna(d_sec_s) and pd.notna(d_sec_l) and abs(den) > 1e-9:
                out.loc[out.condition_id == cid, "M_D_agg"] = (d_sec_l - d_sec_s) / den

        if "M" in rows and {"S", "L"}.issubset(rows):
            rho_ref = w_small * rows["S"].compaction_density + \
                      w_large * rows["L"].compaction_density
            idx_M = (out.condition_id == cid) & (out.precursor == "M")
            out.loc[idx_M, "dRho_M"] = rows["M"].compaction_density - rho_ref

            m_row = rows["M"]
            m_note = getattr(m_row, "note", None)
            if m_note in UNIMODAL_NOTES:
                # 双峰记忆完全擦除的物理端点:产物一级判据确认单峰,比值分子为0,
                # 不依赖 B_prec_M 是否配置(分母非零即可,前驱体M按设计是真双峰)。
                out.loc[idx_M, "M_B"] = 0.0
                out.loc[idx_M, "M_B_note"] = "erased_endpoint"
            elif not np.isnan(B_prec_M) and abs(B_prec_M) > 1e-9 and not np.isnan(m_row.B_bimodal):
                out.loc[idx_M, "M_B"] = m_row.B_bimodal / B_prec_M
                out.loc[idx_M, "M_B_note"] = "computed"
            elif pd.isna(m_note):
                # M 产物样本尚无任何 PSD 测量(如尚未到样),而非"已测出单峰"或
                # "已测出双峰但分母缺配置"——单独标注,避免被误读为 M_B 是真实比值。
                out.loc[idx_M, "M_B_note"] = "no_data"

    m_rows = out["precursor"] == "M"
    if m_rows.any() and out.loc[m_rows, "M_B"].isna().all():
        warnings.warn(
            "M 组 M_B 全部缺失:config.yaml 的 design.precursor_B 未配置"
            "(前驱体 M 自身的双峰特征量 B_prec 无来源)。")
    return out


# ---------------------------------------------------------------------
# 记忆衰减拟合 M(Θ)
# ---------------------------------------------------------------------
def _sigmoid_decay(lnTheta, a, lnTheta_c):
    return 1.0 / (1.0 + np.exp(a * (lnTheta - lnTheta_c)))


def fit_memory_decay(theta: np.ndarray, memory: np.ndarray):
    """拟合 M(Θ)=1/(1+exp[a(lnΘ-lnΘc)]),返回 (a, Theta_c, r2, predict_fn)。

    用于 M_D(Θ) 或 M_B(Θ)。需要 >= 4 个有效点。
    """
    theta = np.asarray(theta, float)
    memory = np.asarray(memory, float)
    mask = np.isfinite(theta) & np.isfinite(memory) & (theta > 0)
    x, y = np.log(theta[mask]), memory[mask]
    if mask.sum() < 4:
        return dict(a=np.nan, Theta_c=np.nan, r2=np.nan, predict=None,
                    note="有效点不足(<4),无法拟合 S 形衰减")

    p0 = [2.0, np.median(x)]
    try:
        popt, _ = curve_fit(_sigmoid_decay, x, y, p0=p0, maxfev=20000)
    except Exception as e:  # noqa
        return dict(a=np.nan, Theta_c=np.nan, r2=np.nan, predict=None,
                    note=f"拟合失败:{e}")

    yhat = _sigmoid_decay(x, *popt)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot

    def predict(theta_new):
        return _sigmoid_decay(np.log(np.asarray(theta_new, float)), *popt)

    return dict(a=float(popt[0]), Theta_c=float(np.exp(popt[1])),
                r2=r2, predict=predict, note="ok")
