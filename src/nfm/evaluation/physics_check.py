# -*- coding: utf-8 -*-
"""
physics_check.py —— 物理一致性核查(X.7.5)
===========================================
四项独立检验:
  ① Q_learn 与文献(180–220 kJ/mol)及 X.4 独立动力学拟合 Q 的三方对比;
     判据:Q_learn 是否落在独立实验值的 95% CI 内("学到某 Q" ≠ "测定真实 Q")。
  ② Arrhenius 还原:训练后模型扫描温度轴,检验 ln D–1/T 线性度(R²)。
  ③ 单调性满足率:参数空间密集采样,∂D/∂T、∂D/∂t 符号正确比例。
  ④ 边界合规率:全部预测落于物理范围比例。
"""
from __future__ import annotations

import numpy as np


def check_Q(Q_learn_kJmol, Q_fit_mean_kJmol, Q_fit_ci_kJmol,
            lit_range_kJmol=(180.0, 220.0)):
    lo_ci, hi_ci = Q_fit_ci_kJmol
    return {
        "Q_learn": Q_learn_kJmol,
        "Q_fit_mean": Q_fit_mean_kJmol,
        "Q_fit_95CI": (lo_ci, hi_ci),
        "in_fit_CI": bool(lo_ci <= Q_learn_kJmol <= hi_ci),
        "in_literature": bool(lit_range_kJmol[0] <= Q_learn_kJmol <= lit_range_kJmol[1]),
    }


def arrhenius_recovery(predict_lnD, T_grid_K, fixed_others):
    """固定其它特征,扫描温度,检验 ln D 对 1/T 的线性度。返回斜率、R²。"""
    invT = 1.0 / np.asarray(T_grid_K, float)
    lnD = np.asarray([predict_lnD(T, fixed_others) for T in T_grid_K], float)
    A = np.vstack([invT, np.ones_like(invT)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, lnD, rcond=None)
    yhat = A @ np.array([slope, intercept])
    ss_res = float(np.sum((lnD - yhat) ** 2))
    ss_tot = float(np.sum((lnD - lnD.mean()) ** 2)) + 1e-12
    return {"slope": float(slope), "r2": 1.0 - ss_res / ss_tot,
            "apparent_Q_kJmol": float(-slope * 8.314 / 1e3)}


def monotonicity_rate(predict_D, T_samples, t_samples, others, dT=5.0, dt=0.5):
    """有限差分估 ∂D/∂T、∂D/∂t 的符号正确率。"""
    okT = okt = n = 0
    for T in T_samples:
        for t in t_samples:
            n += 1
            if predict_D(T + dT, t, others) >= predict_D(T - dT, t, others):
                okT += 1
            if predict_D(T, t + dt, others) >= predict_D(T, t - dt, others):
                okt += 1
    return {"dD_dT_ok_rate": okT / max(n, 1), "dD_dt_ok_rate": okt / max(n, 1)}


def bounds_compliance(preds: dict, schema_bounds: dict):
    """preds: {col: array};schema_bounds: {col: (lo,hi)}。返回各列合规率。"""
    out = {}
    for col, arr in preds.items():
        if col in schema_bounds and schema_bounds[col]:
            lo, hi = schema_bounds[col]
            arr = np.asarray(arr, float)
            out[col] = float(np.mean((arr >= lo) & (arr <= hi)))
    return out
