# -*- coding: utf-8 -*-
"""
thermal_exposure.py —— 归一化热暴露量 Θ (X.2.4 / X.4)
=====================================================
Θ(Q) = ∫₀^t_end exp[ -(Q/R)·(1/T(τ) - 1/T_ref) ] dτ

把 (T, β, t) 三个工艺变量统一到同一动力学标尺。温度程序按两段构造:
  段1(预烧,所有样本一致):室温→550℃,550℃ 保温 t_pre(对比中可并入或忽略);
  段2(变量段):550℃ →(速率 β)→ 目标 T → 保温 t_hold。
积分起点 integrate_from_C:
  - 数值(如 700):从该温度起算(物理锚定,推荐由双端点预实验反推);
  - "auto":暂用 550℃ 起算并打标记,待标定后回填。

输出 Θ 单位为小时(与保温段一致),作为 schema 的 INPUT_DERIVED 列 `Theta`。
同时给出中间量 t_eff(升温段 Arrhenius 折算到目标温度的等效保温时间)。
"""
from __future__ import annotations

import numpy as np

# NumPy 2.0 将 trapz 更名为 trapezoid;兼容两版
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def _temperature_program(T_target_C, beta_C_per_min, t_hold_h,
                         start_C, n_ramp=400, n_hold=200):
    """构造 [start_C → T_target] 升温 + 保温 的离散温度程序 (秒, ℃)。"""
    beta_C_per_s = beta_C_per_min / 60.0
    ramp_seconds = max((T_target_C - start_C) / beta_C_per_s, 1e-9)
    t_ramp = np.linspace(0.0, ramp_seconds, n_ramp)
    T_ramp = start_C + beta_C_per_s * t_ramp

    hold_seconds = t_hold_h * 3600.0
    t_hold = np.linspace(0.0, hold_seconds, n_hold) + ramp_seconds
    T_hold = np.full(n_hold, float(T_target_C))

    t = np.concatenate([t_ramp, t_hold])
    T = np.concatenate([T_ramp, T_hold])
    return t, T  # seconds, degC


def theta(T_target_C, beta_C_per_min, t_hold_h, *,
          Q_J, R, T_ref_C, start_C):
    """计算单个样本的归一化热暴露量 Θ(单位:小时)。"""
    t_s, T_C = _temperature_program(T_target_C, beta_C_per_min, t_hold_h, start_C)
    T_K = T_C + 273.15
    T_ref_K = T_ref_C + 273.15
    integrand = np.exp(-(Q_J / R) * (1.0 / T_K - 1.0 / T_ref_K))
    theta_seconds = _trapz(integrand, t_s)
    return theta_seconds / 3600.0


def t_eff_arrhenius_ramp(T_target_C, beta_C_per_min, t_hold_h, *,
                         Q_J, R, start_C):
    """有效时间 = 保温 + 升温段按 Arrhenius 折算到目标温度的等效保温(小时)。"""
    t_s, T_C = _temperature_program(T_target_C, beta_C_per_min, t_hold_h, start_C)
    T_K = T_C + 273.15
    Tt_K = T_target_C + 273.15
    # 升温段(去掉保温平台后)的折算
    ramp_mask = T_C < (T_target_C - 1e-6)
    w = np.exp(-(Q_J / R) * (1.0 / T_K[ramp_mask] - 1.0 / Tt_K))
    ramp_equiv_s = _trapz(w, t_s[ramp_mask]) if ramp_mask.sum() > 1 else 0.0
    return t_hold_h + ramp_equiv_s / 3600.0


def add_thermal_features(df, cfg):
    """对 master_table 加入 T_K / inv_T_K / Theta / t_eff。返回新 df(不改原表)。"""
    out = df.copy()
    Q_J = cfg.Q_ref_J
    R = cfg.R
    T_ref_C = cfg.raw["physical_constants"]["T_ref_C"]
    ifrom = cfg.raw["thermal_exposure"]["integrate_from_C"]
    start_C = 550.0 if ifrom == "auto" else float(ifrom)

    out["T_K"] = out["T_C"] + 273.15
    out["inv_T_K"] = 1.0 / out["T_K"]
    out["Theta"] = [
        theta(r.T_C, r.beta, r.t_hold, Q_J=Q_J, R=R, T_ref_C=T_ref_C, start_C=start_C)
        for r in out.itertuples()
    ]
    out["t_eff"] = [
        t_eff_arrhenius_ramp(r.T_C, r.beta, r.t_hold, Q_J=Q_J, R=R, start_C=start_C)
        for r in out.itertuples()
    ]
    return out
