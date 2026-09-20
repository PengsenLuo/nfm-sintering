# -*- coding: utf-8 -*-
"""
state_map.py —— 形貌记忆状态图 / 工艺窗口 / 逆向设计(X.7.6)
============================================================
模型最终产出不只预测等高线,而是具物理意义的**反应烧结形貌记忆状态图**:
在 (T, β, t) 或 Θ 空间圈定——
  记忆保留区 / 部分重构区 / 记忆擦除区 / 双峰有效区 / 双峰失效区 /
  过烧结(二次颗粒合并)区 / 高压实-合理动力学 Pareto 区。
并在其上构建多目标工艺窗口、做受约束逆向设计(给定目标压实→反推工艺)。

状态边界由前面的物理量定义(可学习/可标定):
  - 记忆区:M_D(Θ) 阈值(如 M_D>0.7 保留 / 0.3–0.7 部分 / <0.3 擦除);
  - 双峰区:M_B(Θ) 阈值(如 M_B>0.5 有效);
  - 过烧结区:D_pri_sem 或 hier_size_ratio 拐点。
本模块把这些阈值参数化,叠加模型预测生成分区与窗口。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def classify_state(M_D, M_B, *, md_keep=0.7, md_erase=0.3, mb_eff=0.5):
    """按记忆指标给单点分配形貌记忆状态标签。"""
    if M_D is not None and np.isfinite(M_D):
        if M_D >= md_keep:
            base = "memory_retained"
        elif M_D <= md_erase:
            base = "memory_erased"
        else:
            base = "partial_reconstruction"
    else:
        base = "unknown"
    bimodal = "bimodal_effective" if (np.isfinite(M_B) and M_B >= mb_eff) \
        else "bimodal_failed"
    return base, bimodal


def build_state_grid(predict_indices, T_range, t_range, beta_fixed,
                     thresholds=None):
    """在 (T,t) 网格上预测 M_D/M_B 并分区。

    predict_indices(T,beta,t) -> (M_D, M_B);返回 DataFrame(T,t,M_D,M_B,state,bimodal)。
    """
    th = thresholds or {}
    rows = []
    for T in T_range:
        for t in t_range:
            M_D, M_B = predict_indices(T, beta_fixed, t)
            base, bim = classify_state(M_D, M_B, **th)
            rows.append(dict(T_C=T, t_hold=t, beta=beta_fixed,
                             M_D=M_D, M_B=M_B, state=base, bimodal=bim))
    return pd.DataFrame(rows)


def process_window(predict_targets, T_range, beta_range, t_range,
                   constraints):
    """筛选满足多目标约束的工艺点。

    predict_targets(T,beta,t) -> dict(目标名: 值);
    constraints: dict(目标名: (op, threshold)),op ∈ {">=","<="}。
    返回满足全部约束的工艺点 DataFrame。
    """
    def ok(vals):
        for name, (op, thr) in constraints.items():
            v = vals.get(name, np.nan)
            if op == ">=" and not (v >= thr):
                return False
            if op == "<=" and not (v <= thr):
                return False
        return True

    rows = []
    for T in T_range:
        for b in beta_range:
            for t in t_range:
                vals = predict_targets(T, b, t)
                if ok(vals):
                    rows.append({"T_C": T, "beta": b, "t_hold": t, **vals})
    return pd.DataFrame(rows)


def inverse_design(predict_targets, target_density, *,
                   T_bounds=(850, 950), beta_bounds=(2, 8), t_bounds=(10, 20),
                   prefer_low_cost=True, n_restarts=12, seed=42):
    """给定目标压实密度 → 反推工艺(参数物理域内约束优化)。

    目标:最小化 |ρ_pred - target| (+ 低温短时成本偏好)。
    返回 Pareto 候选(按目标偏差排序)。用全局多起点 + 局部精修。
    """
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    lo = np.array([T_bounds[0], beta_bounds[0], t_bounds[0]])
    hi = np.array([T_bounds[1], beta_bounds[1], t_bounds[1]])

    def cost(x):
        T, b, t = x
        rho = predict_targets(T, b, t).get("compaction_density", np.nan)
        err = (rho - target_density) ** 2
        if prefer_low_cost:
            # 轻微偏好低温短时(归一化后加权)
            pen = 1e-3 * (((T - lo[0]) / (hi[0] - lo[0]))
                          + ((t - lo[2]) / (hi[2] - lo[2])))
            err = err + pen
        return err

    cands = []
    for _ in range(n_restarts):
        x0 = lo + rng.random(3) * (hi - lo)
        res = minimize(cost, x0, method="L-BFGS-B",
                       bounds=list(zip(lo, hi)))
        T, b, t = res.x
        rho = predict_targets(T, b, t).get("compaction_density", np.nan)
        cands.append({"T_C": round(T, 1), "beta": round(b, 2),
                      "t_hold": round(t, 1), "pred_density": float(rho),
                      "abs_err": abs(float(rho) - target_density)})
    out = pd.DataFrame(cands).sort_values("abs_err").drop_duplicates(
        subset=["T_C", "beta", "t_hold"]).reset_index(drop=True)
    return out
