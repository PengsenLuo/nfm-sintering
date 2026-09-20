# -*- coding: utf-8 -*-
"""
losses.py —— 物理约束损失(X.7.3)
==================================
L = L_data + λ1·L_Arrhenius + λ2·L_mono + λ3·L_bound

- L_data:多目标 MSE。
- L_Arrhenius:对 D_XRD / D_pri_sem 的预测应满足生长动力学
    Dⁿ - D₀ⁿ = k₀·t·exp(-Q/RT)  ⇒  在 ln 空间约束 ln D 与 (1/T, ln t) 的关系。
    Q、k₀ 可学习(以 X.4 独立拟合 Q 作先验/软约束),n 可固定或离散比较。
- L_mono:∂D/∂T ≥ 0、∂D/∂t ≥ 0 的铰链惩罚(自动微分)。
- L_bound:越出物理范围的铰链惩罚。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

R_GAS = 8.314


def data_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None):
    if mask is None:
        return F.mse_loss(pred, target)
    diff = (pred - target) ** 2
    diff = diff * mask
    return diff.sum() / mask.sum().clamp_min(1.0)


def arrhenius_loss(lnD_pred: torch.Tensor, T_K: torch.Tensor, t_h: torch.Tensor,
                   Q_J: torch.Tensor, lnk0: torch.Tensor, n: float):
    """软约束:n·lnD ≈ ln(k0·t) - Q/(R T) (忽略 D0 的高温近似)。
    返回残差 MSE;Q_J、lnk0 为可学习标量参数。
    """
    target = (lnk0 + torch.log(t_h.clamp_min(1e-6)) - Q_J / (R_GAS * T_K))
    return F.mse_loss(n * lnD_pred, target)


def monotonic_loss(model, X_samp: torch.Tensor, idx_T: int, idx_t: int,
                   out_index_D: int):
    """对采样点施加 ∂D/∂T≥0、∂D/∂t≥0 的铰链惩罚(自动微分)。

    X_samp 需 requires_grad=True;out_index_D 为被约束的尺寸型输出列。
    """
    X_samp = X_samp.clone().requires_grad_(True)
    y = model(X_samp)[:, out_index_D]
    grad = torch.autograd.grad(y.sum(), X_samp, create_graph=True)[0]
    dD_dT = grad[:, idx_T]
    dD_dt = grad[:, idx_t]
    return (F.relu(-dD_dT).mean() + F.relu(-dD_dt).mean())


def bounds_loss(pred: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor):
    """越界铰链惩罚:pred<lower 或 pred>upper。"""
    return (F.relu(lower - pred).mean() + F.relu(pred - upper).mean())


def total_loss(pred, target, *, weights, arr_kwargs=None, mono_kwargs=None,
               bound_kwargs=None, mask=None):
    parts = {"data": data_loss(pred, target, mask)}
    parts["arrhenius"] = arrhenius_loss(**arr_kwargs) if arr_kwargs else torch.tensor(0.0)
    parts["monotonic"] = monotonic_loss(**mono_kwargs) if mono_kwargs else torch.tensor(0.0)
    parts["bounds"] = bounds_loss(**bound_kwargs) if bound_kwargs else torch.tensor(0.0)
    total = (weights["data"] * parts["data"]
             + weights["arrhenius"] * parts["arrhenius"]
             + weights["monotonic"] * parts["monotonic"]
             + weights["bounds"] * parts["bounds"])
    return total, parts
