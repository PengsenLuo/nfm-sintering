# -*- coding: utf-8 -*-
"""
pinn.py —— 物理约束多任务模型(X.7.3)
======================================
共享主干—多输出头(全输出共享表征 = 多任务隐式正则),
可学习 Arrhenius 参数 Q、k₀(初始化于 Q_ref;最终与独立拟合 Q 比对作一致性核查)。

术语上属 physics-informed / physics-guided / physics-constrained ML;
论文标题不必出现 "PINN"(把读者注意力引向材料规律而非算法)。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from nfm import schema
from nfm.models.losses import arrhenius_loss, bounds_loss, data_loss, monotonic_loss

# 反向/无先验目标:热重构使其随 Θ 下降或非单调,强加 ∂D/∂T≥0、∂D/∂t≥0
# 会把误导性趋势硬编码进模型。
FORBIDDEN_MONO_TARGETS = {"circularity", "convexity", "compaction_density"}


class SharedTrunkMultiHead(nn.Module):
    def __init__(self, in_dim: int, target_names: list[str],
                 hidden_dims=(64, 32, 16), dropout=0.2,
                 Q_init_kJmol=200.0, lnk0_init=20.0):
        super().__init__()
        self.target_names = list(target_names)
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleDict({name: nn.Linear(d, 1) for name in self.target_names})
        # 可学习动力学参数(标量),以 Q_ref 初始化。
        # ★ 存成 kJ/mol(~200)而非 J/mol(~2e5):后者在生产 lr=1e-3 下,Adam 单步
        # 位移量级(~lr)小于 float32 在 2e5 量级的 ULP(~0.0156),`param-update`
        # 会被舍入直接抹掉——梯度非零但更新存不下来,表现为"Q 永远等于初始化值"
        # (2026-07-26 真实数据 27 折 LOCO 复现:learned Q 精确等于 200.0000,
        # |Δ|=0.0000)。kJ/mol 量级下 ULP 细得多,不会有这个问题。
        self.Q_kJmol = nn.Parameter(torch.tensor(Q_init_kJmol, dtype=torch.float32))
        self.lnk0 = nn.Parameter(torch.tensor(lnk0_init, dtype=torch.float32))

    @property
    def Q_J(self):
        """物理量纲(J/mol),供 arrhenius_loss 的 Q_J/(R·T_K) 使用;可微。"""
        return self.Q_kJmol * 1e3

    def forward(self, x):
        z = self.trunk(x)
        outs = [self.heads[name](z) for name in self.target_names]
        return torch.cat(outs, dim=1)

    def head_index(self, name: str) -> int:
        return self.target_names.index(name)


def _standardize_columns(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐列 nan-aware mean/std(std 下限 1e-6,防止常量列除零)。"""
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _mask_targets(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把 target 里的 NaN 替换为 0,并返回逐元素 mask(1=有效/0=缺失)。

    data_loss 用 mask 加权求和,故 NaN 必须先置零,否则 NaN*0 仍是 NaN。
    """
    Y = np.asarray(Y, dtype=np.float64)
    mask = (~np.isnan(Y)).astype(np.float32)
    Y_filled = np.nan_to_num(Y, nan=0.0).astype(np.float32)
    return Y_filled, mask


def _physics_scale(epoch: int, warmup_epochs: int) -> float:
    """物理约束热身:0..warmup_epochs 线性从 ~0 爬升到 1,之后恒为 1。"""
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, (epoch + 1) / warmup_epochs)


def _resolve_physics_targets(target_names, physics_cfg):
    """按配置把 arrhenius/mono/bounds 三类约束分派到具体 target 列。

    - 列表:与 target_names 取交集(配置里写了但当前没训练的目标直接忽略)。
    - "all":展开为全部 target_names(bounds_targets 常用)。
    - mono_targets 命中 FORBIDDEN_MONO_TARGETS 直接报错——这是硬约束,
      不是可静默降级的配置项(circularity/convexity/compaction_density
      对 Θ 反向或非单调,不能被单调性铰链惩罚硬编码方向)。
    """
    def _select(key):
        cfg_val = physics_cfg.get(key, [])
        if cfg_val == "all":
            return list(target_names)
        return [n for n in cfg_val if n in target_names]

    arr_names = _select("arrhenius_targets")
    mono_names = _select("mono_targets")
    forbidden = FORBIDDEN_MONO_TARGETS & set(mono_names)
    if forbidden:
        raise ValueError(
            f"mono_targets 不得包含 {sorted(forbidden)}:这些目标随 Θ 下降或非单调"
            "(热重构而非生长动力学主导),强加单调性约束会把误导性趋势硬编码进模型。"
            "见 configs/config.yaml 的 training.pinn.physics 配置块。")

    bounds_cfg = physics_cfg.get("bounds_targets", [])
    candidates = list(target_names) if bounds_cfg == "all" else \
        [n for n in bounds_cfg if n in target_names]
    bounds_names = [n for n in candidates if n in schema.BY_NAME and schema.BY_NAME[n].bounds]
    return arr_names, mono_names, bounds_names


class TorchSklearnWrapper:
    """把上面模型包成 fit/predict 接口,便于在 cross_validation 内复用。

    fit() 接入 losses.py 的 total_loss:data 项用 NaN mask(缺失 target 的样本
    只跳过该 head 的数据项,不影响其他 head,也不影响该样本的 mono/bounds 项);
    arrhenius/monotonic/bounds 三项按 cfg_pinn["physics"] 分派到具体 target 列,
    在全部样本上计算(不受 data mask 限制——物理约束是模型自身预测的自洽性检验,
    与该样本是否有 ground truth 无关)。

    ★ X/Y 标准化(fold 内部 fit,无泄漏):共享主干同时预测 7 个量纲、量级迥异
    的 target(D_XRD~20-90nm、lattice_c~15.8-16.4Å、circularity~0-1、
    compaction_density~2-4g/cm3……),不标准化会让窄范围 target(如 lattice_c)
    的梯度被大尺度 target/输入淹没,训出 R²=-40000 量级的坏解(2026-07-26 真实
    数据跑分复现过)。trunk 前向用标准化后的 Xs、data_loss 用标准化后的 Yn;
    但 arrhenius_loss/bounds_loss 需要物理量纲(Q_J/(R·T_K) 的 T_K 必须是开尔文,
    不能是 z-score;schema bounds 是物理范围),所以在把预测值喂给这两项之前先
    做可微的逆变换 pred_phys = pred_norm*y_std+y_mean;monotonic_loss 只看梯度
    符号,y_std>0 时符号不受仿射变换影响,直接用标准化空间计算即可。
    """
    def __init__(self, in_dim, target_names, cfg_pinn, X_cols, seed=42):
        self.in_dim = in_dim
        self.target_names = list(target_names)
        self.cfg = cfg_pinn
        self.X_cols = list(X_cols)
        self.seed = seed
        self.model = None
        self.loss_history: list[dict] = []

    def fit(self, X, Y):
        torch.manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        x_mean, x_std = _standardize_columns(X)
        y_mean, y_std = _standardize_columns(Y)
        self._x_mean, self._x_std = x_mean, x_std
        self._y_mean, self._y_std = y_mean, y_std

        Xs = ((X - x_mean) / x_std).astype(np.float32)
        Yn = ((Y - y_mean) / y_std)
        Yn_filled, mask_np = _mask_targets(Yn)

        self.model = SharedTrunkMultiHead(
            self.in_dim, self.target_names,
            hidden_dims=tuple(self.cfg["hidden_dims"]),
            dropout=self.cfg["dropout"],
            Q_init_kJmol=self.cfg["Q_init_kJmol"])
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg["lr"],
                               weight_decay=self.cfg["weight_decay"])

        Xt = torch.tensor(Xs, dtype=torch.float32)            # 标准化后,喂给 trunk
        Xt_raw = torch.tensor(X, dtype=torch.float32)          # 物理量纲原值,供 arrhenius 用
        Yt = torch.tensor(Yn_filled, dtype=torch.float32)
        Mt = torch.tensor(mask_np, dtype=torch.float32)
        y_std_t = torch.tensor(y_std, dtype=torch.float32)
        y_mean_t = torch.tensor(y_mean, dtype=torch.float32)

        physics_cfg = self.cfg.get("physics", {})
        arr_names, mono_names, bounds_names = _resolve_physics_targets(
            self.target_names, physics_cfg)
        weights = self.cfg["loss_weights"]
        n_exp = float(self.cfg["growth_exponent_n"])
        warmup_epochs = int(self.cfg.get("warmup_epochs", 0))

        T_K_t = t_eff_t = None
        if arr_names:
            T_K_t = Xt_raw[:, self.X_cols.index("T_K")]
            t_eff_t = Xt_raw[:, self.X_cols.index("t_eff")]
        idx_T = idx_t = None
        if mono_names:
            idx_T = self.X_cols.index("T_C")
            idx_t = self.X_cols.index("t_hold")

        self.loss_history = []
        best, wait = np.inf, 0
        for epoch in range(self.cfg["epochs"]):
            opt.zero_grad()
            pred_norm = self.model(Xt)
            pred_phys = pred_norm * y_std_t + y_mean_t  # 可微逆变换,回到物理量纲

            data_term = data_loss(pred_norm, Yt, Mt)

            if arr_names:
                arr_terms = []
                for name in arr_names:
                    j = self.target_names.index(name)
                    lnD = torch.log(pred_phys[:, j].clamp_min(1e-3))
                    arr_terms.append(arrhenius_loss(
                        lnD, T_K_t, t_eff_t, self.model.Q_J, self.model.lnk0, n_exp))
                arrhenius_term = torch.stack(arr_terms).mean()
            else:
                arrhenius_term = torch.tensor(0.0)

            if mono_names:
                mono_terms = []
                for name in mono_names:
                    j = self.target_names.index(name)
                    # 标准化空间算梯度符号即可:y_std>0,仿射逆变换不改变符号。
                    mono_terms.append(monotonic_loss(self.model, Xt, idx_T, idx_t, j))
                mono_term = torch.stack(mono_terms).mean()
            else:
                mono_term = torch.tensor(0.0)

            if bounds_names:
                bound_terms = []
                for name in bounds_names:
                    j = self.target_names.index(name)
                    lo, hi = schema.BY_NAME[name].bounds
                    bound_terms.append(bounds_loss(
                        pred_phys[:, j], torch.tensor(float(lo)), torch.tensor(float(hi))))
                bounds_term = torch.stack(bound_terms).mean()
            else:
                bounds_term = torch.tensor(0.0)

            scale = _physics_scale(epoch, warmup_epochs)
            total = (weights["data"] * data_term
                     + scale * weights["arrhenius"] * arrhenius_term
                     + scale * weights["monotonic"] * mono_term
                     + scale * weights["bounds"] * bounds_term)
            total.backward()
            opt.step()

            self.loss_history.append({
                "epoch": epoch,
                "data": float(data_term.item()),
                "arrhenius": float(arrhenius_term.item()) if arr_names else 0.0,
                "monotonic": float(mono_term.item()) if mono_names else 0.0,
                "bounds": float(bounds_term.item()) if bounds_names else 0.0,
                "total": float(total.item()),
                # Q 可识别性诊断用:记录每步更新后的 Q 值与刚用于该次更新的梯度,
                # 供事后比较梯度量级与 Adam 实际步长量级。
                "Q_kJmol": float(self.model.Q_kJmol.item()),
                "Q_grad": float(self.model.Q_kJmol.grad.item())
                if self.model.Q_kJmol.grad is not None else None,
            })

            if epoch < warmup_epochs:
                # 热身期物理项权重本身在爬坡,total 的涨落不代表"没在收敛"——
                # 早停判断必须等权重爬满(scale==1)后再开始计数,否则 data 项
                # 一收敛就会在物理约束还没生效前把训练整个停掉(真实数据复现过:
                # 27 折平均只训了 ~60/1500 步,learned Q 纹丝不动)。
                continue
            val = float(total.item())
            if val < best - 1e-6:
                best, wait = val, 0
            else:
                wait += 1
                if wait >= self.cfg["early_stopping_patience"]:
                    break
        return self

    def predict(self, X):
        self.model.eval()
        X = np.asarray(X, dtype=np.float64)
        Xs = ((X - self._x_mean) / self._x_std).astype(np.float32)
        with torch.no_grad():
            pred_norm = self.model(torch.tensor(Xs, dtype=torch.float32))
            pred_phys = pred_norm * torch.tensor(self._y_std, dtype=torch.float32) \
                + torch.tensor(self._y_mean, dtype=torch.float32)
        out = pred_phys.numpy()
        return out if out.shape[1] > 1 else out.ravel()

    @property
    def learned_Q_kJmol(self):
        return float(self.model.Q_kJmol.detach())
