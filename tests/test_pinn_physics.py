# -*- coding: utf-8 -*-
"""
test_pinn_physics.py —— TorchSklearnWrapper.fit() 物理约束接入(2026-07-26)
================================================================
覆盖:
  - NaN target 掩码(_mask_targets)
  - 物理约束目标解析(_resolve_physics_targets)——按目标分派,
    circularity/convexity/compaction_density 禁止出现在 mono_targets
  - warmup 线性 ramp(_physics_scale)
  - fit() 端到端:Q_J/lnk0 真正拿到梯度、NaN target 不传染、loss_history 有分量记录
"""
from __future__ import annotations

import numpy as np
import pytest

from nfm.models.pinn import (
    TorchSklearnWrapper,
    _mask_targets,
    _physics_scale,
    _resolve_physics_targets,
)


def _toy_cfg(**overrides):
    cfg = {
        "hidden_dims": [8, 4],
        "dropout": 0.0,
        "weight_decay": 0.0,
        "lr": 1e-2,
        "epochs": 5,
        "early_stopping_patience": 50,
        "Q_init_kJmol": 200,
        "growth_exponent_n": 3.0,
        "warmup_epochs": 0,
        "loss_weights": {"data": 1.0, "arrhenius": 1.0, "monotonic": 1.0, "bounds": 1.0},
        "physics": {
            "arrhenius_targets": ["D_XRD"],
            "mono_targets": ["D_XRD"],
            "bounds_targets": "all",
        },
    }
    cfg.update(overrides)
    return cfg


def _toy_data(n=20, seed=0):
    rng = np.random.default_rng(seed)
    T_C = rng.uniform(850, 950, n)
    t_hold = rng.uniform(10, 20, n)
    T_K = T_C + 273.15
    t_eff = t_hold + rng.uniform(0, 1, n)
    X_cols = ["T_C", "t_hold", "T_K", "t_eff"]
    X = np.column_stack([T_C, t_hold, T_K, t_eff]).astype(np.float64)
    D_XRD = 40 + 0.5 * (T_C - 850) + rng.normal(0, 1, n)
    circularity = np.clip(0.8 - 0.001 * (T_C - 850) + rng.normal(0, 0.01, n), 0, 1)
    Y = np.column_stack([D_XRD, circularity]).astype(np.float64)
    target_names = ["D_XRD", "circularity"]
    return X, X_cols, Y, target_names


# ---------------------------------------------------------------------
# 纯函数单元测试
# ---------------------------------------------------------------------

def test_mask_targets_replaces_nan_and_builds_mask():
    Y = np.array([[1.0, np.nan], [2.0, 3.0]])
    Y_filled, mask = _mask_targets(Y)
    assert np.array_equal(mask, np.array([[1, 0], [1, 1]]))
    assert Y_filled[0, 1] == 0.0
    assert Y_filled[1, 1] == 3.0


def test_physics_scale_ramps_linearly_then_saturates():
    assert _physics_scale(0, 10) == pytest.approx(0.1)
    assert _physics_scale(9, 10) == 1.0
    assert _physics_scale(100, 10) == 1.0
    assert _physics_scale(0, 0) == 1.0


def test_resolve_physics_targets_filters_and_expands_all():
    target_names = ["D_XRD", "circularity", "unknown_col"]
    phys_cfg = {
        "arrhenius_targets": ["D_XRD", "not_present"],
        "mono_targets": ["D_XRD"],
        "bounds_targets": "all",
    }
    arr_names, mono_names, bounds_names = _resolve_physics_targets(target_names, phys_cfg)
    assert arr_names == ["D_XRD"]
    assert mono_names == ["D_XRD"]
    # unknown_col 在 schema 里没有 bounds 声明,不应出现
    assert set(bounds_names) == {"D_XRD", "circularity"}


@pytest.mark.parametrize("forbidden", ["circularity", "convexity", "compaction_density"])
def test_resolve_physics_targets_rejects_forbidden_mono_targets(forbidden):
    target_names = [forbidden, "D_XRD"]
    phys_cfg = {"arrhenius_targets": [], "mono_targets": [forbidden], "bounds_targets": []}
    with pytest.raises(ValueError):
        _resolve_physics_targets(target_names, phys_cfg)


# ---------------------------------------------------------------------
# fit() 端到端
# ---------------------------------------------------------------------

def test_fit_gives_Q_and_lnk0_real_gradient():
    X, X_cols, Y, target_names = _toy_data()
    cfg = _toy_cfg(epochs=5)
    w = TorchSklearnWrapper(in_dim=X.shape[1], target_names=target_names,
                            cfg_pinn=cfg, X_cols=X_cols, seed=1)
    Q_before = cfg["Q_init_kJmol"] * 1e3
    w.fit(X, Y)
    assert w.model.Q_J.item() != Q_before  # Adam 每步更新量级 ~lr,与 Q_before 精确相等即说明没拿到梯度
    assert np.isfinite(w.learned_Q_kJmol)


def test_fit_masks_nan_targets_without_corrupting_other_head():
    X, X_cols, Y, target_names = _toy_data(n=30)
    Y_nan = Y.copy()
    Y_nan[::2, 0] = np.nan  # D_XRD 缺一半
    cfg = _toy_cfg(epochs=10)
    w = TorchSklearnWrapper(in_dim=X.shape[1], target_names=target_names,
                            cfg_pinn=cfg, X_cols=X_cols, seed=2)
    w.fit(X, Y_nan)
    pred = w.predict(X)
    assert np.isfinite(pred).all()
    assert np.isfinite(w.model.Q_J.item())
    assert np.isfinite(w.model.lnk0.item())


def test_loss_history_records_component_breakdown():
    X, X_cols, Y, target_names = _toy_data(n=16)
    cfg = _toy_cfg(epochs=3, early_stopping_patience=1000)
    w = TorchSklearnWrapper(in_dim=X.shape[1], target_names=target_names,
                            cfg_pinn=cfg, X_cols=X_cols, seed=3)
    w.fit(X, Y)
    assert len(w.loss_history) == 3
    for row in w.loss_history:
        for k in ("data", "arrhenius", "monotonic", "bounds", "total"):
            assert k in row
            assert np.isfinite(row[k])


def test_fit_predict_handles_heterogeneous_target_scales():
    """回归测试:master_table 真实跑分曾出现 lattice_c R2=-41405——
    根因是训练全程未标准化,~16Å 量级的窄范围 target 与 T_K(~1150)、
    Theta(0-200)等大尺度输入/输出共享同一趋势拟合,梯度被大尺度项主导。
    fit() 必须内部做 X/Y 标准化(物理约束仍在物理量纲空间计算),
    否则窄范围 target 训不出来。
    """
    rng = np.random.default_rng(0)
    n = 60
    T_K = rng.uniform(1120, 1220, n)
    other = rng.uniform(-1, 1, n)
    X = np.column_stack([T_K, other])
    X_cols = ["T_K", "other"]
    y_big = 40.0 + 0.3 * (T_K - 1120) + rng.normal(0, 0.5, n)        # ~40-70 量级
    y_narrow = 16.0 - 0.0005 * (T_K - 1120) + rng.normal(0, 0.002, n)  # ~16 量级,窄范围(仿 lattice_c)
    Y = np.column_stack([y_big, y_narrow])
    target_names = ["big", "narrow"]
    cfg = _toy_cfg(epochs=300, lr=1e-2, physics={
        "arrhenius_targets": [], "mono_targets": [], "bounds_targets": [],
    })
    w = TorchSklearnWrapper(in_dim=2, target_names=target_names, cfg_pinn=cfg,
                            X_cols=X_cols, seed=7)
    w.fit(X, Y)
    pred = w.predict(X)

    def r2(yt, yp):
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        return 1.0 - ss_res / ss_tot

    assert r2(y_big, pred[:, 0]) > 0.8
    assert r2(y_narrow, pred[:, 1]) > 0.8


def test_learned_Q_actually_moves_at_production_lr():
    """真实 07b 脚本用 config.yaml 默认 lr=1e-3 跑 27 折 LOCO,learned Q 每折都
    精确停在初始化值 200.0(|Δ|=0.0000)。根因:Q_J 以 J/mol 存储(量级 2e5)的
    float32 在该量级下 ULP≈0.0156,Adam 单步位移量级~lr=1e-3 时,`param - update`
    直接被舍入抹掉——不是没拿到梯度(grad 确实非零),是浮点精度让"学习"发生了但
    存不下来。必须把可学习量重新参数化到 kJ/mol 量级(~200,ULP 细得多)才行。
    """
    X, X_cols, Y, target_names = _toy_data(n=40, seed=11)
    cfg = _toy_cfg(epochs=80, lr=1e-3, weight_decay=1e-4, warmup_epochs=0,
                   early_stopping_patience=1000)
    w = TorchSklearnWrapper(in_dim=X.shape[1], target_names=target_names,
                            cfg_pinn=cfg, X_cols=X_cols, seed=11)
    Q_before = cfg["Q_init_kJmol"]
    w.fit(X, Y)
    assert abs(w.learned_Q_kJmol - Q_before) > 1e-3


def test_loss_history_records_Q_trajectory_and_gradient():
    """任务指令_PINN诊断修复_2026-07-26 任务1 需要逐步 Q 值+梯度来判断 Q 是否
    被数据驱动(而非被浮点精度冻结)。D_XRD 在 arrhenius_targets 里,梯度应非零。
    """
    X, X_cols, Y, target_names = _toy_data(n=20, seed=9)
    cfg = _toy_cfg(epochs=5, warmup_epochs=0)
    w = TorchSklearnWrapper(in_dim=X.shape[1], target_names=target_names,
                            cfg_pinn=cfg, X_cols=X_cols, seed=9)
    w.fit(X, Y)
    for row in w.loss_history:
        assert np.isfinite(row["Q_kJmol"])
        assert row["Q_grad"] is not None
        assert np.isfinite(row["Q_grad"])
    assert row["Q_grad"] != 0.0


def test_early_stopping_does_not_fire_during_physics_warmup():
    """真实数据跑分发现的第二个坑:early stopping 原来盯着 total loss,而
    warmup 期间物理项权重本身在线性爬坡,total 会在权重爬坡时持续"改善"或
    震荡,导致 data 项其实早就收敛时 patience 计数器提前打满、还没爬到
    warmup_epochs、物理约束权重还远小于 1 就整个训练停掉了(2026-07-26
    真实 27 折 LOCO 复现:learned Q 纹丝不动,因为平均只训了 ~60/1500 步)。
    патience 判断必须等 warmup 结束后才开始计数。
    """
    X, X_cols, Y, target_names = _toy_data(n=20, seed=5)
    cfg = _toy_cfg(epochs=60, warmup_epochs=50, early_stopping_patience=3, lr=1e-2)
    w = TorchSklearnWrapper(in_dim=X.shape[1], target_names=target_names,
                            cfg_pinn=cfg, X_cols=X_cols, seed=5)
    w.fit(X, Y)
    assert len(w.loss_history) >= 50


def test_fit_without_physics_targets_still_trains_data_only():
    X, X_cols, Y, target_names = _toy_data(n=12)
    cfg = _toy_cfg(epochs=3, physics={
        "arrhenius_targets": [], "mono_targets": [], "bounds_targets": [],
    })
    w = TorchSklearnWrapper(in_dim=X.shape[1], target_names=target_names,
                            cfg_pinn=cfg, X_cols=X_cols, seed=4)
    w.fit(X, Y)
    for row in w.loss_history:
        assert row["arrhenius"] == 0.0
        assert row["monotonic"] == 0.0
        assert row["bounds"] == 0.0
