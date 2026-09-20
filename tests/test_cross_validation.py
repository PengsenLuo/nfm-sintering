# -*- coding: utf-8 -*-
"""
test_cross_validation.py —— run_cv_multi(多目标 LOCO/LOO,2026-07-26)
=======================================================================
PINN 共享主干一次预测多个 target(部分 target 如 D_XRD 只有 51/81 有真值),
run_cv(单目标)不够用。run_cv_multi 的关键行为:
  - 折切分逻辑复用 loco_splits/loo_splits(与 run_cv 一致,不重复实现)
  - 汇总统计按 target 逐列剔除 NaN 真值,不同 target 的可用样本数互不影响
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfm.evaluation.cross_validation import run_cv_multi, theta_precursor_baseline


class _ZeroModel:
    """恒预测 0 的哑模型,只用来验证 run_cv_multi 的切分/汇总逻辑本身。"""

    def fit(self, X, Y):
        return self

    def predict(self, X):
        n = X.shape[0]
        return np.zeros((n, 1))


def _toy_df():
    return pd.DataFrame({
        "condition_id": ["G1", "G1", "G2", "G2"],
        "x": [0.0, 1.0, 2.0, 3.0],
        "y": [5.0, np.nan, 7.0, 9.0],
    })


def test_run_cv_multi_excludes_nan_ytrue_from_metrics():
    df = _toy_df()
    res = run_cv_multi(lambda: _ZeroModel(), df, ["x"], ["y"], scheme="loco")

    assert res["n_folds"] == 2
    s = res["summary"]["y"]
    # 有效真值只有 5.0(G1 折)、7.0/9.0(G2 折)三个,NaN 那一行被剔除
    assert s["n_eval"] == 3
    assert s["MAE"] == pytest.approx((5.0 + 7.0 + 9.0) / 3.0)
    assert s["R2"] == pytest.approx(1.0 - 155.0 / 8.0)


def test_run_cv_multi_per_target_nan_counts_are_independent():
    df = pd.DataFrame({
        "condition_id": ["G1", "G1", "G2", "G2"],
        "x": [0.0, 1.0, 2.0, 3.0],
        "y1": [1.0, 2.0, 3.0, 4.0],          # 全部有效
        "y2": [np.nan, np.nan, 5.0, 6.0],    # 只有一半有效
    })

    class _TwoHeadZero:
        def fit(self, X, Y):
            return self

        def predict(self, X):
            return np.zeros((X.shape[0], 2))

    res = run_cv_multi(lambda: _TwoHeadZero(), df, ["x"], ["y1", "y2"], scheme="loco")
    assert res["summary"]["y1"]["n_eval"] == 4
    assert res["summary"]["y2"]["n_eval"] == 2


def test_run_cv_multi_scaler_and_model_factory_called_per_fold():
    df = _toy_df()
    calls = {"factory": 0, "scaler_fit": 0}

    class _RecordingModel(_ZeroModel):
        pass

    def factory():
        calls["factory"] += 1
        return _RecordingModel()

    class _NoopScaler:
        def fit(self, X):
            calls["scaler_fit"] += 1
            return self

        def transform(self, X):
            return X

    run_cv_multi(factory, df, ["x"], ["y"], scheme="loco",
                 scaler_factory=lambda: _NoopScaler())
    assert calls["factory"] == 2   # 2 折,每折一个新模型(防止跨折状态泄漏)
    assert calls["scaler_fit"] == 2  # 每折在训练子集内单独 fit(防泄漏)


def test_theta_precursor_baseline_drops_nan_target_and_fits_ridge():
    """任务指令_PINN诊断修复_2026-07-26(任务3)的标准基线:ln(Theta)+Theta+
    precursor one-hot 的 LOCO Ridge。用一个 y 与 ln(Theta) 强线性相关的
    合成数据集验证:R2 应接近 1(基线本身跑得通、跑得对),且 NaN target 行
    被正确剔除、不进入 n_eval。
    """
    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame({
        "condition_id": np.repeat(np.arange(10), 3),
        "precursor": np.tile(["S", "M", "L"], 10),
        "Theta": rng.uniform(4.0, 47.0, n),
    })
    df["y"] = 10.0 + 5.0 * np.log(df["Theta"]) + rng.normal(0, 0.05, n)
    df.loc[0, "y"] = np.nan

    summary = theta_precursor_baseline(df, "y")
    assert summary["n_eval"] == n - 1
    assert summary["R2"] > 0.9


def test_theta_precursor_baseline_auto_scopes_nano_layer_targets():
    """T1(任务指令_稿件数据冻结_2026-08-06):theta_precursor_baseline 对纳米层
    目标(D_XRD/lattice_a/lattice_c/c_a_ratio)必须自动走 nano_layer_frame,
    忽略传入 DataFrame 里混入的仪器2"陷阱"行,不静默扩大样本/折数。
    """
    rng = np.random.default_rng(1)
    # 51 个"真"样本:17 条件 x S/M/L,仪器1,可靠
    n_true = 51
    df_true = pd.DataFrame({
        "condition_id": np.repeat([f"C{i:02d}" for i in range(17)], 3),
        "precursor": np.tile(["S", "M", "L"], 17),
        "Theta": rng.uniform(4.0, 47.0, n_true),
        "xrd_instrument": 1,
        "D_XRD_reliable": True,
    })
    df_true["D_XRD"] = 30.0 + 10.0 * np.log(df_true["Theta"]) + rng.normal(0, 0.5, n_true)

    # 20 个仪器2"陷阱"行:不同 condition_id,D_XRD 非空、且刻意偏离趋势,
    # 若被误纳入,LOCO 折数与 n_eval 都会变、R2 会明显变差
    n_trap = 20
    df_trap = pd.DataFrame({
        "condition_id": [f"X{i:02d}" for i in range(n_trap)],
        "precursor": np.tile(["S", "M", "L"], 7)[:n_trap],
        "Theta": rng.uniform(4.0, 47.0, n_trap),
        "xrd_instrument": 2,
        "D_XRD_reliable": True,
    })
    df_trap["D_XRD"] = 200.0  # 明显偏离,一旦混入 R2 会崩

    df = pd.concat([df_true, df_trap], ignore_index=True)

    summary = theta_precursor_baseline(df, "D_XRD")

    assert summary["n_folds"] == 17
    assert summary["n_eval"] == 51
    assert summary["R2"] > 0.5  # 若混入陷阱行,R2 会被拉到很低甚至负值


def test_theta_precursor_baseline_non_nano_target_unaffected():
    """非纳米层目标(如合成的 'y')不应被 nano_layer_frame 过滤逻辑触碰
    (传入的 df 甚至可以没有 xrd_instrument 列,不应报错)。
    """
    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame({
        "condition_id": np.repeat(np.arange(10), 3),
        "precursor": np.tile(["S", "M", "L"], 10),
        "Theta": rng.uniform(4.0, 47.0, n),
    })
    df["y"] = 10.0 + 5.0 * np.log(df["Theta"]) + rng.normal(0, 0.05, n)

    summary = theta_precursor_baseline(df, "y")
    assert summary["n_eval"] == n
