# -*- coding: utf-8 -*-
"""
cross_validation.py —— 小样本验证(X.7.4,v0.2)
================================================
★ 主口径:LOCO(留整条件,27 折,每折留出同炉三样本)/ 分组嵌套 CV。
   LOO(81 折)仅作辅助,因同炉样本共享热历史,LOO 存在组内泄漏而高估泛化;
   LOCO 与 LOO 差额定量揭示该效应。
另提供:留一温度水平、留一前驱体类型、网格外独立工艺点评估。

防泄漏:标准化等预处理必须在每折训练子集内 fit,故本模块对外暴露 splitter,
由建模脚本在折内做 fit_transform(见 features.scaling)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut


def loco_splits(df: pd.DataFrame):
    """留整条件:按 condition_id 分组,27 折。返回 (train_idx, test_idx) 生成器。"""
    groups = df["condition_id"].to_numpy()
    logo = LeaveOneGroupOut()
    X_dummy = np.zeros((len(df), 1))
    return logo.split(X_dummy, groups=groups), groups


def loo_splits(df: pd.DataFrame):
    """逐样本留一:81 折(辅助口径)。"""
    loo = LeaveOneOut()
    return loo.split(np.zeros((len(df), 1)))


def leave_one_temperature(df: pd.DataFrame):
    """留一温度水平:每折留出某一 T_C 的全部样本(检验温度外推)。"""
    for T in sorted(df["T_C"].unique()):
        test = df.index[df["T_C"] == T].to_numpy()
        train = df.index[df["T_C"] != T].to_numpy()
        yield train, test, f"T={T}"


def leave_one_precursor(df: pd.DataFrame):
    """留一前驱体类型:每折留出某一前驱体的全部样本(检验前驱体外推)。"""
    for p in df["precursor"].unique():
        test = df.index[df["precursor"] == p].to_numpy()
        train = df.index[df["precursor"] != p].to_numpy()
        yield train, test, f"precursor={p}"


def run_cv(model_factory, df, X_cols, y_col, scheme="loco",
           scaler_factory=None, metrics=("R2", "MAE")):
    """通用 CV 驱动。model_factory()->带 fit/predict 的模型;scaler 在折内 fit。

    返回 dict:per_fold 列表 + 汇总(R2、MAE)。
    """
    rows = []
    if scheme == "loco":
        splits, groups = loco_splits(df)
        iterator = ((tr, te, None) for tr, te in splits)
    elif scheme == "loo":
        iterator = ((tr, te, None) for tr, te in loo_splits(df))
    elif scheme == "lot":
        iterator = leave_one_temperature(df)
    elif scheme == "lop":
        iterator = leave_one_precursor(df)
    else:
        raise ValueError(f"未知 CV 方案:{scheme}")

    X_all = df[X_cols].to_numpy(float)
    y_all = df[y_col].to_numpy(float)
    for tr, te, tag in iterator:
        Xtr, Xte = X_all[tr], X_all[te]
        ytr, yte = y_all[tr], y_all[te]
        if scaler_factory is not None:
            sc = scaler_factory().fit(Xtr)        # ★ 仅训练折 fit,防泄漏
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        m = model_factory()
        m.fit(Xtr, ytr)
        yp = m.predict(Xte)
        rows.append(dict(tag=tag, y_true=yte.tolist(), y_pred=np.ravel(yp).tolist()))

    y_true = np.concatenate([np.asarray(r["y_true"]) for r in rows])
    y_pred = np.concatenate([np.asarray(r["y_pred"]) for r in rows])
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) + 1e-12
    summary = {"R2": 1.0 - ss_res / ss_tot,
               "MAE": float(np.mean(np.abs(y_true - y_pred))),
               "n_folds": len(rows), "scheme": scheme}
    return {"per_fold": rows, "summary": summary}


def run_cv_multi(model_factory, df, X_cols, y_cols, scheme="loco", scaler_factory=None):
    """多目标版 run_cv:model.fit(Xtr, Ytr)/predict(Xte) 处理 (n, len(y_cols)) 矩阵。

    与 run_cv 的差异只在于 y——PINN 共享主干一次预测多个 target,其中部分列
    (如 D_XRD 51/81)真值缺失。折切分逻辑与 run_cv 完全一致(复用同一批
    splitter);汇总统计按 target 逐列剔除 NaN 真值,不同 target 的可用样本数
    (n_eval)互不影响,不会因为某个样本缺 D_XRD 就连累其它 target 的评估。
    """
    rows = []
    if scheme == "loco":
        splits, groups = loco_splits(df)
        iterator = ((tr, te, None) for tr, te in splits)
    elif scheme == "loo":
        iterator = ((tr, te, None) for tr, te in loo_splits(df))
    elif scheme == "lot":
        iterator = leave_one_temperature(df)
    elif scheme == "lop":
        iterator = leave_one_precursor(df)
    else:
        raise ValueError(f"未知 CV 方案:{scheme}")

    y_cols = list(y_cols)
    X_all = df[X_cols].to_numpy(float)
    Y_all = df[y_cols].to_numpy(float)
    for tr, te, tag in iterator:
        Xtr, Xte = X_all[tr], X_all[te]
        Ytr, Yte = Y_all[tr], Y_all[te]
        if scaler_factory is not None:
            sc = scaler_factory().fit(Xtr)        # ★ 仅训练折 fit,防泄漏
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        m = model_factory()
        m.fit(Xtr, Ytr)
        Yp = np.atleast_2d(m.predict(Xte))
        if Yp.shape[0] != len(te):
            Yp = Yp.T
        rows.append(dict(tag=tag, test_idx=np.asarray(te).tolist(),
                         y_true=Yte.tolist(), y_pred=Yp.tolist()))

    y_true_all = np.concatenate([np.asarray(r["y_true"]) for r in rows], axis=0)
    y_pred_all = np.concatenate([np.asarray(r["y_pred"]) for r in rows], axis=0)

    summary = {}
    for j, name in enumerate(y_cols):
        yt, yp = y_true_all[:, j], y_pred_all[:, j]
        valid = ~np.isnan(yt)
        n_eval = int(valid.sum())
        if n_eval == 0:
            summary[name] = {"R2": float("nan"), "MAE": float("nan"), "n_eval": 0}
            continue
        yt_v, yp_v = yt[valid], yp[valid]
        ss_res = float(np.sum((yt_v - yp_v) ** 2))
        ss_tot = float(np.sum((yt_v - yt_v.mean()) ** 2)) + 1e-12
        summary[name] = {"R2": 1.0 - ss_res / ss_tot,
                         "MAE": float(np.mean(np.abs(yt_v - yp_v))),
                         "n_eval": n_eval}
    return {"per_fold": rows, "summary": summary, "scheme": scheme, "n_folds": len(rows)}


def theta_precursor_baseline(df: pd.DataFrame, y_col: str, alpha: float = 1.0,
                             scheme: str = "loco"):
    """标准平价基线:ln(Theta)+Theta+前驱体 one-hot 的 LOCO Ridge。

    任何声称"物理约束/更复杂架构有帮助"的建模改动,都必须先跑赢这个基线,
    否则不得声称改进有效。只在目标列非缺失的样本上评估(与 run_cv_multi 的
    多目标 NaN 容忍不同,Ridge 不能吃 NaN 的 y,所以先按该 target 过滤,
    folds 数因此可能少于 27)。

    纳米层目标(D_XRD/lattice_a/lattice_c/c_a_ratio,见 nfm.nano_layer.NANO_COLS)
    自动先走 nfm.nano_layer.nano_layer_frame() 限定到仪器1、分辨率可信的
    51 样:若只做 dropna、不做仪器过滤,会静默把仪器2的可用行也吃进来
    (78样/27折,R2 从 0.50 跌到 0.43)。选择"自动路由"而非"显式 scope= 参数"
    是因为 NANO_COLS 已经是
    nfm.nano_layer 里认定的唯一权威枚举,自动挡比"调用方必须记得传对参数"
    更难被绕过;nano_layer_frame() 内部的 warnings.warn 已经让误用/过滤结果
    在每次调用时可见,不需要额外的显式参数来满足"不能静默降级"的要求。
    """
    from sklearn.linear_model import Ridge

    from nfm.features.scaling import scaler_factory
    from nfm.nano_layer import NANO_COLS, nano_layer_frame

    d = df.copy()
    if y_col in NANO_COLS:
        d = nano_layer_frame(d, require_reliable=True)
    d["ln_Theta"] = np.log(d["Theta"])
    d = pd.get_dummies(d, columns=["precursor"], prefix="precursor")
    precursor_cols = [c for c in d.columns if c.startswith("precursor_")]
    d[precursor_cols] = d[precursor_cols].astype(float)
    X_cols = ["ln_Theta", "Theta"] + precursor_cols
    d_fit = d.dropna(subset=X_cols + [y_col])
    res = run_cv_multi(lambda: Ridge(alpha=alpha), d_fit, X_cols, [y_col],
                       scheme=scheme, scaler_factory=scaler_factory)
    out = dict(res["summary"][y_col])
    out["n_folds"] = res["n_folds"]
    return out


def evaluate_external(model, df_train, ext_df, X_cols, y_col,
                      scaler=None):
    """网格外独立工艺点评估:模型已在全部 81 样本训练好,在 ext_df 上预测。

    ext_df 需含 X_cols(可由 thermal_exposure 对外部工艺点先算 Θ)与实测 y(若已测)。
    """
    Xe = ext_df[X_cols].to_numpy(float)
    if scaler is not None:
        Xe = scaler.transform(Xe)
    pred = np.ravel(model.predict(Xe))
    res = ext_df[["T_C", "beta", "t_hold"]].copy()
    res["pred"] = pred
    if y_col in ext_df.columns:
        res["measured"] = ext_df[y_col].to_numpy(float)
        res["abs_err"] = (res["pred"] - res["measured"]).abs()
    return res
