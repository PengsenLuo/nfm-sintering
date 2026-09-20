# -*- coding: utf-8 -*-
"""dxrd_baseline.py —— compaction_density/D_sec/D_XRD/circularity 统一
Ridge(alpha=1,原始尺度)+LOCO 实现
==========================================================================
模型选择理由(Ridge 而非 OLS):设计矩阵为显式截距 + 3 个 one-hot 前驱体
哑元(不 drop_first),三列之和恒为 1,与截距列线性相关,矩阵完全共线;
`LinearRegression` 内部靠 `lstsq` 给出的是一个隐式依赖实现细节的最小范数
解,不是唯一定义的解。`Ridge(alpha=1)` 通过正则化项打破共线性,解是
确定性的,且在原始(未标准化)尺度上系数同样可直接读出物理意义。

模型规格演变(保留关键结论,供复现参考):
1. 自变量用 `ln(Theta)` 而非 `Theta` 本身——两者是同一底层量的单调变换,
   同时纳入会使两个系数各自不可解读;仅保留 lnΘ 后,LOCO R²/MAE 全面改善
   (D_XRD 从 0.494 提升到 0.562)。
2. 前驱体编码最终选定 **sum(偏差)编码**(而非以 S 为参照类的哑元编码):
   `I_M = 1(precursor==M) / −1(precursor==S) / 0(precursor==L)`,
   `I_L` 同理。参照类编码会把 S 吸收进不受 Ridge 惩罚的截距,使 S/M/L 三者
   在正则化项下不再对称受罚,对 `D_sec`(S 是最极端值)造成实质伤害
   (LOCO R² 从 0.918 降到 0.913);sum 编码下 S 在两列都取 −1,不被截距单独
   吸收,三者对称受罚。

模型规格(现行):target ~ lnΘ + precursor(sum 编码,`I_M`/`I_L` 两列,
S 全部取 −1)+ 显式截距,`sklearn.linear_model.Ridge(alpha=1,
fit_intercept=True)`,**不做特征标准化**(不加 StandardScaler)——与仓库
另一套 `nfm.evaluation.cross_validation.theta_precursor_baseline`(Ridge +
StandardScaler + drop_first 的正则化基线,系数为标准化尺度、不可直接读)
是两回事,不要混用。本项目 ANOVA(`scripts/18`/`19` 等)本来就统一用
`C(precursor, Sum)`,sum 编码的 Ridge 基线与之口径一致。

**sum 编码下系数的读法**:`β_M`/`β_L` 是 M/L 相对**三个前驱体总体均值**
的偏差(不是相对 S 的差);S 的偏差 = −(β_M + β_L)(sum-to-zero 约束);
截距是"lnΘ=0 处三个前驱体的总体均值水平"。

`circularity` 通过本函数走同一条统一路径(`scripts/22_build_
manuscript_numbers.py::build_table7_loco` 内联调用,不进 `TARGETS` 常量、
不写入 `table6_ridge_coef.csv`——Table 6 仍只报 compaction_density/D_sec/
D_XRD 三个目标的系数,circularity 只出现在 Table 7 的 LOCO 性能对照)。

取数口径:仅当 `y_col` 属于 `nfm.nano_layer.NANO_COLS`(D_XRD/lattice_a/
lattice_c/c_a_ratio)时,自动经 `nano_layer_frame(df, require_reliable=True)`
限定到纳米层唯一合法口径(51样,仅 xrd_instrument==1 且
D_XRD_reliable==True)。其余 target(如 compaction_density、D_sec、
circularity)使用 `data/processed/master_table.csv` 的全部样本
(现状 81 样/27 条件)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from nfm.evaluation.cross_validation import loco_splits
from nfm.nano_layer import NANO_COLS, nano_layer_frame

PRECURSOR_SUM_LEVELS = ["M", "L"]  # sum 编码的两个非基准水平;S 在两列都取 -1


def _build_design(df: pd.DataFrame):
    d = df.copy()
    d["ln_Theta"] = np.log(d["Theta"])
    for p in PRECURSOR_SUM_LEVELS:
        d[f"precursor_{p}"] = np.where(
            d["precursor"] == "S", -1.0,
            np.where(d["precursor"] == p, 1.0, 0.0))
    x_cols = ["ln_Theta"] + [f"precursor_{p}" for p in PRECURSOR_SUM_LEVELS]
    return d, x_cols


def ridge_coef_loco(df: pd.DataFrame, y_col: str, alpha: float = 1.0) -> dict:
    """target 的 3特征模型系数(lnΘ + precursor_M + precursor_L,sum 编码,
    全数据拟合)+ LOCO(留整条件)性能,`sklearn.linear_model.Ridge(alpha=alpha)`,
    原始(未标准化)尺度。

    输入:含 target/Theta/precursor/condition_id/sample_id 列的 DataFrame
    (通常是 data/processed/master_table.csv)。若 y_col 属于
    `nfm.nano_layer.NANO_COLS`,先经 nano_layer_frame 限定到 51 样/17 条件
    (不满足该口径直接报错);否则使用全部样本(现状 81 样/27 条件)。
    """
    is_nano = y_col in NANO_COLS
    sub = nano_layer_frame(df, require_reliable=True) if is_nano else df
    sub = sub[sub[y_col].notna()].reset_index(drop=True)

    n = len(sub)
    n_cond = sub["condition_id"].nunique()
    if is_nano:
        assert n == 51 and n_cond == 17, (
            f"预期纳米层唯一合法口径为 51 样/17 条件,实际 n={n}, n_cond={n_cond}"
            f" —— 口径已变化,需先核实原因,不得继续套用本函数的旧断言。"
        )
        counts = sub["precursor"].value_counts().to_dict()
        assert all(counts.get(p, 0) == n_cond for p in ["S", "M", "L"]), (
            f"预期每条件下 S/M/L 各恰好1样,实际前驱体计数={counts},"
            f"LOCO 逐折留出'该条件全部样本'的假设不成立。"
        )
    else:
        assert n == len(df), (
            f"预期 {y_col} 在 master_table.csv 全量样本上非缺失(n={len(df)}),"
            f"实际 n={n} —— 若该 target 出现缺失,需先核实原因,不得静默缩窄口径。"
        )

    d, x_cols = _build_design(sub)
    X_all = d[x_cols].to_numpy(float)
    y_all = d[y_col].to_numpy(float)

    model_full = Ridge(alpha=alpha, fit_intercept=True)
    model_full.fit(X_all, y_all)
    coef_map = dict(zip(x_cols, model_full.coef_))
    coef = {
        "intercept": float(model_full.intercept_),  # lnΘ=0 处三个前驱体总体均值水平
        "ln_Theta": float(coef_map["ln_Theta"]),
        "precursor_M": float(coef_map["precursor_M"]),  # M 相对总体均值的偏差
        "precursor_L": float(coef_map["precursor_L"]),  # L 相对总体均值的偏差
        # S 相对总体均值的偏差 = -(precursor_M + precursor_L)(sum-to-zero 约束,未单列)
    }
    r2_insample = float(model_full.score(X_all, y_all))

    splits, groups = loco_splits(d)
    fold_rows = []
    y_true_all, y_pred_all = [], []
    for tr, te in splits:
        Xtr, Xte = X_all[tr], X_all[te]
        ytr, yte = y_all[tr], y_all[te]
        m = Ridge(alpha=alpha, fit_intercept=True)
        m.fit(Xtr, ytr)
        yp = m.predict(Xte)
        cond = d.iloc[te]["condition_id"].unique()
        assert len(cond) == 1, "LOCO 折内出现跨条件样本,分组逻辑有误"
        for sid, yt, ypi in zip(d.iloc[te]["sample_id"], yte, yp):
            fold_rows.append(dict(condition_id=cond[0], sample_id=sid,
                                  y_true=float(yt), y_pred=float(ypi)))
        y_true_all.append(yte)
        y_pred_all.append(yp)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    ss_res = float(np.sum((y_true_all - y_pred_all) ** 2))
    ss_tot = float(np.sum((y_true_all - y_true_all.mean()) ** 2)) + 1e-12
    r2_loco = 1.0 - ss_res / ss_tot
    mae_loco = float(np.mean(np.abs(y_true_all - y_pred_all)))
    n_folds = len(set(groups.tolist()))

    return {
        "coef": coef, "n": n, "n_cond": n_cond, "r2_insample": r2_insample,
        "R2": r2_loco, "MAE": mae_loco, "n_folds": n_folds,
        "n_eval": len(y_true_all), "fold_rows": fold_rows,
    }
