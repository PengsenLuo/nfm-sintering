# -*- coding: utf-8 -*-
"""29 lnΘ-only 基线 vs 现行 lnΘ+Θ 基线(补充/敏感性分析,不替换稿件口径)
====================================================================
⚠️ **本脚本已被后续模型规格取代,已过时,不要重跑**:当时要判断的问题
正是"要不要采用 lnΘ-only";后来模型规格已拍板去掉 Θ 项,`nfm.dxrd_baseline.
ridge_coef_loco()` 现在本身就是 lnΘ-only(外加参照类编码,S 为参照类)。
本脚本第158-172行读取 `data/interim/table6_ridge_coef.csv` 的 `Theta` 列
做"新旧对照",该列在去掉 Θ 项之后已不存在,重跑会 KeyError——这是预期的
"脚本已过时"信号,不是缺陷,不修复。保留本文件只作历史记录(当时两种
口径 LOCO 对照的独立探针结果)。

背景(历史,当时的问题描述):`scripts/24_table6_ridge_baseline.py`(经 `nfm.dxrd_baseline.
ridge_coef_loco`)对 compaction_density/D_sec/D_XRD 三个 target 拟合
`target ~ lnΘ + Θ + precursor(one-hot,不 drop_first) + 截距` 的
Ridge(alpha=1,原始尺度)模型。lnΘ 与 Θ 高度相关(同一底层量的两种单调
变换),两个系数各自的可解释性存疑("谁是主驱动、谁是搭便车")。

本脚本是独立的旁路敏感性分析:把 Θ 从设计矩阵里去掉,只保留
`target ~ lnΘ + precursor(one-hot) + 截距`,同规格 Ridge(alpha=1,原始
尺度)+ LOCO(留整条件),用于对照决定要不要采用。

**不修改** `src/nfm/dxrd_baseline.py`、`scripts/24_table6_ridge_baseline.py`
——本脚本只读取已有的
`data/interim/table6_ridge_coef.csv` / `table6_ridge_loco.csv`(lnΘ+Θ 版本,
已由 24 号脚本产出,这里不重跑)作对照基准,自己另算 lnΘ-only 版本写到
新文件。

D_XRD 口径同现行做法:限定到 `nfm.nano_layer.nano_layer_frame`(51样/
17条件,仅 xrd_instrument==1 且 D_XRD_reliable==True);compaction_density/
D_sec 用全部 81 样/27 条件。

输出:
  data/interim/s5_1_lntheta_only_coef.csv   系数表(3行)
  data/interim/s5_1_lntheta_only_loco.csv   LOCO表(3行)
  data/interim/s5_1_baseline_comparison.csv 两版本并排对照(6行)
"""
import _bootstrap  # noqa

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from nfm.evaluation.cross_validation import loco_splits
from nfm.nano_layer import NANO_COLS, nano_layer_frame

MASTER_TABLE = "data/processed/master_table.csv"
TARGETS = ["compaction_density", "D_sec", "D_XRD"]
PRECURSOR_ORDER = ["L", "M", "S"]

OUT_COEF = "data/interim/s5_1_lntheta_only_coef.csv"
OUT_LOCO = "data/interim/s5_1_lntheta_only_loco.csv"
OUT_COMPARE = "data/interim/s5_1_baseline_comparison.csv"

EXISTING_COEF = "data/interim/table6_ridge_coef.csv"
EXISTING_LOCO = "data/interim/table6_ridge_loco.csv"


def _build_design(df: pd.DataFrame):
    """同 nfm.dxrd_baseline._build_design,但去掉 Theta 列,只留 lnΘ。"""
    d = df.copy()
    d["ln_Theta"] = np.log(d["Theta"])
    for p in PRECURSOR_ORDER:
        d[f"precursor_{p}"] = (d["precursor"] == p).astype(float)
    x_cols = ["ln_Theta"] + [f"precursor_{p}" for p in PRECURSOR_ORDER]
    return d, x_cols


def ridge_coef_loco_lntheta_only(df: pd.DataFrame, y_col: str, alpha: float = 1.0) -> dict:
    """lnΘ-only 版:target ~ lnΘ + precursor(one-hot) + 截距,Ridge(alpha=1),
    原始尺度。与 nfm.dxrd_baseline.ridge_coef_loco 完全同规格,唯一差异是
    设计矩阵去掉 Theta 列。口径判定(nano_layer_frame / 全量断言)逐字复用。
    """
    is_nano = y_col in NANO_COLS
    sub = nano_layer_frame(df, require_reliable=True) if is_nano else df
    sub = sub[sub[y_col].notna()].reset_index(drop=True)

    n = len(sub)
    n_cond = sub["condition_id"].nunique()
    if is_nano:
        assert n == 51 and n_cond == 17, (
            f"预期纳米层唯一合法口径为 51 样/17 条件,实际 n={n}, n_cond={n_cond}"
        )
    else:
        assert n == len(df), (
            f"预期 {y_col} 在 master_table.csv 全量样本上非缺失(n={len(df)}),实际 n={n}"
        )

    d, x_cols = _build_design(sub)
    X_all = d[x_cols].to_numpy(float)
    y_all = d[y_col].to_numpy(float)

    model_full = Ridge(alpha=alpha, fit_intercept=True)
    model_full.fit(X_all, y_all)
    coef_map = dict(zip(x_cols, model_full.coef_))
    coef = {
        "intercept": float(model_full.intercept_),
        "ln_Theta": float(coef_map["ln_Theta"]),
        "precursor_L": float(coef_map["precursor_L"]),
        "precursor_M": float(coef_map["precursor_M"]),
        "precursor_S": float(coef_map["precursor_S"]),
    }
    r2_insample = float(model_full.score(X_all, y_all))

    splits, groups = loco_splits(d)
    y_true_all, y_pred_all = [], []
    for tr, te in splits:
        Xtr, Xte = X_all[tr], X_all[te]
        ytr, yte = y_all[tr], y_all[te]
        m = Ridge(alpha=alpha, fit_intercept=True)
        m.fit(Xtr, ytr)
        yp = m.predict(Xte)
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
        "R2": r2_loco, "MAE": mae_loco, "n_folds": n_folds, "n_eval": len(y_true_all),
    }


def main():
    df = pd.read_csv(MASTER_TABLE)

    coef_rows, loco_rows = [], []
    for target in TARGETS:
        out = ridge_coef_loco_lntheta_only(df, target)
        c = out["coef"]
        print(f"\n=== {target} lnΘ-only (n={out['n']}, n_cond={out['n_cond']}) ===")
        print(f"  截距={c['intercept']:+.4f} lnΘ={c['ln_Theta']:+.4f} "
              f"L={c['precursor_L']:+.4f} M={c['precursor_M']:+.4f} S={c['precursor_S']:+.4f}")
        print(f"  LOCO R²={out['R2']:.4f}  MAE={out['MAE']:.4f}  "
              f"n_folds={out['n_folds']}  n_eval={out['n_eval']}")
        coef_rows.append({
            "target": target, "n": out["n"], "n_cond": out["n_cond"],
            "intercept": c["intercept"], "ln_Theta": c["ln_Theta"],
            "L": c["precursor_L"], "M": c["precursor_M"], "S": c["precursor_S"],
            "r2_insample": out["r2_insample"],
        })
        loco_rows.append({
            "target": target, "R2": out["R2"], "MAE": out["MAE"],
            "n_folds": out["n_folds"], "n_eval": out["n_eval"],
        })

    coef_df = pd.DataFrame(coef_rows)
    loco_df = pd.DataFrame(loco_rows)
    coef_df.to_csv(OUT_COEF, index=False)
    loco_df.to_csv(OUT_LOCO, index=False)
    print(f"\n已写出 {OUT_COEF}\n已写出 {OUT_LOCO}")

    # ---- 对照现行 lnΘ+Θ 基线(读已有文件,不重跑 script 24) ----
    existing_coef = pd.read_csv(EXISTING_COEF)
    existing_loco = pd.read_csv(EXISTING_LOCO)

    compare_rows = []
    for target in TARGETS:
        old_loco = existing_loco[existing_loco["target"] == target].iloc[0]
        new_loco = loco_df[loco_df["target"] == target].iloc[0]
        old_coef = existing_coef[existing_coef["target"] == target].iloc[0]
        new_coef = coef_df[coef_df["target"] == target].iloc[0]
        compare_rows.append({
            "target": target,
            "model": "lnTheta+Theta (existing)",
            "R2_loco": old_loco["R2"], "MAE_loco": old_loco["MAE"],
            "n_folds": old_loco["n_folds"], "n_eval": old_loco["n_eval"],
            "ln_Theta_coef": old_coef["ln_Theta"], "Theta_coef": old_coef["Theta"],
        })
        compare_rows.append({
            "target": target,
            "model": "lnTheta only (new)",
            "R2_loco": new_loco["R2"], "MAE_loco": new_loco["MAE"],
            "n_folds": new_loco["n_folds"], "n_eval": new_loco["n_eval"],
            "ln_Theta_coef": new_coef["ln_Theta"], "Theta_coef": np.nan,
        })
    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(OUT_COMPARE, index=False)
    print(f"\n=== lnΘ-only vs lnΘ+Θ 对照 ===")
    print(compare_df.to_string(index=False))
    print(f"\n已写出 {OUT_COMPARE}")


if __name__ == "__main__":
    main()
