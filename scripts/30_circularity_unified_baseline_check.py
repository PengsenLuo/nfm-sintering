# -*- coding: utf-8 -*-
"""30 circularity 走统一 Ridge/LOCO 路径 vs 冻结快照(补充/敏感性分析)
====================================================================
⚠️ **本脚本已被后续生产实现取代,已过时,不要重跑**:当时要回答的问题是
"circularity 若改走统一路径会怎样"这个敏感性问题;后来这个答案已经落地
成生产实现本身——
`scripts/22_build_manuscript_numbers.py::build_table7_loco` 现在直接内联
调用 `nfm.dxrd_baseline.ridge_coef_loco(df,"circularity")`,`table7.
circularity.*` 已经就是这里说的"统一路径"数字,不再来自下面第66行断言
里比较的快照文件。若重跑本脚本,第66-69行的断言(比较 manuscript_numbers.csv
与快照是否一致)会失败——那不是 bug,是这个问题已经不需要再问了。保留本
脚本文件只作历史记录(当时的独立探针结果),不删除、不修复。三个 coef
访问点(`c['Theta']`/`c['precursor_S']`)在下方保持原样(与当时实际跑出的
`data/interim/s5_2_circularity_unified_vs_snapshot.csv` 对应),因为
`nfm.dxrd_baseline.ridge_coef_loco()` 的返回结构后来已经改变(去掉了
`Theta`/`precursor_S` 两个 key),原样保留意味着这两行现在会 KeyError——
这是预期的、留作"脚本已过时"的信号,不是需要修的缺陷。

背景(历史,当时的问题描述):compaction_density/D_sec/D_XRD 的表6/表7数字都来自统一实现
`nfm.dxrd_baseline.ridge_coef_loco`(由 `scripts/24_table6_ridge_baseline.py`
调用)。但 circularity 的表7 LOCO 数字来自另一条代码路径的冻结快照
`manuscript/numbers/sources/target_baseline_comparison.snapshot.csv`
(该快照由 `scripts/07e_target_baseline_table.py` 产出,那个脚本跑的是更重的
PINN 全套交叉验证对照,不是单纯 Ridge 基线;快照里 circularity 一行的
`baseline_R2`/`baseline_MAE` 列才是这里要对照的"非统一路径"数字)。

这是"同一张稿件表格、两条不同代码路径"的风险点。本脚本把 circularity
接到与 compaction_density/D_sec/D_XRD 完全相同的统一路径
(`nfm.dxrd_baseline.ridge_coef_loco`,81样,不做纳米层限定——circularity
是 SEM 形貌描述量,不是 XRD 量),报告统一路径算出来的 LOCO R²/MAE/系数,
和冻结快照里的数字精确对比出差异。

**不修改** `scripts/22_build_manuscript_numbers.py`、
`manuscript/numbers/manuscript_numbers.csv`、
`manuscript/numbers/sources/target_baseline_comparison.snapshot.csv`——
仅读取后两者作对照基准。

输出:
  data/interim/s5_2_circularity_unified_vs_snapshot.csv
"""
import _bootstrap  # noqa

import pandas as pd

from nfm.dxrd_baseline import ridge_coef_loco

MASTER_TABLE = "data/processed/master_table.csv"
MANUSCRIPT_NUMBERS = "manuscript/numbers/manuscript_numbers.csv"
SNAPSHOT = "manuscript/numbers/sources/target_baseline_comparison.snapshot.csv"
OUT_COMPARE = "data/interim/s5_2_circularity_unified_vs_snapshot.csv"

TARGET = "circularity"


def main():
    df = pd.read_csv(MASTER_TABLE)

    out = ridge_coef_loco(df, TARGET)
    c = out["coef"]
    print(f"=== {TARGET} 走统一路径 nfm.dxrd_baseline.ridge_coef_loco "
          f"(n={out['n']}, n_cond={out['n_cond']}) ===")
    print(f"  截距={c['intercept']:+.6f} lnΘ={c['ln_Theta']:+.6f} "
          f"Θ={c['Theta']:+.6f} L={c['precursor_L']:+.6f} "
          f"M={c['precursor_M']:+.6f} S={c['precursor_S']:+.6f}")
    print(f"  LOCO R²={out['R2']:.6f}  MAE={out['MAE']:.6f}  "
          f"n_folds={out['n_folds']}  n_eval={out['n_eval']}")

    # ---- 读快照(非统一路径的现行稿件来源)----
    snap = pd.read_csv(SNAPSHOT)
    snap_row = snap[snap["target"] == TARGET].iloc[0]
    snap_r2 = float(snap_row["baseline_R2"])
    snap_mae = float(snap_row["baseline_MAE"])
    snap_n_folds = int(snap_row["baseline_n_folds"])

    # ---- 也读 manuscript_numbers.csv 里的 table7.circularity 行做交叉核实 ----
    mnum = pd.read_csv(MANUSCRIPT_NUMBERS)
    mn_r2 = float(mnum.loc[mnum["id"] == "table7.circularity.R2", "value"].iloc[0])
    mn_mae = float(mnum.loc[mnum["id"] == "table7.circularity.MAE", "value"].iloc[0])
    mn_n_folds = int(mnum.loc[mnum["id"] == "table7.circularity.n_folds", "value"].iloc[0])

    assert abs(mn_r2 - snap_r2) < 1e-9 and abs(mn_mae - snap_mae) < 1e-9, (
        "manuscript_numbers.csv 的 table7.circularity 与快照文件数字不一致,"
        "需先核实(理论上应逐位相同,manuscript_numbers.csv 只是从快照里搬运)。"
    )
    print(f"\n=== 冻结快照(scripts/07e_target_baseline_table.py 路径,"
          f"= manuscript_numbers.csv table7.circularity) ===")
    print(f"  R²={snap_r2:.6f}  MAE={snap_mae:.6f}  n_folds={snap_n_folds}")

    d_r2 = out["R2"] - snap_r2
    d_mae = out["MAE"] - snap_mae
    print(f"\n=== 差异(统一路径 - 快照) ===")
    print(f"  ΔR²={d_r2:+.6f}  ΔMAE={d_mae:+.6f}")

    compare_df = pd.DataFrame([
        {"source": "unified (nfm.dxrd_baseline.ridge_coef_loco)",
         "R2": out["R2"], "MAE": out["MAE"], "n_folds": out["n_folds"], "n": out["n"],
         "intercept": c["intercept"], "ln_Theta": c["ln_Theta"], "Theta": c["Theta"],
         "precursor_L": c["precursor_L"], "precursor_M": c["precursor_M"], "precursor_S": c["precursor_S"]},
        {"source": "snapshot (scripts/07e_target_baseline_table.py)",
         "R2": snap_r2, "MAE": snap_mae, "n_folds": snap_n_folds, "n": out["n"],
         "intercept": None, "ln_Theta": None, "Theta": None,
         "precursor_L": None, "precursor_M": None, "precursor_S": None},
        {"source": "diff (unified - snapshot)",
         "R2": d_r2, "MAE": d_mae, "n_folds": out["n_folds"] - snap_n_folds, "n": None,
         "intercept": None, "ln_Theta": None, "Theta": None,
         "precursor_L": None, "precursor_M": None, "precursor_S": None},
    ])
    compare_df.to_csv(OUT_COMPARE, index=False)
    print(f"\n已写出 {OUT_COMPARE}")


if __name__ == "__main__":
    main()
