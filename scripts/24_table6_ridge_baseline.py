# -*- coding: utf-8 -*-
"""24 表6系数 + 表7 LOCO(compaction_density/D_sec/D_XRD 三个目标,统一实现)
====================================================================
这三个目标同时出现在稿件表6(系数)与表7(LOCO性能)里,统一调用
nfm.dxrd_baseline.ridge_coef_loco(),coefficient 与 LOCO 报告保证来自
同一个拟合对象,避免同一格数字出自两套独立实现导致不一致。

模型规格不含 Θ 项(lnΘ 与 Θ 是同一底层量的两种单调变换,系数各自不可
解读)。D_XRD 限定到 nano_layer_frame(51样/17条件),compaction_density/
D_sec 用全部81样/27条件。

前驱体编码采用 **sum(偏差)编码**,而非"以 S 为参照类"的编码——参照类
编码下 S 被吸收进不受 Ridge 惩罚的截距,S/M/L 三者在正则化项下不再对称
受罚,对 `D_sec`(S 是最极端值)造成实质伤害(LOCO 变差);sum 编码下
三个前驱体重新对称受罚,`D_sec` 的 LOCO 恢复且反超切换前。系数含义:
`M`/`L` 是相对**三个前驱体总体均值**的偏差,S 的偏差 = -(M+L)
(sum-to-zero 约束,未单列)。

输出:
  data/interim/table6_ridge_coef.csv   系数表(3行,一行一个target)
  data/interim/table6_ridge_loco.csv   LOCO表(3行)
"""
import _bootstrap  # noqa

import pandas as pd

from nfm.dxrd_baseline import ridge_coef_loco

MASTER_TABLE = "data/processed/master_table.csv"
TARGETS = ["compaction_density", "D_sec", "D_XRD"]
OUT_COEF = "data/interim/table6_ridge_coef.csv"
OUT_LOCO = "data/interim/table6_ridge_loco.csv"


def main():
    df = pd.read_csv(MASTER_TABLE)
    coef_rows, loco_rows = [], []
    for target in TARGETS:
        out = ridge_coef_loco(df, target)
        c = out["coef"]
        print(f"\n=== {target} (n={out['n']}, n_cond={out['n_cond']}) ===")
        print(f"  截距(总体均值)={c['intercept']:+.4f} lnΘ={c['ln_Theta']:+.4f} "
              f"M(相对总体均值)={c['precursor_M']:+.4f} "
              f"L(相对总体均值)={c['precursor_L']:+.4f} "
              f"S(相对总体均值,推算)={-(c['precursor_M']+c['precursor_L']):+.4f}")
        print(f"  LOCO R²={out['R2']:.4f}  MAE={out['MAE']:.4f}  "
              f"n_folds={out['n_folds']}  n_eval={out['n_eval']}")
        coef_rows.append({
            "target": target, "n": out["n"], "n_cond": out["n_cond"],
            "intercept": c["intercept"], "ln_Theta": c["ln_Theta"],
            "M": c["precursor_M"], "L": c["precursor_L"],
            "r2_insample": out["r2_insample"],
        })
        loco_rows.append({
            "target": target, "R2": out["R2"], "MAE": out["MAE"],
            "n_folds": out["n_folds"], "n_eval": out["n_eval"],
        })
    pd.DataFrame(coef_rows).to_csv(OUT_COEF, index=False)
    pd.DataFrame(loco_rows).to_csv(OUT_LOCO, index=False)
    print(f"\n已写出 {OUT_COEF}\n已写出 {OUT_LOCO}")


if __name__ == "__main__":
    main()
