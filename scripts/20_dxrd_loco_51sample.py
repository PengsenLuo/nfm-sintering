# -*- coding: utf-8 -*-
"""
20 D_XRD 5特征模型系数 + LOCO CV,在唯一合法纳米层口径(51样/17条件)上重算
==============================================================================
背景:论文 §3.6 表6的 D_XRD
系数行来自纳米层口径定档(2026-07-30)**之前**的旧算,样本量口径不明确
——`scripts/07d_dxrd_baseline_sweep.py`/`theta_precursor_baseline()` 直接
对 `data/processed/master_table.csv` 的 81 行做 `dropna(subset=[...,"D_XRD"])`,
未过滤 `xrd_instrument`,而仪器2(MiniFlex)的 D_XRD 大多数不是 NaN
(30 样里只有 3 样 NaN),因此旧算大概率混入了仪器2样本——与 §2.2.1/§3.3
现在唯一声明的口径(仅仪器1,51样=17条件)自相矛盾。

本脚本只在 `nfm.nano_layer.nano_layer_frame(df, require_reliable=True)`
(51样,仅 xrd_instrument==1 且 D_XRD_reliable==True)上重新拟合,产出
可与表6直接对照的系数 + LOCO(留整条件)交叉验证 R²。

模型规格:`D_XRD ~ lnΘ + Θ + precursor(one-hot: S/M/L)`,用
`sklearn.linear_model.Ridge(alpha=1, fit_intercept=True)`(不做特征标准化)
在**全部**one-hot三列(S/M/L,不 drop_first)+ 显式截距上拟合——这是复刻
表6原表格的列结构(截距、lnΘ、Θ、L、M、S 六个数),而不是本仓库另一套
`nfm.evaluation.cross_validation.theta_precursor_baseline`(那是 Ridge +
StandardScaler + drop_first 的正则化基线,系数不可比)。选用 Ridge(alpha=1)
而非普通最小二乘(OLS/LinearRegression)的原因:表6另外两行
(compaction_density/D_sec)独立复核后确认本来就是 Ridge(alpha=1)原始尺度
算出的、与稿件已发表系数逐位吻合,同一规格下 D_XRD 的 LOCO 才能复现论文
原稿的 R²=0.494(改用 OLS 重算得到的是 0.4886,对不上)。根本原因:
intercept 列与三个 one-hot 列本身共线(和恒为1),`LinearRegression`
(内部走最小二乘 lstsq)给出的是一个隐式依赖实现细节的最小范数解,不是
唯一定义的解;`Ridge(alpha=1)` 通过正则化项打破共线性,解是确定性的。
详见 `src/nfm/dxrd_baseline.py` 顶部说明。

本脚本不再内联实现上述模型/LOCO 逻辑,已提取为
`nfm.dxrd_baseline.ridge_coef_loco()`(从最初的 D_XRD 专用
`dxrd_ols_loco()` 推广为任意 target 的 Ridge 版本)——那是这套
计算的唯一实现,本脚本只是读 master_table.csv、调用它(`y_col="D_XRD"`)、
把返回的 dict 打印/落盘成下面两个 csv 的薄 I/O 包装。这样做是因为
`data/interim/target_baseline_comparison.csv` 的 D_XRD 行此前另有一套 Ridge
(标准化尺度)实现给出不同数字,稿件表6/表7必须同源,详见
`src/nfm/dxrd_baseline.py` 顶部说明与
`scripts/21_patch_target_baseline_dxrd.py`。

LOCO:按 `condition_id` 分组留一(17 折,每折留出该条件下 S/M/L 全部
3 个样本),复用 `nfm.evaluation.cross_validation.loco_splits`(与
`LeaveOneGroupOut` 完全一致,只是分组自动从 51 样本中的 17 个
`condition_id` 取值,不是全 27 条件)。

不使用 D_sec/Θ 之外的 81 样本口径列——本脚本自变量只有 lnΘ、Θ、
precursor one-hot,因变量只有 D_XRD,不涉及 D_sec。

输出:
  - stdout 打印系数表与 LOCO 汇总
  - data/interim/dxrd_loco_51sample_coef.csv   (系数表)
  - data/interim/dxrd_loco_51sample_folds.csv  (逐折预测明细)
"""
import _bootstrap  # noqa

import os

import pandas as pd

from nfm.dxrd_baseline import ridge_coef_loco

MASTER_TABLE = "data/processed/master_table.csv"
OUT_COEF = "data/interim/dxrd_loco_51sample_coef.csv"
OUT_FOLDS = "data/interim/dxrd_loco_51sample_folds.csv"


def main():
    df = pd.read_csv(MASTER_TABLE)
    out = ridge_coef_loco(df, "D_XRD")

    print(f"[口径确认] n={out['n']} 样本, n_cond={out['n_cond']} 条件"
          f"(仅 xrd_instrument==1,D_XRD_reliable==True)")

    # 模型规格去掉了Θ项;前驱体编码从"S为参照类"改为 sum(偏差)编码;
    # 下面的打印/落盘随 nfm.dxrd_baseline.ridge_coef_loco() 的 coef 字典
    # 同步更新。
    print("\n=== D_XRD 3特征模型系数(51样/17条件,对照表6格式) ===")
    coef = out["coef"]
    print(f"  截距(总体均值): {coef['intercept']:+.4f}")
    print(f"  lnΘ: {coef['ln_Theta']:+.4f}")
    print(f"  M(相对总体均值): {coef['precursor_M']:+.4f}")
    print(f"  L(相对总体均值): {coef['precursor_L']:+.4f}")
    print(f"  S(相对总体均值,推算): {-(coef['precursor_M']+coef['precursor_L']):+.4f}")
    print(f"  (全数据同集拟合 R²={out['r2_insample']:.4f},仅供参考,非样本外指标)")

    coef_row = {
        "截距": coef["intercept"], "lnΘ": coef["ln_Theta"],
        "M": coef["precursor_M"], "L": coef["precursor_L"],
    }
    coef_df = pd.DataFrame([coef_row])
    coef_df.insert(0, "target", "D_XRD")
    coef_df.insert(1, "n", out["n"])
    coef_df.insert(2, "n_cond", out["n_cond"])
    coef_df["r2_insample"] = out["r2_insample"]
    coef_df["loco_R2"] = out["R2"]
    coef_df["loco_MAE_nm"] = out["MAE"]
    coef_df["loco_n_folds"] = out["n_folds"]
    coef_df["loco_n_eval"] = out["n_eval"]

    print(f"\n=== LOCO(留整条件,{out['n_folds']} 折,每折留出该条件下 S/M/L 全部样本) ===")
    print(f"  R²={out['R2']:.4f}  MAE={out['MAE']:.4f} nm  n_eval={out['n_eval']}")

    os.makedirs("data/interim", exist_ok=True)
    coef_df.to_csv(OUT_COEF, index=False)
    pd.DataFrame(out["fold_rows"]).to_csv(OUT_FOLDS, index=False)
    print(f"\n已写出 {OUT_COEF}")
    print(f"已写出 {OUT_FOLDS}")


if __name__ == "__main__":
    main()
