# -*- coding: utf-8 -*-
"""32 同炉配对 M vs L 差值分布(补充/敏感性分析)
====================================================================
背景:稿件正文用组均值±合并标准差(如 `manuscript_numbers.csv` 里
`table3.M.D_sec_mean/std`≈10.33±1.16µm、`table3.L.D_sec_mean/std`≈10.46±
1.03µm)论证"M 达到与 L 相当的二次颗粒尺寸/压实密度"。但组内合并标准差
(±1.16/±1.03µm)比组间均值差(10.46−10.33≈0.13µm)还大——这种比较方式
掩盖了 M/L 本来是同炉共烧(每个 `condition_id` 下 M 和 L 各一个样本)的
配对结构,统计上更合适的是同炉配对差值检验,而不是比较合并组统计量。

本脚本用 `data/processed/master_table.csv`(81样,27 condition_id,每条件
S/M/L 各一样本),对每个 condition_id 算 (D_sec_M − D_sec_L) 和
(compaction_density_M − compaction_density_L) 的配对差,报告:
  - 27 个配对差的中位数、IQR(Q1/Q3)
  - 精确符号检验(scipy.stats.binomtest,正/负计数 out of 27,H0: p=0.5)
    ——字面意义的符号检验,不是 Wilcoxon 符号秩检验

如中位配对差接近零且符号检验不显著:说明"M≈L"在配对/条件层面是真实
成立的,不是合并组统计量掩盖噪声的假象。如符号检验显著:如实报告,并
指出稿件"M≈L"表述需要加限定语。两种结果都如实报告,不预设结论。

**不修改** 任何 `src/nfm/` 代码。

输出:
  data/interim/s5_3_paired_mvsl_diff.csv        27条件×2变量的配对差明细
  data/interim/s5_3_paired_mvsl_summary.csv     中位数/IQR/符号检验汇总
"""
import _bootstrap  # noqa

import numpy as np
import pandas as pd
from scipy.stats import binomtest

MASTER_TABLE = "data/processed/master_table.csv"
OUT_DIFF = "data/interim/s5_3_paired_mvsl_diff.csv"
OUT_SUMMARY = "data/interim/s5_3_paired_mvsl_summary.csv"

VARS = ["D_sec", "compaction_density"]


def main():
    df = pd.read_csv(MASTER_TABLE)

    n_cond = df["condition_id"].nunique()
    print(f"condition_id 数量: {n_cond}")

    m = df[df["precursor"] == "M"].set_index("condition_id")
    l = df[df["precursor"] == "L"].set_index("condition_id")
    common = sorted(set(m.index) & set(l.index))
    assert len(common) == n_cond == 27, (
        f"预期 27 个 condition_id 下 M/L 均齐全,实际交集 n={len(common)}, "
        f"condition_id 总数 n={n_cond} —— 配对结构不完整,需先核实原因。"
    )

    diff_rows = []
    for cond in common:
        row = {"condition_id": cond}
        for v in VARS:
            row[f"{v}_M"] = m.loc[cond, v]
            row[f"{v}_L"] = l.loc[cond, v]
            row[f"diff_{v}"] = m.loc[cond, v] - l.loc[cond, v]
        diff_rows.append(row)
    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv(OUT_DIFF, index=False)
    print(f"已写出 {OUT_DIFF}")

    summary_rows = []
    for v in VARS:
        d = diff_df[f"diff_{v}"].to_numpy(float)
        n = len(d)
        median = float(np.median(d))
        q1 = float(np.percentile(d, 25))
        q3 = float(np.percentile(d, 75))
        iqr = q3 - q1
        n_pos = int(np.sum(d > 0))
        n_neg = int(np.sum(d < 0))
        n_zero = int(np.sum(d == 0))
        # 精确符号检验:正/负计数 out of (n_pos+n_neg),H0 p=0.5,双侧
        n_eff = n_pos + n_neg
        res = binomtest(n_pos, n_eff, p=0.5, alternative="two-sided")
        p_value = res.pvalue

        print(f"\n=== {v}: 27条件配对差 (M - L) ===")
        print(f"  median={median:+.4f}  Q1={q1:+.4f}  Q3={q3:+.4f}  IQR={iqr:.4f}")
        print(f"  n_pos={n_pos}  n_neg={n_neg}  n_zero={n_zero}  "
              f"(n_eff={n_eff} used in sign test)")
        print(f"  符号检验(binomtest, two-sided, H0 p=0.5): p={p_value:.4f}")

        summary_rows.append({
            "variable": v, "n": n, "median_diff": median, "Q1": q1, "Q3": q3, "IQR": iqr,
            "n_pos": n_pos, "n_neg": n_neg, "n_zero": n_zero, "n_eff": n_eff,
            "sign_test_p": p_value,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    print(f"\n已写出 {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
