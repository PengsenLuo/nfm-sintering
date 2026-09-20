# -*- coding: utf-8 -*-
"""36 27条件同炉配对 M-vs-L 差值重新计数与算术核对
====================================================================
背景:稿件 v5 版正文写"27 个条件中 M 高于 L 出现
15 次,但低于 L 出现 10 次,恰好相等 1 次"——15+10+1=26≠27,存在计数
错误。此前从 `master_table.csv` 直接重算得 compaction_density
15/11/1,D_sec 14/13/0(0 容差)。此前 `scripts/32_paired_mvsl_difference.py`
已经算过一次配对差值+符号检验(compaction_density 15/11/1、D_sec 14/13/0,
与本轮重算数字一致)——本脚本是**同一份数据的两个侧面**:上一轮脚本关注
符号检验统计推断,本脚本关注**计数本身的核对**(补上 1e-3 容差版本,并把
结果正式写入 `manuscript_numbers.csv` 供正文直接引用,取代那句算术错误的
原文)。

产出:
  data/interim/mvsl_paired_recount.csv        27行明细(两个变量×两种容差)
  data/interim/mvsl_paired_recount_summary.csv 汇总(供 scripts/22 读取)
  reports/s5_supplementary_baselines.md        追加本节小节(不覆盖已有内容)
"""
import _bootstrap  # noqa

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

MASTER_TABLE = "data/processed/master_table.csv"
OUT_DETAIL_CSV = Path("data/interim/mvsl_paired_recount.csv")
OUT_SUMMARY_CSV = Path("data/interim/mvsl_paired_recount_summary.csv")
REPORT_MD = Path("reports/s5_supplementary_baselines.md")

VARIABLES = ["compaction_density", "D_sec"]
TIE_TOLERANCES = {"tol0": 0.0, "tol1e-3": 1e-3}

# 既有参考数字(0 容差),不采信,只作对照
COWORK_TOL0 = dict(
    compaction_density=dict(n_pos=15, n_neg=11, n_zero=1),
    D_sec=dict(n_pos=14, n_neg=13, n_zero=0),
)
COWORK_TOL1E3 = dict(
    D_sec=dict(n_pos=13, n_neg=13, n_zero=1),
)


def main():
    df = pd.read_csv(MASTER_TABLE)
    pivot = {}
    for var in VARIABLES:
        p = df.pivot(index="condition_id", columns="precursor", values=var)
        assert p.shape[0] == 27, f"{var}: 预期 27 条件,实际 {p.shape[0]}"
        pivot[var] = p

    detail_rows = []
    summary_rows = []
    for var in VARIABLES:
        p = pivot[var]
        diff = (p["M"] - p["L"]).rename(f"{var}_diff_M_minus_L")
        for cond, d in diff.items():
            detail_rows.append(dict(condition_id=cond, variable=var, diff_M_minus_L=d))

        n = len(diff)
        median_diff = float(diff.median())
        q1, q3 = float(diff.quantile(0.25)), float(diff.quantile(0.75))
        iqr = q3 - q1

        for tol_name, tol in TIE_TOLERANCES.items():
            n_pos = int((diff > tol).sum())
            n_neg = int((diff < -tol).sum())
            n_zero = int((diff.abs() <= tol).sum())
            assert n_pos + n_neg + n_zero == n, (
                f"{var}/{tol_name}: n_pos+n_neg+n_zero={n_pos+n_neg+n_zero} != n={n},"
                "计数逻辑有误,停下来排查")
            # 符号检验:剔除恰好为0(容差内)的样本,对 n_pos/(n_pos+n_neg) 做双侧精确二项检验
            if n_pos + n_neg > 0:
                sign_p = float(binomtest(n_pos, n_pos + n_neg, 0.5, alternative="two-sided").pvalue)
            else:
                sign_p = float("nan")
            summary_rows.append(dict(
                variable=var, tie_tolerance=tol_name, tol_value=tol,
                n=n, median_diff=median_diff, q1=q1, q3=q3, iqr=iqr,
                n_pos=n_pos, n_neg=n_neg, n_zero=n_zero, sign_test_p=sign_p,
            ))
            print(f"{var} [{tol_name}]: median={median_diff:+.4f}, IQR=[{q1:+.4f},{q3:+.4f}], "
                 f"n_pos={n_pos}, n_neg={n_neg}, n_zero={n_zero}, sign_p={sign_p:.4f}, "
                 f"sum={n_pos+n_neg+n_zero} (预期27)")

    detail_df = pd.DataFrame(detail_rows)
    summary_df = pd.DataFrame(summary_rows)
    OUT_DETAIL_CSV.parent.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(OUT_DETAIL_CSV, index=False)
    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)
    print(f"\n[done] {OUT_DETAIL_CSV}\n[done] {OUT_SUMMARY_CSV}")

    # ---- 与既有参考数字对照 ----
    print("\n=== 与既有参考数字对照(不采信,独立复算为准) ===")
    for var in VARIABLES:
        row0 = summary_df[(summary_df["variable"] == var) & (summary_df["tie_tolerance"] == "tol0")].iloc[0]
        claim0 = COWORK_TOL0.get(var)
        if claim0:
            match0 = (row0["n_pos"] == claim0["n_pos"] and row0["n_neg"] == claim0["n_neg"]
                      and row0["n_zero"] == claim0["n_zero"])
            print(f"  {var} tol0: 本轮={row0['n_pos']}/{row0['n_neg']}/{row0['n_zero']}  "
                 f"既有={claim0['n_pos']}/{claim0['n_neg']}/{claim0['n_zero']}  "
                 f"{'一致' if match0 else '不一致'}")
        row1 = summary_df[(summary_df["variable"] == var) & (summary_df["tie_tolerance"] == "tol1e-3")].iloc[0]
        claim1 = COWORK_TOL1E3.get(var)
        if claim1:
            match1 = (row1["n_pos"] == claim1["n_pos"] and row1["n_neg"] == claim1["n_neg"]
                      and row1["n_zero"] == claim1["n_zero"])
            print(f"  {var} tol1e-3: 本轮={row1['n_pos']}/{row1['n_neg']}/{row1['n_zero']}  "
                 f"既有={claim1['n_pos']}/{claim1['n_neg']}/{claim1['n_zero']}  "
                 f"{'一致' if match1 else '不一致'}")

    # ---- v5 稿件算术错误核对(15+10+1=26≠27) ----
    cd0 = summary_df[(summary_df["variable"] == "compaction_density") & (summary_df["tie_tolerance"] == "tol0")].iloc[0]
    print(f"\n=== v5 稿件算术错误核对 ===")
    print(f"  v5 原文声称: M>L 15次, M<L 10次, 相等1次, 合计 15+10+1=26 (≠27, 算术错误)")
    print(f"  本轮独立复算(compaction_density, 0容差): "
         f"M>L {cd0['n_pos']}次, M<L {cd0['n_neg']}次, 相等 {cd0['n_zero']}次, "
         f"合计 {cd0['n_pos']}+{cd0['n_neg']}+{cd0['n_zero']}={cd0['n_pos']+cd0['n_neg']+cd0['n_zero']}")

    append_report(summary_df, detail_df)
    print(f"\n[done,追加] {REPORT_MD}")


def append_report(summary_df: pd.DataFrame, detail_df: pd.DataFrame):
    lines = []
    lines.append("\n---\n\n## 27条件配对 M-vs-L 差值重新计数(算术核对)\n\n")
    lines.append("> 由 `scripts/36_paired_mvsl_recount.py` 追加。"
                 "稿件 v5 版正文写 M>L 15次、M<L 10次、"
                 "相等1次(合计15+10+1=26≠27,存在计数错误)。本节从 "
                 "`master_table.csv` 自动重新统计,给出可直接引用的正确数字,并"
                 "**同时给出 0 容差与 1e-3 容差两种\"相等\"判定口径**"
                 "(浮点精确相等 vs 容差内视为相等,两者对 `D_sec` 给出不同计数,"
                 "对 `compaction_density` 一致)。\n\n")
    lines.append("**注**:本节与 `scripts/32_paired_mvsl_difference.py` 产出的上一节"
                 "是**同一份底层数据的两个"
                 "侧面**——上一节关注符号检验的统计推断结论(\"M≈L 不是系统性偏移\"),"
                 "本节关注**计数本身**(补上 1e-3 容差版本,并直接对上 v5 稿件那句"
                 "算术错误的原文)。两节数字在 0 容差口径下核对一致"
                 "(compaction_density 15/11/1,D_sec 14/13/0)。\n\n")
    lines.append("### 完整统计表(0 容差 与 1e-3 容差并列)\n\n")
    lines.append("| variable | tie容差 | n_pos(M>L) | n_neg(M<L) | n_zero(相等) | 合计 | "
                 "median(M−L) | IQR | 符号检验p |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|\n")
    for _, r in summary_df.iterrows():
        total = r["n_pos"] + r["n_neg"] + r["n_zero"]
        lines.append(f"| {r['variable']} | {r['tie_tolerance']} | {int(r['n_pos'])} | "
                     f"{int(r['n_neg'])} | {int(r['n_zero'])} | {int(total)} | "
                     f"{r['median_diff']:+.4f} | [{r['q1']:+.3f}, {r['q3']:+.3f}] | "
                     f"{r['sign_test_p']:.4f} |\n")
    lines.append("\n")
    cd0 = summary_df[(summary_df["variable"] == "compaction_density") & (summary_df["tie_tolerance"] == "tol0")].iloc[0]
    lines.append(f"**v5 算术错误的正确替换句(compaction_density,0容差)**:27 个条件中,"
                 f"M 高于 L 出现 **{int(cd0['n_pos'])}** 次,低于 L 出现 **{int(cd0['n_neg'])}** 次,"
                 f"恰好相等 **{int(cd0['n_zero'])}** 次"
                 f"({int(cd0['n_pos'])}+{int(cd0['n_neg'])}+{int(cd0['n_zero'])}=27,核对无误)。"
                 "此数字已被 `scripts/22_build_manuscript_numbers.py` 汇总收录"
                 "(id 前缀 `table5.MvsL.*`),重跑不会丢失。\n\n")
    dsec_tol0 = summary_df[(summary_df["variable"] == "D_sec") & (summary_df["tie_tolerance"] == "tol0")].iloc[0]
    dsec_tol1 = summary_df[(summary_df["variable"] == "D_sec") & (summary_df["tie_tolerance"] == "tol1e-3")].iloc[0]
    if (dsec_tol0["n_pos"], dsec_tol0["n_neg"], dsec_tol0["n_zero"]) != \
       (dsec_tol1["n_pos"], dsec_tol1["n_neg"], dsec_tol1["n_zero"]):
        lines.append(f"**D_sec 的计数对容差选择敏感**:0容差下 "
                     f"{int(dsec_tol0['n_pos'])}/{int(dsec_tol0['n_neg'])}/{int(dsec_tol0['n_zero'])}"
                     f",1e-3容差下变为 "
                     f"{int(dsec_tol1['n_pos'])}/{int(dsec_tol1['n_neg'])}/{int(dsec_tol1['n_zero'])}"
                     "——存在一个条件的 D_sec(M)−D_sec(L) 绝对值落在 (0, 1e-3] 区间内,"
                     "在浮点精确相等口径下判定为\"不相等\"、在 1e-3 容差口径下判定为"
                     "\"相等\"。正文引用时应明确写出用的是哪种容差,不要笼统写"
                     "\"相等N次\"而不说明判定标准。\n\n")
    else:
        lines.append("D_sec 在两种容差下计数一致,无需额外说明。\n\n")

    with open(REPORT_MD, "a", encoding="utf-8") as f:
        f.write("".join(lines))


if __name__ == "__main__":
    main()
