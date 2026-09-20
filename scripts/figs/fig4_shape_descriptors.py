# -*- coding: utf-8 -*-
"""fig4:四个形状描述子(D_f/solidity/elongation/roughness)vs Θ,分前驱体,
2×2 子图,散点 + 简单线性拟合线 + 图例标注 T7 已算好的 Spearman ρ 与
95% bootstrap CI。

**数据**:
  - 散点本身:`data/interim/sem_shape_descriptors_summary.csv`(81 样本,
    500× SEM 形状描述子逐样汇总,T7 `scripts/17_shape_descriptor_full_analysis.py`
    的输入)。
  - ρ 与 95% bootstrap CI:`data/interim/shape_descriptor_full_analysis.csv`
    的 `section=='spearman_theta'` 部分,`metric=='rho'` 行的 value/ci_low/
    ci_high——**不重新计算一套不同口径的置信区间**,直接读用。

**"拟合线 + 置信带"的处理方式**(任务书原话的两难:T7 算的是相关系数的
CI,不是回归线本身的逐点置信带,不能张冠李戴):图上只画**点估计 ρ 对应
方向的简单线性拟合线**(OLS,仅用于指示趋势方向和斜率符号,不代表统计
显著性);**CI 以文字形式写在图例里**(`ρ=+0.70 [0.40, 0.87]`),即
"散点+拟合线+图例标注 ρ[CI]"这一任务书明确认可的备选简单形式,不编造
逐点置信带。
"""
import _bootstrap  # noqa

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _style import (PRECURSOR_COLOR, PRECURSOR_MARKER, PRECURSOR_LABEL,
                     THETA_NOTE, save_fig, skip)

NAME = "fig4_shape_descriptors"

SUMMARY_CSV = Path("data/interim/sem_shape_descriptors_summary.csv")
FULL_ANALYSIS_CSV = Path("data/interim/shape_descriptor_full_analysis.csv")

DESCRIPTORS = [
    ("D_f", "分形维数 D_f"),
    ("solidity", "solidity(实心度)"),
    ("elongation", "elongation(伸长率)"),
    ("roughness", "roughness_ratio(粗糙度比)"),
]
PRECURSORS = ["S", "M", "L"]


def _load_rho_ci() -> dict:
    """{(target, group): (rho, ci_low, ci_high, p)}"""
    df = pd.read_csv(FULL_ANALYSIS_CSV)
    sub = df[df["section"] == "spearman_theta"]
    out = {}
    for (target, group), g in sub.groupby(["target", "group"]):
        g = g.set_index("metric")
        rho = g.loc["rho", "value"]
        ci_low = g.loc["rho", "ci_low"]
        ci_high = g.loc["rho", "ci_high"]
        p = g.loc["p_value_asymptotic", "value"] if "p_value_asymptotic" in g.index else np.nan
        out[(target, group)] = (rho, ci_low, ci_high, p)
    return out


def main():
    if not (SUMMARY_CSV.exists() and FULL_ANALYSIS_CSV.exists()):
        skip(NAME, "sem_shape_descriptors_summary.csv 或 "
                   "shape_descriptor_full_analysis.csv 缺失")
        return

    df = pd.read_csv(SUMMARY_CSV)
    rho_ci = _load_rho_ci()

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.0))
    for ax, (col, label) in zip(axes.flat, DESCRIPTORS):
        for prec in PRECURSORS:
            g = df[df["precursor"] == prec]
            color = PRECURSOR_COLOR[prec]
            marker = PRECURSOR_MARKER[prec]
            rho, ci_lo, ci_hi, p = rho_ci.get((col, prec), (np.nan, np.nan, np.nan, np.nan))
            leg = (f"{PRECURSOR_LABEL[prec]}  ρ={rho:+.2f} "
                  f"[{ci_lo:+.2f},{ci_hi:+.2f}]")
            ax.scatter(g["Theta"], g[col], color=color, marker=marker, s=42,
                      edgecolor="black", linewidth=0.35, label=leg, zorder=3)
            if g["Theta"].nunique() >= 2:
                b, a = np.polyfit(g["Theta"], g[col], 1)
                x_line = np.linspace(g["Theta"].min(), g["Theta"].max(), 50)
                ax.plot(x_line, a + b * x_line, color=color, linestyle="--",
                       linewidth=1.4, alpha=0.85, zorder=2)
        ax.set_xlabel("Θ (h)")
        ax.set_ylabel(label)
        ax.legend(loc="best", fontsize=7.2)

    fig.tight_layout()

    caption = (
        "**数据**:散点为 `data/interim/sem_shape_descriptors_summary.csv`"
        "(81 样本,500× SEM 逐样汇总,来自 T7 `scripts/17_shape_descriptor_"
        "full_analysis.py` 的输入);图例 ρ 与 95% bootstrap CI 取自 "
        "`data/interim/shape_descriptor_full_analysis.csv`"
        "(`section=='spearman_theta'`,`metric=='rho'` 行,n_boot=5000,"
        "**未重新计算**,与 T7 报告数值一致)。虚线为各组 Θ~描述子简单 OLS "
        "线性拟合(仅示意方向/斜率符号,不代表显著性——显著性以图例 CI 是否"
        "跨零判断)。\n\n"
        "**结论呼应论文表4**:roughness_ratio 三组 CI 均不跨零,是唯一跨"
        "三组一致的形貌演化信号;S 组在 D_f/solidity/elongation 三个描述子"
        "上 CI 均不跨零、演化模式完整,M/L 组多数不跨零 CI 落在零附近——"
        "重构幅度与前驱体尺度呈反向关系(详见 "
        "`reports/shape_descriptor_full_analysis_report.md`)。\n\n"
        f"**Θ 标定状态**:{THETA_NOTE}"
    )
    save_fig(fig, NAME, caption)
    plt.close(fig)


if __name__ == "__main__":
    main()
