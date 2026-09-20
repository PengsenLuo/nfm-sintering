# -*- coding: utf-8 -*-
"""fig7:条件级残差点图(27 条件 × S/M/L)。

数据:`data/interim/sieving_confound_diagnostic.csv`(T1,
`scripts/13_sieving_confound.py`),直接读用 `residual_precursor_adj_full27`
列(扣除前驱体主效应后的 compaction_density 条件级残差,81 样本口径,
**不重新定义残差计算逻辑**)。

横轴用 Θ(而非条件序号),纵轴为残差;按前驱体着色/形状双编码。

2026-08-02 决策:论文正文不再讨论"过筛"这一构念,故本图不再标出具体
过筛条件,图注也不再展开结构性混杂的具体细节;仅保留"无重复设计、
温度与条件编号存在共线,该现象降为讨论项"这一层限定。T1 完整分析
(含过筛条件识别与 Fisher exact 检验)存档于
`reports/sieving_confound_report.md`,该文件本身不随本次口径调整改动。
"""
import _bootstrap  # noqa

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _style import (PRECURSOR_COLOR, PRECURSOR_MARKER, PRECURSOR_LABEL,
                     THETA_NOTE, save_fig, skip)

NAME = "fig7_condition_residuals"

DIAG_CSV = Path("data/interim/sieving_confound_diagnostic.csv")


def main():
    if not DIAG_CSV.exists():
        skip(NAME, f"{DIAG_CSV} 不存在,需先跑 scripts/13_sieving_confound.py")
        return
    df = pd.read_csv(DIAG_CSV)
    col = "residual_precursor_adj_full27"

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for prec in ["S", "M", "L"]:
        g = df[df["precursor"] == prec]
        color = PRECURSOR_COLOR[prec]
        marker = PRECURSOR_MARKER[prec]
        ax.scatter(g["Theta"], g[col], color=color, marker=marker, s=60,
                  edgecolor="black", linewidth=0.4,
                  label=f"{PRECURSOR_LABEL[prec]} (n={len(g)})", zorder=3)

    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":", zorder=1)
    ax.set_xlabel("Θ (归一化热暴露量, h)")
    ax.set_ylabel("compaction_density 条件级残差\n(扣除前驱体主效应, g/cm³)")
    ax.legend(loc="best", fontsize=8.5)

    n_cond = df["condition_id"].nunique()
    caption = (
        f"**数据**:`data/interim/sieving_confound_diagnostic.csv` 的 "
        f"`residual_precursor_adj_full27` 列(T1,"
        f"`scripts/13_sieving_confound.py`;compaction_density 扣除前驱体"
        f"主效应后的样本级残差,81 样 = {n_cond} 条件 × S/M/L,本图直接沿用,"
        f"不重新定义残差计算逻辑)。\n\n"
        f"**限定**:本设计无重复(每条件每前驱体仅 1 个样本),且温度与条件"
        f"编号之间存在共线性,因此条件级残差跨前驱体的重现性只能作讨论项,"
        f"不构成独立统计证据。\n\n"
        f"**Θ 标定状态**:{THETA_NOTE}"
    )
    save_fig(fig, NAME, caption)
    plt.close(fig)


if __name__ == "__main__":
    main()
