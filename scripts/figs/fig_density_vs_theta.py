# -*- coding: utf-8 -*-
"""compaction_density vs Θ,按前驱体分组(散点+趋势线)(+ _byT 版)。20 条件 × S/M/L。"""
import _bootstrap  # noqa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from _style import (MASTER_TABLE, PRECURSOR_COLOR, PRECURSOR_MARKER,
                     PRECURSOR_LABEL, TEMP_COLOR, THETA_NOTE, save_fig, skip)

NAME = "fig_density_vs_theta"
NAME_BYT = "fig_density_vs_theta_byT"


def plot_vs_theta(sub: pd.DataFrame) -> None:
    # T8 fig2b 要求:三条"水平带"视觉强调(呼应论文 §3.1"组内相关不显著、
    # 组间清晰分层"的描述),而不是暗示组内存在显著线性趋势的虚线斜率。
    # 故把原来的 OLS 斜率虚线换成组均值±1std 的水平阴影带 + 均值实线,
    # 图例同时标注组内 Spearman ρ[p],让读者看到"不显著"这件事本身。
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    x_min, x_max = sub["Theta"].min(), sub["Theta"].max()
    x_pad = 0.03 * (x_max - x_min)
    band_x = (x_min - x_pad, x_max + x_pad)
    for prec in ["S", "M", "L"]:
        g = sub[sub["precursor"] == prec]
        if g.empty:
            continue
        color = PRECURSOR_COLOR[prec]
        mean_d = g["compaction_density"].mean()
        std_d = g["compaction_density"].std()
        rho, pval = (np.nan, np.nan)
        if g["Theta"].nunique() >= 2:
            rho, pval = stats.spearmanr(g["Theta"], g["compaction_density"])

        ax.axhspan(mean_d - std_d, mean_d + std_d, xmin=0, xmax=1,
                   color=color, alpha=0.10, zorder=1)
        ax.hlines(mean_d, band_x[0], band_x[1], color=color,
                  linewidth=1.6, linestyle="-", zorder=2, alpha=0.8)
        rho_label = f"ρ={rho:+.2f}[p={pval:.2f}]" if np.isfinite(rho) else "ρ=NA"
        ax.scatter(g["Theta"], g["compaction_density"], color=color,
                   marker=PRECURSOR_MARKER[prec], s=55, edgecolor="black",
                   linewidth=0.4,
                   label=f"{PRECURSOR_LABEL[prec]} (n={len(g)}, {rho_label})", zorder=3)

    ax.set_xlim(*band_x)
    ax.set_xlabel("Θ (归一化热暴露量, h)")
    ax.set_ylabel("compaction_density (g/cm³)")
    ax.set_title("压实密度 vs Θ")
    ax.legend(loc="best", fontsize=8.5)

    n, n_cond = len(sub), sub["condition_id"].nunique()
    caption = (
        f"**数据**:`compaction_density`,n={n} 样 = {n_cond} 条件 × S/M/L(手工录入,"
        f"来源见 `density_processor.py`)。散点+marker 按前驱体双编码着色/形状。"
        f"三条水平阴影带 = 各前驱体组 compaction_density 均值±1std,实线=组均值,"
        f"用以呼应论文 §3.1 的描述——**组内 compaction_density~Θ 相关不显著"
        f"(图例标注各组 Spearman ρ 与 p,p 值范围约 0.20–0.87),组间清晰分层**;"
        f"故不再画组内 OLS 斜率虚线,以免暗示不存在的组内趋势。\n\n"
        f"**Θ 标定状态**:{THETA_NOTE}"
    )
    save_fig(fig, NAME, caption)
    plt.close(fig)


def plot_by_T(sub: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    rng = np.random.default_rng(0)
    jitter_map = {"S": -6, "M": 0, "L": 6}
    for prec in ["S", "M", "L"]:
        g = sub[sub["precursor"] == prec]
        if g.empty:
            continue
        color = PRECURSOR_COLOR[prec]
        x = g["T_C"].to_numpy(float) + jitter_map[prec] + rng.uniform(-1.2, 1.2, len(g))
        ax.scatter(x, g["compaction_density"], color=color, marker=PRECURSOR_MARKER[prec],
                   s=55, edgecolor="black", linewidth=0.4, label=PRECURSOR_LABEL[prec])
    ax.set_xticks([850, 900, 950])
    ax.set_xlabel("T_C (°C)  (点做水平抖动区分前驱体)")
    ax.set_ylabel("compaction_density (g/cm³)")
    ax.set_title("压实密度 vs T_C(不依赖 Θ 标定的稳健版)")
    ax.legend(loc="best", fontsize=9)

    n, n_cond = len(sub), sub["condition_id"].nunique()
    caption = f"**数据**:同 `{NAME}`,n={n} 样 = {n_cond} 条件。x 轴改为 T_C,不依赖 Θ 标定。"
    save_fig(fig, NAME_BYT, caption)
    plt.close(fig)


def main():
    df = pd.read_csv(MASTER_TABLE)
    sub = df[df["compaction_density"].notna()].copy()
    if sub.empty:
        skip(NAME, "master_table.csv 中 compaction_density 全部缺失")
        return
    plot_vs_theta(sub)
    plot_by_T(sub)


if __name__ == "__main__":
    main()
