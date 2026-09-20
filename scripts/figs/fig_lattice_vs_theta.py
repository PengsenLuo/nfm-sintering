# -*- coding: utf-8 -*-
"""lattice_a / lattice_c / c_a_ratio 三联子图 vs Θ(+ 按 T_C 分组的 _byT 版)。
不含 D_XRD_WH / microstrain_WH(已定为定性参照,不进正式图)。
"""
import _bootstrap  # noqa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _style import (MASTER_TABLE, TEMP_COLOR, PRECURSOR_MARKER,
                     PRECURSOR_LABEL, THETA_NOTE, save_fig, skip)
from nfm.nano_layer import nano_layer_frame

NAME = "fig_lattice_vs_theta"
NAME_BYT = "fig_lattice_vs_theta_byT"
PANELS = [("lattice_a", "lattice_a (Å)"), ("lattice_c", "lattice_c (Å)"),
          ("c_a_ratio", "c/a")]


def plot_vs_theta(sub: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    fig.subplots_adjust(wspace=0.45)
    for ax, (col, ylabel) in zip(axes, PANELS):
        for t_c, color in TEMP_COLOR.items():
            g_t = sub[sub["T_C"] == t_c]
            for prec, marker in PRECURSOR_MARKER.items():
                g = g_t[g_t["precursor"] == prec]
                if g.empty:
                    continue
                ax.scatter(g["Theta"], g[col], color=color, marker=marker,
                           s=45, edgecolor="black", linewidth=0.4)
        ax.set_xlabel("Θ (h)")
        ax.set_ylabel(ylabel)
    axes[0].set_title("lattice_a")
    axes[1].set_title("lattice_c")
    axes[2].set_title("c/a ratio")
    handles = [plt.Line2D([0], [0], marker=m, color="w", markerfacecolor="grey",
                           markeredgecolor="black", markersize=8, label=p)
               for p, m in PRECURSOR_MARKER.items()]
    handles += [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                            markeredgecolor="black", markersize=8, label=f"{t:.0f}°C")
                for t, c in TEMP_COLOR.items()]
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("晶格参数 vs Θ (R-3m 最小二乘)", y=1.18)

    n, n_cond = len(sub), sub["condition_id"].nunique()
    caption = (
        f"**数据**:`lattice_a`/`lattice_c`/`c_a_ratio`(R-3m 多峰最小二乘),"
        f"n={n} 样 = {n_cond} 条件,与 `fig_dxrd_vs_theta` 同一子集(纳米层口径"
        f"定档,仅仪器1/SmartLab 51样=17完整条件三元组,其余10个条件因换用"
        f"另一台仪器按口径限定未纳入,不是处理未完成,见 nfm.nano_layer)。\n\n"
        f"**不含** `D_XRD_WH`/`microstrain_WH`——Williamson-Hall 已定论为定性参照,"
        f"不作定量结果引用(见 src/nfm/data_processing/xrd_processor.py)。\n\n"
        f"颜色=T_C,marker=前驱体。\n\n**Θ 标定状态**:{THETA_NOTE}"
    )
    save_fig(fig, NAME, caption)
    plt.close(fig)


def plot_by_T(sub: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.subplots_adjust(wspace=0.45)
    rng = np.random.default_rng(0)
    jitter_map = {"S": -6, "M": 0, "L": 6}
    for ax, (col, ylabel) in zip(axes, PANELS):
        for prec in ["S", "M", "L"]:
            g = sub[sub["precursor"] == prec]
            if g.empty:
                continue
            x = g["T_C"].to_numpy(float) + jitter_map[prec] + rng.uniform(-1.2, 1.2, len(g))
            color = [TEMP_COLOR[t] for t in g["T_C"]]
            ax.scatter(x, g[col], c=color, marker=PRECURSOR_MARKER[prec],
                       s=45, edgecolor="black", linewidth=0.4)
        ax.set_xticks([850, 900, 950])
        ax.set_xlabel("T_C (°C)")
        ax.set_ylabel(ylabel)
    axes[0].set_title("lattice_a")
    axes[1].set_title("lattice_c")
    axes[2].set_title("c/a ratio")
    handles = [plt.Line2D([0], [0], marker=m, color="w", markerfacecolor="grey",
                           markeredgecolor="black", markersize=8, label=p)
               for p, m in PRECURSOR_MARKER.items()]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("晶格参数 vs T_C(不依赖 Θ 标定的稳健版)", y=1.18)

    n, n_cond = len(sub), sub["condition_id"].nunique()
    caption = (
        f"**数据**:同 `{NAME}`,n={n} 样 = {n_cond} 条件。x 轴改为 T_C(850/900/950),"
        f"marker 水平抖动仅区分前驱体,不依赖 Θ 标定。"
    )
    save_fig(fig, NAME_BYT, caption)
    plt.close(fig)


def main():
    df = pd.read_csv(MASTER_TABLE)
    # 纳米层口径定档(2026-07-30):lattice_* 只用仪器1(SmartLab)51样,见 nfm.nano_layer。
    df = nano_layer_frame(df, require_reliable=True)
    sub = df[df["lattice_a"].notna() & df["lattice_c"].notna()].copy()
    if sub.empty:
        skip(NAME, "master_table.csv 中 lattice_a/lattice_c 全部缺失")
        return
    plot_vs_theta(sub)
    plot_by_T(sub)


if __name__ == "__main__":
    main()
