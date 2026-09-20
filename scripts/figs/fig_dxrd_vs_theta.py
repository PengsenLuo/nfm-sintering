# -*- coding: utf-8 -*-
"""D_XRD vs Θ(+ 按 T_C 分组的 _byT 稳健版)。51 样 = 17 条件 × S/M/L
(纳米层口径定档,仅仪器1/SmartLab,见 nfm.nano_layer.nano_layer_frame)。"""
import _bootstrap  # noqa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _style import (MASTER_TABLE, TEMP_COLOR, PRECURSOR_MARKER,
                     PRECURSOR_LABEL, THETA_NOTE, save_fig, skip)
from nfm.nano_layer import nano_layer_frame

NAME = "fig_dxrd_vs_theta"
NAME_BYT = "fig_dxrd_vs_theta_byT"


def _load():
    df = pd.read_csv(MASTER_TABLE)
    # 纳米层口径定档(2026-07-30):D_XRD 只用仪器1(SmartLab)51样,见 nfm.nano_layer。
    df = nano_layer_frame(df, require_reliable=True)
    sub = df[df["D_XRD"].notna()].copy()
    return sub


def plot_vs_theta(sub: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for t_c, color in TEMP_COLOR.items():
        g_t = sub[sub["T_C"] == t_c]
        for prec, marker in PRECURSOR_MARKER.items():
            g = g_t[g_t["precursor"] == prec]
            if g.empty:
                continue
            ax.scatter(g["Theta"], g["D_XRD"], color=color, marker=marker,
                       s=55, edgecolor="black", linewidth=0.4,
                       label=f"{t_c:.0f}°C / {prec}")
    ax.set_xlabel("Θ (归一化热暴露量, h)")
    ax.set_ylabel("D_XRD (nm, Scherrer (003)/(104))")
    ax.set_title("D_XRD vs Θ")
    n = len(sub)
    n_cond = sub["condition_id"].nunique()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, ncol=3, fontsize=7.5, loc="upper left")
    caption = (
        f"**数据**:`data/processed/master_table.csv` 的 `D_XRD` 列,n={n} 样 = "
        f"{n_cond} 条件 × S/M/L(条件:"
        f"{', '.join(sorted(sub['condition_id'].unique()))})。"
        f" XRD 表征覆盖27个工艺条件中的17个完整条件三元组,共51个样本"
        f"(S/M/L各17),Θ覆盖4.05–47.34h全域,并非仅落在低热暴露段;"
        f"温度/升温速率/保温时间三个因子在这17个完整条件上不均衡"
        f"(β计数15/12/24,t计数18/9/24),其余10个条件(C02/C05/C06/C08/"
        f"C10/C11/C13/C20/C22/C23)因实测换用了另一台仪器(xrd_instrument=2)、"
        f"按纳米层单仪器口径限定(见 nfm.nano_layer)未纳入本图,不是处理"
        f"未完成。\n\n"
        f"**颜色**=T_C(850/900/950,蓝/橙/红),**marker**=前驱体(圆 S / 方 M / 三角 L)。\n\n"
        f"**Θ 标定状态**:{THETA_NOTE}"
    )
    save_fig(fig, NAME, caption)
    plt.close(fig)


def plot_by_T(sub: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    rng = np.random.default_rng(0)
    prec_order = ["S", "M", "L"]
    jitter_map = {"S": -6, "M": 0, "L": 6}
    for prec in prec_order:
        g = sub[sub["precursor"] == prec]
        if g.empty:
            continue
        x = g["T_C"].to_numpy(float) + jitter_map[prec] + rng.uniform(-1.2, 1.2, len(g))
        color = [TEMP_COLOR[t] for t in g["T_C"]]
        ax.scatter(x, g["D_XRD"], c=color, marker=PRECURSOR_MARKER[prec],
                   s=55, edgecolor="black", linewidth=0.4,
                   label=PRECURSOR_LABEL[prec])
    ax.set_xticks([850, 900, 950])
    ax.set_xlabel("T_C (°C)  (点做水平抖动区分前驱体,不代表 x 轴数值差异)")
    ax.set_ylabel("D_XRD (nm)")
    ax.set_title("D_XRD vs T_C(不依赖 Θ 标定的稳健版)")
    ax.legend(loc="upper left", fontsize=9)
    n = len(sub)
    n_cond = sub["condition_id"].nunique()
    caption = (
        f"**数据**:同 `{NAME}`,n={n} 样 = {n_cond} 条件。本版以 T_C(850/900/950,"
        f"仅三档)为横轴,marker 水平抖动仅为区分前驱体、无标定依赖,用于在 Θ "
        f"起算温度未锁定期间提供稳健参照。"
    )
    save_fig(fig, NAME_BYT, caption)
    plt.close(fig)


def main():
    sub = _load()
    if sub.empty:
        skip(NAME, "master_table.csv 中 D_XRD 全部缺失")
        return
    plot_vs_theta(sub)
    plot_by_T(sub)


if __name__ == "__main__":
    main()
