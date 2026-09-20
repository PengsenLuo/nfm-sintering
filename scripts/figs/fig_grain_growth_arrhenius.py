# -*- coding: utf-8 -*-
"""ln(D_XRD) vs 1/T_K,按前驱体分组线性拟合,报斜率换算的表观活化能
(仅作一致性参照,不进入正式建模;真实生长动力学还依赖 t_eff,这里是简化诊断)。
"""
import _bootstrap  # noqa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from _style import MASTER_TABLE, PRECURSOR_COLOR, PRECURSOR_MARKER, save_fig, skip
from nfm.nano_layer import nano_layer_frame

NAME = "fig_grain_growth_arrhenius"
R_GAS = 8.314  # J/(mol K)


def main():
    df = pd.read_csv(MASTER_TABLE)
    # 纳米层口径定档(2026-07-30):D_XRD 只用仪器1(SmartLab)51样,见 nfm.nano_layer。
    df = nano_layer_frame(df, require_reliable=True)
    sub = df[df["D_XRD"].notna()].copy()
    if sub.empty:
        skip(NAME, "master_table.csv 中 D_XRD 全部缺失")
        return
    sub["lnD"] = np.log(sub["D_XRD"])

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    fit_lines = []
    for prec in ["S", "M", "L"]:
        g = sub[sub["precursor"] == prec]
        if len(g) < 2:
            continue
        color = PRECURSOR_COLOR[prec]
        ax.scatter(g["inv_T_K"], g["lnD"], color=color, marker=PRECURSOR_MARKER[prec],
                   s=55, edgecolor="black", linewidth=0.4, label=f"{prec} (n={len(g)})")
        if len(g) >= 3 and g["inv_T_K"].nunique() >= 2:
            res = stats.linregress(g["inv_T_K"], g["lnD"])
            x_line = np.linspace(g["inv_T_K"].min(), g["inv_T_K"].max(), 50)
            ax.plot(x_line, res.intercept + res.slope * x_line, color=color, linestyle="--")
            q_app_kJ = -res.slope * R_GAS / 1000.0
            fit_lines.append((prec, res.slope, res.rvalue ** 2, q_app_kJ, len(g)))

    ax.set_xlabel("1/T_K (K$^{-1}$)")
    ax.set_ylabel("ln(D_XRD)  [D_XRD in nm]")
    ax.set_title("晶粒生长 Arrhenius 图: ln(D_XRD) vs 1/T_K")
    ax.legend(loc="best", fontsize=9)

    fit_text = "\n".join(
        f"- {p}: n={n}, 斜率={slope:.1f} K, R²={r2:.3f}, "
        f"表观 Q≈{q:.0f} kJ/mol{'(落在文献180–220范围)' if 180 <= q <= 220 else '(超出文献180–220范围,仅供参照)'}"
        for p, slope, r2, q, n in fit_lines
    ) if fit_lines else "(各前驱体分组点数不足,未拟合)"

    caption = (
        f"**数据**:`D_XRD` 与 `inv_T_K`(=1/T_K),n={len(sub)} 样 = "
        f"{sub['condition_id'].nunique()} 条件,按前驱体(S/M/L)分组做简单线性回归 "
        f"ln(D_XRD)=lnA-Q/(R·1/T_K)。\n\n"
        f"**注意**:此拟合未计入 t_eff(保温时间折算)差异,是简化的一致性诊断,"
        f"不是正式的生长动力学定量结果;真实活化能应在 PINN 训练中由 "
        f"L_Arrhenius 物理约束项与 t_eff 联合估计。\n\n"
        f"**各前驱体拟合**:\n{fit_text}\n\n"
        f"文献参考范围:活化能 180–220 kJ/mol。"
    )
    save_fig(fig, NAME, caption)
    plt.close(fig)


if __name__ == "__main__":
    main()
