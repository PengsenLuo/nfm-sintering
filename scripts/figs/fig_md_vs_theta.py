# -*- coding: utf-8 -*-
"""M_D(同炉粒径记忆系数)vs Θ,20 条件散点 + S 形衰减拟合(+ _byT 版)。
M_D 定义:同炉 L/S 产物 D50 差 / 前驱体 D50 差,基于 D50,不受过筛伪影影响。
"""
import _bootstrap  # noqa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from _style import (MASTER_TABLE, TEMP_COLOR, THETA_NOTE, save_fig, skip)
from nfm.features.memory_indices import fit_memory_decay

NAME = "fig_md_vs_theta"
NAME_BYT = "fig_md_vs_theta_byT"


def _sigmoid_decay(lnTheta, a, lnTheta_c):
    """与 nfm.features.memory_indices 内部同名私有函数公式一致,用于取
    curve_fit 的协方差矩阵算 Θ_c 的 95% CI(该模块本身只返回点估计)。"""
    return 1.0 / (1.0 + np.exp(a * (lnTheta - lnTheta_c)))


def _load_per_condition():
    df = pd.read_csv(MASTER_TABLE)
    sub = df[df["M_D"].notna()].copy()
    per_cond = sub.groupby("condition_id").agg(
        Theta=("Theta", "mean"), T_C=("T_C", "mean"), M_D=("M_D", "first")).reset_index()
    return per_cond


def _fit_is_sane(fit, theta) -> bool:
    """fit_memory_decay 用 curve_fit 只要不抛异常就标 note='ok',但会静默收敛到
    退化解(a≈0 即完全不衰减、Θ_c 落在数据范围外)。这里补做合理性检查,
    退化解按"拟合失败"处理,走 LOWESS 兜底,而不是画一条没有物理意义的曲线。"""
    if fit["note"] != "ok":
        return False
    if not (np.isfinite(fit["a"]) and np.isfinite(fit["Theta_c"])):
        return False
    if abs(fit["a"]) < 1e-3:
        return False
    if not (theta.min() * 0.2 <= fit["Theta_c"] <= theta.max() * 5):
        return False
    if fit["r2"] < 0.05:
        return False
    return True


def _fit_with_ci(theta, memory):
    fit = fit_memory_decay(theta, memory)
    if not _fit_is_sane(fit, theta):
        fit = dict(fit, note=f"退化解或拟合质量不足(a={fit['a']:.2g}, "
                              f"Θ_c={fit['Theta_c']:.2g}, R²={fit['r2']:.2g}),判定为拟合失败")
        return fit, None
    x = np.log(theta)
    try:
        popt, pcov = curve_fit(_sigmoid_decay, x, memory, p0=[fit["a"], np.log(fit["Theta_c"])],
                                maxfev=20000)
        se_lnTc = float(np.sqrt(pcov[1, 1]))
        ci = (float(np.exp(popt[1] - 1.96 * se_lnTc)), float(np.exp(popt[1] + 1.96 * se_lnTc)))
    except Exception:
        ci = None
    return fit, ci


def plot_vs_theta(per_cond: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for t_c, color in TEMP_COLOR.items():
        g = per_cond[per_cond["T_C"] == t_c]
        ax.scatter(g["Theta"], g["M_D"], color=color, s=60, edgecolor="black",
                   linewidth=0.5, label=f"{t_c:.0f}°C", zorder=3)

    theta = per_cond["Theta"].to_numpy()
    memory = per_cond["M_D"].to_numpy()
    fit, ci = _fit_with_ci(theta, memory)

    if fit["note"] == "ok":
        theta_line = np.linspace(theta.min(), theta.max(), 200)
        ax.plot(theta_line, fit["predict"](theta_line), color="black", linewidth=2,
                zorder=2, label="S 形衰减拟合")
        ax.axvline(fit["Theta_c"], color="grey", linestyle=":", linewidth=1.2)
        ci_txt = f"[{ci[0]:.1f}, {ci[1]:.1f}]" if ci else "不可用"
        fit_txt = (f"拟合:a={fit['a']:.2f}, Θ_c={fit['Theta_c']:.1f} h "
                   f"(95% CI {ci_txt}), R²={fit['r2']:.3f}")
    else:
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            sm = lowess(memory, theta, frac=0.6, return_sorted=True)
            ax.plot(sm[:, 0], sm[:, 1], color="black", linewidth=2, label="LOWESS(未做参数拟合)")
        except Exception:
            pass
        fit_txt = f"S 形拟合未成功({fit['note']}),已退化为散点 + LOWESS,未标 Θ_c。"

    ax.set_xlabel("Θ (归一化热暴露量, h)")
    ax.set_ylabel("M_D (同炉 L/S 产物 D50 差 ÷ 前驱体 D50 差)")
    ax.set_title("形貌记忆衰减: M_D vs Θ")
    ax.legend(loc="best", fontsize=9)

    n = len(per_cond)
    caption = (
        f"**数据**:`M_D`,n={n} 条件(每条件一个值,来自同炉 L/S 产物 D50 差 ÷ "
        f"前驱体 D50 差)。M_D 基于 D50(激光粒度),不依赖 SEM 过筛/分割,不受本轮"
        f"暂停的 SEM 形貌列污染。\n\n"
        f"**拟合**:{fit_txt}\n\n"
        f"**Θ 标定状态**:{THETA_NOTE}"
    )
    save_fig(fig, NAME, caption)
    plt.close(fig)


def plot_by_T(per_cond: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    rng = np.random.default_rng(0)
    for t_c, color in TEMP_COLOR.items():
        g = per_cond[per_cond["T_C"] == t_c]
        if g.empty:
            continue
        x = np.full(len(g), t_c) + rng.uniform(-3, 3, len(g))
        ax.scatter(x, g["M_D"], color=color, s=60, edgecolor="black", linewidth=0.5,
                   label=f"{t_c:.0f}°C")
    ax.set_xticks([850, 900, 950])
    ax.set_xlabel("T_C (°C)  (点做水平抖动,不代表 x 轴数值差异)")
    ax.set_ylabel("M_D")
    ax.set_title("M_D vs T_C(不依赖 Θ 标定的稳健版)")
    ax.legend(loc="best", fontsize=9)

    n = len(per_cond)
    caption = f"**数据**:同 `{NAME}`,n={n} 条件。x 轴改为 T_C,不依赖 Θ 标定。"
    save_fig(fig, NAME_BYT, caption)
    plt.close(fig)


def main():
    per_cond = _load_per_condition()
    if per_cond.empty:
        skip(NAME, "master_table.csv 中 M_D 全部缺失")
        return
    plot_vs_theta(per_cond)
    plot_by_T(per_cond)


if __name__ == "__main__":
    main()
