# -*- coding: utf-8 -*-
"""fig1_design_and_precursor_psd.py —— Fig. 1(投稿级,NEW):3x3x3 全因子设计
(XRD 覆盖度) + 前驱体激光粒度曲线。

预先回答审稿人一眼就会问的两个问题:
  (a) 实验设计长什么样(T x beta x t_hold 全因子,27 条件 x S/M/L 前驱体)
  (b) 为什么 XRD 分析只覆盖 81 样本里的 51 个(=27 条件里的 17 个,仪器1)

数据来源:
  - data/processed/master_table.csv:condition_id -> (T_C, beta, t_hold) 映射;
    XRD 覆盖度通过 nfm.nano_layer.nano_layer_frame() 派生(纳米层数据唯一
    合法取数入口),不手写硬编码覆盖清单。
  - data/raw/psd/precursors/precursor_{S,M,L}.csv:3 个前驱体的真实 101 点
    体积-频率激光粒度曲线(Size (um), Volume (%))。
  - data/raw/psd/samples/psd_dvalues_precursors.csv:每个前驱体的 D10/D50/
    D90/Span,用于给曲线标注 D50。

data/raw/psd/precursors/ 下已有三个前驱体的真实完整体积-频率曲线,
面板(b)直接用真实曲线,不需要退回简化版 D10/D50/D90 示意图。
"""
import _bootstrap  # noqa: F401  (把 src/ 和 scripts/figs/ 加进 sys.path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _journal_style import (
    double_column_figsize,
    add_panel_label,
    save_fig_journal,
    dump_source_data,
    PRECURSOR_COLOR,
    PRECURSOR_MARKER,
    REPO_ROOT,
)
from nfm.nano_layer import nano_layer_frame

NAME = "fig1_design_and_precursor_psd"

MASTER_TABLE = REPO_ROOT / "data" / "processed" / "master_table.csv"
PSD_PRECURSOR_DIR = REPO_ROOT / "data" / "raw" / "psd" / "precursors"
PSD_DVALUES_PATH = REPO_ROOT / "data" / "raw" / "psd" / "samples" / "psd_dvalues_precursors.csv"

T_LEVELS = [850, 900, 950]
BETA_LEVELS = [2, 5, 8]
THOLD_LEVELS = [10, 15, 20]
PRECURSORS = ["S", "M", "L"]

FACE_COVERED = "#c9c9c9"
FACE_NOT_COVERED = "white"
EDGE_COLOR = "black"

# 仅作为运行期自检:与预先核对过的覆盖清单交叉比对,发现不一致就
# 报错而不是默默用脚本重新算出来的结果覆盖掉——交叉核对能同时抓出
# 清单手误和本脚本逻辑错误。
_BRIEF_COVERED = {
    "C01", "C03", "C04", "C07", "C09", "C12", "C14", "C15", "C16", "C17",
    "C18", "C19", "C21", "C24", "C25", "C26", "C27",
}


def _load_condition_coverage():
    """从 master_table.csv 派生 27 条件 -> (T_C, beta, t_hold) 映射,以及
    17 个被 XRD(仪器1, D_XRD_reliable)覆盖的条件集合。不手写覆盖清单。
    """
    master = pd.read_csv(MASTER_TABLE)
    cond_map = (
        master[["condition_id", "T_C", "beta", "t_hold"]]
        .drop_duplicates()
        .set_index("condition_id")
        .sort_index()
    )
    assert len(cond_map) == 27, f"期望 27 条件,实际 {len(cond_map)}"

    nano = nano_layer_frame(master, require_reliable=True)
    covered = set(nano["condition_id"].unique())
    assert len(covered) == 17, f"期望 17 个 XRD 覆盖条件,实际 {len(covered)}"
    assert covered == _BRIEF_COVERED, (
        f"派生覆盖集合与预先核对的清单不一致: "
        f"仅本脚本有={covered - _BRIEF_COVERED}, 仅清单有={_BRIEF_COVERED - covered}"
    )
    return cond_map, covered


def _draw_design_panel(fig, gs_a, cond_map, covered):
    """面板(a):3 个并排的 T_C 子网格(850/900/950),每个是 beta x t_hold
    的 3x3 网格,每格标唯一 condition_id,填充/网纹区分 XRD 是否覆盖
    (颜色留给全图统一的 S/M/L 编码,本面板不用颜色区分覆盖度)。
    """
    axes_a = [fig.add_subplot(gs_a[0, i]) for i in range(3)]
    for j, t_c in enumerate(T_LEVELS):
        ax = axes_a[j]
        for bi, beta in enumerate(BETA_LEVELS):
            for ti, t_hold in enumerate(THOLD_LEVELS):
                match = cond_map[
                    (cond_map["T_C"] == t_c)
                    & (cond_map["beta"] == beta)
                    & (cond_map["t_hold"] == t_hold)
                ]
                assert len(match) == 1, (
                    f"T_C={t_c},beta={beta},t_hold={t_hold} 应恰好 1 条,实际 {len(match)}"
                )
                cond_id = match.index[0]
                is_covered = cond_id in covered
                rect = Rectangle(
                    (bi - 0.5, ti - 0.5), 1.0, 1.0,
                    facecolor=FACE_COVERED if is_covered else FACE_NOT_COVERED,
                    edgecolor=EDGE_COLOR,
                    linewidth=0.6,
                    # 单斜线密度(而非 "////")让格内 condition_id 文字更易读,
                    # 同时仍与实心灰底形成清楚的 fill vs hatch 区分。
                    hatch=None if is_covered else "//",
                )
                ax.add_patch(rect)
                ax.text(
                    bi, ti, cond_id, ha="center", va="center",
                    fontsize=6.3, color="black",
                )
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, 2.5)
        ax.set_aspect("equal")
        ax.set_xticks(range(3))
        ax.set_xticklabels(BETA_LEVELS, fontsize=6.5)
        ax.set_yticks(range(3))
        if j == 0:
            ax.set_yticklabels(THOLD_LEVELS, fontsize=6.5)
            ax.set_ylabel("$t_{hold}$ (h)", fontsize=7)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel(r"$\beta$ (°C/min)", fontsize=7)
        ax.set_title(f"$T$ = {t_c:.0f} °C", fontsize=7.5)
        ax.grid(False)
        ax.tick_params(length=2, pad=1.5)

    # 覆盖度图例,放在三个子网格下方、只属于面板(a)的独立小轴里
    ax_legend = fig.add_subplot(gs_a[1, :])
    ax_legend.axis("off")
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=FACE_COVERED, edgecolor=EDGE_COLOR,
                  linewidth=0.6, label="XRD-covered (17 conditions)"),
        Rectangle((0, 0), 1, 1, facecolor=FACE_NOT_COVERED, edgecolor=EDGE_COLOR,
                  linewidth=0.6, hatch="//", label="Not covered (10 conditions)"),
    ]
    ax_legend.legend(
        handles=handles, loc="center", ncol=1, fontsize=6.5, frameon=False,
        handlelength=1.4, handleheight=1.1, borderaxespad=0,
    )
    return axes_a[0]


def _draw_psd_panel(ax_b, dvalues):
    """面板(b):3 个前驱体的真实激光粒度体积-频率曲线(101 点),对数 x
    轴,标注每条曲线的 D50。
    """
    curves = {}
    x_lo, x_hi = np.inf, -np.inf
    for p in PRECURSORS:
        curve = pd.read_csv(PSD_PRECURSOR_DIR / f"precursor_{p}.csv")
        curve = curve.dropna()
        x = curve["Size (μm)"].to_numpy(dtype=float)
        y = curve["Volume (%)"].to_numpy(dtype=float)
        curves[p] = (x, y)
        ax_b.plot(
            x, y, color=PRECURSOR_COLOR[p], marker=PRECURSOR_MARKER[p],
            markevery=6, markersize=3.2, linewidth=1.1,
            label=f"{p} (D50={dvalues.loc[f'precursor_{p}', 'D50']:.1f} µm)",
        )
        sig = x[y > 0.05]
        if len(sig):
            x_lo = min(x_lo, sig.min())
            x_hi = max(x_hi, sig.max())

        # D50 value is already stated in the legend label above; the dashed
        # vertical line alone marks its position on the curve without adding
        # a second floating text label that would collide with the legend
        # box (S's peak sits directly under "upper left", where the legend
        # is anchored).
        d50 = dvalues.loc[f"precursor_{p}", "D50"]
        ax_b.axvline(d50, color=PRECURSOR_COLOR[p], linestyle="--",
                     linewidth=0.6, alpha=0.6, ymax=0.92)

    ax_b.set_xscale("log")
    ax_b.set_xlim(x_lo * 0.55, x_hi * 1.6)
    ax_b.set_xlabel("Particle size (µm)")
    ax_b.set_ylabel("Volume frequency (%)")
    ax_b.legend(loc="upper left", fontsize=6.5, frameon=False)
    ax_b.grid(True, which="major", alpha=0.25)
    return curves


def main():
    cond_map, covered = _load_condition_coverage()
    dvalues = pd.read_csv(PSD_DVALUES_PATH).set_index("sample_id")

    fig = plt.figure(figsize=double_column_figsize(80))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.38,
                              left=0.06, right=0.98, top=0.87, bottom=0.15)
    gs_a = outer[0].subgridspec(2, 3, height_ratios=[1.0, 0.22], wspace=0.12, hspace=0.15)
    ax_b = fig.add_subplot(outer[1])

    ax_a0 = _draw_design_panel(fig, gs_a, cond_map, covered)
    curves = _draw_psd_panel(ax_b, dvalues)

    add_panel_label(ax_a0, "(a)")
    add_panel_label(ax_b, "(b)")

    design_rows = cond_map.reset_index()
    design_rows["xrd_covered"] = design_rows["condition_id"].isin(covered)
    dump_source_data(
        NAME, "a", design_rows,
        note="27-condition full-factorial design grid; xrd_covered marks "
             "the 17 conditions (51 samples) with nano-layer XRD data, via "
             "nfm.nano_layer.nano_layer_frame(require_reliable=True).",
        source_files=[MASTER_TABLE],
    )

    psd_rows = pd.concat([
        pd.DataFrame({
            "precursor": p,
            "size_um": x,
            "volume_pct": y,
            "D50_um": dvalues.loc[f"precursor_{p}", "D50"],
        })
        for p, (x, y) in curves.items()
    ], ignore_index=True)
    dump_source_data(
        NAME, "b", psd_rows,
        note="Real volume-frequency laser-diffraction curves per precursor "
             "(dropna'd), plus each curve's D50 (dashed vertical line).",
        source_files=[PSD_PRECURSOR_DIR / f"precursor_{p}.csv" for p in PRECURSORS]
                     + [PSD_DVALUES_PATH],
    )

    caption = (
        "Fig. 1. (a) 3x3x3 full-factorial sintering design (sintering "
        "temperature T, heating rate beta, holding time t_hold; 27 conditions "
        "C01-C27, each fired with three precursor particle-size variants "
        "S/M/L in the same furnace batch). Each cell shows its unique "
        "condition ID; filled cells mark the 17 conditions (51 samples) with "
        "XRD-derived nano-layer data (grain size and lattice parameters, via "
        "nfm.nano_layer.nano_layer_frame(); data/processed/master_table.csv), "
        "hatched cells mark the remaining 10 conditions. XRD thus covers 17 "
        "complete condition triplets (51 samples), spanning the full Theta "
        "range; the panel lets this coverage be read directly off the design "
        "grid. (b) Laser diffraction particle-size distributions of the three "
        "precursor powders (S, M, L) used to seed the 27 conditions, plotted "
        "as volume frequency vs. particle size (log scale); dashed vertical "
        "lines mark each precursor's D50. M is a 7:3 mass blend of large and "
        "small particles and shows a bimodal shape with two local maxima "
        "(data/raw/psd/precursors/precursor_{S,M,L}.csv; D-values from "
        "data/raw/psd/samples/psd_dvalues_precursors.csv)."
    )
    save_fig_journal(fig, NAME, caption)
    plt.close(fig)
    return fig


if __name__ == "__main__":
    main()
