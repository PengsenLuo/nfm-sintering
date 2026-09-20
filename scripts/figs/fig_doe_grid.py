# -*- coding: utf-8 -*-
"""实验设计 3×3×3(T/β/t)完成度网格 + table_conditions.csv(27 条件清单)。
零依赖:不需要任何单一 target 齐全,直接从 master_table 的非空情况自动统计。

fig_doe_grid 本身只按本轮"三条硬数据线"(D_XRD、M_D、compaction_density)打分,
与本轮 hard rule(禁止出 SEM/PSD 受污染列的图)保持一致;table_conditions.csv
是纯粹的覆盖度元数据(布尔标记,不展示数值),额外多记了几列(D50/D_sec/D_pri_sem/
B_bimodal)方便统筹全局进度,不受该 hard rule 约束。
"""
import _bootstrap  # noqa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from _style import MASTER_TABLE, REPO_ROOT, save_fig, skip
from nfm.nano_layer import nano_layer_frame

NAME = "fig_doe_grid"
TABLE_PATH = REPO_ROOT / "outputs" / "tables" / "table_conditions.csv"

T_LEVELS = [850, 900, 950]
BETA_LEVELS = [2, 5, 8]
THOLD_LEVELS = [10, 15, 20]

# 本轮记分用的三条硬数据线(与 fig_doe_grid 打分一致)
SCORE_COLS = ["D_XRD", "M_D", "compaction_density"]
# table_conditions.csv 额外记录的覆盖度列(不进打分、不进图,只是元数据)
EXTRA_COLS = ["D50", "D_sec", "D_pri_sem", "B_bimodal"]


def _write_table(df: pd.DataFrame) -> None:
    cond = df.groupby("condition_id").agg(
        T_C=("T_C", "first"), beta=("beta", "first"), t_hold=("t_hold", "first"),
        Theta=("Theta", "mean"),
        **{f"has_{c}": (c, lambda s: bool(s.notna().any())) for c in SCORE_COLS + EXTRA_COLS},
    ).sort_index()
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cond.to_csv(TABLE_PATH, encoding="utf-8-sig")
    print(f"[table] table_conditions.csv 写入 {TABLE_PATH}(n={len(cond)} 条件)")


def main():
    df = pd.read_csv(MASTER_TABLE)
    if df.empty:
        skip(NAME, "master_table.csv 为空")
        return

    # 纳米层口径定档(2026-07-30):D_XRD 覆盖度打分只认仪器1(SmartLab)51样,
    # 见 nfm.nano_layer——只清空非合法样本的 D_XRD,不动 M_D/compaction_density
    # (它们与 XRD 仪器无关,不受此限制)。
    nano_ok = set(nano_layer_frame(df, require_reliable=True)["sample_id"])
    df = df.copy()
    df.loc[~df["sample_id"].isin(nano_ok), "D_XRD"] = np.nan

    _write_table(df)

    cond = df.groupby("condition_id").agg(
        T_C=("T_C", "first"), beta=("beta", "first"), t_hold=("t_hold", "first"),
        **{f"has_{c}": (c, lambda s: bool(s.notna().any())) for c in SCORE_COLS},
    )
    cond["score"] = cond[[f"has_{c}" for c in SCORE_COLS]].sum(axis=1)

    cmap = ListedColormap(["#f2f2f2", "#ffd166", "#84a98c", "#2a9d8f"])
    norm = BoundaryNorm([0, 1, 2, 3, 4], cmap.N)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    im = None
    for j, t_c in enumerate(T_LEVELS):
        ax = axes[j]
        grid = np.zeros((3, 3))
        for bi, beta in enumerate(BETA_LEVELS):
            for ti, t_hold in enumerate(THOLD_LEVELS):
                row = cond[(cond.T_C == t_c) & (cond.beta == beta) & (cond.t_hold == t_hold)]
                if row.empty:
                    grid[ti, bi] = -1
                    label = "无此条件"
                else:
                    r = row.iloc[0]
                    grid[ti, bi] = r["score"]
                    marks = "".join(
                        ("X" if r["has_D_XRD"] else "-",
                         "M" if r["has_M_D"] else "-",
                         "ρ" if r["has_compaction_density"] else "-"))
                    label = f"{row.index[0]}\n{marks}"
                ax.text(bi, ti, label, ha="center", va="center", fontsize=8)
        plot_grid = np.where(grid < 0, np.nan, grid)
        im = ax.imshow(plot_grid, cmap=cmap, norm=norm, origin="lower")
        ax.set_xticks(range(3)); ax.set_xticklabels(BETA_LEVELS)
        ax.set_yticks(range(3)); ax.set_yticklabels(THOLD_LEVELS)
        ax.set_xlabel("β (°C/min)")
        if j == 0:
            ax.set_ylabel("t_hold (h)")
        ax.set_title(f"T_C={t_c:.0f}°C")

    fig.subplots_adjust(bottom=0.28, wspace=0.3)
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.04])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", ticks=[0.5, 1.5, 2.5, 3.5])
    cbar.ax.set_xticklabels(["0/3 完成", "1/3", "2/3", "3/3 全完成"])
    fig.suptitle("实验设计完成度: T×β×t_hold(本轮三条硬数据线打分)", y=1.03)

    n_full = int((cond["score"] == 3).sum())
    n_none = int((cond["score"] == 0).sum())
    caption = (
        f"**数据**:27 条件(C01–C27)× 本轮三条硬数据线(D_XRD、M_D、"
        f"compaction_density)是否有数据,取任一前驱体非空即算该条件命中。"
        f"格内 `X/M/ρ` 分别标 D_XRD/M_D/compaction_density 是否命中(字母=命中,"
        f"`-`=缺失),灰底=不存在该条件组合(3×3×3 设计外)。\n\n"
        f"全 27 条件中 {n_full} 个三线全齐、{n_none} 个三线全无。"
        f"本图**不含** SEM/PSD 受污染列(D_sec/D_pri_sem/D50/B_bimodal 等),"
        f"这些列的覆盖度只记录在 `outputs/tables/table_conditions.csv` 里"
        f"(纯布尔元数据,不展示数值,不受本轮 SEM 图禁令约束)。"
    )
    save_fig(fig, NAME, caption)
    plt.close(fig)


if __name__ == "__main__":
    main()
