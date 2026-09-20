# -*- coding: utf-8 -*-
"""压实密度 3(T)×3(β)×3(t) 汇总矩阵,前驱体分面,缺条件留白标注。
同时导出 table_density.csv(条件 × 前驱体密度矩阵,供论文表5-5取用)。
"""
import _bootstrap  # noqa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _style import MASTER_TABLE, REPO_ROOT, save_fig, skip

NAME = "fig_density_matrix"
TABLE_PATH = REPO_ROOT / "outputs" / "tables" / "table_density.csv"

T_LEVELS = [850, 900, 950]
BETA_LEVELS = [2, 5, 8]
THOLD_LEVELS = [10, 15, 20]
PRECURSORS = ["S", "M", "L"]


def _write_table(df: pd.DataFrame) -> None:
    conds = df[["condition_id", "T_C", "beta", "t_hold"]].drop_duplicates() \
        .set_index("condition_id").sort_index()
    piv = df.pivot_table(index="condition_id", columns="precursor",
                          values="compaction_density", aggfunc="first")
    out = conds.join(piv[[c for c in PRECURSORS if c in piv.columns]])
    out = out.reindex(columns=["T_C", "beta", "t_hold"] + PRECURSORS)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLE_PATH, encoding="utf-8-sig")
    print(f"[table] table_density.csv 写入 {TABLE_PATH}(n_conditions={len(out)})")


def main():
    df = pd.read_csv(MASTER_TABLE)
    sub = df[df["compaction_density"].notna()].copy()
    if sub.empty:
        skip(NAME, "master_table.csv 中 compaction_density 全部缺失")
        return

    _write_table(sub)

    vmin, vmax = sub["compaction_density"].min(), sub["compaction_density"].max()
    fig, axes = plt.subplots(3, 3, figsize=(10, 10.5))
    im = None
    n_missing_cells = 0
    for i, t_c in enumerate(T_LEVELS):
        for j, prec in enumerate(PRECURSORS):
            ax = axes[i, j]
            grid = np.full((3, 3), np.nan)
            for bi, beta in enumerate(BETA_LEVELS):
                for ti, t_hold in enumerate(THOLD_LEVELS):
                    row = sub[(sub.T_C == t_c) & (sub.beta == beta) &
                              (sub.t_hold == t_hold) & (sub.precursor == prec)]
                    if not row.empty:
                        grid[ti, bi] = row["compaction_density"].iloc[0]
                    else:
                        n_missing_cells += 1
            im = ax.imshow(grid, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
            for bi in range(3):
                for ti in range(3):
                    v = grid[ti, bi]
                    if np.isnan(v):
                        ax.text(bi, ti, "N/A", ha="center", va="center", fontsize=8, color="grey")
                    else:
                        ax.text(bi, ti, f"{v:.3f}", ha="center", va="center", fontsize=8,
                                color="white" if v < (vmin + vmax) / 2 else "black")
            ax.set_xticks(range(3)); ax.set_xticklabels(BETA_LEVELS)
            ax.set_yticks(range(3)); ax.set_yticklabels(THOLD_LEVELS)
            if i == 2:
                ax.set_xlabel("β (°C/min)")
            if j == 0:
                ax.set_ylabel(f"{t_c:.0f}°C\nt_hold (h)")
            if i == 0:
                ax.set_title(f"precursor={prec}")

    fig.subplots_adjust(right=0.88, hspace=0.35, wspace=0.3)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="compaction_density (g/cm³)")
    fig.suptitle("压实密度汇总矩阵: T × β × t_hold × 前驱体", y=0.98)

    n, n_cond = len(sub), sub["condition_id"].nunique()
    caption = (
        f"**数据**:`compaction_density`,n={n} 样 = {n_cond}/27 条件 × S/M/L(每格一个"
        f"值)。3 行=T_C(850/900/950),3 列=前驱体(S/M/L);每个子图内 3×3 为 "
        f"β(x 轴 2/5/8)× t_hold(y 轴 10/15/20)。灰色 `N/A` 标注缺条件"
        f"(全表 27 条件中有 {27 - n_cond} 条件尚无压实密度:"
        f"{', '.join(sorted(set(df.condition_id) - set(sub.condition_id)))})。\n\n"
        f"配套表:`outputs/tables/table_density.csv`(条件 × 前驱体密度矩阵)。"
    )
    save_fig(fig, NAME, caption)
    plt.close(fig)


if __name__ == "__main__":
    main()
