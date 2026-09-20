# -*- coding: utf-8 -*-
"""figS4_density_matrix.py -- Fig. S4 (journal submission set, supplementary
information): compaction_density summary matrix, 3(T_C) x 3(precursor) grid
of 3(beta) x 3(t_hold) heatmap cells, restyled port of
scripts/figs/fig_density_matrix.py (read-only source; not imported, not
modified).

Same data source and logic as that script: master_table.csv rows with
non-null compaction_density; for each (T_C, precursor) combination, a 3x3
grid of (beta, t_hold) cells is filled with compaction_density where a
sample exists for that exact condition, and left as a labelled "N/A" cell
where the source script found no matching row (missing-condition handling
identical to the source's grid = np.full((3,3), np.nan) + per-cell lookup
loop). Every 3x3 grid uses the same colour scale (vmin/vmax = global min/max
across all non-null compaction_density), same as the source script.

No instrument-disclosure concern applies to this figure: compaction_density
comes from the manual density-measurement pipeline (density_processor.py),
not from XRD, so the missing conditions listed in the caption are genuine
not-yet-measured conditions, unrelated to the XRD dual-instrument split
that fig3/figS1/figS3 must avoid disclosing.
"""
import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _journal_style import (
    double_column_figsize,
    save_fig_journal,
    dump_source_data,
    REPO_ROOT,
)
from _style import MASTER_TABLE

NAME = "figS4_density_matrix"

T_LEVELS = [850, 900, 950]
BETA_LEVELS = [2, 5, 8]
THOLD_LEVELS = [10, 15, 20]
PRECURSORS = ["S", "M", "L"]


def _load() -> pd.DataFrame:
    df = pd.read_csv(MASTER_TABLE)
    sub = df[df["compaction_density"].notna()].copy()
    return df, sub


def main():
    df, sub = _load()
    if sub.empty:
        print(f"[skip] {NAME}: master_table.csv has no compaction_density")
        return

    vmin, vmax = sub["compaction_density"].min(), sub["compaction_density"].max()

    grid_rows = []
    for t_c in T_LEVELS:
        for prec in PRECURSORS:
            for beta in BETA_LEVELS:
                for t_hold in THOLD_LEVELS:
                    row = df[(df.T_C == t_c) & (df.beta == beta) &
                            (df.t_hold == t_hold) & (df.precursor == prec)]
                    grid_rows.append(dict(
                        T_C=t_c, precursor=prec, beta=beta, t_hold=t_hold,
                        condition_id=row["condition_id"].iloc[0],
                        sample_id=row["sample_id"].iloc[0],
                        compaction_density=row["compaction_density"].iloc[0],
                    ))
    grid_df = pd.DataFrame(grid_rows)

    fig = plt.figure(figsize=double_column_figsize(200))
    outer = fig.add_gridspec(3, 3, wspace=0.18, hspace=0.30,
                             left=0.075, right=0.87, top=0.92, bottom=0.075)
    im = None
    for i, t_c in enumerate(T_LEVELS):
        for j, prec in enumerate(PRECURSORS):
            ax = fig.add_subplot(outer[i, j])
            grid = np.full((3, 3), np.nan)
            for bi, beta in enumerate(BETA_LEVELS):
                for ti, t_hold in enumerate(THOLD_LEVELS):
                    row = sub[(sub.T_C == t_c) & (sub.beta == beta) &
                              (sub.t_hold == t_hold) & (sub.precursor == prec)]
                    if not row.empty:
                        grid[ti, bi] = row["compaction_density"].iloc[0]
            im = ax.imshow(grid, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
            for bi in range(3):
                for ti in range(3):
                    v = grid[ti, bi]
                    if np.isnan(v):
                        ax.text(bi, ti, "N/A", ha="center", va="center",
                               fontsize=5.2, color="grey")
                    else:
                        ax.text(bi, ti, f"{v:.3f}", ha="center", va="center",
                               fontsize=5.2,
                               color="white" if v < (vmin + vmax) / 2 else "black")
            ax.set_xticks(range(3))
            ax.set_yticks(range(3))
            if i == 2:
                ax.set_xticklabels(BETA_LEVELS, fontsize=6.0)
                ax.set_xlabel(r"$\beta$ (C/min)", fontsize=6.5)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_yticklabels(THOLD_LEVELS, fontsize=6.0)
                ax.set_ylabel(f"{t_c:.0f} C\nt_hold (h)", fontsize=6.5)
            else:
                ax.set_yticklabels([])
            if i == 0:
                ax.set_title(f"precursor = {prec}", fontsize=7.0)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

    cbar_ax = fig.add_axes([0.90, 0.12, 0.02, 0.76])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r"Compaction density (g/cm$^3$)", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6.0)

    n, n_cond = len(sub), sub["condition_id"].nunique()
    n_missing = 27 - n_cond
    missing_ids = ", ".join(sorted(set(df.condition_id) - set(sub.condition_id)))
    if n_missing > 0:
        missing_note = (
            f"Grey 'N/A' cells mark conditions with no compaction-density "
            f"measurement yet ({n_missing}/27 conditions missing across the "
            f"full design: {missing_ids}); all three precursor variants of "
            f"a missing condition are affected together since each "
            f"condition is one co-fired furnace run."
        )
    else:
        missing_note = (
            "All 27/27 conditions have a compaction-density measurement for "
            "all three precursor variants (no missing cells; the grey "
            "'N/A' convention used elsewhere in this figure set for missing "
            "conditions does not apply here)."
        )

    dump_source_data(
        NAME, "single", grid_df,
        note=f"9 (T_C x precursor) subplots x 9 (beta x t_hold) cells = 81 "
             f"rows; compaction_density is NaN for the {n_missing}/27 "
             f"condition(s) with no measurement yet ({missing_ids}), "
             f"rendered as a grey 'N/A' cell.",
        source_files=[MASTER_TABLE],
    )

    caption = (
        f"Fig. S4. Compaction density summary matrix: n={n} samples = "
        f"{n_cond}/27 conditions x S/M/L (one value per cell). 3 rows = "
        f"sintering temperature T_C (850/900/950 C), 3 columns = precursor "
        f"(S/M/L); within each subplot, a 3x3 grid of beta (x-axis: "
        f"2/5/8 C/min) x t_hold (y-axis: 10/15/20 h) reports the measured "
        f"compaction density (g/cm3). {missing_note} Colour scale "
        f"(viridis) is shared across all nine subplots, spanning the global "
        f"min-max of all measured compaction-density values "
        f"({vmin:.3f}-{vmax:.3f} g/cm3). A companion machine-readable table "
        f"(condition x precursor compaction-density matrix) is written to "
        f"outputs/tables/table_density.csv by the internal-review pipeline "
        f"this figure is restyled from."
    )
    save_fig_journal(fig, NAME, caption, panel_labels=False)
    plt.close(fig)
    return fig


if __name__ == "__main__":
    main()
