# -*- coding: utf-8 -*-
"""fig6_sem_psd_gradation.py -- Fig. 6 (journal submission set): main-text
figure for the SEM-caliber product PSD gradation-transfer evidence
(originally produced as figS12_sem_psd_gradation.py, then promoted to the
main text and renamed here: the SEM-caliber gradation-transfer result
belongs in main-text §3.5 as its primary evidence (p<1e-4), not
supplementary material. scripts/figs/journal/fig8_condition_residuals.py
(the former Fig. 6) is a different figure -- see that script's own
docstring. Rename only: no data/colour/marker/scale change from the
original figS12 version. This figure supersedes the manuscript's earlier
placeholder Figure 6 slot, which relied on a 9-sample water-dispersed
laser-diffraction measurement now excluded from the manuscript.

Panel (a): per-precursor Span_num distribution (81 samples, colour-coded by
sintering temperature, following this figure set's TEMP_COLOR convention).
Panel (b): 27-condition M-L paired differences for D10_num and Span_num
(the two metrics found significant), shown as two columns of points at
x=0/1 with the zero line marked. Full statistics (Wilcoxon signed-rank
test, per-metric p-values, precursor-group medians) are computed by
scripts/45_sem_psd_gradation_pairwise.py -- not recomputed here beyond
what each panel plots.
"""
import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _journal_style import (
    double_column_figsize,
    save_fig_journal,
    dump_source_data,
    add_panel_label,
    TEMP_COLOR,
    REPO_ROOT,
)

NAME = "fig6_sem_psd_gradation"
SRC = REPO_ROOT / "data" / "interim" / "sem_psd_quantiles.csv"
MASTER_TABLE = REPO_ROOT / "data" / "processed" / "master_table.csv"
PRECURSOR_ORDER = ["S", "M", "L"]


def main():
    if not SRC.exists():
        print(f"[skip] {NAME}: {SRC} missing")
        return

    df = pd.read_csv(SRC)
    t_c_map = pd.read_csv(MASTER_TABLE)[["sample_id", "T_C"]]
    df_tc = df.merge(t_c_map, on="sample_id", how="left")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=double_column_figsize(85))

    # Panel (a): Span_num by precursor, temperature-coloured jitter scatter
    rng = np.random.default_rng(0)
    for i, prec in enumerate(PRECURSOR_ORDER):
        sub = df_tc[df_tc.precursor == prec]
        for t_c, color in TEMP_COLOR.items():
            g = sub[sub["T_C"] == t_c]
            if g.empty:
                continue
            jitter = rng.uniform(-0.12, 0.12, len(g))
            ax_a.scatter(np.full(len(g), i) + jitter, g["Span_num"], color=color,
                        s=16, edgecolor="black", linewidth=0.3, zorder=3,
                        label=f"{t_c:.0f} C" if i == 0 else None)
    box_data = [df[df.precursor == p]["Span_num"].to_numpy() for p in PRECURSOR_ORDER]
    ax_a.boxplot(box_data, positions=range(3), widths=0.5, showfliers=False,
                zorder=1, patch_artist=True,
                boxprops=dict(facecolor="none", edgecolor="#888888"))
    ax_a.set_xticks(range(3))
    ax_a.set_xticklabels(PRECURSOR_ORDER)
    ax_a.set_xlabel("Precursor architecture")
    ax_a.set_ylabel(r"$\mathrm{Span}_{num}$ (number-weighted)")
    ax_a.legend(loc="upper right", fontsize=6.2, title="Sintering T", title_fontsize=6.2)
    add_panel_label(ax_a, "(a)")

    dump_source_data(
        NAME, "a", df[["sample_id", "condition_id", "precursor", "Span_num"]],
        note="Number-weighted Span (D90-D10)/D50 per sample, 81 samples, "
             "grouped by precursor architecture; colour in the rendered "
             "figure encodes sintering temperature (joined from "
             "master_table.csv T_C, not part of this tidy table).",
        source_files=[SRC, MASTER_TABLE],
    )

    # Panel (b): 27-condition paired M-L differences for D10_num, Span_num
    pair_rows = []
    for j, col in enumerate(["D10_num", "Span_num"]):
        piv = df.pivot(index="condition_id", columns="precursor", values=col)
        diff = piv["M"] - piv["L"]
        jitter = np.random.default_rng(1).uniform(-0.12, 0.12, len(diff))
        ax_b.scatter(np.full(len(diff), j) + jitter, diff.values, color="#457b9d",
                    s=16, edgecolor="black", linewidth=0.3, zorder=3)
        for cid, v in diff.items():
            pair_rows.append(dict(condition_id=cid, metric=col, diff_M_minus_L=v))
    ax_b.axhline(0, color="#999999", linestyle="--", linewidth=0.8, zorder=1)
    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels([r"$D10_{num}$", r"$\mathrm{Span}_{num}$"])
    ax_b.set_ylabel("Paired difference (M - L), 27 conditions")
    add_panel_label(ax_b, "(b)")

    dump_source_data(
        NAME, "b", pd.DataFrame(pair_rows),
        note="27-condition same-furnace paired M-L differences for D10_num "
             "and Span_num (the two metrics found significant by a "
             "Wilcoxon signed-rank test across the 27 conditions).",
        source_files=[SRC],
    )

    fig.tight_layout()
    caption = (
        "Fig. 6. SEM-caliber (500x, number-weighted particle-size "
        "distribution) evidence for gradation retention. (a) Number-weighted "
        "Span, all 81 samples, grouped by precursor architecture (S/M/L), "
        "point colour encodes sintering temperature. (b) 27-condition "
        "same-furnace paired differences (M - L) for D10 and Span "
        "(number-weighted); a Wilcoxon signed-rank test finds both "
        "significantly different from zero, while D50 and volume-weighted "
        "Span are not distinguishable between M and L."
    )
    save_fig_journal(fig, NAME, caption)
    plt.close(fig)
    return fig


if __name__ == "__main__":
    main()
