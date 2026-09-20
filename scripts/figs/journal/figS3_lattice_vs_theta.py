# -*- coding: utf-8 -*-
"""figS3_lattice_vs_theta.py -- Fig. S3 (journal submission set, supplementary
information): lattice_a / lattice_c / c_a_ratio vs. Theta, 3-panel, restyled
port of scripts/figs/fig_lattice_vs_theta.py's plot_vs_theta() (read-only
source; not imported, not modified).

Same data source and scoping as that script: master_table.csv, restricted to
the nano-layer 51-sample/17-condition subset via
nfm.nano_layer.nano_layer_frame(require_reliable=True), rows with both
lattice_a and lattice_c non-null. Colour = T_C, marker = precursor, same
double encoding as fig3/figS1 in this directory.

IMPORTANT (house rule, 2026-08-02 authorial decision -- same rule fig2/fig3/
figS1 in this directory follow, and explicitly required again for this
figure per the task brief: fig_lattice_vs_theta.py's own internal caption in
the source script names the reason for the 51-sample split, which is exactly
the disclosure this house rule forbids in the journal-facing figure set).
Main-text and SI figures must never explain that the 51-vs-81 sample-count
split (for D_XRD/lattice_a/lattice_c/c_a_ratio) is due to a second XRD
measurement instrument, and must never list the excluded condition IDs. Use
exactly the sanctioned sentence: "XRD covers 17 complete condition triplets
(51 samples), spanning the full Theta range." Do not write "instrument",
"SmartLab", "MiniFlex", or "xrd_instrument" anywhere in this file's
on-figure text or caption.
"""
import _bootstrap  # noqa: F401

import pandas as pd
import matplotlib.pyplot as plt

from _journal_style import (
    double_column_figsize,
    add_panel_label,
    save_fig_journal,
    dump_source_data,
    PRECURSOR_MARKER,
    TEMP_COLOR,
)
from _style import MASTER_TABLE
from nfm.nano_layer import nano_layer_frame

NAME = "figS3_lattice_vs_theta"
PANELS = [("lattice_a", r"lattice_a ($\AA$)"), ("lattice_c", r"lattice_c ($\AA$)"),
          ("c_a_ratio", "c/a")]
PANEL_TITLE = {"lattice_a": "lattice_a", "lattice_c": "lattice_c", "c_a_ratio": "c/a ratio"}

PRECURSOR_LABEL_EN = {
    "S": "S (D50~3 um)",
    "M": "M (7:3 blend)",
    "L": "L (D50~11 um)",
}


def _load() -> pd.DataFrame:
    df = pd.read_csv(MASTER_TABLE)
    df = nano_layer_frame(df, require_reliable=True)
    sub = df[df["lattice_a"].notna() & df["lattice_c"].notna()].copy()
    return sub


def _draw_panel(ax, sub: pd.DataFrame, col: str, ylabel: str) -> None:
    for t_c, color in TEMP_COLOR.items():
        g_t = sub[sub["T_C"] == t_c]
        for prec, marker in PRECURSOR_MARKER.items():
            g = g_t[g_t["precursor"] == prec]
            if g.empty:
                continue
            ax.scatter(g["Theta"], g[col], color=color, marker=marker,
                       s=16, edgecolor="black", linewidth=0.3, zorder=3)
    ax.set_xlabel(r"$\Theta$ (h)")
    ax.set_ylabel(ylabel)
    ax.set_title(PANEL_TITLE[col], fontsize=7.5)
    ax.grid(True, alpha=0.25)


def main():
    sub = _load()
    if sub.empty:
        print(f"[skip] {NAME}: master_table.csv has no lattice_a/lattice_c after "
              f"nano-layer scoping")
        return
    n, n_cond = len(sub), sub["condition_id"].nunique()
    assert n == 51 and n_cond == 17, (
        f"expected n=51/17 conditions after nano-layer scoping, got n={n}/{n_cond}"
    )

    panel_letter = {"lattice_a": "a", "lattice_c": "b", "c_a_ratio": "c"}
    for col, _ylabel in PANELS:
        dump_source_data(
            NAME, panel_letter[col],
            sub[["sample_id", "condition_id", "precursor", "T_C", "Theta", col]].copy(),
            note=f"Nano-layer-scoped n=51 {col} vs Theta scatter.",
            source_files=[MASTER_TABLE],
        )

    fig = plt.figure(figsize=double_column_figsize(80))
    gs = fig.add_gridspec(1, 3, wspace=0.42,
                          left=0.07, right=0.98, top=0.78, bottom=0.16)
    panel_labels = ["(a)", "(b)", "(c)"]
    axes = []
    for i, (col, ylabel) in enumerate(PANELS):
        ax = fig.add_subplot(gs[0, i])
        _draw_panel(ax, sub, col, ylabel)
        add_panel_label(ax, panel_labels[i])
        axes.append(ax)

    handles = [plt.Line2D([0], [0], marker=m, color="w", markerfacecolor="grey",
                           markeredgecolor="black", markersize=6,
                           label=PRECURSOR_LABEL_EN[p])
               for p, m in PRECURSOR_MARKER.items()]
    handles += [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                            markeredgecolor="black", markersize=6, label=f"{t:.0f} C")
                for t, c in TEMP_COLOR.items()]
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=6.2,
              handlelength=1.1, columnspacing=1.0, labelspacing=0.35,
              bbox_to_anchor=(0.5, 0.99), frameon=False)

    caption = (
        "Fig. S3. R-3m lattice parameters (multi-peak least-squares fit) "
        "vs. normalized thermal exposure (Theta): (a) lattice_a, "
        "(b) lattice_c, (c) c/a ratio. Colour codes sintering temperature "
        "(850/900/950 C), marker shape codes precursor (circle S, square M, "
        "triangle L) -- same double encoding as the main-text figures. "
        "XRD covers 17 complete condition triplets (51 samples), spanning "
        "the full Theta range; same subset as Fig. 3a (D_XRD vs. Theta) and "
        "Fig. S1 (Arrhenius consistency plot). Williamson-Hall size/"
        "microstrain (D_XRD_WH, microstrain_WH) are deliberately excluded "
        "from this figure set -- Williamson-Hall analysis is retained as a "
        "qualitative cross-reference only and must not be cited as a "
        "quantitative result, so it is not plotted here."
    )
    # 3-panel figure sits close to the 190mm double-column width budget;
    # matplotlib's default savefig pad_inches (0.1in/side, applied on top of
    # the tight bbox by save_fig_journal's bbox_inches="tight") pushed the
    # exported PNG to ~191mm. Same fix as fig2_variance_decomposition.py:
    # tighten the pad locally (this script only, restored after saving).
    _old_pad = plt.rcParams["savefig.pad_inches"]
    plt.rcParams["savefig.pad_inches"] = 0.02
    try:
        save_fig_journal(fig, NAME, caption)
    finally:
        plt.rcParams["savefig.pad_inches"] = _old_pad
    plt.close(fig)
    return fig


if __name__ == "__main__":
    main()
