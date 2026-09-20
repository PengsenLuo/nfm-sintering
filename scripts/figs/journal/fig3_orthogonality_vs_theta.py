# -*- coding: utf-8 -*-
"""fig3_orthogonality_vs_theta.py -- Fig. 3 (journal submission set):
(a) D_XRD vs Theta (nano-layer scoped, n=51) and (b) compaction_density vs
Theta (n=81, precursor-banded mean +/- 1 std), merged into one double-column
figure.

Panel (a) is a direct restyle of scripts/figs/fig_dxrd_vs_theta.py's
plot_vs_theta(): same data source (_load() below is a line-for-line port of
that script's _load(), reading master_table.csv through
nfm.nano_layer.nano_layer_frame(require_reliable=True) to get the 51-sample
nano-layer-scoped subset), same temperature-color / precursor-marker double
encoding; only the on-figure text is translated to English and the styling
switched to journal conventions.

Panel (b) is a direct restyle of scripts/figs/fig_density_vs_theta.py's
plot_vs_theta(): same data source (all 81 samples with non-null
compaction_density, no nano-layer scoping), same deliberate T8 design choice
of plotting per-precursor mean +/- 1 std horizontal bands (not OLS trend
lines) to visually emphasize "no significant within-group Theta trend,
clear between-group separation" -- see that script's own comments. This
script keeps that choice; it is not reverted to a trend-line style here.

IMPORTANT (house rule, 2026-08-02 authorial decision -- same rule fig2 in
this directory follows): main-text figures and captions must never explain
that the 51-vs-81 nano-layer sample-count split is due to a second XRD
measurement instrument, and must never list the excluded condition IDs
(doing so would let a careful reader infer the same instrument-split fact
that the plain sentence is designed to avoid disclosing). State "n=51,
17 complete condition triplets" as a plain fact, using exactly the sanctioned
sentence: "XRD covers 17 complete condition triplets (51 samples), spanning
the full Theta range." Do not write "instrument", "SmartLab", "MiniFlex", or
"xrd_instrument" anywhere in this file's on-figure text or caption, and do
not enumerate the 10 excluded condition IDs anywhere in this file.
"""
import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from _journal_style import (
    double_column_figsize,
    add_panel_label,
    save_fig_journal,
    dump_source_data,
    PRECURSOR_COLOR,
    PRECURSOR_MARKER,
    TEMP_COLOR,
)
from _style import MASTER_TABLE
from nfm.nano_layer import nano_layer_frame

NAME = "fig3_orthogonality_vs_theta"

# English precursor legend labels -- _style.py's PRECURSOR_LABEL contains
# Chinese text ("7:3混合"); on-figure text must be English-only, so this
# module defines its own English variant instead of importing that one.
PRECURSOR_LABEL_EN = {
    "S": "S (D50~3 um)",
    "M": "M (7:3 blend)",
    "L": "L (D50~11 um)",
}


# ---------------------------------------------------------------------
# Panel (a) data -- line-for-line port of fig_dxrd_vs_theta.py's _load()
# ---------------------------------------------------------------------
def _load_panel_a() -> pd.DataFrame:
    df = pd.read_csv(MASTER_TABLE)
    df = nano_layer_frame(df, require_reliable=True)
    sub = df[df["D_XRD"].notna()].copy()
    return sub


# ---------------------------------------------------------------------
# Panel (b) data -- same filter as fig_density_vs_theta.py's main()
# ---------------------------------------------------------------------
def _load_panel_b() -> pd.DataFrame:
    df = pd.read_csv(MASTER_TABLE)
    sub = df[df["compaction_density"].notna()].copy()
    return sub


def _draw_panel_a(ax, sub: pd.DataFrame) -> None:
    for t_c, color in TEMP_COLOR.items():
        g_t = sub[sub["T_C"] == t_c]
        for prec, marker in PRECURSOR_MARKER.items():
            g = g_t[g_t["precursor"] == prec]
            if g.empty:
                continue
            ax.scatter(g["Theta"], g["D_XRD"], color=color, marker=marker,
                       s=18, edgecolor="black", linewidth=0.3,
                       label=f"{t_c:.0f} C / {prec}")
    ax.set_xlabel(r"$\Theta$ (h)")
    ax.set_ylabel(r"$D_{XRD}$ (nm)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, ncol=3, fontsize=5.6, loc="upper left",
              handlelength=1.1, columnspacing=0.8, labelspacing=0.35)
    ax.grid(True, alpha=0.25)


def _draw_panel_b(ax, sub: pd.DataFrame) -> tuple[float, float]:
    """Draws panel (b) and returns (min_p, max_p) across the three
    per-precursor within-group Spearman tests, computed here (not
    hardcoded) so the caption's stated p-value range is independently
    verified against whatever is in master_table.csv at render time."""
    x_min, x_max = sub["Theta"].min(), sub["Theta"].max()
    x_pad = 0.03 * (x_max - x_min)
    band_x = (x_min - x_pad, x_max + x_pad)
    pvals = []
    for prec in ["S", "M", "L"]:
        g = sub[sub["precursor"] == prec]
        if g.empty:
            continue
        color = PRECURSOR_COLOR[prec]
        mean_d = g["compaction_density"].mean()
        std_d = g["compaction_density"].std()
        rho, pval = (np.nan, np.nan)
        if g["Theta"].nunique() >= 2:
            rho, pval = stats.spearmanr(g["Theta"], g["compaction_density"])
        if np.isfinite(pval):
            pvals.append(pval)

        ax.axhspan(mean_d - std_d, mean_d + std_d, xmin=0, xmax=1,
                   color=color, alpha=0.10, zorder=1)
        ax.hlines(mean_d, band_x[0], band_x[1], color=color,
                  linewidth=1.1, linestyle="-", zorder=2, alpha=0.8)
        rho_label = f"$\\rho$={rho:+.2f} [p={pval:.2f}]" if np.isfinite(rho) else "NA"
        ax.scatter(g["Theta"], g["compaction_density"], color=color,
                   marker=PRECURSOR_MARKER[prec], s=18, edgecolor="black",
                   linewidth=0.3,
                   label=f"{PRECURSOR_LABEL_EN[prec]} (n={len(g)}, {rho_label})",
                   zorder=3)

    ax.set_xlim(*band_x)
    ax.set_xlabel(r"$\Theta$ (h)")
    ax.set_ylabel(r"Compaction density (g/cm$^3$)")
    ax.legend(loc="best", fontsize=5.6, handlelength=1.1, labelspacing=0.4)
    ax.grid(True, alpha=0.25)
    if pvals:
        return min(pvals), max(pvals)
    return float("nan"), float("nan")


def main():
    sub_a = _load_panel_a()
    sub_b = _load_panel_b()
    if sub_a.empty or sub_b.empty:
        print(f"[skip] {NAME}: master_table.csv missing D_XRD or "
              f"compaction_density data")
        return

    n_a, n_cond_a = len(sub_a), sub_a["condition_id"].nunique()
    n_b, n_cond_b = len(sub_b), sub_b["condition_id"].nunique()
    assert n_a == 51 and n_cond_a == 17, (
        f"expected panel (a) n=51/17 conditions, got n={n_a}/{n_cond_a} "
        f"-- nano_layer_frame scoping may have changed"
    )
    assert n_b == 81 and n_cond_b == 27, (
        f"expected panel (b) n=81/27 conditions, got n={n_b}/{n_cond_b}"
    )

    fig = plt.figure(figsize=double_column_figsize(95))
    outer = fig.add_gridspec(1, 2, wspace=0.32,
                             left=0.08, right=0.98, top=0.90, bottom=0.14)
    ax_a = fig.add_subplot(outer[0])
    ax_b = fig.add_subplot(outer[1])

    _draw_panel_a(ax_a, sub_a)
    p_lo, p_hi = _draw_panel_b(ax_b, sub_b)

    add_panel_label(ax_a, "(a)")
    add_panel_label(ax_b, "(b)")

    dump_source_data(
        NAME, "a", sub_a[["sample_id", "condition_id", "precursor", "T_C", "Theta", "D_XRD"]].copy(),
        note="Nano-layer-scoped (n=51, 17 complete condition triplets) "
             "D_XRD vs Theta scatter points.",
        source_files=[MASTER_TABLE],
    )
    dump_source_data(
        NAME, "b", sub_b[["sample_id", "condition_id", "precursor", "Theta", "compaction_density"]].copy(),
        note=f"All 81 samples; compaction_density vs Theta scatter points. "
             f"Per-precursor horizontal mean+/-1-std bands are a trivial "
             f"aggregate of these same rows, not separately tabulated. "
             f"Within-group Spearman p-values span "
             f"[{p_lo:.4f},{p_hi:.4f}] across the 3 precursor groups.",
        source_files=[MASTER_TABLE],
    )

    p_range = (f"p in the {p_lo:.2f}-{p_hi:.2f} range" if np.isfinite(p_lo)
               else "p not computable (fewer than 2 distinct Theta values)")

    caption = (
        "Fig. 3. (a) Grain size (D_XRD, Scherrer analysis of the (003)/(104) "
        "reflections) vs. normalized thermal exposure (Theta). Colour codes "
        "sintering temperature (850/900/950 C), marker shape codes precursor "
        "(circle S, square M, triangle L). XRD covers 17 complete condition "
        "triplets (51 samples), spanning the full Theta range. (b) Compaction "
        "density vs. Theta for all 81 samples (27 conditions x S/M/L), grouped "
        "by precursor (colour and marker, same double encoding). Horizontal "
        "shaded bands mark each precursor group's compaction-density mean +/- "
        "1 std with the mean as a solid line, rather than an ordinary-least- "
        "squares trend line, because the within-group Theta dependence is not "
        "statistically significant for any of the three precursor groups "
        f"(legend states each group's Spearman rho and p-value, {p_range}); a "
        "sloped trend line would visually overstate a within-group relationship "
        "the data do not support. Together the two panels support an "
        "orthogonality argument: sintering process conditions (captured by "
        "Theta) drive grain-size evolution in panel (a) essentially independent "
        "of precursor identity, whereas compaction density in panel (b) is "
        "separated by precursor identity with no resolvable within-group "
        "process dependence -- i.e., process governs grain size but not "
        "packing, while precursor governs packing but not grain size."
    )
    save_fig_journal(fig, NAME, caption)
    plt.close(fig)
    return fig


if __name__ == "__main__":
    main()
