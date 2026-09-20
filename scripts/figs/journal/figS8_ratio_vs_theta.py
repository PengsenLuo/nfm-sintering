# -*- coding: utf-8 -*-
"""figS8_ratio_vs_theta.py -- Fig. S8 (journal submission set, supplementary
information, W7): does the D(104)/D(003) anisotropy ratio drift
systematically with thermal exposure or temperature? Two panels: (a) ratio
vs. Theta, (b) ratio vs. T_C, both colour/marker-coded by precursor.

Pure visualization of an already-computed, already-reported result -- no new
analysis is performed here. Data: data/interim/dxrd_hkl_anisotropy.csv
(same file/scope as figS7), columns Theta/T_C/ratio_104_003/precursor.

The annotated Spearman rho/p values are quoted from
reports/dxrd_hkl_anisotropy.md Section 3 (rho=+0.636 vs. Theta,
rho=+0.723 vs. T_C, both p<0.0001). This script also recomputes the same
statistics locally as a sanity check; on disagreement beyond tolerance it
prints a warning and still uses the report's numbers for the caption text.
"""
import _bootstrap  # noqa: F401

import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from _journal_style import (
    double_column_figsize,
    add_panel_label,
    save_fig_journal,
    dump_source_data,
    REPO_ROOT,
    PRECURSOR_COLOR,
    PRECURSOR_MARKER,
)

NAME = "figS8_ratio_vs_theta"
CSV = REPO_ROOT / "data" / "interim" / "dxrd_hkl_anisotropy.csv"
REPORT_MD = REPO_ROOT / "reports" / "dxrd_hkl_anisotropy.md"

PRECURSOR_LABEL_EN = {
    "S": "S (D50~3 um)",
    "M": "M (7:3 blend)",
    "L": "L (D50~11 um)",
}

# Frozen numbers from reports/dxrd_hkl_anisotropy.md Section 3.
REPORT_RHO_THETA = 0.636
REPORT_RHO_TC = 0.723
REPORT_P_THETA = 0.0001  # report prints "0.0000"; use its stated bound p<0.0001
REPORT_P_TC = 0.0001
TOL = 0.02


def _draw_panel(ax, df: pd.DataFrame, x_col: str, xlabel: str, rho: float, p_bound: float) -> None:
    for prec, marker in PRECURSOR_MARKER.items():
        g = df[df["precursor"] == prec]
        if g.empty:
            continue
        ax.scatter(g[x_col], g["ratio_104_003"], color=PRECURSOR_COLOR[prec],
                   marker=marker, s=16, edgecolor="black", linewidth=0.3,
                   zorder=3, label=PRECURSOR_LABEL_EN[prec])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$D_{XRD}$(104) / $D_{XRD}$(003)")
    ax.grid(True, alpha=0.25)
    ax.text(0.04, 0.94, f"Spearman $\\rho$ = {rho:+.3f}\np < {p_bound:.4f}",
           transform=ax.transAxes, fontsize=5.8, color="#333333",
           ha="left", va="top",
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                    edgecolor="#cccccc", linewidth=0.5))


def main():
    if not CSV.exists():
        print(f"[skip] {NAME}: dxrd_hkl_anisotropy.csv missing")
        return
    df = pd.read_csv(CSV)
    if df.empty:
        print(f"[skip] {NAME}: dxrd_hkl_anisotropy.csv is empty")
        return
    n, n_cond = len(df), df["condition_id"].nunique()
    assert n == 51 and n_cond == 17, (
        f"expected n=51/17 conditions (nano-layer scope), got n={n}/{n_cond}"
    )

    rho_theta, p_theta = stats.spearmanr(df["Theta"], df["ratio_104_003"])
    rho_tc, p_tc = stats.spearmanr(df["T_C"], df["ratio_104_003"])

    mismatches = []
    if abs(rho_theta - REPORT_RHO_THETA) > TOL:
        mismatches.append(f"rho(ratio,Theta) local={rho_theta:.3f} vs report={REPORT_RHO_THETA:.3f}")
    if abs(rho_tc - REPORT_RHO_TC) > TOL:
        mismatches.append(f"rho(ratio,T_C) local={rho_tc:.3f} vs report={REPORT_RHO_TC:.3f}")
    if mismatches:
        print(f"[WARNING] {NAME}: local recomputation disagrees with "
              f"{REPORT_MD.name}: " + "; ".join(mismatches) +
              " -- using report numbers for caption/annotation regardless.")

    for x_col, panel_letter, rho_val in [("Theta", "a", REPORT_RHO_THETA), ("T_C", "b", REPORT_RHO_TC)]:
        dump_source_data(
            NAME, panel_letter,
            df[["sample_id", "condition_id", "precursor", x_col, "ratio_104_003"]].copy(),
            note=f"D(104)/D(003) ratio vs {x_col}, nano-layer-scoped n=51. "
                 f"Frozen Spearman rho={rho_val:+.3f} (reports/"
                 f"dxrd_hkl_anisotropy.md Section 3).",
            source_files=[CSV],
        )

    fig = plt.figure(figsize=double_column_figsize(75))
    gs = fig.add_gridspec(1, 2, wspace=0.38,
                          left=0.09, right=0.98, top=0.80, bottom=0.16)

    ax_a = fig.add_subplot(gs[0, 0])
    _draw_panel(ax_a, df, "Theta", r"$\Theta$ (h)", REPORT_RHO_THETA, REPORT_P_THETA)
    add_panel_label(ax_a, "(a)")

    ax_b = fig.add_subplot(gs[0, 1])
    _draw_panel(ax_b, df, "T_C", r"$T$ ($^{\circ}$C)", REPORT_RHO_TC, REPORT_P_TC)
    add_panel_label(ax_b, "(b)")

    handles = [plt.Line2D([0], [0], marker=m, color="w",
                          markerfacecolor=PRECURSOR_COLOR[p],
                          markeredgecolor="black", markersize=6,
                          label=PRECURSOR_LABEL_EN[p])
              for p, m in PRECURSOR_MARKER.items()]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=6.4,
              handlelength=1.1, columnspacing=1.2, labelspacing=0.35,
              bbox_to_anchor=(0.5, 0.99), frameon=False)

    caption = (
        "Fig. S8. The D(104)/D(003) grain-size anisotropy ratio (see Fig. "
        "S7) plotted against (a) normalized thermal exposure Theta and (b) "
        "sintering temperature T, nano-layer-scoped n=51 samples (17 "
        "complete condition triplets), colour/marker-coded by precursor. "
        "Data: data/interim/dxrd_hkl_anisotropy.csv, the frozen output of "
        "a prior, already-completed diagnostic round "
        "(scripts/27_dxrd_hkl_anisotropy.py; not re-run here). Numbers "
        f"quoted from reports/dxrd_hkl_anisotropy.md Section 3: Spearman "
        f"rho(ratio, Theta) = {REPORT_RHO_THETA:+.3f} (p<0.0001, n=51); "
        f"Spearman rho(ratio, T) = {REPORT_RHO_TC:+.3f} (p<0.0001, n=51); "
        f"a Kruskal-Wallis test finds no significant precursor-group "
        f"difference in the ratio (H=1.58, p=0.453). The ratio therefore "
        f"drifts systematically with thermal exposure and temperature -- "
        f"it is not a constant crystallographic offset -- and this "
        f"drift, not precursor identity, is the dominant source of "
        f"variation in the anisotropy; the report this figure visualizes "
        f"concludes the resulting directional bias should be stated "
        f"explicitly rather than dismissed as 'the two peaks are "
        f"parallel'."
    )
    save_fig_journal(fig, NAME, caption)
    plt.close(fig)
    return fig


if __name__ == "__main__":
    main()
