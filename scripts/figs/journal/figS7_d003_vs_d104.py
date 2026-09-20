# -*- coding: utf-8 -*-
"""figS7_d003_vs_d104.py -- Fig. S7 (journal submission set, supplementary
information, W7): D(003) vs. D(104) Scherrer grain size, nano-layer-scoped
n=51 samples, colour/marker-coded by precursor, with a dashed 1:1 reference
line (dashed, not solid, because the report below finds the two peaks are
NOT numerically equal -- see caption).

Pure visualization of an already-computed, already-reported result -- no new
analysis is performed here. Data: data/interim/dxrd_hkl_anisotropy.csv
(columns sample_id, condition_id, precursor, Theta, T_C, D_XRD, D_XRD_003,
D_XRD_104, ratio_104_003), the frozen output of a prior, already-completed
diagnostic round (scripts/27_dxrd_hkl_anisotropy.py; not re-run here).

The annotated Pearson r / Spearman rho / median ratio are the numbers frozen
in reports/dxrd_hkl_anisotropy.md Section 1 (that report's own independent
recomputation, not an earlier alternative set of numbers it explicitly
declined to adopt). This script also recomputes the same three statistics
locally as a sanity check; if the local recomputation disagrees with the
report's frozen numbers beyond a tight tolerance, it prints a warning and
still uses the report's numbers for the caption text, so the caption cannot
silently drift from the frozen report.
"""
import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from _journal_style import (
    single_column_figsize,
    save_fig_journal,
    dump_source_data,
    REPO_ROOT,
    PRECURSOR_COLOR,
    PRECURSOR_MARKER,
)

NAME = "figS7_d003_vs_d104"
CSV = REPO_ROOT / "data" / "interim" / "dxrd_hkl_anisotropy.csv"
REPORT_MD = REPO_ROOT / "reports" / "dxrd_hkl_anisotropy.md"

PRECURSOR_LABEL_EN = {
    "S": "S (D50~3 um)",
    "M": "M (7:3 blend)",
    "L": "L (D50~11 um)",
}

# Frozen numbers from reports/dxrd_hkl_anisotropy.md Section 1 (this
# report's own independent recomputation, quoted verbatim -- not an
# earlier alternative set of numbers the report explicitly declined to
# adopt).
REPORT_PEARSON_R = 0.9896
REPORT_SPEARMAN_RHO = 0.9857
REPORT_MEDIAN_RATIO = 1.2502
TOL = 0.01


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

    d003, d104 = df["D_XRD_003"].to_numpy(), df["D_XRD_104"].to_numpy()
    pearson_r, _ = stats.pearsonr(d003, d104)
    spearman_rho, _ = stats.spearmanr(d003, d104)
    median_ratio = df["ratio_104_003"].median()

    mismatches = []
    if abs(pearson_r - REPORT_PEARSON_R) > TOL:
        mismatches.append(f"pearson_r local={pearson_r:.4f} vs report={REPORT_PEARSON_R:.4f}")
    if abs(spearman_rho - REPORT_SPEARMAN_RHO) > TOL:
        mismatches.append(f"spearman_rho local={spearman_rho:.4f} vs report={REPORT_SPEARMAN_RHO:.4f}")
    if abs(median_ratio - REPORT_MEDIAN_RATIO) > TOL:
        mismatches.append(f"median_ratio local={median_ratio:.4f} vs report={REPORT_MEDIAN_RATIO:.4f}")
    if mismatches:
        print(f"[WARNING] {NAME}: local recomputation disagrees with "
              f"{REPORT_MD.name}: " + "; ".join(mismatches) +
              " -- using report numbers for caption/annotation regardless.")

    dump_source_data(
        NAME, "single",
        df[["sample_id", "condition_id", "precursor", "D_XRD_003", "D_XRD_104"]].copy(),
        note=f"Nano-layer-scoped n=51 D_XRD(003) vs D_XRD(104) scatter. "
             f"Frozen stats quoted from reports/dxrd_hkl_anisotropy.md: "
             f"Pearson r={REPORT_PEARSON_R:+.4f}, Spearman "
             f"rho={REPORT_SPEARMAN_RHO:+.4f}, median "
             f"D(104)/D(003)={REPORT_MEDIAN_RATIO:.4f}.",
        source_files=[CSV],
    )

    fig = plt.figure(figsize=single_column_figsize(85))
    ax = fig.add_subplot(
        fig.add_gridspec(1, 1, left=0.16, right=0.97, top=0.97, bottom=0.13)[0]
    )

    for prec, marker in PRECURSOR_MARKER.items():
        g = df[df["precursor"] == prec]
        if g.empty:
            continue
        ax.scatter(g["D_XRD_003"], g["D_XRD_104"], color=PRECURSOR_COLOR[prec],
                   marker=marker, s=18, edgecolor="black", linewidth=0.3,
                   zorder=3, label=PRECURSOR_LABEL_EN[prec])

    lo = min(d003.min(), d104.min()) * 0.92
    hi = max(d003.max(), d104.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], color="#999999", linewidth=0.9,
            linestyle="--", zorder=1, label="1:1 reference")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(r"$D_{XRD}$(003) (nm)")
    ax.set_ylabel(r"$D_{XRD}$(104) (nm)")
    ax.legend(loc="upper left", fontsize=5.6, handlelength=1.2, labelspacing=0.35)
    ax.grid(True, alpha=0.25)

    ax.text(0.97, 0.06,
           f"Pearson r = {REPORT_PEARSON_R:+.4f}\n"
           f"Spearman $\\rho$ = {REPORT_SPEARMAN_RHO:+.4f}\n"
           f"median D(104)/D(003) = {REPORT_MEDIAN_RATIO:.4f}",
           transform=ax.transAxes, fontsize=5.6, color="#333333",
           ha="right", va="bottom",
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                    edgecolor="#cccccc", linewidth=0.5))

    caption = (
        "Fig. S7. Scherrer grain size from the (003) reflection vs. the "
        "(104) reflection, nano-layer-scoped n=51 samples (17 complete "
        "condition triplets), colour/marker-coded by precursor (circle S, "
        "square M, triangle L). Dashed line: 1:1 reference (dashed, not "
        "solid, because the two peaks are not numerically equal -- see "
        "below). Data: data/interim/dxrd_hkl_anisotropy.csv, the frozen "
        "output of a prior, already-completed diagnostic round "
        "(scripts/27_dxrd_hkl_anisotropy.py; not re-run here). Numbers "
        f"quoted from reports/dxrd_hkl_anisotropy.md Section 1 (that "
        f"report's own independent recomputation): Pearson r = "
        f"{REPORT_PEARSON_R:+.4f}, Spearman rho = {REPORT_SPEARMAN_RHO:+.4f} "
        f"(both p<1e-38, n=51) -- the two peaks are highly consistent in "
        f"ranking -- but a paired Wilcoxon test finds the two peaks "
        f"systematically differ in absolute value (p=5.1e-10, not a random-"
        f"noise difference), with median D(104)/D(003) = "
        f"{REPORT_MEDIAN_RATIO:.4f}, i.e. an approximately 25% systematic "
        f"directional (crystallographic-anisotropy) offset between the two "
        f"reflections. The production D_XRD column (used throughout the "
        f"main text) is the unweighted average of these two peaks; this "
        f"figure is a diagnostic of that averaging choice, not a change to "
        f"it."
    )
    save_fig_journal(fig, NAME, caption, panel_labels=False)
    plt.close(fig)
    return fig


if __name__ == "__main__":
    main()
