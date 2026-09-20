# -*- coding: utf-8 -*-
"""figS9_deconvolution_sensitivity.py -- Fig. S9 (journal submission set,
supplementary information, W7): D_XRD deconvolution-method sensitivity --
Gaussian (current production Kalpha2 subtraction) vs. linear/Lorentzian
alternative, paired per-sample scatter with a 1:1 reference line, n=51.

Pure visualization of an already-computed, already-reported sensitivity
check -- no new analysis is performed here, and the production Gaussian
D_XRD is NOT changed by this figure. Data:
data/interim/dxrd_deconvolution_sensitivity.csv (columns include
D_XRD_gauss_reimpl, D_XRD_lorentz, precursor), the frozen output of a prior,
already-completed round (scripts/28_dxrd_deconvolution_sensitivity.py; not
re-run here).

The annotated cross-method Spearman rho and median relative difference are
quoted from reports/dxrd_deconvolution_sensitivity.md Sections 3/5. This
script also recomputes the same two statistics locally as a sanity check;
on disagreement beyond tolerance it prints a warning and still uses the
report's numbers for the caption text.
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

NAME = "figS9_deconvolution_sensitivity"
CSV = REPO_ROOT / "data" / "interim" / "dxrd_deconvolution_sensitivity.csv"
REPORT_MD = REPO_ROOT / "reports" / "dxrd_deconvolution_sensitivity.md"

PRECURSOR_LABEL_EN = {
    "S": "S (D50~3 um)",
    "M": "M (7:3 blend)",
    "L": "L (D50~11 um)",
}

# Frozen numbers from reports/dxrd_deconvolution_sensitivity.md Sections 3/5.
REPORT_SPEARMAN_RHO = 0.9999
REPORT_MEDIAN_REL_DIFF_PCT = 75.81
TOL_RHO = 0.001
TOL_DIFF = 2.0


def main():
    if not CSV.exists():
        print(f"[skip] {NAME}: dxrd_deconvolution_sensitivity.csv missing")
        return
    df = pd.read_csv(CSV)
    if df.empty:
        print(f"[skip] {NAME}: dxrd_deconvolution_sensitivity.csv is empty")
        return
    df = df[df["D_XRD_gauss_reimpl"].notna() & df["D_XRD_lorentz"].notna()].copy()
    if df.empty:
        print(f"[skip] {NAME}: no rows with both D_XRD_gauss_reimpl and D_XRD_lorentz")
        return
    n = len(df)
    assert n == 51, f"expected n=51 (nano-layer scope), got n={n}"

    gauss = df["D_XRD_gauss_reimpl"].to_numpy()
    lorentz = df["D_XRD_lorentz"].to_numpy()
    spearman_rho, _ = stats.spearmanr(gauss, lorentz)
    rel_diff_pct = (lorentz - gauss) / gauss * 100.0
    median_rel_diff = np.median(rel_diff_pct)

    mismatches = []
    if abs(spearman_rho - REPORT_SPEARMAN_RHO) > TOL_RHO:
        mismatches.append(f"spearman_rho local={spearman_rho:.4f} vs report={REPORT_SPEARMAN_RHO:.4f}")
    if abs(median_rel_diff - REPORT_MEDIAN_REL_DIFF_PCT) > TOL_DIFF:
        mismatches.append(f"median_rel_diff local={median_rel_diff:.2f}% vs report={REPORT_MEDIAN_REL_DIFF_PCT:.2f}%")
    if mismatches:
        print(f"[WARNING] {NAME}: local recomputation disagrees with "
              f"{REPORT_MD.name}: " + "; ".join(mismatches) +
              " -- using report numbers for caption/annotation regardless.")

    dump_source_data(
        NAME, "single",
        df[["sample_id", "precursor", "T_C", "D_XRD_gauss_reimpl", "D_XRD_lorentz"]].copy(),
        note=f"Nano-layer-scoped n=51 paired D_XRD, Gaussian vs "
             f"linear/Lorentzian Kalpha2-subtraction. Frozen stats quoted "
             f"from reports/dxrd_deconvolution_sensitivity.md: Spearman "
             f"rho={REPORT_SPEARMAN_RHO:.4f}, median relative "
             f"difference={REPORT_MEDIAN_REL_DIFF_PCT:+.1f}%.",
        source_files=[CSV],
    )

    fig = plt.figure(figsize=single_column_figsize(88))
    ax = fig.add_subplot(
        fig.add_gridspec(1, 1, left=0.16, right=0.97, top=0.97, bottom=0.13)[0]
    )

    for prec, marker in PRECURSOR_MARKER.items():
        g = df[df["precursor"] == prec]
        if g.empty:
            continue
        ax.scatter(g["D_XRD_gauss_reimpl"], g["D_XRD_lorentz"],
                   color=PRECURSOR_COLOR[prec], marker=marker, s=18,
                   edgecolor="black", linewidth=0.3, zorder=3,
                   label=PRECURSOR_LABEL_EN[prec])

    lo = min(gauss.min(), lorentz.min()) * 0.9
    hi = max(gauss.max(), lorentz.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], color="#999999", linewidth=0.9,
            linestyle="--", zorder=1, label="1:1 reference")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel(r"$D_{XRD}$, Gaussian $K\alpha_2$ subtraction (nm)")
    ax.set_ylabel(r"$D_{XRD}$, linear/Lorentzian subtraction (nm)")
    ax.legend(loc="upper left", fontsize=5.6, handlelength=1.2, labelspacing=0.35)
    ax.grid(True, alpha=0.25)

    ax.text(0.97, 0.06,
           f"Spearman $\\rho$ = {REPORT_SPEARMAN_RHO:.4f}\n"
           f"median rel. diff. = {REPORT_MEDIAN_REL_DIFF_PCT:+.1f}%",
           transform=ax.transAxes, fontsize=5.6, color="#333333",
           ha="right", va="bottom",
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                    edgecolor="#cccccc", linewidth=0.5))

    caption = (
        "Fig. S9. D_XRD deconvolution-method sensitivity: production "
        "Gaussian Kalpha2-subtraction values vs. an alternative "
        "linear/Lorentzian subtraction method, paired per-sample, nano-"
        "layer-scoped n=51, colour/marker-coded by precursor. Dashed line: "
        "1:1 reference. Data: "
        "data/interim/dxrd_deconvolution_sensitivity.csv, the frozen "
        "output of a prior, already-completed round "
        "(scripts/28_dxrd_deconvolution_sensitivity.py; not re-run here); "
        "this is a sensitivity/robustness check only -- it does not change "
        "the production Gaussian-based D_XRD used elsewhere in this study. "
        f"Numbers quoted from reports/dxrd_deconvolution_sensitivity.md "
        f"Sections 3 and 5: median D_XRD shifts from 63.6 nm (Gaussian) to "
        f"111.9 nm (linear/Lorentzian), a median relative difference of "
        f"{REPORT_MEDIAN_REL_DIFF_PCT:+.1f}%, i.e. absolute D_XRD values "
        f"are sensitive to the deconvolution method. However, the two "
        f"methods' per-sample D_XRD values are almost perfectly rank-"
        f"correlated (Spearman rho = {REPORT_SPEARMAN_RHO:.4f}), and "
        f"Spearman(D_XRD, Theta) is +0.818 under both methods -- so "
        f"ranking-based conclusions (e.g. the D_XRD-Theta trend) are "
        f"robust to this methodological choice even though the absolute "
        f"nanometre scale is not."
    )
    save_fig_journal(fig, NAME, caption, panel_labels=False)
    plt.close(fig)
    return fig


if __name__ == "__main__":
    main()
