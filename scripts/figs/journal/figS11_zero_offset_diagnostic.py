# -*- coding: utf-8 -*-
"""figS11_zero_offset_diagnostic.py -- Fig. S11 (journal submission set,
supplementary information, W7): XRD second zero-offset (z) diagnostic.
(a) fitted zero-shift z vs. Theta, coloured by precursor, n=51. (b) per-hkl
2-theta residual vs. cos(theta), split into the 00l family (003, 006) vs.
all other hkl, showing the opposite-sign angular-dependence pattern found
in reports/xrd_peak_position_reconstruction.md Section 1.3.

Pure visualization of an already-computed, already-reported diagnostic --
no new analysis is performed here, and the production
xrd_processor.lattice_params() implementation (which already fits and
applies z) is not changed. Data:
data/interim/xrd_zero_offset_diagnostic.csv (panel a; columns sample_id,
precursor, Theta, T_C, D_XRD, lattice_a, lattice_c, zero_shift,
two_theta_offset_applied) and data/interim/xrd_hkl_residuals.csv (panel b;
columns sample_id, hkl, precursor, two_theta_obs, two_theta_calc,
residual_deg, fwhm, eta, cos_theta, sin_theta, D_XRD), the frozen outputs of
a prior, already-completed round (scripts/34_xrd_zero_offset_diagnostic.py;
not re-run here).

IMPORTANT (per reports/xrd_peak_position_reconstruction.md Section 1.3):
this is a genuinely nuanced, unresolved finding -- neither "hypothesis A"
(a real sample-height shift, which would predict a single-sign
Delta(2theta) ~ cos(theta) relationship across ALL hkl) nor "hypothesis B"
(a peak-width-dependent bias) is supported by the report's own tests. The
00l family shows a strong negative correlation with cos(theta) while the
non-00l family shows a weak POSITIVE one -- opposite signs -- which is
exactly why hypothesis A (uniform sample-height shift) is rejected across
the full peak set even though it holds within the 00l family alone. The
caption below states this as an open, unexplained pattern, not a clean
mechanistic story; do not oversimplify it.
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

NAME = "figS11_zero_offset_diagnostic"
ZERO_CSV = REPO_ROOT / "data" / "interim" / "xrd_zero_offset_diagnostic.csv"
RESID_CSV = REPO_ROOT / "data" / "interim" / "xrd_hkl_residuals.csv"
REPORT_MD = REPO_ROOT / "reports" / "xrd_peak_position_reconstruction.md"

PRECURSOR_LABEL_EN = {
    "S": "S (D50~3 um)",
    "M": "M (7:3 blend)",
    "L": "L (D50~11 um)",
}
# NOTE: data/interim/xrd_hkl_residuals.csv's `hkl` column round-trips through
# CSV as int64, so "003"/"006" lose their leading zeros on read (become 3/6)
# -- match against the int values, not the zero-padded strings, or every row
# silently falls into the "non-00l" bucket (caught here: an earlier version
# of this script did exactly that and its "non-00l" Spearman rho matched the
# report's *all-hkl* rho of -0.1571 instead of the correct +0.2127 for the
# non-00l-only subset -- see reports/xrd_peak_position_reconstruction.md
# Section 1.3).
OOL_HKL = {3, 6}

# Frozen numbers from reports/xrd_peak_position_reconstruction.md Section 1.
REPORT_RHO_Z_THETA = -0.573
REPORT_RHO_OOL = -0.7629
REPORT_RHO_NON_OOL = 0.2127
TOL = 0.02


def main():
    if not (ZERO_CSV.exists() and RESID_CSV.exists()):
        print(f"[skip] {NAME}: xrd_zero_offset_diagnostic.csv or "
              f"xrd_hkl_residuals.csv missing")
        return
    zdf = pd.read_csv(ZERO_CSV)
    rdf = pd.read_csv(RESID_CSV)
    if zdf.empty or rdf.empty:
        print(f"[skip] {NAME}: one of the two input CSVs is empty")
        return

    n_z = len(zdf)
    assert n_z == 51, f"expected n=51 (nano-layer scope) in zero-offset CSV, got n={n_z}"

    rho_z_theta, _ = stats.spearmanr(zdf["Theta"], zdf["zero_shift"])
    if abs(rho_z_theta - REPORT_RHO_Z_THETA) > TOL:
        print(f"[WARNING] {NAME}: local rho(z,Theta)={rho_z_theta:.4f} vs "
              f"{REPORT_MD.name}={REPORT_RHO_Z_THETA:.4f} -- using report "
              f"number for caption/annotation regardless.")

    rdf = rdf.copy()
    is_ool = rdf["hkl"].isin(OOL_HKL)
    ool = rdf[is_ool]
    non_ool = rdf[~is_ool]
    rho_ool, _ = stats.spearmanr(ool["residual_deg"], ool["cos_theta"])
    rho_non_ool, _ = stats.spearmanr(non_ool["residual_deg"], non_ool["cos_theta"])
    if abs(rho_ool - REPORT_RHO_OOL) > TOL:
        print(f"[WARNING] {NAME}: local rho(00l)={rho_ool:.4f} vs "
              f"{REPORT_MD.name}={REPORT_RHO_OOL:.4f} -- using report "
              f"number for caption/annotation regardless.")
    if abs(rho_non_ool - REPORT_RHO_NON_OOL) > TOL:
        print(f"[WARNING] {NAME}: local rho(non-00l)={rho_non_ool:.4f} vs "
              f"{REPORT_MD.name}={REPORT_RHO_NON_OOL:.4f} -- using report "
              f"number for caption/annotation regardless.")

    dump_source_data(
        NAME, "a", zdf[["sample_id", "condition_id", "precursor", "Theta", "zero_shift"]].copy(),
        note=f"Nano-layer-scoped n=51 fitted zero-shift z vs Theta. Frozen "
             f"Spearman(z,Theta)={REPORT_RHO_Z_THETA:+.3f} (reports/"
             f"xrd_peak_position_reconstruction.md Section 1).",
        source_files=[ZERO_CSV],
    )
    rdf_panel = rdf[["sample_id", "hkl", "precursor", "cos_theta", "residual_deg"]].copy()
    rdf_panel["is_ool_family"] = rdf["hkl"].isin(OOL_HKL)
    dump_source_data(
        NAME, "b", rdf_panel,
        note=f"510 residual points (51 samples x 10 hkl): 2theta_obs - "
             f"2theta_calc vs cos(theta), split into the 00l family (hkl "
             f"3,6) vs the other 8 hkl. Frozen "
             f"Spearman(residual,cos_theta): 00l={REPORT_RHO_OOL:+.4f}, "
             f"non-00l={REPORT_RHO_NON_OOL:+.4f} (reports/"
             f"xrd_peak_position_reconstruction.md Section 1.3).",
        source_files=[RESID_CSV],
    )

    fig = plt.figure(figsize=double_column_figsize(78))
    gs = fig.add_gridspec(1, 2, wspace=0.40,
                          left=0.08, right=0.98, top=0.80, bottom=0.17)

    # Panel (a): zero_shift z vs Theta, coloured/marker by precursor.
    ax_a = fig.add_subplot(gs[0, 0])
    for prec, marker in PRECURSOR_MARKER.items():
        g = zdf[zdf["precursor"] == prec]
        if g.empty:
            continue
        ax_a.scatter(g["Theta"], g["zero_shift"], color=PRECURSOR_COLOR[prec],
                    marker=marker, s=16, edgecolor="black", linewidth=0.3,
                    zorder=3, label=PRECURSOR_LABEL_EN[prec])
    ax_a.set_xlabel(r"$\Theta$ (h)")
    ax_a.set_ylabel(r"fitted zero-shift $z$ ($^{\circ}2\theta$)")
    ax_a.grid(True, alpha=0.25)
    ax_a.legend(loc="upper right", fontsize=5.6, handlelength=1.2, labelspacing=0.35)
    ax_a.text(0.04, 0.06, f"Spearman $\\rho$ = {REPORT_RHO_Z_THETA:+.3f}",
             transform=ax_a.transAxes, fontsize=5.8, color="#333333",
             ha="left", va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.5))
    add_panel_label(ax_a, "(a)")

    # Panel (b): per-hkl 2-theta residual vs cos(theta), 00l family vs rest.
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.scatter(non_ool["cos_theta"], non_ool["residual_deg"], color="#457b9d",
                marker="o", s=12, edgecolor="black", linewidth=0.25,
                alpha=0.75, zorder=2, label="non-00l (8 hkl)")
    ax_b.scatter(ool["cos_theta"], ool["residual_deg"], color="#e63946",
                marker="^", s=16, edgecolor="black", linewidth=0.25,
                zorder=3, label="00l family (003, 006)")
    ax_b.axhline(0, color="#999999", linewidth=0.6, linestyle="-", zorder=1)
    ax_b.set_xlabel(r"$\cos\theta$")
    ax_b.set_ylabel(r"$2\theta_{obs} - 2\theta_{calc}$ ($^{\circ}$)")
    ax_b.grid(True, alpha=0.25)
    ax_b.legend(loc="upper right", fontsize=5.6, handlelength=1.2, labelspacing=0.35)
    ax_b.text(0.04, 0.06,
             f"00l: $\\rho$={REPORT_RHO_OOL:+.3f}\n"
             f"non-00l: $\\rho$={REPORT_RHO_NON_OOL:+.3f}",
             transform=ax_b.transAxes, fontsize=5.6, color="#333333",
             ha="left", va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.5))
    add_panel_label(ax_b, "(b)")

    caption = (
        "Fig. S11. XRD second zero-offset (z) diagnostic, nano-layer-"
        "scoped n=51 samples (17 complete condition triplets). (a) Fitted "
        "zero-shift z vs. normalized thermal exposure Theta, colour/marker-"
        "coded by precursor. (b) Per-hkl 2theta residual "
        "(2theta_obs - 2theta_calc, from the final fitted a/c/z for each "
        "sample) vs. cos(theta), split into the 00l family (003, 006; red "
        "triangles) vs. all other measured hkl (012/015/018/101/104/107/"
        "110/113; blue circles), 510 residual points total (51 samples x "
        "10 hkl). Data: data/interim/xrd_zero_offset_diagnostic.csv (panel "
        "a) and data/interim/xrd_hkl_residuals.csv (panel b), the frozen "
        "outputs of a prior, already-completed round "
        "(scripts/34_xrd_zero_offset_diagnostic.py; not re-run here) that "
        "does not change the production lattice_params() implementation. "
        f"Numbers quoted from reports/xrd_peak_position_reconstruction.md "
        f"Section 1: Spearman(z, Theta) = {REPORT_RHO_Z_THETA:+.3f} "
        f"(p=1.1e-05); Spearman(residual, cos(theta)) restricted to the "
        f"00l family = {REPORT_RHO_OOL:+.4f} (p=1.2e-20, strongly "
        f"negative) vs. restricted to non-00l reflections = "
        f"{REPORT_RHO_NON_OOL:+.4f} (p=1.5e-05, weakly positive) -- "
        f"opposite signs. That report treats this as an open, unresolved "
        f"pattern rather than a clean mechanistic explanation: the "
        f"opposite-sign split across the full peak set rules out the "
        f"simplest 'hypothesis A' (a uniform real sample-height shift, "
        f"which predicts a single-sign residual-cos(theta) relationship "
        f"across all hkl) even though that relationship does hold within "
        f"the 00l family alone; a peak-width-dependent bias ('hypothesis "
        f"B') is separately tested and also not supported "
        f"(Spearman(|residual|, FWHM) rho=+0.009, p=0.84). The zero-offset "
        f"z (panel a) is negatively correlated with Theta and, in a "
        f"companion diagnostic in the same report, more strongly "
        f"correlated with D_XRD itself (rho=-0.752); readers should treat "
        f"z as an empirical second-order correction term whose physical "
        f"origin is not yet identified, not as evidence for any single "
        f"specific instrumental or sample-geometry mechanism."
    )
    save_fig_journal(fig, NAME, caption)
    plt.close(fig)
    return fig


if __name__ == "__main__":
    main()
