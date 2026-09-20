# -*- coding: utf-8 -*-
"""figS1_arrhenius.py -- Fig. S1 (journal submission set, supplementary
information): ln(D_XRD) vs. 1/T_K, per-precursor linear fits, restyled port
of scripts/figs/fig_grain_growth_arrhenius.py.

This is a direct restyle of that script's data logic (read-only source; not
imported, not modified): master_table.csv, scoped to the nano-layer 51-sample
subset via nfm.nano_layer.nano_layer_frame(require_reliable=True) (same
scoping fig3/figS3 use for D_XRD), grouped by precursor (S/M/L), each group
with >=3 points and >=2 distinct 1/T_K values gets an ordinary-least-squares
fit of ln(D_XRD) = ln(A) - Q/R * (1/T_K); the apparent activation energy is
q_app_kJ = -slope * R_GAS / 1000, exactly the original script's formula
(R_GAS = 8.314 J/mol/K), computed here fresh from the data rather than
hardcoded from the task spec's quoted "44-63 kJ/mol" figure.

This figure's apparent activation energy must not be read as a physical
kinetic result. The original script's own comment says so explicitly
("仅作一致性参照,不进入正式建模;真实生长动力学还依赖 t_eff,这里是简化
诊断"), and the physical literature range for this activation energy is
180-220 kJ/mol. The caption below states this in English, in the same terms:
t_eff (the ramp-time Arrhenius correction used everywhere else in this
project's kinetics treatment) is not included in this simplified single-slope
diagnostic, so the fitted Q systematically understates the true apparent
activation energy -- it is a consistency/sanity-check plot, not a reported
kinetic result, and must not be cited as one.

IMPORTANT (house rule, 2026-08-02 authorial decision -- same rule fig2/fig3
in this directory follow): this figure's data (D_XRD, nano-layer scoped) is
the same 51-sample/17-condition subset as fig3's panel (a). Main-text and SI
figures must never explain that the 51-vs-81 sample-count split is due to a
second XRD measurement instrument, and must never list the excluded
condition IDs. Use exactly the sanctioned sentence: "XRD covers 17 complete
condition triplets (51 samples), spanning the full Theta range." Do not
write "instrument", "SmartLab", "MiniFlex", or "xrd_instrument" anywhere in
this file's on-figure text or caption.
"""
import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from _journal_style import (
    single_column_figsize,
    save_fig_journal,
    dump_source_data,
    PRECURSOR_COLOR,
    PRECURSOR_MARKER,
)
from _style import MASTER_TABLE
from nfm.nano_layer import nano_layer_frame

NAME = "figS1_arrhenius"
R_GAS = 8.314  # J/(mol K), identical constant to the source script
LIT_Q_LO, LIT_Q_HI = 180.0, 220.0  # kJ/mol, physical literature reference range

PRECURSOR_LABEL_EN = {
    "S": "S (D50~3 um)",
    "M": "M (7:3 blend)",
    "L": "L (D50~11 um)",
}


def _load() -> pd.DataFrame:
    df = pd.read_csv(MASTER_TABLE)
    df = nano_layer_frame(df, require_reliable=True)
    sub = df[df["D_XRD"].notna()].copy()
    sub["lnD"] = np.log(sub["D_XRD"])
    return sub


def main():
    sub = _load()
    if sub.empty:
        print(f"[skip] {NAME}: master_table.csv has no D_XRD after nano-layer scoping")
        return
    n, n_cond = len(sub), sub["condition_id"].nunique()
    assert n == 51 and n_cond == 17, (
        f"expected n=51/17 conditions after nano-layer scoping, got n={n}/{n_cond}"
    )

    fig = plt.figure(figsize=single_column_figsize(85))
    ax = fig.add_subplot(
        fig.add_gridspec(1, 1, left=0.18, right=0.97, top=0.97, bottom=0.13)[0]
    )

    fit_rows = []
    for prec in ["S", "M", "L"]:
        g = sub[sub["precursor"] == prec]
        if len(g) < 2:
            continue
        color = PRECURSOR_COLOR[prec]
        ax.scatter(g["inv_T_K"], g["lnD"], color=color, marker=PRECURSOR_MARKER[prec],
                   s=16, edgecolor="black", linewidth=0.3,
                   label=f"{PRECURSOR_LABEL_EN[prec]} (n={len(g)})", zorder=3)
        if len(g) >= 3 and g["inv_T_K"].nunique() >= 2:
            res = stats.linregress(g["inv_T_K"], g["lnD"])
            x_line = np.linspace(g["inv_T_K"].min(), g["inv_T_K"].max(), 50)
            ax.plot(x_line, res.intercept + res.slope * x_line, color=color,
                    linestyle="--", linewidth=0.9, zorder=2)
            q_app_kJ = -res.slope * R_GAS / 1000.0
            fit_rows.append(dict(precursor=prec, n=len(g), slope=res.slope,
                                 r2=res.rvalue ** 2, q_app_kJ=q_app_kJ))

    assert fit_rows, "no precursor group had >=3 points / >=2 distinct 1/T_K values to fit"
    fit_df = pd.DataFrame(fit_rows)

    fit_note = "; ".join(
        f"{r.precursor}(n={r.n}): slope={r.slope:.6g}, R2={r.r2:.3f}, "
        f"Q_app={r.q_app_kJ:.1f} kJ/mol"
        for r in fit_df.itertuples()
    )
    dump_source_data(
        NAME, "single",
        sub[["sample_id", "condition_id", "precursor", "T_C", "inv_T_K", "lnD"]].copy(),
        note=f"Nano-layer-scoped n=51 ln(D_XRD) vs 1/T_K scatter (lnD = "
             f"ln(D_XRD), D_XRD in nm). Per-precursor OLS fits (NOT a "
             f"reported kinetic result, see caption): {fit_note}.",
        source_files=[MASTER_TABLE],
    )

    # Verify (do not assume) the task spec's quoted "44-63 kJ/mol" range.
    q_lo, q_hi = fit_df["q_app_kJ"].min(), fit_df["q_app_kJ"].max()
    all_below_lit = bool((fit_df["q_app_kJ"] < LIT_Q_LO).all())

    ax.set_xlabel(r"1/$T_K$ (K$^{-1}$)")
    ax.set_ylabel(r"ln($D_{XRD}$)  [$D_{XRD}$ in nm]")
    ax.legend(loc="lower left", fontsize=6.0, handlelength=1.1, labelspacing=0.35)
    ax.grid(True, alpha=0.25)

    fit_text = "\n".join(
        f"{r.precursor}: Q_app={r.q_app_kJ:.0f} kJ/mol, R2={r.r2:.2f}"
        for r in fit_df.itertuples()
    )
    ax.text(0.97, 0.97, fit_text, transform=ax.transAxes, fontsize=6.0,
           va="top", ha="right",
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                     edgecolor="#CCCCCC", alpha=0.9))

    fit_summary = "; ".join(
        f"{r.precursor} (n={r.n}): Q_app={r.q_app_kJ:.0f} kJ/mol, R^2={r.r2:.2f}"
        for r in fit_df.itertuples()
    )
    below_note = (
        f"All three fitted Q_app values ({q_lo:.0f}-{q_hi:.0f} kJ/mol) fall below "
        f"the {LIT_Q_LO:.0f}-{LIT_Q_HI:.0f} kJ/mol literature range."
        if all_below_lit else
        f"Fitted Q_app values span {q_lo:.0f}-{q_hi:.0f} kJ/mol; not all fall "
        f"below the {LIT_Q_LO:.0f}-{LIT_Q_HI:.0f} kJ/mol literature range "
        f"(reported as observed, not adjusted to match the expected range)."
    )
    caption = (
        f"Fig. S1. Grain-growth Arrhenius consistency plot: ln(D_XRD) vs. "
        f"1/T_K, grouped by precursor and fit by ordinary least squares "
        f"(Q_app = -slope x R / 1000, R = 8.314 J/mol/K). XRD covers 17 "
        f"complete condition triplets (51 samples), spanning the full Theta "
        f"range. Per-precursor fits: {fit_summary}. "
        f"IMPORTANT: this apparent activation energy is NOT a reported "
        f"quantitative kinetic result and must not be cited as one. The fit "
        f"is a single-slope diagnostic that does not include t_eff (the "
        f"ramp-time Arrhenius correction applied throughout this study's "
        f"normalized thermal exposure Theta and used in the PINN's "
        f"L_Arrhenius physical-consistency term); omitting that correction "
        f"systematically understates the true apparent activation energy. "
        f"{below_note} The true activation energy, to the extent it is "
        f"identifiable from this dataset at all, should be estimated jointly "
        f"with t_eff inside the full kinetics model, not read off this plot; "
        f"independently, Q has also been found not to be identifiable from "
        f"this dataset via a physics-informed neural network's learnable "
        f"activation-energy parameter. This figure is included only as a "
        f"sanity-check visualization confirming the expected sign and "
        f"monotonic trend of grain growth with thermal exposure, per "
        f"precursor group."
    )
    save_fig_journal(fig, NAME, caption, panel_labels=False)
    plt.close(fig)

    print(f"[figS1] verified Q_app range: {q_lo:.1f}-{q_hi:.1f} kJ/mol "
          f"(task spec quoted ~44-63 kJ/mol)")
    print(fit_df.to_string(index=False))
    return fig


if __name__ == "__main__":
    main()
