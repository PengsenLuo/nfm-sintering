# -*- coding: utf-8 -*-
"""figS6_theta_qref_sensitivity.py -- Fig. S6 (journal submission set,
supplementary information, NEW figure -- the only genuinely new-to-this-
repo plot in Task 8): Spearman(D_XRD, Theta) and Spearman(M_D_agg, Theta) as
a function of the reference activation energy Q_ref used to normalize
thermal exposure (Theta), Q_ref in [150, 250] kJ/mol.

This is pure visualization of already-computed, already-reported results --
no new analysis is performed here. Data source: the frozen R5 sensitivity-
sweep output data/interim/theta_qref_sensitivity_dxrd.csv and
data/interim/theta_qref_sensitivity_mdagg.csv (columns q_ref_kjmol,
spearman_rho, p), produced by a prior, already-completed round
(scripts/26_theta_qref_sensitivity.py) and force-added to git as the frozen
record in commit 5a0136e despite data/interim/ being otherwise .gitignore'd
-- confirmed still present and git-tracked before reading
(`git ls-files data/interim/theta_qref_sensitivity*`, see task report).
scripts/26_theta_qref_sensitivity.py itself is NOT run or re-run here.

The caption states the actual robustness verdict from
reports/theta_qref_sensitivity.md (a prior, already-completed round's
report): both correlations are robust (same sign, p<0.05) across the full
Q_ref sweep -- that verdict is quoted/restated here, not re-derived or
second-guessed.
"""
import _bootstrap  # noqa: F401

import pandas as pd
import matplotlib.pyplot as plt

from _journal_style import (
    single_column_figsize,
    save_fig_journal,
    dump_source_data,
    REPO_ROOT,
)

NAME = "figS6_theta_qref_sensitivity"

DXRD_CSV = REPO_ROOT / "data" / "interim" / "theta_qref_sensitivity_dxrd.csv"
MDAGG_CSV = REPO_ROOT / "data" / "interim" / "theta_qref_sensitivity_mdagg.csv"
REPORT_MD = REPO_ROOT / "reports" / "theta_qref_sensitivity.md"

Q_REF_PRODUCTION = 200.0  # kJ/mol, this study's fixed production Q_ref
ALPHA = 0.05


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    dxrd = pd.read_csv(DXRD_CSV).sort_values("q_ref_kjmol")
    mdagg = pd.read_csv(MDAGG_CSV).sort_values("q_ref_kjmol")
    return dxrd, mdagg


def main():
    if not (DXRD_CSV.exists() and MDAGG_CSV.exists()):
        print(f"[skip] {NAME}: theta_qref_sensitivity_{{dxrd,mdagg}}.csv missing")
        return
    dxrd, mdagg = _load()

    # Confirm (not assume) the sign/significance stability the report
    # verdict quotes below.
    dxrd_sign_stable = (dxrd["spearman_rho"] > 0).all() or (dxrd["spearman_rho"] < 0).all()
    mdagg_sign_stable = (mdagg["spearman_rho"] > 0).all() or (mdagg["spearman_rho"] < 0).all()
    dxrd_sig_stable = (dxrd["p"] < ALPHA).all()
    mdagg_sig_stable = (mdagg["p"] < ALPHA).all()
    assert dxrd_sign_stable and mdagg_sign_stable, (
        "sign of Spearman(D_XRD/M_D_agg, Theta) is not stable across the "
        "Q_ref sweep -- contradicts reports/theta_qref_sensitivity.md; do "
        "not overwrite the caption's robustness verdict without checking "
        "that report first"
    )
    assert dxrd_sig_stable and mdagg_sig_stable, (
        "significance (p<0.05) is not stable across the Q_ref sweep -- "
        "contradicts reports/theta_qref_sensitivity.md"
    )

    panel_df = pd.concat([
        dxrd.assign(series="D_XRD")[["series", "q_ref_kjmol", "spearman_rho", "p"]],
        mdagg.assign(series="M_D_agg")[["series", "q_ref_kjmol", "spearman_rho", "p"]],
    ], ignore_index=True)
    dump_source_data(
        NAME, "single", panel_df,
        note="Spearman(D_XRD,Theta) (n=51) and Spearman(M_D_agg,Theta) "
             "(n=27) vs Q_ref, frozen output of "
             "scripts/26_theta_qref_sensitivity.py (not re-run here).",
        source_files=[DXRD_CSV, MDAGG_CSV],
    )

    fig = plt.figure(figsize=single_column_figsize(78))
    ax = fig.add_subplot(
        fig.add_gridspec(1, 1, left=0.19, right=0.96, top=0.97, bottom=0.15)[0]
    )

    ax.axvline(Q_REF_PRODUCTION, color="#999999", linewidth=0.8, linestyle=":", zorder=1)
    ax.text(Q_REF_PRODUCTION, 0.99, "production Q_ref=200", fontsize=5.4,
           color="#666666", ha="center", va="top", transform=ax.get_xaxis_transform())

    for df, color, marker, label in [
        (dxrd, "#2a78d6", "o", r"Spearman($D_{XRD}$, $\Theta$), n=51"),
        (mdagg, "#e87ba4", "s", r"Spearman($M_{D,agg}$, $\Theta$), n=27"),
    ]:
        ax.plot(df["q_ref_kjmol"], df["spearman_rho"], marker=marker, color=color,
                markersize=4.2, linewidth=1.1, markeredgecolor="black",
                markeredgewidth=0.3, label=label, zorder=3)
        # Every point in both sweeps is significant (p<0.05, verified by the
        # assert above) -- mark each with "*" rather than mixing significant
        # and non-significant annotation styles that would not apply here.
        for _, r in df.iterrows():
            ax.annotate("*", (r["q_ref_kjmol"], r["spearman_rho"]),
                       textcoords="offset points", xytext=(0, 5), fontsize=7,
                       ha="center", color=color)

    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r"$Q_{ref}$ (kJ/mol)")
    ax.set_ylabel(r"Spearman $\rho$")
    ax.legend(loc="lower right", fontsize=5.6, handlelength=1.2, labelspacing=0.4)
    ax.grid(True, alpha=0.25)
    ax.text(0.03, 0.06, "* p < 0.05 (all points, both series)", transform=ax.transAxes,
           fontsize=5.4, color="#444444", va="bottom", ha="left")

    dxrd_rho_lo, dxrd_rho_hi = dxrd["spearman_rho"].min(), dxrd["spearman_rho"].max()
    mdagg_rho_lo, mdagg_rho_hi = mdagg["spearman_rho"].min(), mdagg["spearman_rho"].max()
    dxrd_p_max = dxrd["p"].max()
    mdagg_p_max = mdagg["p"].max()

    q_values_str = "/".join(str(int(q)) for q in sorted(dxrd["q_ref_kjmol"]))
    n_total_points = len(dxrd) + len(mdagg)

    caption = (
        f"Fig. S6. Sensitivity of Theta's two headline correlations to the "
        f"reference activation energy Q_ref used to normalize the thermal-"
        f"exposure integral, swept over Q_ref in [150, 250] kJ/mol ({len(dxrd)} "
        f"values: {q_values_str} kJ/mol). Data: "
        f"data/interim/theta_qref_sensitivity_dxrd.csv and "
        f"_mdagg.csv, the frozen output of a prior, already-completed "
        f"robustness-check round (scripts/26_theta_qref_sensitivity.py; "
        f"not re-run here). Blue circles: Spearman(D_XRD, Theta), nano-"
        f"layer-scoped n=51 (17 complete condition triplets); rho ranges "
        f"[{dxrd_rho_lo:+.2f}, {dxrd_rho_hi:+.2f}], p<{dxrd_p_max:.2e} "
        f"across the full sweep. Magenta squares: Spearman(M_D_agg, "
        f"Theta), n=27 conditions; rho ranges [{mdagg_rho_lo:+.2f}, "
        f"{mdagg_rho_hi:+.2f}], p<{mdagg_p_max:.3f} across the full sweep. "
        f"All {n_total_points} points (both series) are significant at p<0.05 (marked "
        f"'*'); the dotted vertical line marks this study's fixed "
        f"production value, Q_ref=200 kJ/mol. Verdict (from "
        f"reports/theta_qref_sensitivity.md, restated here, not "
        f"re-derived): both correlations are robust -- same sign and "
        f"p<0.05 throughout -- across the entire Q_ref sweep; that report "
        f"additionally finds the 27-condition Theta ranking itself is "
        f"near-invariant to Q_ref (adjacent-Q_ref-pair Spearman rho >= "
        f"0.979) and that this study's main-text ANOVA (Table 1, "
        f"compaction_density/D_XRD main effects) uses the raw categorical "
        f"factors T_C/beta/t_hold/precursor directly and never regresses "
        f"on Theta, so it is structurally independent of the Q_ref choice "
        f"regardless of this sweep's outcome."
    )
    save_fig_journal(fig, NAME, caption, panel_labels=False)
    plt.close(fig)
    return fig


if __name__ == "__main__":
    main()
