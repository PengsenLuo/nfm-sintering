# -*- coding: utf-8 -*-
"""fig8_condition_residuals.py -- Fig. 8 (journal submission set, double
column / "通栏"); renumbered from Fig. 6 to Fig. 8 since the manuscript
moved the SEM-caliber PSD gradation figure into the main text as the new
Fig. 6, bumping this figure to Fig. 8 -- rename only, no
data/colour/marker change): condition-level residual of compaction_density
(precursor main effect removed) vs. normalized thermal exposure (Theta),
27 conditions x S/M/L.

Direct restyle of scripts/figs/fig7_condition_residuals.py: same data source
(data/interim/sieving_confound_diagnostic.csv, column
residual_precursor_adj_full27 -- read as-is, the residual computation itself
is NOT redefined here) and the same scatter-by-precursor-colour/-marker
pattern; only the on-figure text is translated to English and the styling
switched to journal conventions (this repo's house rule for scripts/figs/
journal/*.py, see e.g. fig3/fig5 in this directory).

"Sieved" marker check (verified by reading the source script rather than
assuming sieved markers were already removed):
scripts/figs/fig7_condition_residuals.py was read in full. Its only markers
are the three precursor scatter calls (`for prec in ["S", "M", "L"]:` at
line 41, colour = PRECURSOR_COLOR[prec], marker = PRECURSOR_MARKER[prec]).
There is no code path in that script referencing "sieved", a sieved subset,
or any distinct marker/annotation for sieved conditions -- the word "sieved"
does not appear anywhere in that file. The underlying CSV
(data/interim/sieving_confound_diagnostic.csv) does still carry a `sieved`
column and a second residual column (`residual_precursor_adj_unsieved22`,
the sieved-conditions-excluded variant used in an earlier sieving-sensitivity
diagnostic), but
fig7_condition_residuals.py reads only `residual_precursor_adj_full27` and
never touches either of those two columns. This confirms, by reading the
code rather than trusting the spec's docstring claim, that the 2026-08-02
decision to drop sieving-specific markers/annotations from this figure was
in fact carried out and there is nothing left to remove here. This script
inherits that same restriction (reads only residual_precursor_adj_full27,
does not plot `sieved` in any form).

Caption requirement: this figure stays in the main text rather than being
demoted to the SI, so the caption MUST state the result's status as a
discussion observation, not a main finding, using the sanctioned sentence
verbatim (not paraphrased) -- see the caption string in main() below.
"""
import _bootstrap  # noqa: F401

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from _journal_style import (
    double_column_figsize,
    save_fig_journal,
    dump_source_data,
    PRECURSOR_COLOR,
    PRECURSOR_MARKER,
)
from _style import REPO_ROOT

# NOTE: _style.THETA_NOTE (used by the non-journal fig7 this script restyles)
# is Chinese text; captions in this journal/ directory must be English-only
# (house rule, see fig3/fig5's own PRECURSOR_LABEL_EN for the same reasoning),
# so the Theta-calibration sentence below is a fresh English translation of
# that constant's content rather than an import of it.

NAME = "fig8_condition_residuals"

DIAG_CSV = REPO_ROOT / "data" / "interim" / "sieving_confound_diagnostic.csv"

# English precursor legend labels -- _style.py's PRECURSOR_LABEL contains
# Chinese text ("7:3混合"); on-figure text must be English-only (house rule
# for this journal/ directory, see fig3/fig5's own PRECURSOR_LABEL_EN).
PRECURSOR_LABEL_EN = {
    "S": "S (D50~3 um)",
    "M": "M (7:3 blend)",
    "L": "L (D50~11 um)",
}


def _load() -> pd.DataFrame:
    """Line-for-line port of fig7_condition_residuals.py's data-loading:
    read the diagnostic CSV, use residual_precursor_adj_full27 as-is."""
    df = pd.read_csv(DIAG_CSV)
    return df


def _draw(ax, df: pd.DataFrame, col: str) -> None:
    for prec in ["S", "M", "L"]:
        g = df[df["precursor"] == prec]
        color = PRECURSOR_COLOR[prec]
        marker = PRECURSOR_MARKER[prec]
        ax.scatter(g["Theta"], g[col], color=color, marker=marker, s=28,
                  edgecolor="black", linewidth=0.4,
                  label=f"{PRECURSOR_LABEL_EN[prec]} (n={len(g)})", zorder=3)

    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":", zorder=1)
    ax.set_xlabel(r"$\Theta$ (h)")
    ax.set_ylabel("Condition-level residual of compaction_density\n"
                  "(precursor main effect removed, g/cm$^3$)")
    ax.legend(loc="best", fontsize=6.5, handlelength=1.1, labelspacing=0.4)
    ax.grid(True, alpha=0.25)


def main():
    if not DIAG_CSV.exists():
        print(f"[skip] {NAME}: {DIAG_CSV} missing, run "
              f"scripts/13_sieving_confound.py first")
        return

    df = _load()
    col = "residual_precursor_adj_full27"
    n_samples = len(df)
    n_cond = df["condition_id"].nunique()
    assert n_samples == 81 and n_cond == 27, (
        f"expected 81 samples / 27 conditions, got {n_samples}/{n_cond}"
    )

    dump_source_data(
        NAME, "single", df[["sample_id", "condition_id", "precursor", "Theta", col]].copy(),
        note="Condition-level residual of compaction_density (precursor "
             "main effect removed) vs Theta, all 81 samples; column "
             "residual_precursor_adj_full27, used as-is -- the residual "
             "computation itself is not redefined in this figure script.",
        source_files=[DIAG_CSV],
    )

    fig = plt.figure(figsize=double_column_figsize(100))
    ax = fig.add_subplot(
        fig.add_gridspec(1, 1, left=0.09, right=0.98, top=0.96, bottom=0.13)[0]
    )
    _draw(ax, df, col)

    caption = (
        "Fig. 8. Condition-level residual of compaction density (precursor "
        "main effect removed) vs. normalized thermal exposure (Theta), all "
        "81 samples (27 conditions x S/M/L). Data: "
        "data/interim/sieving_confound_diagnostic.csv, column "
        "residual_precursor_adj_full27 (scripts/13_sieving_confound.py), "
        "used as-is -- this figure does not redefine the residual "
        "calculation. Colour and marker jointly code precursor identity "
        "(circle S, square M, triangle L); the dotted horizontal line marks "
        "zero residual. This condition-level reproducibility is presented "
        "as an observation for discussion rather than an established "
        "finding: the design contains no replicates, and temperature is "
        "collinear with experimental batch. Theta integration start "
        "temperature is fixed at 550 C (the calcination end temperature, "
        "identical for all samples; config.yaml."
        "thermal_exposure.integrate_from_C=550)."
    )
    # Measured at 191.1mm wide after bbox_inches="tight" cropping -- 0.6%
    # over the 190mm double-column limit. Same fix as
    # fig2_variance_decomposition.py/figS3_lattice_vs_theta.py/
    # figS5_mdagg_fusion_overlays.py for the identical issue: tighten the
    # savefig pad locally (this script only, restored after saving) rather
    # than touching the shared _journal_style.py infra used by other figures.
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
