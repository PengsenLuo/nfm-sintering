# -*- coding: utf-8 -*-
"""figS10_df_r2_distribution.py -- Fig. S10 (journal submission set,
supplementary information): distribution of the per-particle Richardson
log-log regression R-squared for the apparent contour fractal dimension
(D_f) fit, split by precursor (S/M/L), 500x SEM secondary-particle
population.

Pure visualization of an already-computed, already-reported diagnostic --
no new analysis is performed here. Data:
data/interim/sem_df_r2_diagnostic.csv (one row per candidate particle,
columns sample_id/precursor/filename/d_um/D_f/r2/n_scales), the frozen
output of a prior, already-completed audit round
(scripts/31_sem_metrology_audit.py; not re-run here). Rows with NaN R2 (a
particle whose contour was too small / had too few scale points to fit)
are excluded before plotting, matching that audit's own filtering.

The annotated scale range (1-16 px, ~0.223-3.573 um at the confirmed 500x
pixel size 0.2233 um/px) and overall median R2 (~0.872) are quoted from
that audit. This script also recomputes the
median locally as a sanity check; on disagreement beyond tolerance it
prints a warning and still uses the audit's number for the caption text.
"""
import _bootstrap  # noqa: F401

import pandas as pd
import matplotlib.pyplot as plt

from _journal_style import (
    single_column_figsize,
    save_fig_journal,
    dump_source_data,
    REPO_ROOT,
    PRECURSOR_COLOR,
)

NAME = "figS10_df_r2_distribution"
CSV = REPO_ROOT / "data" / "interim" / "sem_df_r2_diagnostic.csv"
REPORT_MD = REPO_ROOT / "reports" / "sem_metrology_audit.md"

PRECURSOR_LABEL_EN = {
    "S": "S (D50~3 um)",
    "M": "M (7:3 blend)",
    "L": "L (D50~11 um)",
}
PRECURSOR_ORDER = ["S", "M", "L"]

# Frozen numbers from the scripts/31_sem_metrology_audit.py audit (Section 3).
REPORT_MEDIAN_R2_ALL = 0.872
REPORT_N_VALID = 40413
REPORT_SCALE_PX = "1-16 px"
REPORT_SCALE_UM = "~0.223-3.573 um"
REPORT_PIXEL_SIZE_UM = 0.2233
TOL = 0.01


def main():
    if not CSV.exists():
        print(f"[skip] {NAME}: sem_df_r2_diagnostic.csv missing")
        return
    df = pd.read_csv(CSV, usecols=["sample_id", "precursor", "r2"])
    valid = df[df["r2"].notna()].copy()
    if valid.empty:
        print(f"[skip] {NAME}: no non-NaN r2 rows in sem_df_r2_diagnostic.csv")
        return

    n_valid = len(valid)
    median_r2_all = valid["r2"].median()
    if n_valid != REPORT_N_VALID:
        print(f"[WARNING] {NAME}: local n_valid={n_valid} vs report={REPORT_N_VALID}")
    if abs(median_r2_all - REPORT_MEDIAN_R2_ALL) > TOL:
        print(f"[WARNING] {NAME}: local median_r2_all={median_r2_all:.3f} vs "
              f"{REPORT_MD.name}={REPORT_MEDIAN_R2_ALL:.3f} -- using report "
              f"number for caption/annotation regardless.")

    dump_source_data(
        NAME, "single", valid[["sample_id", "precursor", "r2"]].copy(),
        note=f"Per-particle Richardson log-log fit R2, n={n_valid} "
             f"particles with a valid fit (of 83,366 candidates), split by "
             f"precursor. The violin body itself is a kernel-density "
             f"estimate of this same column (not separately verifiable "
             f"artist-by-artist); the black median bar per group IS "
             f"directly checked against this table's per-precursor median. "
             f"Overall median R2={REPORT_MEDIAN_R2_ALL:.3f}.",
        source_files=[CSV],
    )

    summary_rows = []
    for prec in PRECURSOR_ORDER:
        g = valid.loc[valid["precursor"] == prec, "r2"]
        if g.empty:
            continue
        summary_rows.append(dict(
            precursor=prec, n=len(g), median=g.median(), mean=g.mean(),
            std=g.std(), q25=g.quantile(0.25), q75=g.quantile(0.75),
            min=g.min(), max=g.max(),
        ))
    summary_rows.append(dict(
        precursor="all", n=len(valid), median=valid["r2"].median(),
        mean=valid["r2"].mean(), std=valid["r2"].std(),
        q25=valid["r2"].quantile(0.25), q75=valid["r2"].quantile(0.75),
        min=valid["r2"].min(), max=valid["r2"].max(),
    ))
    dump_source_data(
        NAME, "summary", pd.DataFrame(summary_rows),
        note="Per-precursor (plus an 'all' row) summary statistics of the "
             "same r2 column as the 'single' panel's 40,413-row granular "
             "file -- n/median/mean/std/q25/q75/min/max. This is the "
             "RECOMMENDED table for manuscript submission (small, directly "
             "answers 'what does this distribution look like' without the "
             "per-particle bulk); the granular 'single' file remains the "
             "audit trail this summary is computed from and is not "
             "superseded by it.",
        source_files=[CSV],
    )

    groups = [valid.loc[valid["precursor"] == p, "r2"].to_numpy()
             for p in PRECURSOR_ORDER if (valid["precursor"] == p).any()]
    present = [p for p in PRECURSOR_ORDER if (valid["precursor"] == p).any()]
    if not groups:
        print(f"[skip] {NAME}: no precursor-labelled rows after filtering")
        return

    fig = plt.figure(figsize=single_column_figsize(85))
    ax = fig.add_subplot(
        fig.add_gridspec(1, 1, left=0.15, right=0.97, top=0.97, bottom=0.13)[0]
    )

    positions = list(range(1, len(groups) + 1))
    parts = ax.violinplot(groups, positions=positions, showmedians=True,
                          showextrema=False, widths=0.75)
    for body, prec in zip(parts["bodies"], present):
        body.set_facecolor(PRECURSOR_COLOR[prec])
        body.set_edgecolor("black")
        body.set_linewidth(0.5)
        body.set_alpha(0.65)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.1)

    ax.set_xticks(positions)
    ax.set_xticklabels([PRECURSOR_LABEL_EN[p] for p in present], fontsize=6.2)
    ax.set_ylabel(r"Richardson log-log fit $R^2$")
    ax.set_ylim(0, 1.02)
    ax.axhline(REPORT_MEDIAN_R2_ALL, color="#666666", linewidth=0.8,
              linestyle=":", zorder=1)
    ax.text(0.03, REPORT_MEDIAN_R2_ALL + 0.02,
           f"overall median = {REPORT_MEDIAN_R2_ALL:.3f}",
           transform=ax.get_yaxis_transform(), fontsize=5.6, color="#555555")
    ax.grid(True, alpha=0.25, axis="y")

    ax.text(0.97, 0.05,
           f"scale range: {REPORT_SCALE_PX} ({REPORT_SCALE_UM})\n"
           f"n = {REPORT_N_VALID} particles with valid fit",
           transform=ax.transAxes, fontsize=5.4, color="#333333",
           ha="right", va="bottom",
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                    edgecolor="#cccccc", linewidth=0.5))

    caption = (
        "Fig. S10. Distribution of the per-particle Richardson (divider) "
        "log-log regression R-squared underlying the apparent contour "
        "fractal dimension D_f, split by precursor (500x SEM secondary-"
        "particle population). Violins: kernel density of R2 per "
        "precursor group; black bar: group median. Horizontal dotted line: "
        f"overall median R2 = {REPORT_MEDIAN_R2_ALL:.3f}. Data: "
        "data/interim/sem_df_r2_diagnostic.csv, the frozen output of a "
        "prior, already-completed audit round "
        "(scripts/31_sem_metrology_audit.py; not re-run here); rows with "
        "no valid fit (contour too short, or fewer than 4 usable scale "
        "points after the perimeter-based filter) are excluded, matching "
        "that audit's own filtering, leaving "
        f"n={REPORT_N_VALID} of 83,366 candidate particles (48.5%). Scale "
        f"points span a geometric series of {REPORT_SCALE_PX} "
        f"({REPORT_SCALE_UM} at the confirmed, full-population 500x pixel "
        f"size {REPORT_PIXEL_SIZE_UM} um/px), i.e. under one decade -- per "
        "that report, this range is set by the segmentation algorithm's "
        "pixel resolution and minimum-perimeter threshold rather than by "
        "any intrinsic particle self-similarity, and the R2 distribution "
        "shown here (well below 1 for most particles) indicates the "
        "multi-scale perimeter-length relationship is only an "
        "approximate, not a strict, power law -- supporting that report's "
        "recommendation to describe this quantity as an apparent contour "
        "fractal dimension used only for group/trend comparison, not as an "
        "absolute physical fractal dimension."
    )
    save_fig_journal(fig, NAME, caption, panel_labels=False)
    plt.close(fig)
    return fig


if __name__ == "__main__":
    main()
