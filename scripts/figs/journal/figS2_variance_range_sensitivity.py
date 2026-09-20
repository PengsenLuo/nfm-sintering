# -*- coding: utf-8 -*-
"""figS2_variance_range_sensitivity.py -- Fig. S2 (journal submission set,
supplementary information): variance-share (eta^2) sensitivity to design
range, restyled and materially RE-LAID-OUT from
scripts/15_variance_range_sensitivity.py (read-only source; not imported,
not modified).

Why this is a redesign, not a restyle
--------------------------------------
The source script's make_figure() draws a 4x1 stack of grouped bar charts
(4 targets x 5 series x 13 design subsets = 260 bars per panel, 1040 bars
total) on an 18x20 inch canvas (~46x51 cm). That canvas is roughly 10x this
figure set's 190mm x 230mm SI budget; simply shrinking the figsize tuple
while keeping 260 bars per panel would make every bar and every one of the
13 rotated x-tick labels illegible at 7-8pt final type -- shrinking the
canvas without redesigning the plot type was explicitly ruled out by the
task brief.

Design chosen instead: a 2x2 grid (one panel per target), each panel a LINE
plot (5 series x 13 x-positions, markers + connecting lines) instead of a
grouped bar chart. A line plot needs only 5 visually distinguishable
colour/marker series regardless of x-axis density, whereas a grouped bar
chart's legibility degrades directly with the number of bars per x-position;
this is the same category of "reduce the number of contending", not
"proportionally shrink" strategy this project's SEM h-maxima h-scale fix and
the density-matrix redesign both apply for the same reason (many independent
per-category quantities communicated more compactly as ordered
line-with-marker series than juxtaposed bars). x-axis subset labels are
shortened to compact codes (e.g. "-T950" = "drop T_C=950") to fit 13
categories in the ~190mm double-column width at 7pt; the full subset
description remains available in the underlying
data/interim/variance_range_sensitivity.csv (not reproduced verbatim in the
figure, consistent with an SI figure summarizing, not replacing, the CSV).

Data logic replicated (identical to 15_variance_range_sensitivity.py, not
re-derived): reads the already-computed long-format
data/interim/variance_range_sensitivity.csv (13 subset_id groups x 4 targets
x {precursor, T_C, beta, t_hold, coupling_total} terms' eta2 column); no
ANOVA is recomputed here. Subset ordering and grouping (baseline, then
T_C-drop x3, beta-drop x3, t_hold-drop x3, precursor-drop x3) is identical to
the source script's SUBSETS list order.
"""
import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _journal_style import (
    double_column_figsize,
    add_panel_label,
    save_fig_journal,
    dump_source_data,
    REPO_ROOT,
)

NAME = "figS2_variance_range_sensitivity"
CSV_PATH = REPO_ROOT / "data" / "interim" / "variance_range_sensitivity.csv"

TARGETS = ["compaction_density", "D_sec", "circularity", "solidity"]
SERIES_ORDER = ["precursor", "T_C", "beta", "t_hold", "coupling_total"]

# Identical set/order of subset_id to 15_variance_range_sensitivity.py's
# SUBSETS list (source script lines 56-87), with a compact English code for
# the x-axis in place of the source's long Chinese description strings.
SUBSET_ORDER = [
    ("baseline_81", "Base\n(81)", "baseline"),
    ("drop_T950", "-T950", "T_C"),
    ("drop_T900", "-T900", "T_C"),
    ("drop_T850", "-T850", "T_C"),
    ("drop_beta8", "-β8", "beta"),
    ("drop_beta5", "-β5", "beta"),
    ("drop_beta2", "-β2", "beta"),
    ("drop_thold20", "-t20", "t_hold"),
    ("drop_thold15", "-t15", "t_hold"),
    ("drop_thold10", "-t10", "t_hold"),
    ("drop_precursor_S", "-S", "precursor"),
    ("drop_precursor_M", "-M", "precursor"),
    ("drop_precursor_L", "-L", "precursor"),
]

SERIES_COLOR = {
    "precursor": "#2a78d6",
    "T_C": "#eb6834",
    "beta": "#1baf7a",
    "t_hold": "#eda100",
    "coupling_total": "#e87ba4",
}
SERIES_MARKER = {
    "precursor": "o", "T_C": "s", "beta": "^", "t_hold": "D", "coupling_total": "*",
}
SERIES_LABEL_EN = {
    "precursor": "Precursor", "T_C": "T", "beta": r"$\beta$",
    "t_hold": "t (hold)", "coupling_total": "Precursor x process coupling (total)",
}
TARGET_TITLE = {
    "compaction_density": "compaction_density",
    "D_sec": "D_sec",
    "circularity": "circularity",
    "solidity": "solidity",
}


def _load() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


def _draw_panel(ax, df: pd.DataFrame, target: str) -> None:
    sub = df[df["target"] == target]
    x = np.arange(len(SUBSET_ORDER))
    for term in SERIES_ORDER:
        vals = []
        for sid, _label, _grp in SUBSET_ORDER:
            row = sub[(sub["subset_id"] == sid) & (sub["term"] == term)]
            vals.append(float(row["eta2"].iloc[0]) * 100 if len(row) else np.nan)
        ax.plot(x, vals, marker=SERIES_MARKER[term], color=SERIES_COLOR[term],
                markersize=3.6, linewidth=1.0, markeredgecolor="black",
                markeredgewidth=0.3, label=SERIES_LABEL_EN[term], zorder=3)

    # Vertical separators between the 5 groups (baseline | T_C | beta |
    # t_hold | precursor), same grouping the source script's bar figure
    # marks with dashed vlines.
    prev_grp = None
    for i, (_sid, _label, grp) in enumerate(SUBSET_ORDER):
        if prev_grp is not None and grp != prev_grp:
            ax.axvline(i - 0.5, color="#BBBBBB", linewidth=0.7, linestyle=":", zorder=1)
        prev_grp = grp

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _sid, lbl, _grp in SUBSET_ORDER], fontsize=5.6)
    ax.set_ylabel(r"Variance share $\eta^2$ (%)", fontsize=7.0)
    ax.set_title(TARGET_TITLE[target], fontsize=7.5, loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=-2)


def _verify_amplitude_claim(df: pd.DataFrame) -> tuple[float, float, float, float, int]:
    """Independently recompute the source script's core §4b claim (main-effect
    amplitude vs. coupling-total amplitude across the 12 non-baseline
    sensitivity subsets) from the same CSV, rather than quoting a separately
    written report's numbers verbatim -- this keeps the caption's stated
    numbers tied to whatever is actually in the CSV at render time."""
    sens_ids = [sid for sid, _l, grp in SUBSET_ORDER if grp != "baseline"]
    main_terms = ["precursor", "T_C", "beta", "t_hold"]
    per_target_main_range = []
    per_target_coupling_range = []
    for target in TARGETS:
        sub = df[(df["target"] == target) & (df["subset_id"].isin(sens_ids))]
        main_ranges = []
        for term in main_terms:
            vals = sub[sub["term"] == term]["eta2"] * 100
            main_ranges.append(vals.max() - vals.min())
        per_target_main_range.append(max(main_ranges))
        coup = sub[sub["term"] == "coupling_total"]["eta2"] * 100
        per_target_coupling_range.append(coup.max() - coup.min())
    main_lo, main_hi = min(per_target_main_range), max(per_target_main_range)
    coup_lo, coup_hi = min(per_target_coupling_range), max(per_target_coupling_range)
    n_support = sum(1 for m, c in zip(per_target_main_range, per_target_coupling_range)
                    if c <= 1e-9 or m / c > 1)
    return main_lo, main_hi, coup_lo, coup_hi, n_support


def main():
    if not CSV_PATH.exists():
        print(f"[skip] {NAME}: {CSV_PATH} missing")
        return
    df = _load()
    for target in TARGETS:
        assert (df[df["target"] == target]["subset_id"].nunique()
                == len(SUBSET_ORDER)), f"{target}: subset count mismatch"

    main_lo, main_hi, coup_lo, coup_hi, n_support = _verify_amplitude_claim(df)

    panel_letter = dict(zip(TARGETS, "abcd"))
    for target in TARGETS:
        sub_t = df[df["target"] == target]
        panel_rows = []
        for term in SERIES_ORDER:
            for sid, label, grp in SUBSET_ORDER:
                row = sub_t[(sub_t["subset_id"] == sid) & (sub_t["term"] == term)]
                val = float(row["eta2"].iloc[0]) * 100 if len(row) else float("nan")
                panel_rows.append(dict(subset_id=sid, subset_label=label,
                                       subset_group=grp, term=term, eta2_pct=val))
        dump_source_data(
            NAME, panel_letter[target], pd.DataFrame(panel_rows),
            note=f"target={target}; eta2 x 100 (matches the rendered "
                 f"percent y-axis), 13 design subsets x 5 terms "
                 f"(precursor/T_C/beta/t_hold main effects + total "
                 f"precursor x process coupling).",
            source_files=[CSV_PATH],
        )

    fig = plt.figure(figsize=double_column_figsize(160))
    gs = fig.add_gridspec(2, 2, wspace=0.30, hspace=0.60,
                          left=0.075, right=0.98, top=0.85, bottom=0.10)
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    axes = []
    for i, target in enumerate(TARGETS):
        r, c = divmod(i, 2)
        ax = fig.add_subplot(gs[r, c])
        _draw_panel(ax, df, target)
        add_panel_label(ax, panel_labels[i])
        axes.append(ax)

    # Shared figure-level legend above all four panels -- an in-panel legend
    # collides with data in every panel (all four targets have a data series
    # near the top of the y-range across most of the x-axis), so the legend
    # is drawn once, outside the plotted area, using panel (a)'s handles.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=6.2,
              handlelength=1.3, columnspacing=1.1, labelspacing=0.35,
              bbox_to_anchor=(0.5, 0.995), frameon=False)

    caption = (
        "Fig. S2. Sensitivity of ANOVA variance shares (eta^2 x 100%) to "
        "design-range truncation, for four targets with a full "
        "two-way-interaction model at n=81 (compaction_density, D_sec, "
        "circularity, solidity; restyled and re-laid-out from "
        "scripts/15_variance_range_sensitivity.py's data/interim/"
        "variance_range_sensitivity.csv, read as-is, not recomputed by this "
        "figure script). convexity is deliberately excluded here for the "
        "same reason the main-text Fig. 2 excludes it: it duplicates "
        "solidity's definition under a different implementation with "
        "materially different values. solidity "
        "itself is from scripts/17_shape_descriptor_full_analysis.py's "
        "data/interim/sem_shape_descriptors_summary.csv, merged into "
        "scripts/15_variance_range_sensitivity.py's existing 13-design-"
        "subset sensitivity machinery for this figure (the underlying ANOVA "
        "engine, nfm.stats.anova_interactions.run_anova_for_target, is "
        "unchanged). (a)-(d) each panel: 5 line series (precursor, T, "
        "beta, t-hold main effects, and total precursor x process coupling "
        "= sum of eta^2 over the three precursor:{T,beta,t} interaction "
        "terms) evaluated on 13 design subsets -- the full 81-sample "
        "baseline, 9 subsets each dropping one level of T/beta/t_hold "
        "(27->18 conditions, 81->54 samples), and 3 subsets each dropping "
        "one precursor (81->54 samples, 2 precursors remaining). Dotted "
        "vertical lines separate the five subset groups (baseline | drop-T "
        "| drop-beta | drop-t_hold | drop-precursor). This addresses a "
        "reviewer concern that the main-text variance decomposition "
        "(Fig. 2a; precursor main-effect eta^2 up to 88.8%, precursor x "
        "process coupling always <=3.3%) could be an artefact of the "
        "specific design ranges chosen (precursor D50 spans a 3.4x range, "
        "temperature only a 100 K range) rather than a property of the "
        "effects themselves. Across the 12 non-baseline subsets, the "
        f"largest main-effect eta^2 amplitude (max-min across subsets, per "
        f"target) ranges {main_lo:.1f}-{main_hi:.1f} percentage points, "
        f"while the coupling-total amplitude ranges {coup_lo:.1f}-"
        f"{coup_hi:.1f} percentage points ({n_support}/4 targets showing "
        "coupling amplitude smaller than the largest main-effect "
        "amplitude) -- consistent with the full derivation and per-target "
        "breakdown produced by scripts/15_variance_range_sensitivity.py "
        "(section 4 of its report), which this figure visualizes but does "
        "not re-derive."
    )
    save_fig_journal(fig, NAME, caption)
    plt.close(fig)
    return fig


if __name__ == "__main__":
    main()
