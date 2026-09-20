# -*- coding: utf-8 -*-
"""fig2_variance_decomposition.py -- Fig. 2 (journal submission set):
(a) restyled variance-decomposition stacked bar (10 targets, English labels,
journal styling) and (b) NEW precursor x process coupling-total summary bar.

Panel (a) is a direct restyle of scripts/figs/fig1_variance_stack.py: same
three data sources, same GROUPS/segment_defs/hatch-pattern logic, only the
on-figure text is translated to English and the styling switched to journal
conventions (apply_journal_style / add_panel_label / save_fig_journal). The
aggregation itself is NOT re-derived here -- _load_nano()/_load_partb_style()
below are a line-for-line port of the internal-review script's loading
functions, reading the same three already-computed csv files.

Panel (b) is new: it plots each target's total precursor-x-process coupling
(sum of eta2 over the three precursor:{T_C,beta,t_hold} interaction terms)
from manuscript/numbers/tables/table2_interactions.csv -- a frozen,
already-computed number (scripts/12_anova_interactions.py and
scripts/19_nano_layer_interaction_anova.py); no ANOVA is recomputed here.
A dashed reference line marks 88.8%, D_sec's precursor main-effect eta2 from
manuscript/numbers/tables/table1_main_effects.csv -- the single largest main
effect in the whole study, so the panel reads directly as "coupling is a
small fraction of what a single main effect can explain".

IMPORTANT (house rule, 2026-08-02 authorial decision): main-text figures and
captions must never explain that the 51-vs-81 nano-layer sample-count split
is due to a second measurement instrument. State "n=51" as a plain fact,
never "why". Do not write "instrument", "SmartLab", or "MiniFlex" anywhere
in this file's on-figure text or caption.
"""
import _bootstrap  # noqa: F401

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from _journal_style import (
    double_column_figsize,
    add_panel_label,
    save_fig_journal,
    dump_source_data,
    REPO_ROOT,
)

NAME = "fig2_variance_decomposition"

# ---------------------------------------------------------------------
# Panel (a) data sources -- identical to fig1_variance_stack.py
# ---------------------------------------------------------------------
NANO_INTERACTION_CSV = REPO_ROOT / "data" / "interim" / "nano_layer_interaction_anova.csv"
PARTB_CSV = REPO_ROOT / "data" / "interim" / "anova_interactions.csv"
SHAPE_CSV = REPO_ROOT / "data" / "interim" / "anova_interactions_shape_descriptors.csv"

# Panel (b) data sources -- frozen, already-computed manuscript numbers
TABLE1_CSV = REPO_ROOT / "manuscript" / "numbers" / "tables" / "table1_main_effects.csv"
TABLE2_CSV = REPO_ROOT / "manuscript" / "numbers" / "tables" / "table2_interactions.csv"

MAIN_TERMS = ["precursor", "T_C", "beta", "t_hold"]
INTERACTION_TERMS = ["precursor:T_C", "precursor:beta", "precursor:t_hold",
                     "T_C:beta", "T_C:t_hold", "beta:t_hold"]

# Same Okabe-Ito colour-blind-safe hex values as fig1_variance_stack.py's
# TERM_COLOR, reused literally for visual continuity across the figure set.
TERM_COLOR = {
    "precursor": "#E69F00",
    "T_C": "#0072B2",
    "beta": "#009E73",
    "t_hold": "#CC79A7",
    "interactions": "#999999",
    "residual_clean": "#D3D3D3",
}

SEGMENT_COLS = ["precursor", "T_C", "beta", "t_hold", "interactions", "residual_clean"]

# Group headers deliberately omit any mention of a second instrument -- see
# module docstring. "n=51, main-effects model" states a fact about the
# model design (relevant to the residual-comparability caveat below); it
# does not explain *why* the nano-layer sample count is 51.
GROUPS = [
    ("Nano layer\n(n=51, precursor x process interactions)", ["D_XRD", "lattice_c"]),
    ("Morphology/Packing\n(n=81, full interactions)", ["compaction_density", "D_sec", "circularity"]),
    ("Shape descriptors\n(n=81, full interactions)", ["D_f", "solidity", "elongation", "roughness"]),
]

# Targets covered by table2_interactions.csv, in the same left-to-right
# group order as panel (a) (nano group, then morphology/packing group);
# the 4 shape descriptors have no table2 entry (R4: convexity removed, and
# table2 was never built for the T7 shape-descriptor pipeline -- see
# scripts/22_build_manuscript_numbers.py:_TABLE1_TARGETS).
PANEL_B_GROUPS = [
    ("Nano layer (n=51)", ["D_XRD", "lattice_c"]),
    ("Morphology/Packing (n=81)", ["compaction_density", "D_sec", "circularity"]),
]


def _load_partb_style(path: Path) -> dict:
    df = pd.read_csv(path)
    out = {}
    for target, g in df.groupby("target"):
        g = g.set_index("term")
        row = {m: g.loc[m, "eta2"] for m in MAIN_TERMS}
        row["interactions"] = g.loc[INTERACTION_TERMS, "eta2"].sum()
        row["residual_clean"] = g.loc["Residual", "eta2"]
        out[target] = row
    return out


NANO_INTERACTION_TERMS = ["precursor:T_C", "precursor:beta", "precursor:t_hold"]


def _load_nano() -> dict:
    """Nano-layer panel (a) rows -- same source model as manuscript Table 1
    (scripts/19_nano_layer_interaction_anova.py's precursor x process
    interaction model), NOT scripts/18's main-effects-only model. This model
    only has 3 interaction terms (precursor:{T_C,beta,t_hold}), not the full
    6 two-way interactions the 81-sample targets have -- T_C:beta/T_C:t_hold/
    beta:t_hold are inestimable at n=51 (rank-deficient design matrix) and
    are NOT rows in this file; their variance is already inside Residual.
    """
    df = pd.read_csv(NANO_INTERACTION_CSV)
    out = {}
    for target, g in df.groupby("target"):
        g = g.set_index("term")
        row = {m: g.loc[m, "eta2"] for m in MAIN_TERMS}
        row["interactions"] = g.loc[NANO_INTERACTION_TERMS, "eta2"].sum()
        row["residual_clean"] = g.loc["Residual", "eta2"]
        out[target] = row
    return out


def load_panel_a() -> pd.DataFrame:
    data = {}
    data.update(_load_nano())
    data.update(_load_partb_style(PARTB_CSV))
    data.update(_load_partb_style(SHAPE_CSV))
    order = [t for _, ts in GROUPS for t in ts]
    return pd.DataFrame([data[t] for t in order], index=order)


def load_panel_b() -> pd.DataFrame:
    """Total precursor x process coupling per target, eta2 and omega2, from
    the frozen table2_interactions.csv (already-computed, not recomputed
    here). Row order matches PANEL_B_GROUPS."""
    df = pd.read_csv(TABLE2_CSV).set_index("target")
    order = [t for _, ts in PANEL_B_GROUPS for t in ts]
    return df.loc[order, ["total_eta2", "total_omega2"]]


def _verify_reference_line() -> float:
    """Independently verify the '88.8% main-effect reference' claim against
    table1_main_effects.csv rather than hardcoding the spec's quoted number.
    Returns the verified value (rounded to 1 decimal, matching Table 1's own
    published precision) and asserts it is in fact the largest main-effect
    eta2 across all targets/terms in the table."""
    t1 = pd.read_csv(TABLE1_CSV)
    term_cols = ["precursor", "T", "beta", "t"]
    melted = t1.melt(id_vars="target", value_vars=term_cols,
                     var_name="term", value_name="eta2")
    top = melted.loc[melted["eta2"].idxmax()]
    assert top["target"] == "D_sec" and top["term"] == "precursor", (
        f"expected D_sec/precursor to be the largest main effect, got "
        f"{top['target']}/{top['term']}"
    )
    return round(float(top["eta2"]), 1)


def _draw_panel_a(ax, df: pd.DataFrame) -> None:
    x = np.arange(len(df))
    bottom = np.zeros(len(df))

    segment_defs = [
        (SEGMENT_COLS[0], "Precursor", TERM_COLOR["precursor"], None),
        (SEGMENT_COLS[1], "T", TERM_COLOR["T_C"], None),
        (SEGMENT_COLS[2], "β", TERM_COLOR["beta"], None),
        (SEGMENT_COLS[3], "t", TERM_COLOR["t_hold"], None),
        (SEGMENT_COLS[4], "Interactions", TERM_COLOR["interactions"], None),
        (SEGMENT_COLS[5], "Residual", TERM_COLOR["residual_clean"], None),
    ]
    for col, label, color, hatch in segment_defs:
        vals = df[col].fillna(0.0).to_numpy() * 100
        ax.bar(x, vals, bottom=bottom, color=color, label=label,
              edgecolor="black", linewidth=0.4, hatch=hatch, width=0.62)
        bottom += vals

    # Type II sums of squares are not an orthogonal decomposition of total
    # variance on the unbalanced 51-sample nano-layer design (a known
    # property of Type II SS on this unbalanced design, not re-derived
    # here): the 6
    # fitted terms + Residual for D_XRD/lattice_c sum to ~88.7%/~92.75%, not
    # 100%. This is a property of the Type II method on this specific
    # unbalanced design, NOT a missing/excluded term -- the 3 inestimable
    # process x process interactions' variance is already fully inside the
    # Residual segment above. Draw the shortfall as an unfilled, outlined
    # cap (no colour, no legend entry, no text label) rather than a coloured
    # segment, so it reads as "intentionally left blank", not "excluded
    # interaction X". Only D_XRD/lattice_c have a non-negligible gap; the
    # other 7 (balanced 81-sample) targets sum to 100% already (verified
    # during planning, float noise only, well under this 1e-6 threshold).
    for i, gap in enumerate(100.0 - bottom):
        if gap > 1e-6:
            ax.bar(x[i], gap, bottom=bottom[i], facecolor="none",
                  edgecolor="black", linewidth=0.5, width=0.62, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(df.index, fontsize=6.8, rotation=25, ha="right")
    cursor = 0
    for gname, targets in GROUPS:
        n = len(targets)
        if cursor > 0:
            ax.axvline(cursor - 0.5, color="black", linewidth=0.7, linestyle=":")
        ax.text(cursor + n / 2 - 0.5, 104, gname, ha="center", va="bottom",
               fontsize=6.3, fontweight="bold")
        cursor += n

    ax.set_ylim(0, 118)
    ax.set_ylabel(r"Variance share $\eta^2$ (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=6.3,
             handlelength=1.3, columnspacing=1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def _draw_panel_b(ax_top, ax_bot, df: pd.DataFrame, ref_line: float) -> None:
    """Broken y-axis: ax_top shows the 88.8% main-effect reference line at
    its true scale, ax_bot shows all 5 coupling-total bars (all <=3.2%) at a
    zoomed-in linear scale. A broken axis was chosen over log scale because
    two of the omega2 totals (D_XRD, lattice_c) are exact zeros (see
    data/interim/nano_layer_interaction_anova.csv: every precursor:X
    interaction term for these two targets has omega2==0.0, not a rounding
    artifact) -- log(0) cannot be plotted, and a linear-with-break axis
    represents "zero" as a true zero-height bar rather than requiring an ad
    hoc floor value or an annotation standing in for a bar."""
    targets = list(df.index)
    x = np.arange(len(targets))
    width = 0.34

    eta2 = df["total_eta2"].to_numpy()
    omega2 = df["total_omega2"].to_numpy()

    for ax in (ax_top, ax_bot):
        ax.bar(x - width / 2, eta2, width=width, color=TERM_COLOR["interactions"],
              edgecolor="black", linewidth=0.4, label=r"$\eta^2$ (total coupling)")
        ax.bar(x + width / 2, omega2, width=width, color="#E8E8E8", hatch="//",
              edgecolor="black", linewidth=0.4, label=r"$\omega^2$ (total coupling)")

    # Reference line only needs to render in the top (high-value) strip.
    # Positioned with x in axes-fraction coordinates (not data coordinates)
    # and ha="right" so the text is guaranteed to stay inside the axes'
    # horizontal extent regardless of target count -- a data-coordinate,
    # ha="left" label previously pushed the tight savefig bbox past the
    # 190mm double-column width limit.
    ax_top.axhline(ref_line, color="#CC0000", linestyle="--", linewidth=0.9)
    trans = matplotlib.transforms.blended_transform_factory(ax_top.transAxes, ax_top.transData)
    ax_top.text(0.99, ref_line, f"{ref_line}% (D_sec precursor main effect, Table 1)",
               color="#CC0000", fontsize=6.0, va="bottom", ha="right", transform=trans)

    ax_top.set_ylim(80, 95)
    ax_bot.set_ylim(0, 4.0)

    # explicit zero-total-coupling omega2 annotation (true zero, not missing)
    for xi, om in zip(x, omega2):
        if om == 0.0:
            ax_bot.text(xi + width / 2, 0.06, "0.00", ha="center", va="bottom",
                       fontsize=5.6, color="#555555")

    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(labelbottom=False, bottom=False)
    ax_top.set_yticks([88.8])
    ax_top.set_yticklabels(["88.8"], fontsize=6.5)

    d = 0.5
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=8, linestyle="none",
                 color="k", mec="k", mew=0.8, clip_on=False)
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
    ax_bot.plot([0, 1], [1, 1], transform=ax_bot.transAxes, **kwargs)

    ax_bot.axvline(1.5, color="black", linewidth=0.7, linestyle=":")
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(targets, fontsize=6.8, rotation=20, ha="right")
    ax_bot.set_ylabel(r"Coupling total (%)", fontsize=7.2)
    ax_bot.yaxis.set_label_coords(-0.09, 0.75)
    ax_bot.legend(loc="upper right", fontsize=6.3, frameon=False, ncol=1,
                 handlelength=1.3)
    ax_bot.grid(axis="y", alpha=0.25)
    ax_bot.set_axisbelow(True)
    ax_top.grid(False)


def main():
    if not (NANO_INTERACTION_CSV.exists() and PARTB_CSV.exists() and SHAPE_CSV.exists()):
        print(f"[skip] {NAME}: underlying ANOVA csv files missing")
        return
    if not (TABLE1_CSV.exists() and TABLE2_CSV.exists()):
        print(f"[skip] {NAME}: manuscript/numbers/tables/table1|2 csv missing "
             f"(run scripts/22_build_manuscript_numbers.py)")
        return

    df_a = load_panel_a()
    df_b = load_panel_b()
    ref_line = _verify_reference_line()
    assert abs(ref_line - 88.8) < 1e-9, (
        f"independently verified main-effect reference is {ref_line}%, "
        f"not the 88.8% quoted in the task spec -- update the figure/caption, "
        f"do not hardcode the spec's number"
    )

    fig = plt.figure(figsize=double_column_figsize(200))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.5, 1.0], hspace=0.62,
                             left=0.10, right=0.97, top=0.93, bottom=0.10)
    ax_a = fig.add_subplot(outer[0])
    gs_b = outer[1].subgridspec(2, 1, height_ratios=[0.30, 1.0], hspace=0.06)
    ax_b_top = fig.add_subplot(gs_b[0])
    ax_b_bot = fig.add_subplot(gs_b[1], sharex=ax_b_top)

    _draw_panel_a(ax_a, df_a)
    _draw_panel_b(ax_b_top, ax_b_bot, df_b, ref_line)

    add_panel_label(ax_a, "(a)")
    add_panel_label(ax_b_top, "(b)")

    panel_a_rows = []
    for col in SEGMENT_COLS:
        for target in df_a.index:
            val = df_a.loc[target, col]
            panel_a_rows.append(dict(
                target=target, term=col,
                eta2_pct=(0.0 if pd.isna(val) else float(val) * 100),
                eta2_is_na=bool(pd.isna(val)),
            ))
    dump_source_data(
        NAME, "a", pd.DataFrame(panel_a_rows),
        note="Stacked-bar segment heights, eta2 x 100 (matches the rendered "
             "percent y-axis). eta2_is_na=True marks a term not applicable "
             "for that target (e.g. 'interactions' for the main-effects-"
             "only nano-layer targets D_XRD/lattice_c); such segments "
             "render as a zero-height filler bar via fillna(0.0), not a "
             "true zero.",
        source_files=[NANO_INTERACTION_CSV, PARTB_CSV, SHAPE_CSV],
    )

    panel_b_rows = df_b.reset_index().melt(
        id_vars="target", value_vars=["total_eta2", "total_omega2"],
        var_name="metric", value_name="value_pct",
    )
    dump_source_data(
        NAME, "b", panel_b_rows,
        note=f"Total precursor x process coupling per target, already in "
             f"percent units (matches the rendered y-axis; NOT a 0-1 "
             f"fraction). Reference line = {ref_line}% (D_sec precursor "
             f"main effect, Table 1, independently verified in this script "
             f"via _verify_reference_line()).",
        source_files=[TABLE2_CSV, TABLE1_CSV],
    )

    caption = (
        "Fig. 2. (a) ANOVA variance decomposition (eta^2 = SS_term/SS_total) "
        "for 9 targets across three analysis groups; each bar sums to 100% "
        "except D_XRD and lattice_c (see below). Nano layer (D_XRD, "
        "lattice_c; n=51) is from data/interim/nano_layer_interaction_anova.csv "
        "(scripts/19_nano_layer_interaction_anova.py -- the same model and "
        "file manuscript Table 1's D_XRD/lattice_c rows are computed from, "
        "not scripts/18's main-effects-only model). This design is an "
        "unbalanced 51-sample subset (17 of 27 conditions), so Type II sums "
        "of squares are not an orthogonal decomposition of total variance: "
        "the 6 fitted terms (precursor, T, beta, t, precursor x process "
        "interactions, Residual) sum to only 88.7% (D_XRD) / 92.8% "
        "(lattice_c) of variance, not 100% -- an unfilled, outlined cap "
        "(no colour, no legend entry) marks this shortfall on those two "
        "bars. This is a documented property of Type II SS on this "
        "unbalanced design, not "
        "a missing or excluded term: the three process x process "
        "interactions (T_C:beta, T_C:t_hold, beta:t_hold) are inestimable "
        "at n=51 (rank-deficient design matrix) and their variance is "
        "already fully inside the Residual segment shown. Morphology/Packing "
        "(compaction_density, D_sec, circularity; n=81) is from "
        "data/interim/anova_interactions.csv (scripts/12_anova_interactions.py; "
        "convexity is excluded because it duplicates the T7 shape-descriptor "
        "pipeline's solidity metric under a different implementation with "
        "materially different values). Shape "
        "descriptors (D_f, solidity, elongation, roughness; n=81) are from "
        "data/interim/anova_interactions_shape_descriptors.csv "
        "(scripts/17_shape_descriptor_full_analysis.py). (b) Total precursor x process coupling "
        "(sum of eta^2, and unbiased omega^2, over the three "
        "precursor:{T,beta,t} interaction terms) for the 5 targets with a "
        "frozen coupling estimate in manuscript/numbers/tables/"
        "table2_interactions.csv (scripts/12_anova_interactions.py for the "
        "n=81 targets, scripts/19_nano_layer_interaction_anova.py for the "
        "n=51 nano-layer targets); the 4 shape descriptors have no such "
        "estimate and are not shown. The dashed red line marks 88.8%, "
        "D_sec's precursor main-effect eta^2 from manuscript/numbers/tables/"
        f"table1_main_effects.csv, independently verified here to be the "
        "largest main effect of any target/term in that table. The y-axis "
        "is broken (linear scale, not log) between about 4% and 80% because "
        "every coupling total is <=3.2%, two orders of magnitude below the "
        "reference line, and the omega^2 coupling totals for D_XRD and "
        "lattice_c are exact zeros (every underlying precursor:X interaction "
        "term has omega2==0.0 in data/interim/nano_layer_interaction_anova.csv, "
        "not missing data or a rounding artifact) which a log scale cannot "
        "represent; the '0.00' labels make this explicit rather than leaving "
        "an ambiguous gap."
    )
    # This figure's content sits close to the 190mm double-column width
    # budget (2 stacked panels with a broken-axis sub-panel and long legend
    # labels); matplotlib's default savefig pad_inches (0.1in per side,
    # applied on top of the already-tight bbox by save_fig_journal's
    # bbox_inches="tight") was enough on its own to push the exported PNG a
    # few mm over 190mm even though the actual plotted content is under
    # budget. Tightening the pad locally (this script only, restored after
    # saving) keeps the export within spec without touching the shared
    # _journal_style.py infra used by other figures.
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
