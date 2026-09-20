# -*- coding: utf-8 -*-
"""fig5_memory_index_robustness.py -- Fig. 5 (journal submission set, single-
column layout): the agglomerate
memory index M_D_agg vs Theta (27 conditions), with a per-condition
sensitivity ("amplitude") band overlaid from a 9-parameter segmentation
sweep -- promoted from the SI figure fig_S_mdagg_robustness (produced by
scripts/02j_mdagg_fusion_check.py's make_robustness_figure()) to a main-text
figure.

Why this script exists instead of reusing 02j directly (02j is NOT imported
or edited -- only its data-loading is replicated here): the spec's ask for
Fig. 5 is narrower and differently shaped than what 02j's existing 3-panel
SI figure draws.

  Spec text (Fig. 5 section):
    "M_D,agg vs Theta (27 conditions), overlaid with a value band from the
    9-group segmentation-parameter sensitivity sweep (adapted from
    fig_S_mdagg_robustness, promoted from SI to main text)."
    Layout is explicitly marked single-column (90 mm), not double-column.

  02j's existing fig_S_mdagg_robustness (make_robustness_figure(), lines
  ~342-396 of that script) is a 3-panel *double*-column figure: (1) Spearman
  rho vs percentile with slope held at default, (2) Spearman rho vs slope
  with percentile held at default, (3) a bar chart of rho for a *different*,
  legacy fixed-h segmentation family (explicitly labelled in that script as
  "cross-reference ONLY -- not production sensitivity"). None of those three
  panels is actually "M_D_agg vs Theta" -- they all plot Spearman rho (a
  single summary number per parameter combination) on the y-axis, never the
  27 raw (Theta, M_D_agg) points the spec asks for. So the core scatter the
  spec wants does not exist as a standalone panel anywhere upstream and is
  built fresh here, from data/processed/master_table.csv (M_D_agg is a
  per-condition column there, repeated on all 3 precursor rows of a
  condition; drop_duplicates("condition_id") recovers the 27 unique values
  -- the same pattern already used and verified in
  scripts/26_theta_qref_sensitivity.py's R5 sensitivity check, line 68).

  Panel-count decision: ONE single-column panel, not two, and not a port of
  all 3 SI panels. Reasons:
    1. The spec explicitly marks this figure single-column (90 mm). A
       2-panel side-by-side split would leave ~43 mm per panel -- too
       cramped for a labelled scatter plus legend at 7-8 pt journal type.
    2. The spec's own verb is "overlaid" (叠加), not "alongside" / "next to"
       -- it asks for the sensitivity information layered onto the same
       scatter, not a second panel.
    3. The legacy fixed-h bar panel (original panel 3) is explicitly a
       different algorithm family and is already labelled in 02j as
       "not production sensitivity" -- it is a defensive SI cross-check,
       not part of the promoted main-text claim, and stays in the SI figure
       (untouched; this script does not regenerate or replace
       reports/figures/fig_S_mdagg_robustness.png).
  Concretely, "overlaid value band" is implemented as an asymmetric vertical
  error bar on each of the 27 default-pipeline points, spanning the
  min-max range of that condition's M_D_agg recomputed under all 9
  segmentation-parameter combinations from the sensitivity sweep (1 default
  + 4 non-default percentile values + 4 non-default slope values -- the same
  9-combination grid 02j's --sweep step ran). A per-point error bar was
  chosen over a continuous fill_between ribbon because the 27 conditions are
  discrete experimental points at irregularly spaced Theta, not samples of a
  continuous function; connecting successive conditions' min/max with a
  filled ribbon would visually imply a continuum between conditions that
  does not exist in the design. The corresponding Spearman rho range across
  the same 9 combinations is stated as figure text (computed here, not
  hardcoded) so the reader gets the summary-statistic view the original SI
  bar/line panels conveyed, without a second panel.

Data sources (Step 1 of the task -- confirmed by reading
scripts/02j_mdagg_fusion_check.py end-to-end, not assumed):
  - data/processed/master_table.csv (MASTER in 02j) -- 81-row table;
    drop_duplicates("condition_id") gives the 27 condition-level M_D_agg
    values plotted as the main scatter. Verified exact byte-for-byte match
    (max abs diff = 0.0) against recomputing M_D_agg from the sweep CSV's
    "default" combo below -- the two sources are the same production
    pipeline output, not independently-drifted numbers.
  - data/interim/mdagg_sensitivity_sweep.csv (SWEEP_CSV in 02j, written by
    02j's run_sweep()) -- 729 rows = 9 combo_id groups (see combos_to_run()
    in 02j, lines 140-151) x 81 samples, columns sample_id/D_sec/combo_id/
    percentile/slope. This is the sole input 02j's sweep_correlations()
    (called by make_robustness_figure()'s sweep_corr_df argument) reads via
    pd.read_csv(SWEEP_CSV) at line 228 of that script. Confirmed present on
    disk (40 KB, last written 2026-07-30).
  NOT used here: data/interim/sem_sec_sweep/metrics.csv (LEGACY_METRICS in
  02j, feeds legacy_corr_df / the original panel 3) -- deliberately excluded
  per the panel-count decision above; it remains SI-only content.
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
    TEMP_COLOR,
    REPO_ROOT,
)
from _style import MASTER_TABLE
from nfm.config import load_config, load_precursor_dvalues

NAME = "fig5_memory_index_robustness"

SWEEP_CSV = REPO_ROOT / "data" / "interim" / "mdagg_sensitivity_sweep.csv"

# The sweep CSV's combo_id groups already encode the 9-combination grid
# (1 default + 4 non-default percentile values + 4 non-default slope
# values, see 02j_mdagg_fusion_check.py's combos_to_run()); this script
# reads that grouping directly from the CSV rather than re-declaring the
# grid values, so there is nothing to keep in sync here if that grid ever
# changes upstream.


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def _load_condition_level() -> pd.DataFrame:
    """27-condition default-pipeline (Theta, M_D_agg, T_C) frame -- a fresh
    read of master_table.csv, not reused from 02j (which does not expose
    this as a standalone frame; its compute_mdagg_theta() derives M_D_agg
    from a raw D_sec map + a fresh S/L pairing loop rather than reading the
    already-aggregated master_table.csv column)."""
    df = pd.read_csv(MASTER_TABLE)
    cond = df.drop_duplicates("condition_id").copy()
    assert len(cond) == 27, f"expected 27 conditions, got {len(cond)}"
    assert cond["M_D_agg"].notna().all(), "unexpected NaN in M_D_agg (27-condition frame)"
    return cond[["condition_id", "Theta", "M_D_agg", "T_C"]].sort_values("Theta")


def _mdagg_per_condition(meta: pd.DataFrame, d50: dict, d_sec_map: pd.Series) -> pd.DataFrame:
    """Recompute the 27 per-condition M_D_agg values for one segmentation
    parameter combination. Deliberately a fresh, self-contained
    implementation (not an import of 02j_mdagg_fusion_check.compute_mdagg_theta),
    replicating 02j's data-loading rather than its code: same arithmetic --
    M_D_agg = (D_sec_L - D_sec_S) / (precursor_D50_L - precursor_D50_S),
    paired within each condition_id -- matching this project's schema
    definition of M_D_agg."""
    df = meta.copy()
    df["D_sec"] = df["sample_id"].map(d_sec_map)
    den = d50["L"] - d50["S"]
    rows = []
    for cid, g in df.groupby("condition_id"):
        s = g[g.precursor == "S"]
        l = g[g.precursor == "L"]
        if len(s) != 1 or len(l) != 1:
            rows.append((cid, np.nan))
            continue
        d_s, d_l = s["D_sec"].iloc[0], l["D_sec"].iloc[0]
        m = (d_l - d_s) / den if pd.notna(d_s) and pd.notna(d_l) else np.nan
        rows.append((cid, m))
    return pd.DataFrame(rows, columns=["condition_id", "M_D_agg"]).set_index("condition_id")


def _load_sensitivity_band(cond: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (band_df, combo_rho_df).

    band_df: index=condition_id, columns [combo_min, combo_max] -- the
    per-condition M_D_agg range across all 9 segmentation-parameter
    combinations (the "amplitude/sensitivity band" the spec asks to overlay).

    combo_rho_df: one row per combo_id with Spearman rho(M_D_agg, Theta)
    across that combo's 27 conditions -- used only to state the rho range
    as figure text, mirroring what the original SI panels 1/2 plotted as
    curves.
    """
    cfg = load_config()
    d50 = load_precursor_dvalues(cfg)
    mt = pd.read_csv(MASTER_TABLE)
    meta = mt[["sample_id", "condition_id", "precursor", "Theta"]]

    sweep = pd.read_csv(SWEEP_CSV)
    n_combos_expected = 1 + 4 + 4
    n_combos_found = sweep["combo_id"].nunique()
    assert n_combos_found == n_combos_expected, (
        f"expected {n_combos_expected} sweep combinations, found {n_combos_found}"
    )

    per_combo = {}
    rho_rows = []
    for combo_id, g in sweep.groupby("combo_id"):
        d_sec_map = g.set_index("sample_id")["D_sec"]
        out = _mdagg_per_condition(meta, d50, d_sec_map)
        per_combo[combo_id] = out["M_D_agg"]
        valid = out.dropna(subset=["M_D_agg"]).join(
            cond.set_index("condition_id")["Theta"], how="inner"
        )
        if len(valid) >= 4:
            rho, p = stats.spearmanr(valid["Theta"], valid["M_D_agg"])
        else:
            rho, p = np.nan, np.nan
        rho_rows.append(dict(combo_id=combo_id, rho=rho, p=p, n=len(valid)))

    piv = pd.DataFrame(per_combo)
    assert not piv.isna().any().any(), "NaN in per-condition sensitivity sweep pivot"
    band_df = pd.DataFrame({"combo_min": piv.min(axis=1), "combo_max": piv.max(axis=1)})

    # Sanity check: the "default" combo must reproduce master_table.csv's
    # M_D_agg exactly -- same production segmentation parameters, same
    # arithmetic. A mismatch would mean the two sources have silently
    # diverged (e.g. master_table.csv regenerated with different config).
    default_check = piv["default"].reindex(cond.set_index("condition_id").index)
    max_diff = (default_check.values - cond.set_index("condition_id")["M_D_agg"].values)
    max_diff = np.abs(max_diff).max()
    assert max_diff < 1e-9, (
        f"sweep 'default' combo does not match master_table.csv M_D_agg "
        f"(max abs diff={max_diff:.6g}) -- data sources have diverged"
    )

    combo_rho_df = pd.DataFrame(rho_rows)
    return band_df, combo_rho_df


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
def _draw(ax, cond: pd.DataFrame, band_df: pd.DataFrame, combo_rho_df: pd.DataFrame,
          rho0: float, p0: float) -> None:
    band = band_df.reindex(cond["condition_id"])
    yerr_lo = (cond["M_D_agg"].values - band["combo_min"].values)
    yerr_hi = (band["combo_max"].values - cond["M_D_agg"].values)
    assert (yerr_lo >= -1e-9).all() and (yerr_hi >= -1e-9).all(), (
        "default M_D_agg falls outside its own 9-combo sensitivity range"
    )
    yerr_lo = np.clip(yerr_lo, 0, None)
    yerr_hi = np.clip(yerr_hi, 0, None)

    # Sensitivity ("amplitude") band -- per-condition asymmetric error bars,
    # drawn first / behind so the coloured scatter markers sit on top.
    ax.errorbar(cond["Theta"], cond["M_D_agg"], yerr=[yerr_lo, yerr_hi],
                fmt="none", ecolor="#9AA0A6", elinewidth=1.0, capsize=2.0,
                capthick=1.0, alpha=0.75, zorder=2)

    # OLS trend line (sign indicator only, per this figure set's convention
    # -- see fig4's docstring for the same rationale).
    b, a = np.polyfit(cond["Theta"], cond["M_D_agg"], 1)
    x_line = np.linspace(cond["Theta"].min(), cond["Theta"].max(), 50)
    ax.plot(x_line, a + b * x_line, color="#333333", linestyle="--",
            linewidth=1.0, alpha=0.85, zorder=3)

    # Default-pipeline scatter, coloured by sintering temperature (same
    # TEMP_COLOR double-encoding convention as fig3 panel (a)).
    for t_c, color in TEMP_COLOR.items():
        g = cond[cond["T_C"] == t_c]
        if g.empty:
            continue
        ax.scatter(g["Theta"], g["M_D_agg"], color=color, marker="o", s=20,
                  edgecolor="black", linewidth=0.4, zorder=4,
                  label=f"{t_c:.0f} C")

    ax.set_xlabel(r"$\Theta$ (h)")
    ax.set_ylabel(r"$M_{D,\mathrm{agg}}$")
    ax.legend(loc="lower right", fontsize=6.2, handlelength=1.1,
              labelspacing=0.35, title="Sintering T", title_fontsize=6.2)
    ax.grid(True, alpha=0.25)

    rho_lo, rho_hi = combo_rho_df["rho"].min(), combo_rho_df["rho"].max()
    p_max = combo_rho_df["p"].max()
    text = (f"$\\rho$={rho0:+.2f} (p={p0:.4f}), n=27\n"
           f"9-combo sweep: $\\rho\\in$[{rho_lo:+.2f},{rho_hi:+.2f}],\n"
           f"p<{p_max:.3f} (all combos)\n"
           f"error bars: range over 9 combos")
    ax.text(0.03, 0.97, text, transform=ax.transAxes, fontsize=6.0,
           va="top", ha="left",
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                     edgecolor="#CCCCCC", alpha=0.9))


def main():
    if not MASTER_TABLE.exists():
        print(f"[skip] {NAME}: {MASTER_TABLE} missing")
        return
    if not SWEEP_CSV.exists():
        print(f"[skip] {NAME}: {SWEEP_CSV} missing")
        return

    cond = _load_condition_level()
    rho0, p0 = stats.spearmanr(cond["Theta"], cond["M_D_agg"])
    band_df, combo_rho_df = _load_sensitivity_band(cond)

    assert (combo_rho_df["rho"] > 0).all(), (
        "sign of Spearman(M_D_agg, Theta) is not preserved across all 9 "
        "sensitivity-sweep combinations"
    )
    assert (combo_rho_df["p"] < 0.05).all(), (
        "significance (p<0.05) is not preserved across all 9 "
        "sensitivity-sweep combinations"
    )

    panel_df = cond.merge(band_df, left_on="condition_id", right_index=True, how="left")
    dump_source_data(
        NAME, "single", panel_df,
        note=f"27 conditions: default-pipeline M_D_agg vs Theta, plus its "
             f"9-combo segmentation-parameter sensitivity range "
             f"(combo_min/combo_max, the overlaid error bars). Default "
             f"Spearman rho={rho0:+.3f} p={p0:.4f}; across the 9-combo "
             f"sweep rho in "
             f"[{combo_rho_df['rho'].min():+.3f},{combo_rho_df['rho'].max():+.3f}], "
             f"p<{combo_rho_df['p'].max():.4f}.",
        source_files=[MASTER_TABLE, SWEEP_CSV],
    )

    fig = plt.figure(figsize=single_column_figsize(82))
    ax = fig.add_subplot(
        fig.add_gridspec(1, 1, left=0.16, right=0.97, top=0.97, bottom=0.13)[0]
    )
    _draw(ax, cond, band_df, combo_rho_df, rho0, p0)

    rho_lo, rho_hi = combo_rho_df["rho"].min(), combo_rho_df["rho"].max()
    p_max = combo_rho_df["p"].max()
    caption = (
        "Fig. 5. Agglomerate memory index M_D,agg vs. normalized thermal "
        "exposure (Theta), 27 sintering conditions (S/L within-furnace SEM "
        "secondary-particle-size pairs; M_D,agg = [D_sec(L) - D_sec(S)] / "
        "[precursor_D50(L) - precursor_D50(S)], evaluated on the same "
        "clear-border, shape-adaptive-h watershed segmentation used "
        "throughout this figure set). Marker colour codes sintering "
        f"temperature; the dashed line is an ordinary-least-squares fit "
        "shown only to indicate the sign of the trend. Vertical bars overlay "
        "the amplitude/sensitivity band for each condition: the range of "
        "M_D,agg obtained when the secondary-particle segmentation's two "
        "adaptive-h parameters (percentile, slope) are perturbed across a "
        f"9-combination grid (1 production default + 4 non-default "
        "percentile values + 4 non-default slope values). The M_D,agg-Theta "
        f"correlation is positive and significant at the production default "
        f"(Spearman rho={rho0:+.2f}, p={p0:.4f}, n=27) and remains positive "
        f"and significant across all 9 sensitivity-sweep combinations "
        f"(rho in [{rho_lo:+.2f}, {rho_hi:+.2f}], p<{p_max:.3f} throughout) "
        "-- the segmentation-parameter choice does not drive the trend. "
        "This figure is promoted from the supplementary information to the "
        "main text because it is the only main-text quantitative evidence "
        "for one of this paper's three central conclusions: precursor "
        "particle-size identity leaves a measurable imprint on sintered "
        "product morphology that strengthens, rather than erodes, with "
        "increasing thermal exposure -- i.e., morphological memory is "
        "retained, not erased, as Theta increases. A companion, "
        "algorithmically-independent legacy fixed-h segmentation "
        "cross-check (a different, non-production segmentation family) is "
        "reported separately in the supplementary information as "
        "additional, non-primary corroboration and is not reproduced here."
    )
    save_fig_journal(fig, NAME, caption)
    plt.close(fig)
    return fig


if __name__ == "__main__":
    main()
