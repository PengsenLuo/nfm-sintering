# -*- coding: utf-8 -*-
"""fig4_morphology_and_shape.py -- Fig. 4 (journal submission set): merged
SEM morphology panel + shape-descriptor scatter panel. Six panels total,
the most layout-dense figure in the journal set.

Panels (a)(b): raw 500x SEM images, C01-S (Theta=4.2 h, low thermal budget)
vs. C21-S (Theta=47.3 h, high thermal budget). This is a line-for-line port
of scripts/figs/fig3_sem_panel.py's image-loading and scale-bar logic --
same two files (data/raw/sem/C01-S/C01-S-04.tif,
data/raw/sem/C21-S/HS20-2-04.tif), same
nfm.data_processing.sem_processor._read_image_gray / _crop_info_bar /
_read_pixel_size_um calls, same info-bar crop, same scale-bar drawing.
Deliberately NO segmentation, NO pseudo-color, NO watershed overlay: the
original script's rationale is that the C01-S vs. C21-S morphology contrast
(isolated near-spherical agglomerates at low Theta vs. faceted, mutually
fused blocks at high Theta) is visible in the raw image itself, not a
segmentation artifact -- showing a genuinely unprocessed image is the only
way that claim stays credible to a reviewer. This script keeps that
constraint; the only additions to the raw pixel content are the info-bar
crop (removes an instrument-drawn text bar, not sample content) and the
scale bar (a measurement aid, not a data annotation).

Panels (c)-(f): four shape descriptors (D_f, solidity, elongation,
roughness) vs. Theta, a restyled port of
scripts/figs/fig4_shape_descriptors.py. Same data sources: scatter points
from data/interim/sem_shape_descriptors_summary.csv (81 samples, T7's
500x SEM shape-descriptor summary), rho and 95% bootstrap CI read -- not
recomputed -- from data/interim/shape_descriptor_full_analysis.csv
(section=='spearman_theta', metric=='rho' rows). Same deliberate
methodological choice as the original: the dashed line is a plain OLS fit
that indicates only the sign/direction of the point-estimate correlation
(not a statistical claim), and the CI is stated as legend text rather than
drawn as a fabricated per-point confidence band -- T7's bootstrap CI is for
the correlation coefficient itself, not for the regression line, so drawing
a per-point band would misrepresent what was actually computed.
"""
import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import patheffects

from _journal_style import (
    double_column_figsize,
    add_panel_label,
    save_fig_journal,
    save_fig_journal_tiff,
    dump_source_data,
    PRECURSOR_COLOR,
    PRECURSOR_MARKER,
    REPO_ROOT,
)
from nfm.data_processing.sem_processor import (
    _read_image_gray, _crop_info_bar, _read_pixel_size_um, SegConfig,
)

NAME = "fig4_morphology_and_shape"
SEM_ONLY_NAME = "fig4_morphology_and_shape_sem_panels"

SEM_SAMPLES = [
    ("C01-S", REPO_ROOT / "data/raw/sem/C01-S/C01-S-04.tif", 4.2),
    ("C21-S", REPO_ROOT / "data/raw/sem/C21-S/HS20-2-04.tif", 47.3),
]

SUMMARY_CSV = REPO_ROOT / "data/interim/sem_shape_descriptors_summary.csv"
FULL_ANALYSIS_CSV = REPO_ROOT / "data/interim/shape_descriptor_full_analysis.csv"

DESCRIPTORS = [
    ("D_f", "Fractal dimension $D_f$"),
    ("solidity", "Solidity"),
    ("elongation", "Elongation"),
    ("roughness", "Roughness ratio"),
]
PRECURSORS = ["S", "M", "L"]

# English precursor legend labels -- scripts/figs/_style.py's PRECURSOR_LABEL
# contains Chinese text ("7:3混合"); on-figure text must be English-only, so
# this module defines its own English variant (same convention already used
# by scripts/figs/journal/fig3_orthogonality_vs_theta.py).
PRECURSOR_LABEL_EN = {
    "S": "S (D50~3 um)",
    "M": "M (7:3 blend)",
    "L": "L (D50~11 um)",
}


# ---------------------------------------------------------------------
# Panels (a)(b) -- SEM images
# ---------------------------------------------------------------------
def _load_sem_images():
    cfg = SegConfig()
    imgs = []
    for label, path, theta in SEM_SAMPLES:
        if not path.exists():
            return None
        img = _read_image_gray(path)
        img_c, cut = _crop_info_bar(img, cfg)
        px_um, src = _read_pixel_size_um(path)
        imgs.append((label, img_c, px_um, theta, cut, img.shape[0]))
    return imgs


def _text_outline():
    return [patheffects.withStroke(linewidth=2, foreground="black")]


def _add_scalebar(ax, px_um: float, img_width_px: int, bar_um: float = 10.0):
    """Line-for-line port of fig3_sem_panel.py's _add_scalebar: white bar +
    outlined text, positioned near the bottom-left of the (already-cropped)
    image, sized from the real pixel calibration (px_um)."""
    bar_px = bar_um / px_um
    x0 = img_width_px * 0.05
    y0 = ax.get_ylim()[0] * 0.96  # imshow y-axis is inverted; near bottom
    h = img_width_px * 0.012
    ax.add_patch(Rectangle((x0, y0 - h * 3), bar_px, h, color="white",
                           ec="black", linewidth=0.5, zorder=5))
    ax.text(x0 + bar_px / 2, y0 - h * 3.6, f"{bar_um:.0f} um", color="white",
           fontsize=9, ha="center", va="bottom", zorder=5,
           path_effects=_text_outline())


def _draw_sem_panel(ax, label, img_c, px_um, theta):
    ax.imshow(img_c, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"{label} ($\\Theta$ = {theta:.1f} h)", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    _add_scalebar(ax, px_um, img_c.shape[1])


# ---------------------------------------------------------------------
# Panels (c)-(f) -- shape descriptors vs Theta
# ---------------------------------------------------------------------
def _load_rho_ci() -> dict:
    """{(target, group): (rho, ci_low, ci_high, p)} -- read as-is from T7's
    output, not recomputed."""
    df = pd.read_csv(FULL_ANALYSIS_CSV)
    sub = df[df["section"] == "spearman_theta"]
    out = {}
    for (target, group), g in sub.groupby(["target", "group"]):
        g = g.set_index("metric")
        rho = g.loc["rho", "value"]
        ci_low = g.loc["rho", "ci_low"]
        ci_high = g.loc["rho", "ci_high"]
        p = (g.loc["p_value_asymptotic", "value"]
             if "p_value_asymptotic" in g.index else np.nan)
        out[(target, group)] = (rho, ci_low, ci_high, p)
    return out


def _draw_descriptor_panel(ax, df, rho_ci, col, label):
    for prec in PRECURSORS:
        g = df[df["precursor"] == prec]
        color = PRECURSOR_COLOR[prec]
        marker = PRECURSOR_MARKER[prec]
        rho, ci_lo, ci_hi, p = rho_ci.get((col, prec), (np.nan, np.nan, np.nan, np.nan))
        leg = (f"{prec}  $\\rho$={rho:+.2f} [{ci_lo:+.2f},{ci_hi:+.2f}]")
        ax.scatter(g["Theta"], g[col], color=color, marker=marker, s=16,
                  edgecolor="black", linewidth=0.3, label=leg, zorder=3)
        if g["Theta"].nunique() >= 2:
            b, a = np.polyfit(g["Theta"], g[col], 1)
            x_line = np.linspace(g["Theta"].min(), g["Theta"].max(), 50)
            ax.plot(x_line, a + b * x_line, color=color, linestyle="--",
                   linewidth=1.0, alpha=0.85, zorder=2)
    ax.set_xlabel(r"$\Theta$ (h)")
    ax.set_ylabel(label)
    ax.legend(loc="best", fontsize=5.4, handlelength=1.1, labelspacing=0.3,
              borderaxespad=0.3)


# ---------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------
def _build_combined_figure(imgs, df, rho_ci):
    fig = plt.figure(figsize=double_column_figsize(178))
    outer = fig.add_gridspec(
        2, 1, height_ratios=[0.50, 1.0], hspace=0.08,
        left=0.085, right=0.985, top=0.955, bottom=0.058,
    )
    gs_top = outer[0].subgridspec(1, 2, wspace=0.10)
    gs_bot = outer[1].subgridspec(2, 2, wspace=0.30, hspace=0.55)

    panel_labels_top = ["(a)", "(b)"]
    for i, (label, img_c, px_um, theta, cut, h_full) in enumerate(imgs):
        ax = fig.add_subplot(gs_top[0, i])
        _draw_sem_panel(ax, label, img_c, px_um, theta)
        add_panel_label(ax, panel_labels_top[i])

    panel_labels_bot = ["(c)", "(d)", "(e)", "(f)"]
    for i, (col, label) in enumerate(DESCRIPTORS):
        r, c = divmod(i, 2)
        ax = fig.add_subplot(gs_bot[r, c])
        _draw_descriptor_panel(ax, df, rho_ci, col, label)
        add_panel_label(ax, panel_labels_bot[i])

    return fig


def _build_sem_only_figure(imgs):
    """Standalone 2-panel SEM-only sub-figure, saved separately as TIFF.

    Interpretation of the spec ("SEM 位图面板另存 TIFF" / "SEM bitmap panels
    saved separately as TIFF"): read literally, it names the SEM bitmap
    panels specifically, not "the combined figure". A journal's separate
    bitmap-panel submission requirement is about supplying the
    raster/photographic content (the SEM micrographs) as its own
    high-resolution raster file, distinct from the vector-content combined
    PDF/PNG that also contains the four vector scatter panels (c)-(f). So
    this standalone figure re-draws only panels (a)(b) -- same drawing code,
    same panel labels -- sized as its own small sub-figure and exported via
    save_fig_journal_tiff (>=300 dpi, here 600 dpi to match the journal
    PNG convention).
    """
    fig = plt.figure(figsize=double_column_figsize(70))
    gs = fig.add_gridspec(1, 2, wspace=0.10, left=0.045, right=0.985,
                          top=0.90, bottom=0.05)
    panel_labels = ["(a)", "(b)"]
    for i, (label, img_c, px_um, theta, cut, h_full) in enumerate(imgs):
        ax = fig.add_subplot(gs[0, i])
        _draw_sem_panel(ax, label, img_c, px_um, theta)
        add_panel_label(ax, panel_labels[i])
    return fig


def main():
    imgs = _load_sem_images()
    if imgs is None:
        print(f"[skip] {NAME}: SEM source TIFFs not found")
        return
    if not (SUMMARY_CSV.exists() and FULL_ANALYSIS_CSV.exists()):
        print(f"[skip] {NAME}: sem_shape_descriptors_summary.csv or "
              f"shape_descriptor_full_analysis.csv missing")
        return

    df = pd.read_csv(SUMMARY_CSV)
    rho_ci = _load_rho_ci()
    assert len(df) == 81, f"expected 81 samples in shape descriptor summary, got {len(df)}"

    sem_note = "; ".join(
        f"{lbl}: {path.relative_to(REPO_ROOT).as_posix()}, Theta={theta:.1f} h, "
        f"pixel_size={px_um:.4f} um/px, cropped {h_full - img_c.shape[0]} of {h_full} rows"
        for (lbl, img_c, px_um, theta, cut, h_full), (_, path, _) in zip(imgs, SEM_SAMPLES)
    )
    dump_source_data(
        NAME, "sem_panels", df=None,
        note=f"Panels (a)(b): raw SEM images, no tabular source data -- the "
             f"unprocessed pixel content is the data (see "
             f"{SEM_ONLY_NAME}.tiff). {sem_note}",
    )

    panel_letter = {"D_f": "c", "solidity": "d", "elongation": "e", "roughness": "f"}
    for col, label in DESCRIPTORS:
        panel_df = df[["sample_id", "precursor", "Theta", col]].copy()
        rho_bits = []
        for prec in PRECURSORS:
            rho, ci_lo, ci_hi, _p = rho_ci.get((col, prec), (float("nan"),) * 4)
            rho_bits.append(f"{prec}: rho={rho:+.2f} [{ci_lo:+.2f},{ci_hi:+.2f}]")
        dump_source_data(
            NAME, panel_letter[col], panel_df,
            note=f"{label} vs Theta scatter, n=81. Per-precursor Spearman "
                 f"rho [95% bootstrap CI], read (not recomputed) from "
                 f"data/interim/shape_descriptor_full_analysis.csv: "
                 f"{'; '.join(rho_bits)}. Dashed lines are per-group OLS "
                 f"fits shown only to indicate sign of trend, not "
                 f"separately tabulated.",
            source_files=[SUMMARY_CSV, FULL_ANALYSIS_CSV],
        )

    fig = _build_combined_figure(imgs, df, rho_ci)

    n_removed = [f"{lbl}: cropped {h_full - img_c.shape[0]} of {h_full} rows"
                for lbl, img_c, _, _, _, h_full in imgs]
    caption = (
        "Fig. 4. Morphology contrast and quantitative shape evolution vs. "
        "normalized thermal exposure (Theta). (a)(b) Raw 500x SEM "
        "micrographs, C01-S (Theta = 4.2 h, low thermal budget) vs. C21-S "
        "(Theta = 47.3 h, high thermal budget); both share the same pixel "
        "calibration (0.2233 um/px, read from CZ_SEM metadata tag 34118). "
        "Only the instrument-drawn info bar at the bottom of each raw frame "
        "is cropped (" + "; ".join(n_removed) + "); no segmentation, "
        "pseudo-color, or overlay is applied -- the panels show the "
        "unprocessed images so the low-Theta vs. high-Theta contrast "
        "(isolated near-spherical agglomerates vs. faceted, mutually fused "
        "blocks) can be read directly, not as a segmentation artifact. "
        "White bar = 10 um. (c)-(f) Four 500x SEM shape descriptors -- "
        "fractal dimension D_f, solidity, elongation, and roughness ratio "
        "-- vs. Theta for all 81 samples, grouped by precursor "
        f"({', '.join(PRECURSOR_LABEL_EN[p] for p in PRECURSORS)}; colour "
        "and marker double-encode precursor identity, same convention as "
        "the other figures in this set). Dashed lines are per-group "
        "ordinary-least-squares fits shown only to indicate the sign of "
        "the trend, not a statistical claim; legends state each group's "
        "Spearman rho and its 95% bootstrap confidence interval "
        "(n_boot=5000) as computed once upstream and read here unchanged -- "
        "the interval describes the correlation coefficient, not the "
        "regression line, so no per-point confidence band is drawn. "
        "roughness_ratio is the only descriptor whose CI excludes zero in "
        "all three precursor groups; S shows a complete, CI-excludes-zero "
        "evolution pattern across D_f, solidity, and elongation as well, "
        "while M and L show weaker or zero-crossing intervals on those "
        "three -- reconstruction magnitude scales inversely with precursor "
        "particle size."
    )
    save_fig_journal(fig, NAME, caption)
    plt.close(fig)

    fig_sem = _build_sem_only_figure(imgs)
    save_fig_journal_tiff(fig_sem, SEM_ONLY_NAME)
    plt.close(fig_sem)
    return fig


if __name__ == "__main__":
    main()
