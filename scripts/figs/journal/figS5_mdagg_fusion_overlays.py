# -*- coding: utf-8 -*-
"""figS5_mdagg_fusion_overlays.py -- Fig. S5 (journal submission set,
supplementary information): per-sample raw vs. segmented-overlay diagnostic
panel, restyled and re-laid-out from
scripts/02j_mdagg_fusion_check.py's make_diagnostic_overlays() (lines
~293-336 of that script; read-only source; not imported, not modified).
This is the OTHER of 02j's two figure functions -- distinct from
make_robustness_figure()/fig_S_mdagg_robustness, which Task 6 already
promoted to the main-text Fig. 5 -- and is not otherwise redone here.

Row-count check before committing to a layout (per task brief instruction):
02j's DIAG_SAMPLES = ["C01-S","C01-L","C14-S","C14-L","C21-S","C21-L"] is a
fixed list of 6 samples (verified against 02j's source, not assumed). The
source script's own layout is Nx2 = 6 rows x 2 columns (raw | segmented-
overlay) on a (10, 4.4*6=26.4) inch canvas (that height alone is ~671mm,
nearly 3x this figure set's 230mm SI height ceiling) -- proportionally
shrinking that canvas to fit would leave illegible ~1/3-inch-tall images.

Redesign chosen: the 6 samples are 3 sintering conditions (C01, C14, C21 --
the diagnostic's own low/mid/high Theta ladder) x 2 precursors (S, L) each.
Re-laying out as 3 rows (one per condition) x 4 columns (S-raw, S-segmented,
L-raw, L-segmented) keeps every raw/segmented pair adjacent (same
comparison the source script's 2-column layout preserves) while additionally
placing the two precursors side by side within a row -- which is the
diagnostic's actual purpose (S vs. L segmentation-quality comparison at
matched thermal exposure, the basis for M_D_agg = [D_sec(L) - D_sec(S)] /
[precursor_D50(L) - precursor_D50(S)]) and was not directly visible in the
source's stacked single-column layout. This halves the row count (3 vs. 6)
and roughly quarters the required canvas height for the same reason
figS2's bar-to-line redesign reduces clutter: fewer rows means each row can
be taller/more legible within a fixed height budget.

Segmentation itself: identical production call to 02j's
make_diagnostic_overlays() -- same config.yaml SegConfig (production
defaults, sec_adaptive_h_percentile=90, sec_adaptive_h_slope=0.082), same
nfm.data_processing.sem_processor._segment_secondary() call on the first
500x low-magnification file per sample directory, same skimage
find_boundaries(labels, mode="outer") green boundary overlay. Library
functions (sem_processor, skimage) are imported directly; 02j_mdagg_fusion_
check.py itself is not imported, per this task's replicate-the-logic-not-
the-script convention (same approach fig5_memory_index_robustness.py in
this directory already uses for 02j's other function).
"""
import _bootstrap  # noqa: F401

import warnings

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

from _journal_style import (
    double_column_figsize,
    save_fig_journal,
    dump_source_data,
    REPO_ROOT,
)
from _style import MASTER_TABLE
from nfm.data_processing import sem_processor as sp

NAME = "figS5_mdagg_fusion_overlays"

# Identical sample list to 02j_mdagg_fusion_check.py's DIAG_SAMPLES, just
# reshaped here into (condition_id -> {precursor: sample_id}) for the 3x4
# grid instead of the source's flat 6-row list.
CONDITIONS = ["C01", "C14", "C21"]
PRECURSORS = ["S", "L"]
DIAG_SAMPLES = {c: {p: f"{c}-{p}" for p in PRECURSORS} for c in CONDITIONS}

CONFIG_YAML = REPO_ROOT / "configs" / "config.yaml"


def _load_theta_map() -> dict:
    df = pd.read_csv(MASTER_TABLE)
    out = {}
    for cid in CONDITIONS:
        g = df[df["condition_id"] == cid]
        assert not g.empty, f"condition {cid} not found in master_table.csv"
        out[cid] = float(g["Theta"].iloc[0])
    return out


def _segment_one(sample_id: str, seg_cfg):
    sub = REPO_ROOT / "data" / "raw" / "sem" / sample_id
    low, _high = sp._collect_mag_files(sub)
    if not low:
        return None
    f = low[0]
    px, _src = sp._read_pixel_size_um(f)
    img, _cut = sp._crop_info_bar(sp._read_image_gray(f), seg_cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        props, labels = sp._segment_secondary(img, px, seg_cfg)
    return dict(sample_id=sample_id, file=f.name, img=img, labels=labels,
               n=len(props), d_med=props["equiv_diam_um"].median() if len(props) else np.nan)


def main():
    if not CONFIG_YAML.exists():
        print(f"[skip] {NAME}: {CONFIG_YAML} missing")
        return
    raw_yaml = yaml.safe_load(open(CONFIG_YAML, encoding="utf-8"))
    sem_cfg = sp.config_from_yaml(raw_yaml)
    seg_cfg = sem_cfg.seg  # production defaults, same as 02j

    theta_map = _load_theta_map()

    results = {}
    for cid in CONDITIONS:
        for prec in PRECURSORS:
            sid = DIAG_SAMPLES[cid][prec]
            r = _segment_one(sid, seg_cfg)
            if r is None:
                print(f"[skip] {NAME}: no 500x file for {sid}")
                return
            results[(cid, prec)] = r

    diag_rows = [
        dict(condition_id=cid, precursor=prec, sample_id=results[(cid, prec)]["sample_id"],
            file=results[(cid, prec)]["file"], Theta=theta_map[cid],
            n_particles=results[(cid, prec)]["n"],
            D_sec_median_um=results[(cid, prec)]["d_med"])
        for cid in CONDITIONS for prec in PRECURSORS
    ]
    dump_source_data(
        NAME, "diagnostics", pd.DataFrame(diag_rows),
        note="Per-panel segmentation diagnostics (particle count, median "
             "equivalent diameter) annotated under each raw/segmented image "
             "pair; the raw and segmentation-overlay images themselves are "
             "not tabular source data (this is a text-annotation panel, not "
             "a chart -- the underlying per-particle segmentation numbers "
             "are the same production pipeline used for D_sec everywhere "
             "else in this figure set).",
        source_files=[MASTER_TABLE, CONFIG_YAML],
    )

    fig = plt.figure(figsize=double_column_figsize(150))
    gs = fig.add_gridspec(3, 4, wspace=0.06, hspace=0.30,
                          left=0.045, right=0.975, top=0.90, bottom=0.03)

    col_titles = ["S, raw", "S, segmented", "L, raw", "L, segmented"]
    for r_i, cid in enumerate(CONDITIONS):
        theta = theta_map[cid]
        for p_i, prec in enumerate(PRECURSORS):
            res = results[(cid, prec)]
            img, labels = res["img"], res["labels"]

            ax_raw = fig.add_subplot(gs[r_i, p_i * 2])
            ax_raw.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax_raw.set_xticks([]); ax_raw.set_yticks([])
            for spine in ax_raw.spines.values():
                spine.set_linewidth(0.5)

            ax_seg = fig.add_subplot(gs[r_i, p_i * 2 + 1])
            ov = np.dstack([img] * 3).astype(float) / 255
            b = find_boundaries(labels, mode="outer")
            ov[b] = [0, 1, 0]
            ax_seg.imshow(ov)
            ax_seg.set_xticks([]); ax_seg.set_yticks([])
            for spine in ax_seg.spines.values():
                spine.set_linewidth(0.5)
            d_str = f"{res['d_med']:.2f}" if np.isfinite(res["d_med"]) else "NA"
            ax_seg.text(0.5, -0.06, f"n={res['n']}, D_sec med={d_str} um",
                       transform=ax_seg.transAxes, fontsize=5.0, ha="center", va="top")

            if r_i == 0:
                ax_raw.set_title(col_titles[p_i * 2], fontsize=6.5)
                ax_seg.set_title(col_titles[p_i * 2 + 1], fontsize=6.5)
            if p_i == 0:
                ax_raw.set_ylabel(f"{cid}\n" + r"$\Theta$" + f"={theta:.1f} h",
                                  fontsize=6.5, rotation=0, ha="right", va="center",
                                  labelpad=8)

    caption = (
        "Fig. S5. Per-condition, per-precursor secondary-particle "
        "segmentation diagnostic overlays (raw 500x SEM frame next to its "
        "production watershed segmentation, green = detected particle "
        "boundaries), for the S vs. L precursor pair at three sintering "
        "conditions spanning the Theta ladder used elsewhere in this "
        "figure set: C01 (Theta=4.2 h, low thermal budget), C14 "
        "(Theta=15.2 h, mid), C21 (Theta=47.3 h, high thermal budget). "
        "Same production segmentation pipeline as the rest of this study "
        "(shape-adaptive-h watershed, clear-border, config.yaml production "
        "defaults: sec_adaptive_h_percentile=90, sec_adaptive_h_slope="
        "0.082) -- one representative 500x frame per sample (first file in "
        "acquisition order), not a full-sample aggregate. This is a visual "
        "diagnostic supporting the memory-index fusion-artefact check (is "
        "the M_D,agg vs. Theta trend in Fig. 5 explained by high-Theta "
        "particle fusion/necking being mis-segmented as larger single "
        "particles, rather than genuine precursor-encoded morphological "
        "memory?): each panel reports the segmented particle count (n) and "
        "median equivalent diameter (D_sec med) for that frame. The "
        "quantitative version of this check -- a 9-combination "
        "segmentation-parameter sensitivity sweep across all 81 samples, "
        "showing the M_D,agg-Theta correlation stays positive and "
        "significant throughout -- is reported as the error-bar band in "
        "Fig. 5, not repeated here; this figure supplies the qualitative, "
        "visually-inspectable counterpart at the three anchor conditions."
    )
    # 3x4 image grid sits close to the 190mm double-column width budget;
    # matplotlib's default savefig pad_inches (0.1in/side, applied on top of
    # the tight bbox by save_fig_journal's bbox_inches="tight") pushed the
    # exported PNG past 190mm. Same fix as fig2_variance_decomposition.py /
    # figS3_lattice_vs_theta.py: tighten the pad locally (this script only,
    # restored after saving).
    _old_pad = plt.rcParams["savefig.pad_inches"]
    plt.rcParams["savefig.pad_inches"] = 0.0
    try:
        save_fig_journal(fig, NAME, caption, panel_labels=False)
    finally:
        plt.rcParams["savefig.pad_inches"] = _old_pad
    plt.close(fig)
    return fig


if __name__ == "__main__":
    main()
