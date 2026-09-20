# -*- coding: utf-8 -*-
"""02g_adaptive_h_overlay.py —— 形貌自适应 h 的代表样 overlay(2026-07-23)

对 C07/C14/C21/C27 × S/M/L 共 12 代表样,在 SegConfig.sec_adaptive_h=True 下逐张
500x 图分割,标注该图实际选用的 dt_scale(µm)与映射得到的 h(sec_hmaxima_rel),
只叠边界线不填色,供目视确认自适应映射的校准锚点
(sec_adaptive_h_x0=1.6 / sec_adaptive_h_intercept=0.16 / …)是否需要调整。

本脚本复刻 sem_processor._segment_secondary 的分割步骤以便取出 dt_scale/h 这两个
诊断量(该函数本身不返回中间量);复刻逻辑与被测代码路径一致,不改动
sem_processor.py 本体。

输出: data/interim/sem_adaptive_overlays/{sample}_adaptive.png
用法: python scripts/02g_adaptive_h_overlay.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import exposure, filters, morphology, segmentation
from skimage.segmentation import find_boundaries

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfm.data_processing import sem_processor as sp  # noqa: E402

OUT_DIR = ROOT / "data" / "interim" / "sem_adaptive_overlays"
REP_SAMPLES = [f"{c}-{p}" for c in ("C07", "C14", "C21", "C27") for p in ("S", "M", "L")]


def segment_with_diagnostics(img, pixel_um, cfg):
    img_eq = exposure.equalize_adapthist(img, clip_limit=0.02)
    img_s = filters.gaussian(img_eq, sigma=cfg.sec_gauss_sigma)
    mask = img_s > filters.threshold_otsu(img_s)
    min_area_px = (np.pi * (cfg.sec_min_diam_um / 2) ** 2) / (pixel_um ** 2)
    mask = morphology.remove_small_holes(mask, area_threshold=int(min_area_px))
    mask = morphology.remove_small_objects(mask, min_size=int(min_area_px * 0.5))
    mask = morphology.opening(mask, morphology.disk(cfg.sec_open_disk))
    dist_raw = ndi.distance_transform_edt(mask)
    dt_scale = sp._particle_scale_um(mask, dist_raw, pixel_um, cfg.sec_adaptive_h_percentile)
    h_rel = sp._map_h_from_scale(dt_scale, cfg)
    dist = filters.gaussian(dist_raw, sigma=1.0)
    h = h_rel * dist.max()
    hmax = morphology.h_maxima(dist, h)
    markers, _ = ndi.label(hmax)
    if markers.max() == 0:
        markers, _ = ndi.label(mask)
    labels = segmentation.watershed(-dist, markers, mask=mask, compactness=0.01)
    props = sp._regionprops_um(labels, pixel_um, cfg.sec_min_diam_um, clear_border=cfg.sec_clear_border)
    return props, labels, dt_scale, h_rel


def make_overlay(sample_id, cfg):
    sub = ROOT / "data" / "raw" / "sem" / sample_id
    low, _high = sp._collect_mag_files(sub)
    if not low:
        print(f"[adaptive] {sample_id}: 无 500x 图,跳过")
        return
    n = len(low)
    fig, axes = plt.subplots(n, 1, figsize=(9, 6.5 * n), dpi=150)
    if n == 1:
        axes = [axes]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ax, f in zip(axes, low):
            px, _ = sp._read_pixel_size_um(f)
            img, _ = sp._crop_info_bar(sp._read_image_gray(f), cfg)
            props, labels, dt_scale, h_rel = segment_with_diagnostics(img, px, cfg)
            med = props["equiv_diam_um"].median() if len(props) else float("nan")
            ov = np.dstack([img] * 3).astype(float) / 255
            b = find_boundaries(labels, mode="outer")
            ov[b] = [0, 1, 0]
            ax.imshow(ov)
            ax.set_title(f"{sample_id}  {f.stem}  dt_scale={dt_scale:.2f}um  h={h_rel:.3f}  "
                         f"n={len(props)}  D_sec_med={med:.2f}um", fontsize=10)
            ax.axis("off")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{sample_id}_adaptive.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[adaptive] {out_path}")


def main():
    cfg = sp.SegConfig(sec_adaptive_h=True)
    for sample_id in REP_SAMPLES:
        make_overlay(sample_id, cfg)


if __name__ == "__main__":
    main()
