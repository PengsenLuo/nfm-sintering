# -*- coding: utf-8 -*-
"""02f_sec_seg_sweep.py —— 500× 二次颗粒分割参数扫描(2026-07-23)

目的
----
在 sem_processor.py 之外独立实现可参数化的二次颗粒分割(复用其
_crop_info_bar / _read_pixel_size_um / _regionprops_um,不改动 sem_processor
本体),扫描三个轴,产出全量指标表 + 代表样 overlay 图版,供人工目视定档。
本脚本只出图出表,不写 config 默认、不改 schema、不删 raw。

扫描轴
------
  A. sec_hmaxima_rel ∈ {0.12, 0.15, 0.18, 0.21, 0.25}
  B. 二值化方法 ∈ {otsu(现状), local_adaptive}
     local_adaptive 用 skimage.filters.threshold_local。block_size 由像素
     尺寸换算:500× 二次颗粒尺度(D_sec 实测约 5~40µm)比 _segment_primary_classic
     的一次颗粒尺度(pri_min_diam=0.3µm → block=2.5µm)大一个数量级,窗口按
     比例放大到 LOCAL_BLOCK_UM=12µm(约为裁剪后图高 690px×0.223µm/px≈154µm
     的 8%),在"跟随信息栏裁剪后残留的整体光照梯度"与"仍局部到单个视场"
     之间取中。offset 取 -0.008(任务给定 -0.005~-0.01 区间的中点)。
  C. sec_min_diam_um ∈ {0.6, 1.0}——只在 A/B 参考点(0.18 + local_adaptive)
     上做全量对照,不与 A×B 做满笛卡尔积(控规模)。

clear_border 维持 sem_processor 默认 True 不变。

输出
----
data/interim/sem_sec_sweep/
  ├── metrics.csv                每 (参数组合 × 样本) 一行
  └── overlays/
      ├── ab_sweep/{sample}_ab_sweep.png       10 档 A×B 边界叠加拼图
      └── mindiam_compare/{sample}_mindiam.png  0.6 vs 1.0 对照(2 档)

用法
----
  python scripts/02f_sec_seg_sweep.py                 # 全量扫描 + overlay + 报告
  python scripts/02f_sec_seg_sweep.py --limit N        # 断点续跑,每次只新算 N 条
  python scripts/02f_sec_seg_sweep.py --metrics-only    # 只算指标表,不出 overlay
  python scripts/02f_sec_seg_sweep.py --overlays-only   # 只出 overlay(需先有 metrics)
  python scripts/02f_sec_seg_sweep.py --report          # 只打印汇总(需已有 metrics.csv)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import exposure, filters, morphology, segmentation
from skimage.segmentation import find_boundaries

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfm.data_processing import sem_processor as sp  # noqa: E402

OUT_DIR = ROOT / "data" / "interim" / "sem_sec_sweep"
METRICS_CSV = OUT_DIR / "metrics.csv"
OVERLAY_AB_DIR = OUT_DIR / "overlays" / "ab_sweep"
OVERLAY_MINDIAM_DIR = OUT_DIR / "overlays" / "mindiam_compare"

HMAXIMA_VALUES = [0.12, 0.15, 0.18, 0.21, 0.25]
BIN_METHODS = ["otsu", "local_adaptive"]
MIN_DIAM_DEFAULT = 1.0
MIN_DIAM_EXTRA = 0.6
REF_HMAXIMA = 0.18
REF_METHOD = "local_adaptive"

LOCAL_BLOCK_UM = 12.0
LOCAL_OFFSET = -0.008

REP_SAMPLES = [f"{c}-{p}" for c in ("C07", "C14", "C21", "C27") for p in ("S", "M", "L")]


def combo_id(hmax: float, method: str, min_diam: float) -> str:
    return f"h{hmax:.2f}_{method}_d{min_diam:.1f}"


def ab_combos():
    return [(h, m, MIN_DIAM_DEFAULT) for h in HMAXIMA_VALUES for m in BIN_METHODS]


def all_combos():
    return ab_combos() + [(REF_HMAXIMA, REF_METHOD, MIN_DIAM_EXTRA)]


def segment_secondary_variant(img, pixel_um, hmaxima_rel, method, min_diam_um, seg_cfg):
    """_segment_secondary 的参数化版本(方法/h-maxima/最小直径可调),独立实现,
    不改动 sem_processor._segment_secondary。"""
    img_eq = exposure.equalize_adapthist(img, clip_limit=0.02)
    img_s = filters.gaussian(img_eq, sigma=seg_cfg.sec_gauss_sigma)
    if method == "otsu":
        mask = img_s > filters.threshold_otsu(img_s)
    elif method == "local_adaptive":
        block = max(31, int(round(LOCAL_BLOCK_UM / pixel_um)) | 1)
        local_thresh = filters.threshold_local(img_s, block_size=block, offset=LOCAL_OFFSET)
        mask = img_s > local_thresh
    else:
        raise ValueError(f"unknown method {method!r}")

    min_area_px = (np.pi * (min_diam_um / 2) ** 2) / (pixel_um ** 2)
    mask = morphology.remove_small_holes(mask, area_threshold=int(min_area_px))
    mask = morphology.remove_small_objects(mask, min_size=int(min_area_px * 0.5))
    mask = morphology.opening(mask, morphology.disk(seg_cfg.sec_open_disk))
    dist = ndi.distance_transform_edt(mask)
    dist = filters.gaussian(dist, sigma=1.0)
    h = hmaxima_rel * dist.max()
    hmax = morphology.h_maxima(dist, h)
    markers, _ = ndi.label(hmax)
    if markers.max() == 0:
        markers, _ = ndi.label(mask)
    labels = segmentation.watershed(-dist, markers, mask=mask, compactness=0.01)
    props = sp._regionprops_um(labels, pixel_um, min_diam_um, clear_border=seg_cfg.sec_clear_border)
    return props, labels, mask


def _empty_metric_row(sample_id, hmax, method, min_diam):
    return dict(sample_id=sample_id, precursor=sample_id[-1],
                combo=combo_id(hmax, method, min_diam),
                sec_hmaxima_rel=hmax, bin_method=method, sec_min_diam_um=min_diam,
                n_sec=0, D_sec=np.nan, circularity=np.nan, convexity=np.nan,
                aspect_ratio=np.nan, solid_frac=np.nan, area_weighted_solidity=np.nan)


def sample_metrics(sub, hmax, method, min_diam, seg_cfg):
    low, _high = sp._collect_mag_files(sub)
    if not low:
        return _empty_metric_row(sub.name, hmax, method, min_diam)
    all_props, solid_fracs = [], []
    for f in low:
        px, _ = sp._read_pixel_size_um(f)
        img, _ = sp._crop_info_bar(sp._read_image_gray(f), seg_cfg)
        props, _labels, mask = segment_secondary_variant(img, px, hmax, method, min_diam, seg_cfg)
        all_props.append(props)
        solid_fracs.append(float(mask.mean()))
    cat = pd.concat(all_props, ignore_index=True)
    agg = sp._aggregate_secondary(cat)
    aw_solidity = np.nan
    if len(cat):
        conv = (cat["area_um2"] / cat["convex_area_um2"].clip(lower=1e-6)).clip(0, 1)
        aw_solidity = float(np.average(conv, weights=cat["area_um2"]))
    return dict(sample_id=sub.name, precursor=sub.name[-1],
                combo=combo_id(hmax, method, min_diam),
                sec_hmaxima_rel=hmax, bin_method=method, sec_min_diam_um=min_diam,
                n_sec=agg["n_sec"], D_sec=agg["D_sec"], circularity=agg["circularity"],
                convexity=agg["convexity"], aspect_ratio=agg["aspect_ratio"],
                solid_frac=float(np.mean(solid_fracs)), area_weighted_solidity=aw_solidity)


def all_sample_dirs():
    raw = ROOT / "data" / "raw" / "sem"
    return sorted(p for p in raw.iterdir() if p.is_dir() and sp.SAMPLE_ID_RE.match(p.name))


def run_metrics_sweep(limit):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prev = pd.read_csv(METRICS_CSV) if METRICS_CSV.exists() else pd.DataFrame()
    rows = prev.to_dict("records") if len(prev) else []
    have = set(zip(prev.get("sample_id", []), prev.get("combo", [])))

    seg_cfg = sp.SegConfig()
    n_new = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for hmax, method, min_diam in all_combos():
            cid = combo_id(hmax, method, min_diam)
            for sub in all_sample_dirs():
                if (sub.name, cid) in have:
                    continue
                if limit is not None and n_new >= limit:
                    pd.DataFrame(rows).to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")
                    print(f"[sweep] --limit {limit} 达到,已写回 {len(rows)} 行,断点已保存")
                    return
                r = sample_metrics(sub, hmax, method, min_diam, seg_cfg)
                rows.append(r)
                have.add((sub.name, cid))
                n_new += 1
                tag = f"n_sec={r['n_sec']:3d} D_sec={r['D_sec']:.2f}" if r["n_sec"] else "(无 500x 数据)"
                print(f"[sweep] {cid:24s} {sub.name}  {tag}", flush=True)
                if n_new % 20 == 0:
                    pd.DataFrame(rows).to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")
    print(f"[sweep] 完成,共 {len(rows)} 行 -> {METRICS_CSV}")


def _overlay_ax(ax, img, labels, title):
    ov = np.dstack([img] * 3).astype(float) / 255
    b = find_boundaries(labels, mode="outer")
    ov[b] = [0, 1, 0]
    ax.imshow(ov)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def make_ab_overlay(sample_id, seg_cfg):
    out_path = OVERLAY_AB_DIR / f"{sample_id}_ab_sweep.png"
    sub = ROOT / "data" / "raw" / "sem" / sample_id
    low, _high = sp._collect_mag_files(sub)
    if not low:
        print(f"[overlay] {sample_id}: 无 500x 图,跳过")
        return
    f = low[0]
    px, _ = sp._read_pixel_size_um(f)
    img, _ = sp._crop_info_bar(sp._read_image_gray(f), seg_cfg)

    combos = ab_combos()
    fig, axes = plt.subplots(len(combos), 1, figsize=(9, 6.5 * len(combos)), dpi=150)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ax, (hmax, method, min_diam) in zip(axes, combos):
            props, labels, _mask = segment_secondary_variant(img, px, hmax, method, min_diam, seg_cfg)
            med = props["equiv_diam_um"].median() if len(props) else float("nan")
            title = (f"{sample_id}  h={hmax:.2f}  {method}  min_diam={min_diam:.1f}um  "
                     f"n={len(props)}  D_sec_med={med:.2f}um")
            _overlay_ax(ax, img, labels, title)
    fig.tight_layout()
    OVERLAY_AB_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[overlay] {out_path}")


def make_mindiam_overlay(sample_id, seg_cfg):
    out_path = OVERLAY_MINDIAM_DIR / f"{sample_id}_mindiam.png"
    sub = ROOT / "data" / "raw" / "sem" / sample_id
    low, _high = sp._collect_mag_files(sub)
    if not low:
        print(f"[overlay] {sample_id}: 无 500x 图,跳过")
        return
    f = low[0]
    px, _ = sp._read_pixel_size_um(f)
    img, _ = sp._crop_info_bar(sp._read_image_gray(f), seg_cfg)

    fig, axes = plt.subplots(2, 1, figsize=(9, 13), dpi=150)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ax, min_diam in zip(axes, (MIN_DIAM_DEFAULT, MIN_DIAM_EXTRA)):
            props, labels, _mask = segment_secondary_variant(
                img, px, REF_HMAXIMA, REF_METHOD, min_diam, seg_cfg)
            med = props["equiv_diam_um"].median() if len(props) else float("nan")
            title = (f"{sample_id}  h={REF_HMAXIMA:.2f}  {REF_METHOD}  min_diam={min_diam:.1f}um  "
                     f"n={len(props)}  D_sec_med={med:.2f}um")
            _overlay_ax(ax, img, labels, title)
    fig.tight_layout()
    OVERLAY_MINDIAM_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[overlay] {out_path}")


def run_overlays():
    seg_cfg = sp.SegConfig()
    for sample_id in REP_SAMPLES:
        out_ab = OVERLAY_AB_DIR / f"{sample_id}_ab_sweep.png"
        if out_ab.exists():
            print(f"[overlay] {out_ab} 已存在,跳过")
        else:
            make_ab_overlay(sample_id, seg_cfg)
        out_md = OVERLAY_MINDIAM_DIR / f"{sample_id}_mindiam.png"
        if out_md.exists():
            print(f"[overlay] {out_md} 已存在,跳过")
        else:
            make_mindiam_overlay(sample_id, seg_cfg)


def print_report(df: pd.DataFrame):
    ab = df[np.isclose(df.sec_min_diam_um, MIN_DIAM_DEFAULT)].copy()
    header = (f"{'combo':24s} {'D_S':>7s} {'D_M':>7s} {'D_L':>7s} {'L/S':>6s} "
              f"{'n_S':>6s} {'n_M':>6s} {'n_L':>6s} {'circ_S_med':>11s} {'solidity_L_aw':>14s}")
    print("=" * len(header))
    print("A x B 参数组合总表(min_diam=1.0固定;每样本行来自 metrics.csv)")
    print("=" * len(header))
    print(header)
    for hmax in HMAXIMA_VALUES:
        for method in BIN_METHODS:
            cid = combo_id(hmax, method, MIN_DIAM_DEFAULT)
            g = ab[ab.combo == cid]
            if not len(g):
                continue
            byp = g.groupby("precursor")
            d_mean = byp["D_sec"].mean()
            n_mean = byp["n_sec"].mean()
            ls = d_mean.get("L", np.nan) / d_mean.get("S", np.nan)
            circ_s = g.loc[g.precursor == "S", "circularity"].median()
            gl = g[g.precursor == "L"].dropna(subset=["area_weighted_solidity"])
            sol_l = (np.average(gl["area_weighted_solidity"], weights=gl["n_sec"].clip(lower=1))
                     if len(gl) else np.nan)
            print(f"{cid:24s} {d_mean.get('S', np.nan):7.2f} {d_mean.get('M', np.nan):7.2f} "
                  f"{d_mean.get('L', np.nan):7.2f} {ls:6.2f} "
                  f"{n_mean.get('S', np.nan):6.1f} {n_mean.get('M', np.nan):6.1f} "
                  f"{n_mean.get('L', np.nan):6.1f} {circ_s:11.3f} {sol_l:14.3f}")
    print("=" * len(header))

    ref1 = df[df.combo == combo_id(REF_HMAXIMA, REF_METHOD, MIN_DIAM_DEFAULT)]
    ref06 = df[df.combo == combo_id(REF_HMAXIMA, REF_METHOD, MIN_DIAM_EXTRA)]
    print(f"\nmin_diam 对照 @ 参考组合 h={REF_HMAXIMA}/{REF_METHOD} (1.0 -> 0.6):")
    for p in ["S", "M", "L"]:
        a = ref1[ref1.precursor == p]
        b = ref06[ref06.precursor == p]
        if len(a) and len(b):
            print(f"  {p}: n_sec {a.n_sec.mean():.1f} -> {b.n_sec.mean():.1f}   "
                  f"D_sec {a.D_sec.mean():.2f} -> {b.D_sec.mean():.2f} um")


def main():
    args = sys.argv[1:]
    limit = int(args[args.index("--limit") + 1]) if "--limit" in args else None
    metrics_only = "--metrics-only" in args
    overlays_only = "--overlays-only" in args
    report_only = "--report" in args

    if report_only:
        df = pd.read_csv(METRICS_CSV)
        print_report(df)
        return

    if not overlays_only:
        run_metrics_sweep(limit)

    if not metrics_only:
        if not METRICS_CSV.exists():
            print("[overlay] metrics.csv 不存在,先跑一遍不带 --overlays-only 的扫描")
            return
        run_overlays()

    if limit is None:
        df = pd.read_csv(METRICS_CSV)
        print_report(df)


if __name__ == "__main__":
    main()
