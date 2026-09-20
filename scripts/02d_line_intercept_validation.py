# -*- coding: utf-8 -*-
"""02d_line_intercept_validation.py —— 5000× 脊线分水岭的线截距法验证（2026-07-22）

为什么用线截距法验证
--------------------
脊线分水岭（`ridge watershed`）在 5000× 上目测优于原先的局部阈值前景分割，
但没有 ground truth，"看起来对"不等于"对"。线截距法（ASTM E112）是金相
晶粒度的标准手工方法，专为"晶粒已烧结连成一片、无法逐颗描边"的组织设计，
正好对应本项目 5000× 的处境，且完全独立于任何自动分割。

验证设计的关键：**不比尺寸，比同一次测量**
------------------------------------------
若把自动分割的粒径与手工截距长度直接比，需要引入形状因子换算，换算系数
本身就是一个可争论的自由度。本脚本改为直接比较**同一组测试线上的截点数**：

  - 手工：数测试圆与晶界的交点数 P_manual
  - 自动：数脊线分水岭的分区边界与同一组测试圆的交点数 P_auto

两者是同一个物理量在同一条曲线上的两次测量，无需任何换算即可比较。
平均截距长度 L̄ = 测试线总长 / P，故 P 之比即 L̄ 之比的倒数。

用测试圆而非直线（Hilliard/Abrams 圆截距法）：圆没有端点，不存在
"线段末端落在晶粒中间"的端点效应，无需端点修正；且圆对所有方向等权，
自动消除组织取向带来的各向异性偏差。

圆半径的选取
------------
ASTM 建议每视场截点数落在 50–150 之间：太少统计不稳，太多人工计数易错。
本脚本用 `ridge watershed` 的初步结果自适应选取半径，使 P_auto ≈ 60。
需说明：**用自动结果来定测试圆尺寸不会污染比较本身**——它只决定在哪里
画圆，不影响人工在这条圆上数出多少个交点。

输出
----
data/interim/sem_line_intercept/
  ├── LI_<sample>.png       测试圆叠加在 5000× 图上,高分辨率便于放大计数
  ├── 计数表.csv             待填:sample, circle, P_manual
  └── _auto_勿先看.csv       P_auto 与圆参数(计数完成后再用于比对)

用法
----
python scripts/02d_line_intercept_validation.py [--limit N] [--force]
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
from skimage import exposure, filters, measure, morphology, segmentation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfm.data_processing import sem_processor as sp  # noqa: E402

OUT_DIR = ROOT / "data" / "interim" / "sem_line_intercept"
SHEET = OUT_DIR / "计数表.csv"
AUTO = OUT_DIR / "_auto_勿先看.csv"

# 覆盖热预算与前驱体两个方向的极端与中间
SAMPLES = ["C21-S", "C26-L", "C14-M", "C07-S", "C03-L"]
TARGET_P = 60          # 目标自动截点数,据此定圆半径
N_CIRCLES = 3
SEED = 20260722


def ridge_partition(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """边界检测路线:晶界在 SE 像里是暗沟,用 Sato 管状滤波提脊,再分水岭分区。"""
    f = filters.gaussian(exposure.equalize_adapthist(img, clip_limit=0.02), sigma=1.0)
    ridge = filters.sato(1 - f, sigmas=range(1, 5), black_ridges=False)
    ridge = (ridge - ridge.min()) / (np.ptp(ridge) + 1e-9)
    seeds = morphology.h_maxima(filters.gaussian(1 - ridge, sigma=2.0), 0.10)
    markers, _ = ndi.label(seeds)
    return segmentation.watershed(ridge, markers), ridge


def circle_points(cy, cx, r, n=4000):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return cy + r * np.sin(t), cx + r * np.cos(t)


def foreground_mask(img: np.ndarray, px: float) -> np.ndarray:
    """区分"材料区"与"背景/孔隙暗区"。

    背景在 SE 像里是电子逃逸受阻的深阴影(既暗且平),用低分位阈值 + 大尺度
    形态学闭运算得到。用途:让测试圆在背景上的弧段被排除——那里没有晶界,
    不该数,也不该计入有效线长。这样手工(跳过暗区)与自动(只在材料区统计)
    的口径完全一致,消除 P_auto 系统性偏高。
    """
    s = filters.gaussian(img.astype(float) / 255, sigma=max(2.0, 0.10 / px))
    fg = s > np.percentile(s, 22)
    fg = morphology.binary_closing(fg, morphology.disk(int(round(0.3 / px))))
    fg = morphology.remove_small_holes(fg, area_threshold=int((1.0 / px) ** 2))
    fg = morphology.remove_small_objects(fg, min_size=int((1.0 / px) ** 2))
    return fg


def _arc_samples(cy, cx, r, n=3000):
    yy, xx = circle_points(cy, cx, r, n)
    yi = np.clip(np.round(yy).astype(int), 0, 10 ** 9)
    return yy, xx, yi


def count_crossings(labels: np.ndarray, fg: np.ndarray, cy, cx, r):
    """沿圆周统计晶界交点,只在前景(材料区)弧段上计数。

    返回 (P_auto, valid_frac):valid_frac 为落在前景内的圆周比例,
    用于换算有效线长 L_eff = 圆周 × valid_frac。
    """
    H, W = labels.shape
    yy, xx = circle_points(cy, cx, r)
    yi = np.clip(np.round(yy).astype(int), 0, H - 1)
    xi = np.clip(np.round(xx).astype(int), 0, W - 1)
    v = labels[yi, xi]
    infg = fg[yi, xi]
    # 只在"当前点与前一点均在前景"处计边界变化,避免把材料/背景边界当晶界
    prev_v = np.roll(v, 1)
    prev_fg = np.roll(infg, 1)
    crossings = int(np.sum((v != prev_v) & infg & prev_fg))
    return crossings, float(infg.mean())


def process(sample_id: str, force: bool) -> dict | None:
    png = OUT_DIR / f"LI_{sample_id}.png"
    if png.exists() and not force:
        return None
    sub = ROOT / "data" / "raw" / "sem" / sample_id
    _low, high = sp._collect_mag_files(sub)
    if not high:
        warnings.warn(f"{sample_id}: 无 5000× 图")
        return None
    f = high[0]
    px, _ = sp._read_pixel_size_um(f)
    img, _ = sp._crop_info_bar(sp._read_image_gray(f), sp.SegConfig())
    labels, _ridge = ridge_partition(img)
    fg = foreground_mask(img, px)

    H, W = img.shape
    cy, cx = H / 2, W / 2
    r_max = min(H, W) / 2 * 0.92

    # 自适应半径:用最大圆的前景弧估计单位有效长度截点数,再缩放到 TARGET_P
    p_ref, vf_ref = count_crossings(labels, fg, cy, cx, r_max)
    eff_len_ref = 2 * np.pi * r_max * max(vf_ref, 0.05)
    per_px = p_ref / eff_len_ref if p_ref else 1e-6
    r_tot_needed = TARGET_P / max(per_px, 1e-9) / (2 * np.pi)
    ratios = np.array([1.0, 0.7, 0.4])
    scale = min(r_tot_needed / ratios.sum(), r_max / ratios[0])
    radii = ratios * scale

    fig, ax = plt.subplots(figsize=(14, 14 * H / W))
    ax.imshow(img, cmap="gray")
    colors = ["#ff2d55", "#00e5ff", "#ffd60a"]
    rec = {"sample_id": sample_id, "pixel_um": px}
    for i, (r, c) in enumerate(zip(radii, colors)):
        # 逐点画:落在前景(材料区)的点用实色,落在背景的点用灰色——
        # 灰段是"跳过区",人工不数,程序也不在灰段统计。
        yy, xx = circle_points(cy, cx, r, 3000)
        yi = np.clip(np.round(yy).astype(int), 0, H - 1)
        xi = np.clip(np.round(xx).astype(int), 0, W - 1)
        infg = fg[yi, xi]
        ax.scatter(xx[infg], yy[infg], s=2.2, color=c, edgecolors="none")
        ax.scatter(xx[~infg], yy[~infg], s=1.6, color="#888888",
                   edgecolors="none", alpha=0.7)
        ax.text(cx + r * 0.707, cy - r * 0.707, "ABC"[i], color=c,
                fontsize=20, fontweight="bold",
                bbox=dict(fc="black", alpha=0.5, pad=2))
        p_auto, vfrac = count_crossings(labels, fg, cy, cx, r)
        rec[f"r{'ABC'[i]}_um"] = r * px
        rec[f"circumf{'ABC'[i]}_um"] = 2 * np.pi * r * px
        rec[f"valid_frac_{'ABC'[i]}"] = round(vfrac, 3)
        rec[f"Leff{'ABC'[i]}_um"] = 2 * np.pi * r * px * vfrac
        rec[f"P_auto_{'ABC'[i]}"] = p_auto
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{sample_id}  5000x  —  count grain-boundary crossings\n"
                 f"solid arc = count here   |   gray arc = pore/background, SKIP",
                 fontsize=13)
    _scalebar(ax, W, H, px)
    fig.tight_layout()
    fig.savefig(png, dpi=200)
    plt.close(fig)

    rec["P_auto_total"] = sum(rec[f"P_auto_{k}"] for k in "ABC")
    rec["L_total_um"] = sum(rec[f"Leff{k}_um"] for k in "ABC")
    rec["Lbar_auto_um"] = rec["L_total_um"] / max(rec["P_auto_total"], 1)
    return rec


def _scalebar(ax, W, H, px_um, bar_um=5.0):
    n = bar_um / px_um
    x0, y0 = W * 0.03, H * 0.96
    ax.plot([x0, x0 + n], [y0, y0], lw=4, color="yellow")
    ax.text(x0, y0 - H * 0.02, f"{bar_um:g} µm", color="yellow", fontsize=13)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    args = sys.argv[1:]
    limit = int(args[args.index("--limit") + 1]) if "--limit" in args else None
    force = "--force" in args

    prev = pd.read_csv(AUTO) if AUTO.exists() else pd.DataFrame()
    rows = prev.to_dict("records") if len(prev) else []
    have = set(prev.sample_id) if len(prev) else set()

    n = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sid in SAMPLES:
            if sid in have and not force:
                continue
            if limit is not None and n >= limit:
                break
            r = process(sid, force)
            if r:
                rows.append(r)
                n += 1
                print(f"[LI] {sid}  P_auto={r['P_auto_total']}  "
                      f"L̄_auto={r['Lbar_auto_um']:.3f} µm", flush=True)

    df = pd.DataFrame(rows).drop_duplicates("sample_id", keep="last")
    df.to_csv(AUTO, index=False, encoding="utf-8-sig")

    if not SHEET.exists():
        pd.DataFrame([{"sample_id": s, "circle": c, "P_manual": "", "notes": ""}
                      for s in SAMPLES for c in "ABC"]).to_csv(
            SHEET, index=False, encoding="utf-8-sig")
    print(f"\n图 {len(list(OUT_DIR.glob('LI_*.png')))}/{len(SAMPLES)} 张 → {OUT_DIR}")


if __name__ == "__main__":
    main()
