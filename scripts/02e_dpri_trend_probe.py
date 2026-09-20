# -*- coding: utf-8 -*-
"""02e_dpri_trend_probe.py —— 全量一次颗粒尺寸的 Θ 趋势探查（2026-07-22，无需人工计数）

目的
----
在决定"D_pri 只讲趋势 vs 要绝对值"之前,先摸清脊线分水岭 + 圆截距法在全部
有 5000× 图的样本上能给出什么样的趋势。方法完全自动:每张 5000× 图铺一组
同心测试圆(前景弧段),数晶界交点,得平均截距长度 L̄;同炉/同前驱体聚合后
对热预算 Θ 作图。

L̄ 本身就是一个合法的尺寸度量(ASTM E112),这里不换算成"直径"——趋势分析
只需要一个单调对应真实尺寸的量,避免引入形状因子这个自由度。报告 L̄(µm)。

判据(不下科学结论,只看方法可用性)
---------------------------------
1. 单调性:L̄ 是否随 Θ 增大而增大(晶粒长大)
2. 前驱体分离:同 Θ 下 L 型是否系统性大于 S 型(形貌记忆)
3. 稳健性:同一样本三张图之间 L̄ 的离散度(CV),判断 3 图是否够

输出
----
data/interim/sem_dpri_trend/
  ├── dpri_intercept_all.csv     每样本 L̄ 及三图 CV
  ├── dpri_vs_theta.png          L̄–Θ 散点,前驱体着色,含每前驱体趋势线
  └── (控制台)                    Spearman(L̄,Θ) 分前驱体、L/S 配对检验

用法: python scripts/02e_dpri_trend_probe.py [--limit N]
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
from scipy import stats
from skimage import exposure, filters, morphology, segmentation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfm.data_processing import sem_processor as sp  # noqa: E402

OUT_DIR = ROOT / "data" / "interim" / "sem_dpri_trend"
CSV = OUT_DIR / "dpri_intercept_all.csv"
N_CIRCLES = 6           # 每图同心圆数,半径均匀展开,统计更稳
TARGET_TOTAL = 200      # 每图目标总截点数


def ridge_partition(img):
    f = filters.gaussian(exposure.equalize_adapthist(img, clip_limit=0.02), sigma=1.0)
    ridge = filters.sato(1 - f, sigmas=range(1, 5), black_ridges=False)
    ridge = (ridge - ridge.min()) / (np.ptp(ridge) + 1e-9)
    seeds = morphology.h_maxima(filters.gaussian(1 - ridge, sigma=2.0), 0.10)
    markers, _ = ndi.label(seeds)
    return segmentation.watershed(ridge, markers)


def foreground_mask(img, px):
    s = filters.gaussian(img.astype(float) / 255, sigma=max(2.0, 0.10 / px))
    fg = s > np.percentile(s, 22)
    fg = morphology.binary_closing(fg, morphology.disk(max(1, int(round(0.3 / px)))))
    fg = morphology.remove_small_holes(fg, area_threshold=int((1.0 / px) ** 2))
    return morphology.remove_small_objects(fg, min_size=int((1.0 / px) ** 2))


def intercept_Lbar(img, px):
    """一张图的平均截距长度 L̄ = 前景弧总长 / 晶界交点数。"""
    labels = ridge_partition(img)
    fg = foreground_mask(img, px)
    H, W = img.shape
    cy, cx = H / 2, W / 2
    r_max = min(H, W) / 2 * 0.92
    radii = np.linspace(0.25, 1.0, N_CIRCLES) * r_max
    P_tot, L_tot = 0.0, 0.0
    for r in radii:
        t = np.linspace(0, 2 * np.pi, 3000, endpoint=False)
        yy = np.clip(np.round(cy + r * np.sin(t)).astype(int), 0, H - 1)
        xx = np.clip(np.round(cx + r * np.cos(t)).astype(int), 0, W - 1)
        v = labels[yy, xx]
        infg = fg[yy, xx]
        pv, pf = np.roll(v, 1), np.roll(infg, 1)
        P_tot += np.sum((v != pv) & infg & pf)
        L_tot += 2 * np.pi * r * px * infg.mean()
    return (L_tot / P_tot) if P_tot else np.nan, P_tot


def process(sample_id, cfg):
    sub = ROOT / "data" / "raw" / "sem" / sample_id
    _low, high = sp._collect_mag_files(sub)
    if not high:
        return None
    vals = []
    for f in high:
        px, _ = sp._read_pixel_size_um(f)
        img, _ = sp._crop_info_bar(sp._read_image_gray(f), cfg.seg)
        lbar, _p = intercept_Lbar(img, px)
        if np.isfinite(lbar):
            vals.append(lbar)
    if not vals:
        return None
    vals = np.array(vals)
    return dict(sample_id=sample_id, precursor=sample_id[-1],
                Lbar_um=float(vals.mean()), Lbar_sd=float(vals.std()),
                n_img=len(vals), cv=float(vals.std() / vals.mean()))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    args = sys.argv[1:]
    limit = int(args[args.index("--limit") + 1]) if "--limit" in args else None

    master = pd.read_csv(ROOT / "data" / "processed" / "master_table.csv")
    theta = master.set_index("sample_id")["Theta"].to_dict()

    prev = pd.read_csv(CSV) if CSV.exists() else pd.DataFrame()
    rows = prev.to_dict("records") if len(prev) else []
    have = set(prev.sample_id) if len(prev) else set()

    cfg = sp.SemProcessorConfig(save_overlays=False)
    n = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sub in sorted(p for p in (ROOT / "data/raw/sem").iterdir() if p.is_dir()):
            if not sp.SAMPLE_ID_RE.match(sub.name) or sub.name in have:
                continue
            if limit is not None and n >= limit:
                break
            r = process(sub.name, cfg)
            if r:
                r["Theta"] = theta.get(r["sample_id"], np.nan)
                rows.append(r)
                n += 1
                print(f"[dpri] {r['sample_id']}  L̄={r['Lbar_um']:.3f}µm  "
                      f"CV={r['cv']:.2f}  n_img={r['n_img']}", flush=True)
    df = pd.DataFrame(rows).drop_duplicates("sample_id", keep="last")
    df.to_csv(CSV, index=False, encoding="utf-8-sig")

    if "--report" in args or (limit is None):
        report(df)


def report(df):
    df = df.dropna(subset=["Theta", "Lbar_um"])
    print(f"\n{'='*70}\n样本数 {len(df)}   前驱体分布 {df.precursor.value_counts().to_dict()}")
    print(f"三图 CV 中位数 {df.cv.median():.2f}（衡量单样本 3 图的稳健性）\n")

    print("Spearman(L̄, Θ) 分前驱体：")
    for p in ["S", "M", "L"]:
        s = df[df.precursor == p]
        if len(s) >= 4:
            rho, pv = stats.spearmanr(s.Theta, s.Lbar_um)
            print(f"  {p}: rho={rho:+.3f}  p={pv:.3f}  (n={len(s)})")

    # L/S 同炉配对
    piv = df.pivot_table(index=[df.sample_id.str[:3]], columns="precursor",
                         values="Lbar_um", aggfunc="first")
    if {"S", "L"} <= set(piv.columns):
        pair = piv.dropna(subset=["S", "L"])
        if len(pair) >= 3:
            w = stats.wilcoxon(pair.L, pair.S)
            print(f"\nL vs S 同炉配对（n={len(pair)}）：L>S 占 "
                  f"{(pair.L > pair.S).mean():.0%}，Wilcoxon p={w.pvalue:.3f}")
            print(f"  L̄ 均值  S={pair.S.mean():.3f}  L={pair.L.mean():.3f} µm")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    col = {"S": "#e63946", "M": "#457b9d", "L": "#2a9d8f"}
    for p in ["S", "M", "L"]:
        s = df[df.precursor == p]
        ax.scatter(s.Theta, s.Lbar_um, c=col[p], label=f"{p} (n={len(s)})",
                   s=42, alpha=0.85, edgecolors="white", linewidths=0.5)
        if len(s) >= 4:
            z = np.polyfit(s.Theta, s.Lbar_um, 1)
            xs = np.linspace(s.Theta.min(), s.Theta.max(), 50)
            ax.plot(xs, np.polyval(z, xs), c=col[p], lw=1.5, alpha=0.6)
    ax.set_xlabel("Θ  (normalized thermal exposure, h)")
    ax.set_ylabel("mean linear intercept  L̄  (µm)")
    ax.set_title("Auto primary-grain size vs thermal budget\n"
                 "ridge-watershed circular intercept, all 5000x samples")
    ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dpri_vs_theta.png", dpi=150)
    plt.close(fig)
    print(f"\n图 → {OUT_DIR/'dpri_vs_theta.png'}")


if __name__ == "__main__":
    main()
