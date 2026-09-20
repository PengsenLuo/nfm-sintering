# -*- coding: utf-8 -*-
"""
03c_wh_diagnostic.py —— Williamson-Hall 散点诊断图(只读诊断,不改任何计算逻辑)
====================================================================================
背景:晶格参数已用修复4/5 救活(lattice_c 脱离 16.6 上界),但 Williamson-Hall
回归质量(wh_r2)仍偏低(0.16–0.40)。逐峰诊断显示峰本身拟合干净、无残留
错峰,怀疑是 O3 层状结构 00l 类反射(003/006)的各向异性宽化让"各向同性"
的 WH 线性模型(β·cosθ = Kλ/D + 4ε·sinθ)本身不适配这批数据——但这是个
物理层面的判断,需要看图,不是靠改代码能验证的事,所以本脚本只画图、不改
williamson_hall/fit_peak/lattice_params/HKL_2THETA_GUESS 的任何逻辑,也不改
config.yaml 的 wh_peaks。

每个端点样本画一张散点图:
  - 横轴 4sinθ,纵轴 β·cosθ(和 xrd_processor.williamson_hall 内部算法完全
    一致,直接复用该函数拿回归线/R²,不重新实现一套)。
  - 三类点用不同颜色/marker 区分:
      00l 反射(003/006)           —— 验证各向异性假说的关键点
      其余已在 wh_peaks 里的点(101/104/015/110/113)
      未纳入 wh_peaks 的点(012/107/018,拟合质量其实很好,只是 config
      没把它们列进当前回归)
  - 实线 = 当前 config.yaml 的 wh_peaks(7点)算出的回归线(生产环境实际用的);
    虚线 = 假设把全部 10 个点都纳入回归会是什么样子(纯对比用,是否
    采纳留待复核)。
  - 6 张图拼成一张 2x3 组图,存到 data/interim/wh_diagnostic.png。

用法:
    python scripts/03c_wh_diagnostic.py
"""
import _bootstrap  # noqa

import numpy as np

from nfm.config import load_config
from nfm.data_processing.xrd_processor import (
    HKL_2THETA_GUESS, _beta_sample_rad, fit_peak, read_xy, williamson_hall,
)

SAMPLES = ["C07-L", "C07-M", "C07-S", "C21-L", "C21-M", "C21-S"]
OOL_HKL = {"003", "006"}
OUT_PNG = "data/interim/wh_diagnostic.png"


def _category(hkl, wh_peaks):
    if hkl in OOL_HKL:
        return "00l"
    if hkl in wh_peaks:
        return "in_wh_peaks"
    return "not_in_wh_peaks"


def collect_sample_points(sid, cfg):
    xc = cfg.raw["instruments"]["xrd"]
    lam1, lam2, ratio = xc["wavelength_A"], xc["ka2_wavelength_A"], xc["ka2_ratio"]
    K = xc["scherrer_K"]
    caglioti = tuple(xc.get("caglioti_UVW", (0.001, -0.0005, 0.004)))
    wh_peaks_cfg = xc.get("wh_peaks", list(HKL_2THETA_GUESS.keys()))

    tt, ii = read_xy(f"data/raw/xrd/{sid}.txt")
    fitted = {}
    for hkl, g in HKL_2THETA_GUESS.items():
        pk = fit_peak(tt, ii, g, window=1.0, lam1=lam1, lam2=lam2, ratio=ratio)
        if pk is None or pk["degenerate"]:
            continue
        fitted[hkl] = pk

    points = []
    for hkl, pk in fitted.items():
        beta = _beta_sample_rad(pk["fwhm"], pk["two_theta"], caglioti)
        if not np.isfinite(beta):
            continue
        th = np.radians(pk["two_theta"] / 2.0)
        x = 4.0 * np.sin(th)
        y = beta * np.cos(th)
        points.append(dict(hkl=hkl, x=x, y=y, category=_category(hkl, wh_peaks_cfg)))

    wh_current = williamson_hall([fitted.get(h) for h in wh_peaks_cfg], lam1, K, caglioti)
    wh_all10 = williamson_hall(list(fitted.values()), lam1, K, caglioti)
    return points, wh_current, wh_all10, wh_peaks_cfg


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config()
    style = {
        "00l": dict(marker="^", color="C3", s=90, label="00l (003/006)"),
        "in_wh_peaks": dict(marker="o", color="C0", s=70, label="in wh_peaks"),
        "not_in_wh_peaks": dict(marker="x", color="C2", s=70, label="not in wh_peaks"),
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    summary_rows = []
    for ax, sid in zip(axes.ravel(), SAMPLES):
        points, wh_current, wh_all10, wh_peaks_cfg = collect_sample_points(sid, cfg)

        seen_cat = set()
        for p in points:
            kw = dict(style[p["category"]])
            if p["category"] in seen_cat:
                kw.pop("label", None)
            seen_cat.add(p["category"])
            ax.scatter(p["x"], p["y"], **kw)
            ax.annotate(p["hkl"], (p["x"], p["y"]), textcoords="offset points",
                       xytext=(5, 5), fontsize=8)

        xs_all = np.array([p["x"] for p in points])
        x_line = np.linspace(xs_all.min() * 0.95, xs_all.max() * 1.05, 50)

        # 实线:当前 config.yaml wh_peaks(7点)的回归
        if np.isfinite(wh_current["wh_r2"]):
            xs_cur = np.array([p["x"] for p in points if p["category"] != "not_in_wh_peaks"])
            ys_cur = np.array([p["y"] for p in points if p["category"] != "not_in_wh_peaks"])
            A = np.vstack([xs_cur, np.ones_like(xs_cur)]).T
            (slope_cur, intercept_cur), *_ = np.linalg.lstsq(A, ys_cur, rcond=None)
            ax.plot(x_line, slope_cur * x_line + intercept_cur, "-", color="C0",
                   lw=1.5, label=f"wh_peaks fit (R2={wh_current['wh_r2']:.3f})")

        # 虚线:假设纳入全部10点的回归(纯对比,不代表决定)
        if np.isfinite(wh_all10["wh_r2"]):
            xs_a = np.array([p["x"] for p in points])
            ys_a = np.array([p["y"] for p in points])
            A = np.vstack([xs_a, np.ones_like(xs_a)]).T
            (slope_a, intercept_a), *_ = np.linalg.lstsq(A, ys_a, rcond=None)
            ax.plot(x_line, slope_a * x_line + intercept_a, "--", color="gray",
                   lw=1.2, label=f"all-10 fit (R2={wh_all10['wh_r2']:.3f}, diagnostic only)")

        ax.set_title(sid)
        ax.set_xlabel("4 sin(theta)")
        ax.set_ylabel("beta * cos(theta)")
        ax.legend(fontsize=7, loc="best")

        summary_rows.append(dict(
            sample=sid,
            wh_r2_current=wh_current["wh_r2"], wh_n_current=wh_current["wh_n_peaks"],
            wh_r2_all10=wh_all10["wh_r2"], wh_n_all10=wh_all10["wh_n_peaks"],
        ))

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    print(f"[WH诊断] 组图已存: {OUT_PNG}")
    print(f"{'sample':8s} {'R2(wh_peaks,7pt)':18s} {'R2(全10点)':14s}")
    for r in summary_rows:
        print(f"{r['sample']:8s} {r['wh_r2_current']:.4f} (n={r['wh_n_current']:2d})    "
              f"{r['wh_r2_all10']:.4f} (n={r['wh_n_all10']:2d})")


if __name__ == "__main__":
    main()
