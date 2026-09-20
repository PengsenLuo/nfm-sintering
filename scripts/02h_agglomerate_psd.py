# -*- coding: utf-8 -*-
"""02h_agglomerate_psd.py —— SEM 团聚体粒度分布（500× 定稿口径，2026-07-23）

用定稿的形貌自适应分割（sec_adaptive_h=True，Otsu + clear_border）抽取全部
有 500× 数据样本的**逐颗粒**等效圆直径，得到"SEM 团聚体粒度分布"。

这是与激光"分散单元分布"并列的独立口径：
  - 激光 PSD：无水乙醇分散后测得，团聚体被打散成构成单元（D50≈2–4µm）
  - SEM PSD：干态、烧结态的完整团聚体（D50≈6–16µm）
两者之比 disp_ratio = SEM_volD50 / laser_D50 ≈ 一个团聚体由多少分散单元构成，
即"团聚强度"，服务形貌记忆叙事。

口径与已知边界（诚实标注）：
  - 体积加权分位（D^3）与激光同口径；报 volD50 稳（bootstrap 误差小），
    Span/volD90 尾部靠少数大颗粒、仅供参考。
  - 立体学偏差（2D 投影/遮挡/片状平躺）使绝对值为"SEM 等效圆径口径"，
    组间比较与趋势不受影响。

用法：
  python scripts/02h_agglomerate_psd.py --extract [--limit N]   # 断点续跑抽逐颗粒径
  python scripts/02h_agglomerate_psd.py --report                # 汇总+出图（抽完后）
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfm.data_processing import sem_processor as sp  # noqa: E402

PARTICLES = ROOT / "data" / "interim" / "sem_agglomerate_psd_particles_all.csv"
SUMMARY = ROOT / "data" / "interim" / "sem_agglomerate_psd_summary_all.csv"


def _cfg():
    return sp.config_from_yaml(yaml.safe_load(open(ROOT / "configs" / "config.yaml")))


def extract(limit: int) -> None:
    cfg = _cfg()
    done = pd.read_csv(PARTICLES) if PARTICLES.exists() else pd.DataFrame()
    have = set(done.sample_id.unique()) if len(done) else set()
    chunks = [done] if len(done) else []
    n = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sub in sorted(p for p in (ROOT / "data/raw/sem").iterdir() if p.is_dir()):
            if not sp.SAMPLE_ID_RE.match(sub.name) or sub.name in have:
                continue
            if n >= limit:
                break
            low, _ = sp._collect_mag_files(sub)
            if not low:
                continue
            ds = []
            for f in low:
                px, _ = sp._read_pixel_size_um(f)
                img, _ = sp._crop_info_bar(sp._read_image_gray(f), cfg.seg)
                p, _l = sp._segment_secondary(img, px, cfg.seg)
                ds += list(p["equiv_diam_um"].values)
            chunks.append(pd.DataFrame(
                {"sample_id": sub.name, "precursor": sub.name[-1], "d_um": ds}))
            n += 1
            print(f"[psd] {sub.name}  n={len(ds)}", flush=True)
    out = pd.concat(chunks, ignore_index=True)
    out.to_csv(PARTICLES, index=False)
    print(f"累计样本 {out.sample_id.nunique()} / 51")


def _vq(x, q):
    x = np.sort(x); w = x ** 3
    return x[np.searchsorted(np.cumsum(w) / w.sum(), q / 100)]


def _boot_sd(x, q, B=300):
    r = np.random.default_rng(0)
    return float(np.std([_vq(r.choice(x, len(x), replace=True), q) for _ in range(B)]))


def report() -> None:
    from scipy import stats
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = pd.read_csv(PARTICLES)
    mt = pd.read_csv(ROOT / "data/processed/master_table.csv")
    theta = mt.set_index("sample_id").Theta.to_dict()
    laser = mt.set_index("sample_id").D50.to_dict()

    rows = []
    for sid, g in d.groupby("sample_id"):
        x = g.d_um.values
        rows.append(dict(
            sample_id=sid, precursor=sid[-1], condition=sid[:3],
            Theta=theta.get(sid, np.nan), n=len(x),
            numD50=np.median(x), volD10=_vq(x, 10), volD50=_vq(x, 50),
            volD90=_vq(x, 90), volD50_sd=_boot_sd(x, 50),
            laserD50=laser.get(sid, np.nan)))
    r = pd.DataFrame(rows)
    r["Span"] = (r.volD90 - r.volD10) / r.volD50
    r["disp_ratio"] = r.volD50 / r.laserD50
    r = r.sort_values(["precursor", "Theta"])
    r.to_csv(SUMMARY, index=False)

    print(f"样本数 {len(r)}   每样颗粒数中位 {r.n.median():.0f}\n")
    print("Spearman(volD50, Θ) 分前驱体：")
    for p in "SML":
        s = r[r.precursor == p]
        rho, pv = stats.spearmanr(s.Theta, s.volD50)
        print(f"  {p}: rho={rho:+.2f} p={pv:.3f} (n={len(s)})  "
              f"volD50 {s.volD50.min():.1f}–{s.volD50.max():.1f} µm")
    print("\n分散比 SEM/激光 分前驱体：")
    print(r.groupby("precursor").disp_ratio.agg(["mean", "median", "min", "max"]).round(2).to_string())

    # C19/C21 悬案核查：这两个样的 disp_ratio 是否离群
    print("\n=== M_D 悬案核查：C19/C21 的分散比是否异常 ===")
    for cond in ["C19", "C21"]:
        sub = r[r.condition == cond]
        if len(sub):
            print(f"{cond}:")
            print(sub[["sample_id", "volD50", "laserD50", "disp_ratio", "n"]].round(2).to_string(index=False))
    for p in "SML":
        v = r[r.precursor == p].disp_ratio
        print(f"  {p} 组 disp_ratio 分布: 均值 {v.mean():.2f}  95%区间 [{v.quantile(.05):.2f}, {v.quantile(.95):.2f}]")

    _plot(d, r, plt)
    print(f"\n图 → {ROOT/'data/interim/sem_agglomerate_psd_all.png'}")


def _plot(d, r, plt):
    col = {"S": "#e63946", "M": "#457b9d", "L": "#2a9d8f"}
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.6))
    for p in "SML":
        s = r[r.precursor == p]
        ax[0].errorbar(s.Theta, s.volD50, yerr=s.volD50_sd, fmt="o", color=col[p],
                       label=f"{p} SEM agglomerate", capsize=2, alpha=0.8)
        if len(s) >= 4:
            z = np.polyfit(s.Theta, s.volD50, 1)
            xs = np.linspace(s.Theta.min(), s.Theta.max(), 50)
            ax[0].plot(xs, np.polyval(z, xs), color=col[p], lw=1.3, alpha=0.6)
        ax[1].scatter(s.Theta, s.disp_ratio, color=col[p], label=p, s=40, alpha=0.8)
    ax[0].set_xlabel("Θ (h)"); ax[0].set_ylabel("SEM agglomerate volD50 (µm)")
    ax[0].set_title("Agglomerate size vs Θ (all 51, adaptive-h seg)")
    ax[0].legend(); ax[0].grid(alpha=.25)
    ax[1].set_xlabel("Θ (h)"); ax[1].set_ylabel("dispersion ratio  SEM/laser")
    ax[1].set_title("Dispersion (agglomeration) strength vs Θ")
    ax[1].legend(); ax[1].grid(alpha=.25)
    for cond, mk in [("C19", "C19"), ("C21", "C21")]:
        sub = r[r.condition == cond]
        for _, row in sub.iterrows():
            ax[1].annotate(cond, (row.Theta, row.disp_ratio), fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / "data/interim/sem_agglomerate_psd_all.png", dpi=140)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--extract" in args:
        lim = int(args[args.index("--limit") + 1]) if "--limit" in args else 99
        extract(lim)
    if "--report" in args:
        report()
