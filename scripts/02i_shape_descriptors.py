# -*- coding: utf-8 -*-
"""02i_shape_descriptors.py —— 形状机制分解描述子（2026-07-25）

动机
----
`circularity` 只能说"不圆了"，但不圆有三种物理上完全不同的成因：
  (a) 枝杈化/分叉（颈缩聚集成珊瑚状网络）—— 本项目的机制假说
  (b) 表面粗糙（毛刺、析出物）
  (c) 整体拉长（棒状/板状）
circularity 把三者混为一个数，无法判别。本脚本给出可分离三者的描述子组：

| 描述子 | 定义 | 主要敏感于 |
|---|---|---|
| `D_f` 轮廓分形维数 | 多尺度周长标度（"尺子越短量得越长"的速率） | (a)+(b) 轮廓曲折度 |
| `solidity` | A / A_convex | (a) 凹陷/分叉（枝杈会造成大面积凹口） |
| `elongation` | 1 − minor/major | (c) 拉长 |
| `roughness_ratio` | P / P_convex | (b) 表面细节（凸包周长归一） |

判据：
  枝杈化 → D_f↑ 且 solidity↓ 且 elongation 基本不变
  表面粗糙 → D_f↑ 但 solidity 基本不变
  拉长 → elongation↑ 为主，D_f 变化小

分形维数算法
------------
对每个颗粒轮廓做 **多尺度周长标度**（等价于 divider/Richardson 法）：
以步长 s（像素）沿轮廓重采样并累加折线长度 P(s)，拟合
    log P(s) = (1 − D_f)·log s + c   →   D_f = 1 − slope
s 取几何级数（默认 1–16 px，取轮廓长度允许的范围）。

已知局限（诚实标注，必须写进论文方法节）
--------------------------------------
1. 这是 **2D 投影轮廓** 的分形，不是真实 3D 表面分形；
2. D_f 对图像分辨率与分割质量敏感 → **只做组间/趋势比较，绝对值不过度解读**；
3. 小颗粒可用标度范围窄，故设最小周长阈值（默认 60 px）过滤，
   否则拟合不稳；被过滤比例随样本报告。

用法：
  python scripts/02i_shape_descriptors.py --extract [--limit N]
  python scripts/02i_shape_descriptors.py --report
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfm.data_processing import sem_processor as sp  # noqa: E402

OUT = ROOT / "data" / "interim" / "sem_shape_descriptors.csv"
MIN_PERIM_PX = 60.0          # 周长下限：低于此值标度范围不足，拟合不稳
SCALES = np.array([1, 2, 3, 4, 6, 8, 12, 16], dtype=float)


def _seg_config_hash(cfg) -> str:
    """SegConfig 的稳定短哈希,用作 extract() 增量抽取的缓存键(2026-08-29
    根因修复:发现旧版
    `have = set(done.sample_id.unique())` 只按 sample_id 判断"已处理",
    分割参数(SegConfig,如 sec_adaptive_h 定档)跨轮改变后,早期已写入
    CSV 的样本从未被重新分割,导致本文件与生产 D_sec 管线(sem_features.csv,
    同样调用 _segment_secondary,但每次全量重跑不做增量)逐样本静默漂移,
    最大达 0.42 µm/单样本 132 颗粒差(C24-S)。哈希纳入缓存键后,任何一个
    SegConfig 字段变化都会让全部旧行整体失效,强制重新分割,而不是像之前
    那样只在"这个 sample_id 从未出现过"时才处理。"""
    payload = json.dumps(dataclasses.asdict(cfg.seg), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _valid_cached_rows(done: pd.DataFrame, current_hash: str) -> pd.DataFrame:
    """`done`(上次 extract() 写盘的旧结果)中在当前 SegConfig 哈希下仍然
    有效的行。缺少 `seg_config_hash` 列的旧文件(本修复之前产出的)一律视为
    全部失效——它们的分割参数无法追溯,不能假定与当前配置一致。"""
    if done.empty or "seg_config_hash" not in done.columns:
        return done.iloc[0:0]
    return done[done["seg_config_hash"] == current_hash]


def _contour_fractal_dim(contour: np.ndarray) -> tuple[float, int]:
    """多尺度周长标度求轮廓分形维数。返回 (D_f, 用于拟合的标度点数)。"""
    # 闭合轮廓
    c = np.vstack([contour, contour[:1]])
    seg = np.sqrt(((np.diff(c, axis=0)) ** 2).sum(axis=1))
    total = seg.sum()
    if total < MIN_PERIM_PX:
        return np.nan, 0
    # 弧长参数化后按步长 s 重采样
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    logs, logp = [], []
    for s in SCALES:
        if total / s < 8:            # 至少 8 段才有意义
            continue
        t = np.arange(0.0, total, s)
        xs = np.interp(t, cum, c[:, 0])
        ys = np.interp(t, cum, c[:, 1])
        pts = np.column_stack([xs, ys])
        pts = np.vstack([pts, pts[:1]])
        L = np.sqrt(((np.diff(pts, axis=0)) ** 2).sum(axis=1)).sum()
        if L > 0:
            logs.append(np.log(s)); logp.append(np.log(L))
    if len(logs) < 4:
        return np.nan, len(logs)
    slope = np.polyfit(logs, logp, 1)[0]
    return float(1.0 - slope), len(logs)


def _descriptors_for_image(img, px, cfg) -> pd.DataFrame:
    from skimage import measure
    from scipy import ndimage as ndi
    _props, labels = sp._segment_secondary(img, px, cfg.seg)
    H, W = labels.shape
    rows = []
    for r in measure.regionprops(labels):
        # 与主管线口径一致：剔除触边、过滤小于 sec_min_diam_um
        minr, minc, maxr, maxc = r.bbox
        if cfg.seg.sec_clear_border and (minr <= 0 or minc <= 0 or maxr >= H or maxc >= W):
            continue
        d_um = r.equivalent_diameter * px
        if d_um < cfg.seg.sec_min_diam_um:
            continue
        cs = measure.find_contours(r.image.astype(float), 0.5)
        if not cs:
            continue
        cont = max(cs, key=len)
        df, nsc = _contour_fractal_dim(cont)
        conv_p = measure.perimeter(r.convex_image)
        rows.append(dict(
            d_um=d_um,
            D_f=df,
            solidity=r.area / max(r.convex_area, 1),
            elongation=1.0 - (r.minor_axis_length / max(r.major_axis_length, 1e-9)),
            roughness_ratio=r.perimeter / max(conv_p, 1e-9),
            n_scales=nsc,
        ))
    return pd.DataFrame(rows)


def extract(limit: int) -> None:
    cfg = sp.config_from_yaml(yaml.safe_load(open(ROOT / "configs/config.yaml", encoding="utf-8")))
    current_hash = _seg_config_hash(cfg)
    done_raw = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    done = _valid_cached_rows(done_raw, current_hash)
    if len(done_raw):
        n_stale = done_raw["sample_id"].nunique() - (done["sample_id"].nunique() if len(done) else 0)
        if n_stale:
            print(f"[shape] SegConfig hash={current_hash} differs from "
                 f"{n_stale} previously-cached sample(s)' recorded hash "
                 f"(or file predates hash tracking) -- invalidating and "
                 f"will reprocess them", flush=True)
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
            per = []
            for f in low:
                px, _ = sp._read_pixel_size_um(f)
                img, _ = sp._crop_info_bar(sp._read_image_gray(f), cfg.seg)
                per.append(_descriptors_for_image(img, px, cfg))
            df = pd.concat(per, ignore_index=True)
            df.insert(0, "precursor", sub.name[-1])
            df.insert(0, "sample_id", sub.name)
            df["seg_config_hash"] = current_hash
            chunks.append(df)
            n += 1
            ok = df.D_f.notna().mean() * 100
            print(f"[shape] {sub.name}  n={len(df)}  D_f可用={ok:.0f}%  "
                  f"中位 D_f={df.D_f.median():.3f}", flush=True)
    pd.concat(chunks, ignore_index=True).to_csv(OUT, index=False)
    tot = pd.concat(chunks, ignore_index=True)
    print(f"累计 {tot.sample_id.nunique()} / 81 样本")


def report() -> None:
    from scipy import stats
    d = pd.read_csv(OUT)
    mt = pd.read_csv(ROOT / "data/processed/master_table.csv")
    agg = (d.dropna(subset=["D_f"]).groupby("sample_id")
             .agg(D_f=("D_f", "median"), solidity=("solidity", "median"),
                  elongation=("elongation", "median"),
                  roughness=("roughness_ratio", "median"),
                  n_particles=("D_f", "size")).reset_index())
    m = agg.merge(mt[["sample_id", "precursor", "Theta", "T_C",
                      "compaction_density", "circularity"]], on="sample_id")
    m.to_csv(ROOT / "data/interim/sem_shape_descriptors_summary.csv", index=False)

    print(f"样本数 {len(m)}   每样中位颗粒数 {m.n_particles.median():.0f}\n")
    print("=== 机制判别：各描述子 vs Θ（分前驱体） ===")
    print(f"{'描述子':<14}{'S':>22}{'M':>22}{'L':>22}")
    for c in ["D_f", "solidity", "elongation", "roughness", "circularity"]:
        line = f"{c:<14}"
        for p in "SML":
            s = m[m.precursor == p].dropna(subset=[c, "Theta"])
            r, pv = stats.spearmanr(s.Theta, s[c])
            line += f"{r:>+8.3f}(p={pv:.3f})  "
        print(line)
    print("\n=== 判别规则 ===")
    print("  枝杈化   → D_f↑  solidity↓  elongation~不变")
    print("  表面粗糙 → D_f↑  solidity~不变")
    print("  拉长     → elongation↑ 为主")
    print("\n=== D_f 与压实密度的关联（枝杈架桥假说） ===")
    s = m.dropna(subset=["D_f", "compaction_density"])
    print("  全局: rho=%+.3f p=%.2g" % stats.spearmanr(s.D_f, s.compaction_density))
    for p in "SML":
        t = s[s.precursor == p]
        print("    %s 组内: rho=%+.3f p=%.3f (n=%d)"
              % (p, *stats.spearmanr(t.D_f, t.compaction_density), len(t)))
    print("\n=== 分前驱体均值 ===")
    print(m.groupby("precursor")[["D_f", "solidity", "elongation", "roughness"]]
           .mean().round(4).to_string())


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--extract" in args:
        lim = int(args[args.index("--limit") + 1]) if "--limit" in args else 999
        extract(lim)
    if "--report" in args:
        report()
