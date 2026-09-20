# -*- coding: utf-8 -*-
"""02b_sem_infobar_diagnostic.py —— 信息栏裁剪修复的前后对照诊断(2026-07-22)

只跑 500× 二次颗粒线(不加载 SAM/torch),对全部有 SEM 数据的样本各跑两遍:
  - OLD: 复刻修复前的 _crop_info_bar 行为(自底向上,遇首个近黑行即停)
  - NEW: 当前 sem_processor._crop_info_bar(取最靠上的合格暗带起始行)

输出:
  data/interim/sem_infobar_diagnostic.csv   逐样本前后对照
  控制台                                     S/M/L 分组汇总与 L/S 比值

用法: python scripts/02b_sem_infobar_diagnostic.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfm.data_processing import sem_processor as sp  # noqa: E402


def crop_old(img: np.ndarray, cfg: sp.SegConfig) -> np.ndarray:
    """修复前的实现,仅用于对照。"""
    rowmean = img.mean(axis=1)
    H = img.shape[0]
    cut, in_band = H, False
    for r in range(H - 1, int(H * cfg.infobar_min_keep_frac), -1):
        if rowmean[r] < cfg.infobar_dark_thresh:
            cut, in_band = r, True
        elif in_band:
            break
    return img[:cut, :]


def run_one(sub: Path, cfg: sp.SemProcessorConfig) -> dict | None:
    low, _ = sp._collect_mag_files(sub)
    if not low:
        return None
    rec = {"sample_id": sub.name, "precursor": sub.name[-1], "n_img_500x": len(low)}
    for tag, cropper in (("old", crop_old), ("new", lambda i, c: sp._crop_info_bar(i, c)[0])):
        props, heights = [], []
        for f in low:
            px, _ = sp._read_pixel_size_um(f)
            img = cropper(sp._read_image_gray(f), cfg.seg)
            heights.append(img.shape[0])
            p, _labels = sp._segment_secondary(img, px, cfg.seg)
            props.append(p)
        agg = sp._aggregate_secondary(pd.concat(props, ignore_index=True))
        for k, v in agg.items():
            rec[f"{k}_{tag}"] = v
        rec[f"img_h_{tag}"] = int(np.mean(heights))
    return rec


OUT_CSV = ROOT / "data" / "interim" / "sem_infobar_diagnostic.csv"


def main() -> None:
    """支持断点续跑:已在 OUT_CSV 中的样本跳过,新结果追加写回。

    可选参数 --limit N 只处理 N 个未完成样本(用于受时限约束的环境分批执行);
    不带参数则一次跑完全部。--report 只汇总不计算。
    """
    args = sys.argv[1:]
    limit = int(args[args.index("--limit") + 1]) if "--limit" in args else None
    report_only = "--report" in args

    done = pd.read_csv(OUT_CSV) if OUT_CSV.exists() else pd.DataFrame()
    rows = done.to_dict("records") if len(done) else []
    have = set(done["sample_id"]) if len(done) else set()

    if not report_only:
        cfg = sp.SemProcessorConfig(save_overlays=False)
        raw = ROOT / "data" / "raw" / "sem"
        n_new = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for sub in sorted(p for p in raw.iterdir() if p.is_dir()):
                if not sp.SAMPLE_ID_RE.match(sub.name) or sub.name in have:
                    continue
                if limit is not None and n_new >= limit:
                    break
                r = run_one(sub, cfg)
                if r:
                    rows.append(r)
                    n_new += 1
                    print(f"[SEM-500x] {sub.name}  D_sec {r['D_sec_old']:.2f} -> {r['D_sec_new']:.2f}  "
                          f"n_sec {r['n_sec_old']:.0f} -> {r['n_sec_new']:.0f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    df = pd.DataFrame(rows)
    print(f"\n写出 {OUT_CSV}  (累计 {len(df)} 样本)\n")

    cols = ["D_sec", "n_sec", "circularity", "convexity", "aspect_ratio"]
    print("=" * 78)
    print("S/M/L 分组均值(old = 含信息栏 / new = 已裁剪)")
    print("=" * 78)
    g = df.groupby("precursor")
    for c in cols:
        print(f"\n{c}")
        tbl = pd.DataFrame({"old": g[f"{c}_old"].mean(), "new": g[f"{c}_new"].mean()}).round(3)
        print(tbl.to_string())

    print("\n" + "=" * 78)
    print("D_sec 的 L/S 比值(物理上应 > 1:大前驱体应给出更大的二次颗粒)")
    for tag in ("old", "new"):
        m = g[f"D_sec_{tag}"].mean()
        print(f"  {tag}: L/S = {m.get('L', np.nan) / m.get('S', np.nan):.3f}   "
              f"(S={m.get('S', np.nan):.2f}  M={m.get('M', np.nan):.2f}  L={m.get('L', np.nan):.2f})")
    print("=" * 78)


if __name__ == "__main__":
    main()
