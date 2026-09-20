# -*- coding: utf-8 -*-
"""02c_make_arbitration_panels.py —— 一次颗粒可分割性人工仲裁图版（2026-07-22）

目的
----
`judge_acf_scale_um_min=0.30` 与 `judge_edge_density_max=0.115` 两个阈值是在
**含信息栏伪影**的图上标定的。裁剪修复后两个指标整体偏移（acf 平均降 0.04、
edge 平均升 0.009），10 个样本判定翻转、可分割样本数 19→9。阈值必须重标定。

本脚本为重标定准备 ground truth：按纹理评分 z 分层抽样，把 10000× 图整理成
**盲评图版**，由人工逐张判断"一次颗粒能否逐颗分辨"。

盲评设计（重要）
----------------
图版只标注编号（P01、P02…），**不显示样本号，也不显示 acf / edge 数值**。
若显示指标值，判断会被现有数值锚定，标定就成了循环论证。样本号同样隐藏——
C21 = 最高热预算这类先验会直接暗示答案。编号顺序已随机打乱。

对照关系写在 `_key_勿先看.csv`，评分完成后再用它做联结。

输出
----
data/interim/sem_arbitration_panels/
  ├── P01.png … Pnn.png      每样本一张:上排 3 幅 10000× 全图,下排对应中心放大
  ├── 评分表.csv              待填:panel, verdict(Y/N/?), notes
  └── _key_勿先看.csv         panel → sample_id / acf / edge / 现行判定

用法
----
python scripts/02c_make_arbitration_panels.py            # 全量
python scripts/02c_make_arbitration_panels.py --limit 6  # 分批(受时限约束的环境)
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfm.data_processing import sem_processor as sp  # noqa: E402

OUT_DIR = ROOT / "data" / "interim" / "sem_arbitration_panels"
KEY_CSV = OUT_DIR / "_key_勿先看.csv"
SHEET_CSV = OUT_DIR / "评分表.csv"
METRICS_CSV = ROOT / "data" / "interim" / "sem_texture_infobar_check.csv"
N_PANELS = 18
SEED = 20260722


def select_samples() -> pd.DataFrame:
    """按纹理评分 z 分层抽样:两端各取 2 个作锚点,其余在中段等距抽。

    z = zscore(acf) - zscore(edge),两指标相关系数 -0.894,故一维评分已能
    很好地排序。中段加密是因为新阈值大概率落在那里。
    """
    d = pd.read_csv(METRICS_CSV)
    d["z"] = ((d.acf_new - d.acf_new.mean()) / d.acf_new.std()
              - (d.edge_new - d.edge_new.mean()) / d.edge_new.std())
    d = d.sort_values("z", ascending=False).reset_index(drop=True)

    anchors = pd.concat([d.head(2), d.tail(2)])
    mid = d.iloc[2:-2]
    idx = np.linspace(0, len(mid) - 1, N_PANELS - len(anchors)).round().astype(int)
    sel = pd.concat([anchors, mid.iloc[np.unique(idx)]]).drop_duplicates("sample_id")

    rng = np.random.default_rng(SEED)
    sel = sel.iloc[rng.permutation(len(sel))].reset_index(drop=True)
    sel["panel"] = [f"P{i + 1:02d}" for i in range(len(sel))]
    return sel


def make_panel(sample_id: str, panel: str) -> bool:
    sub = ROOT / "data" / "raw" / "sem" / sample_id
    files = [f for f in sorted(sub.glob("*.tif")) if _is_10kx(f)]
    if not files:
        warnings.warn(f"{sample_id}: 无 10000× 图,跳过")
        return False
    files = files[:3]

    fig, axes = plt.subplots(2, len(files), figsize=(5.2 * len(files), 7.4))
    axes = np.atleast_2d(axes)
    for j, f in enumerate(files):
        px, _ = sp._read_pixel_size_um(f)
        img, _ = sp._crop_info_bar(sp._read_image_gray(f), sp.SegConfig())
        H, W = img.shape
        axes[0, j].imshow(img, cmap="gray")
        axes[0, j].set_title(f"full frame {j + 1}", fontsize=11)
        _scalebar(axes[0, j], W, H, px, 2.0)

        # 中心 1/3 放大
        z = img[H // 3: 2 * H // 3, W // 3: 2 * W // 3]
        axes[1, j].imshow(z, cmap="gray")
        axes[1, j].set_title(f"center zoom {j + 1}", fontsize=11)
        _scalebar(axes[1, j], z.shape[1], z.shape[0], px, 0.5)

    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    # 标注一律用 ASCII:matplotlib 默认字体不含 CJK,中文会渲染成方块,
    # 且此脚本可能在未装中文字体的机器上重跑。
    fig.suptitle(f"{panel}   10000x  —  are primary particles individually resolvable?",
                 fontsize=15, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_DIR / f"{panel}.png", dpi=130)
    plt.close(fig)
    return True


def _is_10kx(f: Path) -> bool:
    try:
        return abs(sp._read_mag(f) - 10000.0) / 10000.0 < sp.MAG_RTOL
    except Exception:
        return False


def _scalebar(ax, W, H, px_um, bar_um):
    n_px = bar_um / px_um
    x0, y0 = W * 0.04, H * 0.94
    ax.plot([x0, x0 + n_px], [y0, y0], lw=3, color="yellow")
    ax.text(x0, y0 - H * 0.025, f"{bar_um:g} µm", color="yellow", fontsize=10)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    args = sys.argv[1:]
    limit = int(args[args.index("--limit") + 1]) if "--limit" in args else None

    sel = select_samples()
    sel.to_csv(KEY_CSV, index=False, encoding="utf-8-sig")

    n = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _, r in sel.iterrows():
            if (OUT_DIR / f"{r.panel}.png").exists() and "--force" not in args:
                continue
            if limit is not None and n >= limit:
                break
            if make_panel(r.sample_id, r.panel):
                n += 1
                print(f"[panel] {r.panel}  <- {r.sample_id}", flush=True)

    sheet = pd.DataFrame({
        "panel": sel.panel,
        "verdict": "",          # Y = 可逐颗分辨 / N = 不可 / ? = 拿不准
        "confidence": "",       # 高 / 中 / 低
        "notes": "",
    }).sort_values("panel")
    if not SHEET_CSV.exists():
        sheet.to_csv(SHEET_CSV, index=False, encoding="utf-8-sig")
    done = len(list(OUT_DIR.glob("P*.png")))
    print(f"\n图版 {done}/{len(sel)} 张,输出目录 {OUT_DIR}")


if __name__ == "__main__":
    main()
