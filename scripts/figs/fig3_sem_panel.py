# -*- coding: utf-8 -*-
"""fig3:SEM 500× 图版,C01-S(Θ=4.2,低热预算)vs C21-S(Θ=47.3,高热预算)。
论文 §3.2 图3:低 Θ 为独立近球形团聚体,高 Θ 为棱角化、相互熔连的块体
(原图即可见,非分割伪影)——本图只做信息栏裁剪+并排展示,不做任何分割/
标注,避免用处理后的痕迹替代"原图即可见"这一论断的证据力。

**选图依据**:两样本文件夹内 500× 各有 3 张(见 `_collect_mag_files` 协议:
每倍率 3 张)。用 `nfm.data_processing.sem_processor._read_mag`/
`_read_pixel_size_um` 逐张核实,C01-S 与 C21-S 的 500× 像素尺寸完全一致
(0.2233 µm/px,同一 CZ_SEM 视场标定,说明拍摄条件/仪器设置相同)。
两样本各自序列中的第 1 张 500× 图(C01-S-04.tif / HS20-2-04.tif,均为
每样本 12 张按 10000×/5000×/2000×/500× 循环排列中的第 4 张,即该循环
第 1 组的 500× 位)在序列位置上互相对应,选用作为代表图。

**C21-S 文件夹命名说明**:该文件夹内除 1 张新命名 `C21-S_5000x_01.tif`
外,其余 11 张沿用旧命名 `HS20-2-*.tif`;样本归属按 `sem_processor.py`
文档化的规则"仅由文件夹决定"(不依赖文件名),已用元数据核实这批文件的
像素尺寸标定与 C01-S 完全同源自洽,视为同一批次正常拍摄文件,非误归属。
"""
import _bootstrap  # noqa

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _style import save_fig, skip
from nfm.data_processing.sem_processor import (
    _read_image_gray, _crop_info_bar, _read_pixel_size_um, SegConfig,
)

NAME = "fig3_sem_panel"

SAMPLES = [
    ("C01-S", "data/raw/sem/C01-S/C01-S-04.tif", 4.2),
    ("C21-S", "data/raw/sem/C21-S/HS20-2-04.tif", 47.3),
]


def _add_scalebar(ax, px_um: float, img_width_px: int, bar_um: float = 10.0):
    bar_px = bar_um / px_um
    x0 = img_width_px * 0.05
    y0 = ax.get_ylim()[0] * 0.96  # 图像坐标 y 轴已翻转(imshow),取靠近底部
    h = img_width_px * 0.012
    ax.add_patch(Rectangle((x0, y0 - h * 3), bar_px, h, color="white",
                           ec="black", linewidth=0.5, zorder=5))
    ax.text(x0 + bar_px / 2, y0 - h * 3.6, f"{bar_um:.0f} µm", color="white",
           fontsize=9, ha="center", va="bottom", zorder=5,
           path_effects=_text_outline())


def _text_outline():
    from matplotlib import patheffects
    return [patheffects.withStroke(linewidth=2, foreground="black")]


def main():
    cfg = SegConfig()
    imgs = []
    for label, path, theta in SAMPLES:
        p = Path(path)
        if not p.exists():
            skip(NAME, f"{path} 不存在")
            return
        img = _read_image_gray(p)
        img_c, cut = _crop_info_bar(img, cfg)
        px_um, src = _read_pixel_size_um(p)
        imgs.append((label, img_c, px_um, theta, cut, img.shape[0]))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.4))
    for ax, (label, img_c, px_um, theta, cut, h_full) in zip(axes, imgs):
        ax.imshow(img_c, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{label}  (Θ={theta:.1f} h)", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        _add_scalebar(ax, px_um, img_c.shape[1])

    fig.suptitle("")  # 无标题,标题信息放图注
    fig.tight_layout()

    n_removed = [f"{lbl}:切除信息栏{h_full - img_c.shape[0]}行/{h_full}行"
                for lbl, img_c, _, _, _, h_full in imgs]
    caption = (
        "**数据**:500× SEM 原图,`data/raw/sem/C01-S/C01-S-04.tif`"
        "(Θ=4.2 h,最低热预算 C07 之外的低 Θ 代表点)vs "
        "`data/raw/sem/C21-S/HS20-2-04.tif`(Θ=47.3 h,最高热预算)。"
        "两图像素标定一致(0.2233 µm/px,`sem_processor._read_pixel_size_um`"
        "读 CZ_SEM 元数据 tag 34118),用 `sem_processor._crop_info_bar` 去除"
        "底部信息栏(" + "; ".join(n_removed) + "),**不做任何分割/伪彩标注**——"
        "呼应论文 §3.2 的论断\"低 Θ 为独立近球形团聚体、高 Θ 为棱角化相互熔连"
        "的块体,原图即可见,非分割伪影\",让读者看到未经处理的对比。白色比例尺"
        "=10 µm。"
    )
    save_fig(fig, NAME, caption)
    plt.close(fig)


if __name__ == "__main__":
    main()
