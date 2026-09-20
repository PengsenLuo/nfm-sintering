# -*- coding: utf-8 -*-
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage as ndi
from skimage.draw import disk

from nfm.data_processing import sem_processor as sp


def _props(diams):
    return pd.DataFrame({"equiv_diam_um": diams})


# ---------------------------------------------------------------- 信息栏裁剪
# 回归测试:2026-07-22 修复。旧实现假设信息栏为暗条,遇到本批 Zeiss 的
# "亮条 + 上下近黑分隔线" 版式时只裁掉 2 行,白条残留污染分割(见 _crop_info_bar docstring)。

_CFG = sp.SegConfig()


def _img_bright_infobar(H=768, W=1024, sep_top=690, sep_bot=766):
    """本批真实版式:图像区 → 近黑分隔线 → 亮信息栏 → 近黑分隔线 → 1 行余白。"""
    img = np.full((H, W), 90, dtype=np.uint8)
    img[sep_top, :] = 0
    img[sep_top + 1:sep_bot, :] = 240
    img[sep_bot, :] = 0
    return img


def test_crop_info_bar_bright_bar_cuts_at_top_separator():
    img = _img_bright_infobar()
    out, cut = sp._crop_info_bar(img, _CFG)
    assert cut == 690                      # 切在最靠上的分隔线,而非最靠下
    assert out.shape[0] == 690
    assert out.max() < 240                 # 白条已完全移除


def test_crop_info_bar_dark_bar_still_supported():
    """信息栏本身为暗条(贴底边的厚暗带)时仍应正确裁剪。"""
    img = np.full((768, 1024), 90, dtype=np.uint8)
    img[700:, :] = 0
    out, cut = sp._crop_info_bar(img, _CFG)
    assert cut == 700
    assert out.shape[0] == 700


def test_crop_info_bar_no_bar_warns_and_keeps_image():
    img = np.full((768, 1024), 90, dtype=np.uint8)
    with pytest.warns(UserWarning, match="未找到近黑分隔行"):
        out, cut = sp._crop_info_bar(img, _CFG)
    assert cut == 768
    assert out.shape[0] == 768


def _labels_with_border_particle():
    """3 个内部颗粒 + 1 个贴左边界的颗粒。"""
    lab = np.zeros((100, 100), dtype=np.int32)
    lab[20:30, 20:30] = 1
    lab[20:30, 50:60] = 2
    lab[60:70, 40:50] = 3
    lab[40:50, 0:10] = 4          # 触左边界
    return lab


def test_regionprops_clear_border_drops_edge_particles():
    lab = _labels_with_border_particle()
    keep = sp._regionprops_um(lab, 1.0, 0.0, clear_border=False)
    drop = sp._regionprops_um(lab, 1.0, 0.0, clear_border=True)
    assert len(keep) == 4
    assert len(drop) == 3


def test_regionprops_clear_border_default_is_off():
    """默认关闭,避免影响尚未改口径的一次颗粒线。"""
    lab = _labels_with_border_particle()
    assert len(sp._regionprops_um(lab, 1.0, 0.0)) == 4


def test_regionprops_clear_border_detects_all_four_edges():
    for sl in ((slice(0, 10), slice(40, 50)),      # 上
               (slice(90, 100), slice(40, 50)),    # 下
               (slice(40, 50), slice(0, 10)),      # 左
               (slice(40, 50), slice(90, 100))):   # 右
        lab = np.zeros((100, 100), dtype=np.int32)
        lab[20:30, 20:30] = 1                       # 内部对照
        lab[sl] = 2
        assert len(sp._regionprops_um(lab, 1.0, 0.0, clear_border=True)) == 1


def test_secondary_segmentation_honours_clear_border_flag():
    """SegConfig.sec_clear_border 应真正传导到 _segment_secondary。"""
    rng = np.random.default_rng(0)
    img = (rng.normal(60, 5, (200, 200))).clip(0, 255).astype(np.uint8)
    img[30:60, 30:60] = 220        # 内部亮颗粒
    img[80:110, 0:25] = 220        # 触左边界亮颗粒
    on = sp.SegConfig(sec_min_diam_um=1.0, sec_clear_border=True)
    off = sp.SegConfig(sec_min_diam_um=1.0, sec_clear_border=False)
    p_on, _ = sp._segment_secondary(img, 1.0, on)
    p_off, _ = sp._segment_secondary(img, 1.0, off)
    assert len(p_on) < len(p_off)


def test_crop_info_bar_ignores_dark_image_region():
    """底部区域内的真实暗区(厚且不贴底边)不应被误判为信息栏。"""
    img = np.full((768, 1024), 90, dtype=np.uint8)
    img[500:560, :] = 0          # 图像内真实暗区
    img[730, :] = 0              # 真正的信息栏上分隔线
    img[731:, :] = 240
    out, cut = sp._crop_info_bar(img, _CFG)
    assert cut == 730


def test_aggregate_primary_reports_both_sam_and_classic():
    props_sam = _props([1.0, 1.2, 0.9])
    props_classic = _props([2.0, 2.4, 1.8])
    out = sp._aggregate_primary(props_sam, props_classic, True, {})
    assert out["D_pri_sem"] == pytest.approx(1.0)          # SAM = 正式值
    assert out["n_pri"] == 3
    assert out["D_pri_sem_classic"] == pytest.approx(2.0)   # 经典 = 诊断列
    assert out["n_pri_classic"] == 3


def test_aggregate_primary_not_segmentable_both_nan():
    out = sp._aggregate_primary(None, None, False, {})
    assert np.isnan(out["D_pri_sem"])
    assert out["n_pri"] == 0
    assert np.isnan(out["D_pri_sem_classic"])
    assert out["n_pri_classic"] == 0


def test_aggregate_primary_sam_unavailable_falls_back_classic_for_official_value():
    # SAM 抛异常/不可用时(如 GPU 不可用),官方 D_pri_sem 应回退到经典值而不是 NaN,
    # 但语义上仍要能与"SAM 正常跑出来的值"区分——由调用方在 warnings 里体现,
    # 这里只测数值行为。
    props_classic = _props([2.0, 2.4, 1.8])
    out = sp._aggregate_primary(None, props_classic, True, {})
    assert out["D_pri_sem"] == pytest.approx(2.0)
    assert out["D_pri_sem_classic"] == pytest.approx(2.0)


# ------------------------------------------------------------ 形貌自适应 h(2026-07-23)
# _particle_scale_um: 标签无关、h 无关的特征尺度描述子(前景 dt 稳健高分位 x 像素尺寸),
# 用于驱动 sec_hmaxima_rel 的连续自适应,替代"全库固定一个 h"。

def _packed_disks_mask(shape, radius, spacing):
    """规则栅格铺同尺寸圆盘,模拟"某一特征尺度下的紧密颗粒床"。"""
    mask = np.zeros(shape, dtype=bool)
    for cy in range(radius + 2, shape[0] - radius - 2, spacing):
        for cx in range(radius + 2, shape[1] - radius - 2, spacing):
            rr, cc = disk((cy, cx), radius, shape=shape)
            mask[rr, cc] = True
    return mask


def test_particle_scale_um_monotonic_small_vs_large_particles():
    """小圆盘密堆图 → 小 dt_scale;大圆盘稀疏图 → 大 dt_scale,方向应正确。"""
    small_mask = _packed_disks_mask((120, 120), radius=3, spacing=8)
    large_mask = _packed_disks_mask((240, 240), radius=15, spacing=40)
    small_dt = ndi.distance_transform_edt(small_mask)
    large_dt = ndi.distance_transform_edt(large_mask)
    small_scale = sp._particle_scale_um(small_mask, small_dt, pixel_um=1.0)
    large_scale = sp._particle_scale_um(large_mask, large_dt, pixel_um=1.0)
    assert small_scale < large_scale
    # 填充圆盘的面积集中在靠近边界处,90 分位 EDT 理论上 ≈0.684*R(非 ≈R),
    # 见 P(EDT<=d)=1-((R-d)/R)^2 令其=0.9 解得 d≈0.684R;两个半径均验证吻合
    assert small_scale == pytest.approx(0.684 * 3, abs=0.5)
    assert large_scale == pytest.approx(0.684 * 15, abs=1.5)


def test_particle_scale_um_scales_with_pixel_size():
    mask = _packed_disks_mask((120, 120), radius=5, spacing=14)
    dt = ndi.distance_transform_edt(mask)
    s1 = sp._particle_scale_um(mask, dt, pixel_um=1.0)
    s2 = sp._particle_scale_um(mask, dt, pixel_um=0.5)
    assert s2 == pytest.approx(s1 * 0.5)


def test_particle_scale_um_empty_mask_returns_nan():
    mask = np.zeros((50, 50), dtype=bool)
    dt = ndi.distance_transform_edt(mask)
    assert np.isnan(sp._particle_scale_um(mask, dt, pixel_um=1.0))


def test_map_h_from_scale_low_anchor_and_monotonic():
    cfg = sp.SegConfig(sec_adaptive_h=True)
    h_low = sp._map_h_from_scale(cfg.sec_adaptive_h_x0, cfg)
    h_mid = sp._map_h_from_scale(cfg.sec_adaptive_h_x0 + 0.5, cfg)
    h_high = sp._map_h_from_scale(cfg.sec_adaptive_h_x0 + 1.0, cfg)
    assert h_low == pytest.approx(cfg.sec_adaptive_h_intercept)
    assert h_low < h_mid < h_high


def test_map_h_from_scale_clips_to_bounds():
    cfg = sp.SegConfig(sec_adaptive_h=True)
    assert sp._map_h_from_scale(-100.0, cfg) == pytest.approx(cfg.sec_adaptive_h_min)
    assert sp._map_h_from_scale(1000.0, cfg) == pytest.approx(cfg.sec_adaptive_h_max)


def test_map_h_from_scale_nan_falls_back_to_fixed_hmaxima():
    cfg = sp.SegConfig(sec_adaptive_h=True, sec_hmaxima_rel=0.22)
    assert sp._map_h_from_scale(float("nan"), cfg) == pytest.approx(0.22)


def _synthetic_secondary_image():
    """与捕获 golden 值时完全一致的合成图(固定 rng seed + 固定圆盘坐标),
    任何改动都会使下面的哈希对不上,用于保护 sec_adaptive_h=False 的旧行为。"""
    rng = np.random.default_rng(42)
    img = (rng.normal(60, 5, (200, 200))).clip(0, 255).astype(np.uint8)
    for (cy, cx, r) in [(30, 30, 10), (30, 70, 8), (70, 30, 12), (70, 70, 9),
                        (100, 100, 14), (150, 150, 7), (150, 50, 11)]:
        rr, cc = disk((cy, cx), r, shape=img.shape)
        img[rr, cc] = 220
    return img


def test_adaptive_h_disabled_matches_legacy_pixelwise():
    """sec_adaptive_h=False(默认)必须与重构前逐像素一致 —— golden hash 在
    2026-07-23 加自适应 h 之前、用本函数同样的合成图跑当时的 _segment_secondary
    捕获,任何回归都会改变这个哈希。"""
    img = _synthetic_secondary_image()
    cfg = sp.SegConfig(sec_min_diam_um=1.0, sec_adaptive_h=False)
    props, labels = sp._segment_secondary(img, 1.0, cfg)
    assert len(props) == 7
    assert props["equiv_diam_um"].median() == pytest.approx(19.67396303746374)
    assert hashlib.md5(labels.astype(np.int32).tobytes()).hexdigest() \
        == "3b9a21b6d770163eaaefb744b1ab64b1"


def test_adaptive_h_enabled_changes_h_but_stays_gated_by_flag(monkeypatch):
    """sec_adaptive_h=True 时才会走 _map_h_from_scale;为 False 时绝不应调用,
    确保旧路径完全不受新分支影响(即便新函数本身有 bug 也不会牵连旧行为)。"""
    calls = []
    orig = sp._map_h_from_scale

    def spy(dt_scale, cfg):
        calls.append(dt_scale)
        return orig(dt_scale, cfg)

    monkeypatch.setattr(sp, "_map_h_from_scale", spy)
    img = _synthetic_secondary_image()

    cfg_off = sp.SegConfig(sec_min_diam_um=1.0, sec_adaptive_h=False)
    sp._segment_secondary(img, 1.0, cfg_off)
    assert calls == []

    cfg_on = sp.SegConfig(sec_min_diam_um=1.0, sec_adaptive_h=True)
    sp._segment_secondary(img, 1.0, cfg_on)
    assert len(calls) == 1


def test_default_config_yaml_enables_adaptive_h():
    """2026-07-23 定档:configs/config.yaml 的默认口径必须是自适应 h 开。
    防止日后有人改 config.yaml 或 SegConfig 默认值时无声退回旧的固定 h 行为。"""
    from nfm.config import load_config

    cfg = load_config()
    sem_cfg = sp.config_from_yaml(cfg.raw)
    assert sem_cfg.seg.sec_adaptive_h is True
