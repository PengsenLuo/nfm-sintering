# -*- coding: utf-8 -*-
"""tests/test_packing_models.py —— T9 堆积模型计算引擎的健全性检查(sanity check)

覆盖任务书 §1(Furnas 二元堆积)/§2(LPM)要求的三类检查:
  1. 单组分退化(f_small->0/1 回到单组分堆积分数)
  2. 尺寸比退化到1(级配不应带来增益)
  3. 中间体积分数存在堆积分数的极大值,且高于两端点(级配增益的模型对应)

不测试论文 §4.1 的产物压实密度结论——那需要产物 PSD + 免水压实密度实测数据,
目前仍是 [待测],本文件与本模块都不解除这个缺口。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfm.packing_models import (
    LPMComponent,
    furnas_binary_packing_fraction,
    lpm_loosening_effect,
    lpm_packing_fraction,
    lpm_wall_effect,
    psd_density_to_lpm_components,
)

PHI0 = 0.64


# =======================================================================
# §1 Furnas 二元堆积模型
# =======================================================================
class TestFurnasSingleComponentLimits:
    def test_f_small_zero_returns_phi_large(self):
        v = furnas_binary_packing_fraction(d_large=100, d_small=5, f_small=0.0,
                                            phi_large=0.60, phi_small=0.68)
        assert v == pytest.approx(0.60)

    def test_f_small_one_returns_phi_small(self):
        v = furnas_binary_packing_fraction(d_large=100, d_small=5, f_small=1.0,
                                            phi_large=0.60, phi_small=0.68)
        assert v == pytest.approx(0.68)

    def test_default_phi0_endpoints(self):
        assert furnas_binary_packing_fraction(100, 5, 0.0) == pytest.approx(PHI0)
        assert furnas_binary_packing_fraction(100, 5, 1.0) == pytest.approx(PHI0)


class TestFurnasSizeRatioDegeneration:
    def test_equal_size_no_gain_at_any_fraction(self):
        # d_small == d_large(尺寸比恰为1):无论 f_small 取何值,不应有级配增益,
        # 预测值应精确回到共同的单组分堆积分数。
        for f in (0.1, 0.26, 0.3, 0.5, 0.7, 0.9):
            v = furnas_binary_packing_fraction(d_large=100, d_small=100, f_small=f)
            assert v == pytest.approx(PHI0, abs=1e-9), f"f_small={f} 时不应有增益"

    def test_near_equal_size_gain_is_small(self):
        # 尺寸比 0.99(接近1)时,增益应远小于极端尺寸比(比如0.05)下的增益。
        f = 0.26
        near_equal = furnas_binary_packing_fraction(d_large=100, d_small=99, f_small=f)
        extreme = furnas_binary_packing_fraction(d_large=100, d_small=5, f_small=f)
        gain_near_equal = near_equal - PHI0
        gain_extreme = extreme - PHI0
        assert 0 <= gain_near_equal < 0.02, "尺寸比->1 时增益应接近0"
        assert gain_extreme > 0.15, "极端尺寸比下应有明显增益(sanity:模型本身工作正常)"
        assert gain_near_equal < gain_extreme


class TestFurnasIntermediateOptimum:
    def test_extreme_ratio_has_interior_maximum_above_both_endpoints(self):
        d_large, d_small = 100.0, 5.0
        phi_large = phi_small = PHI0
        fs = np.linspace(0.01, 0.99, 99)
        vals = [furnas_binary_packing_fraction(d_large, d_small, f, phi_large, phi_small)
                for f in fs]
        vmax = max(vals)
        argmax = fs[int(np.argmax(vals))]
        assert vmax > phi_large, "极大值必须高于纯大颗粒端点"
        assert vmax > phi_small, "极大值必须高于纯小颗粒端点"
        # 文献典型最优细颗粒体积分数约 0.2-0.3(极端尺寸比下的 Furnas 理论)
        assert 0.15 < argmax < 0.40, f"最优 f_small={argmax} 超出文献典型范围"

    def test_theoretical_peak_matches_closed_form(self):
        # 极端尺寸比理想情形下,峰值应接近闭式结果
        # phi_max = phi_large + phi_small*(1-phi_large)(见模块 docstring 推导)。
        d_large, d_small = 1000.0, 1.0  # r~0.001,非常接近理想极限
        phi_large = phi_small = PHI0
        expected_peak = phi_large + phi_small * (1 - phi_large)
        fs = np.linspace(0.01, 0.99, 199)
        vals = [furnas_binary_packing_fraction(d_large, d_small, f, phi_large, phi_small)
                for f in fs]
        assert max(vals) == pytest.approx(expected_peak, abs=0.01)


class TestFurnasInputValidation:
    def test_negative_size_raises(self):
        with pytest.raises(ValueError):
            furnas_binary_packing_fraction(d_large=-1, d_small=5, f_small=0.3)

    def test_f_small_out_of_range_raises(self):
        with pytest.raises(ValueError):
            furnas_binary_packing_fraction(d_large=100, d_small=5, f_small=1.5)

    def test_phi_out_of_range_raises(self):
        with pytest.raises(ValueError):
            furnas_binary_packing_fraction(d_large=100, d_small=5, f_small=0.3, phi_large=1.2)


# =======================================================================
# §2 线性堆积模型 LPM
# =======================================================================
class TestLPMSingleComponentLimits:
    def test_single_class_returns_beta(self):
        v = lpm_packing_fraction([LPMComponent(d=50.0, y=1.0)], beta=0.62)
        assert v == pytest.approx(0.62)

    def test_binary_f_zero_and_one_match_beta(self):
        beta = 0.64
        v0 = lpm_packing_fraction([LPMComponent(100, 1.0), LPMComponent(5, 0.0)], beta=beta)
        v1 = lpm_packing_fraction([LPMComponent(100, 0.0), LPMComponent(5, 1.0)], beta=beta)
        assert v0 == pytest.approx(beta)
        assert v1 == pytest.approx(beta)


class TestLPMSizeRatioDegeneration:
    def test_equal_size_no_gain(self):
        beta = 0.64
        for f in (0.1, 0.3, 0.5, 0.7, 0.9):
            comps = [LPMComponent(d=100.0, y=1 - f), LPMComponent(d=100.0, y=f)]
            v = lpm_packing_fraction(comps, beta=beta)
            assert v == pytest.approx(beta, abs=1e-6), f"等尺寸时 f={f} 不应有增益"

    def test_coefficients_hit_documented_boundaries(self):
        assert lpm_wall_effect(0.0) == pytest.approx(0.0)
        assert lpm_wall_effect(1.0) == pytest.approx(1.0)
        assert lpm_loosening_effect(0.0) == pytest.approx(0.0)
        assert lpm_loosening_effect(0.19) == pytest.approx(0.0)  # 低于临界比 x0=0.2
        assert lpm_loosening_effect(1.0) == pytest.approx(1.0)

    def test_near_equal_size_gain_smaller_than_disparate(self):
        beta = 0.64
        f = 0.25
        near_equal = lpm_packing_fraction(
            [LPMComponent(100.0, 1 - f), LPMComponent(95.0, f)], beta=beta)
        disparate = lpm_packing_fraction(
            [LPMComponent(100.0, 1 - f), LPMComponent(5.0, f)], beta=beta)
        assert (near_equal - beta) < (disparate - beta)


class TestLPMIntermediateOptimum:
    def test_binary_disparate_sizes_has_interior_maximum(self):
        beta = 0.64
        fs = np.linspace(0.01, 0.99, 99)
        vals = []
        for f in fs:
            comps = [LPMComponent(100.0, 1 - f), LPMComponent(5.0, f)]
            vals.append(lpm_packing_fraction(comps, beta=beta))
        vmax = max(vals)
        argmax = fs[int(np.argmax(vals))]
        assert vmax > beta, "极大值必须高于两端点(端点都是 beta)"
        assert 0.10 < argmax < 0.45, f"最优细颗粒体积分数 {argmax} 超出预期范围"

    def test_multi_class_full_psd_style_input_runs_and_beats_baseline(self):
        # 模拟"完整 PSD"(多档,而不只是两组分)的输入形式,验证 LPM 能处理
        # n>2 档且仍能产生高于任一单档 beta 的堆积分数(级配增益在多档场景下
        # 依然存在)。
        beta = 0.64
        comps = [
            LPMComponent(d=200.0, y=0.30),
            LPMComponent(d=50.0, y=0.25),
            LPMComponent(d=12.0, y=0.25),
            LPMComponent(d=3.0, y=0.20),
        ]
        v = lpm_packing_fraction(comps, beta=beta)
        assert v == v, "多档输入不应返回 NaN"
        assert v > beta, "级配良好的多档 PSD 应有堆积增益"
        assert v < 1.0


class TestLPMInputValidation:
    def test_empty_components_raises(self):
        with pytest.raises(ValueError):
            lpm_packing_fraction([])

    def test_negative_diameter_raises(self):
        with pytest.raises(ValueError):
            lpm_packing_fraction([LPMComponent(d=-1.0, y=1.0)])

    def test_negative_fraction_raises(self):
        with pytest.raises(ValueError):
            lpm_packing_fraction([LPMComponent(d=10.0, y=-0.5), LPMComponent(d=5.0, y=1.5)])

    def test_all_zero_fraction_raises(self):
        with pytest.raises(ValueError):
            lpm_packing_fraction([LPMComponent(d=10.0, y=0.0), LPMComponent(d=5.0, y=0.0)])


# =======================================================================
# §3 PSD 曲线 -> LPM 输入 转换工具
# =======================================================================
class TestPsdDensityToLpmComponents:
    def test_filters_zero_and_nan_bins(self):
        d = np.array([1.0, 2.0, 3.0, 4.0])
        pct = np.array([0.0, np.nan, 5.0, 10.0])
        comps = psd_density_to_lpm_components(d, pct)
        assert len(comps) == 2
        assert {c.d for c in comps} == {3.0, 4.0}

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            psd_density_to_lpm_components(np.array([1.0, 2.0]), np.array([1.0]))

    def test_roundtrips_into_lpm_without_error(self):
        d = np.array([1.0, 5.0, 20.0, 100.0])
        pct = np.array([5.0, 15.0, 30.0, 50.0])
        comps = psd_density_to_lpm_components(d, pct)
        v = lpm_packing_fraction(comps, beta=0.64)
        assert 0.0 < v < 1.0
