# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfm.features.thermal_exposure import theta

R = 8.314
Q = 200e3


def test_theta_monotonic_in_T_and_t():
    base = theta(900, 5, 15, Q_J=Q, R=R, T_ref_C=900, start_C=550)
    hotter = theta(950, 5, 15, Q_J=Q, R=R, T_ref_C=900, start_C=550)
    longer = theta(900, 5, 20, Q_J=Q, R=R, T_ref_C=900, start_C=550)
    assert hotter > base, "更高温 Θ 应更大"
    assert longer > base, "更长时 Θ 应更大"


def test_theta_ref_point_near_one_per_hour():
    # 参考温度 900℃ 下,保温段 integrand≈1,Θ 近似等于(有效)小时数量级
    val = theta(900, 5, 10, Q_J=Q, R=R, T_ref_C=900, start_C=550)
    assert 5 < val < 60
