# -*- coding: utf-8 -*-
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from nfm.config import load_config
from nfm.data_processing import xrd_processor as xp

RAW_XRD = Path(__file__).resolve().parents[1] / "data" / "raw" / "xrd"


def test_detect_instrument_miniflex_file():
    assert xp.detect_instrument(RAW_XRD / "C02-S.txt") == 2


def test_detect_instrument_smartlab_file():
    assert xp.detect_instrument(RAW_XRD / "C07-S.txt") == 1


def test_detect_instrument_unrecognized_file_raises(tmp_path):
    bogus = tmp_path / "bogus.txt"
    bogus.write_text("not a real header\n1 2\n3 4\n", encoding="utf-8")
    with pytest.raises(ValueError):
        xp.detect_instrument(bogus)


def _synthetic_pattern(shift=0.0, noise=False, rng=None):
    tt = np.arange(5.0, 90.0, 0.02)
    ref = {"003": 16.537, "006": 33.431, "101": 35.164, "104": 41.534, "110": 62.206}
    if not noise:
        ii = np.full_like(tt, 50.0)
        for two_th in ref.values():
            ii = ii + 3000.0 * np.exp(-0.5 * ((tt - (two_th + shift)) / 0.05) ** 2)
    else:
        rng = rng or np.random.default_rng(0)
        ii = np.full_like(tt, 50.0) + rng.normal(0, 5.0, size=tt.shape)
    return tt, ii


def test_estimate_zero_offset_detects_shift():
    tt, ii = _synthetic_pattern(shift=0.9)
    out = xp.estimate_zero_offset(tt, ii)
    assert out["hits"] == 5
    assert abs(out["delta"] - 0.9) < 0.02


def test_estimate_zero_offset_no_shift():
    tt, ii = _synthetic_pattern(shift=0.0)
    out = xp.estimate_zero_offset(tt, ii)
    assert out["hits"] == 5
    assert abs(out["delta"]) <= 0.01


def test_estimate_zero_offset_noise_only_no_correction():
    tt, ii = _synthetic_pattern(noise=True)
    out = xp.estimate_zero_offset(tt, ii)
    assert out["hits"] < 4
    assert out["delta"] == 0.0


def test_process_all_routes_caglioti_by_instrument(tmp_path, monkeypatch):
    cfg = load_config()
    calls = []
    orig = xp.instrument_fwhm

    def spy(two_theta_deg, caglioti):
        calls.append(tuple(caglioti))
        return orig(two_theta_deg, caglioti)

    monkeypatch.setattr(xp, "instrument_fwhm", spy)

    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("C07-S.txt", "C02-S.txt"):
        (raw / name).write_bytes((RAW_XRD / name).read_bytes())

    xp.process_all(raw, tmp_path / "out.csv", cfg)

    inst1_UVW = tuple(cfg.raw["instruments"]["xrd"]["caglioti_UVW"])
    inst2_UVW = tuple(cfg.raw["instruments"]["xrd"]["caglioti_UVW_inst2"])
    assert inst1_UVW in calls
    assert inst2_UVW in calls


def test_process_all_flags_low_resolution_as_unreliable(tmp_path, monkeypatch):
    cfg = load_config()
    monkeypatch.setitem(cfg.raw["instruments"]["xrd"], "resolution_ratio_min", 1e6)

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "C07-S.txt").write_bytes((RAW_XRD / "C07-S.txt").read_bytes())

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        out = xp.process_all(raw, tmp_path / "out.csv", cfg)

    row = out.iloc[0]
    assert row["D_XRD_reliable"] == False
    assert np.isnan(row["D_XRD"])


def _synthetic_pattern_with_hydrate(hyd_amp=0.0, hyd_center=14.4, hyd_width=0.3, base=50.0):
    """003 峰(16.537°)+ 若干晶格峰,可选叠加水合相峰(14.4° 附近,层间膨胀相)。
    用于 compute_hydrate_index 单测:控制 hyd_amp 的大小/是否为0,检验符号与量级。
    """
    tt = np.arange(5.0, 90.0, 0.02)
    ref = {"003": 16.537, "006": 33.431, "101": 35.164, "104": 41.534, "110": 62.206}
    ii = np.full_like(tt, base)
    for two_th in ref.values():
        ii = ii + 3000.0 * np.exp(-0.5 * ((tt - two_th) / 0.05) ** 2)
    if hyd_amp:
        ii = ii + hyd_amp * np.exp(-0.5 * ((tt - hyd_center) / hyd_width) ** 2)
    return tt, ii


def test_compute_hydrate_index_no_hydrate_peak_near_zero():
    """无水合相峰(仅背景+003主峰)时,hydrate_index 应接近0(不应有虚假正信号)。"""
    tt, ii = _synthetic_pattern_with_hydrate(hyd_amp=0.0)
    val = xp.compute_hydrate_index(tt, ii)
    assert np.isfinite(val)
    assert abs(val) < 1.0  # 纯背景,允许基线插值带来的微小数值噪声


def test_compute_hydrate_index_with_hydrate_peak_positive_and_scaled():
    """叠加一个与003峰同量级的水合相峰后,hydrate_index 应明显为正,
    且峰越大 hydrate_index 越大(符号与单调性检验)。"""
    tt, ii_small = _synthetic_pattern_with_hydrate(hyd_amp=300.0)
    _, ii_big = _synthetic_pattern_with_hydrate(hyd_amp=3000.0)
    val_small = xp.compute_hydrate_index(tt, ii_small)
    val_big = xp.compute_hydrate_index(tt, ii_big)
    assert np.isfinite(val_small) and np.isfinite(val_big)
    assert val_small > 1.0          # 明显偏离"无水合相"的近零基线
    assert val_big > val_small      # 峰面积越大,指标越大(单调性)
    assert val_big < 1000.0         # 量级合理(该合成峰面积已远超003主峰),不应炸到发散/负值


def test_compute_hydrate_index_scan_start_above_threshold_returns_nan():
    """扫描起点高于13.8°(不覆盖 A_hyd 区间起点)时,必须返回 NaN,不报错。"""
    tt, ii = _synthetic_pattern_with_hydrate(hyd_amp=0.0)
    m = tt >= 14.0
    val = xp.compute_hydrate_index(tt[m], ii[m])
    assert np.isnan(val)


def test_process_all_adds_hydrate_index_column_for_real_sample():
    """真实仪器1样本(C07-S,已知规范操作、无污染)应给出较低的 hydrate_index。"""
    cfg = load_config()

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "raw"
        raw.mkdir()
        (raw / "C07-S.txt").write_bytes((RAW_XRD / "C07-S.txt").read_bytes())
        out = xp.process_all(raw, Path(td) / "out.csv", cfg)

    assert "hydrate_index" in out.columns
    val = out.iloc[0]["hydrate_index"]
    assert np.isfinite(val)
    assert 0.0 <= val < 5.0  # 规范操作样本,预期落在仪器1已知量级(中位0.45%/最大1.09%)附近


def test_process_all_adds_new_diagnostic_columns(tmp_path):
    cfg = load_config()
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "C07-S.txt").write_bytes((RAW_XRD / "C07-S.txt").read_bytes())

    out = xp.process_all(raw, tmp_path / "out.csv", cfg)
    row = out.iloc[0]
    assert row["xrd_instrument"] == 1
    assert "two_theta_offset_applied" in out.columns
    assert "zero_offset_hits" in out.columns
    assert "resolution_ratio" in out.columns
    assert row["D_XRD_reliable"] == True
