# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from nfm.data_processing import psd_processor as pp


def _write_sample_csv(path: Path, d_pct_rows):
    pd.DataFrame(d_pct_rows, columns=["Size (um)", "Volume (%)"]).to_csv(path, index=False)


def test_iter_sample_csv_files_excludes_dvalues_files(tmp_path):
    _write_sample_csv(tmp_path / "C01-S.csv", [(1.0, 10.0), (2.0, 50.0), (5.0, 100.0)])
    (tmp_path / "psd_dvalues_samples.csv").write_text(
        "sample_id,D10,D50,D90,Span\nC01-S,0.3,2.0,6.0,2.8\n", encoding="utf-8")
    (tmp_path / "psd_dvalues_precursors.csv").write_text(
        "sample_id,D10,D50,D90,Span\nprecursor_S,2.75,3.67,4.96,0.602\n", encoding="utf-8")
    (tmp_path / "not_a_sample.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    files = pp._iter_sample_csv_files(tmp_path)

    assert [f.name for f in files] == ["C01-S.csv"]


def test_process_all_uses_instrument_dvalues_not_selfcalc(tmp_path):
    raw = tmp_path / "samples"
    raw.mkdir()
    # 单调递增的“真累积分布”，自算 D50 应约为 2.0(与仪器值刻意不同,便于区分来源)
    _write_sample_csv(raw / "C01-S.csv",
                       [(0.5, 2), (1.0, 10), (2.0, 50), (4.0, 90), (8.0, 100)])
    (raw / "psd_dvalues_samples.csv").write_text(
        "sample_id,D10,D50,D90,Span\nC01-S,0.30,1.85,3.41,1.68\n", encoding="utf-8")
    out_csv = tmp_path / "out.csv"

    out = pp.process_all(raw, out_csv, cfg=None)

    row = out.set_index("sample_id").loc["C01-S"]
    assert row["D10"] == pytest.approx(0.30)
    assert row["D50"] == pytest.approx(1.85)
    assert row["D90"] == pytest.approx(3.41)
    assert row["Span"] == pytest.approx(1.68)
    assert row["D_value_source"] == "instrument"
    # 自算值必须留档且与仪器值不同(否则测试没有区分度)
    assert row["D50_selfcalc"] != pytest.approx(1.85)


def test_process_all_falls_back_to_selfcalc_when_dvalues_row_missing(tmp_path):
    raw = tmp_path / "samples"
    raw.mkdir()
    _write_sample_csv(raw / "C09-L.csv",
                       [(0.5, 2), (1.0, 10), (2.0, 50), (4.0, 90), (8.0, 100)])
    (raw / "psd_dvalues_samples.csv").write_text(
        "sample_id,D10,D50,D90,Span\nC01-S,0.30,1.85,3.41,1.68\n", encoding="utf-8")
    out_csv = tmp_path / "out.csv"

    with pytest.warns(UserWarning, match="C09-L"):
        out = pp.process_all(raw, out_csv, cfg=None)

    row = out.set_index("sample_id").loc["C09-L"]
    assert row["D_value_source"] == "selfcalc_fallback"
    assert row["D50"] == pytest.approx(row["D50_selfcalc"])


def test_bimodal_guard_boundaries():
    assert pp._bimodal_guard(0.5) is True
    assert pp._bimodal_guard(0.05) is True
    assert pp._bimodal_guard(0.95) is True
    assert pp._bimodal_guard(0.049) is False
    assert pp._bimodal_guard(0.951) is False


def test_count_modes_c14m_bimodal_c21m_c27m_unimodal():
    base = Path(__file__).resolve().parents[1] / "data/raw/psd/samples"
    if not base.exists():
        pytest.skip("本机无 data/raw/psd/samples,跳过真实数据回归测试")
    for sid, expected_n in [("C14-M", 2), ("C21-M", 1), ("C27-M", 1)]:
        d, density = pp.read_density_csv(base / f"{sid}.csv")
        n_modes, positions = pp.count_modes(d, density, prominence=0.5, height=0.1)
        assert n_modes == expected_n, f"{sid}: 期望 {expected_n} 模,实得 {n_modes}({positions})"


def test_count_modes_prominence_boundary():
    # 构造合成密度曲线:主峰高 5,次峰高 0.8(prominence 约 0.8,谷底 0)
    d = np.linspace(0.1, 10, 100)
    density = 5 * np.exp(-((d - 2) ** 2) / 0.5) + 0.8 * np.exp(-((d - 7) ** 2) / 0.3)
    n_low, _ = pp.count_modes(d, density, prominence=0.5, height=0.1)
    n_high, _ = pp.count_modes(d, density, prominence=1.0, height=0.1)
    assert n_low == 2      # prominence 低于次峰突起度→两模都保留
    assert n_high == 1     # prominence 高于次峰突起度→次峰被滤掉,退化为单模


def test_c24l_intercepted_at_modecount_stage():
    raw = Path(__file__).resolve().parents[1] / "data/raw/psd/samples/C24-L.csv"
    if not raw.exists():
        pytest.skip("本机无 data/raw/psd/samples/C24-L.csv，跳过真实数据回归测试")
    d, cum = pp.read_cumulative_csv(raw)
    _, density = pp.read_density_csv(raw)
    r = pp.fit_bimodal(d, cum, density)
    assert r["note"] == "unimodal_by_modecount"
    assert pd.isna(r["B_bimodal"])
    assert r["n_modes"] == 1


def test_c14m_true_bimodal_passes_modecount_and_fit():
    raw = Path(__file__).resolve().parents[1] / "data/raw/psd/samples/C14-M.csv"
    if not raw.exists():
        pytest.skip("本机无 data/raw/psd/samples/C14-M.csv，跳过真实数据回归测试")
    d, cum = pp.read_cumulative_csv(raw)
    _, density = pp.read_density_csv(raw)
    r = pp.fit_bimodal(d, cum, density)
    assert r["n_modes"] == 2
    assert r["note"] == "ok"
    assert r["B_bimodal"] > 0


def test_c27m_now_intercepted_at_modecount_not_wsmall():
    raw = Path(__file__).resolve().parents[1] / "data/raw/psd/samples/C27-M.csv"
    if not raw.exists():
        pytest.skip("本机无 data/raw/psd/samples/C27-M.csv，跳过真实数据回归测试")
    d, cum = pp.read_cumulative_csv(raw)
    _, density = pp.read_density_csv(raw)
    r = pp.fit_bimodal(d, cum, density)
    # 第十轮里 C27-M 是靠 w_small 守卫(二级判据)拦截的;本轮模式计数(一级判据)
    # 更早拦截,note 来源改变,但仍然被拦截,不产生 B_bimodal。
    assert r["note"] == "unimodal_by_modecount"
    assert pd.isna(r["B_bimodal"])
