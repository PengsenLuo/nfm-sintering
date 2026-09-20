# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from nfm.features.memory_indices import compute_within_furnace_indices


class _FakeCfg:
    def __init__(self, precursor_B_M=2.0):
        self.raw = {
            "design": {
                "precursor_desc": {"M": {"blend_ratio": 0.3}},
                "precursor_B": {"M": precursor_B_M},
            }
        }


def _base_rows(note_M):
    return pd.DataFrame([
        dict(condition_id="C99", precursor="S", D50=3.0, precursor_D50=3.67,
             compaction_density=3.0, B_bimodal=np.nan, note="unimodal_by_modecount"),
        dict(condition_id="C99", precursor="M", D50=6.0, precursor_D50=9.7,
             compaction_density=3.1, B_bimodal=np.nan, note=note_M),
        dict(condition_id="C99", precursor="L", D50=10.0, precursor_D50=12.4,
             compaction_density=3.2, B_bimodal=np.nan, note="unimodal_by_modecount"),
    ])


def test_mb_erased_to_zero_when_product_confirmed_unimodal():
    for note_M in ("unimodal_by_modecount", "degenerate_unimodal"):
        df = _base_rows(note_M)
        out = compute_within_furnace_indices(df, _FakeCfg())
        row = out[out.precursor == "M"].iloc[0]
        assert row["M_B"] == pytest.approx(0.0)
        assert row["M_B_note"] == "erased_endpoint"


def test_mb_computed_ratio_when_product_genuinely_bimodal():
    df = _base_rows("ok")
    df.loc[df.precursor == "M", "B_bimodal"] = 1.0
    out = compute_within_furnace_indices(df, _FakeCfg(precursor_B_M=2.0))
    row = out[out.precursor == "M"].iloc[0]
    assert row["M_B"] == pytest.approx(0.5)
    assert row["M_B_note"] == "computed"


def test_mb_note_no_data_when_product_missing_measurement():
    # M 产物样本尚未到样,无任何 PSD 测量:note=NaN,B_bimodal=NaN。
    # B_prec_M 已正确配置(非 NaN、非 0),证明这是缺分子数据的问题,不是缺分母配置。
    df = _base_rows(np.nan)
    out = compute_within_furnace_indices(df, _FakeCfg(precursor_B_M=2.0))
    row = out[out.precursor == "M"].iloc[0]
    assert pd.isna(row["M_B"])
    assert row["M_B_note"] == "no_data"


def test_m_d_agg_computed_from_sec_particle_size():
    # M_D_agg = (D_sec_L - D_sec_S) / (precursor_L_D50 - precursor_S_D50)
    df = pd.DataFrame([
        dict(condition_id="C50", precursor="S", D50=3.0, precursor_D50=3.67,
             compaction_density=3.0, B_bimodal=np.nan, note="unimodal_by_modecount",
             D_sec=5.0),
        dict(condition_id="C50", precursor="M", D50=6.0, precursor_D50=9.7,
             compaction_density=3.1, B_bimodal=np.nan, note="ok", D_sec=8.0),
        dict(condition_id="C50", precursor="L", D50=10.0, precursor_D50=12.4,
             compaction_density=3.2, B_bimodal=np.nan, note="unimodal_by_modecount",
             D_sec=11.5),
    ])
    out = compute_within_furnace_indices(df, _FakeCfg())
    expected_m_d_agg = (11.5 - 5.0) / (12.4 - 3.67)
    for _, row in out[out.condition_id == "C50"].iterrows():
        assert row["M_D_agg"] == pytest.approx(expected_m_d_agg)
    # 旧 M_D 用产物激光 D50,不受 D_sec 影响(回归)
    expected_m_d = (10.0 - 3.0) / (12.4 - 3.67)
    for _, row in out[out.condition_id == "C50"].iterrows():
        assert row["M_D"] == pytest.approx(expected_m_d)


def test_m_d_agg_nan_when_sec_particle_size_missing_and_isolated_to_condition():
    df = pd.DataFrame([
        # C51: D_sec 缺失(S 行 NaN)→ 该条件 M_D_agg 应为 NaN
        dict(condition_id="C51", precursor="S", D50=3.0, precursor_D50=3.67,
             compaction_density=3.0, B_bimodal=np.nan, note="unimodal_by_modecount",
             D_sec=np.nan),
        dict(condition_id="C51", precursor="M", D50=6.0, precursor_D50=9.7,
             compaction_density=3.1, B_bimodal=np.nan, note="ok", D_sec=8.0),
        dict(condition_id="C51", precursor="L", D50=10.0, precursor_D50=12.4,
             compaction_density=3.2, B_bimodal=np.nan, note="unimodal_by_modecount",
             D_sec=11.5),
        # C52: D_sec 齐全 → M_D_agg 应正常计算,不被 C51 的缺失污染
        dict(condition_id="C52", precursor="S", D50=3.0, precursor_D50=3.67,
             compaction_density=3.0, B_bimodal=np.nan, note="unimodal_by_modecount",
             D_sec=4.0),
        dict(condition_id="C52", precursor="M", D50=6.0, precursor_D50=9.7,
             compaction_density=3.1, B_bimodal=np.nan, note="ok", D_sec=7.0),
        dict(condition_id="C52", precursor="L", D50=10.0, precursor_D50=12.4,
             compaction_density=3.2, B_bimodal=np.nan, note="unimodal_by_modecount",
             D_sec=9.0),
    ])
    out = compute_within_furnace_indices(df, _FakeCfg())
    assert out[out.condition_id == "C51"]["M_D_agg"].isna().all()
    expected_c52 = (9.0 - 4.0) / (12.4 - 3.67)
    for _, row in out[out.condition_id == "C52"].iterrows():
        assert row["M_D_agg"] == pytest.approx(expected_c52)
