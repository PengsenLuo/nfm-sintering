# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from nfm import schema
from nfm.config import load_config, build_sample_index


def test_schema_unique_and_counts():
    assert len(schema.BY_NAME) == len(schema.SCHEMA)
    assert "D_XRD" in schema.BY_NAME
    assert "hier_size_ratio" in schema.BY_NAME
    assert "agglomeration_index" not in schema.BY_NAME   # 已更名
    assert {"M_D", "M_B", "dRho_M"} <= set(schema.diff_cols())


def test_views_obey_iron_law():
    cfg = load_config()
    # 三个视图应能通过校验(load_config 内已校验,不抛即通过)
    assert "view_two_step" in cfg.raw["views"]


def test_sample_index_81():
    cfg = load_config()
    assert len(build_sample_index(cfg)) == 81


def test_precursor_D50_from_instrument_table_not_nominal():
    cfg = load_config()
    idx = build_sample_index(cfg)
    d50 = idx.drop_duplicates("precursor").set_index("precursor")["precursor_D50"]
    assert d50["S"] == pytest.approx(3.67, abs=1e-6)
    assert d50["M"] == pytest.approx(9.7, abs=1e-6)
    assert d50["L"] == pytest.approx(12.4, abs=1e-6)


def test_precursor_D50_upper_bound_covers_L_instrument_value():
    spec = schema.BY_NAME["precursor_D50"]
    assert spec.bounds == (2, 13)
    lo, hi = spec.bounds
    assert lo <= 12.4 <= hi


def test_two_step_view_leak_detection():
    # 人为构造泄漏视图应被拒
    bad = {"X": ["D50"], "y": ["D50"]}
    try:
        schema.validate_view("view_two_step", bad)
        assert False, "应检测到 X 与 y 重叠"
    except ValueError:
        pass
