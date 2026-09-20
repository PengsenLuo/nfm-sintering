# -*- coding: utf-8 -*-
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from nfm import nano_layer as nl


def _fake_df():
    return pd.DataFrame({
        "sample_id": ["C01-S", "C01-M", "C02-S", "C02-M", "C20-M"],
        "xrd_instrument": [1, 1, 2, 2, 1],
        "D_XRD_reliable": [True, True, True, False, True],
        "D_XRD": [50.0, 55.0, 30.0, np.nan, 120.0],
        "lattice_a": [2.98, 2.98, 2.98, 2.98, 2.98],
        "lattice_c": [16.0, 16.0, 16.0, 16.0, 16.0],
        "c_a_ratio": [5.37, 5.37, 5.37, 5.37, 5.37],
    })


def test_nano_layer_frame_keeps_only_instrument_1():
    out = nl.nano_layer_frame(_fake_df(), require_reliable=False)
    assert set(out["sample_id"]) == {"C01-S", "C01-M", "C20-M"}
    assert (out["xrd_instrument"] == 1).all()


def test_nano_layer_frame_require_reliable_filters_unreliable():
    df = _fake_df()
    df.loc[df["sample_id"] == "C20-M", "D_XRD_reliable"] = False
    out = nl.nano_layer_frame(df, require_reliable=True)
    assert set(out["sample_id"]) == {"C01-S", "C01-M"}


def test_nano_layer_frame_require_reliable_default_true():
    df = _fake_df()
    df.loc[df["sample_id"] == "C20-M", "D_XRD_reliable"] = False
    out = nl.nano_layer_frame(df)
    assert set(out["sample_id"]) == {"C01-S", "C01-M"}


def test_nano_layer_frame_warns_with_sample_count():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = nl.nano_layer_frame(_fake_df(), require_reliable=False)
    msgs = [str(w.message) for w in caught]
    assert any(str(len(out)) in m for m in msgs)
