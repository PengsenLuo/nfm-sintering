# -*- coding: utf-8 -*-
import importlib.util as _ilu
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

_spec = _ilu.spec_from_file_location(
    "sem_psd_quantiles",
    Path(__file__).resolve().parents[1] / "scripts" / "44_sem_psd_quantiles.py",
)
sem_psd_quantiles = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(sem_psd_quantiles)


def test_numeric_quantiles_uniform_data():
    d = np.arange(1, 101, dtype=float)  # 1..100
    out = sem_psd_quantiles.numeric_quantiles(d)
    assert out["n_particles"] == 100
    assert out["D50_num"] == pytest.approx(50.5, abs=1e-6)
    assert out["D10_num"] == pytest.approx(np.percentile(d, 10), abs=1e-9)
    assert out["D90_num"] == pytest.approx(np.percentile(d, 90), abs=1e-9)
    assert out["Span_num"] == pytest.approx(
        (out["D90_num"] - out["D10_num"]) / out["D50_num"], abs=1e-9
    )


def test_volume_weighted_quantiles_all_equal_diameter():
    d = np.full(50, 5.0)
    out = sem_psd_quantiles.volume_weighted_quantiles(d)
    assert out["D10_vol"] == pytest.approx(5.0, abs=1e-9)
    assert out["D50_vol"] == pytest.approx(5.0, abs=1e-9)
    assert out["D90_vol"] == pytest.approx(5.0, abs=1e-9)
    assert out["Span_vol"] == pytest.approx(0.0, abs=1e-9)


def test_volume_weighted_quantiles_biased_toward_large_particles():
    # 99 particles of diameter 1, 1 particle of diameter 10 -> volume of the
    # single large particle (10**3=1000) dwarfs the 99 small ones (99*1=99),
    # so the volume-weighted median must sit near the large particle, while
    # the number-weighted median sits near the small ones.
    d = np.concatenate([np.full(99, 1.0), np.full(1, 10.0)])
    num_out = sem_psd_quantiles.numeric_quantiles(d)
    vol_out = sem_psd_quantiles.volume_weighted_quantiles(d)
    assert num_out["D50_num"] == pytest.approx(1.0, abs=1e-6)
    assert vol_out["D50_vol"] > 5.0


def test_n_carry_all_equal_diameter_needs_all_particles():
    d = np.full(20, 3.0)
    assert sem_psd_quantiles.n_carry(d, 0.5) == 10
    assert sem_psd_quantiles.n_carry(d, 0.9) == 18


def test_n_carry_one_dominant_particle_carries_most_volume():
    d = np.concatenate([np.full(99, 1.0), np.full(1, 10.0)])
    # the single d=10 particle alone holds 1000/(1000+99)=91% of volume
    assert sem_psd_quantiles.n_carry(d, 0.5) == 1
    assert sem_psd_quantiles.n_carry(d, 0.9) == 1
