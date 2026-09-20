# -*- coding: utf-8 -*-
"""Regression tests for scripts/02i_shape_descriptors.py's segmentation-
config cache invalidation (2026-08-29 root-cause fix). Before this fix,
extract()'s incremental logic keyed its "already processed" cache purely on
sample_id, so a SegConfig change (e.g. the sec_adaptive_h default flip)
silently left early-processed samples' rows stale forever -- discovered when
those stale rows disagreed with master_table.csv's D_sec by up to 0.42 um
for one sample. These tests lock in that changing the segmentation config
invalidates the cache."""
import importlib.util as _ilu
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_spec = _ilu.spec_from_file_location(
    "shape_descriptors_02i",
    Path(__file__).resolve().parents[1] / "scripts" / "02i_shape_descriptors.py",
)
shape02i = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(shape02i)

from nfm.data_processing.sem_processor import SegConfig, SemProcessorConfig  # noqa: E402


def _cfg(**overrides):
    seg = SegConfig(**overrides)
    return SemProcessorConfig(seg=seg)


def test_seg_config_hash_changes_when_a_field_changes():
    h1 = shape02i._seg_config_hash(_cfg(sec_hmaxima_rel=0.25))
    h2 = shape02i._seg_config_hash(_cfg(sec_hmaxima_rel=0.30))
    assert h1 != h2


def test_seg_config_hash_stable_for_identical_config():
    h1 = shape02i._seg_config_hash(_cfg())
    h2 = shape02i._seg_config_hash(_cfg())
    assert h1 == h2


def test_valid_cached_rows_keeps_only_matching_hash():
    done = pd.DataFrame({
        "sample_id": ["C01-S", "C02-S"],
        "seg_config_hash": ["aaa", "bbb"],
    })
    kept = shape02i._valid_cached_rows(done, "aaa")
    assert list(kept["sample_id"]) == ["C01-S"]


def test_valid_cached_rows_treats_legacy_file_without_hash_column_as_all_stale():
    done = pd.DataFrame({"sample_id": ["C01-S", "C02-S"]})  # no seg_config_hash column
    kept = shape02i._valid_cached_rows(done, "aaa")
    assert kept.empty


def test_valid_cached_rows_empty_input_returns_empty():
    done = pd.DataFrame()
    kept = shape02i._valid_cached_rows(done, "aaa")
    assert kept.empty
