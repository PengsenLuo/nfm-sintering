# -*- coding: utf-8 -*-
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "figs"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "figs" / "journal"))

import pandas as pd
import pytest

import _journal_style as js


@pytest.fixture
def source_data_dir(tmp_path, monkeypatch):
    d = tmp_path / "source_data"
    monkeypatch.setattr(js, "REPO_ROOT", REPO_ROOT)
    monkeypatch.setattr(js, "SOURCE_DATA_DIR", d)
    monkeypatch.setattr(js, "SOURCE_DATA_INDEX", d / "index.csv")
    return d


def test_dump_source_data_writes_tidy_csv(source_data_dir):
    df = pd.DataFrame({"sample_id": ["C01-S", "C01-M"], "Theta": [4.2, 5.1]})
    path = js.dump_source_data("figX_test", "a", df, note="unit test")
    assert path == source_data_dir / "figX_test__a.csv"
    assert path.exists()
    out = pd.read_csv(path)
    pd.testing.assert_frame_equal(out, df)


def test_dump_source_data_upserts_index_row_by_fig_and_panel(source_data_dir):
    js.dump_source_data("figX_test", "a", pd.DataFrame({"v": [1, 2, 3]}), note="n1")
    idx = pd.read_csv(source_data_dir / "index.csv", keep_default_na=False)
    assert len(idx) == 1
    row = idx.iloc[0]
    assert row["fig_name"] == "figX_test"
    assert row["panel"] == "a"
    assert row["n_rows"] == 3
    assert row["columns"] == "v"
    assert row["note"] == "n1"

    # Re-dumping the same (fig_name, panel) upserts in place, not appends.
    js.dump_source_data("figX_test", "a", pd.DataFrame({"v": [1, 2]}), note="n2")
    idx2 = pd.read_csv(source_data_dir / "index.csv", keep_default_na=False)
    assert len(idx2) == 1
    assert idx2.iloc[0]["n_rows"] == 2
    assert idx2.iloc[0]["note"] == "n2"


def test_dump_source_data_second_panel_appends_not_overwrites(source_data_dir):
    js.dump_source_data("figX_test", "a", pd.DataFrame({"v": [1]}))
    js.dump_source_data("figX_test", "b", pd.DataFrame({"v": [1, 2]}))
    idx = pd.read_csv(source_data_dir / "index.csv", keep_default_na=False)
    assert sorted(idx["panel"]) == ["a", "b"]


def test_dump_source_data_none_df_registers_index_only(source_data_dir):
    path = js.dump_source_data("figX_test", "images", df=None, note="image panel, no tabular data")
    assert path is None
    idx = pd.read_csv(source_data_dir / "index.csv", keep_default_na=False)
    row = idx[idx["panel"] == "images"].iloc[0]
    assert row["csv_path"] == ""
    assert row["n_rows"] == 0
    assert not (source_data_dir / "figX_test__images.csv").exists()


def test_dump_source_data_records_source_files_relative_to_repo_root(source_data_dir):
    js.dump_source_data(
        "figX_test", "a", pd.DataFrame({"v": [1]}),
        source_files=[REPO_ROOT / "data" / "processed" / "master_table.csv"],
    )
    idx = pd.read_csv(source_data_dir / "index.csv", keep_default_na=False)
    assert idx.iloc[0]["source_files"] == "data/processed/master_table.csv"


def test_dump_source_data_records_calling_script_path(source_data_dir):
    js.dump_source_data("figX_test", "a", pd.DataFrame({"v": [1]}))
    idx = pd.read_csv(source_data_dir / "index.csv", keep_default_na=False)
    assert idx.iloc[0]["script"] == "tests/test_figure_source_data.py"


def test_dump_source_data_resolves_relative_source_file_strings(source_data_dir):
    """Regression test: plain relative-string source_files inputs must be resolved
    to repo-relative paths, not converted to absolute paths."""
    # Pass a plain relative string (not a Path object)
    js.dump_source_data(
        "figX_test", "a", pd.DataFrame({"v": [1]}),
        source_files=["data/processed/master_table.csv"],
    )
    idx = pd.read_csv(source_data_dir / "index.csv", keep_default_na=False)
    # Should record the relative posix path, not an absolute Windows path
    assert idx.iloc[0]["source_files"] == "data/processed/master_table.csv"
