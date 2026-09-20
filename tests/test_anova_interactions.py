# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from nfm.stats import anova_interactions as ai

MASTER_TABLE = Path(__file__).resolve().parents[1] / "data" / "processed" / "master_table.csv"


@pytest.fixture(scope="module")
def design_df():
    df = pd.read_csv(MASTER_TABLE)[["precursor", "T_C", "beta", "t_hold"]].copy()
    for c in ("T_C", "beta", "t_hold"):
        df[c] = df[c].astype("category")
    df["precursor"] = df["precursor"].astype("category")
    return df


def test_build_design_shape_and_term_slices(design_df):
    X, term_slices, col_names = ai.build_design(design_df)
    assert X.shape == (81, 33)
    assert list(term_slices.keys()) == ["Intercept"] + list(ai.TERM_ORDER)
    # main effects: 2 cols each; interactions: 4 cols each
    for t in ai.TERM_ORDER[:4]:
        assert term_slices[t].stop - term_slices[t].start == 2
    for t in ai.TERM_ORDER[4:]:
        assert term_slices[t].stop - term_slices[t].start == 4


def test_build_design_is_orthogonal_across_terms(design_df):
    X, term_slices, _ = ai.build_design(design_df)
    XtX = X.T @ X
    slices = list(term_slices.values())
    for i, si in enumerate(slices):
        for j, sj in enumerate(slices):
            if i >= j:
                continue
            assert np.abs(XtX[si, sj]).max() < 1e-8


def test_type1_ss_matches_statsmodels_on_real_data(design_df):
    df = pd.read_csv(MASTER_TABLE)
    full = pd.concat([design_df, df[["compaction_density"]]], axis=1)

    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    model = smf.ols(f"compaction_density ~ {ai.FORMULA_RHS}", data=full).fit()
    ref = anova_lm(model, typ=1)["sum_sq"]

    X, term_slices, _ = ai.build_design(design_df)
    y = full["compaction_density"].to_numpy(float)
    ss = ai.type1_ss(X, y, term_slices)

    # anova_lm(typ=1) rows are in formula order, same order as ai.TERM_ORDER,
    # followed by "Residual" -- match positionally rather than by string label
    # since statsmodels labels terms as "C(precursor, Sum)" etc.
    for i, term in enumerate(ai.TERM_ORDER):
        assert ss[term] == pytest.approx(ref.iloc[i], abs=1e-6)
    assert ss["Residual"] == pytest.approx(ref["Residual"], abs=1e-6)


def test_omega_squared_known_values():
    # ss_effect=10, df_effect=2, ss_total=100, ms_error=2 -> omega2=(10-4)/(100+2)
    val = ai.omega_squared(ss_effect=10.0, df_effect=2, ss_total=100.0, ms_error=2.0)
    assert val == pytest.approx((10.0 - 2 * 2.0) / (100.0 + 2.0))


def test_omega_squared_clips_negative_to_zero():
    val = ai.omega_squared(ss_effect=1.0, df_effect=2, ss_total=100.0, ms_error=5.0)
    assert val == 0.0


def test_run_anova_for_target_type_agreement_and_shape(design_df):
    df = pd.read_csv(MASTER_TABLE)
    full = pd.concat([design_df, df[["compaction_density"]]], axis=1)
    result = ai.run_anova_for_target(full, "compaction_density", n_perm=200, seed=0)

    assert result["type_agreement"] is True
    table = result["table"]
    assert set(table.index) == set(ai.TERM_ORDER) | {"Residual"}
    assert table.loc["Residual", "df"] == 48
    assert (table.loc[list(ai.TERM_ORDER), "eta2"] >= 0).all()
    # compaction_density precursor share should reproduce paper's 67.4%
    assert table.loc["precursor", "eta2"] == pytest.approx(0.674, abs=0.002)
    perm_p = table.loc[list(ai.TERM_ORDER), "perm_p"]
    assert perm_p.between(0, 1).all()


def test_permutation_null_is_roughly_uniform_for_unrelated_y(design_df):
    rng = np.random.default_rng(0)
    full = design_df.copy()
    full["noise_y"] = rng.normal(size=len(full))
    result = ai.run_anova_for_target(full, "noise_y", n_perm=500, seed=1)
    perm_p = result["table"].loc[list(ai.TERM_ORDER), "perm_p"]
    # pure noise: no term should look extremely significant
    assert (perm_p < 0.01).sum() <= 1
