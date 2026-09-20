# -*- coding: utf-8 -*-
"""test_dxrd_baseline.py —— compaction_density/D_sec/D_XRD 表6/表7 唯一合法
Ridge(alpha=1,原始尺度)+LOCO 实现

模型规格演变(锁定本文件数字所依据的最终规格,理由见
`src/nfm/dxrd_baseline.py` 模块 docstring):
最初把这套计算实现为 OLS(dxrd_ols_loco,仅 D_XRD)。复核发现表6另外两行
(compaction_density/D_sec)本就是 Ridge(alpha=1)原始尺度(不标准化,不
drop_first)算出的,且数字精确吻合;同一规格下 D_XRD 的 LOCO 才能复现
论文原稿的 R²=0.494(OLS 是 0.4886,标准化 Ridge 是 0.5025,都对不上)。

自变量去掉了 Θ 项,只留 lnΘ。前驱体编码曾改成"以 S 为参照类,仅 M/L 两个
哑元 + 截距",但这个编码选择让 S(被吸收进不受 Ridge 惩罚的截距)与 M/L
(仍受罚)在正则化项下不再对称,对 `D_sec`(S 是最极端值)造成实质伤害
(LOCO R² 0.9180→0.9132,MAE 0.6218→0.6688),是编码选择的副作用,与
"去掉Θ"本身无关。

最终改用 **sum(偏差)编码**(`I_M`/`I_L` 两列,S 在两列都取 -1,不再被
截距单独吸收)——三个前驱体重新对称受罚,`D_sec` 的 LOCO 不仅恢复还反超
此前的生产基线(R²=0.9225,MAE=0.5980,均优于 lnΘ+Θ 旧基线的
0.9180/0.6218)。sum 编码下四个 target(含 circularity)的 LOCO 全部达到
或超过各自此前的最优值。本文件锁定的数字来自这套最终规格下的独立复核。
"""
from __future__ import annotations

import pandas as pd
import pytest

from nfm.dxrd_baseline import ridge_coef_loco


def test_ridge_coef_loco_scope_by_target():
    df = pd.read_csv("data/processed/master_table.csv")
    d_xrd = ridge_coef_loco(df, "D_XRD")
    assert d_xrd["n"] == 51 and d_xrd["n_cond"] == 17 and d_xrd["n_folds"] == 17
    assert d_xrd["n_eval"] == 51
    assert set(d_xrd["coef"].keys()) == {
        "intercept", "ln_Theta", "precursor_M", "precursor_L"
    }
    assert len(d_xrd["fold_rows"]) == 51

    comp = ridge_coef_loco(df, "compaction_density")
    assert comp["n"] == 81 and comp["n_cond"] == 27 and comp["n_folds"] == 27

    dsec = ridge_coef_loco(df, "D_sec")
    assert dsec["n"] == 81 and dsec["n_cond"] == 27 and dsec["n_folds"] == 27


def test_ridge_coef_loco_matches_verified_numbers():
    """回归锁定:lnΘ-only + sum(偏差)编码落地后独立复核的
    Ridge(alpha=1,原始尺度)系数与 LOCO 数字。
    """
    df = pd.read_csv("data/processed/master_table.csv")

    d_xrd = ridge_coef_loco(df, "D_XRD")
    assert d_xrd["R2"] == pytest.approx(0.56203, abs=1e-4)
    assert d_xrd["MAE"] == pytest.approx(8.25155, abs=1e-3)
    c = d_xrd["coef"]
    assert c["intercept"] == pytest.approx(25.68016, abs=1e-3)
    assert c["ln_Theta"] == pytest.approx(14.48207, abs=1e-3)
    assert c["precursor_M"] == pytest.approx(-1.60470, abs=1e-3)
    assert c["precursor_L"] == pytest.approx(-0.47698, abs=1e-3)

    comp = ridge_coef_loco(df, "compaction_density")
    assert comp["R2"] == pytest.approx(0.63786, abs=1e-3)
    assert comp["MAE"] == pytest.approx(0.06151, abs=1e-3)
    cc = comp["coef"]
    assert cc["intercept"] == pytest.approx(3.31718, abs=1e-4)
    assert cc["ln_Theta"] == pytest.approx(0.00018, abs=1e-4)
    assert cc["precursor_M"] == pytest.approx(0.07079, abs=1e-4)
    assert cc["precursor_L"] == pytest.approx(0.07665, abs=1e-4)

    dsec = ridge_coef_loco(df, "D_sec")
    assert dsec["R2"] == pytest.approx(0.92247, abs=1e-3)
    assert dsec["MAE"] == pytest.approx(0.59805, abs=1e-3)
    dc = dsec["coef"]
    assert dc["intercept"] == pytest.approx(6.44573, abs=1e-3)
    assert dc["ln_Theta"] == pytest.approx(0.75983, abs=1e-3)
    assert dc["precursor_M"] == pytest.approx(1.84449, abs=1e-3)
    assert dc["precursor_L"] == pytest.approx(1.96789, abs=1e-3)


def test_ridge_coef_loco_circularity_unified_path():
    """circularity 通过与 compaction_density/D_sec/D_XRD 相同的统一
    路径计算(81样,不做纳米层限定),切到 sum 编码后数值已同步更新。"""
    df = pd.read_csv("data/processed/master_table.csv")
    circ = ridge_coef_loco(df, "circularity")
    assert circ["n"] == 81 and circ["n_cond"] == 27 and circ["n_folds"] == 27
    assert circ["R2"] == pytest.approx(0.84672, abs=1e-3)
    assert circ["MAE"] == pytest.approx(0.04180, abs=1e-3)
