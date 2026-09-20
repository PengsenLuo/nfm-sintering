# -*- coding: utf-8 -*-
"""
mediation.py —— 调节中介分析(X.8,审慎使用)
=============================================
路径:前驱体级配 → 双峰记忆 M_B → 压实密度,以 Θ 为调节变量。
分解:总效应、经 M_B 的间接效应、Θ 是否削弱该间接效应。

定位:作为机制的**支持性**证据呈现,不过度声张因果;更复杂的层级贝叶斯/
结构方程仅在审稿需要时补充,不进主线。本实现给出 Baron–Kenny 风格的两步回归
+ Bootstrap 间接效应置信区间,并显式打印"中介≠因果证明"的告诫。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
    _HAS_SM = True
except Exception:  # noqa
    _HAS_SM = False

_CAVEAT = ("⚠ 中介分析为支持性证据:统计中介成立不等于因果中介成立;"
           "不以'直接模型与两步模型 R² 接近'作为中介的主要证据。")


def moderated_mediation(df: pd.DataFrame,
                        x="blend_ratio", m="M_B", y="compaction_density",
                        mod="Theta", n_boot=2000, seed=42):
    """对 M 组(M_B 有值)做调节中介。返回间接效应点估计与 Bootstrap CI。"""
    if not _HAS_SM:
        raise ImportError("未安装 statsmodels")
    d = df[[x, m, y, mod]].dropna().copy()
    if len(d) < 8:
        return {"note": "有效样本不足(<8),不做中介", "caveat": _CAVEAT}

    # a 路径:x(×mod) → m ;b 路径:m(×mod) → y(控制 x)
    a_model = smf.ols(f"{m} ~ {x} * {mod}", d).fit()
    b_model = smf.ols(f"{y} ~ {m} * {mod} + {x}", d).fit()
    a = a_model.params.get(x, np.nan)
    b = b_model.params.get(m, np.nan)
    indirect = a * b

    rng = np.random.default_rng(seed)
    boots = []
    idx = np.arange(len(d))
    for _ in range(n_boot):
        s = d.iloc[rng.choice(idx, len(idx), replace=True)]
        try:
            aa = smf.ols(f"{m} ~ {x} * {mod}", s).fit().params.get(x, np.nan)
            bb = smf.ols(f"{y} ~ {m} * {mod} + {x}", s).fit().params.get(m, np.nan)
            boots.append(aa * bb)
        except Exception:  # noqa
            continue
    boots = np.asarray([v for v in boots if np.isfinite(v)])
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) \
        if boots.size else (np.nan, np.nan)

    return {
        "a_path_coef": float(a), "b_path_coef": float(b),
        "indirect_effect": float(indirect),
        "indirect_95CI_boot": ci,
        "significant": bool(ci[0] * ci[1] > 0),   # CI 不跨 0
        "caveat": _CAVEAT,
    }
