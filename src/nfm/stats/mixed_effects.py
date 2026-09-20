# -*- coding: utf-8 -*-
"""
mixed_effects.py —— 混合效应模型(X.8,统计主力)
================================================
  Y_ij = β0 + β1·Θ_j + β2·P_i + β3·Θ_j·P_i + u_j + ε_ij
  P_i:前驱体类型;u_j:炉次(condition_id)随机效应;β3:Θ×前驱体交互。

直接检验"前驱体效应是否随热暴露变化",并正确处理同炉样本的非独立性。
交互项 β3 是区分"鲁棒遗传"与"状态转变"两种叙事的统计判据。

依赖 statsmodels;前驱体作为分类变量(以 S 为基准)。
"""
from __future__ import annotations

import pandas as pd

try:
    import statsmodels.formula.api as smf
    _HAS_SM = True
except Exception:  # noqa
    _HAS_SM = False


def fit_mixed_effects(df: pd.DataFrame, y_col: str,
                      theta_col: str = "Theta",
                      precursor_col: str = "precursor",
                      group_col: str = "condition_id"):
    """拟合含 Θ×前驱体交互的混合效应模型。返回结果对象与交互项检验摘要。"""
    if not _HAS_SM:
        raise ImportError("未安装 statsmodels")
    d = df[[y_col, theta_col, precursor_col, group_col]].dropna().copy()
    d[precursor_col] = d[precursor_col].astype("category")
    formula = f"{y_col} ~ {theta_col} * C({precursor_col})"
    md = smf.mixedlm(formula, d, groups=d[group_col])
    res = md.fit(method="lbfgs", reml=True)

    # 抽取交互项(Θ:前驱体)系数与 p 值
    interaction = {k: {"coef": float(v), "pval": float(res.pvalues[k])}
                   for k, v in res.params.items()
                   if (":" in k and theta_col in k)}
    return {"result": res, "summary_text": res.summary().as_text(),
            "interaction_terms": interaction,
            "interpretation": _interpret(interaction)}


def _interpret(interaction: dict) -> str:
    if not interaction:
        return "无交互项(检查模型设定)"
    sig = [k for k, v in interaction.items() if v["pval"] < 0.05]
    if sig:
        return (f"交互项显著({sig}):前驱体效应随热暴露变化 → 倾向'状态转变/记忆擦除'叙事")
    return "交互项不显著:前驱体效应在各热暴露下稳定 → 倾向'鲁棒遗传'叙事"
