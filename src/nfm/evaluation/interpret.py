# -*- coding: utf-8 -*-
"""
interpret.py —— 可解释性(X.7.6)
=================================
SHAP(局部归因)+ Sobol(全局方差分解)。预期:
  - 热暴露量 Θ 主导一次颗粒/相干畴类目标;
  - 前驱体描述符主导粒度遗传类目标;
  - 双峰记忆在压实密度中显著正贡献。
与 X.4–X.6 机制结论互证。
"""
from __future__ import annotations

import numpy as np


def shap_summary(model, X, feature_names):
    """对树模型用 TreeExplainer,其它用 KernelExplainer(小样本可行)。返回 shap 值。"""
    import shap
    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X)
    except Exception:  # noqa
        bg = shap.sample(X, min(20, len(X)), random_state=42)
        explainer = shap.KernelExplainer(model.predict, bg)
        values = explainer.shap_values(X, nsamples=100)
    mean_abs = np.abs(values).mean(axis=0)
    ranking = sorted(zip(feature_names, mean_abs), key=lambda kv: -kv[1])
    return {"shap_values": values, "ranking": ranking}


def sobol_indices(predict_fn, problem, n=512, seed=42):
    """Sobol 一阶/总阶指数。problem 为 SALib 格式 dict(num_vars/names/bounds)。"""
    from SALib.sample import sobol as sobol_sample
    from SALib.analyze import sobol as sobol_analyze

    param_values = sobol_sample.sample(problem, n, seed=seed)
    Y = np.array([predict_fn(row) for row in param_values])
    Si = sobol_analyze.analyze(problem, Y, seed=seed)
    return {"S1": dict(zip(problem["names"], Si["S1"])),
            "ST": dict(zip(problem["names"], Si["ST"]))}
