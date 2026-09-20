# -*- coding: utf-8 -*-
"""
scaling.py —— 折内标准化(防泄漏)
==================================
标准化参数只能在每折训练子集上 fit,再应用于验证样本。
本模块只暴露一个 scaler_factory,交给 cross_validation.run_cv 在折内调用。
"""
from __future__ import annotations

from sklearn.preprocessing import StandardScaler


def scaler_factory():
    return StandardScaler()
