# -*- coding: utf-8 -*-
"""
baselines.py —— 基线模型工厂(X.7.3)
=====================================
三类机理互补的经典模型,均用浅结构 + 强正则以适配小样本(81 / 27 独立条件)。
作用:为物理约束模型提供"必须超越"的精度参照,并以模型族一致性交叉验证结论稳健性。
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:  # noqa
    _HAS_XGB = False


def make_rf(params):
    return RandomForestRegressor(random_state=42, n_jobs=-1, **params)


def make_xgb(params):
    if not _HAS_XGB:
        raise ImportError("未安装 xgboost")
    return XGBRegressor(random_state=42, n_jobs=-1, objective="reg:squarederror",
                        **params)


def make_svr(params):
    return SVR(**params)


def factory_from_config(name: str, cfg):
    base = cfg.raw["training"]["baselines"]
    if name == "rf":
        return lambda: make_rf(base["rf"])
    if name == "xgb":
        return lambda: make_xgb(base["xgb"])
    if name == "svr":
        return lambda: make_svr(base["svr"])
    raise ValueError(f"未知基线:{name}")
