# -*- coding: utf-8 -*-
"""07 基线 + 物理约束模型训练/验证。★ 默认 LOCO 为主,--cv loo 辅助。"""
import _bootstrap  # noqa
import argparse
import pandas as pd
from nfm.config import load_config
from nfm.models.baselines import factory_from_config
from nfm.features.scaling import scaler_factory
from nfm.evaluation.cross_validation import run_cv


def main(cv_scheme):
    cfg = load_config()
    df = pd.read_csv("data/processed/master_table.csv")
    view = cfg.view("view_direct")          # 以压实密度直接路线为例
    X_cols, y_col = view["X"], view["y"][0]
    X_cols = [c for c in X_cols if c in df.columns]
    if df[X_cols + [y_col]].dropna().shape[0] < 10:
        print("⚠ 有效样本不足(数据未齐),仅演示流程。")
    print(f"=== 目标 {y_col} | CV={cv_scheme} | 特征 {X_cols} ===")
    for name in ("rf", "xgb", "svr"):
        try:
            fac = factory_from_config(name, cfg)
            res = run_cv(fac, df.dropna(subset=X_cols + [y_col]),
                         X_cols, y_col, scheme=cv_scheme,
                         scaler_factory=scaler_factory)
            s = res["summary"]
            print(f"  {name.upper():5s}  R²={s['R2']:.3f}  MAE={s['MAE']:.4f}  "
                  f"(folds={s['n_folds']})")
        except Exception as e:  # noqa
            print(f"  {name.upper():5s}  跳过:{e}")
    print("  物理约束模型(共享主干+可学习 Q)结构见 nfm.models.pinn;"
          "TorchSklearnWrapper.fit 已接入 losses.L_Arrhenius/L_mono/L_bound"
          "(按 config.yaml.training.pinn.physics 分派)。"
          "完整 LOCO 评估 + 三模型对照见 scripts/07b_pinn_physics_eval.py。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", default="loco", choices=["loco", "loo", "lot", "lop"])
    main(ap.parse_args().cv)
