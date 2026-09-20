# -*- coding: utf-8 -*-
"""
07d D_XRD 达标修复
====================
硬性验收线:D_XRD 的 LOCO 表现 R² >= 0.49 且 MAE <= 8.8 nm
(即不输给 ln(Theta)+Theta+precursor one-hot 的 Ridge 基线,见
nfm.evaluation.cross_validation.theta_precursor_baseline)。

按顺序做实验,每步都记录对 D_XRD LOCO 的影响:
  2a. D_XRD 独立主干(不与形貌/堆积层共享 trunk)——首选修复方法。
  2b. 若未达标,在独立主干基础上网格扫描 growth_exponent_n in {2,3,4}
      x weight.arrhenius in {0, 0.05, 0.1, 0.5}(monotonic/bounds 权重不变)。
  2c. 若关闭 Arrhenius(weight=0)才能达标 —— 明确记录为结论,不为了保留
      "PINN 头"而保留伤害性约束。

只读 config/master_table,不改 schema/处理层口径。输出完整网格结果:
  data/interim/dxrd_fix_grid.csv
"""
import _bootstrap  # noqa
import time

import pandas as pd

from nfm.config import load_config
from nfm.evaluation.cross_validation import run_cv_multi, theta_precursor_baseline
from nfm.models.pinn import TorchSklearnWrapper

R2_TARGET = 0.49
MAE_TARGET = 8.8


def run_variant(label, X_cols, df_fit, cfg_pinn_variant, target_names=("D_XRD",)):
    t0 = time.time()
    res = run_cv_multi(
        lambda: TorchSklearnWrapper(in_dim=len(X_cols), target_names=list(target_names),
                                    cfg_pinn=cfg_pinn_variant, X_cols=X_cols, seed=42),
        df_fit, X_cols, list(target_names), scheme="loco")
    dt = time.time() - t0
    s = res["summary"]["D_XRD"]
    passed = s["R2"] >= R2_TARGET and s["MAE"] <= MAE_TARGET
    print(f"  [{label}] R2={s['R2']:+.4f}  MAE={s['MAE']:.3f}nm  "
          f"n_eval={s['n_eval']}  {dt:.1f}s  {'PASS' if passed else 'fail'}")
    return {"label": label, "R2": s["R2"], "MAE": s["MAE"], "n_eval": s["n_eval"],
            "time_s": dt, "passed": passed}


def main():
    cfg = load_config()
    df = pd.read_csv("data/processed/master_table.csv")
    view = cfg.view("view_pinn_full")
    X_cols, y_cols = view["X"], view["y"]
    df_fit = df.dropna(subset=X_cols)
    cfg_pinn = cfg.raw["training"]["pinn"]

    baseline = theta_precursor_baseline(df, "D_XRD")
    print("=" * 70)
    print("硬性验收线(ln(Theta)+Theta+precursor one-hot Ridge, 同一批 LOCO 折):")
    print(f"  R2={baseline['R2']:.4f}  MAE={baseline['MAE']:.3f}nm  "
          f"n_eval={baseline['n_eval']}  n_folds={baseline['n_folds']}")
    print(f"  硬目标:R2 >= {R2_TARGET}  且  MAE <= {MAE_TARGET}nm")
    print("=" * 70)

    rows = []

    print("\n--- 现状对照:共享主干(7 target,任务4原配置)---")
    rows.append(run_variant("shared_trunk_7target(baseline_arch)", X_cols, df_fit,
                            cfg_pinn, target_names=y_cols))

    print("\n--- 2a: D_XRD 独立主干(默认超参 n=3, arrhenius weight=0.5)---")
    rows.append(run_variant("solo_trunk_default(n=3,w=0.5)", X_cols, df_fit, cfg_pinn))

    best = max(rows, key=lambda r: r["R2"])
    if best["passed"]:
        print(f"\n2a 已达标({best['label']}),跳过 2b 网格扫描。")
    else:
        print("\n--- 2b: 独立主干上网格扫描 growth_exponent_n x weight.arrhenius ---")
        for n_exp in (2.0, 3.0, 4.0):
            for w_arr in (0.0, 0.05, 0.1, 0.5):
                variant = dict(cfg_pinn)
                variant["growth_exponent_n"] = n_exp
                variant["loss_weights"] = dict(cfg_pinn["loss_weights"])
                variant["loss_weights"]["arrhenius"] = w_arr
                label = f"solo_trunk(n={n_exp:g},w_arr={w_arr:g})"
                rows.append(run_variant(label, X_cols, df_fit, variant))

    df_grid = pd.DataFrame(rows)
    df_grid["baseline_R2"] = baseline["R2"]
    df_grid["baseline_MAE"] = baseline["MAE"]
    df_grid.to_csv("data/interim/dxrd_fix_grid.csv", index=False)

    print("\n" + "=" * 70)
    print("完整网格结果(按 R2 降序):")
    print(df_grid.sort_values("R2", ascending=False).to_string(index=False))

    passing = df_grid[df_grid["passed"]]
    print("\n" + "=" * 70)
    if len(passing) == 0:
        print("结论:没有任何配置达标。D_XRD 在本数据集/本架构范围内无法追平基线。")
    else:
        winner = passing.sort_values("R2", ascending=False).iloc[0]
        print(f"结论:达标配置存在,最优为 [{winner['label']}] "
              f"R2={winner['R2']:.4f} MAE={winner['MAE']:.3f}nm")
        if "w_arr=0" in winner["label"] or "w_arr=0)" in winner["label"]:
            print("★ 该配置关闭了 Arrhenius 约束(weight=0)—— 明确结论:"
                  "该数据集与 Arrhenius 约束抵触,不应为保留物理约束头而牺牲精度。")

    print("\n已写出 data/interim/dxrd_fix_grid.csv")


if __name__ == "__main__":
    main()
