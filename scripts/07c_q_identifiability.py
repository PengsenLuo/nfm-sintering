# -*- coding: utf-8 -*-
"""
07c Q 可识别性诊断
====================
在完全相同的数据/架构(view_pinn_full 共享主干,7 target)下,把
Q_init_kJmol 分别设为 150/200/250,各跑一次全量拟合(81 样本)+ 一次
LOCO(27 折),记录:
  - 每种初始值下 learned Q 的收敛值(全量拟合)及 LOCO 27 折均值±std;
  - 训练过程中 Q 的梯度量级 vs Adam 实际步长量级(用于判断"未移动"是
    数据驱动的收敛还是被某种数值效应冻结)。

判据:
  - 三个初始值都收敛到彼此接近区间(两两差 < 15 kJ/mol)→ Q 可识别;
  - 各自收敛在离自己初始值很近处(位移 < 5%)→ Q 不可识别。

只读 config/master_table,不改 schema/处理层口径。输出:
  data/interim/q_identifiability_summary.csv
  data/interim/q_identifiability_gradient_diag.csv
"""
import _bootstrap  # noqa
import time

import numpy as np
import pandas as pd

from nfm.config import load_config
from nfm.evaluation.cross_validation import run_cv_multi
from nfm.models.pinn import TorchSklearnWrapper

Q_INITS = [150.0, 200.0, 250.0]


def _make_cfg(cfg_pinn, q_init):
    cfg = dict(cfg_pinn)
    cfg["Q_init_kJmol"] = q_init
    return cfg


def run_full_fit(cfg_pinn_variant, X_cols, y_cols, df_fit, q_init):
    w = TorchSklearnWrapper(in_dim=len(X_cols), target_names=y_cols,
                            cfg_pinn=cfg_pinn_variant, X_cols=X_cols, seed=42)
    t0 = time.time()
    w.fit(df_fit[X_cols].to_numpy(float), df_fit[y_cols].to_numpy(float))
    dt = time.time() - t0
    # 首步(epoch 0)与末步的梯度/步长量级,用于判断是否"数据驱动"。
    first = w.loss_history[0]
    last = w.loss_history[-1]
    step0 = abs(first["Q_kJmol"] - q_init)
    grad0 = abs(first["Q_grad"]) if first["Q_grad"] is not None else float("nan")
    n_epochs = len(w.loss_history)
    return {
        "Q_init": q_init,
        "learned_Q_full_fit": w.learned_Q_kJmol,
        "delta_full_fit": w.learned_Q_kJmol - q_init,
        "pct_shift_full_fit": (w.learned_Q_kJmol - q_init) / q_init * 100,
        "n_epochs_ran": n_epochs,
        "grad_epoch0": grad0,
        "step_epoch0": step0,
        "fit_time_s": dt,
    }, w


def run_loco(cfg_pinn_variant, X_cols, y_cols, df_fit, q_init):
    learned_Qs = []

    def factory():
        w = TorchSklearnWrapper(in_dim=len(X_cols), target_names=y_cols,
                                cfg_pinn=cfg_pinn_variant, X_cols=X_cols, seed=42)
        orig_fit = w.fit

        def fit_and_record(X, Y):
            orig_fit(X, Y)
            learned_Qs.append(w.learned_Q_kJmol)
            return w
        w.fit = fit_and_record
        return w

    t0 = time.time()
    res = run_cv_multi(factory, df_fit, X_cols, y_cols, scheme="loco")
    dt = time.time() - t0
    arr = np.asarray(learned_Qs)
    return {
        "Q_init": q_init,
        "learned_Q_loco_mean": float(arr.mean()),
        "learned_Q_loco_std": float(arr.std()),
        "delta_loco_mean": float(arr.mean() - q_init),
        "pct_shift_loco_mean": float((arr.mean() - q_init) / q_init * 100),
        "loco_time_s": dt,
    }, learned_Qs


def main():
    cfg = load_config()
    df = pd.read_csv("data/processed/master_table.csv")
    view = cfg.view("view_pinn_full")
    X_cols, y_cols = view["X"], view["y"]
    df_fit = df.dropna(subset=X_cols)
    cfg_pinn = cfg.raw["training"]["pinn"]

    print("=" * 70)
    print("任务1:Q 可识别性判定 —— Q_init in {150, 200, 250} kJ/mol")
    print("架构:与任务4相同的 view_pinn_full 共享主干(7 target)")
    print("=" * 70)

    full_rows, loco_rows, per_fold_rows = [], [], []
    for q_init in Q_INITS:
        variant = _make_cfg(cfg_pinn, q_init)
        print(f"\n--- Q_init = {q_init} kJ/mol ---")
        full_res, w = run_full_fit(variant, X_cols, y_cols, df_fit, q_init)
        print(f"  全量拟合(81样本):learned Q = {full_res['learned_Q_full_fit']:.4f} "
              f"kJ/mol, Δ={full_res['delta_full_fit']:+.4f} "
              f"({full_res['pct_shift_full_fit']:+.3f}%), "
              f"epoch0 grad={full_res['grad_epoch0']:.6g}, "
              f"epoch0 |step|={full_res['step_epoch0']:.6g}, "
              f"跑了 {full_res['n_epochs_ran']} epoch, {full_res['fit_time_s']:.2f}s")
        full_rows.append(full_res)

        loco_res, learned_Qs = run_loco(variant, X_cols, y_cols, df_fit, q_init)
        print(f"  LOCO(27折):learned Q = {loco_res['learned_Q_loco_mean']:.4f} "
              f"± {loco_res['learned_Q_loco_std']:.4f} kJ/mol, "
              f"Δmean={loco_res['delta_loco_mean']:+.4f} "
              f"({loco_res['pct_shift_loco_mean']:+.3f}%), "
              f"{loco_res['loco_time_s']:.1f}s")
        loco_rows.append(loco_res)
        for i, qv in enumerate(learned_Qs):
            per_fold_rows.append({"Q_init": q_init, "fold": i, "learned_Q_kJmol": qv})

    df_full = pd.DataFrame(full_rows)
    df_loco = pd.DataFrame(loco_rows)
    summary = df_full.merge(df_loco, on="Q_init")
    summary.to_csv("data/interim/q_identifiability_summary.csv", index=False)
    pd.DataFrame(per_fold_rows).to_csv("data/interim/q_identifiability_per_fold.csv", index=False)

    print("\n" + "=" * 70)
    print("汇总:")
    print(summary.to_string(index=False))

    # 判据
    means = summary["learned_Q_loco_mean"].to_numpy()
    pairwise_diffs = [abs(means[i] - means[j]) for i in range(len(means)) for j in range(i + 1, len(means))]
    max_pairwise_diff = max(pairwise_diffs)
    max_pct_shift = summary["pct_shift_loco_mean"].abs().max()

    print(f"\n三个初始值 LOCO learned-Q 均值两两最大差 = {max_pairwise_diff:.4f} kJ/mol"
          f"(阈值 <15 判定可识别)")
    print(f"相对自身初始值最大位移 = {max_pct_shift:.4f}%(阈值 <5% 判定不可识别)")

    if max_pairwise_diff < 15.0:
        verdict = "IDENTIFIABLE"
    elif max_pct_shift < 5.0:
        verdict = "NOT_IDENTIFIABLE"
    else:
        verdict = "AMBIGUOUS"
    print(f"\n判定:{verdict}")

    with open("data/interim/q_identifiability_verdict.txt", "w", encoding="utf-8") as f:
        f.write(f"verdict={verdict}\n")
        f.write(f"max_pairwise_diff_kJmol={max_pairwise_diff:.4f}\n")
        f.write(f"max_pct_shift={max_pct_shift:.4f}\n")

    print("\n已写出 data/interim/q_identifiability_summary.csv、"
          "data/interim/q_identifiability_per_fold.csv、"
          "data/interim/q_identifiability_verdict.txt")


if __name__ == "__main__":
    main()
