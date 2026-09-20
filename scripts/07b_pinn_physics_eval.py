# -*- coding: utf-8 -*-
"""
07b PINN 物理约束接入评估(2026-07-26)
==========================================
本脚本做两件事:

1. LOCO(主口径,27折)评估共享主干 PINN 在 view_pinn_full 全部 7 个
   target 上的表现,按物理约束目标(结构层)vs 未获约束目标
   (形貌/堆积层)分组报告 R²/MAE,并输出 learned Q 与文献 180-220 kJ/mol
   的对照。
2. 围绕 compaction_density 做三模型 LOCO 对照——
   (a) Θ-only(+前驱体 one-hot)的线性基线
       (b) 全特征、同架构但物理权重清零的 MLP(纯数据驱动消融)
       (c) 全特征 + 按目标分派物理约束的 PINN(与任务4同一个模型,
           从其 7 head 输出里取 compaction_density 列)

★ 物理项(arrhenius_loss)读的是 T_K/t_eff 的真实物理量纲,若在 CV 折内标准化
  会把 Kelvin/小时压缩成 z-score,破坏 Q_J/(R·T_K) 的物理意义——因此 (b)/(c)
  两个共享主干模型都不做特征标准化(只有 (a) 的线性基线用标准化,不受影响)。
  这是本轮设计里的一个关键正确性约束,不是遗漏。

只读 view/config,不改 schema、不改历史处理器口径。产出打印到 stdout,
同时写 CSV 到 data/interim/pinn_loco_summary.csv,供 docs 小结引用。
"""
import _bootstrap  # noqa
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from nfm.config import load_config
from nfm.evaluation.cross_validation import run_cv_multi
from nfm.evaluation.physics_check import check_Q
from nfm.models.pinn import TorchSklearnWrapper

STRUCTURE_TARGETS = ["D_XRD", "lattice_c", "c_a_ratio"]        # 工艺主导,受物理约束
MORPH_PACKING_TARGETS = ["D_sec", "circularity", "convexity", "compaction_density"]  # 分散体主导,未获约束


def _fmt_summary(summary: dict, names) -> pd.DataFrame:
    rows = []
    for n in names:
        s = summary[n]
        rows.append({"target": n, "R2": s["R2"], "MAE": s["MAE"], "n_eval": s["n_eval"]})
    return pd.DataFrame(rows)


def run_task4(cfg, df):
    print("\n" + "=" * 70)
    print("任务4:LOCO(27折)—— view_pinn_full 全部 7 个 target,共享主干 PINN")
    print("=" * 70)

    view = cfg.view("view_pinn_full")
    X_cols, y_cols = view["X"], view["y"]
    df_fit = df.dropna(subset=X_cols)  # X 无 NaN(工艺+前驱体+派生量全齐);y 允许 NaN
    print(f"样本数(X 齐全):{len(df_fit)}/81;target:{y_cols}")

    cfg_pinn = cfg.raw["training"]["pinn"]
    learned_Qs = []

    def factory():
        w = TorchSklearnWrapper(in_dim=len(X_cols), target_names=y_cols,
                                cfg_pinn=cfg_pinn, X_cols=X_cols, seed=42)
        orig_fit = w.fit

        def fit_and_record(X, Y):
            orig_fit(X, Y)
            learned_Qs.append(w.learned_Q_kJmol)
            return w
        w.fit = fit_and_record
        return w

    t0 = time.time()
    res = run_cv_multi(factory, df_fit, X_cols, y_cols, scheme="loco")  # ★ 不标准化,见模块 docstring
    dt = time.time() - t0
    print(f"LOCO {res['n_folds']} 折完成,用时 {dt:.1f}s")

    df_struct = _fmt_summary(res["summary"], STRUCTURE_TARGETS)
    df_morph = _fmt_summary(res["summary"], MORPH_PACKING_TARGETS)
    print("\n结构层(物理约束已启用:D_XRD 受 arrhenius+mono,lattice_c/c_a_ratio 仅 bounds):")
    print(df_struct.to_string(index=False))
    print("\n形貌/堆积层(未获约束,仅 bounds 兜底;circularity/convexity/compaction_density 明确排除 mono):")
    print(df_morph.to_string(index=False))

    Q_learn_mean = float(np.mean(learned_Qs))
    Q_learn_std = float(np.std(learned_Qs))
    Q_init = float(cfg_pinn["Q_init_kJmol"])
    q_check = check_Q(Q_learn_mean, Q_fit_mean_kJmol=Q_learn_mean,
                      Q_fit_ci_kJmol=(Q_learn_mean - Q_learn_std, Q_learn_mean + Q_learn_std))
    print(f"\nlearned Q(27 折均值±std)= {Q_learn_mean:.4f} ± {Q_learn_std:.4f} kJ/mol"
          f"(初始化 Q_init={Q_init:.1f},|Δ|={abs(Q_learn_mean - Q_init):.4f})")
    print(f"是否落在文献范围 180–220 kJ/mol:{q_check['in_literature']}")
    pd.DataFrame({"fold": range(len(learned_Qs)), "learned_Q_kJmol": learned_Qs}).to_csv(
        "data/interim/pinn_learned_Q_per_fold.csv", index=False)

    out = pd.concat([df_struct.assign(group="structure"),
                     df_morph.assign(group="morph_packing")], ignore_index=True)
    out.to_csv("data/interim/pinn_loco_summary.csv", index=False)
    return res, learned_Qs


def run_task5(cfg, df, pinn_loco_res):
    print("\n" + "=" * 70)
    print("任务5:compaction_density 三模型 LOCO 对照")
    print("=" * 70)

    # (c) 直接复用任务4已训练的共享主干 PINN 的 compaction_density 结果
    c_summary = pinn_loco_res["summary"]["compaction_density"]
    print(f"(c) PINN(全特征+按目标分派物理约束,共享主干)"
          f" R2={c_summary['R2']:.3f} MAE={c_summary['MAE']:.4f}")

    view = cfg.view("view_pinn_full")
    X_cols, y_cols = view["X"], view["y"]
    df_fit = df.dropna(subset=X_cols)
    cfg_pinn = cfg.raw["training"]["pinn"]
    cfg_noPhys = dict(cfg_pinn)
    cfg_noPhys["physics"] = {"arrhenius_targets": [], "mono_targets": [], "bounds_targets": []}

    t0 = time.time()
    res_b = run_cv_multi(
        lambda: TorchSklearnWrapper(in_dim=len(X_cols), target_names=y_cols,
                                    cfg_pinn=cfg_noPhys, X_cols=X_cols, seed=42),
        df_fit, X_cols, y_cols, scheme="loco")
    b_summary = res_b["summary"]["compaction_density"]
    print(f"(b) 全特征 MLP(同架构,物理权重清零) "
          f"R2={b_summary['R2']:.3f} MAE={b_summary['MAE']:.4f}  (用时 {time.time()-t0:.1f}s)")

    # (a) Θ-only + 前驱体 one-hot 的线性基线
    df_a = pd.get_dummies(df, columns=["precursor"], prefix="precursor")
    precursor_cols = [c for c in df_a.columns if c.startswith("precursor_")]
    df_a[precursor_cols] = df_a[precursor_cols].astype(float)
    Xa_cols = ["Theta"] + precursor_cols
    from nfm.features.scaling import scaler_factory
    res_a = run_cv_multi(lambda: Ridge(alpha=1.0), df_a, Xa_cols, ["compaction_density"],
                         scheme="loco", scaler_factory=scaler_factory)
    a_summary = res_a["summary"]["compaction_density"]
    print(f"(a) Θ-only(+前驱体 one-hot)线性基线 "
          f"R2={a_summary['R2']:.3f} MAE={a_summary['MAE']:.4f}")

    table = pd.DataFrame([
        {"model": "(a) Theta-only + precursor one-hot", "R2": a_summary["R2"], "MAE": a_summary["MAE"]},
        {"model": "(b) full-feature MLP, physics off", "R2": b_summary["R2"], "MAE": b_summary["MAE"]},
        {"model": "(c) full-feature PINN, physics on", "R2": c_summary["R2"], "MAE": c_summary["MAE"]},
    ])
    print("\n三模型对照汇总:")
    print(table.to_string(index=False))
    table.to_csv("data/interim/pinn_task5_compaction_density_comparison.csv", index=False)
    return table


def main():
    cfg = load_config()
    df = pd.read_csv("data/processed/master_table.csv")
    res4, learned_Qs = run_task4(cfg, df)
    table5 = run_task5(cfg, df, res4)
    print("\n已写出 data/interim/pinn_loco_summary.csv、"
          "data/interim/pinn_task5_compaction_density_comparison.csv")


if __name__ == "__main__":
    main()
