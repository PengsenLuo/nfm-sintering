# -*- coding: utf-8 -*-
"""
07f 共享主干(view_pinn_full,7 target)上的 arrhenius 权重扫描
================================================================
arrhenius 权重的网格扫描原定在"独立主干"上做
(scripts/07d_dxrd_baseline_sweep.py)。独立主干扫描完成后发现一个意外结果:
D_XRD 在共享主干、物理关闭时(R2=0.386~0.387)明显优于独立主干、物理关闭时
(R2=0.197~0.200)——说明"给 D_XRD 单独一个 trunk"这个 2a 的首选修复方向本身
是错的方向,真正的杠杆是 arrhenius 权重,不是是否共享主干。

本脚本在生产架构(共享 7-target 主干)上补做 weight.arrhenius in
{0, 0.05, 0.1, 0.5} 的完整扫描,同时报告 D_XRD 与 compaction_density 两列,
用于完整展示"任意非零权重都让 D_XRD 断崖式变差,且通过共享主干拖累其它
target"这一结论,不是只挑一个好看的点。

只读 config/master_table,不改 schema/处理层口径。输出:
  data/interim/shared_trunk_arrhenius_weight_sweep.csv
"""
import _bootstrap  # noqa
import time

import pandas as pd

from nfm.config import load_config
from nfm.evaluation.cross_validation import run_cv_multi
from nfm.models.pinn import TorchSklearnWrapper

WEIGHTS = [0.0, 0.05, 0.1, 0.5]


def main():
    cfg = load_config()
    df = pd.read_csv("data/processed/master_table.csv")
    view = cfg.view("view_pinn_full")
    X_cols, y_cols = view["X"], view["y"]
    df_fit = df.dropna(subset=X_cols)
    cfg_pinn = cfg.raw["training"]["pinn"]

    print("=" * 70)
    print("共享主干(7 target)上的 arrhenius 权重扫描")
    print("=" * 70)

    rows = []
    for w in WEIGHTS:
        variant = dict(cfg_pinn)
        variant["loss_weights"] = dict(cfg_pinn["loss_weights"])
        variant["loss_weights"]["arrhenius"] = w
        t0 = time.time()
        res = run_cv_multi(
            lambda: TorchSklearnWrapper(in_dim=len(X_cols), target_names=y_cols,
                                        cfg_pinn=variant, X_cols=X_cols, seed=42),
            df_fit, X_cols, y_cols, scheme="loco")
        dt = time.time() - t0
        row = {"arrhenius_weight": w, "time_s": dt}
        for name in y_cols:
            s = res["summary"][name]
            row[f"{name}_R2"] = s["R2"]
            row[f"{name}_MAE"] = s["MAE"]
        rows.append(row)
        print(f"  w_arr={w:<5g} D_XRD R2={row['D_XRD_R2']:+.4f} MAE={row['D_XRD_MAE']:.3f}  "
              f"compaction_density R2={row['compaction_density_R2']:.4f}  "
              f"circularity R2={row['circularity_R2']:.4f}  "
              f"convexity R2={row['convexity_R2']:.4f}  "
              f"D_sec R2={row['D_sec_R2']:.4f}  ({dt:.1f}s)")

    out = pd.DataFrame(rows)
    out.to_csv("data/interim/shared_trunk_arrhenius_weight_sweep.csv", index=False)
    print("\n已写出 data/interim/shared_trunk_arrhenius_weight_sweep.csv")


if __name__ == "__main__":
    main()
