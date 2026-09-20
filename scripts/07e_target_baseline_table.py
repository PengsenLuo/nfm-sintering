# -*- coding: utf-8 -*-
"""
07e 全 target 基线对照表
========================
为 view_pinn_full 的全部 7 个 target 建立同一套平价基线
(ln(Theta)+Theta+precursor one-hot 的 LOCO Ridge,见
nfm.evaluation.cross_validation.theta_precursor_baseline),与当前
PINN(config.yaml.training.pinn 默认配置,arrhenius weight=0)
逐个对照,标出 PINN 未跑赢基线的 target。

此表是模型改动的标准验收件——若缺了它,容易出现"结构层负 R²却
迟迟未被发现"的问题。

只读 config/master_table,不改 schema/处理层口径。输出:
  data/interim/target_baseline_comparison.csv
"""
import _bootstrap  # noqa

import pandas as pd

from nfm.config import load_config
from nfm.evaluation.cross_validation import run_cv_multi, theta_precursor_baseline
from nfm.models.pinn import TorchSklearnWrapper


def main():
    cfg = load_config()
    df = pd.read_csv("data/processed/master_table.csv")
    view = cfg.view("view_pinn_full")
    X_cols, y_cols = view["X"], view["y"]
    df_fit = df.dropna(subset=X_cols)
    cfg_pinn = cfg.raw["training"]["pinn"]

    print("=" * 70)
    print("任务3:全部7个 target 的平价基线对照表")
    print("PINN 配置:config.yaml.training.pinn(任务2定稿:arrhenius weight=0)")
    print("=" * 70)

    res = run_cv_multi(
        lambda: TorchSklearnWrapper(in_dim=len(X_cols), target_names=y_cols,
                                    cfg_pinn=cfg_pinn, X_cols=X_cols, seed=42),
        df_fit, X_cols, y_cols, scheme="loco")

    rows = []
    for name in y_cols:
        pinn_s = res["summary"][name]
        base_s = theta_precursor_baseline(df, name)
        beats = (pinn_s["R2"] >= base_s["R2"]) and (pinn_s["MAE"] <= base_s["MAE"])
        rows.append({
            "target": name,
            "baseline_R2": base_s["R2"], "baseline_MAE": base_s["MAE"],
            "baseline_n_folds": base_s["n_folds"],
            "PINN_R2": pinn_s["R2"], "PINN_MAE": pinn_s["MAE"],
            "PINN_n_eval": pinn_s["n_eval"],
            "beats_baseline": beats,
        })

    out = pd.DataFrame(rows)
    out.to_csv("data/interim/target_baseline_comparison.csv", index=False)

    print("\n" + out.to_string(index=False))

    flagged = out[~out["beats_baseline"]]
    print("\n" + "=" * 70)
    if len(flagged) == 0:
        print("全部 target 均跑赢平价基线。")
    else:
        print(f"PINN 未跑赢基线的 target({len(flagged)}/{len(out)}):"
              f" {', '.join(flagged['target'].tolist())}")
    print("\n已写出 data/interim/target_baseline_comparison.csv")


if __name__ == "__main__":
    main()
