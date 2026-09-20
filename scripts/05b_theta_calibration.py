# -*- coding: utf-8 -*-
"""05b Θ 起算温度标定诊断。只做扫描诊断,不写回 config.yaml——
起算温度需人工看完诊断结果后决定,决定前 thermal_exposure.integrate_from_C
维持 'auto'(550℃回退)不变。"""
import _bootstrap  # noqa
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from nfm.config import load_config, build_condition_map
from nfm.features.thermal_exposure import theta
from nfm.nano_layer import nano_layer_frame

CANDIDATES = list(range(500, 701, 25))   # 500,525,...,700


def main():
    cfg = load_config()
    cm = build_condition_map(cfg)  # 27 行:condition_id, T_C, beta, t_hold
    xrd = pd.read_csv("data/interim/xrd_features.csv")
    # 纳米层口径定档(2026-07-30):D_XRD 只用仪器1(SmartLab)51样,见 nfm.nano_layer。
    xrd = nano_layer_frame(xrd, require_reliable=True)
    xrd["condition_id"] = xrd["sample_id"].str.extract(r"^(C\d{2})-")

    Q_J, R = cfg.Q_ref_J, cfg.R
    T_ref_C = cfg.raw["physical_constants"]["T_ref_C"]

    rows, theta_by_candidate = [], {}
    for start_C in CANDIDATES:
        th = {r.condition_id: theta(r.T_C, r.beta, r.t_hold, Q_J=Q_J, R=R,
                                    T_ref_C=T_ref_C, start_C=start_C)
              for r in cm.itertuples()}
        theta_by_candidate[start_C] = th
        merged = xrd.merge(pd.Series(th, name="Theta").rename_axis("condition_id"),
                           on="condition_id", how="left").dropna(subset=["Theta", "D_XRD"])
        x = np.log(merged["Theta"].to_numpy())
        y = merged["D_XRD"].to_numpy()
        b, a = np.polyfit(x, y, 1)
        yhat = a + b * x
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        rows.append(dict(start_C=start_C, r2_D_XRD_vs_lnTheta=r2,
                         n_xrd_samples=len(merged)))

    summary = pd.DataFrame(rows)

    # 27条件排序稳定性:相邻候选值之间的 Spearman 相关
    rank_rows = []
    for i in range(len(CANDIDATES) - 1):
        c1, c2 = CANDIDATES[i], CANDIDATES[i + 1]
        th1 = theta_by_candidate[c1]
        th2 = theta_by_candidate[c2]
        cids = sorted(th1)
        rho, _ = spearmanr([th1[c] for c in cids], [th2[c] for c in cids])
        rank_rows.append(dict(start_C_pair=f"{c1}-{c2}", spearman_rho=rho))
    rank_df = pd.DataFrame(rank_rows)

    out_csv = Path("data/interim/theta_calibration_summary.csv")
    summary.to_csv(out_csv, index=False)
    rank_csv = Path("data/interim/theta_calibration_rank_stability.csv")
    rank_df.to_csv(rank_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(summary.start_C, summary.r2_D_XRD_vs_lnTheta, "o-")
    axes[0].set_xlabel("候选起算温度 (°C)")
    axes[0].set_ylabel("D_XRD ~ ln(Θ) 拟合 R²")
    axes[0].set_title(f"拟合优度 vs 起算温度 (n={summary.n_xrd_samples.iloc[0]} XRD样本)")
    axes[1].plot(range(len(rank_df)), rank_df.spearman_rho, "s-")
    axes[1].set_xticks(range(len(rank_df)))
    axes[1].set_xticklabels(rank_df.start_C_pair, rotation=45)
    axes[1].set_ylabel("相邻候选值 27条件Θ排序 Spearman ρ")
    axes[1].set_title("排序稳定性")
    plt.tight_layout()
    plt.savefig("data/interim/theta_calibration_diagnostic.png", dpi=120)

    print(summary.to_string(index=False))
    print(rank_df.to_string(index=False))
    best = summary.loc[summary.r2_D_XRD_vs_lnTheta.idxmax()]
    print(f"\n候选中 R² 最高: start_C={best.start_C:.0f}, R²={best.r2_D_XRD_vs_lnTheta:.4f}")
    print("⚠️ 本脚本只做诊断,不写回 config.yaml——起算温度需人工确认后手动固化。")


if __name__ == "__main__":
    main()
