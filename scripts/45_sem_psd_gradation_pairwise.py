# -*- coding: utf-8 -*-
"""45 27条件 M-vs-L 级配传递配对检验
====================================================================
用 SEM 口径的 D10_num/D50_num/D90_num/Span_num/Span_vol(scripts/44 产出,
已基于修复根因缓存 bug 后重新生成的 sem_shape_descriptors.csv),在 27 个
工艺条件上做同炉 M vs L 配对差(中位数、方向计数、Wilcoxon 符号秩检验),
并核对 Spearman(D50_num,Theta)/Spearman(D50_vol,Theta) 按前驱体分组。
"""
import _bootstrap  # noqa

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

SRC = Path("data/interim/sem_psd_quantiles.csv")
REPORT_MD = Path("reports/sem_psd_gradation.md")

PAIR_COLS = ["D10_num", "D50_num", "D90_num", "Span_num", "Span_vol"]


def main():
    df = pd.read_csv(SRC)
    assert df["sample_id"].nunique() == 81

    lines = ["# SEM 口径产物粒度级配传递证据\n",
            "\n> 由 `scripts/45_sem_psd_gradation_pairwise.py` 生成。"
            "本报告不引用任何 9 样水测激光粒度数字,基于修复根因缓存 bug 后"
            "重新生成的 `sem_shape_descriptors.csv` 独立计算。\n\n"]

    lines.append("## 前驱体分组中位数(81 样)\n\n")
    g = df.groupby("precursor")[PAIR_COLS].median().reindex(["S", "M", "L"])
    lines.append(g.to_markdown() + "\n\n")
    print("=== 前驱体分组中位数 ===")
    print(g)

    lines.append("## 27 条件配对差(M − L)\n\n")
    lines.append("| 指标 | M中位 | L中位 | M−L中位差 | M>L次数 | Wilcoxon p |\n")
    lines.append("|---|---|---|---|---|---|\n")
    print("\n=== 27 条件配对差(M-L) ===")
    for col in PAIR_COLS:
        piv = df.pivot(index="condition_id", columns="precursor", values=col)
        assert piv.shape[0] == 27
        diff = piv["M"] - piv["L"]
        stat, p = wilcoxon(diff)
        n_pos = int((diff > 0).sum())
        median_diff = float(diff.median())
        lines.append(f"| {col} | {piv['M'].median():.3f} | {piv['L'].median():.3f} | "
                     f"{median_diff:+.4f} | {n_pos}/27 | {p:.4g} |\n")
        print(f"  {col}: median_diff={median_diff:+.4f} n_pos={n_pos}/27 p={p:.4g}")

    lines.append("\n## Spearman(D50, Theta) 分前驱体\n\n")
    lines.append("| 前驱体 | rho(D50_num) | p | rho(D50_vol) | p |\n|---|---|---|---|---|\n")
    print("\n=== Spearman(D50, Theta) 分前驱体 ===")
    for prec in ["S", "M", "L"]:
        sub = df[df.precursor == prec]
        rho_n, p_n = spearmanr(sub["D50_num"], sub["Theta"])
        rho_v, p_v = spearmanr(sub["D50_vol"], sub["Theta"])
        lines.append(f"| {prec} | {rho_n:+.3f} | {p_n:.4g} | {rho_v:+.3f} | {p_v:.4g} |\n")
        print(f"  {prec}: D50_num rho={rho_n:+.3f} p={p_n:.4g}  D50_vol rho={rho_v:+.3f} p={p_v:.4g}")

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("".join(lines), encoding="utf-8")
    print(f"\n[done] {REPORT_MD}")


if __name__ == "__main__":
    main()
