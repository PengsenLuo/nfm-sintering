# -*- coding: utf-8 -*-
"""25 §3.5 条件级残差跨前驱体 Pearson r 表
====================================================================
数据源:data/interim/sieving_confound_diagnostic.csv 的
residual_precursor_adj_full27 列(扣除前驱体主效应后的条件级残差,27条件)。
按 condition_id x precursor 透视后,逐对算 Pearson r/p,得到
S-L/M-L/S-M 三对相关。

输出:data/interim/sieving_residual_pearson.csv
"""
import _bootstrap  # noqa

import pandas as pd
from scipy.stats import pearsonr

SRC = "data/interim/sieving_confound_diagnostic.csv"
COL = "residual_precursor_adj_full27"
OUT = "data/interim/sieving_residual_pearson.csv"
PAIRS = [("S", "L"), ("M", "L"), ("S", "M")]


def main():
    df = pd.read_csv(SRC)
    piv = df.pivot(index="condition_id", columns="precursor", values=COL)
    rows = []
    for a, b in PAIRS:
        r, p = pearsonr(piv[a], piv[b])
        n = len(piv)
        print(f"{a}-{b}: r={r:+.4f}  p={p:.5f}  n={n}")
        rows.append({"pair": f"{a}-{b}", "pearson_r": r, "p": p, "n": n})
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\n已写出 {OUT}")


if __name__ == "__main__":
    main()
