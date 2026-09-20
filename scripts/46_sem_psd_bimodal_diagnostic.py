# -*- coding: utf-8 -*-
"""46 SEM 数量口径双峰分解诊断
====================================================================
逐样本在 log(d_um) 上拟合 1/2 组分高斯混合,输出 BIC 差、两组分的峰位/
权重/分离度,并额外判断混合密度本身是否真的存在两个局部极大(而非两个
组分叠加成一个单峰)——只有同时满足"BIC 支持 2 组分"与"密度真有两个
局部极大"才记为 density_bimodal=True。本节结论为算法敏感的间接结果,
不作为级配传递的主证据(主证据见 scripts/45_sem_psd_gradation_pairwise.py
的配对检验)。

数据源为根因修复后重新生成的 data/interim/sem_shape_descriptors.csv
(见前序 commit),不使用 scripts/44 的分位数汇总表(混合拟合需要完整
逐颗粒样本,不是分位数)。

2026-08-30 执行记录:根因修复后重新生成的文件上,"density_bimodal"这个
离散计数(BIC 支持2组分 且 混合密度确有两个局部极大)首次出现 M(9)
略高于 L(8)——用 GaussianMixture 的 5 个不同 random_state(0/1/2/42/123)
重跑验证,M 稳定在 9-10、L 稳定在 8,不是单次拟合的偶然噪声。但这不构成
"M 比 L 更双峰"的显著证据:(a) 差值只有 27 个样本里的 1-2 个,该计数
是连续分离度在"是否达到两个局部极大"这一判据边界上的离散化,天生对
临界样本敏感;(b) 更稳健的连续量——分离度本身的 27 条件配对 Wilcoxon
检验——依然远不显著(p≈0.58),即"M 与 L 在分离度上无法区分"这一主
判断不受影响。故本脚本**如实报告**这个离散计数的微小、跨种子稳定的
差异,但不将其上升为"M 更双峰"的结论,也不为了凑回旧的"L≥M"而调整
判据定义——"相反的结论"应理解为统计显著意义上的反转,而不是一个跨种子
仅差 1-2 个样本的离散计数波动。
"""
import _bootstrap  # noqa

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import wilcoxon
from sklearn.mixture import GaussianMixture

SHAPE_CSV = Path("data/interim/sem_shape_descriptors.csv")
MASTER_TABLE = Path("data/processed/master_table.csv")
OUT_CSV = Path("data/interim/sem_psd_bimodal_diagnostic.csv")
REPORT_MD = Path("reports/sem_psd_gradation.md")


ROBUSTNESS_SEEDS = (0, 1, 2, 42, 123)


def _is_bimodal_for_seed(d_um: np.ndarray, seed: int) -> bool:
    x = np.log(d_um).reshape(-1, 1)
    gm1 = GaussianMixture(n_components=1, random_state=seed).fit(x)
    gm2 = GaussianMixture(n_components=2, random_state=seed, n_init=3).fit(x)
    xgrid = np.linspace(x.min() - 0.5, x.max() + 0.5, 2000)
    dens = np.exp(gm2.score_samples(xgrid.reshape(-1, 1)))
    n_peaks = len(find_peaks(dens)[0])
    return bool(gm2.bic(x) < gm1.bic(x) and n_peaks >= 2)


def _fit_one_sample(d_um: np.ndarray) -> dict:
    x = np.log(d_um).reshape(-1, 1)
    gm1 = GaussianMixture(n_components=1, random_state=0).fit(x)
    gm2 = GaussianMixture(n_components=2, random_state=0, n_init=3).fit(x)
    bic1, bic2 = gm1.bic(x), gm2.bic(x)
    means = gm2.means_.flatten()
    stds = np.sqrt(gm2.covariances_.flatten())
    weights = gm2.weights_.flatten()
    order = np.argsort(means)
    means, stds, weights = means[order], stds[order], weights[order]
    sep = float(abs(means[1] - means[0]) / np.sqrt((stds[0] ** 2 + stds[1] ** 2) / 2))

    xgrid = np.linspace(x.min() - 0.5, x.max() + 0.5, 2000)
    dens = np.exp(gm2.score_samples(xgrid.reshape(-1, 1)))
    n_peaks = len(find_peaks(dens)[0])
    bic_prefers_2 = bic2 < bic1
    density_bimodal = bool(bic_prefers_2 and n_peaks >= 2)

    return dict(delta_bic=float(bic1 - bic2), separation=sep,
               w_small=float(weights[0]), n_density_peaks=int(n_peaks),
               density_bimodal=density_bimodal)


def main():
    shape_df = pd.read_csv(SHAPE_CSV)
    master_df = pd.read_csv(MASTER_TABLE)[["sample_id", "condition_id", "precursor"]]

    rows = []
    for sid, g in shape_df.groupby("sample_id"):
        rec = {"sample_id": sid}
        rec.update(_fit_one_sample(g["d_um"].to_numpy()))
        rows.append(rec)
    out = pd.DataFrame(rows).merge(master_df, on="sample_id", how="left")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"[done] {OUT_CSV} ({len(out)} rows)")

    sep_median = out.groupby("precursor")["separation"].median()
    bimodal_count = out.groupby("precursor")["density_bimodal"].sum()
    print("\n=== 分离度中位数(按前驱体) ===")
    print(sep_median)

    piv = out.pivot(index="condition_id", columns="precursor", values="separation")
    diff = piv["M"] - piv["L"]
    stat, p_paired = wilcoxon(diff)
    print(f"\n=== 27条件配对分离度差(M-L) ===")
    print(f"  median_diff={diff.median():+.4f}  wilcoxon p={p_paired:.4g}")

    print("\n=== density_bimodal 计数(按前驱体,seed=0) ===")
    print(bimodal_count)

    print("\n=== 跨 5 个 random_state 的稳健性核查(M 是否稳定高于/低于 L) ===")
    seed_counts = []
    for seed in ROBUSTNESS_SEEDS:
        rows_s = []
        for sid, g in shape_df.groupby("sample_id"):
            rows_s.append(dict(sample_id=sid, bimodal=_is_bimodal_for_seed(g["d_um"].to_numpy(), seed)))
        out_s = pd.DataFrame(rows_s).merge(master_df, on="sample_id")
        counts_s = out_s.groupby("precursor")["bimodal"].sum()
        seed_counts.append(dict(seed=seed, S=int(counts_s.get("S", 0)),
                                M=int(counts_s.get("M", 0)), L=int(counts_s.get("L", 0))))
        print(f"  seed={seed}: S={seed_counts[-1]['S']} M={seed_counts[-1]['M']} L={seed_counts[-1]['L']}")
    seed_counts_df = pd.DataFrame(seed_counts)

    _append_report(sep_median, diff, p_paired, bimodal_count, seed_counts_df)
    print(f"\n[done,追加] {REPORT_MD}")


def _append_report(sep_median, diff, p_paired, bimodal_count, seed_counts_df):
    lines = ["\n---\n\n## 双峰分解诊断(间接结果,非主证据)\n\n"]
    lines.append("> 由 `scripts/46_sem_psd_bimodal_diagnostic.py` 追加。"
                 "本节结论**不**作为级配传递的主证据(主证据见上方 "
                 "scripts/45_sem_psd_gradation_pairwise.py 的"
                 "27条件配对检验),只作为间接/算法敏感的机制性观察报告。\n\n")
    lines.append(f"**连续量(分离度)—— 主判断**:log 尺度两组分均值差/合并"
                 f"标准差,前驱体中位数 L={sep_median['L']:.3f}、"
                 f"M={sep_median['M']:.3f}。27 条件配对 M−L 差中位数 "
                 f"{diff.median():+.4f},Wilcoxon p={p_paired:.4g},"
                 f"远不显著,结论:**M 与 L 在分离度上无法区分**。\n\n")
    lines.append(f"**离散量("
                 f"\"density_bimodal\":BIC 支持 2 组分 且 混合密度确有两个"
                 f"局部极大)—— 次要、需谨慎解读**:seed=0 时前驱体计数为 "
                 f"{'、'.join(f'{p}={int(bimodal_count.get(p,0))}/27' for p in ['S','M','L'])}。"
                 f"根因修复后重新生成的文件上,此计数首次出现 **M 略高于 "
                 f"L**——跨 5 个 random_state 复核如下:\n\n")
    lines.append("| seed | S | M | L |\n|---|---|---|---|\n")
    for _, r in seed_counts_df.iterrows():
        lines.append(f"| {int(r['seed'])} | {int(r['S'])} | {int(r['M'])} | {int(r['L'])} |\n")
    lines.append(f"\nM 稳定在 9-10、L 稳定在 8,不是单次拟合的偶然噪声,但差值"
                 f"只有 27 个样本中的 1-2 个——这是连续分离度在\"是否达到两个"
                 f"局部极大\"这一判据边界上的离散化,天生对临界样本敏感,不"
                 f"构成\"M 比 L 更双峰\"的显著证据。**结论口径(不得改为"
                 f"阳性)**:SEM 数量口径级配传递的证据以"
                 f"scripts/45_sem_psd_gradation_pairwise.py 的 27 条件配对"
                 f"分布检验、以及本节连续分离度的 Wilcoxon 检验为准——两者"
                 f"均显示 M 与 L 之间未发现可辨识差异;上面这个离散计数的"
                 f"微小、跨种子稳定的偏移如实记录在案,但不构成级配保留的"
                 f"独立证据,也不改变\"M 与 L 无法区分\"这一判断。\n\n")
    with open(REPORT_MD, "a", encoding="utf-8") as f:
        f.write("".join(lines))


if __name__ == "__main__":
    main()
