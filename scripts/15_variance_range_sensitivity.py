# -*- coding: utf-8 -*-
"""15 设计范围敏感性诊断(T5)——主效应/交互项方差份额是否被设计范围本身决定。
==========================================================================
审稿人质疑:前驱体跨 3.4 倍、温度只跨 100K,方差份额(η²)是设计范围决定的,
不是效应本身的属性。论文 §4.3 用完整二阶交互 ANOVA(见
`scripts/12_anova_interactions.py`)论证
"前驱体×工艺交互总量远小于主效应份额,两个尺度可独立优化"。

本脚本对 compaction_density/D_sec/circularity/convexity 四个目标,依次:
  §2.1 剔除一个温度/beta/t_hold 水平(27条件81样本 → 18条件54样本),共9个子集
  §2.2 剔除一个前驱体(81样本 → 54样本,只剩两种前驱体),共3个子集
  §2.3 对比 §2.1/§2.2 全部12个子集中,主效应份额 vs 前驱体×工艺耦合总量
       (precursor:T_C+precursor:beta+precursor:t_hold 的 η² 之和)的跨子集振幅

2026-08-26 追加:新增第 5 个目标 `solidity`——
`master_table.csv` 本身没有这一列,`main()` 里从
`data/interim/sem_shape_descriptors_summary.csv`(scripts/17 产物)按
`sample_id` 合并进来,合并后复用与另外 4 个目标完全相同的 13 子集×
`run_anova_for_target` 流程,不改任何统计逻辑。动机:figS2 面板 (d)
原来画的是 `convexity`,但 `convexity` 已在稿件正文中因与 `solidity`
定义重复、数值实质不同而被排除,
figS2 若继续画 `convexity` 就与正文口径不一致——故让 figS2 改画
`solidity`,需要这里补出对应的设计范围敏感性数据。`convexity` 本身
仍保留在 `TARGETS` 里(该列产出继续可用于其他用途),只是 figS2 不再读它。

复用 `nfm.stats.anova_interactions.run_anova_for_target`(不重写 SS 计算逻辑)。
每个子集在跑 ANOVA 前都实测验证:(a) 仍是完整平衡析因设计(每格 n=1),
(b) Sum 编码设计矩阵任意两个 term 的列块内积是否仍恒为 0(正交性,不假设)。

产出:
  data/interim/variance_range_sensitivity.csv   长表(subset×target×term)
  reports/figures/fig_variance_range_sensitivity.{pdf,png}
  reports/variance_range_sensitivity_report.md
"""
import _bootstrap  # noqa

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from nfm.stats.anova_interactions import (
    FACTORS, TERM_ORDER, PRECURSOR_PROCESS_TERMS, build_design, run_anova_for_target,
)

TARGETS = ("compaction_density", "D_sec", "circularity", "convexity", "solidity")
MAIN_TERMS = ("precursor", "T_C", "beta", "t_hold")
N_PERM = 5000
MASTER_TABLE = "data/processed/master_table.csv"
SHAPE_SUMMARY = "data/interim/sem_shape_descriptors_summary.csv"

OUT_CSV = Path("data/interim/variance_range_sensitivity.csv")
OUT_FIG_PDF = Path("reports/figures/fig_variance_range_sensitivity.pdf")
OUT_FIG_PNG = Path("reports/figures/fig_variance_range_sensitivity.png")
OUT_REPORT = Path("reports/variance_range_sensitivity_report.md")

# 前驱体 D50 跨度(µm),来自 master_table.csv precursor_D50(逐前驱体唯一值)
PRECURSOR_D50 = {"S": 3.67, "M": 9.70, "L": 12.40}

# ---------------------------------------------------------------------------
# 子集定义:§2.1(温度/beta/t_hold 各剔除一个水平)+ §2.2(前驱体剔除一个)
# ---------------------------------------------------------------------------
SUBSETS = [
    dict(id="baseline_81", desc="完整81样本(基线,27条件×3前驱体,3x3x3x3)",
         group="baseline", filt=lambda df: df),

    dict(id="drop_T950", desc="剔除T_C=950,只留{850,900}",
         group="T_C", filt=lambda df: df[df["T_C"] != 950]),
    dict(id="drop_T900", desc="剔除T_C=900,只留{850,950}",
         group="T_C", filt=lambda df: df[df["T_C"] != 900]),
    dict(id="drop_T850", desc="剔除T_C=850,只留{900,950}",
         group="T_C", filt=lambda df: df[df["T_C"] != 850]),

    dict(id="drop_beta8", desc="剔除beta=8,只留{2,5}",
         group="beta", filt=lambda df: df[df["beta"] != 8]),
    dict(id="drop_beta5", desc="剔除beta=5,只留{2,8}",
         group="beta", filt=lambda df: df[df["beta"] != 5]),
    dict(id="drop_beta2", desc="剔除beta=2,只留{5,8}",
         group="beta", filt=lambda df: df[df["beta"] != 2]),

    dict(id="drop_thold20", desc="剔除t_hold=20,只留{10,15}",
         group="t_hold", filt=lambda df: df[df["t_hold"] != 20]),
    dict(id="drop_thold15", desc="剔除t_hold=15,只留{10,20}",
         group="t_hold", filt=lambda df: df[df["t_hold"] != 15]),
    dict(id="drop_thold10", desc="剔除t_hold=10,只留{15,20}",
         group="t_hold", filt=lambda df: df[df["t_hold"] != 10]),

    dict(id="drop_precursor_S", desc="剔除前驱体S,只留{M,L}(9.70/12.40µm,跨度1.28倍)",
         group="precursor", filt=lambda df: df[df["precursor"] != "S"]),
    dict(id="drop_precursor_M", desc="剔除前驱体M,只留{S,L}(3.67/12.40µm,跨度3.38倍)",
         group="precursor", filt=lambda df: df[df["precursor"] != "M"]),
    dict(id="drop_precursor_L", desc="剔除前驱体L,只留{S,M}(3.67/9.70µm,跨度2.64倍)",
         group="precursor", filt=lambda df: df[df["precursor"] != "L"]),
]

SENSITIVITY_IDS = [s["id"] for s in SUBSETS if s["group"] != "baseline"]
PROCESS_IDS = [s["id"] for s in SUBSETS if s["group"] in ("T_C", "beta", "t_hold")]
PRECURSOR_IDS = [s["id"] for s in SUBSETS if s["group"] == "precursor"]


def check_design(work: pd.DataFrame) -> dict:
    """实测(不假设)子集设计是否仍是完整平衡析因 + Sum 编码是否仍正交。

    对照 tests/test_anova_interactions.py::test_build_design_is_orthogonal_across_terms
    的做法:任意两个不同 term 的设计矩阵列块内积应恒为 0。
    """
    n_rows = len(work)
    cell_counts = work.groupby(list(FACTORS), observed=True).size()
    balanced = bool((cell_counts == 1).all())
    n_cells = int(len(cell_counts))

    X, term_slices, _ = build_design(work)
    XtX = X.T @ X
    slices = list(term_slices.values())
    max_offdiag = 0.0
    for i, si in enumerate(slices):
        for j, sj in enumerate(slices):
            if i >= j:
                continue
            max_offdiag = max(max_offdiag, float(np.abs(XtX[si, sj]).max()))
    orthogonal = bool(max_offdiag < 1e-8)

    levels = {f: sorted(work[f].unique().tolist()) for f in FACTORS}

    return dict(n_rows=n_rows, n_cells=n_cells, balanced=balanced,
                orthogonal=orthogonal, max_offdiag=max_offdiag, levels=levels)


def main():
    df = pd.read_csv(MASTER_TABLE)
    shape_df = pd.read_csv(SHAPE_SUMMARY, usecols=["sample_id", "solidity"])
    df = df.merge(shape_df, on="sample_id", how="left", validate="one_to_one")
    for t in TARGETS:
        if df[t].isna().any():
            raise RuntimeError(f"{t} 在完整81样本里就有缺失,子集分析的前提不成立")

    long_rows = []
    design_rows = []
    results = {}  # (subset_id, target) -> run_anova_for_target 返回值

    for spec in SUBSETS:
        sub = spec["filt"](df).copy()
        diag = check_design(sub)
        design_rows.append(dict(subset_id=spec["id"], subset_description=spec["desc"],
                                group=spec["group"], **{k: v for k, v in diag.items() if k != "levels"},
                                levels=str(diag["levels"])))

        print(f"[{spec['id']}] n={diag['n_rows']} n_cells={diag['n_cells']} "
              f"balanced={diag['balanced']} orthogonal={diag['orthogonal']} "
              f"(max_offdiag={diag['max_offdiag']:.2e})")

        if not diag["balanced"]:
            print(f"  !! 警告:{spec['id']} 不是完整平衡析因设计(每格应恰好n=1),"
                  "跳过该子集的 ANOVA(不能假装 Type I/II/III 仍然一致)")
            continue

        for i, target in enumerate(TARGETS):
            res = run_anova_for_target(sub, target, n_perm=N_PERM,
                                       seed=hash((spec["id"], target)) % (2**31))
            results[(spec["id"], target)] = res
            table = res["table"]

            for term in list(TERM_ORDER) + ["Residual"]:
                row = table.loc[term]
                long_rows.append(dict(
                    subset_id=spec["id"], subset_description=spec["desc"],
                    group=spec["group"], target=target, term=term,
                    SS=row["SS"], df=row["df"], MS=row["MS"], F=row["F"],
                    p_parametric=row.get("p_parametric", np.nan),
                    eta2=row["eta2"], omega2=row["omega2"], perm_p=row["perm_p"],
                    n_samples=diag["n_rows"], design_orthogonal=diag["orthogonal"],
                    type_agreement=res["type_agreement"],
                    type_max_abs_diff=res["type_max_abs_diff"],
                ))

            # 合成一行:前驱体×工艺耦合总量(coupling_total),eta2/omega2 直接求和
            # (与 scripts/12_anova_interactions.py 的口径一致,即论文表2头号数字)
            coupling_eta2 = table.loc[list(PRECURSOR_PROCESS_TERMS), "eta2"].sum()
            coupling_omega2 = table.loc[list(PRECURSOR_PROCESS_TERMS), "omega2"].sum()
            coupling_df = int(table.loc[list(PRECURSOR_PROCESS_TERMS), "df"].sum())
            long_rows.append(dict(
                subset_id=spec["id"], subset_description=spec["desc"],
                group=spec["group"], target=target, term="coupling_total",
                SS=np.nan, df=coupling_df, MS=np.nan, F=np.nan, p_parametric=np.nan,
                eta2=coupling_eta2, omega2=coupling_omega2, perm_p=np.nan,
                n_samples=diag["n_rows"], design_orthogonal=diag["orthogonal"],
                type_agreement=res["type_agreement"], type_max_abs_diff=res["type_max_abs_diff"],
            ))

    long_df = pd.DataFrame(long_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(OUT_CSV, index=False)
    print(f"[done] {OUT_CSV} ({len(long_df)} 行)")

    design_df = pd.DataFrame(design_rows)

    make_figure(long_df)
    print(f"[done] {OUT_FIG_PDF}")
    print(f"[done] {OUT_FIG_PNG}")

    write_report(long_df, design_df, results)
    print(f"[done] {OUT_REPORT}")


# ---------------------------------------------------------------------------
# 图:份额 vs 设计子集(分面=target,分组条形=term)
# ---------------------------------------------------------------------------
# dataviz 骨架的固定顺序分类色(references/palette.md,前5槽),用于 5 个系列
# (precursor/T_C/beta/t_hold/coupling_total),严格按固定顺序赋色、不循环。
PALETTE = {
    "precursor": "#2a78d6",       # slot1 blue
    "T_C": "#eb6834",             # slot2 orange
    "beta": "#1baf7a",            # slot3 aqua
    "t_hold": "#eda100",          # slot4 yellow
    "coupling_total": "#e87ba4",  # slot5 magenta
}
SERIES_ORDER = ["precursor", "T_C", "beta", "t_hold", "coupling_total"]
SERIES_LABEL = {
    "precursor": "前驱体主效应", "T_C": "T_C主效应", "beta": "beta主效应",
    "t_hold": "t_hold主效应", "coupling_total": "前驱体×工艺耦合总量",
}
GROUP_LABEL = {
    "baseline": "基线(81)", "T_C": "剔除T水平", "beta": "剔除beta水平",
    "t_hold": "剔除t_hold水平", "precursor": "剔除前驱体",
}
TARGET_LABEL = {
    "compaction_density": "compaction_density", "D_sec": "D_sec",
    "circularity": "circularity", "convexity": "convexity",
    "solidity": "solidity",
}


def make_figure(long_df: pd.DataFrame):
    order = [s["id"] for s in SUBSETS]
    order_labels = [s["desc"] for s in SUBSETS]
    group_of = {s["id"]: s["group"] for s in SUBSETS}

    fig, axes = plt.subplots(len(TARGETS), 1, figsize=(18, 5 * len(TARGETS)), sharex=True)
    bar_w = 0.16
    n_series = len(SERIES_ORDER)

    for ax, target in zip(axes, TARGETS):
        sub = long_df[long_df["target"] == target]
        x0 = np.arange(len(order))
        for k, term in enumerate(SERIES_ORDER):
            vals = []
            for sid in order:
                row = sub[(sub["subset_id"] == sid) & (sub["term"] == term)]
                vals.append(float(row["eta2"].iloc[0]) * 100 if len(row) else np.nan)
            offset = (k - (n_series - 1) / 2) * bar_w
            ax.bar(x0 + offset, vals, width=bar_w * 0.92, color=PALETTE[term],
                  label=SERIES_LABEL[term] if ax is axes[0] else None)

        ax.set_ylabel("方差份额 η² (%)", fontsize=10)
        ax.set_title(TARGET_LABEL[target], fontsize=11, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        # 组分隔竖线 + 顶部分组标签
        prev_g = None
        for i, sid in enumerate(order):
            g = group_of[sid]
            if prev_g is not None and g != prev_g:
                ax.axvline(i - 0.5, color="#c3c2b7", linewidth=1.0, linestyle="--", zorder=0)
            prev_g = g

    axes[-1].set_xticks(np.arange(len(order)))
    axes[-1].set_xticklabels(order_labels, rotation=32, ha="right", fontsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
              bbox_to_anchor=(0.5, 1.0), fontsize=10)
    fig.suptitle("主效应份额 vs 前驱体×工艺耦合总量:随设计子集(缩范围)的变化", fontsize=13, y=1.03)
    fig.tight_layout()

    OUT_FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG_PDF, bbox_inches="tight")
    fig.savefig(OUT_FIG_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def amplitude_table(long_df: pd.DataFrame, subset_ids: list, terms: list) -> pd.DataFrame:
    """对给定子集集合、给定 term 集合,逐 target 算 eta2%(=eta2*100) 的 min/max/range,
    以及取到 min/max 的子集 id。"""
    rows = []
    for target in TARGETS:
        for term in terms:
            sub = long_df[(long_df["target"] == target) & (long_df["term"] == term)
                          & (long_df["subset_id"].isin(subset_ids))]
            if sub.empty:
                rows.append(dict(target=target, term=term, min_pct=np.nan, max_pct=np.nan,
                                 range_pct=np.nan, min_subset=None, max_subset=None))
                continue
            pct = sub["eta2"] * 100
            i_min = pct.idxmin()
            i_max = pct.idxmax()
            rows.append(dict(target=target, term=term,
                             min_pct=float(pct.loc[i_min]), max_pct=float(pct.loc[i_max]),
                             range_pct=float(pct.loc[i_max] - pct.loc[i_min]),
                             min_subset=sub.loc[i_min, "subset_id"],
                             max_subset=sub.loc[i_max, "subset_id"]))
    return pd.DataFrame(rows)


def write_report(long_df: pd.DataFrame, design_df: pd.DataFrame, results: dict):
    lines = []
    lines.append("# 设计范围敏感性诊断报告(T5)")
    lines.append("")
    lines.append(f"生成脚本:`scripts/15_variance_range_sensitivity.py`;"
                 f"数据源:`{MASTER_TABLE}`;复用 `nfm.stats.anova_interactions.run_anova_for_target`。")
    lines.append("")
    lines.append("## 0. 任务背景")
    lines.append("")
    lines.append("审稿人质疑:前驱体设计跨 3.4 倍(3.67–12.40µm),温度只跨 100K(850–950°C),"
                 "`scripts/12_anova_interactions.py`(完整二阶交互 ANOVA,81 样本)报出的"
                 "\"precursor 主效应 η² 最高达 88.8%、precursor×工艺交互总量 ≤3.33%\" 可能只是"
                 "设计范围恰好选得合适的产物,而非效应本身的稳健属性。本报告依次剔除一个"
                 "温度/beta/t_hold 水平(§1)、依次剔除一个前驱体(§2),重算份额,"
                 "检验份额随设计范围收窄的变化幅度,并检验\"交互项份额对范围的依赖性"
                 "显著低于主效应份额\"这一论断本身是否成立(§3)。")
    lines.append("")

    # ---- §1 设计平衡性/正交性验证(硬约束:必须实测,不能假设) ----
    lines.append("## 1. 子集设计的平衡性与正交性验证(实测,非假设)")
    lines.append("")
    lines.append("每个子集在跑 ANOVA 前都用与 "
                 "`tests/test_anova_interactions.py::test_build_design_is_orthogonal_across_terms` "
                 "相同的方法实测:(a) 是否仍是完整平衡析因设计(每个保留下来的 "
                 "(precursor,T_C,beta,t_hold) 组合恰好 n=1);(b) Sum 编码设计矩阵任意两个不同 "
                 "term 的列块内积是否仍恒为 0(数值容差 1e-8)。")
    lines.append("")
    lines.append("| subset_id | 说明 | n_rows | n_cells | balanced | orthogonal | max_offdiag |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in design_df.iterrows():
        lines.append(f"| {r['subset_id']} | {r['subset_description']} | {r['n_rows']} | "
                     f"{r['n_cells']} | {r['balanced']} | {r['orthogonal']} | {r['max_offdiag']:.2e} |")
    lines.append("")
    all_balanced = bool(design_df["balanced"].all())
    all_ortho = bool(design_df["orthogonal"].all())
    if all_balanced and all_ortho:
        lines.append("**结论:全部 13 个子集(1 基线 + 9 工艺范围 + 3 前驱体范围)实测均仍是"
                     "完整平衡析因设计,且 Sum 编码设计矩阵在全部子集上都保持严格正交"
                     "(max_offdiag 全部 < 1e-8)**。这是因为原始 81 样本设计本身是完整"
                     "3⁴ 析因、每格恰好 n=1;剔除某个因子的一个水平后,保留部分仍是"
                     "对其余水平的完整析因、每格仍恰好 n=1,平衡性和正交性在结构上继续成立"
                     "(不是巧合,但仍按任务要求逐个实测确认,不凭理论想当然)。"
                     "故子集设计上 Type I/II/III SS 应继续一致,下方 §1b 逐一核实。")
    else:
        bad = design_df[~(design_df["balanced"] & design_df["orthogonal"])]
        lines.append(f"**警告:以下子集未通过平衡性/正交性检验,其 ANOVA 结果(若有)"
                     f"不应假设 Type I/II/III 一致**:\n\n{bad[['subset_id','balanced','orthogonal','max_offdiag']].to_string(index=False)}")
    lines.append("")

    lines.append("### 1b. Type I vs Type II/III SS 一致性(逐子集×逐target 实测,非假设)")
    lines.append("")
    bad_agreement = []
    for (sid, target), res in results.items():
        if not res["type_agreement"]:
            bad_agreement.append((sid, target, res["type_max_abs_diff"]))
    if not bad_agreement:
        lines.append(f"全部 {len(results)} 个 (子集×target) 组合 Type I/II/III SS 一致"
                     "(`run_anova_for_target` 内部用 statsmodels typ=1/2/3 交叉验证)。")
    else:
        lines.append("**以下组合 Type I/II/III SS 不一致,其 η²/ω² 结果需谨慎解读**:")
        for sid, target, d in bad_agreement:
            lines.append(f"- {sid} / {target}: max_abs_diff={d:.3e}")
    lines.append("")

    # ---- §2.1 process range ----
    lines.append("## 2. 工艺范围敏感性(§2.1:温度/beta/t_hold 各剔除一个水平)")
    lines.append("")
    lines.append("27 条件(81 样本)→ 18 条件(54 样本),9 个子集(温度3+beta3+t_hold3),"
                 "各子集 n_rows 实测均为 54(见上表)。")
    lines.append("")
    amp_process = amplitude_table(long_df, PROCESS_IDS, list(MAIN_TERMS) + ["coupling_total"])
    lines.append("| target | term | min η²% | max η²% | 振幅(pct pt) | min子集 | max子集 |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in amp_process.iterrows():
        lines.append(f"| {r['target']} | {r['term']} | {r['min_pct']:.2f} | {r['max_pct']:.2f} | "
                     f"{r['range_pct']:.2f} | {r['min_subset']} | {r['max_subset']} |")
    lines.append("")

    # ---- §2.2 precursor range ----
    lines.append("## 3. 前驱体范围敏感性(§2.2:S/M/L 各剔除一个)")
    lines.append("")
    lines.append("81 样本 → 54 样本(剩两种前驱体)。剔除后 `precursor` 从 3 水平变 2 水平,"
                 "`precursor:T_C`/`precursor:beta`/`precursor:t_hold` 交互项 df 也相应从 4 变 2,"
                 "下表按各子集实测 df 报告(不套用 27 条件全量设计的 df=8/12/12/48)。")
    lines.append("")
    for _, r in design_df[design_df["group"] == "precursor"].iterrows():
        sid = r["subset_id"]
        sub = long_df[(long_df["subset_id"] == sid) & (long_df["target"] == TARGETS[0])]
        df_map = sub.set_index("term")["df"].to_dict()
        lines.append(f"- **{sid}**({r['subset_description']}):"
                     f"precursor df={int(df_map.get('precursor', -1))}, "
                     f"precursor:T_C df={int(df_map.get('precursor:T_C', -1))}, "
                     f"precursor:beta df={int(df_map.get('precursor:beta', -1))}, "
                     f"precursor:t_hold df={int(df_map.get('precursor:t_hold', -1))}, "
                     f"Residual df={int(df_map.get('Residual', -1))}")
    lines.append("")

    amp_precursor = amplitude_table(long_df, PRECURSOR_IDS, list(MAIN_TERMS) + ["coupling_total"])
    lines.append("| target | term | min η²% | max η²% | 振幅(pct pt) | min子集 | max子集 |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in amp_precursor.iterrows():
        lines.append(f"| {r['target']} | {r['term']} | {r['min_pct']:.2f} | {r['max_pct']:.2f} | "
                     f"{r['range_pct']:.2f} | {r['min_subset']} | {r['max_subset']} |")
    lines.append("")

    lines.append("### 3a. 剔除S后前驱体份额的变化(直接回应审稿人§4.3质疑)")
    lines.append("")
    lines.append("前驱体 D50 跨度:S=3.67µm / M=9.70µm / L=12.40µm。剔除 S 后只剩 M/L,"
                 "跨度从 12.40/3.67≈3.38 倍收窄到 12.40/9.70≈1.28 倍。")
    lines.append("")
    lines.append("| target | 基线(81样本)precursor η²% | 剔除S后(M/L,跨度1.28倍)precursor η²% | 下降(pct pt) | 下降(相对%) |")
    lines.append("|---|---|---|---|---|")
    for target in TARGETS:
        base = long_df[(long_df["subset_id"] == "baseline_81") & (long_df["target"] == target)
                       & (long_df["term"] == "precursor")]["eta2"].iloc[0] * 100
        dropS = long_df[(long_df["subset_id"] == "drop_precursor_S") & (long_df["target"] == target)
                        & (long_df["term"] == "precursor")]["eta2"].iloc[0] * 100
        lines.append(f"| {target} | {base:.2f} | {dropS:.2f} | {base - dropS:.2f} | "
                     f"{(base - dropS) / base * 100:.1f}% |")
    lines.append("")

    # ---- §2.3 core comparison ----
    lines.append("## 4. 交互项份额的稳健性(§2.3,核心论证)")
    lines.append("")
    lines.append("验证论文预期:\"交互项(前驱体×工艺耦合总量)份额对范围的依赖性"
                 "显著低于主效应份额\"。对 §2/§3 全部 12 个敏感性子集(9 工艺范围 + 3 前驱体范围)"
                 "合并统计每个 target 上 4 个主效应各自的跨子集振幅、以及 coupling_total 的振幅。")
    lines.append("")
    amp_all = amplitude_table(long_df, SENSITIVITY_IDS, list(MAIN_TERMS) + ["coupling_total"])
    lines.append("| target | term | min η²% | max η²% | 振幅(pct pt) |")
    lines.append("|---|---|---|---|---|")
    for _, r in amp_all.iterrows():
        lines.append(f"| {r['target']} | {r['term']} | {r['min_pct']:.2f} | {r['max_pct']:.2f} | "
                     f"{r['range_pct']:.2f} |")
    lines.append("")

    summary_rows = []
    for target in TARGETS:
        sub = amp_all[amp_all["target"] == target]
        main_ranges = sub[sub["term"].isin(MAIN_TERMS)]["range_pct"]
        coupling_range = sub[sub["term"] == "coupling_total"]["range_pct"].iloc[0]
        max_main_term = sub[sub["term"].isin(MAIN_TERMS)].loc[main_ranges.idxmax(), "term"]
        summary_rows.append(dict(target=target, max_main_range=float(main_ranges.max()),
                                 max_main_term=max_main_term, coupling_range=float(coupling_range),
                                 ratio=(float(main_ranges.max()) / coupling_range
                                       if coupling_range > 1e-9 else np.inf)))
    summary_df = pd.DataFrame(summary_rows)

    lines.append("### 4a. 主效应最大振幅 vs 耦合总量振幅(逐target)")
    lines.append("")
    lines.append("| target | 振幅最大的主效应项 | 该主效应振幅(pct pt) | coupling_total振幅(pct pt) | 振幅比(主效应/耦合) |")
    lines.append("|---|---|---|---|---|")
    for _, r in summary_df.iterrows():
        ratio_str = f"{r['ratio']:.1f}x" if np.isfinite(r["ratio"]) else "inf(耦合振幅≈0)"
        lines.append(f"| {r['target']} | {r['max_main_term']} | {r['max_main_range']:.2f} | "
                     f"{r['coupling_range']:.2f} | {ratio_str} |")
    lines.append("")

    overall_main_min = summary_df["max_main_range"].min()
    overall_main_max = summary_df["max_main_range"].max()
    overall_coupling_min = summary_df["coupling_range"].min()
    overall_coupling_max = summary_df["coupling_range"].max()
    n_support = int((summary_df["ratio"] > 1).sum())

    lines.append("### 4b. 结论")
    lines.append("")
    lines.append(f"{len(TARGETS)} 个 target 上,主效应(取跨子集振幅最大的那个主效应项,通常是 precursor)的振幅"
                 f"范围为 **{overall_main_min:.2f}–{overall_main_max:.2f} 个百分点**;"
                 f"前驱体×工艺耦合总量(coupling_total)的振幅范围为 "
                 f"**{overall_coupling_min:.2f}–{overall_coupling_max:.2f} 个百分点**。"
                 f"{n_support}/{len(TARGETS)} 个 target 上耦合总量振幅小于主效应振幅(振幅比>1)。")
    lines.append("")
    if n_support == len(TARGETS):
        min_ratio = summary_df["ratio"].min()
        lines.append(f"**结论成立**:全部 {len(TARGETS)} 个 target 上,主效应份额随设计范围收窄的振幅都"
                     f"明显大于交互项耦合总量的振幅,振幅比最小也有 **{min_ratio:.1f} 倍**"
                     "(即便在振幅比最不利的 target 上,主效应对范围的敏感度仍数倍于交互项)。"
                     "论文 §4.3\"交互项份额对范围的依赖性显著低于主效应份额\"这一论断"
                     "**在本次12子集敏感性扫描下成立,给出量化支持**。")
    else:
        bad_targets = summary_df[summary_df["ratio"] <= 1]["target"].tolist()
        lines.append(f"**结论不完全成立**:以下 target 上耦合总量的振幅并不小于主效应振幅"
                     f"(振幅比≤1):{bad_targets}。如实报告——不能因为这与论文 §4.3 预期"
                     "不符就调整子集划分或统计口径去凑成立。论文在引用\"交互项份额对范围的"
                     "依赖性更弱\"这一论断时,应改为逐 target 限定表述,或补充说明"
                     f"该论断仅在 {[t for t in TARGETS if t not in bad_targets]} 上成立。")
    lines.append("")

    # ---- Limitations ----
    lines.append("## 5. 限制声明")
    lines.append("")
    lines.append("- 本报告的\"振幅\"是跨敏感性子集(9 工艺范围 + 3 前驱体范围,共12个,"
                 "不含基线)的 max−min,不是置信区间;每个子集仍是 n=1/格的单次实验,"
                 "残差里仍混有三阶及以上交互和测量误差(与 `interaction_report.md` §7 一致的限制)。")
    lines.append("- 子集之间不是独立观测(共享同一批 81 样本的子集),振幅数字应理解为"
                 "\"当前这批数据在给定缩范围方式下的敏感性\",不是对总体范围依赖性的"
                 "无偏统计推断。")
    lines.append("- 每次剔除一个水平后自由度显著减少(Residual df 从 48 降到约 30 或更低,"
                 "具体见 §1 表和长表 CSV 的 df 列),置换检验的 perm_p 在小样本下功效更低,"
                 "本报告以 η²/ω² 点估计的振幅为主要证据,p 值仅作参考。")
    lines.append("- 未重新评判 T1/T2 已定稿结论、WH 定性参照、5000×口径、Q 不可识别、"
                 "纳米层单仪器口径等既有定论。")
    lines.append("")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
