# -*- coding: utf-8 -*-
"""fig1:方差分解堆叠柱状图 —— 10 个 target(2 纳米层 + 4 Part B 形貌/堆积
+ 4 T7 形状描述子)。

⚠️⚠️⚠️ 重要口径警告(2026-08-26 追加,不得删除)—— 本图纳米层数值
已不代表稿件正文表 1,不得用于稿件 ⚠️⚠️⚠️
================================================================
本脚本的纳米层(D_XRD/lattice_c)数值来自 `data/interim/nano_layer_anova.csv`
(`scripts/18_nano_layer_anova.py`,**主效应模型**:
`y ~ precursor + T_C + beta + t_hold`,不含交互项)。

`scripts/22_build_manuscript_numbers.py` 产出的 `table1.D_XRD.*`/
`table1.lattice_c.*`(稿件正文表 1)以及期刊投稿图
`scripts/figs/journal/fig2_variance_decomposition.py` 面板 (a),
**已改用另一个模型**——
`data/interim/nano_layer_interaction_anova.csv`
(`scripts/18_nano_layer_anova.py` 的后续脚本
`scripts/19_nano_layer_interaction_anova.py`,前驱体×工艺交互模型)。
两个模型的 η² 数值材料性不同(如 D_XRD 温度主效应:主效应模型 61.0% vs
交互模型 54.2%)。

**本脚本刻意维持主效应模型、不切换到交互模型**——本图的"与论文旧表1对照"
段落(见下方 caption)是针对
`scripts/18` 首次可复现产出时、对照一份更早期且当时已无法溯源复现的
旧稿件表格所做的**一次性历史校验**,校验对象就是 `scripts/18` 本身的
数值,不是"当前稿件表1"这个动态目标;切换模型会让这段历史校验记录失去
对应关系,而不会产出一份等价的新校验(那份"更早期旧表"的原始数字未必
仍可追溯)。因此维持现状,但**明确标注**:本图任何纳米层数值都
**不得直接引用进稿件**,稿件权威口径以 `fig2_variance_decomposition.py`
/ 表 1 为准。

底层数据(全部已由既往任务算好,本脚本只画图,不重新做统计判断):
  - data/interim/nano_layer_anova.csv                (本轮 §1.1 新算,ss_type=='II' 行 —— ⚠️主效应模型,已非表1口径,见上方警告)
  - data/interim/anova_interactions.csv               (Part B,12_anova_interactions.py)
  - data/interim/anova_interactions_shape_descriptors.csv (T7,17_shape_descriptor_full_analysis.py)

**关键口径差异,必须在图上可视化区分,不能被平滑掉**:
纳米层(D_XRD/lattice_c)只跑了主效应模型(见 18_nano_layer_anova.py 的
设计理由:51 样本在 beta/t_hold 上不平衡,加交互项会进一步降低每个 cell
自由度),没有单独估计交互项;而另外 8 个 target 用的是含全部 6 个两两
交互项的完整模型。这意味着:
  - 纳米层两个 target 的"残差"段,实际上是"二阶+交互 + 三阶+交互 + 测量
    误差"的混合,不可与另外 8 个 target"残差 = 三阶+交互 + 测量误差"
    (交互项已单独拆出)的残差直接比较绝对大小。
  - 为避免图面上把二者画成同一件事,纳米层的残差段用斜线填充
    (hatch='//')且图例单独命名为"残差(含未拆分交互,主效应模型)",
    其余 8 个 target 的"交互(6项合计)"与"残差(纯净)"分两段画,图例
    分别命名。

主报值用 η²(SS_term/SS_total,与论文表1口径一致,天然加总=100%,适合
堆叠图);ω²(去偏,更保守)在底层 csv 中可查,不在图上重复展示以免堆叠
和不为100%造成误读。
"""
import _bootstrap  # noqa

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _style import save_fig, skip

NAME = "fig1_variance_stack"

NANO_ANOVA_CSV = Path("data/interim/nano_layer_anova.csv")
PARTB_CSV = Path("data/interim/anova_interactions.csv")
SHAPE_CSV = Path("data/interim/anova_interactions_shape_descriptors.csv")

MAIN_TERMS = ["precursor", "T_C", "beta", "t_hold"]
INTERACTION_TERMS = ["precursor:T_C", "precursor:beta", "precursor:t_hold",
                     "T_C:beta", "T_C:t_hold", "beta:t_hold"]

# Okabe-Ito 色盲友好定性色板(与 PRECURSOR_COLOR 无关——这里编码的是 ANOVA
# 项身份,不是前驱体身份,复用 PRECURSOR_COLOR 在语义上是错的)。
TERM_COLOR = {
    "precursor": "#E69F00",
    "T_C": "#0072B2",
    "beta": "#009E73",
    "t_hold": "#CC79A7",
    "interactions": "#999999",
    "residual_clean": "#D3D3D3",
    "residual_mixed": "#56B4E9",
}

# 论文/报告章节归属分组,决定柱的排列顺序与分面标签
# convexity 已从此图移除——convexity(500x主线)
# 与 solidity(T7 形状描述子流水线)定义完全相同(A/A_convex)但两条流水线
# 数值有实质差异,Table 1/2 已只保留 solidity,本图与之保持一致,不再
# 重复画出定义相同但口径不同的 convexity 柱。
GROUPS = [
    ("纳米层【非表1口径】\n(仅仪器1·51样·主效应模型)", ["D_XRD", "lattice_c"]),
    ("形貌/堆积层\n(Part B·81样·含交互)", ["compaction_density", "D_sec", "circularity"]),
    ("形状描述子\n(T7·81样·含交互)", ["D_f", "solidity", "elongation", "roughness"]),
]


def _load_partb_style(path: Path) -> dict:
    df = pd.read_csv(path)
    out = {}
    for target, g in df.groupby("target"):
        g = g.set_index("term")
        row = {m: g.loc[m, "eta2"] for m in MAIN_TERMS}
        row["interactions"] = g.loc[INTERACTION_TERMS, "eta2"].sum()
        row["residual_clean"] = g.loc["Residual", "eta2"]
        row["residual_mixed"] = np.nan
        out[target] = row
    return out


def _load_nano() -> dict:
    df = pd.read_csv(NANO_ANOVA_CSV)
    df = df[df["ss_type"] == "II"]  # 见 18_nano_layer_anova.py:Type II/III 一致,是权威值
    out = {}
    for target, g in df.groupby("target"):
        g = g.set_index("term")
        row = {m: g.loc[m, "eta2"] for m in MAIN_TERMS}
        row["interactions"] = np.nan  # 未拟合交互项,不是 0,不得画成 0
        row["residual_mixed"] = g.loc["Residual", "eta2"]
        row["residual_clean"] = np.nan
        out[target] = row
    return out


def load_all() -> pd.DataFrame:
    data = {}
    data.update(_load_nano())
    data.update(_load_partb_style(PARTB_CSV))
    data.update(_load_partb_style(SHAPE_CSV))
    order = [t for _, ts in GROUPS for t in ts]
    return pd.DataFrame([data[t] for t in order], index=order)


def plot(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    x = np.arange(len(df))
    bottom = np.zeros(len(df))

    segment_defs = [
        ("precursor", "前驱体", TERM_COLOR["precursor"], None),
        ("T_C", "T", TERM_COLOR["T_C"], None),
        ("beta", "β", TERM_COLOR["beta"], None),
        ("t_hold", "t", TERM_COLOR["t_hold"], None),
        ("interactions", "交互(6项合计)", TERM_COLOR["interactions"], None),
        ("residual_clean", "残差(纯净:3阶+交互+测量误差)", TERM_COLOR["residual_clean"], None),
        ("residual_mixed", "残差(含未拆分交互,主效应模型)", TERM_COLOR["residual_mixed"], "//"),
    ]
    for col, label, color, hatch in segment_defs:
        vals = df[col].fillna(0.0).to_numpy() * 100
        ax.bar(x, vals, bottom=bottom, color=color, label=label,
              edgecolor="black", linewidth=0.4, hatch=hatch, width=0.62)
        bottom += vals

    # 组分隔竖线 + 分组标签
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, fontsize=9, rotation=25, ha="right")
    cursor = 0
    for gname, targets in GROUPS:
        n = len(targets)
        if cursor > 0:
            ax.axvline(cursor - 0.5, color="black", linewidth=0.8, linestyle=":")
        ax.text(cursor + n / 2 - 0.5, 103, gname, ha="center", va="bottom",
               fontsize=8.5, fontweight="bold")
        cursor += n

    ax.set_ylim(0, 112)
    ax.set_ylabel("方差份额 η² (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8)

    # 2026-08-26 追加:图上直接标注口径警告,不能只写在 docstring/caption 里
    # ——本图是内部评审图,常被单独截屏/转发,脱离脚本文件本身查看时也必须
    # 能看到这条警告。用 fig.text()(图级坐标,不是 ax.transAxes)+显式收紧
    # 顶部子图边距,确保 save_fig()(不带 bbox_inches="tight")也不会裁掉它
    # ——ax.transAxes 的越界坐标在没有 tight bbox 时会被直接裁剪,不能用。
    fig.subplots_adjust(top=0.80)
    fig.text(0.5, 0.965,
             "警告:纳米层(D_XRD/lattice_c)用主效应模型(scripts/18),已非稿件表1口径\n"
             "(表1/期刊图2 改用 scripts/19 交互模型)"
             "——不得引用进稿件",
             fontsize=7.5, color="#B00000", fontweight="bold",
             va="top", ha="center",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF3F3",
                       edgecolor="#B00000", linewidth=1.2))

    caption = (
        "**数据**:纳米层(D_XRD/lattice_c)来自 `data/interim/nano_layer_anova.csv`"
        "(`scripts/18_nano_layer_anova.py`,51 样=仅 xrd_instrument==1 且 "
        "D_XRD_reliable==True,`ss_type=='II'` 行 —— 该设计不平衡,Type II/III 数值"
        "一致且与顺序无关,是本图权威值,Type I 因项顺序而异,详见脚本输出);"
        "形貌/堆积层(compaction_density/D_sec/circularity;convexity 因与"
        "T7 的 solidity 定义相同——A/A_convex——但两条流水线数值有实质差异,"
        "已从本图与论文表1/2 移除)来自 "
        "`data/interim/anova_interactions.csv`(Part B,81 样完整平衡析因,"
        "`scripts/12_anova_interactions.py`);形状描述子(D_f/solidity/elongation/"
        "roughness)来自 `data/interim/anova_interactions_shape_descriptors.csv`"
        "(T7,`scripts/17_shape_descriptor_full_analysis.py`)。纵轴为 η² "
        "(SS_term/SS_total,与论文表1口径一致,各柱恰好加总100%)。\n\n"
        "**口径差异(务必注意,不是同一件事)**:纳米层两个 target 只拟合了主效应"
        "模型(51 样在 beta/t_hold 上不平衡,加交互项会进一步降低每个 cell 自由度,"
        "本轮不做),其\"残差\"(斜线填充,浅蓝)实际混杂了全部未拆分的二阶及以上"
        "交互项;另外 8 个 target 用完整二阶交互模型,交互项(灰色)已单独拆出、"
        "残差(浅灰)只含三阶及以上交互+测量误差。两类残差**不可直接比较绝对大小**,"
        "图上刻意用不同填充与图例名称区分,不做同口径拼接。\n\n"
        "**⚠️ 追加提醒(不得忽略)**:上面这段\"仅主效应模型\"的口径说明"
        "本身已经过时于稿件正文——稿件表 1"
        "与期刊图 `fig2_variance_decomposition.py` 面板 (a) 的纳米层数据源"
        "改为 `data/interim/nano_layer_interaction_anova.csv`"
        "(`scripts/19_nano_layer_interaction_anova.py`,前驱体×工艺交互模型),"
        "不再是本图仍在使用的 `scripts/18` 主效应模型。**本图刻意维持"
        "旧模型不切换**,理由是下面这段\"与论文旧表1对照\"是针对 `scripts/18`"
        "首次可复现产出时的一次性历史校验,校验对象是 `scripts/18` 自己的"
        "数值,不是一个会跟着稿件表 1 变化的动态目标。"
        "**因此:本图纳米层数值(D_XRD/lattice_c)不得直接引用进稿件,稿件"
        "权威口径以 `fig2_variance_decomposition.py` / 表 1 为准**;本图"
        "另外 8 个 target(Part B + T7)不受影响,口径未变。\n\n"
        "**与论文旧表1的对照**(仅主效应,新值 vs 论文值,pp=百分点差;此"
        "对照是 `scripts/18` 的历史校验记录,与当前稿件表1无关,见上方警告):D_XRD "
        "precursor +0.1/T −5.3/β +0.9/t −9.3;lattice_c precursor +0.1/T −6.6/"
        "β −0.5/t −6.7(其余 3+4 个 target 的对照结果类似,此处从略)。差异原因、"
        "Type I/II/III 一致性判定详见脚本输出。"
    )
    save_fig(fig, NAME, caption)
    plt.close(fig)


def main():
    if not (NANO_ANOVA_CSV.exists() and PARTB_CSV.exists() and SHAPE_CSV.exists()):
        skip(NAME, "底层 ANOVA 结果 csv 缺失,需先跑 18_nano_layer_anova.py / "
                   "12_anova_interactions.py / 17_shape_descriptor_full_analysis.py")
        return
    df = load_all()
    plot(df)


if __name__ == "__main__":
    main()
