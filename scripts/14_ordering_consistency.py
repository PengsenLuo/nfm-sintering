# -*- coding: utf-8 -*-
"""14 T4 —— S/M/L 排序模式稳健性诊断（一次性分析，不产出可复用模块）。
==========================================================================
背景：论文拟直接引用一句话——"S < M ≲ L 的密度排序在 27 个条件中成立 __/27 次"。
本脚本对 27 个条件（每条件 S/M/L 各一个产物样本）逐条件核算 compaction_density
的实际排序模式，配套符号检验（sign test）判断是否显著偏离随机排序；并对
D_sec/circularity/convexity 三个形貌描述子做同样的排序统计，交叉引用 T1
（`scripts/13_sieving_confound.py` / `reports/sieving_confound_report.md`）
已确认的过筛结构性混杂条件（C01/C04/C07 与 T_C=850°C & t_hold=10h 联合格
100% 重合，Fisher exact p=0.0034）。

数据源：`data/processed/master_table.csv`（81 行 = 27 条件 × S/M/L，
compaction_density/D_sec/circularity/convexity 均 81/81 完整，已核实）。

方向依据（不凭空假设，逐一说明来源）：
  - compaction_density：任务书直接给出预期方向 S<M≲L。
  - D_sec：已定稿的形貌自适应分割给出 D_sec 的 L/S 比值=2.29，方向与
    compaction_density 一致（S 最小、L 最大，前驱体粒径记忆的直接证据）——
    故 D_sec 采用与 compaction_density 相同的"increasing"（S<M≲L）预期方向。
  - circularity/convexity：仓库内没有既有的显式方向陈述，本脚本基于以下
    物理逻辑推断预期方向与 D_sec 相反（"decreasing"，即 S>M≳L）：circularity/
    convexity 是颗粒形状规整度指标（4πA/P²、A/A_convex），S 前驱体本身颗粒小、
    烧结程度相对温和，产物二次颗粒更接近前驱体原始形貌（保留更规整的形状）；
    M/L 前驱体粒径大、同一 Θ 下经历更充分的颈缩/融合式热重构，边界更不规则，
    circularity/convexity 应更低。这与 D_sec 方向相反（D_sec 增大 = 颗粒更大，
    circularity/convexity 降低 = 颗粒更不规则）是同一"形貌记忆 vs 热重构"叙事的
    两个互补侧面，但本脚本对此方向假设不做二次裁决式验证，只如实标注为
    "本脚本推断"，不是仓库既有定论。

`SIEVED_CONDITIONS_T1` / `LOW_THETA_SIEVED_TRIO` 直接取自 T1 报告结论，仅用于
交叉引用（是否与本次排序反转条件重合），不重复计算/不重新裁决 T1。

δ=0.02 g/cm³（compaction_density 的 |M-L| 并列判定阈值）是任务书直接给定的
**工程判定阈值**，不是本脚本估计的仪器重复性标准差——约为压实密度量程
（3.04–3.56 g/cm³）的 4%。本轮没有仪器重复性测量数据（σ₀），不编造替代值。
该阈值只对 compaction_density（单位 g/cm³）有意义，D_sec/circularity/convexity
没有任务书给定或仓库既有的对应工程阈值，本脚本对这三个变量的并列判定一律输出
NaN 并在报告中说明原因，不外推 0.02 这个数字到其它单位/量纲的变量上。

产出：
  data/interim/ordering_consistency.csv
  reports/ordering_consistency_report.md
"""
import _bootstrap  # noqa

from pathlib import Path
from itertools import permutations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

MASTER_TABLE = "data/processed/master_table.csv"
OUT_CSV = Path("data/interim/ordering_consistency.csv")
OUT_REPORT = Path("reports/ordering_consistency_report.md")

# T1 结论（reports/sieving_confound_report.md），仅交叉引用，不重新计算/裁决。
SIEVED_CONDITIONS_T1 = {"C01", "C04", "C07", "C14", "C21"}
LOW_THETA_SIEVED_TRIO = {"C01", "C04", "C07"}  # T1 §2 联合格：T_C=850 & t_hold=10，Fisher p=0.0034

DELTA_TIE = 0.02  # g/cm3，任务书给定的工程判定阈值（compaction_density 专用）

# direction: "increasing" 预期 S<M(<L)；"decreasing" 预期 S>M(>L)
VARIABLES = {
    "compaction_density": dict(direction="increasing", label="压实密度",
                                unit="g/cm3", has_tie_threshold=True),
    "D_sec": dict(direction="increasing", label="SEM二次颗粒尺寸D_sec",
                  unit="um", has_tie_threshold=False),
    "circularity": dict(direction="decreasing", label="圆形度circularity",
                         unit="-", has_tie_threshold=False),
    "convexity": dict(direction="decreasing", label="凸度convexity",
                       unit="-", has_tie_threshold=False),
}

ALL_PATTERNS = ["<".join(p) for p in permutations(["S", "M", "L"])]


def load_data():
    df = pd.read_csv(MASTER_TABLE)
    return df


def pivot_var(df: pd.DataFrame, col: str) -> pd.DataFrame:
    piv = df.pivot(index="condition_id", columns="precursor", values=col)
    piv = piv[["S", "M", "L"]]
    return piv


def condition_meta(df: pd.DataFrame) -> pd.DataFrame:
    meta = df.groupby("condition_id", observed=True)[["T_C", "beta", "t_hold", "Theta"]].first()
    return meta


def ordering_pattern_row(row: pd.Series) -> str:
    """行 = 一个条件的 S/M/L 三个值；返回按升序排列的字符串，如 'S<M<L'。
    显式处理并列（exact tie）：两值相等时用 '=' 而非 '<' 连接，不把 tie 静默
    并入某个严格全排列桶里（例如 compaction_density C27 条件 M=L=3.38 精确相等，
    若用 pandas.sort_values 的稳定排序直接拼字符串会把它错误标成 'S<M<L' 严格
    模式，本函数显式检测数值相等并标注为 'S<M=L'）。"""
    vals = row[["S", "M", "L"]]
    order = sorted(vals.index, key=lambda k: vals[k])
    parts = [order[0]]
    for i in range(1, 3):
        op = "=" if vals[order[i]] == vals[order[i - 1]] else "<"
        parts.append(op)
        parts.append(order[i])
    return "".join(parts)


def full_order_match(row: pd.Series, direction: str) -> bool:
    """严格布尔链判定，不经过 pattern 字符串（避免 tie 被字符串相等判断误判为
    匹配）。direction='increasing' 要求 S<M 且 M<L（严格）；'decreasing' 要求
    S>M 且 M>L（严格）。"""
    s, m, l = row["S"], row["M"], row["L"]
    if direction == "increasing":
        return bool(s < m and m < l)
    return bool(s > m and m > l)


def pattern_frequency_table(patterns: pd.Series) -> pd.DataFrame:
    """频次表：先列出 6 种严格全排列（含未出现的补 0），再把实际观测到但不属于
    这 6 种严格全排列的模式（即含并列 '=' 的行，若存在）作为附加行列在后面，
    并加注说明，不强行塞进某个严格排列桶。"""
    counts_all = patterns.value_counts()
    counts_strict = counts_all.reindex(ALL_PATTERNS, fill_value=0)
    out = counts_strict.reset_index()
    out.columns = ["pattern", "n"]
    out["frac"] = out["n"] / len(patterns)
    extra_patterns = [p for p in counts_all.index if p not in ALL_PATTERNS]
    if extra_patterns:
        extra_rows = pd.DataFrame({
            "pattern": extra_patterns,
            "n": [int(counts_all[p]) for p in extra_patterns],
        })
        extra_rows["frac"] = extra_rows["n"] / len(patterns)
        out = pd.concat([out, extra_rows], ignore_index=True)
    return out


def directional_props(piv: pd.DataFrame, direction: str):
    """返回 (prop_S_vs_M bool序列, prop_S_vs_L bool序列)，按 direction 定义比较方向。"""
    if direction == "increasing":
        p_sm = piv["S"] < piv["M"]
        p_sl = piv["S"] < piv["L"]
    else:
        p_sm = piv["S"] > piv["M"]
        p_sl = piv["S"] > piv["L"]
    return p_sm, p_sl


def sign_test(n_success: int, n_total: int):
    res = binomtest(n_success, n_total, p=0.5, alternative="two-sided")
    return float(res.pvalue)


def build_variable_stats(piv: pd.DataFrame, direction: str, has_tie_threshold: bool):
    n = len(piv)
    patterns = piv.apply(ordering_pattern_row, axis=1)
    freq_table = pattern_frequency_table(patterns)
    full_match = piv.apply(lambda row: full_order_match(row, direction), axis=1)
    p_sm, p_sl = directional_props(piv, direction)

    n_sm = int(p_sm.sum())
    n_sl = int(p_sl.sum())
    n_full = int(full_match.sum())

    p_sm_test = sign_test(n_sm, n)
    p_sl_test = sign_test(n_sl, n)
    # 完整排序 exact-match 的机会水平是 1/6（6 种等可能全排列之一），不是 1/2；
    # 用 binomtest(p=1/6) 明确标注零假设，不套用 0.5（那样会是编造的错误零假设）。
    p_full_test = float(binomtest(n_full, n, p=1 / 6, alternative="two-sided").pvalue)

    if has_tie_threshold:
        tie = (piv["M"] - piv["L"]).abs() < DELTA_TIE
        n_tie = int(tie.sum())
        p_tie_test = sign_test(n_tie, n)
    else:
        tie = pd.Series([np.nan] * n, index=piv.index)
        n_tie = np.nan
        p_tie_test = np.nan

    return dict(
        patterns=patterns, freq_table=freq_table, full_match=full_match,
        p_sm=p_sm, p_sl=p_sl, tie=tie,
        n=n, n_sm=n_sm, n_sl=n_sl, n_full=n_full, n_tie=n_tie,
        p_sm_test=p_sm_test, p_sl_test=p_sl_test, p_full_test=p_full_test,
        p_tie_test=p_tie_test,
    )


def reversal_breakdown(meta: pd.DataFrame, is_reversal: pd.Series, factor: str) -> pd.DataFrame:
    tmp = meta.copy()
    tmp["is_reversal"] = is_reversal.reindex(tmp.index)
    grp = tmp.groupby(factor, observed=True)["is_reversal"].agg(["sum", "count"])
    grp["frac"] = grp["sum"] / grp["count"]
    grp = grp.rename(columns={"sum": "n_reversal", "count": "n_total"})
    return grp


def main():
    df = load_data()
    for col in ("compaction_density", "D_sec", "circularity", "convexity"):
        assert df[col].notna().sum() == 81, f"{col} 应 81/81 完整"

    meta = condition_meta(df)
    assert len(meta) == 27

    stats_by_var = {}
    pivots = {}
    for col, spec in VARIABLES.items():
        piv = pivot_var(df, col)
        pivots[col] = piv
        stats_by_var[col] = build_variable_stats(piv, spec["direction"], spec["has_tie_threshold"])

    # ---------------------------------------------------------------
    # 组装逐条件 CSV
    # ---------------------------------------------------------------
    rows = []
    for cond in meta.index:
        row = dict(condition_id=cond, T_C=meta.loc[cond, "T_C"], beta=meta.loc[cond, "beta"],
                   t_hold=meta.loc[cond, "t_hold"], Theta=meta.loc[cond, "Theta"])
        for col in VARIABLES:
            piv = pivots[col]
            st = stats_by_var[col]
            row[f"{col}_S"] = piv.loc[cond, "S"]
            row[f"{col}_M"] = piv.loc[cond, "M"]
            row[f"{col}_L"] = piv.loc[cond, "L"]
            row[f"{col}_pattern"] = st["patterns"].loc[cond]
            row[f"{col}_full_order_match"] = bool(st["full_match"].loc[cond])
            row[f"{col}_is_reversal"] = not bool(st["full_match"].loc[cond])
            row[f"{col}_prop_S_vs_M"] = bool(st["p_sm"].loc[cond])
            row[f"{col}_prop_S_vs_L"] = bool(st["p_sl"].loc[cond])
            tie_val = st["tie"].loc[cond]
            row[f"{col}_ML_tie_within_delta"] = (bool(tie_val) if pd.notna(tie_val) else np.nan)
        row["in_T1_sieved_conditions"] = cond in SIEVED_CONDITIONS_T1
        row["in_T1_low_theta_sieved_trio"] = cond in LOW_THETA_SIEVED_TRIO
        rows.append(row)
    diag = pd.DataFrame(rows).set_index("condition_id").reset_index()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    diag.to_csv(OUT_CSV, index=False)

    # ---------------------------------------------------------------
    # 报告
    # ---------------------------------------------------------------
    lines = []
    lines.append("# T4 —— S/M/L 排序模式稳健性诊断报告\n")
    lines.append("生成脚本：`scripts/14_ordering_consistency.py`；数据源："
                  "`data/processed/master_table.csv`（81 样本 = 27 条件 × S/M/L，"
                  "compaction_density/D_sec/circularity/convexity 均 81/81 完整）。"
                  "交叉引用：`reports/sieving_confound_report.md`（T1，过筛结构性"
                  "混杂）、`reports/interaction_report.md`（Part B 交互 ANOVA）。"
                  "本报告不重新计算/不重新裁决 T1 结论，仅交叉引用其条件清单。\n")

    cd = stats_by_var["compaction_density"]

    # --- headline ---
    lines.append("## 0. 论文可直接引用的结论句\n")
    lines.append(
        f"**compaction_density 完整排序 S<M<L 在 27 个条件中成立 "
        f"{cd['n_full']}/27 次，二项检验（零假设=完整排序随机出现的机会水平 "
        f"1/6，非 50%——6 种全排列等可能是本检验的正确零假设，用 50% 会是"
        f"错误地把三元排序当成二元命题）p={cd['p_full_test']:.4f}。**\n"
    )
    lines.append(
        f"更贴近任务书原始措辞 \"S < M ≲ L\"（允许 M、L 顺序在工程判定阈值 "
        f"δ=0.02 g/cm³ 内互换/接近）的复合判据：S<M **且**（M<L 或 |M-L|<δ）——"
        f"即 S 严格最小，且 L 不比 M 明显更小。该复合命题成立 "
    )
    ml_lenient = ((pivots["compaction_density"]["M"] < pivots["compaction_density"]["L"]) |
                  ((pivots["compaction_density"]["M"] - pivots["compaction_density"]["L"]).abs() < DELTA_TIE))
    s_min = pivots["compaction_density"]["S"] < pivots["compaction_density"]["M"]
    combo = (s_min & ml_lenient)
    n_combo = int(combo.sum())
    lines.append(f"{n_combo}/27 次。此复合命题是两个子命题的合取，不是单一二项"
                  f"随机变量，不适用 50% 零假设的二项检验，此处只报计数，不报 p 值"
                  f"（避免编造一个不成立的零假设）。\n")
    lines.append(
        f"三条具体命题（§2/§3 逐一给出）：ρ_S<ρ_M 成立 {cd['n_sm']}/27 次"
        f"（sign test p={cd['p_sm_test']:.4f}）；ρ_S<ρ_L 成立 {cd['n_sl']}/27 次"
        f"（sign test p={cd['p_sl_test']:.4f}）；|ρ_M-ρ_L|<0.02 成立 "
        f"{cd['n_tie']}/27 次（sign test p={cd['p_tie_test']:.4f}，见 §2 注意事项，"
        f"该命题的 50% 零假设本身不是标准的并列判定检验框架，解读需谨慎）。\n"
    )

    # --- §1 频次表 ---
    lines.append("## 1. compaction_density 排序模式频次表（27 条件，6 种全排列）\n")
    lines.append("| pattern | n | frac |")
    lines.append("|---|---|---|")
    for _, r in cd["freq_table"].iterrows():
        lines.append(f"| {r['pattern']} | {int(r['n'])} | {r['frac']:.4f} |")
    n_extra = len(cd["freq_table"]) - len(ALL_PATTERNS)
    if n_extra > 0:
        lines.append(f"\n⚠️ 上表最后 {n_extra} 行是精确并列（tie）条件，不属于 6 种"
                      f"严格全排列之一——C27 条件 compaction_density M=L=3.38（精确"
                      f"相等，非四舍五入巧合），记为 'S<M=L'，不计入 S<M<L 严格匹配"
                      f"计数，也不计入其它 5 种严格排列。\n")
    lines.append(f"\n完整排序 S<M<L 精确匹配（严格布尔链 S<M 且 M<L，不经过字符串"
                  f"比较，避免 tie 被误判）：{cd['n_full']}/27，"
                  f"binomtest(p0=1/6) p={cd['p_full_test']:.4f}。\n")
    top_pattern = cd["freq_table"].sort_values("n", ascending=False).iloc[0]
    if top_pattern["pattern"] != "S<M<L":
        lines.append(
            f"**如实指出（不回避）**：6 种模式里出现次数最多的**不是** 'S<M<L'"
            f"（{cd['n_full']}/27），而是 '{top_pattern['pattern']}'"
            f"（{int(top_pattern['n'])}/27，{top_pattern['frac']:.1%}）——即 M 组"
            f"压实密度经常**超过** L 组，只是 S 严格最小这一点几乎恒成立"
            f"（ρ_S<ρ_M 26/27、ρ_S<ρ_L 27/27，见 §2-3）。'S<M≲L' 这句表述的"
            f"\"S 最小\"部分证据极强，但\"M≲L\"（M 略小于/接近 L）部分与实际数据"
            f"的主流模式方向相反（更常见的是 M 略大于 L），论文引用时应注意这一"
            f"区别，不应把\"S 最小\"的强证据延伸为\"M<L 也基本成立\"的印象。\n"
        )

    # --- §2/§3 三条命题 + sign test ---
    lines.append("## 2-3. 三个具体命题的成立次数与符号检验（sign test，binomtest p0=0.5）\n")
    lines.append("δ=0.02 g/cm³ 是任务书直接给定的**工程判定阈值**，不是本脚本"
                  "估计的仪器重复性标准差——约为压实密度量程（3.04–3.56 g/cm³）"
                  "的 4%。本轮没有仪器重复性测量数据（σ₀），不编造\"2σ₀\"之类"
                  "的替代方案，直接使用任务书给定的 0.02，重复性实验数据缺失，"
                  "留待未来补测。\n")
    lines.append("| 命题 | 成立次数/27 | 比例 | sign test p (H0=50%) | 备注 |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| ρ_S<ρ_M | {cd['n_sm']} | {cd['n_sm']/27:.4f} | {cd['p_sm_test']:.4f} | "
                  "标准双侧符号检验，H0=随机排序下 50% |")
    lines.append(f"| ρ_S<ρ_L | {cd['n_sl']} | {cd['n_sl']/27:.4f} | {cd['p_sl_test']:.4f} | "
                  "标准双侧符号检验，H0=随机排序下 50% |")
    lines.append(f"| \\|ρ_M-ρ_L\\|<0.02 | {cd['n_tie']} | {cd['n_tie']/27:.4f} | "
                  f"{cd['p_tie_test']:.4f} | ⚠️ 并列判定不是天然的二元 50/50 随机过程"
                  "（连续变量下\"落在某窄带内\"的机会水平本不是 50%，取决于该变量的"
                  "分布方差与 δ 的相对大小），本行按任务书字面要求套用 binomtest(p0=0.5)"
                  "算出，但**不应把这个 p 值解读为\"是否显著偏离随机排序\"的标准证据"
                  "**，仅供参照，解读以左侧计数（7/27 ≈ 26%）为主 |")
    lines.append("")

    # --- §4 形貌描述子 ---
    lines.append("## 4. 形貌描述子（D_sec/circularity/convexity）排序一致性对比\n")
    lines.append("预期方向依据见脚本文件头注释：D_sec 与 compaction_density 同向"
                  "（S<M≲L，已定稿的形貌自适应分割给出 D_sec 的 L/S=2.29）；"
                  "circularity/convexity 预期反向（S>M≳L），此推断基于\"S 前驱体产物"
                  "更接近前驱体原始形貌、M/L 经历更多颈缩式热重构故形状更不规则\"的"
                  "物理逻辑，**本脚本推断，非仓库既有定论**，报告中如实标注。\n")
    lines.append("| 变量 | 预期方向 | 完整排序精确匹配 n/27 (binomtest p0=1/6) | "
                  "ρ_S vs ρ_M 命题成立 n/27 (sign test p) | ρ_S vs ρ_L 命题成立 n/27 (sign test p) |")
    lines.append("|---|---|---|---|---|")
    dir_label = {"increasing": "S<M≲L（同 compaction_density）",
                 "decreasing": "S>M≳L（反向，本脚本推断）"}
    for col, spec in VARIABLES.items():
        st = stats_by_var[col]
        lines.append(
            f"| {spec['label']} ({col}) | {dir_label[spec['direction']]} | "
            f"{st['n_full']}/27 (p={st['p_full_test']:.4f}) | "
            f"{st['n_sm']}/27 (p={st['p_sm_test']:.4f}) | "
            f"{st['n_sl']}/27 (p={st['p_sl_test']:.4f}) |"
        )
    lines.append("")
    lines.append("各变量完整 6-模式频次表：\n")
    for col, spec in VARIABLES.items():
        st = stats_by_var[col]
        lines.append(f"**{spec['label']} ({col})**：")
        parts = [f"{r['pattern']}={int(r['n'])}" for _, r in st["freq_table"].iterrows()]
        lines.append(", ".join(parts) + "\n")
    lines.append("D_sec/circularity/convexity 三个变量没有任务书给定或仓库既有的"
                  "工程判定阈值，本报告不外推 compaction_density 的 δ=0.02 g/cm³ 到"
                  "这些不同量纲/量程的变量上，其\"|M-L|<δ\"并列命题一律输出 NaN"
                  "（见 CSV `_ML_tie_within_delta` 列），不编造替代阈值。\n")

    # --- §5 反转条件交叉检查 ---
    lines.append("## 5. 排序反转条件的交叉检查\n")
    rev_mask = ~cd["full_match"]
    rev_conditions = rev_mask[rev_mask].index.tolist()
    lines.append(f"compaction_density 排序不满足完整 S<M<L 全排列的条件共 "
                  f"{len(rev_conditions)}/27 个：{', '.join(rev_conditions)}。\n")

    lines.append("### 5.1 与 T1 结构性混杂条件的重合核查\n")
    lines.append("| condition_id | 排序模式 | 是否 T1 过筛条件(C01/C04/C07/C14/C21) | "
                  "是否 T1 低Θ联合格三条件(C01/C04/C07，Fisher p=0.0034) |")
    lines.append("|---|---|---|---|")
    overlap_trio = []
    overlap_sieved_only = []
    for c in rev_conditions:
        pat = cd["patterns"].loc[c]
        in_sieved = c in SIEVED_CONDITIONS_T1
        in_trio = c in LOW_THETA_SIEVED_TRIO
        if in_trio:
            overlap_trio.append(c)
        elif in_sieved:
            overlap_sieved_only.append(c)
        lines.append(f"| {c} | {pat} | {'是' if in_sieved else '否'} | "
                      f"{'是' if in_trio else '否'} |")
    lines.append("")
    if overlap_trio:
        lines.append(
            f"**发现（与 T1 交叉验证，非独立新发现）**：反转条件中 "
            f"{', '.join(overlap_trio)} 恰好落在 T1（`reports/sieving_confound_report.md` "
            f"§2）已确认的结构性混杂三元组——T_C=850°C 且 t_hold=10h 联合格"
            f"（27 条件里唯一满足该联合条件的 3 个，Fisher exact p=0.0034）。"
            f"这为 T1 的过筛混杂判定提供了一次**独立方向的交叉验证**：不仅 T1 用"
            f"ANOVA/条件级残差检验发现该三元组与 sieved 结构性关联，本次单纯看"
            f"排序反转也命中了同一批条件——但这里只做交叉引用，不对 T1"
            f"\"过筛混杂是否真实\"这一结论做二次裁决，T1 的判定（结构性混杂确凿，"
            f"论文 §3.5 降级为讨论项）保持不变。\n"
        )
    else:
        lines.append(
            "**发现**：反转条件中**没有**任何一个恰好落在 T1 低Θ联合格三元组 "
            "(C01/C04/C07)——排序反转与 T1 已确认的过筛结构性混杂条件**不重合**，"
            "两者是相互独立的现象，本次未提供额外交叉验证证据（也未提供反证）。\n"
        )
    if overlap_sieved_only:
        lines.append(f"另有 {', '.join(overlap_sieved_only)} 属于 T1 五个过筛条件"
                      f"（C01/C04/C07/C14/C21）中的成员，但不在 T_C=850&t_hold=10 "
                      f"联合格三元组内。\n")

    lines.append("### 5.2 反转比例按 T_C 水平拆分\n")
    tc_break = reversal_breakdown(meta, rev_mask, "T_C")
    lines.append("| T_C | n_reversal | n_total | frac |")
    lines.append("|---|---|---|---|")
    for lvl, r in tc_break.iterrows():
        lines.append(f"| {lvl} | {int(r['n_reversal'])} | {int(r['n_total'])} | {r['frac']:.4f} |")
    lines.append("")

    lines.append("### 5.3 反转比例按 beta 水平拆分\n")
    beta_break = reversal_breakdown(meta, rev_mask, "beta")
    lines.append("| beta | n_reversal | n_total | frac |")
    lines.append("|---|---|---|---|")
    for lvl, r in beta_break.iterrows():
        lines.append(f"| {lvl} | {int(r['n_reversal'])} | {int(r['n_total'])} | {r['frac']:.4f} |")
    lines.append("")

    lines.append("### 5.4 反转比例按 t_hold 水平拆分\n")
    th_break = reversal_breakdown(meta, rev_mask, "t_hold")
    lines.append("| t_hold | n_reversal | n_total | frac |")
    lines.append("|---|---|---|---|")
    for lvl, r in th_break.iterrows():
        lines.append(f"| {lvl} | {int(r['n_reversal'])} | {int(r['n_total'])} | {r['frac']:.4f} |")
    lines.append("")

    lines.append(
        "**解读**：n=27 条件下每个因子只有 3 个水平、每水平 9 个条件，反转计数"
        "本身样本量很小，下方仅作描述性观察，不做进一步显著性检验（若要对\"反转"
        "是否集中在某水平\"做正式检验，需要 3×2 列联表 Fisher/卡方，样本量同样"
        "受限于 n=27，本报告不展开二次检验，只如实列出频次分布供交叉参考）。\n"
    )

    # --- §6 限制声明 ---
    lines.append("## 6. 限制声明\n")
    lines.append("- δ=0.02 g/cm³ 为任务书给定的工程判定阈值，不是统计估计的仪器"
                  "重复性标准差；本轮无重复性测量数据，未来补测后应重新核算"
                  "|ρ_M-ρ_L|<δ 命题。\n")
    lines.append("- 完整 3 元排序精确匹配的正确随机基线是 1/6（6 种等可能全排列），"
                  "本报告对\"S<M<L 完整匹配\"统一使用 binomtest(p0=1/6)，不套用"
                  "50%；任务书§3 明确要求 50% 零假设的三条命题（ρ_S<ρ_M、ρ_S<ρ_L、"
                  "|ρ_M-ρ_L|<δ）本身是二元/成对比较，50% 零假设对它们是恰当的，"
                  "两套零假设不能混用，本报告严格区分标注。\n")
    lines.append("- |ρ_M-ρ_L|<δ 的\"sign test p\"是按任务书字面要求套用 binomtest"
                  "(p0=0.5) 算出的数字，但连续变量落入固定窄带内的机会水平本不"
                  "天然是 50%（取决于该变量的分布方差相对 δ 的大小），此 p 值不能"
                  "作为\"过筛/工艺是否显著导致 M-L 接近\"的标准证据，仅供参照。\n")
    lines.append("- circularity/convexity 的预期方向（S>M≳L）是本脚本基于物理逻辑"
                  "的推断，不是仓库既有定论（D_sec 方向有既有记录支持，"
                  "circularity/convexity 方向没有）；若后续有更明确的方向依据"
                  "（如 SEM 处理器文档更新），应重新核算本节。\n")
    lines.append("- D_sec/circularity/convexity 没有 δ 工程阈值，本报告未对它们计算"
                  "|M-L|<δ 类命题，CSV 中对应列为 NaN。\n")
    lines.append("- §5.2-5.4 的按因子水平拆分反转比例为描述性统计，未做正式的"
                  "显著性检验（样本量 n=27、每水平 9 个条件，检验效力有限）。\n")
    lines.append("- 不修改 `src/nfm/schema.py`；本诊断全部计算逻辑限于 "
                  "`scripts/14_ordering_consistency.py` 脚本内部，未导出为 "
                  "`src/nfm/` 下的可复用函数/类；不重新裁决 T1 的过筛混杂判定、"
                  "T2 的熔连伪影判定、WH 定性参照、5000×口径、Q 不可识别、"
                  "纳米层单仪器口径等任何已定稿结论。\n")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(f"写出 {OUT_CSV}（{len(diag)} 行）")
    print(f"写出 {OUT_REPORT}")
    print(f"headline: compaction_density S<M<L 完整匹配 {cd['n_full']}/27, "
          f"p(1/6)={cd['p_full_test']:.4f}")
    print(f"S<M: {cd['n_sm']}/27 p={cd['p_sm_test']:.4f}; "
          f"S<L: {cd['n_sl']}/27 p={cd['p_sl_test']:.4f}; "
          f"|M-L|<0.02: {cd['n_tie']}/27 p={cd['p_tie_test']:.4f}")


if __name__ == "__main__":
    main()
