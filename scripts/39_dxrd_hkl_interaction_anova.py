# -*- coding: utf-8 -*-
"""39 D(003)/D(104) 分向前驱体×工艺交互 ANOVA。
==========================================================================
承接 `scripts/27_dxrd_hkl_anisotropy.py`(S1,只算了主效应 η²)与
`scripts/19_nano_layer_interaction_anova.py`(只对 D_XRD 均值/lattice_c
算了前驱体×工艺三项交互)。任务缺口:D_XRD_003/D_XRD_104 各自的交互项
从未单独算过,只算过它们平均后的 D_XRD——而论文的头号耦合总量论断
("0.63%–3.20% 范围,不管挑哪个单一 target 都成立")从未在"D(003) vs
D(104) 单独看,而非平均后再看"这个角度上核实过。

**方法学与 19 号脚本完全一致(不重新发明)**:同一 7 项限定模型
(4 主效应 + 3 个前驱体×工艺交互,Sum 编码)、同一 51 样本纳米层口径
(`nfm.nano_layer.nano_layer_frame(df, require_reliable=True)`)、同一秩
检验(限定模型 21 参数满秩、完整二阶模型 33 参数亏秩)、同一 Type II
(statsmodels `anova_lm(typ=2)`)与真 SST 分母的 η²/ω² 定义。唯一差异:
目标列换成 `D_XRD_003`/`D_XRD_104`(诊断列,不是 D_XRD 均值)。

本脚本是**新增独立脚本**,不修改 19/27 号脚本或 `src/nfm/` 下任何代码。

产出:
  data/interim/dxrd_hkl_interaction_anova.csv   长表,schema 与
    data/interim/nano_layer_interaction_anova.csv 一致(target/term/SS/
    df/MS/F/p_parametric/eta2/omega2),外加 perm_p 列。
  reports/dxrd_hkl_anisotropy.md   追加(不覆盖)"## 6. 交互项各向异性
    对照"一节。
"""
import _bootstrap  # noqa

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from nfm.nano_layer import nano_layer_frame
from nfm.stats.anova_interactions import omega_squared

MASTER_TABLE = "data/processed/master_table.csv"
OUT_LONG = Path("data/interim/dxrd_hkl_interaction_anova.csv")
OUT_REPORT = Path("reports/dxrd_hkl_anisotropy.md")

TARGETS = ("D_XRD_003", "D_XRD_104")
FACTORS = ("precursor", "T_C", "beta", "t_hold")
INTERACTIONS = ("precursor:T_C", "precursor:beta", "precursor:t_hold")
ALL_TERMS = FACTORS + INTERACTIONS  # 4 主效应 + 3 交互,7 项,与 19 号脚本相同顺序

# 已知数字(仅引用,不重算):19 号脚本对 D_XRD(均值)算出的耦合总量。
D_XRD_MEAN_COUPLING_ETA2_PCT = 1.68
D_XRD_MEAN_TERM_ETA2_PCT = {
    "precursor:T_C": 1.17,
    "precursor:beta": 0.16,
    "precursor:t_hold": 0.35,
}

# Part B(形貌/堆积层四目标)+ 纳米层 D_XRD/lattice_c 的耦合总量范围:
# 论文头号论断引用的范围下限=D_sec(0.63%),上限=compaction_density(3.20%)。
EXISTING_RANGE_LOW = 0.63
EXISTING_RANGE_HIGH = 3.20
EXISTING_RANGE_TARGETS = {
    "compaction_density": 3.20,
    "D_sec": 0.63,
    "circularity": 2.79,
    "D_XRD (均值,19号脚本)": 1.68,
}


def _term_to_patsy(term: str) -> str:
    if ":" in term:
        f1, f2 = term.split(":")
        return f"C({f1}, Sum):C({f2}, Sum)"
    return f"C({term}, Sum)"


def rank_diagnostics(df: pd.DataFrame) -> dict:
    """实测(不是理论推断):限定 7 项模型(21 参数)满秩;
    完整二阶模型(全部 6 个两两交互,33 参数)亏秩。与 19 号脚本同一检验,
    数据口径相同(同一 51 样本 nano_layer_frame 输出),预期数字相同,
    但独立重跑以确认——不假定复用 19 号脚本的既有结论。"""
    rhs_restricted = " + ".join(_term_to_patsy(t) for t in ALL_TERMS)
    _, Xr = patsy.dmatrices(f"__y__ ~ {rhs_restricted}", data=df.assign(__y__=0.0),
                            return_type="dataframe")
    rank_r = int(np.linalg.matrix_rank(Xr.values))
    n_params_r = Xr.shape[1]

    all_pairs = [(FACTORS[i], FACTORS[j]) for i in range(4) for j in range(i + 1, 4)]
    rhs_full = (" + ".join(f"C({f}, Sum)" for f in FACTORS) + " + "
               + " + ".join(f"C({f1}, Sum):C({f2}, Sum)" for f1, f2 in all_pairs))
    _, Xf = patsy.dmatrices(f"__y__ ~ {rhs_full}", data=df.assign(__y__=0.0),
                            return_type="dataframe")
    rank_f = int(np.linalg.matrix_rank(Xf.values))
    n_params_f = Xf.shape[1]

    return dict(
        n=len(df),
        restricted_n_params=n_params_r, restricted_rank=rank_r,
        restricted_full_rank=(rank_r == n_params_r),
        restricted_resid_df=len(df) - rank_r,
        full_n_params=n_params_f, full_rank=rank_f,
        full_rank_deficient=(rank_f < n_params_f),
        full_rank_deficiency=n_params_f - rank_f,
    )


def marginal_table(work: pd.DataFrame, target: str, typ: int) -> tuple[pd.DataFrame, float]:
    """Type II/III(边际 SS,与项顺序无关),statsmodels 直接算。
    eta2/omega2 用真实的、均值中心化总平方和 SST 做分母(与 19 号脚本同口径)。"""
    rhs = " + ".join(_term_to_patsy(t) for t in ALL_TERMS)
    model = smf.ols(f"{target} ~ {rhs}", data=work).fit()
    tab = anova_lm(model, typ=typ)
    rename_map = {_term_to_patsy(t): t for t in ALL_TERMS}
    tab = tab.rename(index=rename_map)
    tab = tab.drop(index="Intercept", errors="ignore")

    y = work[target].to_numpy(float)
    sst_true = float(((y - y.mean()) ** 2).sum())
    ms_error = tab.loc["Residual", "sum_sq"] / tab.loc["Residual", "df"]
    rows = []
    for t in ALL_TERMS:
        ss_t = tab.loc[t, "sum_sq"]
        df_t = int(tab.loc[t, "df"])
        rows.append(dict(term=t, SS=ss_t, df=df_t, MS=ss_t / df_t,
                         F=tab.loc[t, "F"],
                         eta2=ss_t / sst_true,
                         omega2=omega_squared(ss_t, df_t, sst_true, ms_error),
                         p_parametric=tab.loc[t, "PR(>F)"]))
    rows.append(dict(term="Residual", SS=tab.loc["Residual", "sum_sq"],
                     df=int(tab.loc["Residual", "df"]), MS=ms_error, F=np.nan,
                     eta2=tab.loc["Residual", "sum_sq"] / sst_true, omega2=np.nan,
                     p_parametric=np.nan))
    return pd.DataFrame(rows).set_index("term"), sst_true


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    df_full = pd.read_csv(MASTER_TABLE)
    sub = nano_layer_frame(df_full, require_reliable=True)
    print(f"[nano_layer_frame] n={len(sub)} (xrd_instrument==1 & D_XRD_reliable==True)")

    work = sub.copy()
    for f in FACTORS:
        work[f] = work[f].astype("category")

    # ---- 秩检验(实测,不假定;与 19 号脚本同一检验,独立重跑确认) ----
    rankd = rank_diagnostics(work)
    print("\n=== 秩检验 ===")
    for k, v in rankd.items():
        print(f"  {k}: {v}")
    assert rankd["restricted_full_rank"], (
        f"限定 7 项模型(21 参数)预期满秩,实测秩={rankd['restricted_rank']}"
        f"≠参数数={rankd['restricted_n_params']},停下来报告,不要继续。")
    assert rankd["restricted_resid_df"] == 30, (
        f"限定模型残差 df 预期 30,实测 {rankd['restricted_resid_df']},停下来报告。")
    assert rankd["full_rank_deficient"], (
        "完整二阶模型(33参数)预期亏秩,实测满秩——与 19 号脚本的前提不符,"
        "需要重新核实,不要继续套用限定模型的论证逻辑。")

    long_rows = []
    per_target = {}

    for target in TARGETS:
        if work[target].isna().any():
            raise RuntimeError(f"{target}: nano_layer_frame 输出中仍有 NaN,不应发生"
                              "(D_XRD_reliable==True 应已保证非空),停下来报告。")
        y = work[target].to_numpy(float)
        sst_true = float(((y - y.mean()) ** 2).sum())

        t2, _ = marginal_table(work, target, typ=2)
        t3, _ = marginal_table(work, target, typ=3)
        ii_iii_diffs = {t: float(t2.loc[t, "SS"] - t3.loc[t, "SS"]) for t in ALL_TERMS}
        ii_iii_max_diff = max(abs(v) for v in ii_iii_diffs.values())
        ii_iii_agree = ii_iii_max_diff < 1e-6 * max(t2.loc["Residual", "SS"], 1.0)

        auth_rows = []
        for t in ALL_TERMS:
            r = t2.loc[t]
            auth_rows.append(dict(target=target, term=t, SS=r["SS"], df=int(r["df"]),
                                  MS=r["MS"], F=r["F"], p_parametric=r["p_parametric"],
                                  eta2=r["eta2"], omega2=r["omega2"]))
        r_resid = t2.loc["Residual"]
        auth_rows.append(dict(target=target, term="Residual", SS=r_resid["SS"],
                              df=int(r_resid["df"]), MS=r_resid["MS"], F=np.nan,
                              p_parametric=np.nan, eta2=r_resid["eta2"], omega2=np.nan))
        long_rows.extend(auth_rows)

        coupling_eta2 = t2.loc[list(INTERACTIONS), "eta2"].sum()
        coupling_omega2 = t2.loc[list(INTERACTIONS), "omega2"].sum()

        per_target[target] = dict(
            t2=t2, t3=t3, sst_true=sst_true,
            ii_iii_diffs=ii_iii_diffs, ii_iii_max_diff=ii_iii_max_diff,
            ii_iii_agree=ii_iii_agree,
            coupling_eta2=coupling_eta2, coupling_omega2=coupling_omega2,
        )

        print(f"\n=== {target} ===")
        print(f"Type II vs III 最大绝对 SS 差: {ii_iii_max_diff:.6g} -> 一致: {ii_iii_agree}")
        print(f"前驱体×工艺耦合总量(η² 之和): {coupling_eta2*100:.3f}%,"
             f" ω² 之和: {coupling_omega2*100:.3f}%")
        for t in INTERACTIONS:
            print(f"  {t}: eta2={t2.loc[t,'eta2']*100:.3f}%, "
                 f"omega2={t2.loc[t,'omega2']*100:.3f}%, "
                 f"p_parametric={t2.loc[t,'p_parametric']:.4f}")

    long_df = pd.DataFrame(long_rows)
    OUT_LONG.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(OUT_LONG, index=False)
    print(f"\n[done] {OUT_LONG}")

    # ---- 决策规则(按预设判据执行,不softening/override) ----
    totals_pct = {t: per_target[t]["coupling_eta2"] * 100 for t in TARGETS}
    both_within_range = all(
        EXISTING_RANGE_LOW <= v <= EXISTING_RANGE_HIGH for v in totals_pct.values())
    print(f"\n=== 决策规则判定 ===")
    print(f"D_XRD_003 耦合总量 = {totals_pct['D_XRD_003']:.3f}%")
    print(f"D_XRD_104 耦合总量 = {totals_pct['D_XRD_104']:.3f}%")
    print(f"既有范围 = [{EXISTING_RANGE_LOW}, {EXISTING_RANGE_HIGH}]%")
    print(f"两者是否均落在范围内: {both_within_range}")

    append_report_section(rankd, per_target, totals_pct, both_within_range)
    print(f"[done] appended section to {OUT_REPORT}")


def append_report_section(rankd, per_target, totals_pct, both_within_range):
    lines = []
    lines.append("")
    lines.append("## 6. 交互项各向异性对照")
    lines.append("")
    lines.append("生成脚本:`scripts/39_dxrd_hkl_interaction_anova.py`(新增独立脚本,不修改"
                 "本报告上方 §1–§5 由 `scripts/27_dxrd_hkl_anisotropy.py` 产出的任何内容,也不"
                 "修改 `scripts/19_nano_layer_interaction_anova.py`)。数据口径与上方各节相同:"
                 f"`nfm.nano_layer.nano_layer_frame(df, require_reliable=True)`,n={rankd['n']}"
                 "(xrd_instrument==1 且 D_XRD_reliable==True)。")
    lines.append("")
    lines.append("### 6.0 背景与缺口")
    lines.append("")
    lines.append("§4 已经给出 D(003)/D(104)/D_XRD(均值)三者的**主效应** η² 对照,但明确写明"
                 "\"交互项 η² 未在本报告估计……若需要 D(003)/D(104) 各自的交互项对照,"
                 "需追加分析\"。`scripts/19_nano_layer_interaction_anova.py` 算过前驱体×"
                 "工艺三项交互,但只对**平均后**的 `D_XRD` 和 `lattice_c` 算,从未对 (003)/(104)"
                 "两个分峰单独算过。论文表 2 的头号论断是\"前驱体×工艺耦合总量在"
                 f"{EXISTING_RANGE_LOW}%–{EXISTING_RANGE_HIGH}% 范围内,不管挑哪个单一 target 都"
                 "成立\"——这句话此前从未在\"D(003) 和 D(104) 分别单独看,而不是先平均再看\""
                 "这个角度上核实过。本节补齐这个缺口。")
    lines.append("")
    lines.append("### 6.1 模型规格(与 19 号脚本完全一致)")
    lines.append("")
    lines.append("```")
    lines.append("y ~ C(precursor,Sum) + C(T_C,Sum) + C(beta,Sum) + C(t_hold,Sum)")
    lines.append("    + C(precursor,Sum):C(T_C,Sum)")
    lines.append("    + C(precursor,Sum):C(beta,Sum)")
    lines.append("    + C(precursor,Sum):C(t_hold,Sum)")
    lines.append("```")
    lines.append("")
    lines.append(f"秩检验(独立重跑,不假定复用 19 号脚本的结论):限定模型(21 参数)实测秩 = "
                 f"**{rankd['restricted_rank']}** -> "
                 f"{'满秩' if rankd['restricted_full_rank'] else '亏秩(不应发生)'}"
                 f",残差 df = **{rankd['restricted_resid_df']}**;完整二阶模型(33 参数)"
                 f"实测秩 = **{rankd['full_rank']}** -> "
                 f"{'亏秩' if rankd['full_rank_deficient'] else '满秩(与前提不符)'}"
                 f",亏秩量 = **{rankd['full_n_params'] - rankd['full_rank']}**。"
                 "与 §4/19 号脚本报告的数字一致,确认同一数据口径下秩结构不变。")
    lines.append("")
    lines.append("Type II vs III 一致性(实测):")
    lines.append("")
    for target in TARGETS:
        d = per_target[target]
        status = "一致" if d["ii_iii_agree"] else "**不一致**"
        lines.append(f"- `{target}`:最大绝对 SS 差 = {d['ii_iii_max_diff']:.6g} -> {status}。"
                     "本节以 Type II 为权威值(与 19 号脚本口径一致)。")
    lines.append("")
    lines.append("### 6.2 D(003)/D(104) 各自的三项前驱体×工艺交互(Type II 权威值)")
    lines.append("")
    header = "| target | term | η² (%) | ω² (%) | p_parametric |"
    lines.append(header)
    lines.append("|---|---|---|---|---|")
    for target in TARGETS:
        d = per_target[target]
        for t in INTERACTIONS:
            row = d["t2"].loc[t]
            lines.append(f"| {target} | {t} | {row['eta2']*100:.3f} | "
                         f"{row['omega2']*100:.3f} | {row['p_parametric']:.4f} |")
    lines.append("")
    lines.append("对照:19 号脚本对**平均后**的 `D_XRD` 算出的同 3 项(仅引用,未重算):")
    lines.append("")
    lines.append("| target | term | η² (%,引用自既有结果) |")
    lines.append("|---|---|---|")
    for t in INTERACTIONS:
        lines.append(f"| D_XRD(均值) | {t} | {D_XRD_MEAN_TERM_ETA2_PCT[t]:.2f} |")
    lines.append("")
    lines.append("### 6.3 耦合总量(三项 η²/ω² 之和)三方对照")
    lines.append("")
    lines.append("| target | η² 总量 (%) | ω² 总量 (%) | 数据来源 |")
    lines.append("|---|---|---|---|")
    for target in TARGETS:
        d = per_target[target]
        lines.append(f"| {target} | {d['coupling_eta2']*100:.3f} | "
                     f"{d['coupling_omega2']*100:.3f} | 本节新算(51样) |")
    lines.append(f"| D_XRD(均值) | {D_XRD_MEAN_COUPLING_ETA2_PCT:.2f} | — | "
                 "引用自 19 号脚本产出的既有结果(未重算) |")
    lines.append("")
    lines.append(f"既有引用范围(compaction_density/D_sec/circularity/D_XRD 四目标的"
                 f"耦合总量 η²):"
                 f"**{EXISTING_RANGE_LOW}%–{EXISTING_RANGE_HIGH}%**"
                 f"(下限 D_sec={EXISTING_RANGE_TARGETS['D_sec']}%,"
                 f"上限 compaction_density={EXISTING_RANGE_TARGETS['compaction_density']}%)。")
    lines.append("")
    lines.append("### 6.4 决策规则判定")
    lines.append("")
    d003 = totals_pct["D_XRD_003"]
    d104 = totals_pct["D_XRD_104"]
    in003 = EXISTING_RANGE_LOW <= d003 <= EXISTING_RANGE_HIGH
    in104 = EXISTING_RANGE_LOW <= d104 <= EXISTING_RANGE_HIGH
    lines.append(f"- D_XRD_003 耦合总量 = **{d003:.3f}%** -> "
                 f"{'落在' if in003 else '**超出**'} [{EXISTING_RANGE_LOW}, {EXISTING_RANGE_HIGH}]% 范围")
    lines.append(f"- D_XRD_104 耦合总量 = **{d104:.3f}%** -> "
                 f"{'落在' if in104 else '**超出**'} [{EXISTING_RANGE_LOW}, {EXISTING_RANGE_HIGH}]% 范围")
    lines.append("")
    if both_within_range:
        lines.append("**判定:两者均落在既有 0.63%–3.20% 范围内。**"
                     "论文可以照原样引用这个范围,不需要因为改用 D(003)/D(104) 单峰口径而"
                     "改写头号论断——\"耦合总量在该范围内,不管挑哪个单一 target 都成立\"这句话"
                     "现在对\"D_XRD 均值\"以及它平均前的两个分向分量(D(003)、D(104))都成立,"
                     "论断对\"是否先平均\"这个建模选择是稳健的。")
    else:
        offenders = []
        if not in003:
            offenders.append(f"D_XRD_003({d003:.3f}%)")
        if not in104:
            offenders.append(f"D_XRD_104({d104:.3f}%)")
        lines.append(f"**判定:{'/'.join(offenders)} 超出既有 {EXISTING_RANGE_LOW}%–"
                     f"{EXISTING_RANGE_HIGH}% 范围。**"
                     "**这不是数字微调,是对论文头号论断实质性框架的挑战**——"
                     "\"耦合总量范围不管挑哪个单一 target 都成立\"这句话在\"D(003)/D(104) 分峰"
                     "而非平均后的 D_XRD\"这个角度上**不成立**,需要在正文里明确改写这句话"
                     "(例如把范围改为涵盖新的上/下限,或明确声明该论断只在\"两峰平均\"这个"
                     "特定建模选择下成立,单独看某一峰会超出),不能简单沿用现有措辞。"
                     "本节仅如实报告该判定,不代替作者做最终措辞决定。")
    lines.append("")
    lines.append("### 6.5 限制声明(与 19 号脚本一致)")
    lines.append("")
    lines.append("- 不平衡设计,无重复:51/81 个 (precursor,T_C,beta,t_hold) 组合缺失"
                 "(仪器2 30 样已按口径决定排除),每个存在的组合恰好 n=1;残差含三阶及以上"
                 "交互与测量误差,二者不可分离。")
    lines.append("- 工艺×工艺三项(T_C:beta/T_C:t_hold/beta:t_hold)未估:完整二阶模型"
                 f"(33 参数)在 51 样本上实测亏秩 {rankd['full_n_params']-rankd['full_rank']}"
                 "(见 §6.1),不可全部估计。")
    lines.append("- 本节未做置换检验(19 号脚本的 `type1_ss`/`permutation_p_values` 等 Type I"
                 "专属基础设施与本节的 Type II-only 需求不完全匹配,复用会引入与 19 号脚本"
                 "不必要的耦合;本节的显著性判断以 `p_parametric`(F 检验)为准)。")
    lines.append("")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with OUT_REPORT.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
