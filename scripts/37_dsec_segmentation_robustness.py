# -*- coding: utf-8 -*-
"""37 D_sec 两条头号结论(前驱体主效应88.8% / 交互总量0.63%)对 SEM 500x
分割参数选择的稳健性审计。
==========================================================================
背景:`scripts/02j_mdagg_fusion_check.py` 之前已产出 9 组分割参数(percentile/
slope 在生产默认值附近扰动)敏感性扫描 `data/interim/mdagg_sensitivity_sweep.csv`
(729 行 = 81 样本 x 9 组合),但那一轮只检验了 Spearman(M_D_agg, Theta) 在 9 组
参数下是否保持显著(9/9 通过),**没有**检验 D_sec 自身的两条论文头号结论
(precursor 主效应 η²=88.8%,表1;precursor×工艺交互总量 η²=0.63%,表2)是否
对同一批参数扰动稳健。

本脚本:
  1. 对 9 个 combo_id,各自取该 combo 的逐样本 D_sec(来自 mdagg_sensitivity_sweep.csv),
     替换进 master_table.csv 的一份拷贝(precursor/T_C/beta/t_hold/condition_id
     等其它列不随分割参数变化,原样保留),用
     `nfm.stats.anova_interactions.run_anova_for_target` 精确复用生产 ANOVA 引擎
     (81 样本完整平衡析因设计,Type I≡II≡III)重算:
       - precursor 主效应 η²(表1头号数字口径)
       - precursor:T_C + precursor:beta + precursor:t_hold 三项 η² 之和(表2口径)
       - D_sec(S)<D_sec(M) 与 D_sec(S)<D_sec(L) 在 27 个条件三元组中成立的次数
         (逐 combo 直接从该 combo 的 D_sec 透视表计数,不复用
         production-only 的 ordering_consistency.csv)
  2. 汇总 9 行对照表 + 相对 default(生产参数)组合的最大绝对偏差
  3. 按预设的判定规则(不软化/不回避)写出 reports/dsec_segmentation_robustness.md
  4. 落盘 data/interim/dsec_segmentation_robustness.csv

**不修改** src/nfm/、scripts/12_anova_interactions.py、scripts/14_ordering_consistency.py、
data/processed/master_table.csv、data/interim/mdagg_sensitivity_sweep.csv、
data/interim/ordering_consistency.csv 下任何文件——本脚本只读取
这些文件,只写自己的输出。
"""
import _bootstrap  # noqa

from pathlib import Path

import numpy as np
import pandas as pd

from nfm.stats.anova_interactions import PRECURSOR_PROCESS_TERMS, run_anova_for_target

MASTER_TABLE = "data/processed/master_table.csv"
SWEEP_CSV = "data/interim/mdagg_sensitivity_sweep.csv"
OUT_CSV = Path("data/interim/dsec_segmentation_robustness.csv")
OUT_REPORT = Path("reports/dsec_segmentation_robustness.md")

PRODUCTION_COMBO = "default"  # percentile=90, slope=0.082,生产设置

# 判定规则(预设,原样照抄,不软化)
INTERACTION_MAX_PCT = 3.20      # interaction_total_eta2_pct 全程必须 < 此值
ORDERING_TOLERANCE = 2          # 两个排序计数相对 default 组合的容许偏差(±)


def compute_combo_row(df_master: pd.DataFrame, sweep_combo: pd.DataFrame, combo_id: str) -> dict:
    """对单个 combo_id:用该 combo 的逐样本 D_sec 替换 master 里的 D_sec 列,
    跑 run_anova_for_target,并直接从透视表数排序计数。返回一行 dict。"""
    d_sec_map = sweep_combo.set_index("sample_id")["D_sec"]
    work = df_master.copy()
    # sample_id 覆盖率检查:sweep 里的 81 个样本应与 master 完全对齐
    missing = set(work["sample_id"]) - set(d_sec_map.index)
    if missing:
        raise RuntimeError(f"combo={combo_id}: master 中 {len(missing)} 个 sample_id "
                           f"在 sweep 里找不到对应 D_sec,例如 {sorted(missing)[:5]}")
    work["D_sec"] = work["sample_id"].map(d_sec_map)
    if work["D_sec"].isna().any():
        raise RuntimeError(f"combo={combo_id}: 替换后 D_sec 出现 NaN,"
                           f"n_na={int(work['D_sec'].isna().sum())}")

    res = run_anova_for_target(work, "D_sec", n_perm=5000, seed=2000)
    table = res["table"]
    precursor_eta2_pct = float(table.loc["precursor", "eta2"] * 100)
    interaction_total_eta2_pct = float(table.loc[list(PRECURSOR_PROCESS_TERMS), "eta2"].sum() * 100)

    piv = work.pivot(index="condition_id", columns="precursor", values="D_sec")
    piv = piv[["S", "M", "L"]]
    n_s_lt_m = int((piv["S"] < piv["M"]).sum())
    n_s_lt_l = int((piv["S"] < piv["L"]).sum())

    return dict(
        combo_id=combo_id,
        precursor_eta2_pct=precursor_eta2_pct,
        interaction_total_eta2_pct=interaction_total_eta2_pct,
        D_sec_S_lt_M_count=n_s_lt_m,
        D_sec_S_lt_L_count=n_s_lt_l,
        type_agreement=bool(res["type_agreement"]),
    )


def main():
    df_master = pd.read_csv(MASTER_TABLE)
    sweep = pd.read_csv(SWEEP_CSV)

    combo_ids = list(sweep["combo_id"].unique())
    assert len(combo_ids) == 9, f"预期 9 个 combo_id,实际 {len(combo_ids)}: {combo_ids}"
    assert PRODUCTION_COMBO in combo_ids

    rows = []
    for combo_id in combo_ids:
        sweep_combo = sweep[sweep["combo_id"] == combo_id]
        assert len(sweep_combo) == 81, f"combo={combo_id}: 应有 81 行,实际 {len(sweep_combo)}"
        rows.append(compute_combo_row(df_master, sweep_combo, combo_id))

    out = pd.DataFrame(rows)
    if not out["type_agreement"].all():
        bad = out.loc[~out["type_agreement"], "combo_id"].tolist()
        raise RuntimeError(f"Type I/II/III SS 不一致的 combo: {bad} —— 停下来报告,"
                          "不要继续套用平衡设计假设。")

    # 相对 default 组合的最大绝对偏差(逐行算,每行都存,不只存汇总)
    default_row = out.loc[out["combo_id"] == PRODUCTION_COMBO].iloc[0]
    for col in ("precursor_eta2_pct", "interaction_total_eta2_pct",
                "D_sec_S_lt_M_count", "D_sec_S_lt_L_count"):
        out[f"abs_dev_from_default_{col}"] = (out[col] - default_row[col]).abs()

    out = out.drop(columns=["type_agreement"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    # ---- 判定规则(预设,原样应用,不软化/不挑favorable子集) ----
    max_precursor = out["precursor_eta2_pct"].max()
    min_precursor = out["precursor_eta2_pct"].min()
    same_order_of_magnitude = int(np.floor(np.log10(max_precursor))) == int(np.floor(np.log10(min_precursor)))

    max_interaction = out["interaction_total_eta2_pct"].max()
    interaction_violations = out.loc[out["interaction_total_eta2_pct"] >= INTERACTION_MAX_PCT]

    sm_dev_max = out["abs_dev_from_default_D_sec_S_lt_M_count"].max()
    sl_dev_max = out["abs_dev_from_default_D_sec_S_lt_L_count"].max()
    sm_violations = out.loc[out["abs_dev_from_default_D_sec_S_lt_M_count"] > ORDERING_TOLERANCE]
    sl_violations = out.loc[out["abs_dev_from_default_D_sec_S_lt_L_count"] > ORDERING_TOLERANCE]

    all_pass = (same_order_of_magnitude and interaction_violations.empty
                and sm_violations.empty and sl_violations.empty)

    write_report(out, default_row, same_order_of_magnitude, max_precursor, min_precursor,
                max_interaction, interaction_violations, sm_dev_max, sl_dev_max,
                sm_violations, sl_violations, all_pass)

    print(f"[done] {OUT_CSV}")
    print(f"[done] {OUT_REPORT}")
    print(out.to_string(index=False))
    print(f"\n判定:{'稳健(robust)' if all_pass else '不稳健——存在超阈值组合,见报告'}")


def write_report(out, default_row, same_order_of_magnitude, max_precursor, min_precursor,
                 max_interaction, interaction_violations, sm_dev_max, sl_dev_max,
                 sm_violations, sl_violations, all_pass):
    lines = []
    lines.append("# D_sec 头号结论对 SEM 500x 分割参数选择的稳健性审计\n")
    lines.append(
        "生成脚本:`scripts/37_dsec_segmentation_robustness.py`;数据源:"
        f"`{SWEEP_CSV}`(9 个 combo_id x 81 样本,`scripts/02j_mdagg_fusion_check.py` "
        f"已产出)+ `{MASTER_TABLE}`(precursor/T_C/beta/t_hold/condition_id,"
        "这些列不随分割参数变化)。ANOVA 引擎直接复用生产代码 "
        "`nfm.stats.anova_interactions.run_anova_for_target`(81 样本完整平衡析因"
        "设计,Type I≡II≡III),不重新实现。\n"
    )
    lines.append(
        "> 背景:此前的 9 组合敏感性扫描只检验了 `Spearman(M_D_agg, Theta)` "
        "是否在 9 个组合下保持显著(9/9 通过,见 `scripts/02j_mdagg_fusion_check.py` "
        "产出),**没有**检验 D_sec 自身的两条论文头号数字"
        "(`table1.D_sec.precursor_eta2_pct=88.8%`、"
        "`table2.D_sec.total_coupling_eta2_pct=0.63%`)和两条排序论断"
        "(`D_sec(S)<D_sec(M)` 27/27、`D_sec(S)<D_sec(L)` 27/27,来自 "
        "`data/interim/ordering_consistency.csv`)是否对同一批分割参数扰动稳健。"
        "本审计补齐这一空白。\n"
    )

    lines.append("## 1. 方法\n")
    lines.append(
        "9 个 combo_id 各自的逐样本 D_sec(729 行来自 `mdagg_sensitivity_sweep.csv`,"
        "按 `sample_id` join)替换进 `master_table.csv` 拷贝的 D_sec 列"
        "(precursor/T_C/beta/t_hold/condition_id 原样保留,这些列不随分割参数"
        "变化),对每份 81 样本的替换后表跑 "
        "`run_anova_for_target(df, \"D_sec\")`,取:\n"
        "- `precursor` 主效应 η²(表1口径)\n"
        "- `precursor:T_C + precursor:beta + precursor:t_hold` 三项 η² 之和(表2口径)\n"
        "- 27 个 `condition_id` 三元组里 `D_sec(S)<D_sec(M)`、`D_sec(S)<D_sec(L)` "
        "各自成立的次数(逐 combo 从该 combo 的 D_sec 透视表直接计数,不复用 "
        "production-only 的 `ordering_consistency.csv`)\n\n"
        "9 个 combo:`default`(生产设置,percentile=90/slope=0.082)、"
        "percentile 在 {75,80,85,95} 扰动(slope 固定 0.082)、"
        "slope 在 {0.033,0.057,0.107,0.131} 扰动(percentile 固定 90)。\n"
    )

    lines.append("## 2. 9 组合对照表\n")
    lines.append(
        "| combo_id | precursor_eta2_pct | interaction_total_eta2_pct | "
        "D_sec_S_lt_M_count | D_sec_S_lt_L_count | "
        "abs_dev(precursor_eta2) | abs_dev(interaction_total) | "
        "abs_dev(S_lt_M) | abs_dev(S_lt_L) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in out.iterrows():
        marker = " **(生产默认)**" if r["combo_id"] == "default" else ""
        lines.append(
            f"| {r['combo_id']}{marker} | {r['precursor_eta2_pct']:.2f} | "
            f"{r['interaction_total_eta2_pct']:.2f} | "
            f"{int(r['D_sec_S_lt_M_count'])}/27 | {int(r['D_sec_S_lt_L_count'])}/27 | "
            f"{r['abs_dev_from_default_precursor_eta2_pct']:.2f} | "
            f"{r['abs_dev_from_default_interaction_total_eta2_pct']:.2f} | "
            f"{int(r['abs_dev_from_default_D_sec_S_lt_M_count'])} | "
            f"{int(r['abs_dev_from_default_D_sec_S_lt_L_count'])} |"
        )
    lines.append("")
    lines.append(
        f"生产默认(`default`)组合:precursor_eta2_pct={default_row['precursor_eta2_pct']:.2f}%,"
        f"interaction_total_eta2_pct={default_row['interaction_total_eta2_pct']:.2f}%,"
        f"S_lt_M={int(default_row['D_sec_S_lt_M_count'])}/27,"
        f"S_lt_L={int(default_row['D_sec_S_lt_L_count'])}/27"
        "(与论文表1/表2引用值、`ordering_consistency.csv` 生产值应一致,"
        "作为本审计内部一致性的交叉核验)。\n"
    )

    lines.append("## 3. 判定规则(预设,原样应用)\n")
    lines.append(
        "若同时满足下列三条,判定两条头号结论(precursor 主效应占主导；交互总量落在 "
        "0.63–3.20% 区间)对分割参数选择稳健,可直接引用:\n\n"
        "1. 9 个 combo 的 `precursor_eta2_pct` 同数量级(以 10 为底数量级判定);\n"
        "2. 9 个 combo 的 `interaction_total_eta2_pct` 全程 < 3.20%;\n"
        "3. 两个排序计数(`D_sec_S_lt_M_count`/`D_sec_S_lt_L_count`)相对 `default` "
        "组合的偏差全程在 ±2 以内。\n\n"
        "若任意一个 combo 违反以上任意一条,如实指出具体是哪个/哪些 combo、"
        "超出多少,不回避、不挑选有利子集。\n"
    )

    lines.append("## 4. 逐条判定结果\n")
    lines.append(
        f"**规则1(数量级)**:9 个 combo 的 `precursor_eta2_pct` 取值范围 "
        f"[{min_precursor:.2f}%, {max_precursor:.2f}%],"
        f"{'均为同一数量级(10¹,%)——**通过**。' if same_order_of_magnitude else '**跨数量级——不通过**。'}\n"
    )
    if interaction_violations.empty:
        lines.append(
            f"**规则2(交互总量<3.20%)**:9 个 combo 的 `interaction_total_eta2_pct` "
            f"最大值 {max_interaction:.2f}%(生产默认 {default_row['interaction_total_eta2_pct']:.2f}%),"
            "全程 < 3.20% —— **通过**。\n"
        )
    else:
        bad_list = "; ".join(
            f"{r['combo_id']}={r['interaction_total_eta2_pct']:.2f}%"
            for _, r in interaction_violations.iterrows()
        )
        lines.append(
            f"**规则2(交互总量<3.20%)**:**不通过**。以下 {len(interaction_violations)} "
            f"个 combo 的 `interaction_total_eta2_pct` ≥ 3.20%:{bad_list}。\n"
        )
    if sm_violations.empty and sl_violations.empty:
        lines.append(
            f"**规则3(排序计数偏差≤±2)**:`D_sec_S_lt_M_count` 相对 default 的最大绝对"
            f"偏差 {int(sm_dev_max)},`D_sec_S_lt_L_count` 最大绝对偏差 {int(sl_dev_max)},"
            "均 ≤ 2 —— **通过**。\n"
        )
    else:
        bad_parts = []
        if not sm_violations.empty:
            bad_parts.append("S_lt_M: " + "; ".join(
                f"{r['combo_id']}(计数{int(r['D_sec_S_lt_M_count'])}/27,偏差"
                f"{int(r['abs_dev_from_default_D_sec_S_lt_M_count'])})"
                for _, r in sm_violations.iterrows()))
        if not sl_violations.empty:
            bad_parts.append("S_lt_L: " + "; ".join(
                f"{r['combo_id']}(计数{int(r['D_sec_S_lt_L_count'])}/27,偏差"
                f"{int(r['abs_dev_from_default_D_sec_S_lt_L_count'])})"
                for _, r in sl_violations.iterrows()))
        lines.append(
            f"**规则3(排序计数偏差≤±2)**:**不通过**。" + "；".join(bad_parts) + "。\n"
        )

    lines.append("## 5. 结论\n")
    if all_pass:
        lines.append(
            "**三条规则全部通过 —— 判定 D_sec 的两条头号结论"
            "(前驱体主效应占主导;前驱体×工艺交互总量落在 0.63–3.20% 区间)"
            "对 SEM 500x 分割参数(percentile/slope)在生产默认值附近的选择稳健,"
            "可按现状引用,不需要因分割参数不确定性而降级或加限制声明。**\n"
        )
    else:
        lines.append(
            "**未能全部通过 —— 如实报告,不作softening、不挑favorable子集**:"
            "详见 §4 逐条判定结果中标注为“不通过”的规则与具体 combo。"
            "这意味着 D_sec 的相应头号结论在给定的分割参数扰动范围内"
            "不能不加限制地判定为稳健,论文引用前应结合 §4 的具体超阈值情况"
            "补充限制声明或做进一步的分割参数敏感性说明。**\n"
        )

    lines.append("## 6. 限制声明\n")
    lines.append(
        "- 本审计只复用已有的 9 组合分割参数扫描(`mdagg_sensitivity_sweep.csv`),"
        "未重新做任何图像分割计算,不引入新的分割结果。\n"
        "- 9 个 combo 均为 percentile/slope 单变量扰动(另一参数固定于生产默认),"
        "不是两参数联合网格扫描,不能外推到 percentile 和 slope 同时偏离默认值的组合。\n"
        "- `run_anova_for_target` 内部的置换检验(5000 次,seed=2000,与生产脚本"
        "`scripts/12_anova_interactions.py` 的 seed 不同)不影响本审计关心的 η² 点估计,"
        "仅影响未在本报告中使用的 perm_p 列。\n"
        "- 本脚本不修改 `src/nfm/`、`scripts/12_anova_interactions.py`、"
        "`scripts/14_ordering_consistency.py`、`data/processed/master_table.csv`、"
        "`data/interim/mdagg_sensitivity_sweep.csv`、`data/interim/ordering_consistency.csv`"
        " 下任何文件,只读取、只写自己的输出。\n"
    )

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
