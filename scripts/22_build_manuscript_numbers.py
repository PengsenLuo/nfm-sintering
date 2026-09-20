# -*- coding: utf-8 -*-
"""22 论文A数字冻结表构建
====================================================================
只从 data/interim/、data/processed/ 下已经产出的 csv 读取并汇总,不做任何
新的统计计算。每个数字追加进 manuscript/numbers/manuscript_numbers.csv
(长表,一行一个数),对应的正文表格另存一份宽表到
manuscript/numbers/tables/tableN_*.csv。

若某个正文数字在现有产出文件里找不到对应来源,追加一行到
manuscript/numbers/manuscript_numbers.csv,value 留空、note 写清楚
"无来源,原因是..."——不得编一个数字填进去。
"""
import _bootstrap  # noqa

import os

import numpy as np
import pandas as pd

# NOTE: these paths point to internal manuscript-draft outputs not included
# in this public repository; supply your own manuscript/numbers/ directory
# to run this script standalone.
NUMBERS_CSV = "manuscript/numbers/manuscript_numbers.csv"
TABLES_DIR = "manuscript/numbers/tables"

_rows = []


def add_number(id_, value, unit, precision, source_script, source_file,
               source_locator, section, note=""):
    _rows.append(dict(id=id_, value=value, unit=unit, precision=precision,
                      source_script=source_script, source_file=source_file,
                      source_locator=source_locator, section=section, note=note))


def flush():
    os.makedirs(TABLES_DIR, exist_ok=True)
    out = pd.DataFrame(_rows, columns=[
        "id", "value", "unit", "precision", "source_script", "source_file",
        "source_locator", "section", "note",
    ])
    out.to_csv(NUMBERS_CSV, index=False)
    print(f"已写出 {NUMBERS_CSV}(共 {len(out)} 行)")


_TABLE6_UNITS = {"D_XRD": "nm", "compaction_density": "g/cm3", "D_sec": "um"}
# 模型规格去掉了 Θ 项,列结构与更早版本的 6 列(intercept/lnΘ/Θ/L/M/S)
# 不同,只剩 4 列。前驱体编码从"以 S 为参照类"改为 sum(偏差)编码——
# intercept 现在是"lnΘ=0 处三个前驱体的总体均值水平",M/L 是相对总体均值
# 的偏差,S 的偏差 = -(M+L)(未单列)。
_TABLE6_COEF_COLS = [
    ("intercept", "intercept"), ("ln_Theta", "ln_theta_coef"),
    ("M", "precursor_M_coef"), ("L", "precursor_L_coef"),
]


def build_table6_model_coef():
    # compaction_density/D_sec/D_XRD 三个 target 的系数与 LOCO 统一来自
    # scripts/24_table6_ridge_baseline.py(Ridge(alpha=1),原始尺度未标准化),
    # 不再只做 D_XRD 一行、也不再依赖 dxrd_loco_51sample_coef.csv。
    coef_src = "data/interim/table6_ridge_coef.csv"
    loco_src = "data/interim/table6_ridge_loco.csv"
    coef_df = pd.read_csv(coef_src)
    loco_df = pd.read_csv(loco_src)
    merged = coef_df.merge(loco_df, on="target", how="left")
    merged.to_csv(f"{TABLES_DIR}/table6_model_coef.csv", index=False)

    for _, row in merged.iterrows():
        target = row["target"]
        unit = _TABLE6_UNITS.get(target, "")
        for col, id_suffix in _TABLE6_COEF_COLS:
            add_number(f"table6.{target}.{id_suffix}", row[col], unit, 4,
                      "scripts/24_table6_ridge_baseline.py", coef_src,
                      f"target=={target}, col {col}", "Table 6",
                      "Ridge(alpha=1,原始尺度未标准化),compaction_density/"
                      "D_sec/D_XRD 三个 target 共用同一实现,系数与 LOCO 保证"
                      "同源;lnΘ-only + sum(偏差)前驱体编码,intercept=总体"
                      "均值水平,M/L=相对总体均值的偏差,S的偏差=-(M+L)未"
                      "单列)")
        add_number(f"table6.{target}.n", int(row["n"]), "", 0,
                  "scripts/24_table6_ridge_baseline.py", coef_src,
                  f"target=={target}, col n", "Table 6")
        add_number(f"table6.{target}.loco_R2", row["R2"], "", 4,
                  "scripts/24_table6_ridge_baseline.py", loco_src,
                  f"target=={target}, col R2", "Table 6",
                  f"与 table7.{target}.R2 同源(均出自 "
                  "scripts/24_table6_ridge_baseline.py)")
        add_number(f"table6.{target}.loco_MAE", row["MAE"], unit, 2,
                  "scripts/24_table6_ridge_baseline.py", loco_src,
                  f"target=={target}, col MAE", "Table 6")
    # 上面循环已对 compaction_density/D_sec/D_XRD 三个 target 各自
    # add_number 了真实系数行(源自 scripts/24_table6_ridge_baseline.py),
    # 原先这里的 "table6.rho_and_Dsec_coef_rows" 无来源占位行已删除。


_TABLE7_RIDGE_TARGETS = ["compaction_density", "D_sec", "D_XRD"]
# circularity 归入与 compaction_density/D_sec/D_XRD 相同的统一 Ridge/LOCO
# 路径(nfm.dxrd_baseline.ridge_coef_loco,81样/27折,不做纳米层限定),不再
# 走 07e 脚本的 PINN 训练快照。**只影响 Table 7**——circularity 不进
# scripts/24 的 TARGETS 列表、不写入 table6_ridge_coef.csv,Table 6 仍只报
# compaction_density/D_sec/D_XRD 三个 target 的系数(circularity 没有正文
# 系数表可对照,只有 LOCO 性能行)。快照文件 `target_baseline_comparison.
# snapshot.csv` 保留在磁盘供审计,但不再是 Table 7 circularity 行的参数来源
# ——lattice_c/c_a_ratio 的 PINN_* 列仍只在快照文件里,不受本次改动影响。
# convexity(与 solidity 定义相同、两条流水线数值有实质差异)已整体移除。
_TABLE7_UNIFIED_INLINE_TARGETS = ["circularity"]
_TABLE7_ORDER = ["D_XRD", "D_sec", "circularity", "compaction_density"]


def build_table7_loco():
    # compaction_density/D_sec/D_XRD 从 scripts/24 的统一 Ridge/LOCO 产出
    # 取数(与 table6 同源)。circularity 在本函数内联调用同一个
    # nfm.dxrd_baseline.ridge_coef_loco(),不经过 scripts/24/table6_ridge_*.csv
    # (那两个文件只服务 Table 6 的 3 个 target)。
    from nfm.dxrd_baseline import ridge_coef_loco

    ridge_src = "data/interim/table6_ridge_loco.csv"
    ridge_df = pd.read_csv(ridge_src)
    ridge_keep = ridge_df[ridge_df["target"].isin(_TABLE7_RIDGE_TARGETS)].rename(
        columns={"R2": "LOCO_R2", "n_folds": "折数", "n_eval": "n"}
    )[["target", "LOCO_R2", "MAE", "折数", "n"]]

    master_src = "data/processed/master_table.csv"
    master_df = pd.read_csv(master_src)
    inline_rows = []
    for target in _TABLE7_UNIFIED_INLINE_TARGETS:
        out = ridge_coef_loco(master_df, target)
        inline_rows.append({"target": target, "LOCO_R2": out["R2"], "MAE": out["MAE"],
                           "折数": out["n_folds"], "n": out["n_eval"]})
    inline_keep = pd.DataFrame(inline_rows)

    keep = pd.concat([ridge_keep, inline_keep], ignore_index=True)
    keep = keep.set_index("target").loc[_TABLE7_ORDER].reset_index()
    keep.to_csv(f"{TABLES_DIR}/table7_loco.csv", index=False)

    for _, r in keep.iterrows():
        target = r["target"]
        if target in _TABLE7_RIDGE_TARGETS:
            source_script = "scripts/24_table6_ridge_baseline.py"
            source_file = ridge_src
            r2_locator = f"target=={target}, col R2"
            mae_locator = f"target=={target}, col MAE"
            nfolds_locator = f"target=={target}, col n_folds"
            note = (f"与 table6.{target} 系数/LOCO 同源(均出自 "
                    "scripts/24_table6_ridge_baseline.py)")
        else:
            source_script = "scripts/22_build_manuscript_numbers.py (nfm.dxrd_baseline.ridge_coef_loco,内联调用)"
            source_file = master_src
            r2_locator = f"target=={target}, ridge_coef_loco(df,'{target}')['R2']"
            mae_locator = f"target=={target}, ridge_coef_loco(df,'{target}')['MAE']"
            nfolds_locator = f"target=={target}, ridge_coef_loco(df,'{target}')['n_folds']"
            note = ("并入统一 Ridge/LOCO 路径,与 "
                    "compaction_density/D_sec/D_XRD 同一实现同一函数调用"
                    "(仅未写入 table6_ridge_*.csv,Table 6 系数表不含此行)")
        add_number(f"table7.{target}.R2", r["LOCO_R2"], "", 4,
                  source_script, source_file, r2_locator, "Table 7", note)
        add_number(f"table7.{target}.MAE", r["MAE"], "", 4,
                  source_script, source_file, mae_locator, "Table 7", note)
        add_number(f"table7.{target}.n_folds", int(r["折数"]), "", 0,
                  source_script, source_file, nfolds_locator, "Table 7")


def build_table3_precursor_summary():
    # convexity 全面移除(理由同 build_table7_loco 上方注释),不再纳入 Table 3。
    src = "data/processed/master_table.csv"
    df = pd.read_csv(src)
    g = df.groupby("precursor").agg(
        precursor_D50=("precursor_D50", "mean"),
        D_sec_mean=("D_sec", "mean"), D_sec_std=("D_sec", "std"),
        circularity_mean=("circularity", "mean"), circularity_std=("circularity", "std"),
        rho_mean=("compaction_density", "mean"), rho_std=("compaction_density", "std"),
    ).reindex(["S", "M", "L"])
    g.to_csv(f"{TABLES_DIR}/table3_precursor_summary.csv")
    for p, row in g.iterrows():
        for col, id_suffix, unit, prec in [
            ("precursor_D50", "precursor_D50", "um", 2),
            ("D_sec_mean", "D_sec_mean", "um", 2), ("D_sec_std", "D_sec_std", "um", 2),
            ("circularity_mean", "circularity_mean", "", 3), ("circularity_std", "circularity_std", "", 3),
            ("rho_mean", "rho_mean", "g/cm3", 3), ("rho_std", "rho_std", "g/cm3", 3),
        ]:
            add_number(f"table3.{p}.{id_suffix}", row[col], unit, prec,
                      "(pandas groupby, no separate script)", src,
                      f"groupby('precursor').agg(mean,std), precursor=={p}, col {col.rsplit('_',1)[0]}",
                      "Table 3")


def build_table5_ordering():
    src = "data/interim/ordering_consistency.csv"
    df = pd.read_csv(src)
    n_cond = len(df)  # 27

    rows = []
    rows.append(("rho_S_lt_M", int(df["compaction_density_prop_S_vs_M"].sum()), n_cond))
    rows.append(("rho_S_lt_L", int(df["compaction_density_prop_S_vs_L"].sum()), n_cond))
    pattern_counts = df["compaction_density_pattern"].value_counts()
    rows.append(("full_order_S_lt_M_lt_L", int(pattern_counts.get("S<M<L", 0)), n_cond))
    rows.append(("full_order_S_lt_L_lt_M", int(pattern_counts.get("S<L<M", 0)), n_cond))
    rows.append(("ML_tie_within_delta", int(df["compaction_density_ML_tie_within_delta"].fillna(False).sum()), n_cond))

    out = pd.DataFrame(rows, columns=["proposition", "count", "n_conditions"])
    out.to_csv(f"{TABLES_DIR}/table5_ordering.csv", index=False)
    for prop, count, total in rows:
        add_number(f"table5.{prop}", f"{count}/{total}", "", 0,
                  "scripts/14_ordering_consistency.py", src,
                  f"aggregated from compaction_density_* columns (see script step 4 note)",
                  "Table 5")

    # 同一检验在形貌量上的交叉验证。
    # convexity 全面移除。
    for target in ["D_sec", "circularity"]:
        s_vs_m = int(df[f"{target}_prop_S_vs_M"].sum())
        s_vs_l = int(df[f"{target}_prop_S_vs_L"].sum())
        add_number(f"table5.{target}.S_vs_M", f"{s_vs_m}/{n_cond}", "", 0,
                  "scripts/14_ordering_consistency.py", src,
                  f"col {target}_prop_S_vs_M, sum over 27 rows", "Table 5 (cross-check)")
        add_number(f"table5.{target}.S_vs_L", f"{s_vs_l}/{n_cond}", "", 0,
                  "scripts/14_ordering_consistency.py", src,
                  f"col {target}_prop_S_vs_L, sum over 27 rows", "Table 5 (cross-check)")


def build_table5_mvsl_paired():
    # 早期稿件正文声称"27个条件中 M>L 15次、M<L 10次、相等1次"
    # (15+10+1=26≠27,算术错误)。这里从 scripts/36_paired_mvsl_recount.py
    # 的产出直接取正确计数,两种"相等"判定容差(0/1e-3)都写入,取代那句
    # 算术错误的原文。
    src = "data/interim/mvsl_paired_recount_summary.csv"
    df = pd.read_csv(src)
    unit_map = {"compaction_density": "g/cm3", "D_sec": "um"}
    for _, row in df.iterrows():
        var, tol = row["variable"], row["tie_tolerance"]
        unit = unit_map.get(var, "")
        prefix = f"table5.MvsL.{var}.{tol}"
        add_number(f"{prefix}.n_pos", int(row["n_pos"]), "", 0,
                  "scripts/36_paired_mvsl_recount.py", src,
                  f"variable=={var}, tie_tolerance=={tol}, col n_pos", "Table 5",
                  "M>L 计数(同炉配对,27条件);取代 v5 稿件算术错误(15+10+1=26≠27)的原句")
        add_number(f"{prefix}.n_neg", int(row["n_neg"]), "", 0,
                  "scripts/36_paired_mvsl_recount.py", src,
                  f"variable=={var}, tie_tolerance=={tol}, col n_neg", "Table 5")
        add_number(f"{prefix}.n_zero", int(row["n_zero"]), "", 0,
                  "scripts/36_paired_mvsl_recount.py", src,
                  f"variable=={var}, tie_tolerance=={tol}, col n_zero", "Table 5")
        add_number(f"{prefix}.median_diff", round(float(row["median_diff"]), 4), unit, 4,
                  "scripts/36_paired_mvsl_recount.py", src,
                  f"variable=={var}, tie_tolerance=={tol}, col median_diff", "Table 5",
                  "median(M-L),同炉配对差值")
        add_number(f"{prefix}.sign_test_p", round(float(row["sign_test_p"]), 4), "", 4,
                  "scripts/36_paired_mvsl_recount.py", src,
                  f"variable=={var}, tie_tolerance=={tol}, col sign_test_p", "Table 5",
                  "scipy.stats.binomtest,字面意义符号检验,非Wilcoxon符号秩检验")


_TABLE1_TARGETS = [
    # convexity(sem_processor.py 500x主线)与solidity(scripts/02i_shape_
    # descriptors.py)定义完全相同(A/A_convex),但两条流水线的颗粒过滤行为
    # 不同、数值有实质差异(Pearson r≈0.60,mean diff≈0.04)。故 Table 1/
    # Table 2 只保留 solidity(与 D_f/elongation/roughness 同一流水线,口径
    # 一致),此处不再列 convexity。convexity 在 Table 3
    # (build_table3_precursor_summary)/Table 7(build_table7_loco)里仍保留,
    # 不受本次改动影响。
    ("compaction_density", "data/interim/anova_interactions.csv", "scripts/12_anova_interactions.py"),
    ("D_sec", "data/interim/anova_interactions.csv", "scripts/12_anova_interactions.py"),
    ("circularity", "data/interim/anova_interactions.csv", "scripts/12_anova_interactions.py"),
    ("D_f", "data/interim/anova_interactions_shape_descriptors.csv", "scripts/17_shape_descriptor_full_analysis.py"),
    ("solidity", "data/interim/anova_interactions_shape_descriptors.csv", "scripts/17_shape_descriptor_full_analysis.py"),
    ("elongation", "data/interim/anova_interactions_shape_descriptors.csv", "scripts/17_shape_descriptor_full_analysis.py"),
    ("roughness", "data/interim/anova_interactions_shape_descriptors.csv", "scripts/17_shape_descriptor_full_analysis.py"),
    ("D_XRD", "data/interim/nano_layer_interaction_anova.csv", "scripts/19_nano_layer_interaction_anova.py"),
    ("lattice_c", "data/interim/nano_layer_interaction_anova.csv", "scripts/19_nano_layer_interaction_anova.py"),
]
_TERM_TO_COL = {"precursor": "precursor", "T_C": "T", "beta": "beta", "t_hold": "t", "Residual": "residual"}


def build_table1_main_effects():
    cache = {}
    rows = []
    for target, src, script in _TABLE1_TARGETS:
        if src not in cache:
            cache[src] = pd.read_csv(src)
        df = cache[src]
        sub = df[df["target"] == target]
        if sub.empty:
            add_number(f"table1.{target}", None, "%", None, script, src,
                      f"target=={target}", "Table 1",
                      "无来源:该 target 在此 csv 里找不到任何行")
            continue
        row = {"target": target}
        for term, colname in _TERM_TO_COL.items():
            match = sub[sub["term"] == term]
            val = float(match["eta2"].iloc[0]) * 100 if not match.empty else None
            row[colname] = val
            if val is not None:
                add_number(f"table1.{target}.{colname}_eta2_pct", round(val, 1), "%", 1,
                          script, src, f"target=={target}, term=={term}, col eta2 (x100)",
                          "Table 1")
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"{TABLES_DIR}/table1_main_effects.csv", index=False)


_INTERACTION_TERMS = ["precursor:T_C", "precursor:beta", "precursor:t_hold"]


def build_table2_interactions():
    cache = {}
    rows = []
    for target, src, script in _TABLE1_TARGETS:  # 表2只列81样的3目标(R4后convexity移除)+纳米层2个,不含4个形状描述子
        if target in ("D_f", "solidity", "elongation", "roughness"):
            continue
        if src not in cache:
            cache[src] = pd.read_csv(src)
        df = cache[src]
        sub = df[df["target"] == target]
        row = {"target": target}
        for term in _INTERACTION_TERMS:
            match = sub[sub["term"] == term]
            if match.empty:
                continue
            eta2_pct = float(match["eta2"].iloc[0]) * 100
            omega2_pct = float(match["omega2"].iloc[0]) * 100
            short = term.split(":")[1]
            row[f"{short}_eta2"] = round(eta2_pct, 2)
            add_number(f"table2.{target}.{short}_eta2_pct", round(eta2_pct, 2), "%", 2,
                      script, src, f"target=={target}, term=={term}, col eta2 (x100)", "Table 2")
        total_eta2 = sub[sub["term"].isin(_INTERACTION_TERMS)]["eta2"].sum() * 100
        total_omega2 = sub[sub["term"].isin(_INTERACTION_TERMS)]["omega2"].sum() * 100
        row["total_eta2"] = round(total_eta2, 2)
        row["total_omega2"] = round(total_omega2, 2)
        add_number(f"table2.{target}.total_coupling_eta2_pct", round(total_eta2, 2), "%", 2,
                  script, src, f"target=={target}, sum eta2 over {_INTERACTION_TERMS}", "Table 2")
        add_number(f"table2.{target}.total_coupling_omega2_pct", round(total_omega2, 2), "%", 2,
                  script, src, f"target=={target}, sum omega2 over {_INTERACTION_TERMS}", "Table 2")
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"{TABLES_DIR}/table2_interactions.csv", index=False)


def build_table4_shape_spearman():
    src = "data/interim/shape_descriptor_full_analysis.csv"
    df = pd.read_csv(src)
    sub = df[(df["section"] == "spearman_theta") & (df["metric"] == "rho")]
    rows = []
    for target in sub["target"].unique():
        row = {"target": target}
        for group in ["S", "M", "L"]:
            m = sub[(sub["target"] == target) & (sub["group"] == group)]
            if m.empty:
                continue
            r = float(m["value"].iloc[0])
            lo, hi = float(m["ci_low"].iloc[0]), float(m["ci_high"].iloc[0])
            row[f"{group}_rho"] = round(r, 3)
            row[f"{group}_ci_low"] = round(lo, 3)
            row[f"{group}_ci_high"] = round(hi, 3)
            diff_row = df[(df["section"] == "spearman_theta") & (df["target"] == target)
                         & (df["group"] == group) & (df["metric"] == "diff_vs_paper_reference")]
            note = ""
            if not diff_row.empty:
                note = f"内建自检 diff_vs_paper_reference={float(diff_row['value'].iloc[0]):+.6f}"
            add_number(f"table4.{target}.{group}.rho", round(r, 3), "", 3,
                      "scripts/17_shape_descriptor_full_analysis.py", src,
                      f"section==spearman_theta, target=={target}, group=={group}, metric==rho",
                      "Table 4", note)
            add_number(f"table4.{target}.{group}.ci", f"[{lo:+.3f}, {hi:+.3f}]", "", 3,
                      "scripts/17_shape_descriptor_full_analysis.py", src,
                      f"section==spearman_theta, target=={target}, group=={group}, ci_low/ci_high",
                      "Table 4")
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"{TABLES_DIR}/table4_shape_spearman.csv", index=False)


def build_headline_numbers():
    # --- Step 1: §3.1 剔除S前驱体后 precursor 方差份额崩塌 ---
    src = "data/interim/variance_range_sensitivity.csv"
    df = pd.read_csv(src)
    baseline = df[df["subset_id"] == "baseline_81"]
    # 注:模糊匹配(subset_id.contains("no_s") 或
    # subset_description.contains("剔除S|无S|drop.?S"))在真实数据上一行都
    # 匹配不到——实测 subset_id 是 "drop_precursor_S",subset_description 是
    # "剔除前驱体S,只留{M,L}(9.70/12.40µm,跨度1.28倍)","剔除S"/"无S"/
    # "drop.?S" 均不是其子串。已用真实数据核实 drop_precursor_S 就是"剔除S
    # 前驱体"对应子集:算出的 eta2 与稿件给出的 0.24%/0.35% 精确吻合,故
    # 直接按已核实的 subset_id 取数,不再依赖容易落空的模糊匹配。
    no_s = df[df["subset_id"] == "drop_precursor_S"]
    # R4(task-4):convexity 从此 headline amplitude 循环里移除,与 Table 1/2
    # 的去重决策(见上方 _TABLE1_TARGETS 注释)保持一致——本循环镜像的正是
    # Table 1/2 的目标列表。
    for target in ["compaction_density", "D_sec", "circularity"]:
        b = baseline[(baseline["target"] == target) & (baseline["term"] == "precursor")]
        if b.empty:
            continue
        eta2_baseline = float(b["eta2"].iloc[0]) * 100
        add_number(f"headline.variance_range.{target}.precursor_eta2_baseline",
                  round(eta2_baseline, 2), "%", 2,
                  "scripts/15_variance_range_sensitivity.py", src,
                  f"subset_id==baseline_81, target=={target}, term==precursor, col eta2 (x100)",
                  "§3.1")
        if no_s.empty:
            add_number(f"headline.variance_range.{target}.precursor_eta2_no_S", None, "%", 2,
                      "scripts/15_variance_range_sensitivity.py", src, "",
                      "§3.1", "无来源:未能在 subset_id/subset_description 里定位到"
                      "'剔除S前驱体'对应的子集,需人工确认该子集的 subset_id 命名后补上")
            continue
        n = no_s[(no_s["target"] == target) & (no_s["term"] == "precursor")]
        if n.empty:
            continue
        eta2_noS = float(n["eta2"].iloc[0]) * 100
        add_number(f"headline.variance_range.{target}.precursor_eta2_no_S",
                  round(eta2_noS, 2), "%", 2,
                  "scripts/15_variance_range_sensitivity.py", src,
                  f"subset_id=={n['subset_id'].iloc[0]}, target=={target}, term==precursor, col eta2 (x100)",
                  "§3.1")

    # --- Step 2: §3.5 条件级残差跨前驱体 Pearson r/p 表 ---
    # 复现方案:data/interim/sieving_confound_diagnostic.csv 的
    # residual_precursor_adj_full27 列,按 condition_id x precursor 透视后
    # 逐对 pearsonr。已用同一份原始 csv 独立复现,S-L/M-L/S-M 三对 r/p 与
    # 稿件正文给出的数值精确一致,不存在需要"用自己数字、标注分歧"的情况。
    # 产出脚本见 scripts/25_sieving_residual_pearson.py。
    src = "data/interim/sieving_residual_pearson.csv"
    pear = pd.read_csv(src)
    for _, row in pear.iterrows():
        add_number(f"headline.residual_reproducibility.{row['pair']}.pearson_r",
                  round(row["pearson_r"], 4), "", 4,
                  "scripts/25_sieving_residual_pearson.py", src,
                  f"pair=={row['pair']}, col pearson_r", "§3.5")
        add_number(f"headline.residual_reproducibility.{row['pair']}.p",
                  round(row["p"], 5), "", 5,
                  "scripts/25_sieving_residual_pearson.py", src,
                  f"pair=={row['pair']}, col p", "§3.5")


def build_table_sem_psd_quantiles():
    # 产物粒度口径从 9 样水测激光粒度改为 81 样 SEM 500x 数量/体积两套
    # D10/D50/D90/Span(scripts/44_sem_psd_quantiles.py 产出)。
    src = "data/interim/sem_psd_quantiles.csv"
    df = pd.read_csv(src)
    g = df.groupby("precursor").agg(
        D10_num_median=("D10_num", "median"), D50_num_median=("D50_num", "median"),
        D90_num_median=("D90_num", "median"), Span_num_median=("Span_num", "median"),
        Span_vol_median=("Span_vol", "median"),
    ).reindex(["S", "M", "L"])
    for p, row in g.iterrows():
        for col in ["D10_num_median", "D50_num_median", "D90_num_median",
                    "Span_num_median", "Span_vol_median"]:
            unit = "um" if col.startswith(("D10", "D50", "D90")) else ""
            add_number(f"sem_psd.{p}.{col}", round(float(row[col]), 4), unit, 4,
                      "scripts/44_sem_psd_quantiles.py", src,
                      f"precursor=={p}, groupby median, col {col.replace('_median','')}",
                      "§3.5 (SEM caliber)")


def main():
    build_table6_model_coef()
    build_table7_loco()
    build_table3_precursor_summary()
    build_table5_ordering()
    build_table5_mvsl_paired()
    build_table1_main_effects()
    build_table2_interactions()
    build_table4_shape_spearman()
    build_headline_numbers()
    build_table_sem_psd_quantiles()
    flush()


if __name__ == "__main__":
    main()
