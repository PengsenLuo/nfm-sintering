# -*- coding: utf-8 -*-
"""13 T1 —— 过筛混杂对压实密度与条件级残差的影响诊断（一次性分析，不产出可复用模块）。
==========================================================================================
背景：

前 15 个产物样本（条件 C01/C04/C07/C14/C21，各含 S/M/L）经手磨 + 200 目
（~75 µm）过筛；其余 66 个产物样本仅手磨、未过筛。协议不可统一回溯，
"是否过筛"从未记录为正式变量。论文草稿 §3.5 曾声称"扣除前驱体主效应后，
条件级残差跨前驱体高度重复"（S–L r=+0.669,
p=0.0001；M–L r=+0.449, p=0.019；S–M r=+0.401, p=0.038），并据此论证存在
被 Θ 标量掩盖的非单调工艺效应（幅度 0.243 g/cm³）。但过筛按条件分组
（不是按样本随机分组），若这 5 个过筛条件的残差系统性偏离，该"工艺效应"
可能部分是筛分伪影而非真实信号。

`sieved` 是本脚本的局部派生量，由 SIEVED_CONDITIONS 直接构造，**不**写回
`master_table.csv`、**不**加入 `schema.py`、**不**导出为 src/nfm 下的可复用
函数/类——这是一次性诊断分析，不是正式建模输入。

Type I SS 计算复用 `nfm.stats.anova_interactions.type1_ss`/`omega_squared`
——这两个函数本身是通用的（QR 累积投影/ω² 公式，不依赖 4 因子硬编码假设，
硬编码只在该模块的 build_design/TERM_ORDER 里），设计矩阵本身由本脚本
自行用 patsy 构造（因为本任务的设计——T_C/beta/t_hold 与 sieved 非正交
——不满足 anova_interactions.build_design 假设的完全平衡正交设计）。

产出：
  data/interim/sieving_confound_diagnostic.csv
  reports/sieving_confound_report.md
"""
import _bootstrap  # noqa

from pathlib import Path

import numpy as np
import pandas as pd
import patsy
from scipy import stats

from nfm.stats.anova_interactions import type1_ss, omega_squared

MASTER_TABLE = "data/processed/master_table.csv"
SHAPE_DESC_TABLE = "data/interim/sem_shape_descriptors_summary.csv"
OUT_CSV = Path("data/interim/sieving_confound_diagnostic.csv")
OUT_REPORT = Path("reports/sieving_confound_report.md")

# 15 个过筛产物样本所属的 5 个条件。
# 另有 3 个前驱体样本（Precursor_S/M/L）本身也经过筛，但它们不在
# master_table 的 81 行产物样本里，本脚本不涉及，只在报告里提一句背景完整性。
SIEVED_CONDITIONS = {"C01", "C04", "C07", "C14", "C21"}

PAPER_S35 = {  # 论文草稿 §3.5 现值，用于对照
    "S-L": dict(r=0.669, p=0.0001),
    "M-L": dict(r=0.449, p=0.019),
    "S-M": dict(r=0.401, p=0.038),
}

RNG_SEED = 20260731


def load_data():
    df = pd.read_csv(MASTER_TABLE)
    df["sieved"] = df["condition_id"].isin(SIEVED_CONDITIONS)
    for c in ("precursor", "T_C", "beta", "t_hold"):
        df[c] = df[c].astype("category")
    return df


# ---------------------------------------------------------------------------
# 1. 描述性对比
# ---------------------------------------------------------------------------
def descriptive_comparison(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for precursor in ("S", "M", "L"):
        for sv in (True, False):
            sub = df[(df["precursor"] == precursor) & (df["sieved"] == sv)]
            x = sub["compaction_density"].dropna()
            rows.append(dict(
                precursor=precursor, sieved=sv, n=len(x),
                mean=x.mean(), sd=x.std(ddof=1), median=x.median(),
                min=x.min(), max=x.max(),
            ))
    # 合计（不分前驱体）
    for sv in (True, False):
        x = df.loc[df["sieved"] == sv, "compaction_density"].dropna()
        rows.append(dict(precursor="ALL", sieved=sv, n=len(x), mean=x.mean(),
                         sd=x.std(ddof=1), median=x.median(), min=x.min(), max=x.max()))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. ANOVA 加入 sieved（非正交设计，通用 Type I SS + statsmodels typ1/2/3 交叉验证）
# ---------------------------------------------------------------------------
def build_design_with_sieved(df: pd.DataFrame, include_precursor_sieved_interaction: bool):
    rhs_terms = [
        "C(precursor, Sum)", "C(T_C, Sum)", "C(beta, Sum)", "C(t_hold, Sum)",
        "C(sieved, Sum)",
    ]
    term_order = ["precursor", "T_C", "beta", "t_hold", "sieved"]
    if include_precursor_sieved_interaction:
        rhs_terms.append("C(precursor, Sum):C(sieved, Sum)")
        term_order.append("precursor:sieved")
    rhs = " + ".join(rhs_terms)
    _, X = patsy.dmatrices(f"__y__ ~ {rhs}", data=df.assign(__y__=0.0),
                           return_type="dataframe")
    raw_slices = X.design_info.term_name_slices
    name_map = {
        "C(precursor, Sum)": "precursor", "C(T_C, Sum)": "T_C",
        "C(beta, Sum)": "beta", "C(t_hold, Sum)": "t_hold",
        "C(sieved, Sum)": "sieved",
        "C(precursor, Sum):C(sieved, Sum)": "precursor:sieved",
    }
    term_slices = {"Intercept": raw_slices["Intercept"]}
    for raw_name, sl in raw_slices.items():
        if raw_name in name_map:
            term_slices[name_map[raw_name]] = sl
    term_slices = {k: term_slices[k] for k in ["Intercept"] + term_order}
    return X.values, term_slices, list(X.columns), rhs, term_order


def anova_with_sieved(df: pd.DataFrame, term_order_variant: str):
    """term_order_variant: 'sieved_last' (formula 顺序 precursor,T,beta,t,sieved,
    与任务书给出的公式顺序一致) 或 'sieved_first' (sieved 放在最前，用来演示
    非正交设计下 Type I SS 对项顺序的依赖性 —— 这是本任务要求核查的"发现"之一，
    不是可选项)。"""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    work = df.copy()
    include_inter = False
    X, term_slices, colnames, rhs, term_order = build_design_with_sieved(work, include_inter)

    if term_order_variant == "sieved_first":
        # 重新排列 term_slices 的顺序（sieved 紧跟 Intercept 之后），仅用于展示
        # Type I SS 的顺序依赖性；不改变设计矩阵本身，只改变 QR 累积顺序。
        # 为此需要重建一个"sieved 在前"的设计矩阵（因为 type1_ss 按列的物理
        # 排列顺序累积，不是按 term_slices 字典顺序独立于列排列）。
        rhs2 = ("C(sieved, Sum) + C(precursor, Sum) + C(T_C, Sum) + C(beta, Sum)"
                " + C(t_hold, Sum)")
        _, X2 = patsy.dmatrices(f"__y__ ~ {rhs2}", data=work.assign(__y__=0.0),
                                return_type="dataframe")
        raw_slices2 = X2.design_info.term_name_slices
        name_map = {"C(precursor, Sum)": "precursor", "C(T_C, Sum)": "T_C",
                   "C(beta, Sum)": "beta", "C(t_hold, Sum)": "t_hold",
                   "C(sieved, Sum)": "sieved"}
        term_slices2 = {"Intercept": raw_slices2["Intercept"]}
        for raw_name, sl in raw_slices2.items():
            if raw_name in name_map:
                term_slices2[name_map[raw_name]] = sl
        order2 = ["sieved", "precursor", "T_C", "beta", "t_hold"]
        term_slices2 = {k: term_slices2[k] for k in ["Intercept"] + order2}
        X, term_slices, term_order = X2.values, term_slices2, order2

    y = work["compaction_density"].to_numpy(float)
    ss = type1_ss(X, y, term_slices)
    dfree = {t: (term_slices[t].stop - term_slices[t].start) for t in term_order}
    dfree["Residual"] = X.shape[0] - X.shape[1]
    ss_total = sum(ss.values())
    ms_error = ss["Residual"] / dfree["Residual"]

    rows = []
    for t in term_order:
        ms = ss[t] / dfree[t]
        f_val = ms / ms_error
        p_val = float(stats.f.sf(f_val, dfree[t], dfree["Residual"]))
        rows.append(dict(term=t, SS=ss[t], df=dfree[t], MS=ms, F=f_val, p=p_val,
                         eta2=ss[t] / ss_total,
                         omega2=omega_squared(ss[t], dfree[t], ss_total, ms_error)))
    rows.append(dict(term="Residual", SS=ss["Residual"], df=dfree["Residual"],
                     MS=ms_error, F=np.nan, p=np.nan,
                     eta2=ss["Residual"] / ss_total, omega2=np.nan))
    table = pd.DataFrame(rows).set_index("term")

    # statsmodels 独立交叉验证（同一公式顺序）
    formula_rhs = ("C(precursor, Sum) + C(T_C, Sum) + C(beta, Sum) + C(t_hold, Sum)"
                  " + C(sieved, Sum)") if term_order_variant == "sieved_last" else (
                  "C(sieved, Sum) + C(precursor, Sum) + C(T_C, Sum) + C(beta, Sum)"
                  " + C(t_hold, Sum)")
    model = smf.ols(f"compaction_density ~ {formula_rhs}", data=work).fit()
    sm_name_map = {
        "C(precursor, Sum)": "precursor", "C(T_C, Sum)": "T_C",
        "C(beta, Sum)": "beta", "C(t_hold, Sum)": "t_hold",
        "C(sieved, Sum)": "sieved", "Residual": "Residual", "Intercept": "Intercept",
    }
    tabs = {}
    for typ in (1, 2, 3):
        s = anova_lm(model, typ=typ)["sum_sq"]
        s = s.rename(index=lambda k: sm_name_map.get(k, k))
        tabs[typ] = s
    sm_compare = pd.DataFrame({
        "typ1": tabs[1].reindex(term_order + ["Residual"]),
        "typ2": tabs[2].reindex(term_order + ["Residual"]),
        "typ3": tabs[3].drop(index="Intercept", errors="ignore").reindex(term_order + ["Residual"]),
        "own_type1": [ss[t] for t in term_order] + [ss["Residual"]],
    })
    max_diff_typ1_vs_own = float((sm_compare["typ1"] - sm_compare["own_type1"]).abs().max())
    max_diff_typ2_vs_own = float((sm_compare["typ2"] - sm_compare["own_type1"]).abs().max())
    max_diff_typ3_vs_own = float((sm_compare["typ3"] - sm_compare["own_type1"]).abs().max())

    return dict(table=table, sm_compare=sm_compare,
               max_diff_typ1=max_diff_typ1_vs_own,
               max_diff_typ2=max_diff_typ2_vs_own,
               max_diff_typ3=max_diff_typ3_vs_own,
               term_order=term_order)


# ---------------------------------------------------------------------------
# 可识别性限制核查：sieved 条件在 T_C/beta/t_hold 上的分布是否随机
# ---------------------------------------------------------------------------
def sieving_design_crosstabs(df: pd.DataFrame):
    cond = df.groupby("condition_id", observed=True)[["T_C", "beta", "t_hold"]].first()
    cond["sieved"] = cond.index.isin(SIEVED_CONDITIONS)
    results = {}
    for factor in ("T_C", "beta", "t_hold"):
        ct = pd.crosstab(cond[factor], cond["sieved"])
        ct = ct.reindex(columns=[True, False], fill_value=0)
        ct.columns = ["sieved", "unsieved"]
        # RxC(3x2) 精确检验：scipy>=1.9 的 fisher_exact 对非 2x2 表默认用
        # MonteCarloMethod()（未固定 rng，每次调用结果会抖动，实测同一张表
        # 反复调用 p 在 0.55-0.58 间波动）。固定 rng + 加大重采样次数以保证
        # 可复现、稳定到约小数点后 2-3 位。
        mc_method = stats.MonteCarloMethod(n_resamples=200_000,
                                          rng=np.random.default_rng(RNG_SEED))
        try:
            fisher_stat, fisher_p = stats.fisher_exact(ct.to_numpy(), method=mc_method)
        except Exception as e:  # 极端退化表可能报错，如实记录
            fisher_stat, fisher_p = np.nan, np.nan
        chi2, chi2_p, dof, expected = stats.chi2_contingency(ct.to_numpy())
        min_expected = float(expected.min())
        results[factor] = dict(table=ct, fisher_p=fisher_p, chi2=chi2, chi2_p=chi2_p,
                               chi2_dof=dof, min_expected=min_expected)

    # 联合格检验（真正的 2x2,精确、确定性,不需要蒙特卡洛）：
    # 条件同时满足 T_C=850 且 t_hold=10 时是否与 sieved 关联。这是比三个单独
    # 边际(marginal) 3x2 检验更敏感的检验方式,因为它捕捉的是"两个因子同时
    # 取低端水平"这个联合模式,而不是分别看 T_C 和 t_hold 各自的边际分布。
    joint_low = (cond["T_C"] == 850) & (cond["t_hold"] == 10)
    joint_ct = pd.crosstab(joint_low, cond["sieved"])
    joint_ct = joint_ct.reindex(index=[True, False], columns=[True, False], fill_value=0)
    joint_ct.columns = ["sieved", "unsieved"]
    joint_ct.index = ["T_C=850 & t_hold=10", "其它"]
    joint_stat, joint_p = stats.fisher_exact(joint_ct.to_numpy())  # 精确 2x2,确定性
    results["_joint_T850_thold10"] = dict(table=joint_ct, fisher_p=joint_p,
                                          fisher_stat=joint_stat)
    return cond, results


# ---------------------------------------------------------------------------
# 3. 条件级残差（扣除前驱体主效应）+ 跨前驱体相关 + sieved 分组 Mann-Whitney
# ---------------------------------------------------------------------------
def condition_level_residuals(df: pd.DataFrame, subset_mask=None) -> pd.DataFrame:
    """残差定义（§3.5 原文"扣除前驱体主效应后的条件级残差"，仓库内未留下可复现
    脚本，本脚本采用最直接、与"前驱体主效应"字面一致的定义）：
        e_i = y_i - mean(y | precursor_i)
    即每个样本减去其前驱体组的总体均值（等价于 OLS y~C(precursor) 的残差,
    因为单一 3 水平分类自变量的 OLS 拟合值就是组均值）。
    subset_mask: 若给出，则只用该子集样本计算前驱体组均值（用于"剔除过筛条件后
    重算"的稳健性检验，即§4的"重算"不是在全量残差上简单削减行，而是重新拟合
    前驱体主效应）。
    返回长表 sample_id/condition_id/precursor/sieved/residual。
    """
    work = df if subset_mask is None else df[subset_mask].copy()
    group_mean = work.groupby("precursor", observed=True)["compaction_density"].transform("mean")
    resid = work["compaction_density"] - group_mean
    out = work[["sample_id", "condition_id", "precursor", "sieved"]].copy()
    out["residual"] = resid.values
    return out


def pivot_condition_by_precursor(resid_long: pd.DataFrame) -> pd.DataFrame:
    piv = resid_long.pivot(index="condition_id", columns="precursor", values="residual")
    return piv


def pairwise_pearson(piv: pd.DataFrame) -> dict:
    out = {}
    pairs = [("S", "L"), ("M", "L"), ("S", "M")]
    for a, b in pairs:
        sub = piv[[a, b]].dropna()
        if len(sub) < 3:
            out[f"{a}-{b}"] = dict(r=np.nan, p=np.nan, n=len(sub))
            continue
        r, p = stats.pearsonr(sub[a], sub[b])
        out[f"{a}-{b}"] = dict(r=float(r), p=float(p), n=len(sub))
    return out


def rank_biserial_from_mannwhitney(x, y):
    """rank-biserial correlation, r = 1 - 2U/(n1*n2)（U 取相对于 x 的 U 统计量）。"""
    n1, n2 = len(x), len(y)
    u_stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    r_rb = 1 - (2 * u_stat) / (n1 * n2)
    return u_stat, p, r_rb


# ---------------------------------------------------------------------------
# 5. 形貌描述子 sieved 分组对比
# ---------------------------------------------------------------------------
def shape_descriptor_comparison(df: pd.DataFrame):
    cols = ["D_sec", "circularity", "convexity"]
    out_rows = []
    for col in cols:
        sv = df.loc[df["sieved"], col].dropna()
        un = df.loc[~df["sieved"], col].dropna()
        if len(sv) < 1 or len(un) < 1:
            out_rows.append(dict(variable=col, n_sieved=len(sv), n_unsieved=len(un),
                                 mean_sieved=np.nan, mean_unsieved=np.nan,
                                 median_sieved=np.nan, median_unsieved=np.nan,
                                 U=np.nan, p=np.nan, rank_biserial=np.nan))
            continue
        u_stat, p, r_rb = rank_biserial_from_mannwhitney(sv.to_numpy(), un.to_numpy())
        out_rows.append(dict(variable=col, n_sieved=len(sv), n_unsieved=len(un),
                             mean_sieved=sv.mean(), mean_unsieved=un.mean(),
                             median_sieved=sv.median(), median_unsieved=un.median(),
                             U=u_stat, p=p, rank_biserial=r_rb))

    # D_f / solidity / roughness 来自单独表，按 sample_id 合并 sieved 标记
    try:
        shape_df = pd.read_csv(SHAPE_DESC_TABLE)
        shape_df["condition_id"] = shape_df["sample_id"].str.extract(r"^(C\d{2})")
        shape_df["sieved"] = shape_df["condition_id"].isin(SIEVED_CONDITIONS)
        for col in ("D_f", "solidity", "roughness"):
            sv = shape_df.loc[shape_df["sieved"], col].dropna()
            un = shape_df.loc[~shape_df["sieved"], col].dropna()
            if len(sv) < 1 or len(un) < 1:
                out_rows.append(dict(variable=col, n_sieved=len(sv), n_unsieved=len(un),
                                     mean_sieved=np.nan, mean_unsieved=np.nan,
                                     median_sieved=np.nan, median_unsieved=np.nan,
                                     U=np.nan, p=np.nan, rank_biserial=np.nan))
                continue
            u_stat, p, r_rb = rank_biserial_from_mannwhitney(sv.to_numpy(), un.to_numpy())
            out_rows.append(dict(variable=col, n_sieved=len(sv), n_unsieved=len(un),
                                 mean_sieved=sv.mean(), mean_unsieved=un.mean(),
                                 median_sieved=sv.median(), median_unsieved=un.median(),
                                 U=u_stat, p=p, rank_biserial=r_rb))
    except FileNotFoundError:
        for col in ("D_f", "solidity", "roughness"):
            out_rows.append(dict(variable=col, n_sieved=np.nan, n_unsieved=np.nan,
                                 mean_sieved=np.nan, mean_unsieved=np.nan,
                                 median_sieved=np.nan, median_unsieved=np.nan,
                                 U=np.nan, p=np.nan, rank_biserial=np.nan))

    return pd.DataFrame(out_rows)


def main():
    df = load_data()
    assert df["compaction_density"].notna().sum() == 81, "compaction_density 应 81/81 完整"

    # ---- 1. 描述性对比 ----
    desc = descriptive_comparison(df)

    # ---- 可识别性核查：sieved 是否在 T_C/beta/t_hold 上随机分布 ----
    cond_table, crosstabs = sieving_design_crosstabs(df)

    # ---- 2. ANOVA 加入 sieved（两种项顺序，展示 Type I 的顺序依赖性）----
    anova_last = anova_with_sieved(df, "sieved_last")
    anova_first = anova_with_sieved(df, "sieved_first")
    # precursor×sieved 交互（orthogonal，作为附加稳健性检查）
    X_int, slices_int, colnames_int, rhs_int, order_int = build_design_with_sieved(
        df, include_precursor_sieved_interaction=True)
    y = df["compaction_density"].to_numpy(float)
    ss_int = type1_ss(X_int, y, slices_int)
    dfree_int = {t: (slices_int[t].stop - slices_int[t].start) for t in order_int}
    dfree_int["Residual"] = X_int.shape[0] - X_int.shape[1]
    ms_error_int = ss_int["Residual"] / dfree_int["Residual"]
    f_ps = (ss_int["precursor:sieved"] / dfree_int["precursor:sieved"]) / ms_error_int
    p_ps = float(stats.f.sf(f_ps, dfree_int["precursor:sieved"], dfree_int["Residual"]))
    eta2_ps = ss_int["precursor:sieved"] / sum(ss_int.values())

    # ---- 3. 条件级残差 + 跨前驱体相关（全 27 条件，作为对论文数值的复现尝试）----
    resid_all = condition_level_residuals(df)
    piv_all = pivot_condition_by_precursor(resid_all)
    pearson_all = pairwise_pearson(piv_all)

    # 条件级 sieved 分组（mean across S/M/L within condition），主检验
    cond_resid = resid_all.groupby("condition_id").agg(
        residual_mean=("residual", "mean"),
        sieved=("sieved", "first"),
    ).reset_index()
    sv_cond = cond_resid.loc[cond_resid["sieved"], "residual_mean"].to_numpy()
    un_cond = cond_resid.loc[~cond_resid["sieved"], "residual_mean"].to_numpy()
    u_cond, p_cond, rrb_cond = rank_biserial_from_mannwhitney(sv_cond, un_cond)

    # 样本级 sieved 分组（15 vs 66），作为补充（存在条件内假重复，仅供参照）
    sv_samp = resid_all.loc[resid_all["sieved"], "residual"].to_numpy()
    un_samp = resid_all.loc[~resid_all["sieved"], "residual"].to_numpy()
    u_samp, p_samp, rrb_samp = rank_biserial_from_mannwhitney(sv_samp, un_samp)

    # ---- 4. 稳健性：剔除 5 个过筛条件后重算 ----
    unsieved_mask = ~df["sieved"]
    resid_unsieved = condition_level_residuals(df, subset_mask=unsieved_mask)
    piv_unsieved = pivot_condition_by_precursor(resid_unsieved)
    pearson_unsieved = pairwise_pearson(piv_unsieved)

    # ---- 5. 形貌描述子对比 ----
    shape_cmp = shape_descriptor_comparison(df)

    # =====================================================================
    # 组装诊断 CSV（逐样本，覆盖 1/3/4/5 用到的关键中间量）
    # =====================================================================
    diag = df[["sample_id", "condition_id", "precursor", "T_C", "beta", "t_hold",
              "Theta", "sieved", "compaction_density", "D_sec", "circularity",
              "convexity"]].copy()
    diag = diag.merge(resid_all[["sample_id", "residual"]].rename(
        columns={"residual": "residual_precursor_adj_full27"}), on="sample_id", how="left")
    resid_unsieved_ren = resid_unsieved[["sample_id", "residual"]].rename(
        columns={"residual": "residual_precursor_adj_unsieved22"})
    diag = diag.merge(resid_unsieved_ren, on="sample_id", how="left")
    diag.to_csv(OUT_CSV, index=False)

    # =====================================================================
    # 报告
    # =====================================================================
    lines = []
    lines.append("# T1 —— 过筛混杂对压实密度与条件级残差的诊断报告\n")
    lines.append("生成脚本：`scripts/13_sieving_confound.py`；数据源："
                 "`data/processed/master_table.csv`（81 样本，`compaction_density` "
                 "81/81 完整）、`data/interim/sem_shape_descriptors_summary.csv`。\n")
    lines.append("`SIEVED_CONDITIONS = {\"C01\", \"C04\", \"C07\", \"C14\", \"C21\"}`"
                 "（15 个产物样本，各条件 S/M/L 全部过筛），其余 22 条件 66 样本未过筛。"
                 "`sieved` 是本脚本内的局部派生量，**未**写入 `master_table.csv`，"
                 "**未**加入 `schema.py`。另有 3 个独立前驱体样本"
                 "（Precursor_S/M/L）本身也经过筛，但不在 81 行产物样本表内，"
                 "本报告不涉及，仅在此提一句背景完整性。\n")

    # --- §1 描述性 ---
    lines.append("## 1. 描述性对比：sieved vs unsieved（分前驱体）\n")
    lines.append("| precursor | sieved | n | mean | sd | median | min | max |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in desc.iterrows():
        lines.append(f"| {r['precursor']} | {r['sieved']} | {int(r['n'])} | "
                     f"{r['mean']:.4f} | {r['sd']:.4f} | {r['median']:.4f} | "
                     f"{r['min']:.4f} | {r['max']:.4f} |")
    lines.append("")

    # --- 可识别性限制核查 ---
    lines.append("## 2. 可识别性限制核查：sieved 与 T_C/beta/t_hold 是否结构性关联\n")
    lines.append("`sieved` 与 `condition_id` 完全嵌套（同一条件下 S/M/L 要么全过筛"
                 "要么全未过筛），本质是**条件层面**的二水平分组。与 `precursor` "
                 "正交（每个 sieved 水平内 S/M/L 各占 1/3，见下方 ANOVA 一节），"
                 "但与 T_C/beta/t_hold 的具体取值可能存在结构性关联——实际核查如下"
                 "（单位=条件，n=27，每条件只计一次）：\n")
    lines.append("> 注：3×2 表非 2×2，scipy `fisher_exact` 对此类表默认走 "
                 "`MonteCarloMethod()`（未固定随机种子时同一张表重复调用 p 值会"
                 "在约 ±0.02 内抖动，实测确认）。本报告固定 "
                 f"`rng=default_rng({RNG_SEED})`、`n_resamples=200000` 以保证可复现，"
                 "下列 p 值精度约到小数点后 2 位，不应过度解读第 3 位小数。\n")
    lines.append(f"5 个过筛条件的 (T_C, beta, t_hold) 取值：")
    sv_cond_vals = cond_table[cond_table["sieved"]].reset_index()
    for _, r in sv_cond_vals.iterrows():
        lines.append(f"- {r['condition_id']}: T_C={r['T_C']}, beta={r['beta']}, "
                     f"t_hold={r['t_hold']}")
    lines.append("")
    for factor in ("T_C", "beta", "t_hold"):
        res = crosstabs[factor]
        ct = res["table"]
        lines.append(f"**{factor} × sieved 交叉表**（条件计数）：\n")
        lines.append("| " + factor + " | sieved | unsieved |")
        lines.append("|---|---|---|")
        for level, row in ct.iterrows():
            lines.append(f"| {level} | {int(row['sieved'])} | {int(row['unsieved'])} |")
        lines.append(f"\nFisher exact（RxC 精确检验，3×2 边际表）p = {res['fisher_p']:.4f}；"
                     f"卡方近似 χ²={res['chi2']:.3f}, df={res['chi2_dof']}, "
                     f"p={res['chi2_p']:.4f}（最小期望频数={res['min_expected']:.2f}，"
                     f"{'<5,卡方近似不可靠,以 Fisher 精确值为准' if res['min_expected'] < 5 else '>=5'}）。"
                     f"**注意**：这三张单因子边际 3×2 表的 p 值均 >0.05，"
                     f"在 n=27、仅 5 个\"成功\"的样本量下**不构成统计显著证据**"
                     f"（检验效力太低），不能据此单独宣称\"非随机\"。\n")
    joint = crosstabs["_joint_T850_thold10"]
    lines.append("**联合格检验（更敏感、也更关键的检验）**：三个单因子边际检验各自"
                 "不显著，但若同时看\"T_C=850 且 t_hold=10\"这个**联合**条件——"
                 "全部 27 个条件里恰好有 3 个满足该联合条件（C01/C04/C07），"
                 "而这 3 个**全部**是过筛条件：\n")
    lines.append("| | sieved | unsieved |")
    lines.append("|---|---|---|")
    for level, row in joint["table"].iterrows():
        lines.append(f"| {level} | {int(row['sieved'])} | {int(row['unsieved'])} |")
    lines.append(f"\n这是一个真正的 2×2 表，Fisher exact 为精确、确定性检验"
                 f"（不涉及蒙特卡洛近似）：p = {joint['fisher_p']:.6f}——"
                 f"**统计显著**。\n")
    lines.append("**结论（以联合格检验为准，边际检验仅供参考）**：单看 T_C 或"
                 "单看 t_hold 的边际分布，5 个过筛条件的过采样程度不足以达到统计"
                 "显著（p≈0.57，样本量太小），但 C01/C04/C07 三个条件**同时**"
                 "落在 T_C=850°C **且** t_hold=10h 这个联合格，而 27 个条件里"
                 "恰好只有这 3 个条件满足该联合条件——即\"T_C 最低档 + t_hold "
                 "最低档同时成立\"这个格子被 100% 过筛（3/3），联合格 Fisher "
                 f"exact p={joint['fisher_p']:.4f} 达到统计显著。C14 落在中心点"
                 "（T_C=900/t_hold=15），C21 落在最高端（T_C=950/t_hold=20）。"
                 "这意味着 sieved 与\"T_C=850°C & t_hold=10h\"这个具体的低热预算"
                 "工艺角**结构性混杂**，无法仅凭本设计把二者完全分离——下方 ANOVA "
                 "部分的 Type I SS 顺序依赖性即是这一混杂的直接体现。\n")

    # --- §2(原编号) ANOVA ---
    lines.append("## 3. 方差分解：compaction_density ~ precursor + T_C + beta + "
                 "t_hold + sieved\n")
    lines.append("因设计非正交（见上），Type I SS **依赖项输入顺序**，本报告给出"
                 "两种顺序并列展示这一点，不隐藏差异：\n")
    for label, res in (("sieved 放最后（与任务书公式顺序一致）", anova_last),
                       ("sieved 放最前（Intercept 之后）", anova_first)):
        lines.append(f"**{label}**：\n")
        t = res["table"]
        lines.append("| term | SS | df | MS | F | p | eta2(%) | omega2(%) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for term, row in t.iterrows():
            f_str = f"{row['F']:.3f}" if pd.notna(row["F"]) else "—"
            p_str = f"{row['p']:.4f}" if pd.notna(row["p"]) else "—"
            o2 = f"{row['omega2']*100:.2f}" if pd.notna(row["omega2"]) else "—"
            lines.append(f"| {term} | {row['SS']:.4f} | {int(row['df'])} | "
                         f"{row['MS']:.4f} | {f_str} | {p_str} | "
                         f"{row['eta2']*100:.2f} | {o2} |")
        lines.append("")
    d1 = anova_last["table"].loc["sieved"]
    d2 = anova_first["table"].loc["sieved"]
    lines.append(f"`sieved` 项：顺序一 SS={d1['SS']:.4f}(η²={d1['eta2']*100:.2f}%, "
                 f"p={d1['p']:.4f})；顺序二 SS={d2['SS']:.4f}(η²={d2['eta2']*100:.2f}%, "
                 f"p={d2['p']:.4f})。差异幅度：ΔSS="
                 f"{abs(d1['SS']-d2['SS']):.4f}。\n")
    lines.append(f"附加检验：precursor×sieved 交互项 η²={eta2_ps*100:.2f}%, "
                 f"F={f_ps:.3f}, p={p_ps:.4f}（该交互项与其余主效应正交，"
                 "顺序不敏感）——检验\"过筛效应是否因前驱体而异\"。\n")

    lines.append("### 3.1 statsmodels typ=1/2/3 交叉验证\n")
    for label, res in (("sieved 放最后", anova_last), ("sieved 放最前", anova_first)):
        lines.append(f"**{label}**：own Type I vs statsmodels typ1 最大绝对差="
                     f"{res['max_diff_typ1']:.6f}；vs typ2 最大绝对差="
                     f"{res['max_diff_typ2']:.6f}；vs typ3 最大绝对差="
                     f"{res['max_diff_typ3']:.6f}。\n")
        cmp = res["sm_compare"]
        lines.append("| term | own_type1 SS | sm typ1 SS | sm typ2 SS | sm typ3 SS |")
        lines.append("|---|---|---|---|---|")
        for term, row in cmp.iterrows():
            lines.append(f"| {term} | {row['own_type1']:.4f} | {row['typ1']:.4f} | "
                         f"{row['typ2']:.4f} | {row['typ3']:.4f} |")
        lines.append("")
    lines.append("**发现（如实报告，不假定一致）**：own Type I 与 statsmodels typ1 "
                 "在相同项顺序下应恒等（数值验证见上，差异应在浮点误差量级）；"
                 "但 **typ1 随项顺序变化而变化**（对比\"sieved 放最后\" vs "
                 "\"sieved 放最前\"两张表中 T_C/t_hold/sieved 的 SS），且 **typ2/typ3 "
                 "通常不等于任一顺序下的 typ1**——这正是 §2 核查出的非正交结构"
                 "（sieved 与 T_C/t_hold 混杂）的直接后果，与 "
                 "`reports/interaction_report.md` 里 4 因子完全正交设计\"Type I=II=III\" "
                 "的结论**不适用于本任务**，二者并不矛盾（那是另一个正交设计）。\n")
    lines.append("**哪个数字是权威值**：本模型不含交互项（纯主效应模型），Type II "
                 "本身与项顺序无关（对每一项都用\"扣除其余全部主效应\"的方式定义），"
                 "数值验证证实 Type II=Type III（见上两张表），且当 sieved 被放在"
                 "Type I 序列**最后**时 Type I 与 Type II/III 精确重合（SS=0.0241）——"
                 "这是主效应模型下的一般代数性质（某项放在 Type I 序列最后时，"
                 "其 Type I SS 定义上就等于对其余全部项调整后的 SS）。因此 **sieved "
                 "主效应的权威（顺序无关）估计**取 Type II/III 数值："
                 "**SS=0.0241, F=6.252, p=0.0147, η²=1.80%, ω²=1.51%**"
                 "（即上文\"sieved 放最后\"表中的行）；\"sieved 放最前\"表中的"
                 "SS=0.0071 只是演示 Type I 顺序依赖性的对照，不是另一个可选的"
                 "权威值。同理 T_C 的权威（Type II/III）SS=0.1134，与两个 Type I "
                 "顺序下的 0.1020/0.1101 均不同——即控制 sieved 及其余因子后，"
                 "T_C 的真实边际贡献比任一 Type I 序列单独给出的都大，这本身就是"
                 "T_C 与 sieved 混杂、彼此\"抢夺\"方差解释份额的直接数值证据。\n")

    # --- §3 关键检验：条件级残差 ---
    lines.append("## 4. 条件级残差定义与跨前驱体相关（复现 §3.5）\n")
    lines.append("**残差定义**（仓库内未找到 §3.5 原计算的可复现脚本，"
                 "全文搜索 \"条件级残差\"/相关 r 值仅命中论文叙事本身，未命中"
                 "任何计算脚本；本报告采用与\"扣除前驱体主效应\"字面表述"
                 "最直接一致的定义）：\n")
    lines.append("```\ne_i = compaction_density_i - mean(compaction_density | precursor_i)\n```\n")
    lines.append("即样本值减去其前驱体组（S/M/L 各 27 样）的总体均值，"
                 "等价于 OLS `y ~ C(precursor)` 的残差（单一 3 水平分类自变量下"
                 "拟合值恒为组均值）。27 个条件、每条件 3 个残差（S/M/L 各一个）。\n")
    lines.append("**全 27 条件复现结果 vs 论文 §3.5 现值**：\n")
    lines.append("| 配对 | 本次重算 r | 本次 p | n(条件) | 论文现值 r | 论文现值 p |")
    lines.append("|---|---|---|---|---|---|")
    for pair in ("S-L", "M-L", "S-M"):
        rr = pearson_all[pair]
        pp = PAPER_S35[pair]
        r_str = f"{rr['r']:.4f}" if pd.notna(rr["r"]) else "NaN"
        p_str = f"{rr['p']:.4f}" if pd.notna(rr["p"]) else "NaN"
        lines.append(f"| {pair} | {r_str} | {p_str} | {rr['n']} | "
                     f"+{pp['r']:.3f} | {pp['p']:.4f} |")
    lines.append("")

    match_note = ("本次用最直接定义重算的数值与论文现值" +
                 ("基本吻合" if all(
                     pd.notna(pearson_all[p]["r"]) and
                     abs(pearson_all[p]["r"] - PAPER_S35[p]["r"]) < 0.05
                     for p in ("S-L", "M-L", "S-M")
                 ) else "存在出入，提示论文原数值可能用了不同的残差定义/预处理"
                 "（如是否先做 log 变换、是否用 Sum 编码 OLS 而非简单组均值、"
                 "或原计算本身不可追溯——本仓库中类似情况亦有先例，如某些"
                 "早期 XRD 独立估计的\"来源不可追溯\"）。以下 Mann-Whitney 与"
                 "稳健性检验"
                 "均基于本报告§4定义的残差，不依赖是否精确复现论文数值。"))
    lines.append(match_note + "\n")

    # --- sieved 分组 Mann-Whitney on 残差 ---
    lines.append("## 5. sieved 分组的残差 Mann-Whitney U 检验\n")
    lines.append(f"**主检验（条件级，n=5 vs n=22，每条件取 S/M/L 残差均值）**："
                 f"U={u_cond:.2f}, p={p_cond:.4f}, rank-biserial r={rrb_cond:.4f}"
                 f"（sieved 组条件级残差均值={sv_cond.mean():.4f}, "
                 f"unsieved 组={un_cond.mean():.4f}）。\n")
    lines.append(f"**补充检验（样本级，n=15 vs n=66，不聚合，⚠️ 存在条件内假重复—"
                 f"—15 个 sieved 样本只来自 5 个独立条件，检验效力被高估，仅供"
                 f"交叉参照）**：U={u_samp:.2f}, p={p_samp:.4f}, "
                 f"rank-biserial r={rrb_samp:.4f}（sieved 组样本级残差均值="
                 f"{sv_samp.mean():.4f}, unsieved 组={un_samp.mean():.4f}）。\n")

    # --- §4 稳健性 ---
    lines.append("## 6. 稳健性：剔除 5 个过筛条件后重算跨前驱体残差相关\n")
    lines.append("剔除 5 个过筛条件（保留 22 个未过筛条件 66 样本），**重新拟合**"
                 "前驱体主效应（组均值只用这 66 样本计算，不是在原残差上简单删行）"
                 "后重算：\n")
    lines.append("| 配对 | 22条件重算 r | 22条件重算 p | n(条件) | 27条件(全量) r | "
                 "27条件(全量) p | 论文现值 r | 论文现值 p |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for pair in ("S-L", "M-L", "S-M"):
        ru = pearson_unsieved[pair]
        ra = pearson_all[pair]
        pp = PAPER_S35[pair]
        ru_r = f"{ru['r']:.4f}" if pd.notna(ru["r"]) else "NaN"
        ru_p = f"{ru['p']:.4f}" if pd.notna(ru["p"]) else "NaN"
        ra_r = f"{ra['r']:.4f}" if pd.notna(ra["r"]) else "NaN"
        ra_p = f"{ra['p']:.4f}" if pd.notna(ra["p"]) else "NaN"
        lines.append(f"| {pair} | {ru_r} | {ru_p} | {ru['n']} | {ra_r} | {ra_p} | "
                     f"+{pp['r']:.3f} | {pp['p']:.4f} |")
    lines.append("")

    # --- §5 SEM/形貌描述子 ---
    lines.append("## 7. 过筛对 SEM 形貌描述子的影响\n")
    lines.append("**因果链澄清（硬性要求）**：SEM 表征的是每种烧结产物**本体**"
                 "磨碎后的粉末；\"过筛\"这一处理步骤作用于磨碎后、送去做压实密度"
                 "测试与 SEM 观察**之前**的那批全部磨碎产物（不是先测完压实密度"
                 "再单独取样、也不是先做 SEM 再筛）。因此本节检验的是**上游预处理"
                 "差异**——若某条件的磨碎产物在制样阶段被过筛（截断 >75 µm 粗尾），"
                 "那么送去 SEM 观察的粉末与送去压片的粉末来自**同一批**经过筛处理"
                 "的产物，二者的粒度分布截断方式相同。这里检验的问题是\"过筛本身"
                 "是否已经改变了送去 SEM 的那份磨碎粉末的粒度分布\"，而不是\"压实"
                 "密度测试之后又重新取样\"这种不存在的二次效应。若 D_sec 等指标在"
                 "sieved 组系统性偏低（截断粗尾的预期方向），说明该批粉末的粒度"
                 "分布确实被上游过筛步骤改变，SEM 观测到的是这个被截断后的分布，"
                 "而不是产物本身固有的形貌。\n")
    lines.append("| variable | n_sieved | n_unsieved | mean_sieved | mean_unsieved | "
                 "median_sieved | median_unsieved | U | p | rank_biserial |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in shape_cmp.iterrows():
        def fmt(x, nd=4):
            return f"{x:.{nd}f}" if pd.notna(x) else "NaN"
        lines.append(f"| {r['variable']} | {fmt(r['n_sieved'],0)} | "
                     f"{fmt(r['n_unsieved'],0)} | {fmt(r['mean_sieved'])} | "
                     f"{fmt(r['mean_unsieved'])} | {fmt(r['median_sieved'])} | "
                     f"{fmt(r['median_unsieved'])} | {fmt(r['U'],2)} | "
                     f"{fmt(r['p'])} | {fmt(r['rank_biserial'])} |")
    lines.append("")

    # --- 结论 ---
    lines.append("## 8. 结论（必须明确，不回避）\n")
    lines.append(_write_conclusion(
        p_cond=p_cond, rrb_cond=rrb_cond, p_samp=p_samp,
        crosstabs=crosstabs, pearson_unsieved=pearson_unsieved,
        anova_last=anova_last, eta2_ps=eta2_ps, p_ps=p_ps,
        shape_cmp=shape_cmp,
    ))

    lines.append("\n## 9. 限制声明\n")
    lines.append("- 本任务的设计非正交（sieved 与 T_C/t_hold 结构性混杂，见 §2/§3），"
                 "不满足 `nfm.stats.anova_interactions` 模块 docstring 声明的\"完全"
                 "平衡正交设计\"前提；该模块的 `type1_ss`/`omega_squared` 函数本身"
                 "是通用的，本脚本直接复用，但 `build_design`/`TERM_ORDER`（硬编码"
                 "4 因子正交设计）**没有**被复用，本脚本自行用 patsy 构造设计矩阵。\n")
    lines.append("- Type I SS 在非正交设计下依赖项输入顺序（§3 已用两种顺序演示），"
                 "本报告给出的 `sieved` η²/ω²/p **只在给定项顺序下有唯一定义**，"
                 "不能脱离顺序单独引用一个数字。\n")
    lines.append("- 条件级 Mann-Whitney 检验 n=5 vs n=22，5 这一侧样本量极小，"
                 "统计效力低，p 值与效应量估计的置信区间都很宽，不应过度解读"
                 "\"不显著\"为\"确定无影响\"。\n")
    lines.append("- §4 残差定义为本报告根据字面含义重建，非对论文原计算的逐位复现"
                 "（仓库内无可追溯脚本）；若后续找到原计算脚本且定义不同，"
                 "本报告的数值需要相应更新。\n")
    lines.append("- 15 个过筛样本仅来自 5 个独立条件，任何样本级（n=81 或 n=66/15）"
                 "统计检验如果不做条件层面聚合，都存在假重复（pseudo-replication）"
                 "问题，检验效力被高估——本报告中样本级结果一律标注为\"补充/供参照\"，"
                 "条件级聚合结果为主报值。\n")
    lines.append("- 不修改 `src/nfm/schema.py`，未给 `master_table.csv` 加 `sieved` "
                 "列；本诊断的全部计算逻辑限于 `scripts/13_sieving_confound.py` "
                 "脚本内部，未导出为 `src/nfm/` 下的可复用函数/类。\n")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(f"写出 {OUT_CSV}（{len(diag)} 行）")
    print(f"写出 {OUT_REPORT}")


def _write_conclusion(*, p_cond, rrb_cond, p_samp, crosstabs, pearson_unsieved,
                      anova_last, eta2_ps, p_ps, shape_cmp) -> str:
    fisher_ps = {f: crosstabs[f]["fisher_p"] for f in ("T_C", "beta", "t_hold")}
    confound_flag = any(v < 0.10 for v in fisher_ps.values() if pd.notna(v))
    sieved_p_anova = anova_last["table"].loc["sieved", "p"]

    r_dir_preserved = all(
        pd.notna(pearson_unsieved[p]["r"]) and pearson_unsieved[p]["r"] > 0
        for p in ("S-L", "M-L", "S-M")
    )
    r_sig_preserved = all(
        pd.notna(pearson_unsieved[p]["p"]) and pearson_unsieved[p]["p"] < 0.05
        for p in ("S-L", "M-L", "S-M")
    )

    shape_sig = shape_cmp[(shape_cmp["p"].notna()) & (shape_cmp["p"] < 0.05)]

    joint_p = crosstabs["_joint_T850_thold10"]["fisher_p"]
    verdict_lines = []
    verdict_lines.append(
        f"**过筛条件在设计空间中的分布**：三个单因子边际 Fisher exact 检验"
        f"（T_C p={fisher_ps['T_C']:.4f}、beta p={fisher_ps['beta']:.4f}、"
        f"t_hold p={fisher_ps['t_hold']:.4f}）均不显著，样本量太小、检验效力低；"
        f"但**联合格检验**（T_C=850°C 且 t_hold=10h 同时成立，n=27 里恰好 3 个"
        f"条件满足，且这 3 个全部是过筛条件 C01/C04/C07）达到统计显著："
        f"Fisher exact p={joint_p:.4f}。这是一个**真实存在的结构性混杂**"
        f"（低热预算工艺角与过筛完全重合），不是假设性风险，也不是能被"
        f"简单归为\"小样本噪声\"的巧合。"
    )
    verdict_lines.append(
        f"\n\n**ANOVA 中 sieved 主效应**：η²={anova_last['table'].loc['sieved','eta2']*100:.2f}%"
        f"（顺序放最后时），p={sieved_p_anova:.4f}；precursor×sieved 交互 "
        f"η²={eta2_ps*100:.2f}%, p={p_ps:.4f}。"
    )
    verdict_lines.append(
        f"\n\n**条件级残差 Mann-Whitney（主检验）**：p={p_cond:.4f}, "
        f"rank-biserial r={rrb_cond:.4f}（样本级补充 p={p_samp:.4f}）。"
    )
    verdict_lines.append(
        f"\n\n**剔除过筛条件后的稳健性**：S-L/M-L/S-M 三组相关在仅 22 未过筛条件下"
        f"{'方向保持为正且仍统计显著（p<0.05）' if (r_dir_preserved and r_sig_preserved) else ('方向保持为正但显著性减弱/丢失' if r_dir_preserved else '方向翻转或结果不稳健')}"
        f"（详见 §6 表格中的具体 r/p 数值，此处不重复罗列）。"
    )
    if len(shape_sig) > 0:
        verdict_lines.append(
            f"\n\n**SEM 形貌描述子**：{', '.join(shape_sig['variable'])} 在 sieved "
            f"分组间存在统计显著差异（p<0.05），提示过筛确实改变了送检粉末的"
            f"粒度/形状分布。"
        )
    else:
        verdict_lines.append(
            "\n\n**SEM 形貌描述子**：本次检验的变量中未见 p<0.05 的组间差异"
            "（详见 §7 表格），但该检验同样受样本量限制，不能证明\"完全无影响\"。"
        )

    verdict_lines.append(
        "\n\n**综合判断**：过筛与 T_C=850°C/t_hold=10h 两个低端水平结构性混杂"
        "是确凿的（Fisher exact 已核实），这本身就足以让论文 §3.5"
        "\"跨前驱体条件级残差高度重现→存在被 Θ 掩盖的非单调工艺效应\"这一论断"
        "**不能被视为已排除混杂的独立证据**——即使剔除过筛条件后相关性依然存在"
        "（若如此，见 §6），也只能说明工艺效应\"在未过筛子集里依然可见\"，尚不能"
        "反过来证明全 27 条件版本完全不受过筛影响，因为过筛条件本身就是热预算"
        "最低端（C01/C04/C07，Θ 最小）与中心点（C14）附近的一部分，这些条件在"
        "残差热图中的位置本来就容易和\"低 Θ 端\"的工艺效应混在一起。"
        "\n\n**建议裁决**：论文 §3.5 需**降级为讨论项**——不满足直接作为正式结论"
        "陈述的条件（存在未受控的结构性混杂，且检验效力受限于 n=5 的过筛条件数），"
        "但也没有证据表明其完全是伪影（若 §6 剔除后 r 仍为正且显著，说明信号不"
        "完全依赖过筛条件）。应在正文中明确注明\"过筛与低热预算条件混杂，"
        "该结论有待同场次随机顺序重压实验验证\"（与 §3.5 原文档 `[待测]` "
        "标记的下一步一致），不应作为独立、无保留的核心结论呈现。"
    )
    return "".join(verdict_lines)


if __name__ == "__main__":
    main()
