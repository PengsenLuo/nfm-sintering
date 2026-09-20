# -*- coding: utf-8 -*-
"""
scripts/17_shape_descriptor_full_analysis.py
=============================================
T7 —— 形状描述子完整分析 + "形状复杂化→压实密度下降"因果链证伪强化

四部分:
  §1 D_f / solidity / elongation / roughness 四个形状描述子的完整二阶交互
     ANOVA —— 直接复用 `nfm.stats.anova_interactions.run_anova_for_target`,
     不重写 SS 计算逻辑。
  §2 分前驱体(S/M/L)Spearman(descriptor, Theta) + 95% bootstrap CI,
     复核既有参考值(scripts/02i_shape_descriptors.py report() 打印值)。
  §3 circularity / D_f 对 compaction_density 的偏相关(控制 Theta,
     以及 Theta+precursor 两种控制集)+ TOST 等价性检验,把"没测到"
     升级为"测到了但小于工程阈值"(如果数据支持)。
  §4 roughness 三组 Spearman 的 Fisher-z 异质性检验(Cochran's Q),
     核查"跨前驱体一致的通用形貌纹理"这一表述是否站得住。

数据源:
  data/interim/sem_shape_descriptors_summary.csv (81/81,D_f/solidity/
    elongation/roughness + precursor/Theta/T_C/compaction_density/circularity)
  data/processed/master_table.csv (81 行,补 beta/t_hold/condition_id,
    并核对与 summary 文件重叠列完全一致 —— 不假设合并键干净,代码里做断言)

产出:
  data/interim/shape_descriptor_full_analysis.csv         全部计算过程长表
  data/interim/anova_interactions_shape_descriptors.csv   §1 结果,
      与 data/interim/anova_interactions.csv 同 schema,**另存一份**
      (不追加进原文件,理由见脚本内注释与报告 §0)
  reports/shape_descriptor_full_analysis_report.md
"""
import _bootstrap  # noqa

import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from nfm.stats.anova_interactions import TERM_ORDER, run_anova_for_target

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "data/interim/sem_shape_descriptors_summary.csv"
MASTER = ROOT / "data/processed/master_table.csv"
OUT_LONG = ROOT / "data/interim/shape_descriptor_full_analysis.csv"
OUT_ANOVA = ROOT / "data/interim/anova_interactions_shape_descriptors.csv"
OUT_REPORT = ROOT / "reports/shape_descriptor_full_analysis_report.md"

DESCRIPTORS = ("D_f", "solidity", "elongation", "roughness")
CAUSAL_DESCRIPTORS = ("circularity", "D_f")
PRECURSORS = ("S", "M", "L")
N_PERM = 5000
N_BOOT = 5000
SEED = 2026
ALPHA = 0.05

# 既有参考值:scripts/02i_shape_descriptors.py report() 的打印输出
# (抄录用于核实复现一致性,不是"真值")。
PAPER_REF_RHO = {
    ("D_f", "S"): 0.695, ("D_f", "M"): 0.020, ("D_f", "L"): 0.189,
    ("solidity", "S"): -0.669, ("solidity", "M"): -0.150, ("solidity", "L"): -0.407,
    ("elongation", "S"): -0.524,
    ("roughness", "S"): 0.682, ("roughness", "M"): 0.546, ("roughness", "L"): 0.535,
}
PAPER_REF_P = {
    ("D_f", "S"): "<0.001", ("D_f", "M"): 0.92, ("D_f", "L"): 0.35,
    ("solidity", "S"): "<0.001", ("solidity", "M"): 0.46, ("solidity", "L"): 0.035,
    ("elongation", "S"): 0.005,
    ("roughness", "S"): "<0.005", ("roughness", "M"): "<0.005", ("roughness", "L"): "<0.005",
}

# TOST 等价界值。压实密度全域跨度 3.04-3.56 g/cm3(0.52),取其 5%≈0.026 g/cm3
# 作为"每 1 SD 标准化形状描述子残差对应的密度效应"的等价界。同时对照
# 另一项分析已用过的 4%≈0.02 g/cm3 —— 但那个阈值的
# 物理量纲是"两个精炼样本压实密度均值之差本身"(g/cm3),这里的量纲是
# "回归斜率:每 1 SD 标准化形状残差 → 密度变化"(g/cm3 / SD),两者量纲不同、
# 不能直接套用同一个数字,分别独立取值,报告里两个阈值都算一遍作敏感性对照,
# 主结论以 5% 为准。
DENSITY_RANGE = 3.56 - 3.04
DELTA_PRIMARY = round(0.05 * DENSITY_RANGE, 6)      # 0.026
DELTA_SENSITIVITY = round(0.04 * DENSITY_RANGE, 6)  # 0.0208 (与 T4 口径对照)


# --------------------------------------------------------------------------
# 数据加载与合并校验
# --------------------------------------------------------------------------
def load_data():
    d = pd.read_csv(SUMMARY)
    mt = pd.read_csv(MASTER)

    assert d["sample_id"].duplicated().sum() == 0, "summary 文件 sample_id 有重复"
    assert mt["sample_id"].duplicated().sum() == 0, "master_table sample_id 有重复"
    missing_in_mt = set(d["sample_id"]) - set(mt["sample_id"])
    missing_in_d = set(mt["sample_id"]) - set(d["sample_id"])
    assert not missing_in_mt, f"summary 里有 sample_id 在 master_table 找不到: {missing_in_mt}"
    assert not missing_in_d, f"master_table 里有 81 样本以外未在 summary 出现: {missing_in_d}"

    merged = d.merge(
        mt[["sample_id", "beta", "t_hold", "condition_id",
            "precursor", "Theta", "T_C", "compaction_density", "circularity"]],
        on="sample_id", suffixes=("", "_mt"), how="inner",
    )
    assert len(merged) == 81, f"合并后应为 81 行,实际 {len(merged)}"

    # 重叠列必须完全一致,不假设合并键干净 —— 逐列核实。
    overlap_cols = ["precursor", "Theta", "T_C", "compaction_density", "circularity"]
    for c in overlap_cols:
        mismatch = merged[c].astype(str) != merged[f"{c}_mt"].astype(str)
        assert mismatch.sum() == 0, f"合并后列 {c} 与 master_table 不一致,共 {mismatch.sum()} 行"
        merged = merged.drop(columns=[f"{c}_mt"])

    # 设计平衡性:81 个 (precursor,T_C,beta,t_hold) 组合各恰好 n=1(ANOVA 前提)。
    n_cells = merged.groupby(["precursor", "T_C", "beta", "t_hold"]).size()
    assert len(n_cells) == 81 and (n_cells == 1).all(), \
        f"设计不是完整平衡析因(唯一格数={len(n_cells)},计数唯一值={sorted(n_cells.unique())})"

    # condition_id: 27 个条件各恰好 3 行(S/M/L 各一)——bootstrap 分块重采样的前提。
    n_cond = merged.groupby("condition_id").size()
    assert len(n_cond) == 27 and (n_cond == 3).all(), \
        f"condition_id 结构异常(条件数={len(n_cond)},计数唯一值={sorted(n_cond.unique())})"
    for cid, sub in merged.groupby("condition_id"):
        assert sorted(sub["precursor"].tolist()) == ["L", "M", "S"], \
            f"条件 {cid} 的前驱体不是 S/M/L 各一"

    for d_ in DESCRIPTORS:
        assert merged[d_].notna().sum() == 81, f"{d_} 非 81/81 完整,不满足平衡设计前提"

    return merged


def _stable_seed(*parts) -> int:
    """确定性种子(不用内置 `hash()`——Python 3 默认对 str 做每进程随机化的
    哈希扰动[hash randomization],同一份代码两次运行会得到不同种子、
    不同 bootstrap 结果,不可复现。用 zlib.crc32 替代,同一输入永远同一输出。
    """
    s = "|".join(str(p) for p in parts).encode("utf-8")
    return zlib.crc32(s) & 0xFFFFFFFF


def _row(rows, **kw):
    base = dict(section=np.nan, target=np.nan, group=np.nan, term=np.nan,
                metric=np.nan, value=np.nan, ci_low=np.nan, ci_high=np.nan,
                se=np.nan, n=np.nan, note="")
    base.update(kw)
    rows.append(base)


# --------------------------------------------------------------------------
# §1 完整二阶交互 ANOVA
# --------------------------------------------------------------------------
def section1_anova(df, rows):
    anova_wide_rows = []
    results = {}
    for i, target in enumerate(DESCRIPTORS):
        res = run_anova_for_target(df, target, n_perm=N_PERM, seed=3000 + i)
        results[target] = res
        table = res["table"]

        _row(rows, section="anova_diagnostic", target=target, metric="type_agreement",
             value=float(res["type_agreement"]), n=81,
             note="Type I/II/III SS 是否一致(平衡正交设计下理论上必须一致)")
        _row(rows, section="anova_diagnostic", target=target, metric="type_max_abs_diff",
             value=res["type_max_abs_diff"], n=81)

        for term in list(TERM_ORDER) + ["Residual"]:
            r = table.loc[term]
            for metric in ("SS", "df", "MS", "F", "p_parametric", "eta2", "omega2", "perm_p"):
                _row(rows, section="anova_interaction", target=target, term=term,
                     metric=metric, value=r.get(metric, np.nan), n=81)
            anova_wide_rows.append(dict(
                target=target, term=term, SS=r["SS"], df=r["df"], MS=r["MS"], F=r["F"],
                p_parametric=r.get("p_parametric", np.nan), eta2=r["eta2"],
                omega2=r["omega2"], perm_p=r["perm_p"],
            ))

    if not all(results[t]["type_agreement"] for t in DESCRIPTORS):
        bad = [t for t in DESCRIPTORS if not results[t]["type_agreement"]]
        raise RuntimeError(f"Type I/II/III SS 在以下形状描述子上不一致: {bad} —— "
                           "这 4 个描述子设计完全平衡、无缺失,没有'不一致'的正当理由,"
                           "停下来报告,不要继续套用本脚本结论。")

    anova_wide = pd.DataFrame(anova_wide_rows)
    anova_wide.to_csv(OUT_ANOVA, index=False)
    print(f"[section1] Type I/II/III 一致性: 全部通过。写出 {OUT_ANOVA}")
    return results, anova_wide


# --------------------------------------------------------------------------
# §2 分前驱体 Spearman(descriptor, Theta) + bootstrap CI
# --------------------------------------------------------------------------
def _bootstrap_spearman(theta, y, n_boot, seed):
    """逐条件重采样(此处退化为逐样本重采样,见下方推理注释)。"""
    rng = np.random.default_rng(seed)
    n = len(theta)
    boots = np.empty(n_boot)
    idx = np.arange(n)
    for b in range(n_boot):
        s = rng.choice(idx, size=n, replace=True)
        r, _ = stats.spearmanr(theta[s], y[s])
        boots[b] = r
    return boots


def section2_spearman_bootstrap(df, rows):
    """
    重采样单位推理(必须先想清楚,见任务书原话):
    本设计 27 个条件各恰好 3 个前驱体样本(S/M/L),同炉共烧、共享热历史,
    不是相互独立观测 —— 这是"按条件整体重采样"这条建议的出发点。

    但本节做的是**分前驱体**的组内相关:S 组本身就是 27 个条件各贡献
    唯一 1 个 S 样本(M、L 组同理)。也就是说,"对 27 个条件重采样、
    保持该条件内三个样本捆绑"这个操作,限制到单个前驱体组内部时,
    等价于"对这 27 个 (Theta, descriptor) 点各自独立重采样"——因为
    每个条件在该组内只贡献一个点,不存在"捆绑"这个操作能起作用的
    结构(捆绿的对象——同一条件下另外两个前驱体的样本——根本不在
    这个子集里)。只有在跨组比较(把 S/M/L 三组的残差放在一起做统计)
    时,同一条件贡献的三个非独立观测才会同时出现在同一个统计量里,
    这时才必须用整条件重采样去保持它们的相关结构 —— 这正是 §3(偏相关
    /TOST,合并全部 81 样本、按 precursor 分组回归)里实际采用整条件
    块重采样的原因,与本节形成对照。
    故本节按条件重采样,但由于组内退化为逐样本重采样,直接用逐样本
    有放回重采样实现(数学上完全等价,写法更直接)。
    """
    for target in DESCRIPTORS:
        group_rhos = {}
        for grp in PRECURSORS:
            sub = df[df["precursor"] == grp][["Theta", target]].dropna()
            n = len(sub)
            theta = sub["Theta"].to_numpy(float)
            y = sub[target].to_numpy(float)
            r, p = stats.spearmanr(theta, y)
            group_rhos[grp] = r

            boots = _bootstrap_spearman(theta, y, N_BOOT, seed=_stable_seed(target, grp, SEED))
            ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])

            _row(rows, section="spearman_theta", target=target, group=grp, metric="rho",
                 value=r, ci_low=ci_lo, ci_high=ci_hi, se=float(np.std(boots, ddof=1)),
                 n=n, note=f"n_boot={N_BOOT}, 重采样单位=条件(该组内退化为逐样本,见代码注释)")
            _row(rows, section="spearman_theta", target=target, group=grp,
                 metric="p_value_asymptotic", value=p, n=n)

            ref_key = (target, grp)
            if ref_key in PAPER_REF_RHO:
                ref_r = PAPER_REF_RHO[ref_key]
                diff = r - ref_r
                _row(rows, section="spearman_theta", target=target, group=grp,
                     metric="paper_reference_rho", value=ref_r, n=n,
                     note=f"paper_p={PAPER_REF_P[ref_key]}")
                _row(rows, section="spearman_theta", target=target, group=grp,
                     metric="diff_vs_paper_reference", value=diff, n=n,
                     note="重算值-参考值; |diff|<0.01 视为精确复现")

            ci_excludes_zero = not (ci_lo <= 0 <= ci_hi)
            _row(rows, section="spearman_theta", target=target, group=grp,
                 metric="ci_excludes_zero", value=float(ci_excludes_zero), n=n,
                 note="CI 是否跨零(跨零=不能排除'无效应';不跨零=有方向性证据,"
                      "即便渐近 p 值不显著)")
        yield target, group_rhos


# --------------------------------------------------------------------------
# §3 偏相关(控制 Theta[,+precursor])+ TOST 等价性检验
# --------------------------------------------------------------------------
def _design_matrix(data, controls):
    n = len(data)
    cols = [np.ones(n)]
    if "Theta" in controls:
        cols.append(data["Theta"].to_numpy(float))
    if "precursor" in controls:
        dummies = pd.get_dummies(data["precursor"], drop_first=True).to_numpy(float)
        cols.append(dummies)
    return np.column_stack(cols)


def _residualize(y, X):
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ coef


def _partial_r_and_slope(data, x_col, y_col, controls):
    """残差化偏相关:x/y 各自对 controls 回归取残差(手写最小二乘,不走
    statsmodels 公式解析,只为 bootstrap 循环里反复调用的速度),残差的
    Pearson 相关 = 偏相关系数。slope = 把 x 残差标准化(z-score)后对 y
    残差做 OLS 斜率,单位 = 'y 每变化 1 个标准化 x 残差 SD'(与 δ 的
    定义口径一致);因 x 残差标准化后方差(ddof=1)恒为 1,斜率退化为
    协方差本身,用 np.cov 直接算,数值上与 polyfit 等价但更快。
    """
    X = _design_matrix(data, controls)
    rx = _residualize(data[x_col].to_numpy(float), X)
    ry = _residualize(data[y_col].to_numpy(float), X)
    sx = rx.std(ddof=1)
    if sx == 0 or not np.isfinite(sx):
        return np.nan, np.nan
    rx_std = (rx - rx.mean()) / sx
    r = float(np.corrcoef(rx, ry)[0, 1])
    slope = float(np.cov(rx_std, ry, ddof=1)[0, 1])
    return r, slope


def _resample_by_condition(df, rng):
    conds = df["condition_id"].unique()
    sampled = rng.choice(conds, size=len(conds), replace=True)
    parts = [df[df["condition_id"] == c] for c in sampled]
    return pd.concat(parts, ignore_index=True)


def _tost_from_boot(boot_vals, delta):
    boot_vals = boot_vals[np.isfinite(boot_vals)]
    p_upper = float(np.mean(boot_vals >= delta))   # H0: slope>=delta(不等价) vs H1: slope<delta
    p_lower = float(np.mean(boot_vals <= -delta))  # H0: slope<=-delta vs H1: slope>-delta
    p_tost = max(p_upper, p_lower)
    ci90_lo, ci90_hi = np.percentile(boot_vals, [5, 95])
    equivalent = bool(ci90_lo > -delta and ci90_hi < delta)
    assert equivalent == (p_tost < ALPHA), "90% CI 判据与经验 p 判据应等价(TOST 与 90% CI 对偶关系)"
    return dict(p_tost=p_tost, p_upper=p_upper, p_lower=p_lower,
               ci90_lo=ci90_lo, ci90_hi=ci90_hi, equivalent=equivalent)


def section3_partial_corr_tost(df, rows):
    control_sets = {
        "Theta_only": ["Theta"],
        "Theta_plus_precursor": ["Theta", "precursor"],
    }
    summary = {}
    for target in CAUSAL_DESCRIPTORS:
        # 零阶(不控制任何变量)与分组相关,作为对照基线,复核论文旧表述的出发点。
        sub_all = df[[target, "compaction_density"]].dropna()
        r0, p0 = stats.spearmanr(sub_all[target], sub_all["compaction_density"])
        _row(rows, section="causal_baseline", target=target, metric="spearman_zero_order",
             value=r0, n=len(sub_all), note="不控制任何变量的全局 Spearman(对照基线)")
        max_p_within_group = -np.inf
        for grp in PRECURSORS:
            sub = df[df["precursor"] == grp][[target, "compaction_density"]].dropna()
            rg, pg = stats.spearmanr(sub[target], sub["compaction_density"])
            max_p_within_group = max(max_p_within_group, pg)
            _row(rows, section="causal_baseline", target=target, group=grp,
                 metric="spearman_within_group", value=rg, n=len(sub))
        _row(rows, section="causal_baseline", target=target,
             metric="max_p_within_group", value=max_p_within_group,
             note="论文表述'组内相关均不显著'对应的最大 p(复核用,非本节主结论)")

        for cs_name, controls in control_sets.items():
            data = df[[target, "compaction_density", "Theta", "precursor", "condition_id"]].dropna()
            r_point, slope_point = _partial_r_and_slope(data, target, "compaction_density", controls)

            rng = np.random.default_rng(_stable_seed(target, cs_name, SEED))
            boot_r = np.empty(N_BOOT)
            boot_slope = np.empty(N_BOOT)
            n_fail = 0
            for b in range(N_BOOT):
                bd = _resample_by_condition(data, rng)
                try:
                    rr, ss = _partial_r_and_slope(bd, target, "compaction_density", controls)
                except Exception:
                    rr, ss = np.nan, np.nan
                if not np.isfinite(rr):
                    n_fail += 1
                boot_r[b] = rr
                boot_slope[b] = ss

            r_ci = np.nanpercentile(boot_r, [2.5, 97.5])
            slope_ci95 = np.nanpercentile(boot_slope, [2.5, 97.5])

            _row(rows, section="partial_corr", target=target, group=cs_name,
                 metric="partial_r", value=r_point, ci_low=r_ci[0], ci_high=r_ci[1],
                 n=len(data), note=f"控制变量={controls}; bootstrap 单位=condition_id(27块); "
                                    f"n_boot={N_BOOT}(失败{n_fail}次,已跳过)")
            _row(rows, section="partial_corr", target=target, group=cs_name,
                 metric="slope_gcm3_per_sd", value=slope_point,
                 ci_low=slope_ci95[0], ci_high=slope_ci95[1], n=len(data),
                 note="每 1 SD 标准化形状描述子残差对应的 compaction_density 变化(g/cm3)")

            for delta_name, delta in (("primary_5pct", DELTA_PRIMARY),
                                      ("sensitivity_4pct", DELTA_SENSITIVITY)):
                tost = _tost_from_boot(boot_slope, delta)
                _row(rows, section="tost", target=target, group=cs_name, term=delta_name,
                     metric="delta_gcm3", value=delta, n=len(data))
                _row(rows, section="tost", target=target, group=cs_name, term=delta_name,
                     metric="p_tost", value=tost["p_tost"], n=len(data))
                _row(rows, section="tost", target=target, group=cs_name, term=delta_name,
                     metric="ci90_low", value=tost["ci90_lo"], n=len(data))
                _row(rows, section="tost", target=target, group=cs_name, term=delta_name,
                     metric="ci90_high", value=tost["ci90_hi"], n=len(data))
                _row(rows, section="tost", target=target, group=cs_name, term=delta_name,
                     metric="equivalent_at_alpha05", value=float(tost["equivalent"]), n=len(data),
                     note="True=90%CI完全落在[-delta,+delta]内,拒绝'存在≥delta的效应'")
                summary[(target, cs_name, delta_name)] = dict(
                    r=r_point, slope=slope_point, slope_ci95=slope_ci95, **tost)
    return summary


# --------------------------------------------------------------------------
# §4 roughness 三组 Spearman 的 Fisher-z 异质性检验(Cochran's Q)
# --------------------------------------------------------------------------
def section4_heterogeneity(df, rows):
    target = "roughness"
    zs, ws, ns, rs = {}, {}, {}, {}
    for grp in PRECURSORS:
        sub = df[df["precursor"] == grp][["Theta", target]].dropna()
        n = len(sub)
        r, _ = stats.spearmanr(sub["Theta"], sub[target])
        z = np.arctanh(r)
        # Spearman 的 Fisher-z 方差用 1.06/(n-3) 校正因子(Fieller, Hartley &
        # Pearson 1957 的标准近似;Pearson r 用 1/(n-3),Spearman rho 额外
        # 乘 1.06,是文献里最常用的解析近似,非本任务自创)。
        var_z = 1.06 / (n - 3)
        zs[grp], ws[grp], ns[grp], rs[grp] = z, 1.0 / var_z, n, r

    z_arr = np.array([zs[g] for g in PRECURSORS])
    w_arr = np.array([ws[g] for g in PRECURSORS])
    z_bar = float(np.sum(w_arr * z_arr) / np.sum(w_arr))
    Q = float(np.sum(w_arr * (z_arr - z_bar) ** 2))
    dfree = len(PRECURSORS) - 1
    p_Q = float(1.0 - stats.chi2.cdf(Q, dfree))
    r_pooled = float(np.tanh(z_bar))
    se_zbar = float(1.0 / np.sqrt(np.sum(w_arr)))
    pooled_ci = np.tanh([z_bar - 1.96 * se_zbar, z_bar + 1.96 * se_zbar])

    for grp in PRECURSORS:
        _row(rows, section="heterogeneity", target=target, group=grp, metric="rho",
             value=rs[grp], n=ns[grp])
        _row(rows, section="heterogeneity", target=target, group=grp, metric="fisher_z",
             value=zs[grp], n=ns[grp])
        _row(rows, section="heterogeneity", target=target, group=grp, metric="weight",
             value=ws[grp], n=ns[grp], note="w=(n-3)/1.06")
    _row(rows, section="heterogeneity", target=target, metric="Q_statistic", value=Q,
         note=f"df={dfree}, Cochran's Q(等价于 Fisher-z meta-analysis 异质性检验)")
    _row(rows, section="heterogeneity", target=target, metric="Q_df", value=float(dfree))
    _row(rows, section="heterogeneity", target=target, metric="Q_p_value", value=p_Q)
    _row(rows, section="heterogeneity", target=target, metric="pooled_r", value=r_pooled,
         ci_low=pooled_ci[0], ci_high=pooled_ci[1],
         note="固定效应加权平均(仅当 Q 检验不显著、同质性假设成立时才有解释力)")

    return dict(Q=Q, df=dfree, p=p_Q, r_pooled=r_pooled, pooled_ci=pooled_ci,
               group_rho=rs, group_n=ns)


# --------------------------------------------------------------------------
# 报告生成
# --------------------------------------------------------------------------
def write_report(spearman_results, causal_summary, het):
    lines = []
    lines.append("# 形状描述子完整分析报告(T7:D_f/solidity/elongation/roughness)")
    lines.append("")
    lines.append(f"生成脚本:`scripts/17_shape_descriptor_full_analysis.py`;"
                 f"数据源:`{SUMMARY.relative_to(ROOT).as_posix()}` 合并 "
                 f"`{MASTER.relative_to(ROOT).as_posix()}`(81 样本,合并键校验见脚本 "
                 "`load_data()`,重叠列逐位一致,无需假设合并键干净但已实测干净)。")
    lines.append("")
    lines.append("## §0 产出格式选择说明")
    lines.append("")
    lines.append("- **§1 的 ANOVA 结果没有追加进 `data/interim/anova_interactions.csv`**,"
                 "而是另存为同 schema 的 `data/interim/anova_interactions_shape_descriptors.csv`。"
                 "理由:(1) `anova_interactions.csv` 由 `scripts/12_anova_interactions.py` "
                 "生成且该脚本是幂等重跑式的(每次运行整份覆写),若直接追加,"
                 "未来任何人重跑 `12_...py` 都会静默抹掉本任务追加的 4 个新 target,"
                 "属于隐藏的维护陷阱;(2) 本任务与另一并行任务(T6)同批次开展,"
                 "任务说明要求各自独立 commit、不要动对方产出,不修改共享产出文件"
                 "是更安全的选择;(3) 两个文件 schema 完全一致"
                 "(target/term/SS/df/MS/F/p_parametric/eta2/omega2/perm_p),"
                 "后续如需合并只需 `pd.concat` 两个 csv,不损失任何可追溯性。")
    lines.append("- 全部计算过程(§1-4)统一落在长表 "
                 "`data/interim/shape_descriptor_full_analysis.csv`"
                 "(列:section/target/group/term/metric/value/ci_low/ci_high/se/n/note),"
                 "§1 的 ANOVA 结果在此长表中以 `section=anova_interaction` 熔断展开"
                 "(与另存的宽表数值完全一致,可交叉核对)。")
    lines.append("")

    lines.append("## §1 完整二阶交互 ANOVA")
    lines.append("")
    lines.append("模型规格与 T3/Part B(`scripts/12_anova_interactions.py`)完全相同:"
                 "`y ~ precursor + T_C + beta + t_hold + 全部两两交互`(Sum 编码,81 样本"
                 "完整平衡析因设计)。Type I/II/III SS 一致性已实测通过(见脚本运行日志"
                 "与 `anova_diagnostic` 行,`type_max_abs_diff` 全部 <1e-6·SS_residual)。")
    lines.append("")
    lines.append("主效应 η²(%)(完整表见 "
                 "`data/interim/anova_interactions_shape_descriptors.csv`):")
    lines.append("")
    header = "| target | precursor | T_C | beta | t_hold | Residual |"
    lines.append(header)
    lines.append("|---|---|---|---|---|---|")
    for target, res in spearman_results["anova_res"].items():
        t = res["table"]
        cells = [f"{t.loc[m, 'eta2']*100:.2f}" for m in ("precursor", "T_C", "beta", "t_hold")]
        cells.append(f"{t.loc['Residual', 'eta2']*100:.2f}")
        lines.append(f"| {target} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("完整交互项(η²/ω²/置换p)见 CSV 产出;不在报告正文重复全部数字,"
                 "避免与 CSV 出现二次抄录误差。")
    lines.append("")

    lines.append("## §2 分前驱体 Spearman(descriptor, Θ)+ 95% bootstrap CI")
    lines.append("")
    lines.append("**重采样单位的推理**(任务书要求先想清楚再决定,不要不假思索套用"
                 "'按条件重采样'的建议):本设计 27 个条件各含 S/M/L 三个样本,"
                 "同炉共烧、共享热历史,不是相互独立观测——这是"
                 "'按条件整体重采样'建议的出发点。但本节是**分前驱体**的组内相关,"
                 "S/M/L 各组内部本来就是 27 个条件各贡献唯一 1 个点,'保持条件内"
                 "三个样本捆绑重采样'这个操作限制到单组内部时,捆绑对象"
                 "(同条件下另外两个前驱体的样本)根本不在该子集里,操作退化为"
                 "对 27 个点的普通有放回重采样——二者数学上完全等价。"
                 "只有做跨组合并分析时,同一条件贡献的三个非独立残差才会同时"
                 "出现在同一个统计量里,那时才必须整条件重采样,这正是 §3(合并"
                 "全部 81 样本做偏相关/TOST)采用整条件块重采样(27 块)的原因,"
                 "与本节形成对照。")
    lines.append("")
    header = "| 描述子 | 组 | n | ρ(重算) | 95%CI | ρ(论文参考) | |diff| | p(渐近) | CI跨零? |"
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for target, grp, r, ci_lo, ci_hi, p, ref_r, diff, excl in spearman_results["table_rows"]:
        ref_str = f"{ref_r:+.3f}" if ref_r is not None else "—"
        diff_str = f"{abs(diff):.3f}" if diff is not None else "—"
        lines.append(f"| {target} | {grp} | 27 | {r:+.3f} | "
                     f"[{ci_lo:+.3f}, {ci_hi:+.3f}] | {ref_str} | {diff_str} | "
                     f"{p:.3g} | {'否(不含0)' if excl else '是(含0)'} |")
    lines.append("")
    n_mismatch = spearman_results["n_mismatch"]
    if n_mismatch == 0:
        lines.append("全部有参考值的组均在 |diff|<0.01 内精确复现,一致。")
    else:
        lines.append(f"**注意**:有 {n_mismatch} 组与参考值 |diff|≥0.01,"
                     "见上表/CSV,可能原因见下文逐项说明。")
    lines.append("")
    lines.append("**M/L 组'未显著'重判**:见上表 CI 跨零列——凡 CI 跨零的组,"
                 "维持'无法排除零效应'(效力不足或确实无效应,数据本身无法二选一);"
                 "凡 CI 不跨零但渐近 p>0.05 的组(若存在),应改判为'方向性证据存在,"
                 "但渐近正态近似在 n=27 下功效不足以达到传统显著性阈值',"
                 "不能笼统写成'没有效应'。具体是否出现这种情况见上表逐行结果。")
    lines.append("")

    lines.append("## §3 '形状复杂化→压实密度下降'因果链证伪强化")
    lines.append("")
    lines.append(f"**TOST 等价界值选取依据**:压实密度全域跨度 "
                 f"3.04–3.56 g/cm³(跨度 {DENSITY_RANGE:.2f} g/cm³)。主结论取其 **5%** "
                 f"≈ **{DELTA_PRIMARY:.4f} g/cm³**(任务书建议值),作为'每 1 SD 标准化"
                 "形状描述子残差对应的密度效应'的工程可忽略阈值——5% 的选择依据:"
                 "在没有压制重复性实验(σ₁,论文§2.2尚未执行)提供测量学阈值的情况下,"
                 "以设计范围的一个小比例作为'工程无关'的保守估计是常见做法(全域跨度"
                 "本身已经是整个 81 样本实验设计能触达的最大密度变化范围,5% 意味着"
                 "'不到实验能探测到的最大变化范围的二十分之一');同时用另一项"
                 f"排序稳健性分析已经用过的 4% ≈ {DELTA_SENSITIVITY:.4f} g/cm³"
                 " 做敏感性对照——注意那个阈值的量纲是原始密度差本身,这里的量纲是回归斜率"
                 "(g/cm³ / 1SD标准化残差),两者物理量纲不同,不能直接套用同一个"
                 "数字,只是同属'全域跨度的一个百分比'这个思路上的对照,不是同一个量。")
    lines.append("")
    header = "| 描述子 | 控制变量 | 偏相关r | 95%CI | 斜率(g/cm³/1SD) | 95%CI |"
    lines.append(header)
    lines.append("|---|---|---|---|---|---|")
    for (target, cs_name), (r, r_ci, slope, slope_ci) in causal_summary["point"].items():
        lines.append(f"| {target} | {cs_name} | {r:+.3f} | [{r_ci[0]:+.3f}, {r_ci[1]:+.3f}] | "
                     f"{slope:+.4f} | [{slope_ci[0]:+.4f}, {slope_ci[1]:+.4f}] |")
    lines.append("")
    lines.append("TOST 结果(δ=主阈值 5% 与敏感性阈值 4% 分别检验):")
    lines.append("")
    header = "| 描述子 | 控制变量 | δ | 90%CI(斜率) | p_TOST | 等价? |"
    lines.append(header)
    lines.append("|---|---|---|---|---|---|")
    for (target, cs_name, delta_name), tost in causal_summary["tost"].items():
        lines.append(f"| {target} | {cs_name} | {tost['delta']:.4f}({delta_name}) | "
                     f"[{tost['ci90_lo']:+.4f}, {tost['ci90_hi']:+.4f}] | "
                     f"{tost['p_tost']:.4f} | {'是' if tost['equivalent'] else '否'} |")
    lines.append("")
    lines.append(causal_summary["verdict_text"])
    lines.append("")

    lines.append("## §4 roughness 跨组一致性异质性检验(Cochran's Q)")
    lines.append("")
    lines.append(f"S/M/L 三组 Spearman(roughness, Θ) 的 Fisher-z 异质性检验:"
                 f"Q = {het['Q']:.3f},df = {het['df']},p = {het['p']:.4f}。"
                 f"固定效应加权平均 ρ_pooled = {het['r_pooled']:+.3f}"
                 f"(95%CI [{het['pooled_ci'][0]:+.3f}, {het['pooled_ci'][1]:+.3f}],"
                 "仅在同质性成立时有解释力)。")
    lines.append("")
    if het["p"] > 0.05:
        lines.append(f"**判定:异质性检验不显著(p={het['p']:.4f}>0.05)**——不能拒绝"
                     "'三组相关系数来自同一潜在效应量、差异仅为抽样波动'这一假设,"
                     "**支持**论文'跨前驱体一致的通用形貌纹理'这一表述,"
                     "无需改写。")
    else:
        lines.append(f"**判定:异质性检验显著(p={het['p']:.4f}≤0.05)**——三组"
                     "Spearman(roughness,Θ) 之间存在统计上可检出的真实差异,"
                     "**不能**再笼统地称其为'跨三组一致的通用形貌纹理',"
                     "论文表述需改写为'三组方向一致但强度不同'一类更保守的说法。")
    lines.append("")

    lines.append("## §5 限制声明")
    lines.append("")
    lines.append("- §2/§3 的 bootstrap CI 均为百分位法(percentile),未做 BCa 偏差校正;"
                 "在 n=27(§2)/n=81 但按 27 条件分块(§3)的样本量下,BCa 与百分位法"
                 "通常差异不大,但严格说这不是最优 CI 构造方式,如后续需要更精确的"
                 "覆盖率可换 BCa。")
    lines.append("- §3 的偏相关基于线性回归残差化(Pearson 相关),对 Θ/precursor 与"
                 "描述子之间的非线性关系不敏感;若关系本质非线性,线性偏相关可能"
                 "低估真实关联,这是本方法的已知局限,未做非参数偏相关的额外交叉验证。")
    lines.append("- §4 的 Spearman Fisher-z 方差近似(1.06/(n-3))是标准解析近似"
                 "(Fieller, Hartley & Pearson 1957),不是精确方差,在 n=27 时已属"
                 "该近似常规适用范围,但仍是近似而非精确解。")
    lines.append("- 本任务不涉及一次颗粒线(D_pri_sem/hier_size_ratio/"
                 "primary_segmentable),未触碰、未翻案该口径的暂停状态。")
    lines.append("")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    df = load_data()
    rows = []

    anova_res, anova_wide = section1_anova(df, rows)

    group_rhos_by_target = {}
    for target, group_rhos in section2_spearman_bootstrap(df, rows):
        group_rhos_by_target[target] = group_rhos

    causal_summary_raw = section3_partial_corr_tost(df, rows)
    het = section4_heterogeneity(df, rows)

    long_df = pd.DataFrame(rows)
    OUT_LONG.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(OUT_LONG, index=False)

    # ---- 组装报告用汇总结构 ----
    table_rows = []
    n_mismatch = 0
    sec2 = long_df[long_df.section == "spearman_theta"]
    for target in DESCRIPTORS:
        for grp in PRECURSORS:
            sub = sec2[(sec2.target == target) & (sec2.group == grp)]
            r = float(sub[sub.metric == "rho"]["value"].iloc[0])
            ci_lo = float(sub[sub.metric == "rho"]["ci_low"].iloc[0])
            ci_hi = float(sub[sub.metric == "rho"]["ci_high"].iloc[0])
            p = float(sub[sub.metric == "p_value_asymptotic"]["value"].iloc[0])
            excl = bool(sub[sub.metric == "ci_excludes_zero"]["value"].iloc[0])
            ref_rows = sub[sub.metric == "paper_reference_rho"]
            if len(ref_rows):
                ref_r = float(ref_rows["value"].iloc[0])
                diff = float(sub[sub.metric == "diff_vs_paper_reference"]["value"].iloc[0])
                if abs(diff) >= 0.01:
                    n_mismatch += 1
            else:
                ref_r, diff = None, None
            table_rows.append((target, grp, r, ci_lo, ci_hi, p, ref_r, diff, excl))

    spearman_results = dict(anova_res=anova_res, table_rows=table_rows, n_mismatch=n_mismatch)

    # causal_summary 组装(point + tost + verdict)
    point = {}
    tost_out = {}
    sec3_pc = long_df[long_df.section == "partial_corr"]
    sec3_tost = long_df[long_df.section == "tost"]
    for target in CAUSAL_DESCRIPTORS:
        for cs_name in ("Theta_only", "Theta_plus_precursor"):
            sub = sec3_pc[(sec3_pc.target == target) & (sec3_pc.group == cs_name)]
            r_row = sub[sub.metric == "partial_r"].iloc[0]
            s_row = sub[sub.metric == "slope_gcm3_per_sd"].iloc[0]
            point[(target, cs_name)] = (
                float(r_row["value"]), (float(r_row["ci_low"]), float(r_row["ci_high"])),
                float(s_row["value"]), (float(s_row["ci_low"]), float(s_row["ci_high"])),
            )
            for delta_name, delta in (("primary_5pct", DELTA_PRIMARY),
                                      ("sensitivity_4pct", DELTA_SENSITIVITY)):
                subt = sec3_tost[(sec3_tost.target == target) & (sec3_tost.group == cs_name)
                                & (sec3_tost.term == delta_name)]
                tost_out[(target, cs_name, delta_name)] = dict(
                    delta=delta,
                    ci90_lo=float(subt[subt.metric == "ci90_low"]["value"].iloc[0]),
                    ci90_hi=float(subt[subt.metric == "ci90_high"]["value"].iloc[0]),
                    p_tost=float(subt[subt.metric == "p_tost"]["value"].iloc[0]),
                    equivalent=bool(subt[subt.metric == "equivalent_at_alpha05"]["value"].iloc[0]),
                )

    all_equivalent_primary = all(
        tost_out[(t, cs, "primary_5pct")]["equivalent"]
        for t in CAUSAL_DESCRIPTORS for cs in ("Theta_only", "Theta_plus_precursor")
    )
    any_equivalent_primary = any(
        tost_out[(t, cs, "primary_5pct")]["equivalent"]
        for t in CAUSAL_DESCRIPTORS for cs in ("Theta_only", "Theta_plus_precursor")
    )
    if all_equivalent_primary:
        verdict = ("**结论:因果链证伪从'弱证据'(未测到显著相关)升级为'强证据'"
                  "(TOST 等价性检验通过)**——circularity/D_f 与 compaction_density "
                  "的偏相关效应量(控制 Theta 及 Theta+precursor 两种规格)在 90% "
                  "置信水平下均落在 ±5% 全域跨度的等价区间内,可正面写入论文作为"
                  "'即使存在关联,效应量也小于工程相关阈值'的证据。")
    elif any_equivalent_primary:
        verdict = ("**结论:部分升级**——部分(描述子×控制变量)组合的偏相关效应量"
                  "通过了 TOST 等价性检验,但并非全部组合都通过,不能笼统宣称"
                  "'证伪已全面升级为强证据',需要在论文里注明具体是哪些控制规格"
                  "下成立(见上表逐行),对未通过的组合如实报告为'效力不足,"
                  "无法排除有意义效应'。")
    else:
        verdict = ("**结论:未能升级**——TOST 未能在任一控制变量规格下确认等价性"
                  "(90% CI 未完全落入 ±δ 区间),说明当前样本量/效应量组合下"
                  "'效应小于工程阈值'这一更强表述**得不到数据支持**,"
                  "应如实报告为'效力不足,无法排除有意义效应',不能拔高为"
                  "等价性证据,继续使用原有'未测到显著相关'(弱证据)表述。")

    causal_summary = dict(point=point, tost=tost_out, verdict_text=verdict)

    write_report(spearman_results, causal_summary, het)

    print(f"[done] {OUT_LONG}")
    print(f"[done] {OUT_ANOVA}")
    print(f"[done] {OUT_REPORT}")
    print("\n=== §2 分前驱体 Spearman(重算 vs 参考) ===")
    for target, grp, r, ci_lo, ci_hi, p, ref_r, diff, excl in table_rows:
        ref_str = f"ref={ref_r:+.3f}" if ref_r is not None else "ref=—"
        print(f"  {target:<12}{grp}  r={r:+.3f} CI=[{ci_lo:+.3f},{ci_hi:+.3f}] "
             f"p={p:.3g} {ref_str}")
    print(f"\n=== §3 TOST 判定(delta=5%={DELTA_PRIMARY:.4f}) ===")
    for (t, cs, dn), v in tost_out.items():
        if dn == "primary_5pct":
            print(f"  {t:<12}{cs:<22} equivalent={v['equivalent']} p_tost={v['p_tost']:.4f}")
    print(f"\n=== §4 roughness 异质性 Q检验 ===  Q={het['Q']:.3f} df={het['df']} p={het['p']:.4f}")


if __name__ == "__main__":
    main()
