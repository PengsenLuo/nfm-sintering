# -*- coding: utf-8 -*-
"""
anova_interactions.py —— 完整二阶交互 ANOVA(形貌/堆积层,Part B)
====================================================================
模型:y ~ precursor + T_C + beta + t_hold + 全部 6 个两两交互(Sum 编码,
均为分类变量,各 3 水平)。81 样本 = 完整平衡析因设计(每个
(precursor,T_C,beta,t_hold) 组合恰好 n=1),4 因子间两两正交
(build_design 用 Sum 对比编码,实测任意两个 term 的设计矩阵子块内积
恒为 0)——这是 Type I(序贯)/II/III 三种 SS 在此设计下必然给出相同结果的
根本原因,而不是巧合;本模块用一次 QR 分解算 Type I SS 作为权威数值,
调用方应在报告里用 statsmodels 的 typ=1/2/3 独立复核一致性(而不是假定)。

n=1 无重复 → 残差 = 三阶及以上交互 + 纯测量误差,二者不可分离,不得把
残差解释为测量误差。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import patsy

FACTORS = ("precursor", "T_C", "beta", "t_hold")

TERM_ORDER = (
    "precursor", "T_C", "beta", "t_hold",
    "precursor:T_C", "precursor:beta", "precursor:t_hold",
    "T_C:beta", "T_C:t_hold", "beta:t_hold",
)

# 前驱体×工艺耦合总量(论文表2头号数字,"两个尺度可独立优化"论断的直接证据)
PRECURSOR_PROCESS_TERMS = ("precursor:T_C", "precursor:beta", "precursor:t_hold")

FORMULA_RHS = (
    "C(precursor, Sum) + C(T_C, Sum) + C(beta, Sum) + C(t_hold, Sum)"
    " + C(precursor, Sum):C(T_C, Sum) + C(precursor, Sum):C(beta, Sum)"
    " + C(precursor, Sum):C(t_hold, Sum)"
    " + C(T_C, Sum):C(beta, Sum) + C(T_C, Sum):C(t_hold, Sum)"
    " + C(beta, Sum):C(t_hold, Sum)"
)


def build_design(df: pd.DataFrame):
    """构造 Sum 编码设计矩阵。df 须含 FACTORS 四列(建议先转 category dtype)。

    返回 (X: (n,p) ndarray, term_slices: {"Intercept": slice, "precursor": slice, ...},
    column_names: list[str])。term_slices 的键顺序为 Intercept + TERM_ORDER。
    """
    _, X = patsy.dmatrices(f"__dummy_y__ ~ {FORMULA_RHS}",
                           data=df.assign(__dummy_y__=0.0), return_type="dataframe")
    raw_slices = X.design_info.term_name_slices
    # patsy 用 "C(precursor, Sum)"/"C(precursor, Sum):C(T_C, Sum)" 命名;
    # 映射回本模块用的简洁 term 名(TERM_ORDER)。
    name_map = {f"C({f}, Sum)": f for f in FACTORS}
    for t in TERM_ORDER:
        if ":" in t:
            f1, f2 = t.split(":")
            name_map[f"C({f1}, Sum):C({f2}, Sum)"] = t
    term_slices = {"Intercept": raw_slices["Intercept"]}
    for raw_name, sl in raw_slices.items():
        if raw_name in name_map:
            term_slices[name_map[raw_name]] = sl
    term_slices = {k: term_slices[k] for k in ["Intercept"] + list(TERM_ORDER)}
    return X.values, term_slices, list(X.columns)


def type1_ss(X: np.ndarray, y: np.ndarray, term_slices: dict) -> dict:
    """序贯(Type I)SS,按 term_slices 的列顺序(Intercept 在前)累积用 QR 分解。

    对完整平衡正交设计(见模块docstring),Type I == Type II == Type III,
    调用方应独立用 statsmodels 交叉验证这个假设是否在具体数据上成立,
    不要凭空信任。
    """
    Q, _ = np.linalg.qr(X, mode="reduced")
    y = np.asarray(y, dtype=float)
    sst_y = float(y @ y)

    cum_cols = 0
    sse_cum = {}  # 累积到每个 term 末尾时的残差平方和
    boundaries = ["Intercept"] + [t for t in term_slices if t != "Intercept"]
    for name in boundaries:
        sl = term_slices[name]
        cum_cols = max(cum_cols, sl.stop)
        proj = Q[:, :cum_cols].T @ y
        sse_cum[name] = sst_y - float(proj @ proj)

    ss = {}
    prev_sse = sst_y  # 空模型(0列)的 SSE = ||y||^2
    order = ["Intercept"] + [t for t in term_slices if t != "Intercept"]
    for name in order:
        ss[name] = prev_sse - sse_cum[name]
        prev_sse = sse_cum[name]
    ss.pop("Intercept", None)
    ss["Residual"] = sse_cum[order[-1]]
    return ss


def omega_squared(ss_effect: float, df_effect: int, ss_total: float, ms_error: float) -> float:
    """ω² = (SS_effect - df_effect·MS_error) / (SS_total + MS_error),负值截断为 0
    (小样本无偏估计,负值代表效应不可区分于噪声,不代表"负方差")。"""
    val = (ss_effect - df_effect * ms_error) / (ss_total + ms_error)
    return max(val, 0.0)


def permutation_p_values(X: np.ndarray, y: np.ndarray, term_slices: dict,
                         observed_ss: dict, n_perm: int, seed: int) -> dict:
    """置换方案:置换响应 y(设计矩阵 X 固定不变),对每次置换重算 Type I SS
    (因设计正交,置换后的 Type I 仍等于该次置换下的 Type II/III)。
    双侧检验的备择假设是"方差份额异常大",故用单尾(右尾)经验 p:
    p = (1 + #{permuted_SS >= observed_SS}) / (n_perm + 1)。

    注:这是响应置换(permute y),不是按 term 置换设计矩阵的更精细方案
    (如 Freedman-Lane);对交互项的检验严格来说后者更稳健,但更复杂,
    本函数采用前者并在此声明所用置换方案。
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(X, mode="reduced")

    order = [t for t in term_slices if t != "Intercept"]
    intercept_stop = term_slices["Intercept"].stop
    cum_bounds = []
    cum_cols = intercept_stop
    for name in order:
        cum_cols = max(cum_cols, term_slices[name].stop)
        cum_bounds.append(cum_cols)

    counts = {name: 0 for name in order}
    y = np.asarray(y, dtype=float)
    for _ in range(n_perm):
        yp = rng.permutation(y)
        sst_y = float(yp @ yp)
        # 先扣掉截距(与 type1_ss 一致),否则第一个 term 的 SS 会被截距的
        # SS(=n·mean²,量级可能远大于任何 term)污染。
        proj0 = Q[:, :intercept_stop].T @ yp
        prev_sse = sst_y - float(proj0 @ proj0)
        for name, cc in zip(order, cum_bounds):
            proj = Q[:, :cc].T @ yp
            sse = sst_y - float(proj @ proj)
            ss_term = prev_sse - sse
            if ss_term >= observed_ss[name] - 1e-12:
                counts[name] += 1
            prev_sse = sse

    return {name: (1 + counts[name]) / (n_perm + 1) for name in order}


def run_anova_for_target(df: pd.DataFrame, target: str, n_perm: int = 5000,
                         seed: int = 0) -> dict:
    """对单个 target 跑完整流程:Type I SS(权威值)+ 独立用 statsmodels 验证
    Type I/II/III 一致性 + omega2/eta2 + 置换检验。

    返回 dict(table=DataFrame[term-> SS/df/MS/F/p/eta2/omega2/perm_p],
              type_agreement=bool, type_max_abs_diff=float)。
    """
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    work = df.copy()
    for f in FACTORS:
        work[f] = work[f].astype("category")

    X, term_slices, _ = build_design(work)
    y = work[target].to_numpy(float)
    ss = type1_ss(X, y, term_slices)

    dfree = {t: (term_slices[t].stop - term_slices[t].start) for t in TERM_ORDER}
    dfree["Residual"] = X.shape[0] - X.shape[1]

    model = smf.ols(f"{target} ~ {FORMULA_RHS}", data=work).fit()
    tabs = {typ: anova_lm(model, typ=typ)["sum_sq"] for typ in (1, 2, 3)}
    ref = tabs[1]
    ss_from_sm = {term: float(ref.iloc[i]) for i, term in enumerate(TERM_ORDER)}
    ss_from_sm["Residual"] = float(ref["Residual"])

    max_diff = 0.0
    for typ in (2, 3):
        # typ=3 表比 typ=1/2 多一行 "Intercept"(typ=1/2 不含),先去掉再按位置对齐,
        # 否则会把 Intercept 的 SS(量级远大于任何 term)错位比成某个 term 的 SS。
        t = tabs[typ]
        t = t.drop(index="Intercept", errors="ignore")
        for i, term in enumerate(TERM_ORDER):
            max_diff = max(max_diff, abs(float(t.iloc[i]) - ss[term]))
        max_diff = max(max_diff, abs(float(t["Residual"]) - ss["Residual"]))
    type_agreement = bool(max_diff < 1e-6 * max(ss["Residual"], 1.0))

    ss_total = sum(ss.values())
    ms_error = ss["Residual"] / dfree["Residual"]

    perm_p = permutation_p_values(X, y, term_slices, ss, n_perm=n_perm, seed=seed)

    rows = []
    for t in TERM_ORDER:
        ms = ss[t] / dfree[t]
        f_val = ms / ms_error
        rows.append(dict(term=t, SS=ss[t], df=dfree[t], MS=ms, F=f_val,
                         eta2=ss[t] / ss_total,
                         omega2=omega_squared(ss[t], dfree[t], ss_total, ms_error),
                         perm_p=perm_p[t]))
    rows.append(dict(term="Residual", SS=ss["Residual"], df=dfree["Residual"],
                     MS=ms_error, F=np.nan, eta2=ss["Residual"] / ss_total,
                     omega2=np.nan, perm_p=np.nan))
    table = pd.DataFrame(rows).set_index("term")

    # 用 statsmodels 的 F 检验 p 值补充(参数化 p,与置换 p 并列报告)
    table["p_parametric"] = np.nan
    for i, t in enumerate(TERM_ORDER):
        table.loc[t, "p_parametric"] = float(anova_lm(model, typ=1)["PR(>F)"].iloc[i])

    return dict(table=table, type_agreement=type_agreement, type_max_abs_diff=max_diff,
               ss_from_statsmodels=ss_from_sm)
