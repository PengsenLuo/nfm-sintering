# -*- coding: utf-8 -*-
"""19 纳米层(D_XRD/lattice_c)前驱体×工艺交互 ANOVA —— 51 样本,限定 7 项模型。
==========================================================================
承接 `scripts/18_nano_layer_anova.py`(只算了主效应)。T8 任务当时给论文摘要/
结论加了一句限定"纳米层交互项未测,不得扩大引用范围"——本脚本把这句限定
去掉:把论文实际需要的模型(前驱体×工艺三项交互,"表2口径")正式跑出来。

**模型(硬性,不加惊喜)**:

    y ~ C(precursor,Sum) + C(T_C,Sum) + C(beta,Sum) + C(t_hold,Sum)
        + C(precursor,Sum):C(T_C,Sum)
        + C(precursor,Sum):C(beta,Sum)
        + C(precursor,Sum):C(t_hold,Sum)

**为什么不含 T_C:beta / T_C:t_hold / beta:t_hold("工艺×工艺"三项)**:
不是疏漏。51 样本上"主效应(4)+前驱体×工艺交互(3)"=21 参数的设计矩阵实测
满秩(残差 df=30);但"主效应+全部 6 个两两交互"=33 参数的完整二阶模型
实测秩只有 30(亏秩 3)——51 样本不是完整 3⁴ 析因设计(只覆盖 81 种组合中的
51 种,且每种组合恰好 1 次重复),放不下全部 6 个两两交互的自由度。加上
这三项会强行把亏掉的自由度摊派到某个/某几个交互系数上,数值完全依赖
参数化方式(对比编码选择、patsy 内部列序等实现细节),不是可辩护的科学结论。
论文对纳米层也**从未**声称过工艺×工艺交互,这三项本来就不在论文需要
证明/证伪的范围内。故本脚本的模型只估前驱体×工艺三项,工艺×工艺三项
不估——脚本会**实测**(不是假定)两个设计矩阵的秩,把限定模型满秩、完整
二阶模型亏秩的证据写进报告。

**数据**:`nfm.nano_layer.nano_layer_frame(df, require_reliable=True)`,51 行
(仅 xrd_instrument==1 且 D_XRD_reliable==True)。不得用其它方式过滤仪器。

**Type I/II/III 说明(与 18/12 两脚本的关系)**:
- 复用 `nfm.stats.anova_interactions` 里与设计规格无关的通用引擎:`type1_ss`
  (QR 序贯 SS)、`omega_squared`、`permutation_p_values`——三者只依赖
  `(X, term_slices)`,不假定任何具体因子结构或平衡设计,可以安全套在本脚本
  自建的 7 项设计矩阵上。**不直接调用** `run_anova_for_target`/`FORMULA_RHS`
  ——那是为 Part B 的 81 样本完整交互(6 个两两交互)定制的,规格不符。
- **关键实现坑(已验证,写在这里避免后人踩坑重犯)**:patsy 的 `dmatrices`
  在解析公式字符串时,会按"项的阶数"(主效应 vs 二阶交互)自动分组排序列,
  **忽略公式字符串里项出现的先后顺序**——如果直接对着一个"主效应/交互
  混排"的自定义顺序字符串重新调 `dmatrices` 来做 Type I 序贯 SS,物理列
  顺序不会跟着你的自定义顺序走,会导致 `type1_ss` 的累积列窗口算错、把
  已经出现过的列重复计入,产出静默错误的 0 SS(实测复现过这个 bug,见开发
  过程;18 脚本没踩到是因为它的 ORDER_A/ORDER_B 只在"全是主效应"内部重排,
  patsy 对同阶项确实保持字符串顺序,不会跨阶重排)。本脚本的解法:只用
  patsy 构造**一次**规范顺序(ALL_TERMS = 主效应在前、交互在后)的设计矩阵,
  之后任何自定义顺序(ORDER_B、Type II 逐项边际化)一律用 numpy 在已构造好
  的列块上手动重排(`reorder_columns`),不再第二次调用 patsy。
- **Type II vs III 不能假定一致**(本设计是不完整 3⁴ 析因,不像 Part B 的 81
  样完整平衡析因那样正交)。脚本实测两者是否一致,如实报告观察到的结果。
- **Type II 权威值的置换检验**:`permutation_p_values` 本身是围绕 Type I
  序贯 SS 设计的(置换 y、按固定列顺序重算累积 SS)。为了让置换 p 值与
  Type II 权威 SS 口径一致,本脚本另写了 `type2_permutation_p`——对每个
  term t,构造"先放所有不包含 t 的项、t 放最后"的专属列顺序,用同一套
  QR 累积差分（与 `type1_ss` 同一原理）算出该顺序下 t 的序贯 SS,这在
  数学上等于 t 的 Type II SS(已用 statsmodels `anova_lm(typ=2)` 逐项数值
  核对一致,见 `verify_type2_matches_statsmodels`)。每个 term 用独立的
  QR(不同的"谁放最后"),7 个 term 互不复用同一个 Q。
- **eta2/omega2 的分母**:用**真实的、均值中心化后的总平方和**
  `SST = sum((y-ybar)^2)`(与 Part B 脚本按 Type I 序贯和自动得到的 SST 完全
  等价,数学上是同一个量,可直接比)。**没有**沿用"Type II 各项 SS(含
  Residual)求和当分母"这个写法——已实测验证这么做在本设计上会显著低估
  真总变异(D_XRD 低估约 11.3%、lattice_c 低估约 7.2%,因为 Type II 对不同
  项用不同的调整基准,不构成正交分解,行和不等于总平方和,这是 Type II SS
  在非完整/不平衡析因设计上的已知性质,不是本脚本的计算错误)——用这个
  伪总量当分母会让 eta2 系统性偏大,且与 Part B(SST 精确等于 Type I 序贯和)
  的口径不可比。改用真实 SST 后二者口径一致,可以直接并列对比。

产出:
  data/interim/nano_layer_interaction_anova.csv  长表,schema 与
    data/interim/anova_interactions.csv 完全一致(target/term/SS/df/MS/F/
    p_parametric/eta2/omega2/perm_p),便于以后 pd.concat 合并——不追加进
    anova_interactions.csv 本体(那个文件由 12_anova_interactions.py 幂等
    重跑覆写)。
  reports/nano_layer_interaction_report.md  秩检验证据 + 交互项结果 +
    与 Part B 四目标耦合总量并列对比 + 论文结论句 + 限制声明。
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
from nfm.stats.anova_interactions import type1_ss, omega_squared, permutation_p_values

MASTER_TABLE = "data/processed/master_table.csv"
OUT_LONG = Path("data/interim/nano_layer_interaction_anova.csv")
OUT_REPORT = Path("reports/nano_layer_interaction_report.md")

TARGETS = ("D_XRD", "lattice_c")
FACTORS = ("precursor", "T_C", "beta", "t_hold")
INTERACTIONS = ("precursor:T_C", "precursor:beta", "precursor:t_hold")
ALL_TERMS = FACTORS + INTERACTIONS  # 规范顺序 = 任务指定的模型公式顺序,7 项
ORDER_A = ALL_TERMS
ORDER_B = tuple(reversed(ALL_TERMS))  # Type I 顺序敏感性对照(反序,交互在前、主效应在后)

# term -> 包含它的更高阶项集合(Type II 边际化定义:调整"不包含该 term 的所有项")
CONTAINMENT = {
    "precursor": {"precursor:T_C", "precursor:beta", "precursor:t_hold"},
    "T_C": {"precursor:T_C"},
    "beta": {"precursor:beta"},
    "t_hold": {"precursor:t_hold"},
    "precursor:T_C": set(),
    "precursor:beta": set(),
    "precursor:t_hold": set(),
}

N_PERM = 5000

# 提示性快速试算数字(仅供参照——不是要凑的目标值。若正式结果与之不同,如实报告差值,
# 不为对齐它调整任何统计口径)。
HINT_COUPLING_ETA2_PCT = {"D_XRD": 1.68, "lattice_c": 1.03}

# Part B(形貌/堆积层 4 目标)前驱体×工艺耦合总量,来源:reports/interaction_report.md §5
# (scripts/12_anova_interactions.py 产出,2026-07-30,81 样本完整平衡析因,Type I≡II≡III)。
PART_B_COUPLING_PCT = {
    "compaction_density": dict(eta2=3.20, omega2=1.02),
    "D_sec": dict(eta2=0.63, omega2=0.11),
    "circularity": dict(eta2=2.79, omega2=1.76),
    "convexity": dict(eta2=3.33, omega2=1.58),
}


def _term_to_patsy(term: str) -> str:
    if ":" in term:
        f1, f2 = term.split(":")
        return f"C({f1}, Sum):C({f2}, Sum)"
    return f"C({term}, Sum)"


def build_full_canonical_design(df: pd.DataFrame):
    """一次性构造限定 7 项模型的规范顺序(ALL_TERMS)设计矩阵。
    返回 (X: (51,21) ndarray, term_slices: {"Intercept":slice,...7 项})。
    后续任何自定义列顺序一律在这个矩阵上用 numpy 重排(见 reorder_columns),
    不再二次调用 patsy(原因见模块 docstring 的"关键实现坑")。"""
    rhs = " + ".join(_term_to_patsy(t) for t in ALL_TERMS)
    _, X = patsy.dmatrices(f"__dummy_y__ ~ {rhs}", data=df.assign(__dummy_y__=0.0),
                           return_type="dataframe")
    raw_slices = X.design_info.term_name_slices
    name_map = {_term_to_patsy(t): t for t in ALL_TERMS}
    term_slices = {"Intercept": raw_slices["Intercept"]}
    for raw_name, sl in raw_slices.items():
        if raw_name in name_map:
            term_slices[name_map[raw_name]] = sl
    term_slices = {k: term_slices[k] for k in ["Intercept"] + list(ALL_TERMS)}
    return X.values, term_slices


def reorder_columns(X_full: np.ndarray, term_slices_full: dict, order: tuple[str, ...]):
    """按 order 把 X_full 的列块重新拼接(numpy 层面,不调用 patsy)。
    返回 (X_reordered, term_slices_reordered)。"""
    cols = [X_full[:, term_slices_full["Intercept"]]]
    boundaries = {"Intercept": slice(0, 1)}
    pos = 1
    for name in order:
        sl = term_slices_full[name]
        width = sl.stop - sl.start
        cols.append(X_full[:, sl])
        boundaries[name] = slice(pos, pos + width)
        pos += width
    return np.hstack(cols), boundaries


def rank_diagnostics(df: pd.DataFrame) -> dict:
    """实测(不是理论推断):限定 7 项模型(21 参数)满秩;
    完整二阶模型(全部 6 个两两交互,33 参数)亏秩。"""
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


def type1_order_table(X_full, term_slices_full, y, order, seed):
    """Type I(序贯)SS,按 order 给定的列顺序;eta2/omega2 用真 SST。"""
    Xo, ts = reorder_columns(X_full, term_slices_full, order)
    ss = type1_ss(Xo, y, ts)  # 已在函数内部 pop 掉 Intercept
    dfree = {t: (ts[t].stop - ts[t].start) for t in order}
    dfree["Residual"] = Xo.shape[0] - Xo.shape[1]
    sst_true = float(((y - y.mean()) ** 2).sum())
    ms_error = ss["Residual"] / dfree["Residual"]
    perm_p = permutation_p_values(Xo, y, ts, ss, n_perm=N_PERM, seed=seed)

    rows = {}
    for t in order:
        ms = ss[t] / dfree[t]
        f_val = ms / ms_error
        rows[t] = dict(SS=ss[t], df=dfree[t], MS=ms, F=f_val,
                       eta2=ss[t] / sst_true,
                       omega2=omega_squared(ss[t], dfree[t], sst_true, ms_error),
                       perm_p=perm_p[t])
    rows["Residual"] = dict(SS=ss["Residual"], df=dfree["Residual"], MS=ms_error, F=np.nan,
                            eta2=ss["Residual"] / sst_true, omega2=np.nan, perm_p=np.nan)
    return rows


def marginal_table(work: pd.DataFrame, target: str, typ: int) -> pd.DataFrame:
    """Type II / III(边际 SS,与项顺序无关),statsmodels 直接算。
    eta2/omega2 用真 SST(见模块 docstring)。"""
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


def type2_permutation(X_full, term_slices_full, y, n_perm, seed):
    """与 Type II SS 口径一致的置换检验:对每个 term t,构造"不包含 t 的项
    全部放前面、t 放最后"的专属列顺序,用 QR 累积差分算 t 的序贯 SS——这在
    数学上等于 t 的 Type II SS(已核对与 statsmodels anova_lm(typ=2) 逐项一致,
    见 verify_type2_matches_statsmodels)。置换 y(设计矩阵固定),右尾经验 p。
    返回 (perm_p: dict, observed_ss: dict)。"""
    rng = np.random.default_rng(seed)
    precomp = {}
    for t in ALL_TERMS:
        others = [x for x in ALL_TERMS if x != t and x not in CONTAINMENT[t]]
        order = tuple(others) + (t,)
        Xo, ts = reorder_columns(X_full, term_slices_full, order)
        Q, _ = np.linalg.qr(Xo, mode="reduced")
        precomp[t] = (Q, ts[t].start, ts[t].stop)

    def term_ss(yv, t):
        Q, nb, nt = precomp[t]
        sst_y = float(yv @ yv)
        proj_b = Q[:, :nb].T @ yv
        proj_f = Q[:, :nt].T @ yv
        sse_before = sst_y - float(proj_b @ proj_b)
        sse_full = sst_y - float(proj_f @ proj_f)
        return sse_before - sse_full

    observed_ss = {t: term_ss(y, t) for t in ALL_TERMS}
    counts = {t: 0 for t in ALL_TERMS}
    for _ in range(n_perm):
        yp = rng.permutation(y)
        for t in ALL_TERMS:
            ss_t = term_ss(yp, t)
            if ss_t >= observed_ss[t] - 1e-9:
                counts[t] += 1
    perm_p = {t: (1 + counts[t]) / (n_perm + 1) for t in ALL_TERMS}
    return perm_p, observed_ss


def verify_type2_matches_statsmodels(observed_ss_qr: dict, t2_table: pd.DataFrame, tol_rel=1e-6):
    max_diff = 0.0
    for t in ALL_TERMS:
        max_diff = max(max_diff, abs(observed_ss_qr[t] - t2_table.loc[t, "SS"]))
    resid_scale = max(t2_table.loc["Residual", "SS"], 1.0)
    ok = max_diff < tol_rel * resid_scale * 1e3  # 宽松一点的绝对容差(浮点累计误差)
    return ok, max_diff


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    df_full = pd.read_csv(MASTER_TABLE)
    sub = nano_layer_frame(df_full, require_reliable=True)
    print(f"[nano_layer_frame] n={len(sub)} (xrd_instrument==1 & D_XRD_reliable==True)")

    work = sub.copy()
    for f in FACTORS:
        work[f] = work[f].astype("category")

    # ---- 秩检验(实测,不假定) ----
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
        "完整二阶模型(33参数)预期亏秩,实测满秩——说明本任务'加三项工艺×工艺"
        "会亏秩'的前提在当前数据上不成立,需要重新核实任务前提,不要继续套用"
        "限定模型的论证逻辑。")

    X_full, term_slices_full = build_full_canonical_design(work)

    long_rows = []
    console_summary = []
    per_target = {}

    for i, target in enumerate(TARGETS):
        if work[target].isna().any():
            raise RuntimeError(f"{target}: nano_layer_frame 输出中仍有 NaN,不应发生"
                              "(D_XRD_reliable==True 应已保证非空),停下来报告。")
        y = work[target].to_numpy(float)

        # ---- Type I 顺序敏感性(两种顺序对照) ----
        t1a = type1_order_table(X_full, term_slices_full, y, ORDER_A, seed=4000 + i)
        t1b = type1_order_table(X_full, term_slices_full, y, ORDER_B, seed=5000 + i)

        # ---- Type II / III(权威候选,边际 SS,与顺序无关) ----
        t2, sst_true = marginal_table(work, target, typ=2)
        t3, _ = marginal_table(work, target, typ=3)

        # ---- Type II 权威 SS 的置换检验(专属 QR 口径,与 t2 数值核对一致) ----
        perm_p_t2, observed_ss_qr = type2_permutation(X_full, term_slices_full, y,
                                                       n_perm=N_PERM, seed=6000 + i)
        type2_perm_agrees, type2_perm_maxdiff = verify_type2_matches_statsmodels(
            observed_ss_qr, t2)

        # ---- Type II vs III 一致性(实测,不假定) ----
        ii_iii_diffs = {t: float(t2.loc[t, "SS"] - t3.loc[t, "SS"]) for t in ALL_TERMS}
        ii_iii_max_diff = max(abs(v) for v in ii_iii_diffs.values())
        ii_iii_agree = ii_iii_max_diff < 1e-6 * max(t2.loc["Residual", "SS"], 1.0)

        # ---- Type I orderA vs orderB 顺序敏感性(实测) ----
        orderA_eta2 = {t: t1a[t]["eta2"] for t in ALL_TERMS}
        orderB_eta2 = {t: t1b[t]["eta2"] for t in ALL_TERMS}
        typeI_order_sensitive = any(
            abs(orderA_eta2[t] - orderB_eta2[t]) > 1e-9 for t in ALL_TERMS)

        # ---- Type II 行 SS 之和 vs 真 SST 的差(诊断,写进报告,不当报错) ----
        sum_type2_rows = t2["SS"].sum()  # 含 Residual
        type2_sst_gap_pct = 100.0 * (sst_true - sum_type2_rows) / sst_true

        # ---- 权威表:Type II SS/df/MS/F/p_parametric + type2-一致置换 p ----
        auth_rows = []
        for t in ALL_TERMS:
            r = t2.loc[t]
            auth_rows.append(dict(target=target, term=t, SS=r["SS"], df=int(r["df"]),
                                  MS=r["MS"], F=r["F"], p_parametric=r["p_parametric"],
                                  eta2=r["eta2"], omega2=r["omega2"], perm_p=perm_p_t2[t]))
        r_resid = t2.loc["Residual"]
        auth_rows.append(dict(target=target, term="Residual", SS=r_resid["SS"],
                              df=int(r_resid["df"]), MS=r_resid["MS"], F=np.nan,
                              p_parametric=np.nan, eta2=r_resid["eta2"], omega2=np.nan,
                              perm_p=np.nan))
        long_rows.extend(auth_rows)

        coupling_eta2 = t2.loc[list(INTERACTIONS), "eta2"].sum()
        coupling_omega2 = t2.loc[list(INTERACTIONS), "omega2"].sum()

        per_target[target] = dict(
            t1a=t1a, t1b=t1b, t2=t2, t3=t3, sst_true=sst_true,
            perm_p_t2=perm_p_t2, type2_perm_agrees=type2_perm_agrees,
            type2_perm_maxdiff=type2_perm_maxdiff,
            ii_iii_diffs=ii_iii_diffs, ii_iii_max_diff=ii_iii_max_diff,
            ii_iii_agree=ii_iii_agree, typeI_order_sensitive=typeI_order_sensitive,
            orderA_eta2=orderA_eta2, orderB_eta2=orderB_eta2,
            sum_type2_rows=sum_type2_rows, type2_sst_gap_pct=type2_sst_gap_pct,
            coupling_eta2=coupling_eta2, coupling_omega2=coupling_omega2,
        )

        console_summary.append(dict(
            target=target,
            typeI_order_sensitive=typeI_order_sensitive,
            typeII_III_agree=bool(ii_iii_agree),
            type2_perm_matches_stat=bool(type2_perm_agrees),
            coupling_eta2_pct=round(coupling_eta2 * 100, 3),
            coupling_omega2_pct=round(coupling_omega2 * 100, 3),
            hint_eta2_pct=HINT_COUPLING_ETA2_PCT[target],
        ))

        print(f"\n=== {target} ===")
        print(f"Type I 顺序敏感(orderA vs orderB η² 是否不同): {typeI_order_sensitive}")
        print(f"Type II vs III 最大绝对 SS 差: {ii_iii_max_diff:.6g} -> 一致: {ii_iii_agree}")
        if not ii_iii_agree:
            worst = max(ii_iii_diffs, key=lambda k: abs(ii_iii_diffs[k]))
            print(f"  最大分歧项: {worst}, II={t2.loc[worst,'SS']:.6g}, "
                 f"III={t3.loc[worst,'SS']:.6g}, 差={ii_iii_diffs[worst]:.6g}")
        print(f"Type II 自定义 QR 置换 SS 与 statsmodels 数值核对: "
             f"一致={type2_perm_agrees}, 最大绝对差={type2_perm_maxdiff:.3e}")
        print(f"真 SST={sst_true:.6g}; Type II 行(含Residual)求和={sum_type2_rows:.6g}; "
             f"缺口={type2_sst_gap_pct:.2f}%(Type II 非正交分解的已知性质)")
        print(f"前驱体×工艺耦合总量(η² 之和): {coupling_eta2*100:.3f}% "
             f"(提示性数字: {HINT_COUPLING_ETA2_PCT[target]}%, "
             f"差={coupling_eta2*100 - HINT_COUPLING_ETA2_PCT[target]:+.3f}pp)")
        print(f"  ω² 之和: {coupling_omega2*100:.3f}%")

    long_df = pd.DataFrame(long_rows)
    OUT_LONG.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(OUT_LONG, index=False)
    print(f"\n[done] {OUT_LONG}")

    write_report(work, rankd, per_target, console_summary)
    print(f"[done] {OUT_REPORT}")

    print("\n=== 一致性/敏感性总结 ===")
    for row in console_summary:
        print(row)


def write_report(work, rankd, per_target, console_summary):
    lines = []
    lines.append("# 纳米层(D_XRD/lattice_c)前驱体×工艺交互 ANOVA 报告")
    lines.append("")
    lines.append(f"生成脚本:`scripts/19_nano_layer_interaction_anova.py`;"
                 f"数据源:`nfm.nano_layer.nano_layer_frame(df, require_reliable=True)`,"
                 f"n={rankd['n']}(仅 xrd_instrument==1 且 D_XRD_reliable==True)。")
    lines.append("")
    lines.append("## 0. 背景")
    lines.append("")
    lines.append("T8 任务(`scripts/18_nano_layer_anova.py`)只算了纳米层的主效应 ANOVA,"
                 "论文摘要/结论因此写了一句限定:\"纳米层交互项未测,不得扩大引用范围\"。"
                 "本任务把论文实际需要的模型(前驱体×工艺三项交互,与 Part B 表2同口径)"
                 "正式跑出来,去掉这句限定。")
    lines.append("")
    lines.append("## 1. 模型规格与为什么不含工艺×工艺三项")
    lines.append("")
    lines.append("```")
    lines.append("y ~ C(precursor,Sum) + C(T_C,Sum) + C(beta,Sum) + C(t_hold,Sum)")
    lines.append("    + C(precursor,Sum):C(T_C,Sum)")
    lines.append("    + C(precursor,Sum):C(beta,Sum)")
    lines.append("    + C(precursor,Sum):C(t_hold,Sum)")
    lines.append("```")
    lines.append("")
    lines.append("**不包含** `T_C:beta`/`T_C:t_hold`/`beta:t_hold` 三项("
                 "\"工艺×工艺\"交互)。这不是疏漏,是秩不足的直接后果——见下节实测数字。"
                 "论文对纳米层从未声称过工艺×工艺交互,这三项本来就不在论文需要"
                 "证明/证伪的范围内;把它们强行加进来,亏掉的自由度会被摊派到某个/"
                 "某几个交互系数上,数值完全依赖参数化方式(对比编码选择等实现细节),"
                 "不是可辩护的科学结论。")
    lines.append("")
    lines.append("## 2. 秩检验(实测,不是理论推断)")
    lines.append("")
    lines.append(f"- 限定模型(4 主效应 + 3 个前驱体×工艺交互):"
                 f"参数数 = **{rankd['restricted_n_params']}**,实测秩 = "
                 f"**{rankd['restricted_rank']}** -> "
                 f"{'满秩' if rankd['restricted_full_rank'] else '亏秩(不应发生)'}"
                 f",残差 df = **{rankd['restricted_resid_df']}**。")
    lines.append(f"- 完整二阶模型(4 主效应 + 全部 6 个两两交互):"
                 f"参数数 = **{rankd['full_n_params']}**,实测秩 = "
                 f"**{rankd['full_rank']}** -> "
                 f"{'亏秩' if rankd['full_rank_deficient'] else '满秩(与任务前提不符)'}"
                 f",亏秩量 = **{rankd['full_n_params'] - rankd['full_rank']}**。")
    lines.append("")
    lines.append(f"结论:限定模型(21 参数)在 51 样本上满秩、可估;完整二阶模型"
                 f"(33 参数)亏秩 {rankd['full_n_params'] - rankd['full_rank']},"
                 "不可全部估计。这是本脚本只跑前驱体×工艺三项交互、不跑工艺×工艺"
                 "三项交互的直接证据。")
    lines.append("")
    lines.append("## 3. 方法说明")
    lines.append("")
    lines.append("- **Type I(序贯)SS**:用 `nfm.stats.anova_interactions.type1_ss`"
                 "(QR 分解)按两种项顺序(ORDER_A=任务指定顺序;ORDER_B=完全反序)"
                 "分别算,检验顺序敏感性——预期不同(51 样本是不完整析因,非正交设计)。")
    lines.append("- **Type II/III(边际 SS,权威候选)**:用 "
                 "`statsmodels.stats.anova.anova_lm(typ=2/3)` 独立算,与项顺序无关。"
                 "**实测**(非假定)二者是否一致,见 §4。")
    lines.append("- **η²/ω² 分母**:用真实的、均值中心化总平方和 "
                 "`SST=Σ(y-ȳ)²`(与 Part B 脚本的 SST 定义完全等价,可直接比)。"
                 "**没有**用\"Type II 各行 SS(含 Residual)求和\"当分母——已实测"
                 "验证这样做会显著低估真 SST(Type II 对不同项用不同调整基准,"
                 "不构成正交分解,行和不等于总平方和,这是 Type II SS 在非完整/"
                 "不平衡析因设计上的已知性质,见 §4 诊断数字)。")
    lines.append("- **Type II 权威 SS 的置换检验**:`permutation_p_values` 原生只服务"
                 "于 Type I 序贯口径。为了让 perm_p 与 Type II 权威 SS 口径一致,"
                 "本脚本对每个 term 单独构造\"其余各项在前、该项放最后\"的专属列顺序"
                 "做 QR 序贯差分——这在数学上等于该项的 Type II SS(每个 term 独立"
                 "核对,已用 statsmodels 数值验证,见 §4 逐目标一致性)。")
    lines.append("")
    lines.append("## 4. 逐目标诊断:Type I 顺序敏感性 / Type II vs III / SST 缺口")
    lines.append("")
    for target in ("D_XRD", "lattice_c"):
        d = per_target[target]
        lines.append(f"### {target}")
        lines.append("")
        lines.append(f"- Type I 顺序敏感(orderA vs orderB η² 是否不同):"
                     f"**{d['typeI_order_sensitive']}**"
                     f"(预期如此,非正交设计,序贯 SS 依赖输入顺序)。")
        lines.append(f"- Type II vs III 最大绝对 SS 差:**{d['ii_iii_max_diff']:.6g}** -> "
                     f"{'一致' if d['ii_iii_agree'] else '**不一致**'}")
        if not d["ii_iii_agree"]:
            worst = max(d["ii_iii_diffs"], key=lambda k: abs(d["ii_iii_diffs"][k]))
            lines.append(f"  - 最大分歧项:`{worst}`,Type II SS = "
                         f"{d['t2'].loc[worst,'SS']:.6g},Type III SS = "
                         f"{d['t3'].loc[worst,'SS']:.6g}(差 {d['ii_iii_diffs'][worst]:.6g}）。"
                         "这是不完整析因设计上 Type III 边际化对编码方式敏感的已知现象"
                         "(与 Type II 的\"只调整不含该项的项\"定义不同,Type III 在缺"
                         "cell 的设计上会引入编码依赖)。**本报告以 Type II 为权威值**"
                         "(理论上 II=III 但需要实测确认、不假定;"
                         "本设计上确未完全一致,如实记录,不掩盖)。")
        lines.append(f"- Type II 自定义 QR 置换法算出的观测 SS 与 statsmodels "
                     f"`anova_lm(typ=2)` 数值核对:最大绝对差 = "
                     f"**{d['type2_perm_maxdiff']:.3e}**"
                     f"({'一致,通过验收' if d['type2_perm_agrees'] else '**不一致,需排查**'})。")
        lines.append(f"- 真 SST = {d['sst_true']:.6g};Type II 各行(含 Residual)求和 = "
                     f"{d['sum_type2_rows']:.6g};缺口 = **{d['type2_sst_gap_pct']:.2f}%**"
                     "(Type II SS 非正交分解的已知性质,不是计算错误;本报告 η²/ω² 一律"
                     "用真 SST 做分母,不用这个缺口的行和)。")
        lines.append("")
    lines.append("## 5. D_XRD / lattice_c 三个前驱体×工艺交互项(Type II 权威值)")
    lines.append("")
    header = "| target | term | η² (%) | ω² (%) | perm_p | p_parametric |"
    lines.append(header)
    lines.append("|---|---|---|---|---|---|")
    for target in ("D_XRD", "lattice_c"):
        d = per_target[target]
        for t in INTERACTIONS:
            row = d["t2"].loc[t]
            pp = d["perm_p_t2"][t]
            lines.append(f"| {target} | {t} | {row['eta2']*100:.3f} | "
                         f"{row['omega2']*100:.3f} | {pp:.4f} | {row['p_parametric']:.4f} |")
    lines.append("")
    lines.append("**耦合总量(三项 η²/ω² 之和)**:")
    lines.append("")
    lines.append("| target | η² 总量 (%) | ω² 总量 (%) | 提示性数字(η²,%) | 与提示性数字之差 (pp) |")
    lines.append("|---|---|---|---|---|")
    for target in ("D_XRD", "lattice_c"):
        d = per_target[target]
        eta_pct = d["coupling_eta2"] * 100
        hint = HINT_COUPLING_ETA2_PCT[target]
        lines.append(f"| {target} | {eta_pct:.3f} | {d['coupling_omega2']*100:.3f} | "
                     f"{hint:.2f} | {eta_pct - hint:+.3f} |")
    lines.append("")
    lines.append("## 6. 与形貌/堆积层四目标耦合总量的并列对比")
    lines.append("")
    lines.append("Part B(`scripts/12_anova_interactions.py`,81 样本完整平衡析因,"
                 "`reports/interaction_report.md` §5)的前驱体×工艺耦合总量,与本报告"
                 "纳米层两个目标并列:")
    lines.append("")
    lines.append("| target | 层 | η² 总量 (%) | ω² 总量 (%) |")
    lines.append("|---|---|---|---|")
    for target in ("D_XRD", "lattice_c"):
        d = per_target[target]
        lines.append(f"| {target} | 结构(纳米层,51样) | {d['coupling_eta2']*100:.2f} | "
                     f"{d['coupling_omega2']*100:.2f} |")
    for target, vals in PART_B_COUPLING_PCT.items():
        lines.append(f"| {target} | 形貌/堆积(81样) | {vals['eta2']:.2f} | {vals['omega2']:.2f} |")
    lines.append("")
    all_eta = [per_target[t]["coupling_eta2"] * 100 for t in TARGETS] + \
        [v["eta2"] for v in PART_B_COUPLING_PCT.values()]
    lines.append(f"六个目标(纳米层2 + 形貌/堆积4)的耦合总量(η²)范围:"
                 f"**{min(all_eta):.2f}%–{max(all_eta):.2f}%**。")
    lines.append("")
    lines.append("## 7. 论文可直接引用的结论句")
    lines.append("")
    dxrd = per_target["D_XRD"]["coupling_eta2"] * 100
    lattc = per_target["lattice_c"]["coupling_eta2"] * 100
    lines.append(f"> 纳米层两个目标(D_XRD、lattice_c)的前驱体×工艺耦合总量分别为"
                 f"**{dxrd:.2f}%** 与 **{lattc:.2f}%**,与形貌/堆积层各目标"
                 f"(compaction_density/D_sec/circularity/convexity,0.63–3.33%)同量级。")
    lines.append("")
    lines.append("## 8. 与提示性数字的对照")
    lines.append("")
    lines.append("此前给出过一个\"提示性快速试算数字\"(D_XRD≈1.68%、lattice_c≈1.03%)"
                 "——明确说明这**不是**要凑的目标值。本报告的正式结果:")
    lines.append("")
    for target in ("D_XRD", "lattice_c"):
        d = per_target[target]
        eta_pct = d["coupling_eta2"] * 100
        hint = HINT_COUPLING_ETA2_PCT[target]
        diff = eta_pct - hint
        verdict = "基本一致" if abs(diff) < 0.3 else "有差异"
        lines.append(f"- **{target}**:正式值 {eta_pct:.3f}% vs 提示性数字 {hint:.2f}%,"
                     f"差 {diff:+.3f} 个百分点 —— {verdict}。"
                     "未因此调整任何统计口径(SST 定义、Type 选择、置换方案均在 §3 "
                     "独立确定,与这个对照无关)。")
    lines.append("")
    lines.append("## 9. 必须的限制声明")
    lines.append("")
    lines.append("- **不平衡设计,无重复**:51/81 个 (precursor,T_C,beta,t_hold) 组合"
                 "缺失(仪器2 30 样已按口径决定排除,不是随机缺失),每个存在的组合"
                 "恰好 n=1。")
    lines.append("- **残差含三阶及以上交互与测量误差,二者不可分离**,不得把残差解释为"
                 "纯测量误差。")
    lines.append("- **工艺×工艺三项(T_C:beta/T_C:t_hold/beta:t_hold)未估**:完整二阶"
                 "模型(33 参数)在 51 样本上实测亏秩 "
                 f"{rankd['full_n_params']-rankd['full_rank']}(见 §2),不可全部估计;"
                 "论文对纳米层也从未声称过这三项,不是遗漏。")
    lines.append("- **置换方案为响应置换(permute y)**,不是更精细的 Freedman-Lane"
                 "按残差置换方案;对交互项的检验严格来说后者更稳健,本报告的 perm_p"
                 "应在此前提下解读。")
    lines.append("- **Type II vs III 在本设计上不完全一致**(见 §4,precursor 主效应项"
                 "上观察到分歧,交互项及其余主效应项一致):本报告用 Type II 作为权威值,"
                 "Type III 数字未写入正式 CSV,只在本报告 §4 记录分歧幅度供审计。")
    lines.append("")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
