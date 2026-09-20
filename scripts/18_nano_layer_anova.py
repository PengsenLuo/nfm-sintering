# -*- coding: utf-8 -*-
"""18 纳米层(D_XRD/lattice_c)主效应 ANOVA —— 51 样本,不平衡设计。
==========================================================================
数据来源:`nfm.nano_layer.nano_layer_frame(df, require_reliable=True)`,
只含 xrd_instrument==1 且 D_XRD_reliable==True 的 51 行。

本脚本首次计算这两列的主效应方差分解(此前没有脚本能复现论文表1的
D_XRD/lattice_c 两行数字)。

**模型**(只含主效应,不含二阶交互——51 样本在 beta/t_hold 上本就不平衡,
加交互项会进一步降低每个 cell 的自由度,超出本任务范围):

    y ~ C(precursor, Sum) + C(T_C, Sum) + C(beta, Sum) + C(t_hold, Sum)

**为什么不能直接调用 `nfm.stats.anova_interactions.run_anova_for_target`**:
该函数的 `FORMULA_RHS` 硬编码了全部 6 个两两交互项(为 Part B 的 81 样本
完整平衡析因设计定制),不满足本任务"只跑主效应"的规格,也不适合套在
本表的不平衡 cell 计数上(交互项会让某些 cell 自由度进一步降低)。
本脚本改为直接复用该模块里与设计规格无关的通用引擎:
`type1_ss`(QR 序贯 SS)、`omega_squared`(ω² 公式)、
`permutation_p_values`(响应置换检验)——这三个函数只依赖
`(X, term_slices)`,不假定平衡设计或某个具体因子结构,可以安全套在
这里的 4 项主效应设计矩阵上;只有 `run_anova_for_target`/`FORMULA_RHS`
本身(硬编码交互项)不能直接用。

**Type I/II/III 在不平衡设计上预期分歧**:与 Part B(81 样完整平衡析因,
设计正交,I≡II≡III 是结构性必然)不同,本设计在 beta(15/12/24)、
t_hold(18/9/24)上不平衡,Type I(序贯)SS 依赖项输入顺序,Type II/III
(边际 SS)与顺序无关、是本报告的权威值。**这是预期,不是断言**——
脚本实测两种项顺序的 Type I 是否真的不同、Type II 与 III 是否真的一致,
如实报告观察到的结果,不预设。

产出:
  data/interim/nano_layer_anova.csv   长表:target×term×ss_type×SS×df×MS×F×p×eta2×omega2
  控制台打印:与论文表1 D_XRD/lattice_c 两行的对照
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
from nfm.stats.anova_interactions import (
    type1_ss, omega_squared, permutation_p_values,
)

MASTER_TABLE = "data/processed/master_table.csv"
OUT_LONG = Path("data/interim/nano_layer_anova.csv")

TARGETS = ("D_XRD", "lattice_c")
FACTORS = ("precursor", "T_C", "beta", "t_hold")
ORDER_A = ("precursor", "T_C", "beta", "t_hold")   # 任务指定顺序
ORDER_B = ("t_hold", "beta", "T_C", "precursor")   # 反序,检验 Type I 顺序敏感性
N_PERM = 5000

# 论文已发表的表1(百分比 η²),D_XRD/lattice_c 两行 —— 用于对照。
PAPER_TABLE1 = {
    "D_XRD":     dict(precursor=1.1, T_C=66.3, beta=1.0, t_hold=16.0),
    "lattice_c": dict(precursor=0.3, T_C=29.0, beta=7.9, t_hold=17.9),
}


def build_main_effects_design(df: pd.DataFrame, order: tuple[str, ...]):
    """构造只含主效应(4 项,Sum 编码)的设计矩阵,项按 `order` 顺序排列。
    返回 (X, term_slices, formula_rhs)。"""
    rhs = " + ".join(f"C({f}, Sum)" for f in order)
    _, X = patsy.dmatrices(f"__dummy_y__ ~ {rhs}",
                           data=df.assign(__dummy_y__=0.0), return_type="dataframe")
    raw_slices = X.design_info.term_name_slices
    name_map = {f"C({f}, Sum)": f for f in FACTORS}
    term_slices = {"Intercept": raw_slices["Intercept"]}
    for raw_name, sl in raw_slices.items():
        if raw_name in name_map:
            term_slices[name_map[raw_name]] = sl
    term_slices = {k: term_slices[k] for k in ["Intercept"] + list(order)}
    return X.values, term_slices, rhs


def type1_table(df: pd.DataFrame, target: str, order: tuple[str, ...], seed: int) -> pd.DataFrame:
    """用本模块自建的主效应设计矩阵 + anova_interactions 里通用的
    type1_ss/omega_squared/permutation_p_values 算 Type I(序贯)SS。"""
    X, term_slices, _ = build_main_effects_design(df, order)
    y = df[target].to_numpy(float)
    ss = type1_ss(X, y, term_slices)

    dfree = {t: (term_slices[t].stop - term_slices[t].start) for t in order}
    dfree["Residual"] = X.shape[0] - X.shape[1]
    ss_total = sum(ss.values())
    ms_error = ss["Residual"] / dfree["Residual"]

    perm_p = permutation_p_values(X, y, term_slices, ss, n_perm=N_PERM, seed=seed)

    rows = []
    for t in order:
        ms = ss[t] / dfree[t]
        f_val = ms / ms_error
        rows.append(dict(term=t, SS=ss[t], df=dfree[t], MS=ms, F=f_val,
                         eta2=ss[t] / ss_total,
                         omega2=omega_squared(ss[t], dfree[t], ss_total, ms_error),
                         perm_p=perm_p[t]))
    rows.append(dict(term="Residual", SS=ss["Residual"], df=dfree["Residual"],
                     MS=ms_error, F=np.nan, eta2=ss["Residual"] / ss_total,
                     omega2=np.nan, perm_p=np.nan))
    return pd.DataFrame(rows).set_index("term")


def marginal_table(df: pd.DataFrame, target: str, typ: int) -> pd.DataFrame:
    """Type II / III(边际 SS,与项顺序无关),用 statsmodels 直接算,
    与 Type I 用同一套 Sum 编码保持可比。"""
    work = df.copy()
    for f in FACTORS:
        work[f] = work[f].astype("category")
    rhs = " + ".join(f"C({f}, Sum)" for f in FACTORS)
    model = smf.ols(f"{target} ~ {rhs}", data=work).fit()
    tab = anova_lm(model, typ=typ)
    tab = tab.rename(index={f"C({f}, Sum)": f for f in FACTORS})
    tab = tab.drop(index="Intercept", errors="ignore")

    ss_total = tab["sum_sq"].sum()
    ms_error = tab.loc["Residual", "sum_sq"] / tab.loc["Residual", "df"]
    rows = []
    for t in FACTORS:
        ss_t = tab.loc[t, "sum_sq"]
        df_t = int(tab.loc[t, "df"])
        rows.append(dict(term=t, SS=ss_t, df=df_t, MS=ss_t / df_t,
                         F=tab.loc[t, "F"],
                         eta2=ss_t / ss_total,
                         omega2=omega_squared(ss_t, df_t, ss_total, ms_error),
                         p_parametric=tab.loc[t, "PR(>F)"]))
    rows.append(dict(term="Residual", SS=tab.loc["Residual", "sum_sq"],
                     df=int(tab.loc["Residual", "df"]), MS=ms_error, F=np.nan,
                     eta2=tab.loc["Residual", "sum_sq"] / ss_total, omega2=np.nan,
                     p_parametric=np.nan))
    return pd.DataFrame(rows).set_index("term")


def main():
    warnings.filterwarnings("ignore", category=UserWarning)  # 压掉 nano_layer_frame 的提示,下面手动打印一次
    df_full = pd.read_csv(MASTER_TABLE)
    sub = nano_layer_frame(df_full, require_reliable=True)
    print(f"[nano_layer_frame] n={len(sub)} (xrd_instrument==1 & D_XRD_reliable==True)")

    n_cells = sub.groupby(list(FACTORS)).size()
    print(f"因子水平计数:precursor={sub['precursor'].value_counts().to_dict()}, "
          f"T_C={sub['T_C'].value_counts().to_dict()}, "
          f"beta={sub['beta'].value_counts().to_dict()}, "
          f"t_hold={sub['t_hold'].value_counts().to_dict()}")
    print(f"最小/最大 cell 计数:{n_cells.min()} / {n_cells.max()} "
          f"(非 1,证实设计不平衡,不满足 anova_interactions.py 81样脚本的前提假设)")

    long_rows = []
    console_summary = []

    for i, target in enumerate(TARGETS):
        if sub[target].isna().any():
            raise RuntimeError(f"{target}: nano_layer_frame 输出中仍有 NaN,"
                              "不应发生(D_XRD_reliable==True 应已保证非空),停下来报告。")

        t1a = type1_table(sub, target, ORDER_A, seed=2000 + i)
        t1b = type1_table(sub, target, ORDER_B, seed=3000 + i)
        t2 = marginal_table(sub, target, typ=2)
        t3 = marginal_table(sub, target, typ=3)

        for ss_type, tab, cols in (
            ("I_orderA", t1a, ("SS", "df", "MS", "F", "eta2", "omega2", "perm_p")),
            ("I_orderB", t1b, ("SS", "df", "MS", "F", "eta2", "omega2", "perm_p")),
            ("II", t2, ("SS", "df", "MS", "F", "eta2", "omega2", "p_parametric")),
            ("III", t3, ("SS", "df", "MS", "F", "eta2", "omega2", "p_parametric")),
        ):
            for term in list(tab.index):
                row = tab.loc[term]
                d = dict(target=target, ss_type=ss_type, term=term)
                for c in cols:
                    d[c] = row.get(c, np.nan)
                long_rows.append(d)

        # ---- Type I 顺序敏感性 + II/III 一致性,实测,不预设 ----
        orderA_eta2 = {t: t1a.loc[t, "eta2"] for t in FACTORS}
        orderB_eta2 = {t: t1b.loc[t, "eta2"] for t in FACTORS}
        typeI_order_sensitive = any(
            abs(orderA_eta2[t] - orderB_eta2[t]) > 1e-9 for t in FACTORS)
        ii_iii_ss_diff = max(abs(t2.loc[t, "SS"] - t3.loc[t, "SS"]) for t in FACTORS)
        ii_iii_agree = ii_iii_ss_diff < 1e-6 * max(t2.loc["Residual", "SS"], 1.0)
        typeI_vs_II_diff = max(abs(t1a.loc[t, "SS"] - t2.loc[t, "SS"]) for t in FACTORS)
        typeI_matches_II = typeI_vs_II_diff < 1e-6 * max(t2.loc["Residual", "SS"], 1.0)

        console_summary.append(dict(
            target=target,
            typeI_order_sensitive=typeI_order_sensitive,
            typeII_III_agree=bool(ii_iii_agree),
            typeI_orderA_matches_typeII=bool(typeI_matches_II),
        ))

        print(f"\n=== {target} ===")
        print(f"Type I 顺序敏感(orderA vs orderB η² 是否不同): {typeI_order_sensitive}")
        print(f"  orderA η%: " + ", ".join(f"{t}={orderA_eta2[t]*100:.2f}" for t in FACTORS))
        print(f"  orderB η%: " + ", ".join(f"{t}={orderB_eta2[t]*100:.2f}" for t in FACTORS))
        print(f"Type II vs III SS 最大绝对差: {ii_iii_ss_diff:.6g} -> 一致: {ii_iii_agree}")
        print(f"Type I(orderA) vs Type II SS 最大绝对差: {typeI_vs_II_diff:.6g} "
              f"-> 一致: {typeI_matches_II}")

        print(f"权威值(Type II,η²%)vs 论文表1 {target} 行:")
        for t in FACTORS:
            new_pct = t2.loc[t, "eta2"] * 100
            old_pct = PAPER_TABLE1[target][t]
            print(f"  {t:10s} 新={new_pct:6.2f}%  论文={old_pct:6.2f}%  "
                 f"差={new_pct - old_pct:+6.2f}pp")

    long_df = pd.DataFrame(long_rows)
    OUT_LONG.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(OUT_LONG, index=False)
    print(f"\n[done] {OUT_LONG}")

    print("\n=== Type I/II/III 一致性总结(权威判定) ===")
    for row in console_summary:
        print(row)


if __name__ == "__main__":
    main()
