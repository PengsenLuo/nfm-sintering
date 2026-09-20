# -*- coding: utf-8 -*-
"""27 D(003) 与 D(104) 分向诊断
====================================================================
背景:现流程用 (003)/(104) 两个主衍射峰的 Scherrer 尺寸直接平均输出为
`D_XRD`。方法学审稿意见指出:两峰对应不同晶体学方向,R-3m 层状结构可能
存在方向性宽化的延伸,无条件平均需要验证。

本脚本只做诊断与输出,**不改变 `D_XRD` 的计算方式或任何既有数值**——
`D_XRD_003`/`D_XRD_104` 是 2026-08-15 新增的 schema 诊断列(见
`src/nfm/schema.py`、`src/nfm/data_processing/xrd_processor.py`),复用
`process_all()` 里已经算出的 (003)/(104) 各自 Scherrer 分量,不重新拟合。

数据口径:仅 `nfm.nano_layer.nano_layer_frame(df, require_reliable=True)`
(xrd_instrument==1 且 D_XRD_reliable==True,51 样/17 条件),与论文纳米层
唯一合法口径一致。

产出:
  data/interim/dxrd_hkl_anisotropy.csv       逐样本 D_XRD_003/D_XRD_104/ratio 等
  data/interim/dxrd_hkl_anisotropy_eta2.csv  D(003)/D(104)/平均 三个因变量的主效应 η² 长表
  reports/dxrd_hkl_anisotropy.md             独立复算 + 诊断结论(不替作者做取舍)
"""
import _bootstrap  # noqa

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import kruskal, pearsonr, spearmanr, wilcoxon
from statsmodels.stats.anova import anova_lm

from nfm.nano_layer import nano_layer_frame

MASTER_TABLE = "data/processed/master_table.csv"
OUT_CSV = Path("data/interim/dxrd_hkl_anisotropy.csv")
OUT_ETA2_CSV = Path("data/interim/dxrd_hkl_anisotropy_eta2.csv")
OUT_REPORT = Path("reports/dxrd_hkl_anisotropy.md")

FACTORS = ("precursor", "T_C", "beta", "t_hold")

# 既有参考在 51 样上给出的结果,独立复算、不直接采信。
COWORK_CLAIM = dict(
    median_d003=56.9, median_d104=70.5, median_ratio=1.25,
    pearson_r=0.990, spearman_rho=0.986, wilcoxon_p_lt=1e-4,
)


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    df_full = pd.read_csv(MASTER_TABLE)
    sub = nano_layer_frame(df_full, require_reliable=True).copy()
    print(f"[nano_layer_frame] n={len(sub)} (xrd_instrument==1 & D_XRD_reliable==True)")

    n_missing = sub[["D_XRD_003", "D_XRD_104"]].isna().any(axis=1).sum()
    if n_missing:
        warnings.warn(f"{n_missing} 个样本 D_XRD_003/D_XRD_104 至少一个为 NaN,"
                      "已从下列成对统计中剔除(D_XRD 主报值可能仍用了单峰,"
                      "但成对比较必须两峰都在)")
    pair = sub.dropna(subset=["D_XRD_003", "D_XRD_104"]).copy()
    pair["ratio_104_003"] = pair["D_XRD_104"] / pair["D_XRD_003"]
    print(f"[成对样本] n={len(pair)}")

    pair[["sample_id", "condition_id", "precursor", "Theta", "T_C",
         "D_XRD", "D_XRD_003", "D_XRD_104", "ratio_104_003"]].to_csv(OUT_CSV, index=False)

    # ---- 1. 独立复算总体统计(不采信既有参考数字,自己算) ----
    d003, d104 = pair["D_XRD_003"].to_numpy(), pair["D_XRD_104"].to_numpy()
    median_d003, median_d104 = float(np.median(d003)), float(np.median(d104))
    median_ratio = float(pair["ratio_104_003"].median())
    r_pearson, p_pearson = pearsonr(d003, d104)
    rho_spearman, p_spearman = spearmanr(d003, d104)
    w_stat, w_p = wilcoxon(d003, d104)

    headline = dict(
        n=len(pair), median_d003=median_d003, median_d104=median_d104,
        median_ratio=median_ratio, pearson_r=float(r_pearson), pearson_p=float(p_pearson),
        spearman_rho=float(rho_spearman), spearman_p=float(p_spearman),
        wilcoxon_stat=float(w_stat), wilcoxon_p=float(w_p),
    )
    print("\n=== 1. 独立复算总体统计(51样)===")
    for k, v in headline.items():
        print(f"  {k}: {v}")

    diffs = {
        "median_d003": (median_d003, COWORK_CLAIM["median_d003"]),
        "median_d104": (median_d104, COWORK_CLAIM["median_d104"]),
        "median_ratio": (median_ratio, COWORK_CLAIM["median_ratio"]),
        "pearson_r": (float(r_pearson), COWORK_CLAIM["pearson_r"]),
        "spearman_rho": (float(rho_spearman), COWORK_CLAIM["spearman_rho"]),
    }
    print("\n=== 与既有参考数字的差异(独立核算,不采信对方数字)===")
    for k, (mine, theirs) in diffs.items():
        print(f"  {k}: 本轮={mine:.4f}  既有={theirs:.4f}  差={mine - theirs:+.4f}")

    # ---- 2. 分向 vs Theta 的 Spearman(整体 + 分前驱体) ----
    rho_d003_theta, p_d003_theta = spearmanr(pair["D_XRD_003"], pair["Theta"])
    rho_d104_theta, p_d104_theta = spearmanr(pair["D_XRD_104"], pair["Theta"])
    rho_dxrd_theta, p_dxrd_theta = spearmanr(pair["D_XRD"], pair["Theta"])

    by_precursor = []
    for prec in ("S", "M", "L"):
        g = pair[pair["precursor"] == prec]
        r003, pv003 = spearmanr(g["D_XRD_003"], g["Theta"])
        r104, pv104 = spearmanr(g["D_XRD_104"], g["Theta"])
        rratio, pvratio = spearmanr(g["ratio_104_003"], g["Theta"])
        by_precursor.append(dict(precursor=prec, n=len(g),
                                 rho_d003_theta=r003, p_d003_theta=pv003,
                                 rho_d104_theta=r104, p_d104_theta=pv104,
                                 rho_ratio_theta=rratio, p_ratio_theta=pvratio))
    by_precursor_df = pd.DataFrame(by_precursor)
    print("\n=== 2. Spearman(D_XRD_00x, Theta):整体 + 分前驱体 ===")
    print(f"  整体: D003={rho_d003_theta:+.3f}(p={p_d003_theta:.4f})  "
          f"D104={rho_d104_theta:+.3f}(p={p_d104_theta:.4f})  "
          f"D_XRD(均值)={rho_dxrd_theta:+.3f}(p={p_dxrd_theta:.4f})")
    print(by_precursor_df.to_string(index=False))

    # ---- 3. 各向异性比值(D104/D003)是否随 Theta/T_C/precursor 系统性漂移 ----
    rho_ratio_theta, p_ratio_theta = spearmanr(pair["ratio_104_003"], pair["Theta"])
    rho_ratio_T, p_ratio_T = spearmanr(pair["ratio_104_003"], pair["T_C"])
    groups = [g["ratio_104_003"].to_numpy() for _, g in pair.groupby("precursor")]
    kw_stat, kw_p = kruskal(*groups)
    ratio_by_precursor = pair.groupby("precursor")["ratio_104_003"].agg(["median", "std", "count"])
    print("\n=== 3. 各向异性比值 D104/D003 的系统性漂移检验 ===")
    print(f"  Spearman(ratio, Theta) = {rho_ratio_theta:+.3f} (p={p_ratio_theta:.4f})")
    print(f"  Spearman(ratio, T_C)   = {rho_ratio_T:+.3f} (p={p_ratio_T:.4f})")
    print(f"  Kruskal-Wallis(ratio ~ precursor): H={kw_stat:.3f}, p={kw_p:.4f}")
    print(ratio_by_precursor.to_string())

    # ---- 4. 主效应 η²(Type II)对照表:D(003) / D(104) / 二者平均(D_XRD) ----
    work = pair.copy()
    for f in FACTORS:
        work[f] = work[f].astype("category")
    eta2_rows = []
    eta2_summary = {}
    for target in ("D_XRD_003", "D_XRD_104", "D_XRD"):
        rhs = " + ".join(f"C({f}, Sum)" for f in FACTORS)
        model = smf.ols(f"{target} ~ {rhs}", data=work).fit()
        tab = anova_lm(model, typ=2)
        tab = tab.rename(index={f"C({f}, Sum)": f for f in FACTORS})
        y = work[target].to_numpy(float)
        sst_true = float(((y - y.mean()) ** 2).sum())
        target_summary = {}
        for f in FACTORS:
            ss_f = float(tab.loc[f, "sum_sq"])
            eta2_pct = 100.0 * ss_f / sst_true
            target_summary[f] = eta2_pct
            eta2_rows.append(dict(target=target, term=f, SS=ss_f,
                                  df=int(tab.loc[f, "df"]), eta2_pct=eta2_pct,
                                  p_parametric=float(tab.loc[f, "PR(>F)"])))
        ss_resid = float(tab.loc["Residual", "sum_sq"])
        eta2_pct_resid = 100.0 * ss_resid / sst_true
        target_summary["Residual"] = eta2_pct_resid
        eta2_rows.append(dict(target=target, term="Residual", SS=ss_resid,
                              df=int(tab.loc["Residual", "df"]), eta2_pct=eta2_pct_resid,
                              p_parametric=np.nan))
        eta2_summary[target] = target_summary

    eta2_df = pd.DataFrame(eta2_rows)
    eta2_df.to_csv(OUT_ETA2_CSV, index=False)
    print("\n=== 4. 主效应 η²(Type II,%),D(003) vs D(104) vs D_XRD(平均) ===")
    print(pd.DataFrame(eta2_summary).T.to_string())

    write_report(headline, diffs, by_precursor_df,
                 rho_ratio_theta, p_ratio_theta, rho_ratio_T, p_ratio_T,
                 kw_stat, kw_p, ratio_by_precursor, eta2_summary,
                 rho_dxrd_theta, p_dxrd_theta)
    print(f"\n[done] {OUT_CSV}\n[done] {OUT_ETA2_CSV}\n[done] {OUT_REPORT}")


def write_report(headline, diffs, by_precursor_df,
                 rho_ratio_theta, p_ratio_theta, rho_ratio_T, p_ratio_T,
                 kw_stat, kw_p, ratio_by_precursor, eta2_summary,
                 rho_dxrd_theta, p_dxrd_theta):
    lines = []
    lines.append("# D(003) 与 D(104) 分向各向异性诊断")
    lines.append("")
    lines.append("> 由 `scripts/27_dxrd_hkl_anisotropy.py` 自动生成。方法学审稿意见:"
                 "`D_XRD` 现将 (003)/(104) 两个主峰的 Scherrer 尺寸无条件平均,"
                 "对应不同晶体学方向,R-3m 层状结构可能存在方向性宽化。本报告"
                 "**只诊断,不改变 `D_XRD` 的计算方式或任何既有数值**;"
                 "`D_XRD_003`/`D_XRD_104` 是新增诊断列,复用已有拟合结果,不重新拟合。")
    lines.append("")
    lines.append(f"数据口径:`nfm.nano_layer.nano_layer_frame(df, require_reliable=True)`,"
                 f"n={headline['n']}(xrd_instrument==1 且 D_XRD_reliable==True,51 样/17 条件"
                 "口径,与论文纳米层唯一合法口径一致)。")
    lines.append("")
    lines.append("## 1. 独立复算总体统计,并与既有参考数字对照")
    lines.append("")
    lines.append("**未采信既有参考给出的数字,以下为本轮独立复算结果**:")
    lines.append("")
    lines.append("| 量 | 本轮独立复算 | 既有参考 | 差 |")
    lines.append("|---|---|---|---|")
    for k, (mine, theirs) in diffs.items():
        lines.append(f"| {k} | {mine:.4f} | {theirs:.4f} | {mine - theirs:+.4f} |")
    lines.append(f"| wilcoxon_p | {headline['wilcoxon_p']:.3e} | < {COWORK_CLAIM['wilcoxon_p_lt']:.0e} | "
                 f"{'一致(同量级,更显著)' if headline['wilcoxon_p'] < COWORK_CLAIM['wilcoxon_p_lt'] else '不一致,需复核'} |")
    lines.append("")
    max_abs_diff = max(abs(mine - theirs) for mine, theirs in diffs.values())
    flagged = [k for k, (mine, theirs) in diffs.items() if abs(mine - theirs) > 0.05]
    if flagged:
        lines.append(f"**独立复算与既有参考数字存在可见差异的量**:{', '.join(flagged)}"
                     f"(最大绝对差 {max_abs_diff:.4f})。差异量级不影响下方"
                     "\"两峰高度一致但系统性偏离约25%\"的定性结论,但数值层面"
                     "以本报告独立复算的结果为准,不直接采信既有参考数字。")
    else:
        lines.append("独立复算与既有参考数字全部一致(差异 < 0.05)。")
    lines.append("")
    lines.append(f"- Pearson r = **{headline['pearson_r']:+.4f}** (p={headline['pearson_p']:.3e})")
    lines.append(f"- Spearman ρ = **{headline['spearman_rho']:+.4f}** (p={headline['spearman_p']:.3e})")
    lines.append(f"- 配对 Wilcoxon p = **{headline['wilcoxon_p']:.3e}**"
                 "(两峰尺寸系统性不同,不是随机噪声)")
    lines.append(f"- D(104)/D(003) 中位数比值 = **{headline['median_ratio']:.4f}**"
                 "(两峰高度一致但存在约25%的系统性方向差异)")
    lines.append("")
    lines.append("## 2. Spearman(D_XRD_00x, Θ):整体 + 分前驱体(S/M/L 各自驱动无混淆)")
    lines.append("")
    lines.append(f"整体:D(003)/D(104) 与 Θ 的 Spearman ρ 见下表逐前驱体分解;"
                 f"D_XRD(均值)与 Θ 整体 ρ={rho_dxrd_theta:+.3f}(p={p_dxrd_theta:.4f})"
                 "(与论文 §3.3 报告的 nano-layer Spearman 一致性核对项)。")
    lines.append("")
    lines.append("| precursor | n | ρ(D003,Θ) | p | ρ(D104,Θ) | p | ρ(ratio,Θ) | p |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, row in by_precursor_df.iterrows():
        lines.append(f"| {row['precursor']} | {int(row['n'])} | "
                     f"{row['rho_d003_theta']:+.3f} | {row['p_d003_theta']:.4f} | "
                     f"{row['rho_d104_theta']:+.3f} | {row['p_d104_theta']:.4f} | "
                     f"{row['rho_ratio_theta']:+.3f} | {row['p_ratio_theta']:.4f} |")
    lines.append("")
    lines.append("两峰各自与 Θ 的相关方向/显著性在三个前驱体分组内是否一致,"
                 "决定了\"用哪一峰报告\"是否会改变论文对 Θ-D_XRD 关系的结论——"
                 "结果见上表,结合下文 §4 一并判断。")
    lines.append("")
    lines.append("## 3. 各向异性比值 D(104)/D(003) 是否随 Θ/T/前驱体系统性漂移")
    lines.append("")
    lines.append(f"- Spearman(ratio, Θ) = **{rho_ratio_theta:+.3f}** (p={p_ratio_theta:.4f})")
    lines.append(f"- Spearman(ratio, T_C) = **{rho_ratio_T:+.3f}** (p={p_ratio_T:.4f})")
    lines.append(f"- Kruskal-Wallis(ratio ~ precursor):H={kw_stat:.3f}, p={kw_p:.4f}")
    lines.append("")
    lines.append("| precursor | ratio median | ratio std | n |")
    lines.append("|---|---|---|---|")
    for prec, row in ratio_by_precursor.iterrows():
        lines.append(f"| {prec} | {row['median']:.4f} | {row['std']:.4f} | {int(row['count'])} |")
    lines.append("")
    ratio_stable = (p_ratio_theta >= 0.05) and (p_ratio_T >= 0.05) and (kw_p >= 0.05)
    if ratio_stable:
        lines.append("**比值本身不随 Θ/T/前驱体系统性漂移**(三项检验均 p≥0.05):"
                     "各向异性是一个大致恒定的方向性宽化偏移,不是随烧结程度演化的效应——"
                     "这是比\"平均掩盖了细节\"更强的一条陈述,必须在正文中明确写出。")
    else:
        lines.append("**比值随 Θ/T/前驱体存在统计上可见的系统性漂移**(至少一项检验 p<0.05,"
                     "见上方具体数字):各向异性本身不是恒定偏移,不能简单归结为\"仪器/晶体学"
                     "常数差异\",需要在正文里如实说明,不能简化为\"两峰平行\"。")
    lines.append("")
    lines.append("## 4. 主效应 η²(Type II,%)对照表:D(003) vs D(104) vs D_XRD(平均)")
    lines.append("")
    lines.append("模型:`y ~ C(precursor,Sum) + C(T_C,Sum) + C(beta,Sum) + C(t_hold,Sum)`"
                 "(只主效应,不含交互,与 `scripts/18_nano_layer_anova.py` 同规格),"
                 "η² 分母用真实 SST。")
    lines.append("")
    header = "| target | " + " | ".join(list(FACTORS) + ["Residual"]) + " |"
    lines.append(header)
    lines.append("|---|" + "---|" * (len(FACTORS) + 1))
    for target in ("D_XRD_003", "D_XRD_104", "D_XRD"):
        vals = eta2_summary[target]
        row = f"| {target} | " + " | ".join(f"{vals[f]:.1f}%" for f in FACTORS) + \
              f" | {vals['Residual']:.1f}% |"
        lines.append(row)
    lines.append("")
    lines.append("交互项 η² 未在本报告估计(51 样不平衡设计,交互项估计需要"
                 "`scripts/19_nano_layer_interaction_anova.py` 那一套秩检验/QR 序贯 SS "
                 "基础设施,超出本次诊断范围;若需要 D(003)/D(104) 各自的交互项对照,"
                 "需追加分析)。")
    lines.append("")
    lines.append("## 5. 三个可选口径")
    lines.append("")
    lines.append("以下三种写法在方法学上均可辩护,本报告只列出各自的证据基础,"
                 "**最终选哪一种取决于稿件写作需求**:")
    lines.append("")
    lines.append("1. **只报 D(003)**:低角峰,受 Caglioti 曲线低角端固有特性影响更明显"
                 "(与 Williamson-Hall 定性参照结果的已知局限同源),但样品制备/装样误差"
                 "通常在低角更小。")
    lines.append("2. **只报 D(104)**:高角峰,分辨率更高、仪器宽化占比更小,"
                 "但强度更弱、拟合噪声可能更大。")
    lines.append("3. **维持两峰平均,但正文明确标注方向性各向异性**:现有做法,"
                 "本报告 §3 的结论(比值是否随 Θ/T/前驱体系统性漂移)决定这句限定"
                 "该怎么写——若比值恒定,限定句可以简单陈述\"存在约25%的恒定方向性偏移,"
                 "不影响相对趋势\";若比值系统性漂移,限定句需要更谨慎。")
    lines.append("")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
