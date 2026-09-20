# -*- coding: utf-8 -*-
"""16_primary_particle_mnar.py —— 一次颗粒"不可辨=记忆保留"论证的替代方案（2026-07-30）

背景
----
原任务书方案（对 primary_segmentable ~ precursor + lnTheta 做逻辑回归）已被废弃。
已定论：primary_segmentable 这个二值标签
与人工判定不相关（Spearman rho 全部 p>0.3），而"前驱体类型"单独预测人工判定的
准确率（0.89）反而高于任何纹理指标（0.78）——即该标签已退化为前驱体的代理变量。
若直接对它做 ~ precursor + lnTheta 回归，得到的"precursor 显著"只是在重复验证
仲裁报告已指出的循环论证问题，不是新证据。故本脚本不做这条回归当证据用。

替代方案（本脚本做的事）
----------------------
把"一次颗粒不可辨=记忆保留证据"这个论证主张，从静态二元标签 primary_segmentable
改为连续量 Lbar_um（脊线分水岭 + ASTM E112 平均线截距，data/interim/sem_dpri_trend/
dpri_intercept_all.csv，全自动、不依赖人工判据、不读前驱体标签，81/81 样本齐全）
的行为特征：

  1. 按 (precursor, Theta 分位) 交叉，统计 Lbar_um 的均值/标准差，复用已有的
     逐样本 cv 列（不重算）。重点看 L 组低 Theta 端的 cv 是否相对更小。
  2. Lbar_um ~ precursor + lnTheta 的线性回归（含交互项版本），报系数/标准误/p 值。
     只用于方向性论证，不解读为定量物理结论（Lbar_um 未做绝对刻度校准，
     已定论"趋势可用，绝对值不可用"）。
  3. 仍旧描述旧标签 primary_segmentable/D_pri_sem 的缺失模式（is_missing_old ~
     precursor + lnTheta 逻辑回归），但结论与 1-2 严格分开：这一节只是描述性地
     印证仲裁报告的循环论证担忧，不作为记忆保留假说的证据。

输出
----
data/interim/primary_segmentability_mnar.csv   长表，section 列区分三块分析
reports/primary_particle_mnar_report.md        结论性报告

用法: python scripts/16_primary_particle_mnar.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

LBAR_CSV = ROOT / "data" / "interim" / "sem_dpri_trend" / "dpri_intercept_all.csv"
MASTER_CSV = ROOT / "data" / "processed" / "master_table.csv"
OUT_CSV = ROOT / "data" / "interim" / "primary_segmentability_mnar.csv"
OUT_REPORT = ROOT / "reports" / "primary_particle_mnar_report.md"

N_THETA_BINS = 3
THETA_BIN_LABELS = ["low", "mid", "high"]


def load_data():
    lbar = pd.read_csv(LBAR_CSV)
    master = pd.read_csv(MASTER_CSV)
    assert lbar["sample_id"].is_unique, "dpri_intercept_all.csv 存在重复 sample_id"
    assert len(lbar) == 81, f"期望 81 行，实际 {len(lbar)}"
    return lbar, master


def verify_old_label_correspondence(master: pd.DataFrame) -> str:
    """核实 D_pri_sem 缺失 与 primary_segmentable==False 是否完全对应（不假设）。"""
    is_nan = master["D_pri_sem"].isna()
    is_false = master["primary_segmentable"] == False  # noqa: E712
    n_nan = int(is_nan.sum())
    n_false = int(is_false.sum())
    n_mismatch = int((is_nan != is_false).sum())
    note = (
        f"D_pri_sem 缺失={n_nan}，primary_segmentable==False={n_false}，"
        f"两者不一致的行数={n_mismatch}（0 表示完全对应）。"
    )
    print(note)
    return note


# ---------------------------------------------------------------------------
# Section 1: Lbar_um 分布统计（precursor x Theta 分位）
# ---------------------------------------------------------------------------

def section1_group_stats(lbar: pd.DataFrame) -> pd.DataFrame:
    df = lbar.copy()
    # Theta 三分位，全局切（不分前驱体），与 T1/T4/T5 系列沿用的低/中/高 Theta
    # 惯例一致；用 pd.qcut 保证每档样本量接近均衡。分档边界写入 note 供复核。
    df["theta_tercile"], bin_edges = pd.qcut(
        df["Theta"], N_THETA_BINS, labels=THETA_BIN_LABELS, retbins=True
    )
    edges_str = ", ".join(f"{e:.3f}" for e in bin_edges)

    rows = []
    for (precursor, tercile), g in df.groupby(["precursor", "theta_tercile"], observed=True):
        rows.append(dict(
            section="1_lbar_group_stats",
            precursor=precursor,
            theta_tercile=str(tercile),
            n=len(g),
            mean_or_estimate=g["Lbar_um"].mean(),
            std_or_se=g["Lbar_um"].std(ddof=1) if len(g) > 1 else np.nan,
            cv_mean=g["cv"].mean(),
            cv_median=g["cv"].median(),
            term=np.nan,
            pvalue=np.nan,
            r_squared=np.nan,
            note=f"Theta三分位边界(全局qcut): [{edges_str}]",
        ))
    out = pd.DataFrame(rows)

    # 重点检查：L 组低 Theta 端的 cv 是否相对整体更小
    overall_cv_mean = df["cv"].mean()
    l_low = df[(df["precursor"] == "L") & (df["theta_tercile"] == "low")]
    l_low_cv_mean = l_low["cv"].mean() if len(l_low) else np.nan
    check_row = dict(
        section="1_lbar_group_stats_check",
        precursor="L",
        theta_tercile="low",
        n=len(l_low),
        mean_or_estimate=np.nan,
        std_or_se=np.nan,
        cv_mean=l_low_cv_mean,
        cv_median=l_low["cv"].median() if len(l_low) else np.nan,
        term="cv_L_low_vs_overall_mean",
        pvalue=np.nan,
        r_squared=np.nan,
        note=(
            f"L组低Theta端cv均值={l_low_cv_mean:.4f} vs 全体cv均值={overall_cv_mean:.4f}；"
            f"{'更小' if (not np.isnan(l_low_cv_mean) and l_low_cv_mean < overall_cv_mean) else '未见更小'}"
        ),
    )
    out = pd.concat([out, pd.DataFrame([check_row])], ignore_index=True)
    print("\n=== Section 1: Lbar_um group stats (precursor x Theta tercile) ===")
    print(out[out["section"] == "1_lbar_group_stats"][
        ["precursor", "theta_tercile", "n", "mean_or_estimate", "std_or_se", "cv_mean", "cv_median"]
    ].to_string(index=False))
    print(f"\nL组低Theta cv均值={l_low_cv_mean:.4f}, 全体cv均值={overall_cv_mean:.4f}")
    return out


# ---------------------------------------------------------------------------
# Section 2: Lbar_um ~ precursor + lnTheta (OLS)
# ---------------------------------------------------------------------------

def section2_ols(lbar: pd.DataFrame) -> pd.DataFrame:
    df = lbar.copy()
    df["lnTheta"] = np.log(df["Theta"])
    df["precursor"] = pd.Categorical(df["precursor"], categories=["S", "M", "L"])

    rows = []

    # Main-effects model
    model_main = smf.ols("Lbar_um ~ C(precursor, Treatment('S')) + lnTheta", data=df).fit()
    r2_main = model_main.rsquared
    for term, coef in model_main.params.items():
        rows.append(dict(
            section="2_ols_main_effects",
            precursor=np.nan,
            theta_tercile=np.nan,
            n=len(df),
            mean_or_estimate=coef,
            std_or_se=model_main.bse[term],
            cv_mean=np.nan,
            cv_median=np.nan,
            term=term,
            pvalue=model_main.pvalues[term],
            r_squared=r2_main,
            note="Lbar_um ~ precursor(S为基线) + lnTheta，主效应模型，仅方向性论证",
        ))

    # Interaction model: lets lnTheta slope differ by precursor (analog of
    # per-group Spearman rho reported in D_pri趋势验证_2026-07-22.md)
    model_int = smf.ols(
        "Lbar_um ~ C(precursor, Treatment('S')) * lnTheta", data=df
    ).fit()
    r2_int = model_int.rsquared
    for term, coef in model_int.params.items():
        rows.append(dict(
            section="2b_ols_interaction",
            precursor=np.nan,
            theta_tercile=np.nan,
            n=len(df),
            mean_or_estimate=coef,
            std_or_se=model_int.bse[term],
            cv_mean=np.nan,
            cv_median=np.nan,
            term=term,
            pvalue=model_int.pvalues[term],
            r_squared=r2_int,
            note="Lbar_um ~ precursor(S为基线) * lnTheta，交互模型（各前驱体独立lnTheta斜率），仅方向性论证",
        ))

    out = pd.DataFrame(rows)
    print("\n=== Section 2: OLS main-effects model ===")
    print(model_main.summary())
    print("\n=== Section 2b: OLS interaction model ===")
    print(model_int.summary())

    # Per-precursor slope (intercept + interaction) for interpretability
    print("\n--- per-precursor lnTheta 斜率（来自交互模型）---")
    base_slope = model_int.params.get("lnTheta", np.nan)
    for p in ["S", "M", "L"]:
        key = f"C(precursor, Treatment('S'))[T.{p}]:lnTheta"
        if p == "S":
            slope = base_slope
        else:
            slope = base_slope + model_int.params.get(key, 0.0)
        print(f"  {p}: slope={slope:.4f}")

    return out


# ---------------------------------------------------------------------------
# Section 4: 旧标签缺失模式（描述性，不作证据）
# ---------------------------------------------------------------------------

def section4_old_label_missing(master: pd.DataFrame, correspondence_note: str) -> pd.DataFrame:
    df = master.copy()
    df["lnTheta"] = np.log(df["Theta"])
    df["precursor"] = pd.Categorical(df["precursor"], categories=["S", "M", "L"])
    # is_missing_old 用 primary_segmentable==False（已核实与 D_pri_sem 缺失完全
    # 对应，见 correspondence_note），语义上更直接对应"旧标签认为不可分割"。
    df["is_missing_old"] = (df["primary_segmentable"] == False).astype(int)  # noqa: E712

    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = smf.logit(
                "is_missing_old ~ C(precursor, Treatment('S')) + lnTheta", data=df
            ).fit(disp=0)
            pseudo_r2 = model.prsquared
            for term, coef in model.params.items():
                rows.append(dict(
                    section="4_logit_missing_old",
                    precursor=np.nan,
                    theta_tercile=np.nan,
                    n=len(df),
                    mean_or_estimate=coef,
                    std_or_se=model.bse[term],
                    cv_mean=np.nan,
                    cv_median=np.nan,
                    term=term,
                    pvalue=model.pvalues[term],
                    r_squared=pseudo_r2,
                    note=(
                        "is_missing_old(=primary_segmentable==False) ~ precursor + lnTheta；"
                        "仅描述旧标签失败模式，不作为记忆保留假说的证据。" + correspondence_note
                    ),
                ))
            print("\n=== Section 4: logit(is_missing_old ~ precursor + lnTheta) ===")
            print(model.summary())
        except Exception as e:  # perfect/quasi separation etc.
            warnings.warn(f"Section 4 logit 拟合失败: {e}")
            rows.append(dict(
                section="4_logit_missing_old",
                precursor=np.nan,
                theta_tercile=np.nan,
                n=len(df),
                mean_or_estimate=np.nan,
                std_or_se=np.nan,
                cv_mean=np.nan,
                cv_median=np.nan,
                term="FIT_FAILED",
                pvalue=np.nan,
                r_squared=np.nan,
                note=f"logit拟合失败: {e}. " + correspondence_note,
            ))

    # Descriptive crosstab as well (robust regardless of logit convergence)
    ct = pd.crosstab(df["precursor"], df["is_missing_old"])
    print("\n--- crosstab: precursor x is_missing_old ---")
    print(ct)
    for precursor in ["S", "M", "L"]:
        n_total = int((df["precursor"] == precursor).sum())
        n_missing = int(((df["precursor"] == precursor) & (df["is_missing_old"] == 1)).sum())
        rows.append(dict(
            section="4_crosstab_descriptive",
            precursor=precursor,
            theta_tercile=np.nan,
            n=n_total,
            mean_or_estimate=n_missing / n_total if n_total else np.nan,
            std_or_se=np.nan,
            cv_mean=np.nan,
            cv_median=np.nan,
            term="missing_fraction",
            pvalue=np.nan,
            r_squared=np.nan,
            note="旧标签按前驱体的缺失比例，纯描述性",
        ))

    return pd.DataFrame(rows)


def main():
    lbar, master = load_data()
    correspondence_note = verify_old_label_correspondence(master)

    sec1 = section1_group_stats(lbar)
    sec2 = section2_ols(lbar)
    sec4 = section4_old_label_missing(master, correspondence_note)

    out = pd.concat([sec1, sec2, sec4], ignore_index=True)
    cols = [
        "section", "precursor", "theta_tercile", "n", "mean_or_estimate",
        "std_or_se", "cv_mean", "cv_median", "term", "pvalue", "r_squared", "note",
    ]
    out = out[cols]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n写出: {OUT_CSV} ({len(out)} 行)")


if __name__ == "__main__":
    main()
