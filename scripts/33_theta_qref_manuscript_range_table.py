# -*- coding: utf-8 -*-
"""33 Θ 对 Q_ref 敏感性——稿件引用范围 {180,200,220} kJ/mol 专表
====================================================================
背景:scripts/26_theta_qref_sensitivity.py 已经用 Q_ref∈{150,175,180,200,
220,225,250} kJ/mol 做了完整敏感性扫描(2026-08-16 补充 180/220 两点后),
但那份报告只给出:(a) 27条件Θ排序在相邻Q_ref间的整体 Spearman ρ,
(b) Spearman(D_XRD,Θ) 只有51样池化数字,没有按前驱体(S/M/L)拆分,
(c) Spearman(M_D_agg,Θ) 27条件的数字。

稿件正文实际引用的活化能敏健性范围是 Q_ref∈{180,200,220} kJ/mol(不是
150/175/225/250 那几个纯粹用来"括住"稿件范围的扫描点)。本脚本只聚焦这
三个值,并新增(b)按前驱体拆分的 Spearman(D_XRD,Θ)——这是 26 号脚本没有
算过的——产出一份可直接在稿件补充稳健性脚注里引用的短表。

复用 26 号脚本的 compute_theta_for_qref() 与 nano_layer_frame() 取数口径
（xrd_instrument==1 且 D_XRD_reliable==True，51样）；M_D_agg 部分直接从
26 号脚本重跑生成的 data/interim/theta_qref_sensitivity_mdagg.csv 里读取
180/200/220 三行，不重新计算。

两种结论(稳健/不稳健)都如实报告,不为了让"稳健"成立而收窄/挑选前驱体
子组或 Q_ref 取值。

输出:reports/theta_qref_sensitivity_manuscript_range.md
"""
import _bootstrap  # noqa

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

sweep_mod = import_module("26_theta_qref_sensitivity")
compute_theta_for_qref = sweep_mod.compute_theta_for_qref

from nfm.config import load_config
from nfm.nano_layer import nano_layer_frame

try:
    import tabulate  # noqa: F401
    HAVE_TABULATE = True
except ImportError:
    HAVE_TABULATE = False

Q_REF_MANUSCRIPT = [180, 200, 220]
MASTER_TABLE = "data/processed/master_table.csv"
MDAGG_CSV = "data/interim/theta_qref_sensitivity_mdagg.csv"
OUT_REPORT = "reports/theta_qref_sensitivity_manuscript_range.md"


def _table_block(df: pd.DataFrame) -> str:
    if HAVE_TABULATE:
        return df.to_markdown(index=False) + "\n\n"
    return "```\n" + df.to_string(index=False) + "\n```\n\n"


def main():
    cfg = load_config()
    df = pd.read_csv(MASTER_TABLE)

    theta_cols = {}
    for q in Q_REF_MANUSCRIPT:
        theta_cols[q] = compute_theta_for_qref(df, q, cfg)
        df[f"Theta_q{q}"] = theta_cols[q]

    # ---- 1. 27条件Θ排序稳定性,仅(180,200)与(200,220)相邻对 ----
    cond = df.drop_duplicates("condition_id").set_index("condition_id")
    rank_rows = []
    for q1, q2 in [(180, 200), (200, 220)]:
        rho, _ = spearmanr(cond[f"Theta_q{q1}"], cond[f"Theta_q{q2}"])
        rank_rows.append({"q_ref_pair": f"{q1}-{q2}", "spearman_rho": rho})
    rank_df = pd.DataFrame(rank_rows)
    print("=== 1. 27条件Θ排序稳定性({180,200,220}相邻对) ===")
    print(rank_df.to_string(index=False))

    # ---- 2. Spearman(D_XRD,Θ) 51样纳米层口径,按前驱体拆分 ----
    nano = nano_layer_frame(df, require_reliable=True)
    precursor_rows = []
    for q in Q_REF_MANUSCRIPT:
        # 池化(51样),对照26号脚本口径
        rho_all, p_all = spearmanr(nano[f"Theta_q{q}"], nano["D_XRD"])
        precursor_rows.append({
            "q_ref_kjmol": q, "precursor": "ALL(51)",
            "n": len(nano), "spearman_rho": rho_all, "p": p_all,
        })
        for prec in ["S", "M", "L"]:
            sub = nano[nano["precursor"] == prec]
            if len(sub) < 3:
                rho, p = np.nan, np.nan
            else:
                rho, p = spearmanr(sub[f"Theta_q{q}"], sub["D_XRD"])
            precursor_rows.append({
                "q_ref_kjmol": q, "precursor": prec,
                "n": len(sub), "spearman_rho": rho, "p": p,
            })
    precursor_df = pd.DataFrame(precursor_rows)
    print(f"\n=== 2. Spearman(D_XRD,Θ) 按前驱体拆分,{len(nano)}样纳米层口径 ===")
    print(precursor_df.to_string(index=False))

    # ---- 3. Spearman(M_D_agg,Θ) 27条件,从26号脚本输出里抽取180/200/220 ----
    mdagg_full = pd.read_csv(MDAGG_CSV)
    mdagg_df = mdagg_full[mdagg_full["q_ref_kjmol"].isin(Q_REF_MANUSCRIPT)].reset_index(drop=True)
    print(f"\n=== 3. Spearman(M_D_agg,Θ),27条件,取自 {MDAGG_CSV} ===")
    print(mdagg_df.to_string(index=False))
    if len(mdagg_df) != len(Q_REF_MANUSCRIPT):
        print(f"警告:{MDAGG_CSV} 里缺少 {Q_REF_MANUSCRIPT} 中的某些 q_ref_kjmol 行,"
              "请先重跑 scripts/26_theta_qref_sensitivity.py 再跑本脚本。")

    write_report(rank_df, precursor_df, mdagg_df, n_nano=len(nano))


def write_report(rank_df, precursor_df, mdagg_df, n_nano):
    # 检验1:两对相邻Q_ref排序是否都完美/接近完美稳定
    rank_stable = (rank_df["spearman_rho"] > 0.99).all()

    # 检验2:D_XRD~Θ 逐前驱体子组,符号是否跨 Q_ref、跨前驱体一致
    pivot = precursor_df.pivot(index="precursor", columns="q_ref_kjmol", values="spearman_rho")
    sign_stable_by_group = {}
    for prec in pivot.index:
        row = pivot.loc[prec].dropna()
        sign_stable_by_group[prec] = bool((np.sign(row) == np.sign(row.iloc[0])).all()) if len(row) else None
    all_sign_positive = bool((precursor_df["spearman_rho"].dropna() > 0).all())

    # 检验3:M_D_agg~Θ 符号/显著性稳定
    if len(mdagg_df):
        sign_stable_mdagg = bool((np.sign(mdagg_df["spearman_rho"]) == np.sign(mdagg_df["spearman_rho"].iloc[0])).all())
        sig_stable_mdagg = bool((mdagg_df["p"] < 0.05).all() or (mdagg_df["p"] >= 0.05).all())
    else:
        sign_stable_mdagg, sig_stable_mdagg = None, None

    fallback_note = (
        "" if HAVE_TABULATE else
        "> 注:本环境未安装 `tabulate` 包,`DataFrame.to_markdown()` 不可用,"
        "以下表格改用 `df.to_string(index=False)` 包在代码块中呈现(内容等价,"
        "仅渲染形式从 Markdown 表格换成等宽文本块)。\n\n"
    )

    lines = [
        "# Θ 对 Q_ref 敏感性——稿件引用范围 {180,200,220} kJ/mol 专表\n\n",
        "> 由 scripts/33_theta_qref_manuscript_range_table.py 自动生成,"
        "聚焦稿件正文实际引用的 Q_ref∈{180,200,220} kJ/mol 活化能敏感性范围"
        "(区别于 scripts/26_theta_qref_sensitivity.py 的完整 7 点扫描 "
        "{150,175,180,200,220,225,250});两种结论(稳健/不稳健)都如实报告,"
        "不为凑'稳健'而挑选前驱体子组或收窄取值。\n\n",
        fallback_note,
        "## 1. 27条件Θ排序稳定性({180,200,220}相邻对,Spearman ρ)\n\n",
        _table_block(rank_df),
        f"排序稳定(ρ>0.99):{'是' if rank_stable else '否'}\n\n",
        f"## 2. Spearman(D_XRD,Θ) 按前驱体拆分,{n_nano}样纳米层口径"
        "(xrd_instrument==1 且 D_XRD_reliable==True)\n\n",
        "> ALL(51) 行为池化对照(与26号脚本口径一致),S/M/L 三行为按前驱体子组"
        "(各约17样)——这是26号脚本没有算过的拆分。\n\n",
        _table_block(precursor_df),
        "各前驱体子组符号是否跨 Q_ref∈{180,200,220} 稳定:"
        + "、".join(f"{k}={'是' if v else ('数据不足' if v is None else '否')}" for k, v in sign_stable_by_group.items())
        + f";全部(含池化)行是否同号(均为正):{'是' if all_sign_positive else '否'}\n\n",
        "## 3. Spearman(M_D_agg,Θ),27条件\n\n",
        _table_block(mdagg_df) if len(mdagg_df) else "(未能从 data/interim/theta_qref_sensitivity_mdagg.csv 抽到 180/200/220 三行,请先重跑26号脚本)\n\n",
        f"符号稳定:{'是' if sign_stable_mdagg else ('未知' if sign_stable_mdagg is None else '否')};"
        f"显著性判定(p<0.05)稳定:{'是' if sig_stable_mdagg else ('未知' if sig_stable_mdagg is None else '否')}\n\n",
        "## 结论\n\n",
        f"在稿件引用的 Q_ref∈{{180,200,220}} kJ/mol 范围内:"
        f"27条件Θ排序{'保持' if rank_stable else '未能保持'}近乎完美稳定;"
        f"Spearman(D_XRD,Θ) 按前驱体拆分后{'各子组和池化结果符号一致(均为正)' if all_sign_positive else '子组间符号并不完全一致'};"
        f"Spearman(M_D_agg,Θ){'符号与显著性均稳定' if (sign_stable_mdagg and sig_stable_mdagg) else '并非在符号和显著性上都稳定'}。"
        "完整 7 点扫描(含 150/175/225/250 更极端的 Q_ref)见 "
        "`reports/theta_qref_sensitivity.md`。\n",
    ]
    Path(OUT_REPORT).write_text("".join(lines), encoding="utf-8")
    print(f"\n已写出 {OUT_REPORT}")


if __name__ == "__main__":
    main()
