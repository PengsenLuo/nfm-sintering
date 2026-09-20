# -*- coding: utf-8 -*-
"""26 Θ 对 Q_ref 的敏感性稳健性检验
====================================================================
防御性检验:审稿人可能问"Θ 固定 Q_ref=200 kJ/mol,若改变会怎样"。本脚本
用 Q_ref ∈ {150,175,180,200,220,225,250} kJ/mol 逐个重算 Θ(2026-08-16 补充
180/220 两点,原 150/175/225/250 全部保留,以直接覆盖稿件正文引用的
Q_ref∈{180,200,220} kJ/mol 活化能范围,而非仅靠 175/225 括住;复用
nfm.features.thermal_exposure.theta,R/T_ref_C/integrate_from_C 仍取自
configs/config.yaml,只有 Q_ref 被扫描——不修改 config.yaml 的运行默认值),
报告:
  1. 27 条件 Θ 排序在相邻 Q_ref 间的 Spearman ρ(格式对照
     data/interim/theta_calibration_rank_stability.csv)
  2. Spearman(D_XRD, Θ)(51样纳米层口径)符号/显著性是否稳健
  3. Spearman(M_D_agg, Θ)(27条件)符号/显著性是否稳健
  4. 表1 compaction_density/D_XRD 两行主效应 η² 是否随 Q_ref 变化
     (结构性不依赖 Θ,零计算,直接说明原因)

两种结论(稳健/不稳健)都要如实报告,不为了让"稳健"成立而收窄扫描范围。

输出:data/interim/theta_qref_sensitivity.csv,
     data/interim/theta_qref_sensitivity_dxrd.csv,
     data/interim/theta_qref_sensitivity_mdagg.csv,
     reports/theta_qref_sensitivity.md
"""
import _bootstrap  # noqa

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from nfm.config import load_config
from nfm.features.thermal_exposure import theta
from nfm.nano_layer import nano_layer_frame

try:
    import tabulate  # noqa: F401
    HAVE_TABULATE = True
except ImportError:
    HAVE_TABULATE = False

Q_REF_SWEEP_KJMOL = [150, 175, 180, 200, 220, 225, 250]
MASTER_TABLE = "data/processed/master_table.csv"
OUT_CSV = "data/interim/theta_qref_sensitivity.csv"
OUT_REPORT = "reports/theta_qref_sensitivity.md"


def compute_theta_for_qref(df, q_ref_kjmol, cfg):
    R = cfg.R
    T_ref_C = cfg.raw["physical_constants"]["T_ref_C"]
    ifrom = cfg.raw["thermal_exposure"]["integrate_from_C"]
    start_C = 550.0 if ifrom == "auto" else float(ifrom)
    Q_J = q_ref_kjmol * 1e3
    return np.array([
        theta(r.T_C, r.beta, r.t_hold, Q_J=Q_J, R=R, T_ref_C=T_ref_C, start_C=start_C)
        for r in df.itertuples()
    ])


def main():
    cfg = load_config()
    df = pd.read_csv(MASTER_TABLE)

    # ---- 每个 Q_ref 算一版 Theta,存回 df 的临时列 ----
    theta_cols = {}
    for q in Q_REF_SWEEP_KJMOL:
        theta_cols[q] = compute_theta_for_qref(df, q, cfg)
        df[f"Theta_q{q}"] = theta_cols[q]

    # ---- 检验1:27条件排序稳定性(相邻Q_ref对) ----
    cond = df.drop_duplicates("condition_id").set_index("condition_id")
    rank_rows = []
    for i in range(len(Q_REF_SWEEP_KJMOL) - 1):
        q1, q2 = Q_REF_SWEEP_KJMOL[i], Q_REF_SWEEP_KJMOL[i + 1]
        rho, _ = spearmanr(cond[f"Theta_q{q1}"], cond[f"Theta_q{q2}"])
        rank_rows.append({"q_ref_pair": f"{q1}-{q2}", "spearman_rho": rho})
    rank_df = pd.DataFrame(rank_rows)
    print("=== 检验1:27条件Θ排序稳定性(相邻Q_ref对) ===")
    print(rank_df.to_string(index=False))

    # ---- 检验2:Spearman(D_XRD, Theta) 51样纳米层口径,逐Q_ref ----
    nano = nano_layer_frame(df, require_reliable=True)
    dxrd_rows = []
    for q in Q_REF_SWEEP_KJMOL:
        rho, p = spearmanr(nano[f"Theta_q{q}"], nano["D_XRD"])
        dxrd_rows.append({"q_ref_kjmol": q, "spearman_rho": rho, "p": p})
    dxrd_df = pd.DataFrame(dxrd_rows)
    print(f"\n=== 检验2:Spearman(D_XRD,Θ),{len(nano)}样,逐Q_ref ===")
    print(dxrd_df.to_string(index=False))

    # ---- 检验3:Spearman(M_D_agg, Theta) 27条件,逐Q_ref ----
    mdagg = df.drop_duplicates("condition_id")[["condition_id", "M_D_agg"] + [f"Theta_q{q}" for q in Q_REF_SWEEP_KJMOL]]
    mdagg_rows = []
    for q in Q_REF_SWEEP_KJMOL:
        rho, p = spearmanr(mdagg[f"Theta_q{q}"], mdagg["M_D_agg"])
        mdagg_rows.append({"q_ref_kjmol": q, "spearman_rho": rho, "p": p})
    mdagg_df = pd.DataFrame(mdagg_rows)
    print("\n=== 检验3:Spearman(M_D_agg,Θ),27条件,逐Q_ref ===")
    print(mdagg_df.to_string(index=False))

    # ---- 检验4:表1 η² 不依赖 Θ(零计算说明) ----
    print("\n=== 检验4:表1 compaction_density/D_XRD 主效应 η² ===")
    print("Table 1's ANOVA uses T_C/beta/t_hold/precursor as raw categorical "
          "factors directly, never Theta as a regressor — these two rows are "
          "structurally invariant to Q_ref by construction, not empirically "
          "checked here.")

    rank_df.to_csv(OUT_CSV, index=False)
    dxrd_df.to_csv(OUT_CSV.replace(".csv", "_dxrd.csv"), index=False)
    mdagg_df.to_csv(OUT_CSV.replace(".csv", "_mdagg.csv"), index=False)

    write_report(rank_df, dxrd_df, mdagg_df, n_nano=len(nano))


def _table_block(df: pd.DataFrame) -> str:
    """Markdown table if tabulate available, else fenced to_string() block
    (fallback noted in report body since 'tabulate' package is not installed
    in this environment)."""
    if HAVE_TABULATE:
        return df.to_markdown(index=False) + "\n\n"
    return "```\n" + df.to_string(index=False) + "\n```\n\n"


def write_report(rank_df, dxrd_df, mdagg_df, n_nano):
    sign_stable_dxrd = (np.sign(dxrd_df["spearman_rho"]) == np.sign(dxrd_df["spearman_rho"].iloc[0])).all()
    sig_stable_dxrd = (dxrd_df["p"] < 0.05).all() or (dxrd_df["p"] >= 0.05).all()
    sign_stable_mdagg = (np.sign(mdagg_df["spearman_rho"]) == np.sign(mdagg_df["spearman_rho"].iloc[0])).all()
    sig_stable_mdagg = (mdagg_df["p"] < 0.05).all() or (mdagg_df["p"] >= 0.05).all()

    fallback_note = (
        "" if HAVE_TABULATE else
        "> 注:本环境未安装 `tabulate` 包,`DataFrame.to_markdown()` 不可用,"
        "以下表格改用 `df.to_string(index=False)` 包在代码块中呈现(内容等价,"
        "仅渲染形式从 Markdown 表格换成等宽文本块)。\n\n"
    )

    lines = ["# Θ 对 Q_ref 的敏感性稳健性检验\n\n",
            "> 由 scripts/26_theta_qref_sensitivity.py 自动生成。防御性检验,"
            "两种结论(稳健/不稳健)都如实报告,不为凑'稳健'而收窄扫描范围。\n\n",
            fallback_note,
            "## 1. 27条件Θ排序稳定性(相邻Q_ref对,Spearman ρ)\n\n",
            _table_block(rank_df),
            f"## 2. Spearman(D_XRD,Θ),{n_nano}样纳米层口径(xrd_instrument==1 且 D_XRD_reliable==True)\n\n",
            _table_block(dxrd_df),
            f"符号稳定:{'是' if sign_stable_dxrd else '否'};显著性判定(p<0.05)稳定:"
            f"{'是' if sig_stable_dxrd else '否'}\n\n",
            "## 3. Spearman(M_D_agg,Θ),27条件\n\n",
            _table_block(mdagg_df),
            f"符号稳定:{'是' if sign_stable_mdagg else '否'};显著性判定(p<0.05)稳定:"
            f"{'是' if sig_stable_mdagg else '否'}\n\n",
            "## 4. 表1 compaction_density/D_XRD 主效应 η²\n\n",
            "表1的ANOVA设计直接用 T_C/beta/t_hold/precursor 作为原始类别型因子,"
            "从不把 Θ 作为回归量(独立核实见 `src/nfm/stats/anova_interactions.py` "
            "`FACTORS = (\"precursor\", \"T_C\", \"beta\", \"t_hold\")` / `FORMULA_RHS`,"
            "以及 `table1.compaction_density.*` "
            "行的 `source_script=scripts/12_anova_interactions.py`、"
            "`table1.D_XRD.*` 行的 `source_script=scripts/19_nano_layer_interaction_anova.py`,"
            "两个脚本的设计矩阵均不含 Theta)——这两行结构性地不依赖 Q_ref 的选择,"
            "这是设计层面的事实,不是本次扫描检验出来的结果。\n\n",
            "## 结论\n\n",
            f"检验1-3{'全部' if (sign_stable_dxrd and sig_stable_dxrd and sign_stable_mdagg and sig_stable_mdagg) else '并非全部'}"
            "在 Q_ref∈[150,250] kJ/mol 范围内保持稳健(具体见上表)。"
            "检验4为结构性不依赖,恒稳健。\n"]
    from pathlib import Path
    Path(OUT_REPORT).write_text("".join(lines), encoding="utf-8")
    print(f"\n已写出 {OUT_REPORT}")


if __name__ == "__main__":
    main()
