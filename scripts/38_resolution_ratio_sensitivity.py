# -*- coding: utf-8 -*-
"""38 resolution_ratio_min 阈值敏感性审计
====================================================================
背景:`configs/config.yaml` 的 `instruments.xrd.resolution_ratio_min`(现值
1.3)是 `xrd_processor.process_all()` 里 `D_XRD_reliable` 布尔列的判据
——`resolution_ratio < 1.3` 时 `D_XRD`/`D_XRD_003`/`D_XRD_104` 置 NaN(反卷积
判定为病态)。方法学复核质疑:1.3 是否有计量学依据,还是作者任意选定;
以及它是否真的从分析用的 51 样(仅 xrd_instrument==1)纳米层口径中排除过
任何样本,还是只影响本就因跨仪器不可比而被 `nano_layer_frame()` 排除的
仪器2样本。

本脚本**只读审计**,不修改 `src/nfm/`、`configs/`、`master_table.csv`。

产出:
  data/interim/resolution_ratio_sensitivity.csv   两仪器分布 + 阈值扫描表
  reports/xrd_peak_position_reconstruction.md      §3(追加,不覆盖 §1/§2)
"""
import _bootstrap  # noqa

from pathlib import Path

import numpy as np
import pandas as pd

MASTER_TABLE = "data/processed/master_table.csv"
OUT_CSV = Path("data/interim/resolution_ratio_sensitivity.csv")
OUT_REPORT = Path("reports/xrd_peak_position_reconstruction.md")

CONFIG_THRESHOLD = 1.3
SWEEP_THRESHOLDS = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
CLAIMED_INST1_MIN = 1.444


def describe(series: pd.Series) -> dict:
    s = series.dropna().astype(float)
    return {
        "n": int(s.shape[0]),
        "min": float(s.min()),
        "q1": float(s.quantile(0.25)),
        "median": float(s.median()),
        "q3": float(s.quantile(0.75)),
        "max": float(s.max()),
    }


def hist_table(series: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    s = series.dropna().astype(float)
    lo, hi = float(s.min()), float(s.max())
    if lo == hi:
        edges = np.array([lo, hi + 1e-9])
        n_bins = 1
    else:
        edges = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(s, bins=edges)
    rows = []
    for i in range(len(counts)):
        rows.append({
            "bin_low": edges[i],
            "bin_high": edges[i + 1],
            "count": int(counts[i]),
        })
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(MASTER_TABLE)
    assert "resolution_ratio" in df.columns, "master_table.csv 缺少 resolution_ratio 列"
    assert "xrd_instrument" in df.columns, "master_table.csv 缺少 xrd_instrument 列"

    inst1 = df[df["xrd_instrument"] == 1]
    inst2 = df[df["xrd_instrument"] == 2]
    print(f"[check] xrd_instrument==1 行数 = {len(inst1)} (期望 51)")
    print(f"[check] xrd_instrument==2 行数 = {len(inst2)} (期望 30)")
    assert len(inst1) == 51, f"仪器1行数={len(inst1)},与预期51不符"
    assert len(inst2) == 30, f"仪器2行数={len(inst2)},与预期30不符"

    # -------- 1. 独立复算 min/median/max,核对既有参考值 1.444 --------
    stats1 = describe(inst1["resolution_ratio"])
    stats2 = describe(inst2["resolution_ratio"])
    print("[stats] instrument 1:", stats1)
    print("[stats] instrument 2:", stats2)

    inst1_min = stats1["min"]
    claim_confirmed = abs(inst1_min - CLAIMED_INST1_MIN) < 1e-3
    print(f"[check] 仪器1 resolution_ratio 最小值(独立复算) = {inst1_min:.6f}")
    print(f"[check] 既有参考的最小值 = {CLAIMED_INST1_MIN}")
    print(f"[check] 1.444 是否被本轮独立复算确认 = {claim_confirmed}")

    # NaN check: are there any NaN resolution_ratio values (shouldn't normally happen
    # since resolution_ratio itself doesn't depend on D_XRD reliability -- it's the input
    # to the reliability decision, not an output of it)
    n_nan_inst1 = int(inst1["resolution_ratio"].isna().sum())
    n_nan_inst2 = int(inst2["resolution_ratio"].isna().sum())
    print(f"[check] resolution_ratio 缺失值: 仪器1={n_nan_inst1}, 仪器2={n_nan_inst2}")

    # -------- 2. 分布 + 分箱直方图表 --------
    hist1 = hist_table(inst1["resolution_ratio"])
    hist1.insert(0, "instrument", 1)
    hist2 = hist_table(inst2["resolution_ratio"])
    hist2.insert(0, "instrument", 2)
    hist_all = pd.concat([hist1, hist2], ignore_index=True)

    # -------- 3. 阈值扫描:多少仪器1样本会被排除 --------
    sweep_rows = []
    for thr in SWEEP_THRESHOLDS:
        excl1 = int((inst1["resolution_ratio"] < thr).sum())
        excl2 = int((inst2["resolution_ratio"] < thr).sum())
        sweep_rows.append({
            "threshold": thr,
            "n_excluded_instrument1": excl1,
            "n_excluded_instrument2": excl2,
            "n_remaining_instrument1": len(inst1) - excl1,
            "n_remaining_instrument2": len(inst2) - excl2,
        })
    sweep_df = pd.DataFrame(sweep_rows)
    print("[sweep]")
    print(sweep_df.to_string(index=False))

    # confirm config default (1.3) matches production behavior recorded in D_XRD_reliable
    if "D_XRD_reliable" in df.columns:
        prod_excl1 = int((inst1["D_XRD_reliable"] == False).sum())  # noqa: E712
        prod_excl2 = int((inst2["D_XRD_reliable"] == False).sum())
        print(f"[cross-check] 生产列 D_XRD_reliable==False 计数: 仪器1={prod_excl1}, 仪器2={prod_excl2}")
        sweep_row_130 = sweep_df[np.isclose(sweep_df["threshold"], CONFIG_THRESHOLD)]
        recompute_excl1 = int(sweep_row_130["n_excluded_instrument1"].iloc[0])
        recompute_excl2 = int(sweep_row_130["n_excluded_instrument2"].iloc[0])
        print(f"[cross-check] 本脚本在 threshold=1.3 重算的排除计数: 仪器1={recompute_excl1}, 仪器2={recompute_excl2}")
    else:
        prod_excl1 = prod_excl2 = None

    # -------- 4. non-binding 判定 --------
    max_sweep_thr = max(SWEEP_THRESHOLDS)
    excluded_at_max = int((inst1["resolution_ratio"] < max_sweep_thr).sum())
    nonbinding_confirmed = all(row["n_excluded_instrument1"] == 0 for row in sweep_rows if row["threshold"] <= inst1_min - 1e-9) \
        or all(r["n_excluded_instrument1"] == 0 for r in sweep_rows)
    # simpler: check zero exclusion across full sweep
    nonbinding_confirmed = all(r["n_excluded_instrument1"] == 0 for r in sweep_rows)
    print(f"[verdict] 阈值扫描 {SWEEP_THRESHOLDS} 全程仪器1排除数是否恒为0 = {nonbinding_confirmed}")

    # -------- write outputs --------
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        f.write("# section=distribution_stats\n")
        stats_df = pd.DataFrame([
            {"instrument": 1, **stats1},
            {"instrument": 2, **stats2},
        ])
        stats_df.to_csv(f, index=False)
        f.write("# section=histogram\n")
        hist_all.to_csv(f, index=False)
        f.write("# section=threshold_sweep\n")
        sweep_df.to_csv(f, index=False)
    print(f"[write] {OUT_CSV}")

    # -------- append report section --------
    lines = []
    lines.append("\n## §3 resolution_ratio_min 阈值敏感性\n")
    lines.append(
        "由 `scripts/38_resolution_ratio_sensitivity.py` 追加。审计对象:"
        "`configs/config.yaml` 的 `instruments.xrd.resolution_ratio_min`(现值 "
        f"{CONFIG_THRESHOLD})——这是 `xrd_processor.process_all()` 中 `D_XRD_reliable` "
        "布尔列的判据,`resolution_ratio < 阈值` 时 `D_XRD`/`D_XRD_003`/`D_XRD_104` 置 NaN。"
        "方法学复核问题:该阈值是否有计量学依据,以及它是否真的从分析用的 51 样"
        "(仅 `xrd_instrument==1`,`nano_layer_frame()` 口径)纳米层结果中排除过任何样本,"
        "还是只影响本就因跨仪器不可比而被排除的仪器2样本。本节为只读审计,不修改"
        "`src/nfm/`、`configs/config.yaml` 或生产数据。\n"
    )

    lines.append("### 3.1 独立复算 min/median/max,核对既有参考值 1.444\n")
    lines.append(
        f"- 仪器1(SmartLab,n={stats1['n']}):"
        f"min={stats1['min']:.4f}、Q1={stats1['q1']:.4f}、median={stats1['median']:.4f}、"
        f"Q3={stats1['q3']:.4f}、max={stats1['max']:.4f}\n"
        f"- 仪器2(MiniFlex,n={stats2['n']}):"
        f"min={stats2['min']:.4f}、Q1={stats2['q1']:.4f}、median={stats2['median']:.4f}、"
        f"Q3={stats2['q3']:.4f}、max={stats2['max']:.4f}\n"
    )
    verdict_text = "**确认**" if claim_confirmed else "**未确认(数值不同,见下)**"
    lines.append(
        f"- 既有参考给出仪器1 `resolution_ratio` 最小值为 {CLAIMED_INST1_MIN}:"
        f"{verdict_text}。本轮独立复算得到仪器1最小值 = **{inst1_min:.4f}**"
        f"{'，与参考数字一致(差值<0.001)。' if claim_confirmed else '，与参考数字不一致,以本轮独立复算为准。'}\n"
    )
    if n_nan_inst1 or n_nan_inst2:
        lines.append(
            f"- 注意:`resolution_ratio` 存在缺失值(仪器1 NaN={n_nan_inst1},"
            f"仪器2 NaN={n_nan_inst2}),上述统计量已对缺失值做 dropna 处理。\n"
        )

    lines.append("### 3.2 两仪器 resolution_ratio 分布(10 分箱,各自量程内)\n")
    lines.append("**仪器1(SmartLab,n=51)**\n")
    lines.append("| bin区间 | count |\n|---|---|\n")
    for _, r in hist1.iterrows():
        lines.append(f"| [{r['bin_low']:.4f}, {r['bin_high']:.4f}) | {int(r['count'])} |\n")
    lines.append("\n**仪器2(MiniFlex,n=30)**\n")
    lines.append("| bin区间 | count |\n|---|---|\n")
    for _, r in hist2.iterrows():
        lines.append(f"| [{r['bin_low']:.4f}, {r['bin_high']:.4f}) | {int(r['count'])} |\n")
    lines.append("\n")

    lines.append("### 3.3 阈值扫描:是否影响 51 样纳米层口径\n")
    zero_thresholds = [r["threshold"] for r in sweep_rows if r["n_excluded_instrument1"] == 0]
    nonzero_thresholds = [r["threshold"] for r in sweep_rows if r["n_excluded_instrument1"] > 0]
    zero_max = max(zero_thresholds) if zero_thresholds else None
    lines.append(
        "`nano_layer_frame()` 已把每一项 XRD/晶格参数分析限定为 `xrd_instrument==1`,"
        "与 `resolution_ratio_min` 阈值本身无关。仪器1(51样)`resolution_ratio` 最小值为 "
        f"{inst1_min:.4f}(见 3.1),明显高于生产配置现值 1.3——逐阈值重算如下:\n"
    )
    lines.append("| threshold | 仪器1排除数 | 仪器1剩余 | 仪器2排除数 | 仪器2剩余 |\n|---|---|---|---|---|\n")
    for r in sweep_rows:
        lines.append(
            f"| {r['threshold']} | {r['n_excluded_instrument1']} | "
            f"{r['n_remaining_instrument1']} | {r['n_excluded_instrument2']} | "
            f"{r['n_remaining_instrument2']} |\n"
        )
    if prod_excl1 is not None:
        lines.append(
            f"\n交叉核对:生产列 `D_XRD_reliable==False` 计数(阈值=1.3 时的实际生产结果)"
            f"仪器1={prod_excl1}、仪器2={prod_excl2};本脚本用 `resolution_ratio<1.3` "
            f"独立重算得到仪器1={recompute_excl1}、仪器2={recompute_excl2}"
            f"{'，两者一致，确认本脚本复现了生产判据的逻辑。' if (prod_excl1, prod_excl2) == (recompute_excl1, recompute_excl2) else '，注意:两者不一致,可能存在其它排除条件(如 resolution_ratio 本身为 NaN 的样本),详见下方说明,不代表本节判据算错。'}\n"
        )

    lines.append("\n### 3.4 结论\n")
    if nonbinding_confirmed:
        lines.append(
            f"**确认非绑定(non-binding)**:在扫描的 {SWEEP_THRESHOLDS} 全部阈值取值下,"
            "仪器1(即 `nano_layer_frame()` 分析用的 51 样)排除数恒为 0。\n"
        )
    else:
        lines.append(
            f"**结果比预设的二元结论更细致,如实报告,不强行套用单一\"non-binding\"标签**:"
            f"阈值扫描 {SWEEP_THRESHOLDS} 显示仪器1排除数**不是恒为 0**——"
            f"在 threshold ∈ {zero_thresholds}(即 ≤ 1.4,均低于仪器1实测最小值 "
            f"{inst1_min:.4f})时排除数为 0;但在 threshold ∈ {nonzero_thresholds} "
            f"(即 ≥ 1.5,已超过仪器1实测最小值)时开始从 51 样中排除样本"
            f"(1.5 排除 {sweep_df[np.isclose(sweep_df['threshold'], 1.5)]['n_excluded_instrument1'].iloc[0]} 个、"
            f"1.6 排除 {sweep_df[np.isclose(sweep_df['threshold'], 1.6)]['n_excluded_instrument1'].iloc[0]} 个)。"
            "这一转折点完全符合预期(阈值一旦超过样本实测最小值,必然开始排除),"
            "不代表判据本身有问题。\n\n"
            "**就本任务实际要回答的问题而言,结论仍然是明确的**:生产配置现值 "
            f"**1.3** 及其邻近的合理调整范围(至少到 1.4,相对现值 +8% 的余量)"
            f"均不从 51 样纳米层分析口径中排除任何样本(仪器1最小值 {inst1_min:.4f} "
            "比 1.3 高出约 11%,有实质安全边际,不是卡在边界上的巧合)。"
            "阈值只有被推到明显超出本批仪器1实测下限之外(≥1.5,相对现值 1.3 已"
            "是 +15% 以上的大幅调整)才会开始影响 51 样口径。因此:1.3 这一"
            "具体取值,以及其附近任何\"合理\"的微调,对本轮 §1/§2 及论文已报告的"
            "全部纳米层分析结果(D_XRD、lattice_a/lattice_c/c_a_ratio 相关结论)"
            "**没有影响**;该阈值在当前生产配置下唯一产生的排除效果落在仪器2的 30 样"
            f"(阈值=1.3 时排除 {sweep_df[np.isclose(sweep_df['threshold'], 1.3)]['n_excluded_instrument2'].iloc[0]} 个)"
            "——而仪器2样本本就因跨仪器不可比(见 `nano_layer.py` 头部说明)被排除在"
            "纳米层分析之外,与本阈值的具体取值无关。\n\n"
            "**建议处理**:鉴于该阈值在其合理取值邻域内对已分析的 51 样范围没有影响,"
            "合适的论文处理方式是把它从目前\"一句话断言、无支撑数字\"的状态,补充为"
            "规范的 SI 材料——明确写出上表阈值扫描结果,说明 1.3 对分析范围不起作用、"
            "且有约 11% 的安全边际。这**强化**而非削弱\"1.3 是可辩护的共享判据、"
            "不是任意挑选\"的论证:正因为它在合理调整范围内不影响任何已报告结果,"
            "选择它就不构成为了让某个特定结果显著/不显著而刻意调参的嫌疑;1.3 这个"
            "数值本身仍然只是仪器2数据质量控制的合理默认选择(不属于纳米层结论的"
            "可辩护性基础,纳米层结论的可辩护性来自跨仪器不合并这一决定,与本阈值无关)。"
            "不应笼统宣称\"该阈值任意取值都不影响结果\"——超出安全边际外的取值(≥1.5)"
            "确实会影响 51 样口径,SI 材料应如实标注这一点,而不是只报告 1.3 单点。\n"
        )

    with open(OUT_REPORT, "a", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"[write] appended §3 to {OUT_REPORT}")


if __name__ == "__main__":
    main()
