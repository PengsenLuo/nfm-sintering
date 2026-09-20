# -*- coding: utf-8 -*-
"""35 XRD 二次零点三种校正口径并行对比(W1.2)
====================================================================
背景:见 `scripts/34_xrd_zero_offset_diagnostic.py`(W1.1)诊断——自由参数
`z`(生产 `lattice_params()` 的第二次零点修正)与 Θ/T_C/D_XRD 系统性相关,
逐 hkl 残差分析显示该相关性主要由 (003) 峰一个峰的系统性偏差驱动(003 中位
残差 -0.0115°,是其余 9 个峰的 5-20 倍),既不干净地符合"真实样品高度位移"
(cosθ 依赖在 00l 系与非 00l 系之间方向相反),也不干净地符合"宽化依赖峰位
偏置"(|residual| 与 FWHM/D_XRD 相关系数≈0)。本脚本不预设诊断结论,只做
三种校正口径的**并行实现与对比**,用 §1.2 的判定规则决定生产实现是否需要动。

三种口径(**只读复算,不替换生产 `xrd_processor.lattice_params()`**):
  C0(现行):a,c,z 三参数最小二乘(z 自由,bounds ±0.2°)—— 与生产完全一致的
    独立重实现,先做交叉核对确认重实现正确。
  C1:a,c 两参数最小二乘,z 固定为 0(不做第二次零点修正)。
  C2:a,c,k 三参数最小二乘,z 替换为逐峰依赖 cosθ 的位移项
    Δ2θ = k·cosθ_obs(Bragg-Brentano 样品高度位移误差的标准形式
    Δ2θ=-(2s/R)·cosθ;s=样品位移、R=测角仪半径二者简并为单一系数 k,
    本脚本**未能从 config.yaml 或原始文件头找到 SmartLab 实测测角仪半径 R**
    ——只拟合组合系数 k=-(2s/R),不单独反解 s,如实标注此限制)。

数据源:`data/interim/xrd_hkl_residuals.csv`(W1.1 已产出,51 样×10 hkl 的
逐峰 `two_theta_obs`,复用避免重新拟合)。

判定规则(硬性):
  若 C1/C2 下 `lattice_c`/`lattice_a` 的主效应 η² 与 Θ 相关性(方向、显著性)
  与 C0 保持同量级 → 现有结论稳健,只需 SI 补一段口径说明,生产实现维持 C0。
  若任一变体出现方向反转或显著性丧失 → **立即停止并汇报,不擅自替换**,
  这需要项目负责人决定,不是本脚本的判断权限。

产出:
  data/interim/xrd_calibration_variants.csv       51样×3口径的 a/c/c_a_ratio/
    残差RMS
  reports/xrd_peak_position_reconstruction.md      §2(追加,不覆盖 §1)
"""
import _bootstrap  # noqa

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import least_squares
from scipy.stats import spearmanr
from statsmodels.stats.anova import anova_lm

from nfm.config import load_config

RESID_CSV = Path("data/interim/xrd_hkl_residuals.csv")
MASTER_TABLE = "data/processed/master_table.csv"
OUT_CSV = Path("data/interim/xrd_calibration_variants.csv")
OUT_REPORT = Path("reports/xrd_peak_position_reconstruction.md")

HKL_TABLE = {"003": (0, 0, 3), "006": (0, 0, 6), "101": (1, 0, 1),
            "012": (0, 1, 2), "104": (1, 0, 4), "015": (0, 1, 5),
            "107": (1, 0, 7), "018": (0, 1, 8), "110": (1, 1, 0),
            "113": (1, 1, 3)}
FACTORS = ("precursor", "T_C", "beta", "t_hold")


def _fit_variant(peaks_2theta: dict, lam_A: float, mode: str):
    """mode in {"c0","c1","c2"}. 返回 (a, c, z_or_k, resid_rms_deg)。"""
    def d_from(two_theta_deg, z_eff_deg):
        th = np.radians((two_theta_deg - z_eff_deg) / 2.0)
        return lam_A / (2.0 * np.sin(th))

    hkls = list(peaks_2theta.keys())
    tts = np.array([peaks_2theta[h] for h in hkls])
    hkl_idx = [HKL_TABLE[h] for h in hkls]

    def resid(params):
        if mode == "c0":
            a, c, z = params
            z_eff = np.full_like(tts, z)
        elif mode == "c1":
            a, c = params
            z_eff = np.zeros_like(tts)
        elif mode == "c2":
            a, c, k = params
            th_obs = np.radians(tts / 2.0)
            z_eff = k * np.cos(th_obs)
        else:
            raise ValueError(mode)
        d = d_from(tts, z_eff)
        inv_d2_obs = 1.0 / d ** 2
        r = []
        for (h, k_, l), obs in zip(hkl_idx, inv_d2_obs):
            inv_d2_cal = 4.0 / 3.0 * (h * h + h * k_ + k_ * k_) / a ** 2 + l * l / c ** 2
            r.append(obs - inv_d2_cal)
        return np.asarray(r)

    if mode == "c1":
        x0, lo, hi = [2.98, 16.0], [2.8, 15.5], [3.1, 16.6]
    else:
        x0, lo, hi = [2.98, 16.0, 0.0], [2.8, 15.5, -0.2], [3.1, 16.6, 0.2]

    sol = least_squares(resid, x0=x0, bounds=(lo, hi))
    if mode == "c1":
        a, c = sol.x
        z_or_k = 0.0
    else:
        a, c, z_or_k = sol.x

    # 残差 RMS(角度尺度,°)——2026-08-18(V2)修正:此前用最小二乘 cost
    # (1/d² 空间)除以数值差分导数"换算"回角度,这个换算本身有 bug
    # (mode!="c2" 分支复用了标量 z_or_k 而 mode=="c2" 分支错误地把角度
    # 依赖的 k·cosθ 项当 0 处理),导致 C0 的 RMS 报出 0.00001,比用
    # `residual_deg` 独立算出的 0.0056 小约 560 倍——不是"C0 拟合得特别好",
    # 是单位/换算搞错了。
    # 改为直接在角度空间算残差:residual_deg = 2θ_obs − 2θ_calc(a,c,z_eff,hkl),
    # 与 scripts/34_xrd_zero_offset_diagnostic.py 的 `_two_theta_calc` 完全
    # 同一套公式(z_eff 对 c0/c1 是标量,对 c2 是逐峰 k·cosθ_obs),不再经过
    # 1/d² 空间再换算,三种口径可直接横向比较。
    resid_deg_list = []
    for (h, k_, l), tt_obs in zip(hkl_idx, tts):
        if mode == "c2":
            th_obs = np.radians(tt_obs / 2.0)
            z_eff_peak = z_or_k * np.cos(th_obs)
        else:
            z_eff_peak = z_or_k  # c0: 拟合出的 z;c1: 恒为 0
        inv_d2_cal = 4.0 / 3.0 * (h * h + h * k_ + k_ * k_) / a ** 2 + l * l / c ** 2
        d_cal = 1.0 / np.sqrt(inv_d2_cal)
        s = np.clip(lam_A / (2.0 * d_cal), -1, 1)
        th_cal = np.degrees(np.arcsin(s))
        tt_calc = 2.0 * th_cal + z_eff_peak
        resid_deg_list.append(tt_obs - tt_calc)
    rms_deg = float(np.sqrt(np.mean(np.square(resid_deg_list))))

    return float(a), float(c), float(z_or_k), rms_deg


def main():
    cfg = load_config()
    lam1 = cfg.raw["instruments"]["xrd"]["wavelength_A"]

    resid_df = pd.read_csv(RESID_CSV, dtype={"hkl": str})
    master = pd.read_csv(MASTER_TABLE)
    meta = master.set_index("sample_id")[["condition_id", "precursor", "T_C", "beta",
                                          "t_hold", "Theta", "D_XRD",
                                          "lattice_a", "lattice_c"]]

    rows = []
    for sid, g in resid_df.groupby("sample_id"):
        peaks = dict(zip(g["hkl"], g["two_theta_obs"]))
        rec = dict(sample_id=sid)
        rec.update(meta.loc[sid].to_dict())
        for mode in ("c0", "c1", "c2"):
            a, c, zk, rms = _fit_variant(peaks, lam1, mode)
            rec[f"{mode}_a"] = a
            rec[f"{mode}_c"] = c
            rec[f"{mode}_c_a_ratio"] = c / a
            rec[f"{mode}_z_or_k"] = zk
            rec[f"{mode}_resid_rms_deg"] = rms
        rows.append(rec)

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"[done] {OUT_CSV}  n={len(out)}")

    # ---- 交叉核对:C0 重实现 vs master_table 既有 lattice_a/lattice_c ----
    diff_a = (out["c0_a"] - out["lattice_a"]).abs()
    diff_c = (out["c0_c"] - out["lattice_c"]).abs()
    print("\n=== C0 重实现交叉核对(vs master_table 既有列) ===")
    print(f"  lattice_a 最大绝对差: {diff_a.max():.6f}  平均绝对差: {diff_a.mean():.6f}")
    print(f"  lattice_c 最大绝对差: {diff_c.max():.6f}  平均绝对差: {diff_c.mean():.6f}")
    cross_check_ok = diff_a.max() < 1e-3 and diff_c.max() < 1e-3

    # ---- 三口径各自的分布 + 残差RMS + Spearman(lattice_*, Theta) + 主效应η² ----
    summary = {}
    work = out.copy()
    for f in FACTORS:
        work[f] = work[f].astype("category")

    for mode in ("c0", "c1", "c2"):
        a_col, c_col, ratio_col, rms_col = (f"{mode}_a", f"{mode}_c",
                                            f"{mode}_c_a_ratio", f"{mode}_resid_rms_deg")
        rho_a, p_a = spearmanr(out[a_col], out["Theta"])
        rho_c, p_c = spearmanr(out[c_col], out["Theta"])

        eta2 = {}
        for target_col in (a_col, c_col):
            rhs = " + ".join(f"C({f}, Sum)" for f in FACTORS)
            model = smf.ols(f"{target_col} ~ {rhs}", data=work).fit()
            tab = anova_lm(model, typ=2)
            tab = tab.rename(index={f"C({f}, Sum)": f for f in FACTORS})
            y = work[target_col].to_numpy(float)
            sst = float(((y - y.mean()) ** 2).sum())
            eta2[target_col] = {f: 100.0 * float(tab.loc[f, "sum_sq"]) / sst for f in FACTORS}
            eta2[target_col]["Residual"] = 100.0 * float(tab.loc["Residual", "sum_sq"]) / sst

        summary[mode] = dict(
            a_median=float(out[a_col].median()), a_std=float(out[a_col].std()),
            c_median=float(out[c_col].median()), c_std=float(out[c_col].std()),
            ratio_median=float(out[ratio_col].median()),
            resid_rms_median=float(out[rms_col].median()),
            rho_a_theta=float(rho_a), p_a_theta=float(p_a),
            rho_c_theta=float(rho_c), p_c_theta=float(p_c),
            eta2_a=eta2[a_col], eta2_c=eta2[c_col],
        )
        print(f"\n=== 口径 {mode.upper()} ===")
        print(f"  lattice_a median={summary[mode]['a_median']:.4f} std={summary[mode]['a_std']:.4f}")
        print(f"  lattice_c median={summary[mode]['c_median']:.4f} std={summary[mode]['c_std']:.4f}")
        print(f"  残差RMS(度) median={summary[mode]['resid_rms_median']:.5f}")
        print(f"  Spearman(lattice_a,Θ)={rho_a:+.4f}(p={p_a:.3e})  Spearman(lattice_c,Θ)={rho_c:+.4f}(p={p_c:.3e})")
        print(f"  lattice_a 主效应η%: " + ", ".join(f"{k}={v:.2f}" for k, v in eta2[a_col].items()))
        print(f"  lattice_c 主效应η%: " + ", ".join(f"{k}={v:.2f}" for k, v in eta2[c_col].items()))

    # ---- 判定规则 ----
    c0 = summary["c0"]
    verdicts = {}
    for mode in ("c1", "c2"):
        m = summary[mode]
        sign_a_ok = np.sign(m["rho_a_theta"]) == np.sign(c0["rho_a_theta"])
        sign_c_ok = np.sign(m["rho_c_theta"]) == np.sign(c0["rho_c_theta"])
        sig_a_ok = (m["p_a_theta"] < 0.05) == (c0["p_a_theta"] < 0.05)
        sig_c_ok = (m["p_c_theta"] < 0.05) == (c0["p_c_theta"] < 0.05)
        verdicts[mode] = dict(sign_a_ok=sign_a_ok, sign_c_ok=sign_c_ok,
                              sig_a_ok=sig_a_ok, sig_c_ok=sig_c_ok,
                              all_ok=bool(sign_a_ok and sign_c_ok and sig_a_ok and sig_c_ok))
    all_variants_robust = all(v["all_ok"] for v in verdicts.values())
    print("\n=== 判定 ===")
    for mode, v in verdicts.items():
        print(f"  {mode}: {v}")
    print(f"  全部变体稳健: {all_variants_robust}")
    if not all_variants_robust:
        print("  *** 触发'停下来汇报',不擅自替换生产实现 ***")

    append_report(out, cross_check_ok, diff_a, diff_c, summary, verdicts, all_variants_robust)
    print(f"\n[done,追加] {OUT_REPORT}")


def append_report(out, cross_check_ok, diff_a, diff_c, summary, verdicts, all_variants_robust):
    lines = []
    lines.append("\n## §2 三种校正口径并行对比(W1.2)\n\n")
    lines.append("由 `scripts/35_xrd_calibration_variants.py` 追加。三种口径均为独立"
                 "只读重实现(复用 §1 已产出的 `data/interim/xrd_hkl_residuals.csv` "
                 "逐峰 2θ,不重新拟合峰、不修改生产 `xrd_processor.lattice_params()`)。"
                 f"n={len(out)} 样本(51样纳米层口径)。\n\n")
    lines.append(f"**C0 重实现交叉核对**:lattice_a 最大绝对差 {diff_a.max():.6f}Å,"
                 f"lattice_c 最大绝对差 {diff_c.max():.6f}Å"
                 f"({'高精度吻合,确认重实现正确' if cross_check_ok else '**存在明显分歧,需排查**'})。\n\n")
    lines.append("**C2 说明**:未能从 `configs/config.yaml` 或原始 `.txt` 文件头找到 "
                 "SmartLab 实测测角仪半径 R;样品位移 s 与 R 在 "
                 "Δ2θ=-(2s/R)·cosθ 中简并为单一组合系数 k=-(2s/R),本报告只拟合并报告 "
                 "k,不单独反解物理位移量 s(需要仪器文档确认 R 才能做,超出本轮范围)。\n\n")
    lines.append("### 2.1 三口径分布 + 残差RMS + Spearman(lattice_*,Θ)\n\n")
    lines.append("| 口径 | a median±std | c median±std | c/a median | 残差RMS(°) | "
                 "ρ(a,Θ) | p | ρ(c,Θ) | p |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|\n")
    for mode in ("c0", "c1", "c2"):
        s = summary[mode]
        lines.append(f"| {mode.upper()} | {s['a_median']:.4f}±{s['a_std']:.4f} | "
                     f"{s['c_median']:.4f}±{s['c_std']:.4f} | {s['ratio_median']:.4f} | "
                     f"{s['resid_rms_median']:.5f} | {s['rho_a_theta']:+.4f} | {s['p_a_theta']:.3e} | "
                     f"{s['rho_c_theta']:+.4f} | {s['p_c_theta']:.3e} |\n")
    lines.append("\n### 2.2 主效应 η²(Type II,%),lattice_a / lattice_c,三口径对照\n\n")
    for target_label, key in (("lattice_a", "eta2_a"), ("lattice_c", "eta2_c")):
        lines.append(f"**{target_label}**\n\n")
        lines.append("| 口径 | precursor | T_C | beta | t_hold | Residual |\n|---|---|---|---|---|---|\n")
        for mode in ("c0", "c1", "c2"):
            e = summary[mode][key]
            lines.append(f"| {mode.upper()} | {e['precursor']:.1f}% | {e['T_C']:.1f}% | "
                         f"{e['beta']:.1f}% | {e['t_hold']:.1f}% | {e['Residual']:.1f}% |\n")
        lines.append("\n")

    lines.append("### 2.3 判定结果\n\n")
    for mode, v in verdicts.items():
        lines.append(f"- **{mode.upper()} vs C0**:方向一致(a)={v['sign_a_ok']},"
                     f"方向一致(c)={v['sign_c_ok']},显著性判定一致(a)={v['sig_a_ok']},"
                     f"显著性判定一致(c)={v['sig_c_ok']} → "
                     f"{'**通过**' if v['all_ok'] else '**未通过**'}\n")
    lines.append("\n")
    if all_variants_robust:
        lines.append("**结论:三种口径下 `Spearman(lattice_a/lattice_c, Θ)` 的方向与显著性判定"
                     "全部保持一致,与 C0(现行生产实现)同量级。生产实现维持 C0"
                     "(`xrd_processor.lattice_params()` 不改动),但需在 SI 补充本节"
                     "对照表,并在 §3.3 的 lattice_c/lattice_a 结论后加一句口径稳健性"
                     "说明。**\n\n")
        lines.append("> ⚠️ 但 §1 的逐 hkl 残差诊断已经确认:`z` 主要由 (003) 单峰的系统性"
                     "偏差驱动,不是一个干净的物理“高度位移”或“宽化偏置”参数。"
                     "本节的稳健性只说明**下游结论(lattice_c/lattice_a 与 Θ 的相关性)"
                     "对如何处理这个自由参数不敏感**,不代表 `z` 本身的物理含义已经"
                     "被澄清——这是两件不同的事,SI 措辞需要把这个区别说清楚,"
                     "不能简化成“z 就是仪器零点,没有问题”。\n\n")
    else:
        lines.append("**结论:至少一个变体在 lattice_a/lattice_c 与 Θ 的相关性上出现方向反转"
                     "或显著性判定改变(见 §2.3 逐项判定)。按照本分析的既定规则,这种情况下"
                     "本报告不擅自替换生产实现或挑选一个变体作为新结论,需项目负责人决定"
                     "下一步(是否需要重新表述 §3.3 的 lattice_c/lattice_a 核心论断)。**\n\n")

    with open(OUT_REPORT, "a", encoding="utf-8") as f:
        f.write("".join(lines))


if __name__ == "__main__":
    main()
