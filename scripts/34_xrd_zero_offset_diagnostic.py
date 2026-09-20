# -*- coding: utf-8 -*-
"""34 XRD 二次零点(lattice_params 的自由参数 z)诊断
====================================================================
背景:现流程对 2θ 做两次位置修正——(1) `process_all` 里
`estimate_zero_offset(tt_raw, ii)` 全谱网格搜索,得 `two_theta_offset_applied`,
平移整条谱;(2) `lattice_params()` 在最小二乘里额外拟合一个自由零点 `z`
(bounds ±0.2°,写入 `zero_shift` 列)。方法学复查指出:一个纯仪器零点不该
与 Θ/T_C 系统性相关——若相关,说明 `z` 吸收了某种未建模的物理/方法学效应,
而不是单纯的仪器噪声。

既有参考在 51 样纳米层口径上给出的相关(**本脚本独立复算,不采信**):
  Spearman(z,Θ)=-0.573, Spearman(z,T_C)=-0.574, Spearman(z,D_XRD)=-0.752,
  Spearman(z,lattice_a)=+0.306, Spearman(z,lattice_c)=+0.273,
  z 中位数=-0.0764°,标准差=0.0098°。

本脚本:
  1. 独立复算上述相关系数,并新增 Spearman(z, two_theta_offset_applied)
     ——若强相关,是"两步修正互相补偿"的直接证据。
  2. z 按前驱体分组的分布。
  3. 逐 hkl 2θ 残差角度依赖分析:对 51 样重新拟合全部 10 个 hkl 峰
     (复用 `xp.fit_peak`,只读,不改生产代码),用 `lattice_params()` 给出
     的最终 (a,c,z) 正算每个 hkl 的理论 2θ,残差 = 2θ_obs − 2θ_calc。
     按 2θ/cosθ 分箱汇总,检验:
       - 若残差随 cosθ 呈线性/系统性依赖 → 支持假说A(真实样品高度位移,
         Δ2θ=−(2s/R)·cosθ 形式)
       - 若残差与该样本该峰的 FWHM/D_XRD 相关,而与 2θ 本身无系统性依赖
         → 支持假说B(宽化依赖的峰位偏置,与 Kα1/Kα2 未分辨程度有关)
     两个假说不互斥,如实报告观察到的模式,不预设结论。

产出:
  data/interim/xrd_zero_offset_diagnostic.csv        51 样本汇总(z/Θ/T_C/
    D_XRD/lattice_a/lattice_c/two_theta_offset_applied/precursor)
  data/interim/xrd_hkl_residuals.csv                  逐样本×逐hkl 2θ残差
  reports/xrd_peak_position_reconstruction.md         §1(本脚本)+ 后续
    校正变体对比、分辨率阈值敏感性等分析追加
"""
import _bootstrap  # noqa

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from nfm.config import load_config
from nfm.data_processing import xrd_processor as xp
from nfm.nano_layer import nano_layer_frame

MASTER_TABLE = "data/processed/master_table.csv"
RAW_XRD_DIR = Path("data/raw/xrd")
OUT_SUMMARY_CSV = Path("data/interim/xrd_zero_offset_diagnostic.csv")
OUT_RESID_CSV = Path("data/interim/xrd_hkl_residuals.csv")
OUT_REPORT = Path("reports/xrd_peak_position_reconstruction.md")

HKL_TABLE = {"003": (0, 0, 3), "006": (0, 0, 6), "101": (1, 0, 1),
            "012": (0, 1, 2), "104": (1, 0, 4), "015": (0, 1, 5),
            "107": (1, 0, 7), "018": (0, 1, 8), "110": (1, 1, 0),
            "113": (1, 1, 3)}

COWORK_CLAIMS = dict(
    rho_z_theta=-0.573, rho_z_TC=-0.574, rho_z_dxrd=-0.752,
    rho_z_lattice_a=0.306, rho_z_lattice_c=0.273,
    z_median=-0.0764, z_std=0.0098,
)


def _two_theta_calc(a, c, z, lam_A, h, k, l):
    inv_d2 = 4.0 / 3.0 * (h * h + h * k + k * k) / a ** 2 + l * l / c ** 2
    d = 1.0 / np.sqrt(inv_d2)
    s = np.clip(lam_A / (2.0 * d), -1, 1)
    th = np.degrees(np.arcsin(s))
    return 2.0 * th + z


def main():
    cfg = load_config()
    xc = cfg.raw["instruments"]["xrd"]
    lam1 = xc["wavelength_A"]
    lam2 = xc["ka2_wavelength_A"]
    ratio = xc["ka2_ratio"]
    caglioti_inst1 = tuple(xc["caglioti_UVW"])

    df = pd.read_csv(MASTER_TABLE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nano = nano_layer_frame(df, require_reliable=True)
    print(f"[nano_layer_frame] n={len(nano)} (xrd_instrument==1 & D_XRD_reliable==True)")

    cols = ["sample_id", "condition_id", "precursor", "Theta", "T_C",
           "D_XRD", "lattice_a", "lattice_c", "zero_shift",
           "two_theta_offset_applied"]
    summary = nano[cols].dropna(subset=["zero_shift"]).reset_index(drop=True)
    n = len(summary)
    print(f"[有 zero_shift 的样本] n={n}")
    OUT_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_SUMMARY_CSV, index=False)

    # ================= 1. 独立复算相关系数 =================
    def _sp(x, y):
        r, p = spearmanr(summary[x], summary[y])
        return float(r), float(p)

    rho_z_theta, p_z_theta = _sp("zero_shift", "Theta")
    rho_z_TC, p_z_TC = _sp("zero_shift", "T_C")
    rho_z_dxrd, p_z_dxrd = _sp("zero_shift", "D_XRD")
    rho_z_a, p_z_a = _sp("zero_shift", "lattice_a")
    rho_z_c, p_z_c = _sp("zero_shift", "lattice_c")
    rho_z_offset, p_z_offset = _sp("zero_shift", "two_theta_offset_applied")

    z_median = float(summary["zero_shift"].median())
    z_std = float(summary["zero_shift"].std())

    print("\n=== 1. 独立复算相关系数(51样纳米层口径)===")
    for label, (r, p) in dict(
        z_theta=(rho_z_theta, p_z_theta), z_TC=(rho_z_TC, p_z_TC),
        z_dxrd=(rho_z_dxrd, p_z_dxrd), z_lattice_a=(rho_z_a, p_z_a),
        z_lattice_c=(rho_z_c, p_z_c),
        z_two_theta_offset_applied=(rho_z_offset, p_z_offset),
    ).items():
        print(f"  {label}: rho={r:+.4f}, p={p:.3e}")
    print(f"  z median={z_median:+.4f}, std={z_std:.4f}")

    # ================= 2. z 分前驱体分布 =================
    by_prec = summary.groupby("precursor")["zero_shift"].agg(["median", "std", "count"])
    print("\n=== 2. z 分前驱体分布 ===")
    print(by_prec.to_string())

    # ================= 3. 逐 hkl 2θ 残差角度依赖 =================
    resid_rows = []
    missing = []
    for _, row in summary.iterrows():
        sid = row["sample_id"]
        f = RAW_XRD_DIR / f"{sid}.txt"
        if not f.exists():
            missing.append(sid)
            continue
        inst = xp.detect_instrument(f)
        if inst != 1:
            raise RuntimeError(f"{sid}: instrument={inst}, 预期1(nano_layer_frame 应只含仪器1)")

        tt_raw, ii = xp.read_xy(f)
        zo = xp.estimate_zero_offset(tt_raw, ii)
        tt = tt_raw - zo["delta"]

        a, c, z = row["lattice_a"], row["lattice_c"], row["zero_shift"]
        for hkl, (h, k, l) in HKL_TABLE.items():
            g = xp.HKL_2THETA_GUESS[hkl]
            pk = xp.fit_peak(tt, ii, g, window=1.0, lam1=lam1, lam2=lam2, ratio=ratio)
            if pk is None or pk.get("degenerate"):
                continue
            tt_obs = pk["two_theta"]
            tt_calc = _two_theta_calc(a, c, z, lam1, h, k, l)
            resid = tt_obs - tt_calc
            th_rad = np.radians(tt_obs / 2.0)
            resid_rows.append(dict(
                sample_id=sid, hkl=hkl, precursor=row["precursor"],
                two_theta_obs=tt_obs, two_theta_calc=tt_calc,
                residual_deg=resid, fwhm=pk["fwhm"], eta=pk["eta"],
                cos_theta=float(np.cos(th_rad)), sin_theta=float(np.sin(th_rad)),
                D_XRD=row["D_XRD"],
            ))
    if missing:
        warnings.warn(f"缺少原始文件的样本(跳过): {missing}")

    resid_df = pd.DataFrame(resid_rows)
    OUT_RESID_CSV.parent.mkdir(parents=True, exist_ok=True)
    resid_df.to_csv(OUT_RESID_CSV, index=False)
    print(f"\n[done] {OUT_RESID_CSV} ({len(resid_df)} 行,{resid_df['sample_id'].nunique()} 样本)")

    # 假说A检验:残差 vs cos(theta)(全体 + 仅 00l 系 vs 仅非00l 系)
    resid_df["is_00l"] = resid_df["hkl"].isin(["003", "006"])
    rho_resid_costh_all, p_resid_costh_all = spearmanr(resid_df["cos_theta"], resid_df["residual_deg"])
    rho_resid_costh_00l, p_resid_costh_00l = spearmanr(
        resid_df.loc[resid_df["is_00l"], "cos_theta"], resid_df.loc[resid_df["is_00l"], "residual_deg"])
    rho_resid_costh_other, p_resid_costh_other = spearmanr(
        resid_df.loc[~resid_df["is_00l"], "cos_theta"], resid_df.loc[~resid_df["is_00l"], "residual_deg"])

    # 假说B检验:残差(绝对值,方向无关的"偏置幅度") vs FWHM/D_XRD
    rho_resid_fwhm, p_resid_fwhm = spearmanr(resid_df["fwhm"], resid_df["residual_deg"].abs())
    rho_resid_dxrd, p_resid_dxrd = spearmanr(resid_df["D_XRD"], resid_df["residual_deg"].abs())

    print("\n=== 3. 残差角度/宽化依赖检验 ===")
    print(f"  Spearman(residual, cos_theta) 全体: rho={rho_resid_costh_all:+.4f}, p={p_resid_costh_all:.3e} (n={len(resid_df)})")
    print(f"  Spearman(residual, cos_theta) 仅00l系(003/006): rho={rho_resid_costh_00l:+.4f}, p={p_resid_costh_00l:.3e}")
    print(f"  Spearman(residual, cos_theta) 仅非00l系: rho={rho_resid_costh_other:+.4f}, p={p_resid_costh_other:.3e}")
    print(f"  Spearman(|residual|, fwhm): rho={rho_resid_fwhm:+.4f}, p={p_resid_fwhm:.3e}")
    print(f"  Spearman(|residual|, D_XRD): rho={rho_resid_dxrd:+.4f}, p={p_resid_dxrd:.3e}")

    by_hkl_resid = resid_df.groupby("hkl")["residual_deg"].agg(["median", "std", "count"])
    print("\n=== 逐hkl残差(度) ===")
    print(by_hkl_resid.to_string())

    write_report(summary, n,
                rho_z_theta, p_z_theta, rho_z_TC, p_z_TC, rho_z_dxrd, p_z_dxrd,
                rho_z_a, p_z_a, rho_z_c, p_z_c, rho_z_offset, p_z_offset,
                z_median, z_std, by_prec,
                resid_df, rho_resid_costh_all, p_resid_costh_all,
                rho_resid_costh_00l, p_resid_costh_00l,
                rho_resid_costh_other, p_resid_costh_other,
                rho_resid_fwhm, p_resid_fwhm, rho_resid_dxrd, p_resid_dxrd,
                by_hkl_resid, missing)
    print(f"\n[done] {OUT_REPORT}")


def write_report(summary, n, rho_z_theta, p_z_theta, rho_z_TC, p_z_TC,
                 rho_z_dxrd, p_z_dxrd, rho_z_a, p_z_a, rho_z_c, p_z_c,
                 rho_z_offset, p_z_offset, z_median, z_std, by_prec,
                 resid_df, rho_resid_costh_all, p_resid_costh_all,
                 rho_resid_costh_00l, p_resid_costh_00l,
                 rho_resid_costh_other, p_resid_costh_other,
                 rho_resid_fwhm, p_resid_fwhm, rho_resid_dxrd, p_resid_dxrd,
                 by_hkl_resid, missing):
    lines = []
    lines.append("# XRD 峰位二次零点(z)重构诊断与基线定档\n\n")
    lines.append("> 由 `scripts/34_xrd_zero_offset_diagnostic.py`(§1)自动生成,"
                 "后续校正变体对比、分辨率阈值敏感性等分析追加到本文件"
                 "(不新建文件)。\n\n")
    lines.append("## §1 二次零点 z 的相关性诊断\n\n")
    lines.append(f"数据口径:`nano_layer_frame(require_reliable=True)`,n={n}"
                 "(51样纳米层口径中 zero_shift 非空的样本)。\n\n")
    lines.append("### 1.1 独立复算 vs 既有参考(不采信,独立复算为准)\n\n")
    lines.append("| 相关对象 | 既有参考 ρ | 本轮独立复算 ρ | p | 判定 |\n")
    lines.append("|---|---|---|---|---|\n")
    rows = [
        ("z vs Θ", COWORK_CLAIMS["rho_z_theta"], rho_z_theta, p_z_theta),
        ("z vs T_C", COWORK_CLAIMS["rho_z_TC"], rho_z_TC, p_z_TC),
        ("z vs D_XRD", COWORK_CLAIMS["rho_z_dxrd"], rho_z_dxrd, p_z_dxrd),
        ("z vs lattice_a", COWORK_CLAIMS["rho_z_lattice_a"], rho_z_a, p_z_a),
        ("z vs lattice_c", COWORK_CLAIMS["rho_z_lattice_c"], rho_z_c, p_z_c),
    ]
    for label, claim, mine, p in rows:
        verdict = "一致" if abs(claim - mine) < 0.03 else "**不一致,以本轮为准**"
        lines.append(f"| {label} | {claim:+.3f} | {mine:+.4f} | {p:.3e} | {verdict} |\n")
    lines.append(f"| z vs two_theta_offset_applied(新增,既有参考未给) | — | {rho_z_offset:+.4f} | {p_z_offset:.3e} | "
                 f"{'两步修正强相关,存在互相补偿证据' if abs(rho_z_offset) > 0.3 and p_z_offset < 0.05 else '两步修正不强相关'} |\n\n")
    lines.append(f"z 中位数 = **{z_median:+.4f}°**(既有参考 {COWORK_CLAIMS['z_median']:+.4f}°),"
                 f"std = **{z_std:.4f}°**(既有参考 {COWORK_CLAIMS['z_std']:.4f}°)。\n\n")
    lines.append("### 1.2 z 分前驱体分布\n\n")
    lines.append("| precursor | median | std | n |\n|---|---|---|---|\n")
    for prec, r in by_prec.iterrows():
        lines.append(f"| {prec} | {r['median']:+.4f} | {r['std']:.4f} | {int(r['count'])} |\n")
    lines.append("\n")

    lines.append("### 1.3 逐 hkl 2θ 残差角度依赖(假说A vs 假说B)\n\n")
    lines.append(f"残差定义:`residual = 2θ_obs − 2θ_calc(a,c,z,hkl)`(用该样本"
                 "`lattice_params()` 已拟合出的最终 a/c/z 正算理论峰位),"
                 f"逐样本逐 hkl,共 {len(resid_df)} 个残差点"
                 f"({resid_df['sample_id'].nunique()} 样本)。\n\n")
    if missing:
        lines.append(f"> 警告:{len(missing)} 个样本缺少原始 `.txt` 文件,已跳过:{missing}\n\n")
    lines.append("**假说A(真实样品高度位移,Δ2θ ∝ cosθ)检验**:\n\n")
    lines.append(f"- Spearman(residual, cosθ) 全体:ρ={rho_resid_costh_all:+.4f}, p={p_resid_costh_all:.3e}\n")
    lines.append(f"- 仅 00l 系(003/006):ρ={rho_resid_costh_00l:+.4f}, p={p_resid_costh_00l:.3e}\n")
    lines.append(f"- 仅非 00l 系:ρ={rho_resid_costh_other:+.4f}, p={p_resid_costh_other:.3e}\n\n")
    hyp_a_supported = (p_resid_costh_all < 0.05) and (abs(rho_resid_costh_all) > 0.2)
    lines.append(f"{'支持假说A:残差与 cosθ 存在系统性关联' if hyp_a_supported else '不支持假说A:残差与 cosθ 无明显系统性关联'}。\n\n")
    lines.append("**假说B(宽化依赖的峰位偏置)检验**:\n\n")
    lines.append(f"- Spearman(|residual|, FWHM):ρ={rho_resid_fwhm:+.4f}, p={p_resid_fwhm:.3e}\n")
    lines.append(f"- Spearman(|residual|, D_XRD):ρ={rho_resid_dxrd:+.4f}, p={p_resid_dxrd:.3e}\n\n")
    hyp_b_supported = (p_resid_fwhm < 0.05) and (abs(rho_resid_fwhm) > 0.2)
    lines.append(f"{'支持假说B:残差幅度与峰宽/D_XRD 存在系统性关联' if hyp_b_supported else '不支持假说B:残差幅度与峰宽/D_XRD 无明显系统性关联'}。\n\n")
    lines.append("**逐 hkl 残差汇总(度)**:\n\n")
    lines.append("| hkl | median | std | n |\n|---|---|---|---|\n")
    for hkl, r in by_hkl_resid.iterrows():
        lines.append(f"| {hkl} | {r['median']:+.4f} | {r['std']:.4f} | {int(r['count'])} |\n")
    lines.append("\n")
    if hyp_a_supported and not hyp_b_supported:
        verdict = "证据更支持假说A(真实样品高度位移)。"
    elif hyp_b_supported and not hyp_a_supported:
        verdict = "证据更支持假说B(宽化依赖的峰位偏置)。"
    elif hyp_a_supported and hyp_b_supported:
        verdict = "两个假说的证据都出现,不互斥,可能是复合效应。"
    else:
        verdict = "两个假说的证据都不明显,残差角度/宽化依赖检验未能定位 z 系统性相关的具体机制。"
    lines.append(f"**§1 小结**:{verdict} 详见 §2 三种口径正式对比后的最终判定。\n\n")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
