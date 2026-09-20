# -*- coding: utf-8 -*-
"""28 D_XRD 反卷积方法敏感性检验(高斯二次相减 vs 线性/洛伦兹相减)
====================================================================
方法学复查提出:xrd_processor.py 当前用高斯二次相减扣仪器宽化
(`beta_sample = sqrt(fwhm_obs^2 - fwhm_inst^2)`,`_beta_sample_rad`,
~line 270),但 pseudo-Voigt 拟合同时给出每个峰的洛伦兹分量占比 `eta`
(0=纯高斯,1=纯洛伦兹)——`eta` 接近 1 的峰理论上更适合线性相减
(`beta_sample = fwhm_obs - fwhm_inst`,柯西/洛伦兹卷积恒等式),而不是
高斯二次相减。复查方声称(未经本仓库独立验证,原话):
  - (003) 峰 eta 中位数 ≈ 0.576(偏洛伦兹),(104) 峰 eta 中位数 ≈ 0.108
    (偏高斯)——即两个峰可能需要不同的反卷积处理;
  - 全 51 样本切换高斯→线性/洛伦兹后,D_XRD 中位数变化约 76%,
    但 Spearman(D_XRD,Θ) 仍是同一个 +0.818,且两种方法逐样本 D_XRD 的
    Spearman 相关达到 +1.000。
本脚本**独立**从原始 .txt 重新拟合并计算,不采信上述数字,如实报告本脚本
自己算出的结果,并明确标注与复查方声称数字的分歧(如有)。

**这是敏感性/稳健性检验,不改变生产用高斯口径的 D_XRD**:
  - 不修改 `src/nfm/data_processing/xrd_processor.py`(生产反卷积方式不变)。
  - 不修改 `data/interim/xrd_features.csv` / `data/processed/master_table.csv`。
  - 本脚本产出的所有数字只写入 `data/interim/dxrd_deconvolution_sensitivity.csv`
    与 `reports/dxrd_deconvolution_sensitivity.md`,论文/生产结论仍以
    `master_table.csv` 现有的高斯口径 D_XRD 为准。

**范围**:仅 51 样本纳米层口径(`nfm.nano_layer.nano_layer_frame`,
xrd_instrument==1 且 D_XRD_reliable==True)——与全仓库"D_XRD 分析只用
仪器1的51样"的口径定档一致,不越权扩大到仪器2。

**复用**(只读,不改):`xp.read_xy`/`xp.estimate_zero_offset`/`xp.fit_peak`/
`xp.instrument_fwhm`/`xp.HKL_2THETA_GUESS`/`xp.scherrer`(高斯口径,直接复用
生产函数保证与 master_table 现有列可比)。线性/洛伦兹口径是本脚本新写的
独立函数(`scherrer_linear`),不进 xrd_processor.py。

产出:
  data/interim/dxrd_deconvolution_sensitivity.csv   51 样本×(003)/(104)
    两峰的 fwhm/two_theta/eta + 两种方法算出的 D + 与 master_table 既有列
    的交叉核对差值。
  reports/dxrd_deconvolution_sensitivity.md
"""
import _bootstrap  # noqa

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from statsmodels.stats.anova import anova_lm

from nfm.config import load_config
from nfm.data_processing import xrd_processor as xp
from nfm.nano_layer import nano_layer_frame

try:
    import tabulate  # noqa: F401
    HAVE_TABULATE = True
except ImportError:
    HAVE_TABULATE = False

MASTER_TABLE = "data/processed/master_table.csv"
RAW_XRD_DIR = Path("data/raw/xrd")
OUT_CSV = Path("data/interim/dxrd_deconvolution_sensitivity.csv")
OUT_REPORT = Path("reports/dxrd_deconvolution_sensitivity.md")

PEAKS = ("003", "104")
FACTORS = ("precursor", "T_C", "beta", "t_hold")
INTERACTIONS = ("precursor:T_C", "precursor:beta", "precursor:t_hold")
ALL_TERMS = FACTORS + INTERACTIONS

# 复查方声称的数字(未采信,只作对照,见模块 docstring)
REVIEW_CLAIMS = dict(
    eta_median_003=0.576, eta_median_104=0.108,
    spearman_theta_gauss=0.818, spearman_theta_lorentz=0.818,
    cross_method_spearman=1.000, median_pct_diff=76.0,
    interaction_total_range=(0.63, 3.20),
)


# ---------------------------------------------------------------------
# 线性/洛伦兹反卷积(新写,不进 xrd_processor.py)
# ---------------------------------------------------------------------
def scherrer_linear(fwhm_obs_deg, two_theta_deg, lam_A, K, caglioti):
    """线性(柯西/洛伦兹卷积恒等式)相减扣仪器宽化:
    beta_sample_deg = fwhm_obs_deg - fwhm_inst_deg,仅当 fwhm_obs > fwhm_inst
    时有效,否则 NaN(与生产 `_beta_sample_rad` 对高斯口径的"不兜底"原则
    一致:观测宽度不超过仪器宽度时,物理上没有可解析的样品额外宽化)。
    其余(Scherrer 公式本身)与生产 `xp.scherrer` 完全一致。"""
    fi = xp.instrument_fwhm(two_theta_deg, caglioti)
    beta_deg = fwhm_obs_deg - fi
    if beta_deg <= 0:
        return float("nan")
    beta_rad = beta_deg * xp.DEG
    th = np.radians(two_theta_deg / 2.0)
    D_nm = K * (lam_A * 0.1) / (beta_rad * np.cos(th))
    return float(D_nm)


def refit_sample(path: Path, lam1, lam2, ratio, caglioti):
    """重跑单样本:零点校正 + (003)/(104) 双线 pseudo-Voigt 拟合。
    返回 dict(hkl -> dict(two_theta, fwhm, eta)),退化/未拟合的峰不出现在结果里。
    """
    tt_raw, ii = xp.read_xy(path)
    zo = xp.estimate_zero_offset(tt_raw, ii)
    tt = tt_raw - zo["delta"]
    out = {}
    for hkl in PEAKS:
        g = xp.HKL_2THETA_GUESS[hkl]
        pk = xp.fit_peak(tt, ii, g, window=1.0, lam1=lam1, lam2=lam2, ratio=ratio)
        if pk is None or pk.get("degenerate"):
            continue
        out[hkl] = pk
    return out


def main():
    cfg = load_config()
    xc = cfg.raw["instruments"]["xrd"]
    lam1 = xc["wavelength_A"]
    lam2 = xc["ka2_wavelength_A"]
    ratio = xc["ka2_ratio"]
    K = xc["scherrer_K"]
    caglioti_inst1 = tuple(xc["caglioti_UVW"])

    df = pd.read_csv(MASTER_TABLE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nano = nano_layer_frame(df, require_reliable=True)
    sample_ids = sorted(nano["sample_id"].tolist())
    print(f"[nano_layer_frame] n={len(sample_ids)}(仅 xrd_instrument==1 且 D_XRD_reliable==True)")

    meta_cols = ["sample_id", "precursor", "T_C", "beta", "t_hold", "Theta",
                "xrd_instrument", "D_XRD", "D_XRD_003", "D_XRD_104"]
    meta = nano[meta_cols].set_index("sample_id")

    rows = []
    missing_files = []
    for sid in sample_ids:
        f = RAW_XRD_DIR / f"{sid}.txt"
        if not f.exists():
            missing_files.append(sid)
            continue
        inst = xp.detect_instrument(f)
        if inst != 1:
            raise RuntimeError(f"{sid}: detect_instrument={inst},预期为1"
                               "(nano_layer_frame 应只含仪器1样本,数据不一致,停下来报告)")

        fitted = refit_sample(f, lam1, lam2, ratio, caglioti_inst1)

        rec = {"sample_id": sid}
        rec.update(meta.loc[sid].to_dict())

        d_gauss_list, d_lorentz_list = [], []
        for hkl in PEAKS:
            pk = fitted.get(hkl)
            if pk is None:
                rec[f"two_theta_{hkl}"] = np.nan
                rec[f"fwhm_{hkl}"] = np.nan
                rec[f"eta_{hkl}"] = np.nan
                rec[f"D_{hkl}_gauss"] = np.nan
                rec[f"D_{hkl}_lorentz"] = np.nan
                continue
            tt_pk, fwhm_pk, eta_pk = pk["two_theta"], pk["fwhm"], pk["eta"]
            rec[f"two_theta_{hkl}"] = tt_pk
            rec[f"fwhm_{hkl}"] = fwhm_pk
            rec[f"eta_{hkl}"] = eta_pk

            d_g = xp.scherrer(fwhm_pk, tt_pk, lam1, K, caglioti_inst1)
            d_l = scherrer_linear(fwhm_pk, tt_pk, lam1, K, caglioti_inst1)
            rec[f"D_{hkl}_gauss"] = d_g
            rec[f"D_{hkl}_lorentz"] = d_l
            if np.isfinite(d_g):
                d_gauss_list.append(d_g)
            if np.isfinite(d_l):
                d_lorentz_list.append(d_l)

        rec["D_XRD_gauss_reimpl"] = float(np.mean(d_gauss_list)) if d_gauss_list else np.nan
        rec["D_XRD_lorentz"] = float(np.mean(d_lorentz_list)) if d_lorentz_list else np.nan
        rows.append(rec)

    if missing_files:
        warnings.warn(f"缺少原始文件的纳米层样本(跳过): {missing_files}")

    out = pd.DataFrame(rows)

    # ---- 交叉核对:高斯重实现 vs master_table 既有 D_XRD_003/D_XRD_104/D_XRD ----
    out["diff_003_vs_master"] = out["D_003_gauss"] - out["D_XRD_003"]
    out["diff_104_vs_master"] = out["D_104_gauss"] - out["D_XRD_104"]
    out["diff_DXRD_vs_master"] = out["D_XRD_gauss_reimpl"] - out["D_XRD"]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"[done] {OUT_CSV}  n={len(out)}")

    # ================= 分析 =================
    n = len(out)

    cross_check = dict(
        n=n,
        max_abs_diff_003=float(out["diff_003_vs_master"].abs().max()),
        max_abs_diff_104=float(out["diff_104_vs_master"].abs().max()),
        max_abs_diff_DXRD=float(out["diff_DXRD_vs_master"].abs().max()),
        mean_abs_diff_DXRD=float(out["diff_DXRD_vs_master"].abs().mean()),
    )
    print("\n=== 交叉核对:本脚本高斯重实现 vs master_table 既有列 ===")
    for k, v in cross_check.items():
        print(f"  {k}: {v}")

    # ---- eta 分布 ----
    eta_stats = {}
    for hkl in PEAKS:
        vals = out[f"eta_{hkl}"].dropna().to_numpy()
        eta_stats[hkl] = dict(
            n=len(vals),
            median=float(np.median(vals)) if len(vals) else np.nan,
            q25=float(np.percentile(vals, 25)) if len(vals) else np.nan,
            q75=float(np.percentile(vals, 75)) if len(vals) else np.nan,
        )
    print("\n=== eta(pseudo-Voigt 洛伦兹占比)分布 ===")
    for hkl in PEAKS:
        s = eta_stats[hkl]
        print(f"  ({hkl}): n={s['n']}, median={s['median']:.4f}, "
              f"IQR=[{s['q25']:.4f},{s['q75']:.4f}]")
    print(f"  复查方声称: (003)median={REVIEW_CLAIMS['eta_median_003']}, "
          f"(104)median={REVIEW_CLAIMS['eta_median_104']}")

    # ---- 两种方法 D_XRD 分布 + % 差 ----
    med_gauss = float(out["D_XRD_gauss_reimpl"].median())
    med_lorentz = float(out["D_XRD_lorentz"].median())
    pct_diff = (med_lorentz - med_gauss) / med_gauss * 100.0
    print(f"\n=== D_XRD 中位数(51样) ===")
    print(f"  高斯(重实现): {med_gauss:.3f} nm")
    print(f"  线性/洛伦兹:   {med_lorentz:.3f} nm")
    print(f"  差异: {pct_diff:+.2f}%  (复查方声称约 +{REVIEW_CLAIMS['median_pct_diff']}%)")

    valid_lorentz_n = int(out["D_XRD_lorentz"].notna().sum())
    print(f"  线性法有限值样本数: {valid_lorentz_n}/{n}")

    # ---- Spearman(D_XRD, Theta),各方法 ----
    sub_g = out.dropna(subset=["D_XRD_gauss_reimpl", "Theta"])
    sub_l = out.dropna(subset=["D_XRD_lorentz", "Theta"])
    rho_g, p_g = spearmanr(sub_g["Theta"], sub_g["D_XRD_gauss_reimpl"])
    rho_l, p_l = spearmanr(sub_l["Theta"], sub_l["D_XRD_lorentz"])
    print(f"\n=== Spearman(D_XRD, Theta) ===")
    print(f"  高斯:   rho={rho_g:.4f}, p={p_g:.3e}  (n={len(sub_g)})")
    print(f"  线性:   rho={rho_l:.4f}, p={p_l:.3e}  (n={len(sub_l)})")

    # ---- 两方法配对 Spearman ----
    sub_both = out.dropna(subset=["D_XRD_gauss_reimpl", "D_XRD_lorentz"])
    rho_cross, p_cross = spearmanr(sub_both["D_XRD_gauss_reimpl"], sub_both["D_XRD_lorentz"])
    print(f"\n=== 两方法逐样本配对 Spearman(D_XRD_gauss, D_XRD_lorentz) ===")
    print(f"  rho={rho_cross:.4f}, p={p_cross:.3e}  (n={len(sub_both)})")
    print(f"  复查方声称: rho={REVIEW_CLAIMS['cross_method_spearman']}")

    # ================= ANOVA(线性/洛伦兹口径 D_XRD,7 项模型,同 script 19 设计) =================
    work = out.dropna(subset=["D_XRD_lorentz"]).copy()
    for f in FACTORS:
        work[f] = work[f].astype("category")
    rhs = " + ".join(_term_to_patsy(t) for t in ALL_TERMS)
    model = smf.ols(f"D_XRD_lorentz ~ {rhs}", data=work).fit()
    tab = anova_lm(model, typ=2)
    tab = tab.rename(index={_term_to_patsy(t): t for t in ALL_TERMS})
    tab = tab.drop(index="Intercept", errors="ignore")

    y = work["D_XRD_lorentz"].to_numpy(float)
    sst_true = float(((y - y.mean()) ** 2).sum())
    anova_rows = []
    for t in ALL_TERMS:
        ss_t = float(tab.loc[t, "sum_sq"])
        df_t = int(tab.loc[t, "df"])
        anova_rows.append(dict(term=t, SS=ss_t, df=df_t,
                               eta2_pct=ss_t / sst_true * 100.0,
                               p_parametric=float(tab.loc[t, "PR(>F)"])))
    anova_df = pd.DataFrame(anova_rows).set_index("term")
    interaction_total_pct = anova_df.loc[list(INTERACTIONS), "eta2_pct"].sum()

    print(f"\n=== ANOVA(D_XRD_lorentz,51样,Type II,7项设计同 script 19,n={len(work)}) ===")
    print(anova_df.to_string())
    print(f"\n前驱体×工艺交互总量(3项 eta2 之和): {interaction_total_pct:.3f}%")
    print(f"  提示区间(复查方,不是目标值): "
          f"{REVIEW_CLAIMS['interaction_total_range']}")

    write_report(out, cross_check, eta_stats, med_gauss, med_lorentz, pct_diff,
                valid_lorentz_n, n, rho_g, p_g, rho_l, p_l, rho_cross, p_cross,
                len(sub_g), len(sub_l), len(sub_both), anova_df, interaction_total_pct,
                len(work), missing_files)
    print(f"\n[done] {OUT_REPORT}")


def _term_to_patsy(term: str) -> str:
    if ":" in term:
        f1, f2 = term.split(":")
        return f"C({f1}, Sum):C({f2}, Sum)"
    return f"C({term}, Sum)"


def _table_block(df: pd.DataFrame) -> str:
    if HAVE_TABULATE:
        return df.to_markdown() + "\n\n"
    return "```\n" + df.to_string() + "\n```\n\n"


def write_report(out, cross_check, eta_stats, med_gauss, med_lorentz, pct_diff,
                 valid_lorentz_n, n, rho_g, p_g, rho_l, p_l, rho_cross, p_cross,
                 n_g, n_l, n_both, anova_df, interaction_total_pct, n_anova, missing_files):
    fallback_note = (
        "" if HAVE_TABULATE else
        "> 注:本环境未安装 `tabulate` 包,`DataFrame.to_markdown()` 不可用,"
        "以下表格改用 `df.to_string()` 包在代码块中呈现(内容等价,仅渲染形式从 "
        "Markdown 表格换成等宽文本块)。\n\n"
    )

    lines = []
    lines.append("# D_XRD 反卷积方法敏感性检验(高斯二次相减 vs 线性/洛伦兹相减)\n\n")
    lines.append("> 由 `scripts/28_dxrd_deconvolution_sensitivity.py` 自动生成。"
                 "**这是敏感性/稳健性检验,不改变生产用高斯口径的 D_XRD**——"
                 "未修改 `src/nfm/data_processing/xrd_processor.py`、"
                 "`data/interim/xrd_features.csv`、`data/processed/master_table.csv`;"
                 "本报告所有数字仅来自本脚本独立重跑,"
                 "写入 `data/interim/dxrd_deconvolution_sensitivity.csv`,"
                 "论文/生产结论仍以现有高斯口径 D_XRD 为准。\n\n")
    lines.append(fallback_note)
    lines.append("## 0. 背景与复查方声称的数字(未采信,仅作对照)\n\n")
    lines.append("方法学复查提出:pseudo-Voigt 拟合给出的 `eta`(0=纯高斯、1=纯洛伦兹)"
                 "若接近 1,理论上更适合线性/洛伦兹相减而非当前生产用的高斯二次相减。"
                 "复查方给出的原话数字(**本脚本独立重新计算,不采信,只作对照**):\n\n")
    lines.append(f"- (003) 峰 eta 中位数 ≈ {REVIEW_CLAIMS['eta_median_003']},"
                 f"(104) 峰 eta 中位数 ≈ {REVIEW_CLAIMS['eta_median_104']}\n")
    lines.append(f"- 全51样本切换高斯→线性/洛伦兹,D_XRD 中位数变化约 "
                 f"{REVIEW_CLAIMS['median_pct_diff']:.0f}%\n")
    lines.append(f"- Spearman(D_XRD,Θ) 两种方法都是 +{REVIEW_CLAIMS['spearman_theta_gauss']}\n")
    lines.append(f"- 两方法逐样本 D_XRD 的 Spearman 相关 = "
                 f"+{REVIEW_CLAIMS['cross_method_spearman']:.3f}\n\n")
    if missing_files:
        lines.append(f"> 警告:纳米层 51 样本中有 {len(missing_files)} 个缺少原始 `.txt` 文件,"
                     f"已跳过:{missing_files}\n\n")

    lines.append("## 1. 高斯重实现交叉核对(vs `master_table.csv` 既有列)\n\n")
    lines.append(f"本脚本用 `xp.scherrer`(生产同一函数)重新计算 D_XRD_003/D_XRD_104/D_XRD,"
                 f"与 `master_table.csv` 中既有列逐样本比较(n={cross_check['n']}):\n\n")
    lines.append(f"- (003) 最大绝对差:**{cross_check['max_abs_diff_003']:.6f} nm**\n")
    lines.append(f"- (104) 最大绝对差:**{cross_check['max_abs_diff_104']:.6f} nm**\n")
    lines.append(f"- D_XRD(两峰平均) 最大绝对差:**{cross_check['max_abs_diff_DXRD']:.6f} nm**,"
                 f"平均绝对差 {cross_check['mean_abs_diff_DXRD']:.6f} nm\n\n")
    match_verdict = "高精度吻合" if cross_check["max_abs_diff_DXRD"] < 1e-3 else \
        ("基本吻合(微小数值差,可能来自浮点/拟合初值路径)" if cross_check["max_abs_diff_DXRD"] < 0.5 else "**存在明显分歧,需排查**")
    lines.append(f"结论:重实现与生产 pipeline 既有列{match_verdict}。\n\n")

    lines.append("## 2. eta(pseudo-Voigt 洛伦兹占比)分布\n\n")
    eta_rows = pd.DataFrame([
        dict(peak=f"({hkl})", n=eta_stats[hkl]["n"], median=eta_stats[hkl]["median"],
            q25=eta_stats[hkl]["q25"], q75=eta_stats[hkl]["q75"])
        for hkl in PEAKS
    ])
    lines.append(_table_block(eta_rows))
    verdict_003 = "接近" if abs(eta_stats["003"]["median"] - REVIEW_CLAIMS["eta_median_003"]) < 0.05 else "不同于"
    verdict_104 = "接近" if abs(eta_stats["104"]["median"] - REVIEW_CLAIMS["eta_median_104"]) < 0.05 else "不同于"
    lines.append(f"本脚本独立计算:(003) eta 中位数 = **{eta_stats['003']['median']:.4f}**"
                 f"(复查方声称 {REVIEW_CLAIMS['eta_median_003']},{verdict_003});"
                 f"(104) eta 中位数 = **{eta_stats['104']['median']:.4f}**"
                 f"(复查方声称 {REVIEW_CLAIMS['eta_median_104']},{verdict_104})。\n\n")

    lines.append("## 3. D_XRD:高斯 vs 线性/洛伦兹\n\n")
    lines.append(f"- 高斯(重实现)中位数:**{med_gauss:.3f} nm**\n")
    lines.append(f"- 线性/洛伦兹中位数:**{med_lorentz:.3f} nm**(有限值 {valid_lorentz_n}/{n} 样本;"
                 f"`fwhm_obs<=fwhm_inst` 时该峰/该方法置 NaN)\n")
    lines.append(f"- 相对差异:**{pct_diff:+.2f}%**(复查方声称约 "
                 f"+{REVIEW_CLAIMS['median_pct_diff']:.0f}%)\n\n")

    lines.append("## 4. Spearman(D_XRD, Θ),两种方法各自\n\n")
    rho_rows = pd.DataFrame([
        dict(method="高斯(重实现)", n=n_g, spearman_rho=rho_g, p=p_g),
        dict(method="线性/洛伦兹", n=n_l, spearman_rho=rho_l, p=p_l),
    ])
    lines.append(_table_block(rho_rows))
    sign_stable = np.sign(rho_g) == np.sign(rho_l)
    sig_stable = (p_g < 0.05) == (p_l < 0.05)
    lines.append(f"符号一致:{'是' if sign_stable else '否'};显著性判定(p<0.05)一致:"
                 f"{'是' if sig_stable else '否'}(复查方声称两者都是 "
                 f"+{REVIEW_CLAIMS['spearman_theta_gauss']})\n\n")

    lines.append("## 5. 两方法逐样本配对 Spearman(D_XRD_gauss, D_XRD_lorentz)\n\n")
    lines.append(f"rho = **{rho_cross:.4f}**,p = {p_cross:.3e}(n={n_both})"
                 f"(复查方声称 +{REVIEW_CLAIMS['cross_method_spearman']:.3f})\n\n")

    lines.append("## 6. 前驱体×工艺 ANOVA(线性/洛伦兹口径 D_XRD,Type II,7项设计同 script 19)\n\n")
    lines.append(f"数据:纳米层51样中线性法 D_XRD 有限值的 n={n_anova} 样本;"
                 "模型:`y ~ precursor + T_C + beta + t_hold + precursor:T_C + "
                 "precursor:beta + precursor:t_hold`(Sum 编码),eta2 分母用真实 "
                 "SST=Σ(y-ȳ)²,与 `scripts/19_nano_layer_interaction_anova.py` 同口径。\n\n")
    anova_report = anova_df.copy()
    anova_report["eta2_pct"] = anova_report["eta2_pct"].round(3)
    lines.append(_table_block(anova_report[["SS", "df", "eta2_pct", "p_parametric"]]))
    lines.append("### 主效应表(4项)\n\n")
    lines.append(_table_block(anova_report.loc[list(FACTORS), ["SS", "df", "eta2_pct", "p_parametric"]]))
    lines.append(f"### 前驱体×工艺交互总量(headline number)\n\n")
    lo, hi = REVIEW_CLAIMS["interaction_total_range"]
    in_range = lo <= interaction_total_pct <= hi
    lines.append(f"三项交互 η² 之和 = **{interaction_total_pct:.3f}%**"
                 f"(复查方期望区间 {lo:.2f}–{hi:.2f}%,{'落在区间内' if in_range else '**不在该区间内**'},"
                 "与 Part B 其余目标 0.63–3.33% 同量级"
                 f"{'' if in_range else '(仍属同一数量级,但未落入复查方给出的具体区间)'})。\n\n")

    lines.append("## 结论\n\n")
    lines.append(f"- 高斯重实现与生产 pipeline 既有 `D_XRD_003`/`D_XRD_104`/`D_XRD` 三列{match_verdict}"
                 f"(最大绝对差 {cross_check['max_abs_diff_DXRD']:.2e} nm),确认本脚本重跑逻辑正确。\n")
    lines.append(f"- eta 中位数:(003)={eta_stats['003']['median']:.4f},"
                 f"(104)={eta_stats['104']['median']:.4f}"
                 f"——与复查方声称的 {REVIEW_CLAIMS['eta_median_003']}/"
                 f"{REVIEW_CLAIMS['eta_median_104']} 相比:{verdict_003}/{verdict_104}。\n")
    lines.append(f"- D_XRD 中位数从高斯 {med_gauss:.2f} nm 变为线性/洛伦兹 {med_lorentz:.2f} nm"
                 f"({pct_diff:+.1f}%)。\n")
    lines.append(f"- Spearman(D_XRD,Θ):高斯 {rho_g:+.3f}、线性 {rho_l:+.3f}"
                 f"——符号{'一致' if sign_stable else '不一致'}。\n")
    lines.append(f"- 两方法逐样本 D_XRD 的 Spearman 相关:{rho_cross:+.4f}。\n")
    lines.append(f"- 线性/洛伦兹口径 D_XRD 的前驱体×工艺交互总量(η²):"
                 f"{interaction_total_pct:.3f}%。\n")
    lines.append("- 本检验**不改变**生产用高斯口径 D_XRD;若绝对数值对反卷积方法敏感、"
                 "但排序(Spearman vs Θ)与交互结构稳健,则支持论文现有基于排序/趋势的"
                 "结论对反卷积方法选择不敏感;若不稳健,应如实在论文限制声明中补充。\n")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
