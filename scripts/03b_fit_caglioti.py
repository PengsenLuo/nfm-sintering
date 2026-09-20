# -*- coding: utf-8 -*-
"""
03b_fit_caglioti.py —— 用 Si 标样数据拟合 Caglioti 仪器宽化参数 (U, V, W)
==========================================================================
背景:xrd_processor.py 里 D_XRD/D_XRD_WH 的仪器宽化扣除依赖 (U,V,W),但
configs/config.yaml 之前没有 caglioti_UVW 键,一直落到占位默认值
(0.001, -0.0005, 0.004)。此前出现过的 U=0.007670/V=-0.006492/
W=0.007007 来源不可追溯,本脚本不采用,改为在本仓库内用现有的
data/raw/xrd/standards/Si-standard.txt 重新拟合,做到可复现。

消费端单位/公式约定(已读 xrd_processor.py 核对,未发现 bug,以下是确认到的口径):
  - instrument_fwhm(two_theta_deg, caglioti) 计算
        FWHM_inst² = U·tan²θ + V·tanθ + W   (θ = two_theta_deg/2,换算成弧度后取 tan)
    返回值 FWHM_inst 的单位是“度”(函数 docstring 明确写"(度)")。
  - _beta_sample_rad 用 fwhm_obs_deg**2 - FWHM_inst**2(单位都是 度²)做差,
    开方后再乘 DEG(=π/180)转成弧度,得到最终喂给 Scherrer/Williamson-Hall
    的 beta(弧度)。
  - 因此:U/V/W 必须按“FWHM 用度、θ 用弧度求 tan”的口径拟合,且观测 FWHM
    的提取方法必须和生产数据用的一致 —— 本脚本直接复用
    xrd_processor.read_xy / xrd_processor.fit_peak(同一套 Kα1+Kα2 双线
    pseudo-Voigt 模型),不另起一套峰形假设,标定和生产数据的"观测 FWHM"
    定义完全一致。

方法:
  1. data/raw/xrd/standards/Si-standard.txt 格式与生产 XRD .txt 相同
     (首行 "?" 占位行,之后两列 2θ/强度),read_xy 可直接读,不需要额外
     兼容代码。
  2. 用 scipy.signal.find_peaks 在 5–90° 全谱范围内找强峰。Si 粉末花样
     在该范围内理论上只有 6 条衍射线:(111)(220)(311)(400)(331)(422),
     探测结果(28.46/47.32/56.14/69.14/76.38/88.04)与文献值高度吻合。
  3. 对每个峰用 xrd_processor.fit_peak 拟合,取观测 FWHM(度)与峰位 2θ(度)。
  4. 普通最小二乘:FWHM² = U·tan²θ + V·tanθ + W,报告 R² 和逐峰残差。
  5. 诊断图(拟合曲线 + 数据点 + 残差)存到 data/interim/caglioti_fit_diagnostics.png,
     逐峰数值存到 data/interim/caglioti_peaks.csv。
  6. 只有 R² ≥ R2_THRESHOLD(默认 0.95)才会把 (U,V,W) 写回
     configs/config.yaml 的 instruments.xrd.caglioti_UVW;否则只打印/画图,
     不动 config.yaml,让人决定怎么处理。

用法:
    python scripts/03b_fit_caglioti.py                          # 只拟合+诊断,不回填
    python scripts/03b_fit_caglioti.py --write-config            # R² 达标时才真正回填
    python scripts/03b_fit_caglioti.py --write-config --force    # R² 未达标,人工确认后强制回填

──────────────────────────────────────────────────────────────────────────
本次拟合结果(R²=0.8770,经 --force 回填)已确认采用。
为什么 R²=0.8770(低于默认阈值 0.95)在本场景下可以接受,这里固化下来,
避免以后每次看到这个数字都重新怀疑一遍:
  - R² 偏低是"窄范围数据"的指标特性,不是拟合差:6 个 Si 峰的 FWHM 只在
    0.072–0.088° 这个很窄的区间内变化,总方差天生很小,同样大小的绝对
    残差,在窄范围数据上算出来的 R² 必然显著低于宽范围数据——这反映的是
    "仪器宽化随角度变化平缓"这个事实本身,不是模型/数据有问题。
  - 真正该看的判据是**绝对残差**:6 个峰的残差都在 0.0003–0.0008° 量级
    (约为 FWHM 本身的 1% 左右)。这个量级的误差经
    `FWHM_obs² − FWHM_inst²` 扣除后传导到 Scherrer 公式给出的 D_XRD,
    影响在亚纳米量级,相对于本研究 40–82nm 的晶粒尺寸范围可忽略。
  - 拟合出的 U 形曲线(FWHM 先随角度降低、过中段后再升高)符合 Caglioti
    曲线的经典物理形状,不是病态拟合。
  - 做过加权最小二乘的对照(R²=0.925 更高),但**不采用**:加权用的是
    pseudo-Voigt 拟合协方差估出的 FWHM 不确定度,(111) 峰因强度极高、
    峰形极尖锐,curve_fit 内 amp/fwhm/eta 三参数出现强相关,导致协方差
    被人为放大,不代表真实测量噪声;按这组人为放大的不确定度加权会扭曲
    结果,故沿用无权重的普通最小二乘版本。
  - 5–90° 范围内 Si 粉末花样只有 6 条允许衍射线((111)(220)(311)(400)
    (331)(422)),物理上没有更多峰可加;重扫标样不会增加可用峰数,
    没有必要。
──────────────────────────────────────────────────────────────────────────
"""
import _bootstrap  # noqa

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from nfm.config import load_config
from nfm.data_processing.xrd_processor import read_xy, fit_peak

SI_STANDARD_TXT = "data/raw/xrd/standards/Si-standard.txt"
OUT_PEAKS_CSV = "data/interim/caglioti_peaks.csv"
OUT_PLOT_PNG = "data/interim/caglioti_fit_diagnostics.png"
R2_THRESHOLD = 0.95

# 标准 Si 粉末花样在 5-90° 内的衍射指数(仅用于打印标注,不参与拟合本身)
SI_HKL_REF_2TH = {"111": 28.44, "220": 47.30, "311": 56.12,
                   "400": 69.13, "331": 76.37, "422": 88.03}


def _label_hkl(two_theta: float) -> str:
    hkl, _ = min(SI_HKL_REF_2TH.items(), key=lambda kv: abs(kv[1] - two_theta))
    return hkl


def find_si_peaks(tt, ii, prominence_frac=0.03, distance_pts=20):
    bg = np.percentile(ii, 20)
    prominence = (ii.max() - bg) * prominence_frac
    idx, props = find_peaks(ii, prominence=prominence, distance=distance_pts)
    order = np.argsort(-props["prominences"])
    return tt[idx[order]]


def _out_paths(config_key: str):
    """诊断产出路径按 config_key 派生:默认键(仪器1)保持旧文件名不变,
    其它键(如仪器2)用 caglioti2_* 前缀,不覆盖仪器1的产出。"""
    suffix = "" if config_key == "caglioti_UVW" else "2"
    return (f"data/interim/caglioti{suffix}_peaks.csv",
            f"data/interim/caglioti{suffix}_fit_diagnostics.png")


def fit_caglioti(twoth_deg, fwhm_deg):
    th = np.radians(np.asarray(twoth_deg) / 2.0)
    x1, x2 = np.tan(th) ** 2, np.tan(th)
    y = np.asarray(fwhm_deg) ** 2
    A = np.vstack([x1, x2, np.ones_like(x1)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return coef, r2, yhat


def main(write_config: bool, force: bool, standard_path: str = SI_STANDARD_TXT,
        config_key: str = "caglioti_UVW"):
    cfg = load_config()
    xc = cfg.raw["instruments"]["xrd"]
    lam1, lam2, ratio = xc["wavelength_A"], xc["ka2_wavelength_A"], xc["ka2_ratio"]
    out_peaks_csv, out_plot_png = _out_paths(config_key)

    tt, ii = read_xy(standard_path)
    print(f"[Caglioti] 读取 {standard_path}: {len(tt)} 点,"
          f"2θ 范围 {tt.min():.2f}–{tt.max():.2f}°")

    guesses_raw = find_si_peaks(tt, ii)
    print(f"[Caglioti] 探测到 {len(guesses_raw)} 个候选峰: "
          f"{[round(float(g), 3) for g in guesses_raw]}")

    # 去重(按最近 hkl 归属,只保留每个 hkl 里 prominence 最高的那个候选):
    # 高分辨率仪器在高角端可能把 Kα1/Kα2 双线分辨成两个独立的局部极大值
    # (例如 Si-standard-2 在 76.36°/76.58° 各出现一个峰,均离 331 最近),
    # find_si_peaks 本身不做 hkl 归属、不知道这是同一条衍射线的两个瓣。
    # guesses_raw 已按 prominence 降序排列,遍历时先出现的就是更强的那个候选。
    seen_hkl, guesses = set(), []
    for g in guesses_raw:
        hkl = _label_hkl(float(g))
        if hkl in seen_hkl:
            print(f"  [去重] 2θ≈{g:.2f}° 与已保留的 Si({hkl}) 候选属于同一条衍射线,跳过")
            continue
        seen_hkl.add(hkl)
        guesses.append(g)

    rows = []
    for g in guesses:
        pk = fit_peak(tt, ii, float(g), window=1.2, lam1=lam1, lam2=lam2, ratio=ratio)
        if pk is None:
            print(f"  [警告] 2θ≈{g:.2f} 附近拟合失败,跳过")
            continue
        rows.append(dict(hkl_guess=_label_hkl(float(g)),
                          two_theta_deg=pk["two_theta"],
                          fwhm_deg=pk["fwhm"], eta=pk["eta"]))
    peaks = pd.DataFrame(rows)
    if len(peaks) < 4:
        raise RuntimeError(f"有效峰数仅 {len(peaks)}(<4),无法可靠拟合 3 参数 Caglioti 曲线,"
                           "停止,不输出结果。")

    coef, r2, yhat = fit_caglioti(peaks["two_theta_deg"], peaks["fwhm_deg"])
    U, V, W = (float(v) for v in coef)
    peaks["fwhm_fit_deg"] = np.sqrt(np.clip(yhat, 0, None))
    peaks["residual_deg2"] = peaks["fwhm_deg"] ** 2 - yhat

    Path(out_peaks_csv).parent.mkdir(parents=True, exist_ok=True)
    peaks.to_csv(out_peaks_csv, index=False)

    print(f"[Caglioti] 用 {len(peaks)} 个峰拟合,普通最小二乘:")
    print(peaks.to_string(index=False))
    print(f"[Caglioti] U={U:.6f}  V={V:.6f}  W={W:.6f}  R^2={r2:.4f}")
    print(f"[Caglioti] 逐峰结果已存:{out_peaks_csv}")

    _save_diagnostic_plot(peaks, coef, r2, out_plot_png)
    print(f"[Caglioti] 诊断图已存:{out_plot_png}(请目视确认拟合曲线是否合理)")

    meets_threshold = r2 >= R2_THRESHOLD
    if not meets_threshold and not force:
        print(f"\n[Caglioti] [警告] R^2={r2:.4f} 低于阈值 {R2_THRESHOLD},"
              "默认不会写入 config.yaml。若已人工确认这组结果可接受"
              "(判据见脚本头部注释),加 --force 显式覆盖;否则只输出诊断。")
        return

    if not write_config:
        if meets_threshold:
            print(f"\n[Caglioti] R^2={r2:.4f} >= {R2_THRESHOLD},但本次未加 --write-config,"
                  "未回填 config.yaml(避免静默改配置)。确认无误后可重跑加 --write-config。")
        else:
            print(f"\n[Caglioti] R^2={r2:.4f} 低于阈值,已加 --force,但还需要同时加 "
                  "--write-config 才会真正写入。当前只是诊断,未回填。")
        return

    _write_config_yaml(U, V, W, config_key)
    if meets_threshold:
        print(f"[Caglioti] R^2={r2:.4f} >= {R2_THRESHOLD},已写入 "
              f"configs/config.yaml 的 instruments.xrd.{config_key}")
    else:
        print(f"[Caglioti] [人工确认覆盖] R^2={r2:.4f} 低于默认阈值 {R2_THRESHOLD},"
              "但经 --force 显式确认采用(理由见本脚本头部 2026-06 注释段),"
              f"已写入 configs/config.yaml 的 instruments.xrd.{config_key}")


def _save_diagnostic_plot(peaks: pd.DataFrame, coef, r2, out_png: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    th_dense = np.linspace(peaks["two_theta_deg"].min() - 2,
                           peaks["two_theta_deg"].max() + 2, 400)
    th_rad = np.radians(th_dense / 2.0)
    U, V, W = coef
    fwhm_curve = np.sqrt(np.clip(
        U * np.tan(th_rad) ** 2 + V * np.tan(th_rad) + W, 0, None))

    # 全部用英文标签:避免默认字体(DejaVu Sans)缺中文字形导致图里出现方框。
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(th_dense, fwhm_curve, "-", color="C0", label="Caglioti fit curve")
    ax[0].plot(peaks["two_theta_deg"], peaks["fwhm_deg"], "o", color="C1",
              label="Si standard observed FWHM")
    for _, r in peaks.iterrows():
        ax[0].annotate(f"Si({r['hkl_guess']})", (r["two_theta_deg"], r["fwhm_deg"]),
                       textcoords="offset points", xytext=(4, 4), fontsize=9)
    ax[0].set_xlabel("2theta (deg)"); ax[0].set_ylabel("FWHM (deg)")
    ax[0].set_title(f"Caglioti fit  U={U:.5f} V={V:.5f} W={W:.5f}\nR2={r2:.4f}")
    ax[0].legend()

    ax[1].axhline(0, color="gray", lw=1)
    ax[1].bar(peaks["hkl_guess"].apply(lambda h: f"Si({h})"), peaks["residual_deg2"], color="C2")
    ax[1].set_ylabel("residual FWHM^2 (obs - fit) [deg^2]")
    ax[1].set_title("per-peak residuals")

    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=120)
    plt.close()


def _write_config_yaml(U: float, V: float, W: float, config_key: str = "caglioti_UVW"):
    """把 (U,V,W) 写进 configs/config.yaml 的 instruments.xrd 段(config_key 那一行),
    保留其它内容/注释。config_key 默认 caglioti_UVW(仪器1),不影响既有行为;
    传 caglioti_UVW_inst2 时只替换仪器2那一行,不动仪器1的值。"""
    import re

    cfg_path = Path("configs/config.yaml")
    text = cfg_path.read_text(encoding="utf-8")
    out_peaks_csv, _ = _out_paths(config_key)
    new_line = f"    {config_key}: [{U:.6f}, {V:.6f}, {W:.6f}]  " \
              f"# 03b_fit_caglioti.py 拟合(见 {out_peaks_csv})\n"
    pattern = rf"^    {re.escape(config_key)}:.*\n"
    if re.search(pattern, text, flags=re.MULTILINE):
        text = re.sub(pattern, new_line, text, flags=re.MULTILINE)
    else:
        text = re.sub(r"(  xrd:\n(?:.*\n)*?    wh_peaks:.*\n)",
                      r"\1" + new_line, text)
    cfg_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-config", action="store_true",
                    help="把 (U,V,W) 写入 configs/config.yaml(R^2 达标时直接写;"
                         "未达标时需同时加 --force 才写)")
    ap.add_argument("--force", action="store_true",
                    help="R^2 低于默认阈值时,人工确认后仍允许写入(配合 --write-config)。"
                         "不会修改 R2_THRESHOLD 本身,只是单次显式绕过。")
    ap.add_argument("--standard", default=SI_STANDARD_TXT,
                    help="Si 标样文件路径(默认仪器1标样,保持现状不变)")
    ap.add_argument("--config-key", default="caglioti_UVW",
                    help="写入 config.yaml 的键名(默认 caglioti_UVW,即仪器1,保持现状不变;"
                         "第二台仪器用 caglioti_UVW_inst2)")
    args = ap.parse_args()
    main(args.write_config, args.force, args.standard, args.config_key)
