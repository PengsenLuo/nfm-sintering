# -*- coding: utf-8 -*-
"""
psd_processor.py —— 激光粒度处理器(模块二·处理器 1,v0.2)
==========================================================
输入 : data/raw/psd/<sample>.csv   (两列:粒径 µm vs 体积累积 %)
输出 : data/interim/psd_features.csv  每样本一行:
         D10 / D50 / D90 / Span + B_bimodal + 双峰拟合参数 + QC

v0.2 新增:**双对数正态混合拟合**,从完整分布提取双峰特征量 B_bimodal,
作为双峰保留指数 M_B 的数据来源(X.2.4)。B 综合"双峰分离度 × 小峰权重",
单峰样品(S/L)拟合后小峰权重趋零 → B≈0,符合物理直觉。

禁水:分散介质为无水乙醇(仅作元数据记录与校验)。
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import norm

SAMPLE_ID_RE = re.compile(r"^(C\d{2}-[SML])$")
DVALUES_PREFIX = "psd_dvalues_"


def _iter_sample_csv_files(raw_dir: Path) -> list[Path]:
    """样本分布 CSV 列表：显式排除仪器权威 D 值表(psd_dvalues_*.csv)，
    再按 sample_id 正则过滤。两者独立生效，不依赖正则恰好不匹配的副作用。"""
    out = []
    for f in sorted(raw_dir.glob("*.csv")):
        if f.stem.startswith(DVALUES_PREFIX):
            continue
        if not SAMPLE_ID_RE.match(f.stem):
            continue
        out.append(f)
    return out


# ---------------------------------------------------------------------
# 1. 读取累积分布 / 体积密度分布(共享底层 xy 读取)
# ---------------------------------------------------------------------
def _read_raw_xy(path: Path):
    df = pd.read_csv(path)
    df = df.iloc[:, :2]
    df.columns = ["d_um", "pct"]
    df = df.dropna().sort_values("d_um")
    d = df["d_um"].to_numpy(float)
    pct = df["pct"].to_numpy(float)
    if pct.max() <= 1.5:        # 若是 0–1 比例,转百分数
        pct = pct * 100.0
    return d, pct


def read_cumulative_csv(path: Path):
    d, pct = _read_raw_xy(path)
    # 仪器导出的实际是逐区间体积密度(先升后降的钟形),不是累积分布;
    # 用单调性判断:非单调不减则视为微分分布,累加成累积分布再用。
    if np.all(np.diff(pct) >= -1e-6):
        cum = pct
    else:
        cum = np.cumsum(pct)
    cum = np.clip(cum, 0.0, 100.0)
    return d, cum


def read_density_csv(path: Path):
    """逐区间体积密度(非累积),用于原始曲线模式计数。仪器导出通常已是密度
    (先升后降的钟形);若输入恰好是单调不减的累积曲线,用差分还原密度。"""
    d, pct = _read_raw_xy(path)
    if np.all(np.diff(pct) >= -1e-6):
        density = np.diff(pct, prepend=0.0)
    else:
        density = pct
    return d, density


def d_values(d, cum):
    """由累积曲线插值 D10/D50/D90,并算 Span。"""
    def dx(p):
        return float(np.interp(p, cum, d))
    D10, D50, D90 = dx(10), dx(50), dx(90)
    span = (D90 - D10) / D50 if D50 > 0 else np.nan
    return D10, D50, D90, span


def load_instrument_dvalues(path: Path) -> pd.DataFrame:
    """仪器报告 D 值权威表(sample_id 索引,列 D10/D50/D90/Span)。"""
    df = pd.read_csv(path)
    return df.set_index("sample_id")[["D10", "D50", "D90", "Span"]]


# ---------------------------------------------------------------------
# 2. 原始曲线模式计数(一级判据)+ 双对数正态混合拟合 → B_bimodal
# ---------------------------------------------------------------------
def count_modes(d, density, prominence: float = 0.5, height: float = 0.1):
    """在原始体积密度曲线(非累积)上做模式计数。

    prominence/height 单位与密度列一致(体积百分比%)。默认值来自对本项目
    60 个真实样本的敏感性扫描(prominence∈[0.2,2.0] 结果完全稳定,只在 0.1
    附近出现噪声级假阳性)。
    """
    peaks, _ = find_peaks(density, prominence=prominence, height=height)
    return len(peaks), [float(d[p]) for p in peaks]



def _bilognorm_cdf(d, w1, mu1, s1, mu2, s2):
    """两条对数正态的加权累积(w1 为小峰权重,mu/s 在 ln 空间)。"""
    x = np.log(d)
    c1 = norm.cdf(x, loc=mu1, scale=s1)
    c2 = norm.cdf(x, loc=mu2, scale=s2)
    return w1 * c1 + (1 - w1) * c2


def _bimodal_guard(w_small: float, bounds: tuple[float, float] = (0.05, 0.95)) -> bool:
    """判断小峰权重 w_small 是否落在合理双峰区间内(越界视为退化单峰拟合)。"""
    lo, hi = bounds
    return lo <= w_small <= hi


def fit_bimodal(d, cum, density, w_small_bounds=(0.05, 0.95),
                mode_prominence=0.5, mode_height=0.1):
    """一级判据:原始体积密度曲线模式计数。模式数<2→直接判单峰,不进入混合拟合。
    模式数>=2→走双对数正态混合拟合 + 既有 w_small 二级守卫(退化单峰兜底)。

    B = separation × min(w_small, ...) 形式:
        separation = |mu2 - mu1| / sqrt(s1^2 + s2^2)   (峰间距/峰宽,无量纲)
        small_w    = 小峰的权重
        B = separation * small_w * 2          (经验缩放至 ~[0, 几])
    单峰样品:小峰权重→0 或两峰位置重合 → B≈0。

    w_small_bounds: 小峰权重的合理区间。落界外说明"双峰"拟合实际退化为
    单峰(如 w_small≈0.99 意味着几乎全部质量在一个峰上),此时返回
    B_bimodal=NaN、note="degenerate_unimodal",其余诊断字段仍照实填。
    """
    n_modes, mode_positions = count_modes(d, density, prominence=mode_prominence,
                                          height=mode_height)
    if n_modes < 2:
        return dict(B_bimodal=np.nan,
                    peak1_um=(mode_positions[0] if mode_positions else np.nan),
                    peak2_um=np.nan, w_small=np.nan, separation=np.nan,
                    bimodal_r2=np.nan, n_modes=n_modes,
                    mode_positions_um=str(mode_positions),
                    note="unimodal_by_modecount")

    y = cum / 100.0
    # 初值:两峰分别落在分布的 25% 与 75% 分位附近
    d25, d50, d75 = (np.interp(p, cum, d) for p in (25, 50, 75))
    p0 = [0.3, np.log(max(d25, 1e-3)), 0.3, np.log(max(d75, 1e-3)), 0.3]
    bounds = ([0.0, np.log(d.min()), 0.05, np.log(d.min()), 0.05],
              [1.0, np.log(d.max()), 1.5, np.log(d.max()), 1.5])
    try:
        popt, _ = curve_fit(_bilognorm_cdf, d, y, p0=p0, bounds=bounds, maxfev=30000)
        w1, mu1, s1, mu2, s2 = popt
        # 规范:让 mu1 < mu2(峰1=小颗粒峰)
        if mu1 > mu2:
            w1, mu1, s1, mu2, s2 = (1 - w1), mu2, s2, mu1, s1
        separation = abs(mu2 - mu1) / np.sqrt(s1 ** 2 + s2 ** 2 + 1e-9)
        small_w = w1
        yhat = _bilognorm_cdf(d, *popt)
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        common = dict(n_modes=n_modes, mode_positions_um=str(mode_positions))
        if not _bimodal_guard(small_w, w_small_bounds):
            return dict(B_bimodal=np.nan, peak1_um=float(np.exp(mu1)),
                        peak2_um=float(np.exp(mu2)), w_small=float(small_w),
                        separation=float(separation), bimodal_r2=r2,
                        note="degenerate_unimodal", **common)
        B = float(separation * small_w * 2.0)
        return dict(B_bimodal=B, peak1_um=float(np.exp(mu1)),
                    peak2_um=float(np.exp(mu2)), w_small=float(small_w),
                    separation=float(separation), bimodal_r2=r2, note="ok", **common)
    except Exception as e:  # noqa
        return dict(B_bimodal=np.nan, peak1_um=np.nan, peak2_um=np.nan,
                    w_small=np.nan, separation=np.nan, bimodal_r2=np.nan,
                    n_modes=n_modes, mode_positions_um=str(mode_positions),
                    note=f"双峰拟合失败:{e}")


# ---------------------------------------------------------------------
# 3. 批处理
# ---------------------------------------------------------------------
def process_all(raw_dir: str | Path, out_csv: str | Path, cfg=None,
                dvalues_csv: str | Path | None = None):
    raw_dir = Path(raw_dir)
    if dvalues_csv is None:
        if cfg is not None:
            dvalues_csv = cfg.raw["instruments"]["laser_psd"].get(
                "dvalues_samples_csv", raw_dir / "psd_dvalues_samples.csv")
        else:
            dvalues_csv = raw_dir / "psd_dvalues_samples.csv"
    dvalues_csv = Path(dvalues_csv)
    instrument = load_instrument_dvalues(dvalues_csv) if dvalues_csv.exists() else None
    if instrument is None:
        warnings.warn(f"仪器 D 值权威表不存在: {dvalues_csv},全部样本回退自算 D 值"
                      "(非正式口径,仅供流程演练)")

    w_small_bounds = (0.05, 0.95)
    mode_prom, mode_height = 0.5, 0.1
    if cfg is not None:
        wb = cfg.raw["instruments"]["laser_psd"].get("bimodal_w_small_bounds")
        if wb:
            w_small_bounds = tuple(wb)
        mc = cfg.raw["instruments"]["laser_psd"].get("bimodal_modecount", {})
        mode_prom = mc.get("prominence", mode_prom)
        mode_height = mc.get("height", mode_height)

    rows = []
    for f in _iter_sample_csv_files(raw_dir):
        sid = f.stem
        d, cum = read_cumulative_csv(f)
        _, density = read_density_csv(f)
        D10_sc, D50_sc, D90_sc, span_sc = d_values(d, cum)
        if instrument is not None and sid in instrument.index:
            inst = instrument.loc[sid]
            D10, D50, D90, span = (float(inst.D10), float(inst.D50),
                                   float(inst.D90), float(inst.Span))
            source = "instrument"
        else:
            if instrument is not None:
                warnings.warn(f"{sid}: 仪器 D 值表缺该样本,回退自算值(非正式口径)")
            D10, D50, D90, span = D10_sc, D50_sc, D90_sc, span_sc
            source = "selfcalc_fallback"
        rec = {
            "sample_id": sid, "D10": D10, "D50": D50, "D90": D90, "Span": span,
            "D10_selfcalc": D10_sc, "D50_selfcalc": D50_sc,
            "D90_selfcalc": D90_sc, "Span_selfcalc": span_sc,
            "D_value_source": source,
        }
        do_bimodal = cfg is None or cfg.raw["instruments"]["laser_psd"]["bimodal_fit"]
        if do_bimodal:
            rec.update(fit_bimodal(d, cum, density, w_small_bounds=w_small_bounds,
                                   mode_prominence=mode_prom, mode_height=mode_height))
        rows.append(rec)
    out = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out
