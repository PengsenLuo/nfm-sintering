# -*- coding: utf-8 -*-
"""
xrd_processor.py —— XRD 处理器(模块二·处理器 3,v0.2)
======================================================
输入 : data/raw/xrd/<sample>.txt   (两列 2θ / 强度;首行可能为占位)
输出 : data/interim/xrd_features.csv  每样本一行:
         D_XRD / D_XRD_WH / microstrain_WH / lattice_a / lattice_c / c_a_ratio + QC

方法学(对应论文 X.2.2):
  1) 峰形 = Cu Kα1+Kα2 双线 pseudo-Voigt(强度比固定 0.5,峰位由波长比锁定,
     共享 FWHM/η)。仪器分辨率高(FWHM≈0.08–0.10°)时双线部分分裂,单峰模型会
     系统性高估 FWHM,故必须双线。所有峰位/FWHM 均为 Kα1 口径(λ=1.540593 Å)。
  2) 晶格参数:R-3m 指标化峰位最小二乘(同时拟合零点偏移)。
  3) 尺寸:
     - 主报值 D_XRD:Scherrer D=Kλ/(β·cosθ),β 为扣仪器宽化后的 Kα1 FWHM(弧度);
       仪器宽化用 Si 标样 Caglioti 曲线。
     - v0.2 新增稳健性校验 Williamson–Hall:β·cosθ = Kλ/D + 4ε·sinθ,
       对多峰线性回归,截距给 D_XRD_WH、斜率给微应变 ε(microstrain_WH)。
       这是对"仅取 (003)/(104) 平均"的稳健性校验,无需新增实验。
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from nfm import schema

SAMPLE_ID_RE = re.compile(r"^(C\d{2}-[SML])$")
DEG = np.pi / 180.0

# 退化解判据的容差(2026-06 修复):curve_fit 收敛到这些边界附近,说明拟合
# 没找到真实峰、只是被参数边界顶住,结果不可信(必须弃用,不能当正常值用)。
FWHM_LO_BOUND, FWHM_HI_BOUND = 0.02, 1.0
FWHM_BOUND_TOL = 1e-3
TWO_THETA_EDGE_TOL_DEG = 0.05

# 峰归属容差(2026-06 修复4):find_peaks 在窗口内找到的候选峰,离 guess_2th
# 先验位置超过这个距离就判定"没找到该 hkl 的可信峰"(很可能是邻近 hkl 的
# 旁峰漏进了窗口),不强行拟合。HKL_2THETA_GUESS 已用修复5 校准到接近真实
# 峰位(残差 <0.15°,见下方注释),0.5° 留足余量,不会误杀真实峰。
ATTRIBUTION_TOL_DEG = 0.5

# O3 相 R-3m 指标化峰(2θ 先验位置,用于定位拟合窗口中心;实际峰位由拟合给出)
#
# 2026-06 修复5:原表(003:16.0/006:32.2/101:35.8/012:38.5/104:40.8/
# 015:43.6/107:48.2/018:53.9/110:64.5/113:68.0)系统性偏离生产数据真实峰位
# (实测 003≈16.54、104≈41.53),部分峰(107/018)偏差甚至到 4–5°,导致窗口
# 经常完全没盖住真实峰、find_peaks 在错误窗口里随便抓一个峰交差,污染
# Williamson-Hall 回归和晶格参数拟合(这是修复2之后 wh_r2 仍然很低、
# lattice_c 仍然顶界的根因)。
#
# 重新推导:用本批 6 个端点样本(C07/C21 × L/M/S)里已确认可靠、样本间高度
# 一致的 (003)(104) 实测峰位(均值 16.537°/41.534°,6 个样本标准差 <0.01°),
# 反推 R-3m 六方晶格 a=2.9823Å、c=16.0689Å(公式:1/d²=4/3·(h²+hk+k²)/a²+l²/c²,
# d=λ/(2sinθ)),再用这组 (a,c) 正向算出其余 hkl 的理论 2θ(λ=Kα1=1.540593Å,
# zero_shift 取 0,因为这里只是给拟合窗口定中心,不是最终晶格解)。
#
# 已用 find_peaks 在全谱范围独立探测真实强峰交叉验证(对 C07-L/C07-M/C21-S
# 三个样本核对),理论值与实测峰位全部对应,残差 <0.15°(006/018 最大,
# 0.11–0.14°;其余 <0.05°),区别于旧表 4–5° 的偏差量级,确认这是自洽且
# 与数据吻合的先验表,不是凑出来的数字。
HKL_2THETA_GUESS = {
    "003": 16.537, "006": 33.431, "101": 35.164, "012": 36.514,
    "104": 41.534, "015": 44.992, "107": 53.352, "018": 58.132,
    "110": 62.206, "113": 64.853,
}


# ---------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------
def read_xy(path: Path):
    tt, ii = [], []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            a, b = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        tt.append(a); ii.append(b)
    return np.asarray(tt), np.asarray(ii)


def detect_instrument(path) -> int:
    """按文件内容(而非文件名/目录)判定仪器归属。

    - 含 `*FILE_SYSTEM_NAME "MiniFlex"` → 仪器2(MiniFlex,RAS_RAW 格式)
    - 首个非空行以 `?` 开头的纯文本格式 → 仪器1(SmartLab)
    - 都不匹配 → 抛错,不猜测(config 里的条件清单只用于交叉校验,见 process_all)
    """
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    if '*FILE_SYSTEM_NAME "MiniFlex"' in text:
        return 2
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("?"):
            return 1
        break
    raise ValueError(f"{path}: 无法识别仪器归属(既非 MiniFlex header 也非 '?' 开头纯文本)")


# 逐样全谱 2θ 零点估计只用这 5 个强且分离良好的参考峰(不用弱峰),
# 直接取自 HKL_2THETA_GUESS,不重复定义常量。
ZERO_OFFSET_REF = {k: HKL_2THETA_GUESS[k] for k in ("003", "006", "101", "104", "110")}


ZERO_OFFSET_TOP_K = 30  # 候选峰只取 prominence 最高的前 N 个


def estimate_zero_offset(tt, ii) -> dict:
    """全谱平移量 δ 网格搜索。

    δ ∈ [-1.5, 1.5],步长 0.01°:取 REF 中命中候选峰(±0.15° 内)数最多、
    并列时残差和最小的 δ。hits<4 视为"零点估计不可信",不施加校正(delta=0.0)。

    候选峰只取 prominence 最高的前 ZERO_OFFSET_TOP_K 个(真实衍射花样的强峰数量
    远小于此值,10 个 hkl 顶多);不加此上限时,纯噪声谱里 find_peaks 会在全谱
    找出成百上千个低显著性局部极大值,301 个 δ 网格点扫描下几乎总能凑出某个 δ
    让全部 5 个参考峰都"碰巧"命中,hits<4 的判据形同虚设。
    """
    prominence = 0.03 * (ii.max() - ii.min())
    pk_idx, props = find_peaks(ii, prominence=prominence)
    order = np.argsort(-props["prominences"])
    pk_idx = pk_idx[order][:ZERO_OFFSET_TOP_K]
    cand = tt[pk_idx]

    deltas = np.arange(-1.5, 1.5 + 1e-9, 0.01)
    best_hits, best_resid, best_delta = -1, np.inf, 0.0
    for d in deltas:
        hits = 0
        resid = 0.0
        for prior in ZERO_OFFSET_REF.values():
            if len(cand) == 0:
                continue
            target = prior + d
            diffs = np.abs(cand - target)
            j = int(np.argmin(diffs))
            if diffs[j] <= 0.15:
                hits += 1
                resid += float(diffs[j])
        if hits > best_hits or (hits == best_hits and resid < best_resid):
            best_hits, best_resid, best_delta = hits, resid, float(d)

    if best_hits < 4:
        return dict(delta=0.0, hits=int(best_hits), resid=float(best_resid))
    return dict(delta=float(best_delta), hits=int(best_hits), resid=float(best_resid))


# ---------------------------------------------------------------------
# Kα1+Kα2 双线 pseudo-Voigt
# ---------------------------------------------------------------------
def _pv(x, x0, fwhm, eta):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    g = np.exp(-0.5 * ((x - x0) / sigma) ** 2)
    l = 1.0 / (1.0 + ((x - x0) / (fwhm / 2)) ** 2)
    return eta * l + (1 - eta) * g


def _doublet(x, amp, x0_1, fwhm, eta, bg, lam1, lam2, ratio):
    """两条 pseudo-Voigt:峰2(Kα2)位置由 Bragg 锁定。"""
    th1 = np.radians(x0_1 / 2.0)
    s2 = np.clip(np.sin(th1) * lam2 / lam1, -1, 1)
    x0_2 = 2.0 * np.degrees(np.arcsin(s2))
    p1 = _pv(x, x0_1, fwhm, eta)
    p2 = _pv(x, x0_2, fwhm, eta)
    return bg + amp * (p1 + ratio * p2)


def fit_peak(tt, ii, guess_2th, window, lam1, lam2, ratio):
    """在 [guess_2th-window, guess_2th+window] 内拟合一个 Kα1+Kα2 doublet。

    guess_2th 只作为"搜索中心"(窗口边界 + curve_fit 里 x0 的搜索范围),
    不直接当 curve_fit 的初值——真实峰位可能偏离 guess_2th 0.5°+(生产数据
    与硬编码的 HKL_2THETA_GUESS 表之间常见这种偏差),若直接拿 guess_2th 当
    初值,容易让 curve_fit 收敛到参数边界,产出"看似成功但实际是退化解"的
    结果(2026-06 发现的 D_XRD 炸到 8e4nm 的根因)。
    改为:先用 find_peaks 在窗口内找真实局部强度最大值作真正初值;找不到
    显著峰就直接返回 None,不强行拟合。

    2026-06 修复4 追加:窗口内可能不止一个候选峰(邻近 hkl 的强峰落进同一
    窗口),取离 guess_2th 最近、且 prominence 达标的候选作初值,而不是
    简单取窗口内强度最高的——否则容易把邻近 hkl 的峰错当成当前 hkl 拟合,
    污染 Williamson-Hall/晶格参数(这是修复2 的边界检测看不出来的另一类
    退化:拟合本身收敛得很"正常",只是认错了峰)。若最近候选峰仍偏离
    guess_2th 超过 ATTRIBUTION_TOL_DEG,直接标记 degenerate(峰归属不可信),
    不进 curve_fit。

    返回:
      None                          —— 窗口内点数不足 / 没有任何候选峰
      dict(two_theta, fwhm, eta,
           degenerate, degenerate_reason)
                                    —— degenerate=True 时表示"峰归属不可信"
                                       或"收敛到了参数边界",结果不可信,
                                       调用方应当弃用(不要参与任何计算)。
    """
    m = (tt > guess_2th - window) & (tt < guess_2th + window)
    if m.sum() < 8:
        return None
    x, y = tt[m], ii[m]
    bg0 = np.percentile(y, 10)
    amp0 = max(y.max() - bg0, 1e-6)

    # 动态初值定位(修复1)+ 峰归属校验(修复4):窗口内可能不止一个候选峰
    # (邻近 hkl 的强峰也可能落进同一窗口),取离 guess_2th 先验位置最近、且
    # prominence 达标的那个作初值——不能简单取强度最高的,强度最高的可能是
    # 别的 hkl 漏进来的旁峰。若最近的候选峰仍偏离先验位置超过
    # ATTRIBUTION_TOL_DEG,判定窗口内没有该 hkl 的可信峰,标记退化(不进入
    # curve_fit,也不参与任何下游计算),而不是接受一个归属错误的峰。
    prominence = max((y.max() - bg0) * 0.1, 1e-9)
    pk_idx, _ = find_peaks(y, prominence=prominence)
    if len(pk_idx) == 0:
        return None
    cand_x = x[pk_idx]
    nearest_i = int(np.argmin(np.abs(cand_x - guess_2th)))
    x0_guess = float(cand_x[nearest_i])
    attribution_offset = abs(x0_guess - guess_2th)
    if attribution_offset > ATTRIBUTION_TOL_DEG:
        return dict(
            two_theta=x0_guess, fwhm=float("nan"), eta=float("nan"),
            degenerate=True,
            degenerate_reason=(f"峰归属不可信:最近候选峰在{x0_guess:.3f}°,"
                              f"偏离先验{guess_2th:.3f}°达{attribution_offset:.3f}°"
                              f"(>{ATTRIBUTION_TOL_DEG}°)"))

    p0 = [amp0, x0_guess, 0.12, 0.5, bg0]
    try:
        popt, _ = curve_fit(
            lambda x, amp, x0, fwhm, eta, bg:
                _doublet(x, amp, x0, fwhm, eta, bg, lam1, lam2, ratio),
            x, y, p0=p0, maxfev=20000,
            bounds=([0, guess_2th - window, 0.02, 0, -np.inf],
                    [np.inf, guess_2th + window, 1.0, 1, np.inf]))
        amp, x0, fwhm, eta, bg = popt
    except Exception:  # noqa
        return None

    # 退化解检测(修复2):curve_fit 跑完不代表拟合可信,贴边界 = 没找到真实峰。
    reasons = []
    if fwhm <= FWHM_LO_BOUND + FWHM_BOUND_TOL or fwhm >= FWHM_HI_BOUND - FWHM_BOUND_TOL:
        reasons.append(f"fwhm={fwhm:.4f}贴边界[{FWHM_LO_BOUND},{FWHM_HI_BOUND}]")
    lo_edge, hi_edge = guess_2th - window, guess_2th + window
    if x0 <= lo_edge + TWO_THETA_EDGE_TOL_DEG or x0 >= hi_edge - TWO_THETA_EDGE_TOL_DEG:
        reasons.append(f"two_theta={x0:.3f}贴窗口边界[{lo_edge:.2f},{hi_edge:.2f}]")

    return dict(two_theta=float(x0), fwhm=float(fwhm), eta=float(eta),
               degenerate=bool(reasons), degenerate_reason=";".join(reasons))


# ---------------------------------------------------------------------
# 仪器宽化(Caglioti,Si 标样)
# ---------------------------------------------------------------------
def instrument_fwhm(two_theta_deg, caglioti):
    """FWHM_inst = sqrt(U tan²θ + V tanθ + W) (度)。caglioti=(U,V,W)。"""
    U, V, W = caglioti
    th = np.radians(two_theta_deg / 2.0)
    val = U * np.tan(th) ** 2 + V * np.tan(th) + W
    return float(np.sqrt(max(val, 1e-8)))


def _beta_sample_rad(fwhm_obs_deg, two_theta_deg, caglioti):
    """扣仪器宽化后的样品宽化(弧度),高斯近似平方相减。

    若 FWHM_obs² ≤ FWHM_inst²(观测宽度不超过仪器宽度),物理上没有可解析的
    样品额外宽化,返回 NaN——2026-06 之前这里用 max(b2, 1e-8) 把负数/零兜成
    一个极小的假正数,导致 Scherrer 公式分母趋零、D_XRD 炸到 8e4nm 量级,
    是当时那个 bug 的直接元凶,现在不再兜底。"""
    fi = instrument_fwhm(two_theta_deg, caglioti)
    b2 = fwhm_obs_deg ** 2 - fi ** 2
    if b2 <= 0:
        return float("nan")
    return np.sqrt(b2) * DEG


# ---------------------------------------------------------------------
# 尺寸:Scherrer(主报)+ Williamson–Hall(校验)
# ---------------------------------------------------------------------
def scherrer(fwhm_obs_deg, two_theta_deg, lam_A, K, caglioti):
    beta = _beta_sample_rad(fwhm_obs_deg, two_theta_deg, caglioti)
    if not np.isfinite(beta) or beta <= 0:
        return float("nan")
    th = np.radians(two_theta_deg / 2.0)
    D_nm = K * (lam_A * 0.1) / (beta * np.cos(th))   # λ: Å→nm
    return float(D_nm)


def williamson_hall(peaks, lam_A, K, caglioti):
    """对多峰做 WH 回归:y = β·cosθ, x = 4 sinθ;y = Kλ/D + ε·x。
    返回 (D_WH_nm, microstrain, r2)。peaks: list of dict(two_theta, fwhm)。
    """
    xs, ys = [], []
    for pk in peaks:
        if pk is None or pk.get("degenerate"):
            continue
        tt = pk["two_theta"]
        beta = _beta_sample_rad(pk["fwhm"], tt, caglioti)   # rad
        if not np.isfinite(beta):
            continue
        th = np.radians(tt / 2.0)
        ys.append(beta * np.cos(th))
        xs.append(4.0 * np.sin(th))
    if len(xs) < 3:
        return dict(D_XRD_WH=np.nan, microstrain_WH=np.nan, wh_r2=np.nan,
                   wh_n_peaks=len(xs))
    xs, ys = np.asarray(xs), np.asarray(ys)
    A = np.vstack([xs, np.ones_like(xs)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, ys, rcond=None)
    yhat = A @ np.array([slope, intercept])
    ss_res = float(np.sum((ys - yhat) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    D_WH = (K * (lam_A * 0.1) / intercept) if intercept > 1e-9 else np.nan
    return dict(D_XRD_WH=float(D_WH), microstrain_WH=float(max(slope, 0.0)),
                wh_r2=float(r2), wh_n_peaks=len(xs))


# ---------------------------------------------------------------------
# 水合/残碱相污染指标(hydrate_index,2026-08-05 新增)
# ---------------------------------------------------------------------
# 层状氧化物正极吸潮/吸CO2后在 (003) 峰低角侧(2θ≈14.4°)出现层间膨胀相
# (d≈6.14Å,比正常 (003) 层间距 5.36Å 大 14.6%)。hydrate_index 度量该相
# 相对 (003) 主峰的面积占比,用作样品退化/污染的质控标志(非建模目标,
# 见 schema.py Role.QC)。定义(硬性,不得自行改动窗口/公式):
#   A_hyd = 13.8–15.2° 区间积分面积,先扣两侧线性基线(左窗 12.8–13.6°、
#           右窗 15.4–16.0°,取各窗口中位数强度作为基线两端点、线性插值
#           到积分区间),负值截断为 0。
#   A_003 = 16.0–17.2° 区间积分面积,先扣水平基线(左窗 15.4–15.9°、右窗
#           17.3–17.8° 合并后取中位数强度,作为常数基线)。
#   hydrate_index = 100 * A_hyd / A_003   (%)
# 积分方法:np.trapezoid(梯形积分),下同两个区域一致。
# 输入的 tt 必须是零点校正后的角度(与 process_all 中 estimate_zero_offset
# 校正后的 tt 一致),不在本函数内重复估计/施加零点校正。
HYDRATE_A_HYD_WINDOW = (13.8, 15.2)
HYDRATE_A_HYD_BASE_LEFT = (12.8, 13.6)
HYDRATE_A_HYD_BASE_RIGHT = (15.4, 16.0)
HYDRATE_A003_WINDOW = (16.0, 17.2)
HYDRATE_A003_BASE_LEFT = (15.4, 15.9)
HYDRATE_A003_BASE_RIGHT = (17.3, 17.8)
HYDRATE_SCAN_START_MAX_DEG = 13.8  # 扫描起点高于此值 → 数据不覆盖 A_hyd 起点,返回 NaN


def _window_median(tt, ii, lo, hi):
    m = (tt >= lo) & (tt <= hi)
    if not m.any():
        return None
    return float(np.median(ii[m]))


def compute_hydrate_index(tt, ii):
    """水合/残碱相污染指标(%)= 100 * A_hyd / A_003。详见上方模块注释。

    tt: 零点校正后的 2θ(度),ii: 对应强度。
    扫描起点高于 13.8° 的文件(不覆盖 A_hyd 区间起点)返回 NaN,不报错。
    任一所需窗口(积分区间或基线窗)在数据范围内点数不足,同样返回 NaN。
    """
    tt = np.asarray(tt, dtype=float)
    ii = np.asarray(ii, dtype=float)
    if tt.size == 0 or tt.min() > HYDRATE_SCAN_START_MAX_DEG:
        return float("nan")

    # --- A_hyd:线性基线(左右窗中位数作为基线两端点,线性插值) ---
    m_hyd = (tt >= HYDRATE_A_HYD_WINDOW[0]) & (tt <= HYDRATE_A_HYD_WINDOW[1])
    left_med = _window_median(tt, ii, *HYDRATE_A_HYD_BASE_LEFT)
    right_med = _window_median(tt, ii, *HYDRATE_A_HYD_BASE_RIGHT)
    if m_hyd.sum() < 2 or left_med is None or right_med is None:
        return float("nan")
    x_hyd, y_hyd = tt[m_hyd], ii[m_hyd]
    x_left_c = sum(HYDRATE_A_HYD_BASE_LEFT) / 2.0
    x_right_c = sum(HYDRATE_A_HYD_BASE_RIGHT) / 2.0
    baseline_hyd = left_med + (right_med - left_med) * (x_hyd - x_left_c) / (x_right_c - x_left_c)
    A_hyd = float(np.trapezoid(y_hyd - baseline_hyd, x_hyd))
    A_hyd = max(A_hyd, 0.0)

    # --- A_003:水平基线(左右窗合并后取中位数) ---
    m_003 = (tt >= HYDRATE_A003_WINDOW[0]) & (tt <= HYDRATE_A003_WINDOW[1])
    m_003_base = (
        ((tt >= HYDRATE_A003_BASE_LEFT[0]) & (tt <= HYDRATE_A003_BASE_LEFT[1]))
        | ((tt >= HYDRATE_A003_BASE_RIGHT[0]) & (tt <= HYDRATE_A003_BASE_RIGHT[1]))
    )
    if m_003.sum() < 2 or not m_003_base.any():
        return float("nan")
    x_003, y_003 = tt[m_003], ii[m_003]
    baseline_003 = float(np.median(ii[m_003_base]))
    A_003 = float(np.trapezoid(y_003 - baseline_003, x_003))

    if not np.isfinite(A_hyd) or not np.isfinite(A_003) or A_003 <= 0:
        return float("nan")
    return 100.0 * A_hyd / A_003


# ---------------------------------------------------------------------
# 晶格参数(R-3m 六方,最小二乘 + 零点偏移)
# ---------------------------------------------------------------------
def _dspacing(lam_A, two_theta, zero_shift):
    th = np.radians((two_theta - zero_shift) / 2.0)
    return lam_A / (2.0 * np.sin(th))


def lattice_params(indexed_peaks, lam_A):
    """indexed_peaks: dict{hkl: two_theta}。最小二乘解 a,c 与零点。
    1/d² = 4/3·(h²+hk+k²)/a² + l²/c²。
    """
    HKL = {"003": (0, 0, 3), "006": (0, 0, 6), "101": (1, 0, 1),
           "012": (0, 1, 2), "104": (1, 0, 4), "015": (0, 1, 5),
           "107": (1, 0, 7), "018": (0, 1, 8), "110": (1, 1, 0),
           "113": (1, 1, 3)}

    def resid(params):
        a, c, z = params
        r = []
        for hkl, tt in indexed_peaks.items():
            if hkl not in HKL:
                continue
            h, k, l = HKL[hkl]
            d = _dspacing(lam_A, tt, z)
            inv_d2_obs = 1.0 / d ** 2
            inv_d2_cal = 4.0 / 3.0 * (h * h + h * k + k * k) / a ** 2 + l * l / c ** 2
            r.append(inv_d2_obs - inv_d2_cal)
        return np.asarray(r)

    from scipy.optimize import least_squares
    sol = least_squares(resid, x0=[2.98, 16.0, 0.0],
                        bounds=([2.8, 15.5, -0.2], [3.1, 16.6, 0.2]))
    a, c, z = sol.x
    return float(a), float(c), float(z)


# ---------------------------------------------------------------------
# 批处理
# ---------------------------------------------------------------------
def _clip_to_schema_bounds(rec: dict, sid: str) -> dict:
    """出口拦截(修复3):对 rec 里凡是 schema.py 声明了 bounds 的列,越界值
    置 NaN 并告警,不放宽 schema 里的 bounds 本身——bounds 是科学约束,
    应该是数据/拟合去满足它,不是反过来改 bounds 迁就坏数据。"""
    for col in list(rec.keys()):
        spec = schema.BY_NAME.get(col)
        if spec is None or spec.bounds is None:
            continue
        val = rec[col]
        if val is None:
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        lo, hi = spec.bounds
        if not (lo <= v <= hi):
            warnings.warn(f"{sid}: {col}={v:.4g} 超出 schema bounds [{lo},{hi}],置 NaN")
            rec[col] = float("nan")
    return rec


def process_all(raw_dir, out_csv, cfg):
    raw_dir = Path(raw_dir)
    xc = cfg.raw["instruments"]["xrd"]
    lam1 = xc["wavelength_A"]; lam2 = xc["ka2_wavelength_A"]; ratio = xc["ka2_ratio"]
    K = xc["scherrer_K"]
    caglioti_inst1 = tuple(xc.get("caglioti_UVW", (0.001, -0.0005, 0.004)))  # 占位,标样标定后回填
    caglioti_inst2 = tuple(xc.get("caglioti_UVW_inst2", caglioti_inst1))
    instrument2_conditions = set(xc.get("instrument2_conditions", []))
    resolution_ratio_min = xc.get("resolution_ratio_min", 1.3)
    wh_peaks = xc.get("wh_peaks", list(HKL_2THETA_GUESS.keys()))

    rows = []
    for f in sorted(raw_dir.glob("*.txt")):
        sid = f.stem
        if not SAMPLE_ID_RE.match(sid):
            continue

        # 逐样仪器路由:以文件内容为准,config 的 instrument2_conditions 只作交叉
        # 校验,分歧时告警但以文件检测结果为准。
        inst = detect_instrument(f)
        condition_id = sid.split("-")[0]
        cross_check_inst2 = condition_id in instrument2_conditions
        if (inst == 2) != cross_check_inst2:
            warnings.warn(
                f"{sid}: 文件检测仪器={inst},但 config.instrument2_conditions "
                f"{'包含' if cross_check_inst2 else '不包含'}其条件 {condition_id}"
                f"(存在分歧,以文件检测结果为准)")
        caglioti = caglioti_inst2 if inst == 2 else caglioti_inst1

        # 逐样全谱 2θ 零点预对齐(§2.3):对全部样本一律执行,不按仪器/阈值门控。
        tt_raw, ii = read_xy(f)
        zo = estimate_zero_offset(tt_raw, ii)
        if zo["hits"] < 4:
            warnings.warn(f"{sid}: 零点估计不可信(hits={zo['hits']}<4),不施加校正")
        tt = tt_raw - zo["delta"]

        fitted = {}
        for hkl, g in HKL_2THETA_GUESS.items():
            pk = fit_peak(tt, ii, g, window=1.0, lam1=lam1, lam2=lam2, ratio=ratio)
            if pk is None:
                continue
            if pk["degenerate"]:
                warnings.warn(f"{sid}: {hkl} 峰拟合退化({pk['degenerate_reason']}),弃用此峰")
                continue
            fitted[hkl] = pk

        rec = {"sample_id": sid, "xrd_instrument": inst,
              "two_theta_offset_applied": zo["delta"], "zero_offset_hits": zo["hits"]}

        # 水合/残碱相污染质控指标(2026-08-05 新增,Role.QC,非建模目标)。
        # 用同一个零点校正后的 tt,不重复估计零点。
        rec["hydrate_index"] = compute_hydrate_index(tt, ii)

        # 分辨率上限判据(§2.4):用未校零的观测 FWHM(零点校正只平移峰位,不改变
        # FWHM)对本样本所属仪器的 Caglioti 曲线算 β_obs/β_inst,取 (003)(104) 较小值。
        ratios = []
        for hkl in ("003", "104"):
            if hkl in fitted:
                fi = instrument_fwhm(fitted[hkl]["two_theta"], caglioti)
                if fi > 0:
                    ratios.append(fitted[hkl]["fwhm"] / fi)
        resolution_ratio = float(min(ratios)) if ratios else float("nan")
        reliable = bool(np.isfinite(resolution_ratio) and resolution_ratio >= resolution_ratio_min)
        rec["resolution_ratio"] = resolution_ratio
        rec["D_XRD_reliable"] = reliable

        # 主报 D_XRD:003/104 各自算 Scherrer,只用有限值的那些平均;
        # 两个都不可用就是 NaN,不用兜底值假装算出来。超出仪器分辨率上限时
        # 即便算得出有限值也置 NaN(反卷积病态,数值不可信)。
        # 同时把 (003)/(104) 单峰分量各自存成 D_XRD_003/D_XRD_104(诊断列,进
        # schema),供各向异性诊断使用;D_XRD 本身的计算方式/数值完全不变,
        # 仍是两者平均。
        d_list = []
        d_per_hkl = {}
        for hkl in ("003", "104"):
            if hkl in fitted:
                d = scherrer(fitted[hkl]["fwhm"], fitted[hkl]["two_theta"], lam1, K, caglioti)
                if np.isfinite(d):
                    d_list.append(d)
                    d_per_hkl[hkl] = d
        if d_list and not reliable:
            warnings.warn(f"{sid}: resolution_ratio={resolution_ratio:.3f} < {resolution_ratio_min},"
                          "超出该仪器分辨率上限,反卷积病态,D_XRD 置 NaN")
        rec["D_XRD"] = float(np.mean(d_list)) if (d_list and reliable) else np.nan
        rec["D_XRD_003"] = d_per_hkl.get("003", np.nan) if reliable else np.nan
        rec["D_XRD_104"] = d_per_hkl.get("104", np.nan) if reliable else np.nan
        rec["D_XRD_n_peaks"] = len(d_list)  # 诊断列,非 schema 列:实际用了几个有效峰
        if not d_list:
            warnings.warn(f"{sid}: (003)(104) 均不可用,D_XRD 置 NaN")
        # WH 校验
        rec.update(williamson_hall([fitted.get(h) for h in wh_peaks], lam1, K, caglioti))
        # 晶格
        if len(fitted) >= 4:
            idx = {hkl: pk["two_theta"] for hkl, pk in fitted.items()}
            a, c, z = lattice_params(idx, lam1)
            rec.update(lattice_a=a, lattice_c=c, c_a_ratio=c / a, zero_shift=z)
        rec = _clip_to_schema_bounds(rec, sid)
        rows.append(rec)

    out = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out
