# -*- coding: utf-8 -*-
"""packing_models.py —— 颗粒堆积模型计算引擎

背景与范围（务必先读）
======================
论文 §4.1 现有"简单混合预测"段落（0.7ρ_L+0.3ρ_S=3.327 与实测差 +0.062）把压实密度
当线性混合量处理，这在物理上不成立——密度（本质是堆积效率的函数）对粒径级配是
非线性的，小颗粒填充大颗粒间隙会产生级配增益，这正是本模块要计算的东西。

**本模块只是"计算引擎"本身，不产出可替换论文 §4.1 的结论**：真正要替换 §4.1，
需要（1）产物（免水）压实密度实测值、（2）产物 PSD 实测曲线，两者目前均是
`[待测]`。本模块用前驱体 PSD（已实测）验证模型实现是否正确，但前驱体的"堆积分数"
是几何堆积计算量，不是产物压实密度实测值，两者是不同物理量，不能相互替代，
见 `scripts/19_packing_model_demo.py`。

两个模型
========
1. Furnas 二元堆积模型（`furnas_binary_packing_fraction`）
   出处：Furnas, C.C. (1929), "Flow of Gases through Beds of Broken Solids",
   U.S. Bureau of Mines Bulletin 307；及更易获取的后续版本 Furnas, C.C. (1931),
   "Grading Aggregates I — Mathematical Relations for Beds of Broken Solids of
   Maximum Density", Industrial & Engineering Chemistry, 23(9), 1052–1058。
   本实现中"粗颗粒主导支路"与"细颗粒主导支路"两条曲线取 min 的经典表述，及
   最大理论堆积分数 φ_max = φ_c + φ_f·(1-φ_c) 的闭式结果，同样见于教科书整理版本
   R. M. German, *Particle Packing Characteristics*, Metal Powder Industries
   Federation, 1989, Ch. 4（二元堆积一节）。

   Furnas 原始理论假设"极端尺寸比"（小颗粒相对大颗粒尺寸→0，二者互不干扰）。
   为了让函数能接受任意 d_small/d_large 并在比值→1 时正确退化（不产生虚假增益，
   这是三条健全性检查里明确要求的），本模块在 Furnas 两支路曲线与"无增益基线"
   （比表体积可加的调和平均，phi_c=phi_f 时精确退化为常数 phi0）之间，用一个
   尺寸比阻尼因子 η(r)=1-r 做线性插值。**这个阻尼因子是本实现为扩展到有限尺寸比
   而添加的工程手段，不是 Furnas(1929) 原文的一部分**——Furnas 原文完全没有处理
   有限尺寸比，这正是后续 Toufar/LPM/CPM 模型要解决的问题（见下）。阻尼方向（尺寸比
   越接近1，级配增益越弱）有明确定性文献依据：二元堆积的 loosening/wall 效应"在
   尺寸比不是极端情况时才重要"，级配增益随尺寸比趋近1而消失（见 LPM 部分的
   loosening/wall 效应系数，以及综述如 Yu, A.B. & Standish, N. (1991), "Estimation
   of the porosity of multi-component mixtures of particles", Powder Technology,
   52, 233–241 的定性表述）。此阻尼因子的具体函数形式（线性）是工程近似，代码里
   明确标注，不冒充是某篇论文的原始公式。

2. 线性堆积模型 LPM（`lpm_packing_fraction` / `LPMComponent`）
   出处：Stovall, T., de Larrard, F., Buil, M. (1986), "Linear Packing Density
   Model of Grain Mixtures", Powder Technology, 48(1), 1–12
   (doi:10.1016/0032-5910(86)80058-4)。核心假设：把连续 PSD 离散成 n 个粒径档
   （class），每档有自己的体积分数 y_i 与单组分堆积分数 β_i；某一档 i "主导"
   堆积骨架时，比它粗的档（j<i，按粒径降序编号）通过"wall effect"系数 b_ij 扰动
   它，比它细的档（j>i）通过"loosening effect"系数 a_ij 扰动它；混合物实际堆积
   分数取所有可能主导档 i 算出的虚拟堆积分数 γ_i 中的最小值（min，与 Furnas
   两支路取 min 是同一逻辑的多组分推广）。

   loosening/wall 效应系数作为尺寸比 r=d_j/d_i（细/粗）的函数：
   - wall effect:      b(r) = 1 - (1-r)^1.6
   - loosening effect: a(r) = 0                                  (r < x0)
                        a(r) = ((r-x0)/(1-x0))^1.6                (r >= x0)
     其中临界尺寸比 x0 ≈ 0.2（低于此值，细颗粒可无干扰地填入粗颗粒孔隙——这是
     Stovall/de Larrard/Buil (1986) 原文明确提出的"临界孔穴尺寸比"概念）。
   b(r) 的具体幂函数形式（指数 1.6）在本任务开发过程中通过检索到的二手文献转述
   （含 PMC 综述文章对该论文的公式复述）交叉确认；a(r) 在临界比 x0 以上的精确
   代数形式在原文中更复杂（分段有理式），本实现未能从可得的二手来源可靠核实其
   精确系数，因此**没有照抄一个不确定的公式**，改用一个满足原文档明确记录的边界
   条件（a(x0)=0，a(1)=1，单调）的透明幂函数近似，并在此明确注明是近似，不是
   逐字复现原文公式。两个系数函数都满足 r→1 时 a=b=1（无级配增益）、r→0 附近
   a=0（细颗粒不扰动粗骨架，与 Furnas 理想极限一致）的物理边界，因此模型在
   Furnas 二元极限下与本模块的 Furnas 实现方向一致（不要求数值逐位相同，两者
   是不同参数化）。

   本实现用"共享单组分堆积分数 β"简化（β_i=β_j=β for all i,j）：因为本任务的
   应用场景是同一种前驱体/产物粉体只是粒径切割不同，同种材料同种形貌的球形/
   准球形颗粒，单颗粒随机堆积分数近似与绝对尺寸无关（只依赖形状/摩擦），故取
   共享值是合理简化；若未来需要处理形状随尺寸系统变化的体系（如本项目 SEM 观察
   到的形貌记忆现象本身），需要扩展为逐档 β_i，本模块当前不支持，接口里显式
   注明。

单位与约定
==========
- 所有"堆积分数"（packing fraction）φ 定义为 固体体积 / 表观（堆积）体积，
  取值范围 (0, 1)；孔隙率 = 1-φ。
- 尺寸：任意一致长度单位均可（本模块用 µm，对应前驱体 PSD 数据）。
- 体积分数 f_small / y_i：颗粒体系里"固体体积"的分数（不是质量分数），
  Σy_i = 1。

算不出来的情形（如 size ratio 超出 (0,1] 或体积分数不在 [0,1]）一律抛
`ValueError`，不静默返回猜测值；调用方若拿到 NaN 应视为"模型未定义"。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------
# 通用单组分随机堆积分数默认值
# ---------------------------------------------------------------------
# 文献典型值：随机疏堆积(random loose packing)≈0.60，随机密堆积
# (random close packing)≈0.64（等径球，重力沉降）。这里取 0.64 作为默认，
# 属于模型自带参数，可由调用方覆盖；健全性检查只要求"自洽"，不要求精确匹配
# 某个具体文献数值。
DEFAULT_PHI0 = 0.64


# =======================================================================
# 1. Furnas 二元堆积模型
# =======================================================================
def _furnas_ideal_branches(f_small: np.ndarray, phi_large: float, phi_small: float):
    """Furnas(1929/1931) 理想极端尺寸比二元堆积的两条支路（推导见模块 docstring）。

    支路 1（粗颗粒主导 / 细颗粒填充粗骨架孔隙、不扰动粗骨架体积）：
        phi_1(f_s) = phi_large / (1 - f_s)
    支路 2（细颗粒主导 / 粗颗粒作为"大块"分散在细颗粒基体中，直接按自身固体
        体积挤占表观体积，不利用细基体已有孔隙）：
        phi_2(f_s) = phi_small / (f_s + phi_small*(1 - f_s))

    两支路交点即级配增益最大点；理论最大堆积分数（在交点处）化简为经典结果
    phi_max = phi_large + phi_small*(1-phi_large)（German 1989 Ch.4 给出同一结果）。
    """
    phi_1 = phi_large / (1.0 - f_small)
    phi_2 = phi_small / (f_small + phi_small * (1.0 - f_small))
    return phi_1, phi_2


def furnas_binary_packing_fraction(
    d_large: float,
    d_small: float,
    f_small: float,
    phi_large: float = DEFAULT_PHI0,
    phi_small: float = DEFAULT_PHI0,
) -> float:
    """Furnas 二元堆积模型预测的混合堆积分数（含有限尺寸比阻尼，见模块 docstring）。

    Parameters
    ----------
    d_large, d_small : float
        大 / 小颗粒特征尺寸（任意一致长度单位，如 µm）。若传入 d_small > d_large，
        内部按大小自动纠正（不假设调用方一定传对顺序），但 f_small 仍按"体积分数
        较小特征尺寸那一组分"解释。
    f_small : float
        小颗粒组分的体积分数，∈[0,1]。
    phi_large, phi_small : float
        大 / 小颗粒单独存在时的单组分堆积分数（默认共用 DEFAULT_PHI0=0.64，
        即假设两者是同材料同形状、只是尺寸不同）。

    Returns
    -------
    float
        预测堆积分数 φ_mix ∈ (0,1)。d_large<=0 或 d_small<=0 或 f_small 不在
        [0,1] 时抛 ValueError（不返回猜测值）。
    """
    if d_large <= 0 or d_small <= 0:
        raise ValueError(f"颗粒特征尺寸必须为正: d_large={d_large}, d_small={d_small}")
    if not (0.0 <= f_small <= 1.0):
        raise ValueError(f"f_small 必须在 [0,1]: 传入 {f_small}")
    if not (0.0 < phi_large < 1.0) or not (0.0 < phi_small < 1.0):
        raise ValueError("phi_large/phi_small 必须在 (0,1)")

    d_hi, d_lo = max(d_large, d_small), min(d_large, d_small)
    r = d_lo / d_hi  # ∈(0,1]，1=等尺寸

    # 端点直接返回单组分堆积分数，避免除零（f_small=0 时支路1分母为1，本可
    # 直接算，但 f_small=1 时支路1分母为0——显式端点处理更稳健、更易读）。
    if f_small == 0.0:
        return float(phi_large)
    if f_small == 1.0:
        return float(phi_small)

    phi_1, phi_2 = _furnas_ideal_branches(np.array([f_small]), phi_large, phi_small)
    phi_ideal = float(min(phi_1[0], phi_2[0]))

    # 无级配增益基线：比堆积体积(1/phi)按体积分数线性可加(harmonic mixing)，
    # phi_large=phi_small 时此基线精确等于该共同值(常数, 对 f_small 不变)，
    # 正是"尺寸比→1 时不应有增益"这一健全性检查要求的基线形状。
    f_large = 1.0 - f_small
    phi_linear = 1.0 / (f_large / phi_large + f_small / phi_small)

    eta = 1.0 - r  # 尺寸比阻尼因子(工程近似，见模块 docstring，非 Furnas 原文)
    phi_mix = phi_linear + eta * (phi_ideal - phi_linear)
    return float(phi_mix)


# =======================================================================
# 2. 线性堆积模型 LPM（Stovall / de Larrard / Buil, 1986）
# =======================================================================
def lpm_wall_effect(r: np.ndarray | float) -> np.ndarray | float:
    """wall effect 系数 b(r)，r=d_fine/d_coarse ∈[0,1]（细/粗尺寸比）。

    b(r) = 1 - (1-r)^1.6，出处见模块 docstring(Stovall/de Larrard/Buil 1986,
    经二手文献转述交叉确认)。r=0 → b=0(粗颗粒不扰动细骨架)；r=1 → b=1(等尺寸,
    扰动最大)。
    """
    r = np.clip(np.asarray(r, dtype=float), 0.0, 1.0)
    return 1.0 - (1.0 - r) ** 1.6


def lpm_loosening_effect(r: np.ndarray | float, x0: float = 0.2) -> np.ndarray | float:
    """loosening effect 系数 a(r)，r=d_fine/d_coarse ∈[0,1]。

    分段:r<x0 时 a=0(细颗粒尺寸低于临界孔穴尺寸比,可无扰动地填入粗骨架孔隙,
    这是 Stovall/de Larrard/Buil(1986)原文明确提出的临界比概念,x0≈0.2);
    r>=x0 时用满足边界条件 a(x0)=0、a(1)=1 的单调幂函数近似
    a(r)=((r-x0)/(1-x0))^1.6 —— **此幂函数的具体形式是本实现的透明近似**，
    原文在 x0 以上的精确代数式更复杂，未能从可得二手来源可靠核实系数，故不
    冒充逐字复现，见模块 docstring。
    """
    r = np.clip(np.asarray(r, dtype=float), 0.0, 1.0)
    out = np.where(r < x0, 0.0, ((np.clip(r, x0, 1.0) - x0) / (1.0 - x0)) ** 1.6)
    return out


@dataclass
class LPMComponent:
    """LPM 的一个粒径档(size class)。d: 特征尺寸(µm等), y: 该档体积分数(占总固体)。"""

    d: float
    y: float


def _lpm_gamma_for_dominant(
    idx_sorted: np.ndarray,
    d_sorted: np.ndarray,
    y_sorted: np.ndarray,
    beta: float,
    x0: float,
) -> float:
    """给定按粒径降序排好的 (d_sorted, y_sorted)，算以 idx_sorted 位置为主导档
    的虚拟堆积分数 gamma_i(共享 beta 简化形式，推导见模块 docstring)。"""
    n = len(d_sorted)
    i = idx_sorted
    d_i = d_sorted[i]
    wall_term = 0.0
    for j in range(0, i):  # j < i：比 i 粗，wall effect
        r = d_i / d_sorted[j]  # 细(i)/粗(j)
        b = float(lpm_wall_effect(r))
        wall_term += (1.0 - b) * y_sorted[j]
    loosen_term = 0.0
    for j in range(i + 1, n):  # j > i：比 i 细，loosening effect
        r = d_sorted[j] / d_i  # 细(j)/粗(i)
        a = float(lpm_loosening_effect(r, x0=x0))
        loosen_term += (1.0 - a) * y_sorted[j]
    denom = 1.0 - (1.0 - beta) * wall_term - loosen_term
    if denom <= 0:
        return np.nan  # 模型在此组合下无物理解(骨架被完全挤垮)，如实返回 NaN
    return beta / denom


def lpm_packing_fraction(
    components: list[LPMComponent],
    beta: float = DEFAULT_PHI0,
    x0: float = 0.2,
) -> float:
    """LPM 预测的多组分(可含 n>=2 档，支持完整 PSD 离散化后的输入)混合堆积分数。

    Parameters
    ----------
    components : list[LPMComponent]
        每个元素是一个粒径档 (d, y)。y 之和必须 >0；内部会自动归一化到 1(允许
        调用方传入未归一化的体积权重，比如某个档在总 PSD 里的原始体积%)。
        d 必须全部为正；两个不同档不要求 d 各不相同，但若出现完全重复的 d 会被
        自动合并(否则同 d 档互相的 r=1 会使某些主导档配置产生退化解)。
    beta : float
        共享的单组分堆积分数(所有档同材料同形状假设，见模块 docstring)。
    x0 : float
        loosening effect 临界尺寸比，默认 0.2(Stovall/de Larrard/Buil 1986)。

    Returns
    -------
    float
        预测堆积分数 φ ∈ (0,1)，为所有候选主导档 gamma_i 的最小值(与 Furnas
        二元模型两支路取 min 同一逻辑的多组分推广)。若所有候选都无物理解，
        返回 NaN。
    """
    if not components:
        raise ValueError("components 不能为空")
    if any(c.d <= 0 for c in components):
        raise ValueError("所有粒径档的 d 必须为正")
    if any(c.y < 0 for c in components):
        raise ValueError("体积分数不能为负")

    # 合并重复 d(避免 r=1 自比较带来的除零/退化歧义)，并按体积分数归一化。
    merged: dict[float, float] = {}
    for c in components:
        merged[c.d] = merged.get(c.d, 0.0) + c.y
    total_y = sum(merged.values())
    if total_y <= 0:
        raise ValueError("体积分数总和必须 >0")

    items = sorted(merged.items(), key=lambda kv: kv[0], reverse=True)  # 粗→细
    d_sorted = np.array([d for d, _ in items], dtype=float)
    y_sorted = np.array([y / total_y for _, y in items], dtype=float)

    if len(d_sorted) == 1:
        return float(beta)

    # 只有 y_i>0 的档才有资格作为"主导档"候选(y_i=0 的档不存在，不该主导堆积)。
    gammas = []
    for i in range(len(d_sorted)):
        if y_sorted[i] <= 0:
            continue
        gammas.append(_lpm_gamma_for_dominant(i, d_sorted, y_sorted, beta, x0))
    gammas = [g for g in gammas if not np.isnan(g)]
    if not gammas:
        return float("nan")
    return float(min(gammas))


# =======================================================================
# 3. PSD 曲线 → 模型输入 的转换工具
# =======================================================================
def psd_density_to_lpm_components(d_um: np.ndarray, vol_pct: np.ndarray) -> list[LPMComponent]:
    """把"逐档体积密度分布"(体积% per bin，总和≈100，不是累积分布)转换成
    LPM 需要的 (d, y) 档列表。d_um/vol_pct 需等长、已按仪器原始分档对齐；
    vol_pct 为 0 的档会被跳过(不参与堆积计算，减少无意义的 O(n^2) 主导档枚举)。

    与 `data_processing/psd_processor.py::read_density_csv` 的输出口径一致
    (逐档体积密度，非累积)；本函数不做单位换算，也不做累积→密度的判别，
    调用方需先用 psd_processor 里的读取逻辑把仪器导出的原始曲线整理成这个
    形式(仪器导出实际是密度分布，见 psd_processor.py 里的说明与探测逻辑)。
    """
    d_um = np.asarray(d_um, dtype=float)
    vol_pct = np.asarray(vol_pct, dtype=float)
    if d_um.shape != vol_pct.shape:
        raise ValueError("d_um 和 vol_pct 长度必须一致")
    mask = np.isfinite(d_um) & np.isfinite(vol_pct) & (vol_pct > 0) & (d_um > 0)
    return [LPMComponent(d=float(d), y=float(v)) for d, v in zip(d_um[mask], vol_pct[mask])]
