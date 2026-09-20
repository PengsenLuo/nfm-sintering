# -*- coding: utf-8 -*-
"""19_packing_model_demo.py —— 用前驱体 PSD 验证堆积模型计算引擎(T9,2026-07-30)

编号说明:任务书原建议用 scripts/16_packing_model_demo.py,但 16 号已被
scripts/16_primary_particle_mnar.py(T6)占用,故改用当前(委派时确认)可用的
19 号。若执行时发现 19 号又被别的任务占了,请改下一个可用编号并在报告里注明。

做什么
------
本脚本**只验证 Furnas / LPM 两个堆积模型实现的正确性**,用的是前驱体(烧结前)
PSD 实测曲线——这是几何堆积计算,不是压实密度实测值,两者是不同物理量,
不能相互替代。**不产出可替换论文 §4.1 线性混合叙述的结论**——那需要产物
(免水)压实密度 + 产物 PSD 实测数据,目前仍是 [待测],本脚本不解除这个缺口。

步骤:
  1. 读取 data/raw/psd/precursors/precursor_{S,M,L}.csv,先打印真实解码后的列名
     字符串做人工核验(不硬编码列名匹配,复用 psd_processor.read_cumulative_csv/
     read_density_csv 的按列位置读取,那两个函数已实现"自动判别累积/密度分布"
     的逻辑,不重复造轮子)。
  2. 本地重算 D50,与 nfm.config.load_precursor_dvalues() 权威表交叉核对
     (S=3.67/M=9.70/L=12.40 µm,不重新硬编码这三个数,直接从权威表读)。
  3. 计算"若 M 前驱体是 S/L 按 0.7:0.3(大:小,来自 config precursor_desc.M.
     blend_ratio=0.3)机械混合"时:
       (a) Furnas 二元模型,用 D50(L)/D50(S) 作特征尺寸
       (b) LPM 两组分近似,同样只用 D50(与 Furnas 可比)
       (c) LPM 全 PSD 机械混合:S、L 的完整实测曲线按体积权重 0.3/0.7 直接合并
           成一组多档输入(这是 LPM 相对 Furnas 的核心优势——接受完整分布,
           不必先压成两个特征尺寸)
  4. 另外直接用 M 前驱体自身的实测 PSD 全曲线跑 LPM,得到 (d)。
  5. 把 (c) 与 (d) 做**定性**对照(不是统计检验):M 实测双峰位置
     [3.12, 11.2] µm、w_small≈0.331(见 data/interim/psd_precursor_features.csv,
     01b 脚本产出)与"7:3 机械混合"构造出的多档分布在数量级上是否吻合。

输出:
  data/interim/packing_model_precursor_demo.csv  —— 逐项数值结果
  终端打印一份人类可读摘要
"""
import _bootstrap  # noqa

from pathlib import Path

import numpy as np
import pandas as pd

from nfm.config import load_config, load_precursor_dvalues
from nfm.data_processing import psd_processor as pp
from nfm.packing_models import (
    LPMComponent,
    furnas_binary_packing_fraction,
    lpm_packing_fraction,
    psd_density_to_lpm_components,
)

PRECURSOR_DIR = Path("data/raw/psd/precursors")
OUT_CSV = Path("data/interim/packing_model_precursor_demo.csv")
BIMODAL_FEATURES_CSV = Path("data/interim/psd_precursor_features.csv")


def load_precursor_curve(letter: str):
    f = PRECURSOR_DIR / f"precursor_{letter}.csv"
    # 只做人工核验用的打印:实际解析走 psd_processor 的按列位置读取,不依赖
    # 列名字符串匹配(原始文件带 UTF-8 BOM,列名解码后是 "Size (µm)"/"Volume (%)",
    # 但在 GBK 终端下打印可能显示乱码——这不影响下面 iloc[:, :2] 的位置读取)。
    header_cols = list(pd.read_csv(f, encoding="utf-8-sig", nrows=0).columns)
    print(f"  {f.name} 解码后列名(utf-8-sig): {header_cols!r}  (仅供人工核验,不参与解析逻辑)")
    d, cum = pp.read_cumulative_csv(f)
    _, density = pp.read_density_csv(f)
    return d, cum, density


def main():
    cfg = load_config()
    d50_auth = load_precursor_dvalues(cfg)  # 权威 D50:{'S':3.67,'M':9.70,'L':12.40}

    print("=== 1. 读取三条前驱体 PSD 曲线 ===")
    curves = {}
    for letter in ("S", "M", "L"):
        curves[letter] = load_precursor_curve(letter)

    print("\n=== 2. D50 交叉核对(本地重算 vs 权威表 load_precursor_dvalues) ===")
    d50_local = {}
    for letter in ("S", "M", "L"):
        d, cum, _ = curves[letter]
        d50_local[letter] = pp.d_values(d, cum)[1]
        diff_pct = 100 * (d50_local[letter] - d50_auth[letter]) / d50_auth[letter]
        print(f"  {letter}: 权威表={d50_auth[letter]:.3f} µm  本地重算={d50_local[letter]:.3f} µm"
              f"  差异={diff_pct:+.1f}%")

    f_small = cfg.raw["design"]["precursor_desc"]["M"]["blend_ratio"]
    f_large = 1.0 - f_small
    print(f"\n=== 3. M 前驱体配方比例:大:小 = {f_large:.1f}:{f_small:.1f}"
          f" (blend_ratio={f_small}, 来自 config precursor_desc.M) ===")

    # (a) Furnas:两个特征尺寸(权威 D50)
    furnas_pred = furnas_binary_packing_fraction(
        d_large=d50_auth["L"], d_small=d50_auth["S"], f_small=f_small)

    # (b) LPM 两组分近似(同样只用 D50,与 Furnas 直接可比)
    lpm_pred_2class = lpm_packing_fraction(
        [LPMComponent(d=d50_auth["L"], y=f_large), LPMComponent(d=d50_auth["S"], y=f_small)])

    # (c) LPM 全 PSD 机械混合:S/L 完整曲线按体积权重合并(LPM 相对 Furnas 的优势)
    dS, _, densS = curves["S"]
    dL, _, densL = curves["L"]
    comps_mix = (psd_density_to_lpm_components(dS, densS * f_small)
                 + psd_density_to_lpm_components(dL, densL * f_large))
    lpm_pred_full_mix = lpm_packing_fraction(comps_mix)

    # (d) LPM 直接用 M 自身实测 PSD 全曲线
    dM, _, densM = curves["M"]
    comps_M = psd_density_to_lpm_components(dM, densM)
    lpm_pred_M_actual = lpm_packing_fraction(comps_M)

    print("\n=== 4. 堆积分数预测(几何堆积计算量,不是压实密度实测值) ===")
    print(f"  (a) Furnas, D50(L)/D50(S), f_small={f_small}       -> phi = {furnas_pred:.4f}")
    print(f"  (b) LPM 两组分(D50 近似,同 Furnas 可比)            -> phi = {lpm_pred_2class:.4f}")
    print(f"  (c) LPM 全 PSD 机械混合(S 0.3 + L 0.7 完整曲线)     -> phi = {lpm_pred_full_mix:.4f}")
    print(f"  (d) LPM 直接用 M 自身实测 PSD 全曲线                -> phi = {lpm_pred_M_actual:.4f}")

    print("\n=== 5. 与 M 前驱体实测双峰形状的定性对照(非统计检验) ===")
    if BIMODAL_FEATURES_CSV.exists():
        bf = pd.read_csv(BIMODAL_FEATURES_CSV).set_index("precursor")
        m_row = bf.loc["M"]
        print(f"  M 实测双峰位置(01b 脚本, data/interim/psd_precursor_features.csv): "
              f"{m_row['mode_positions_um']}, w_small={m_row['w_small']:.3f}")
    else:
        print(f"  [警告] {BIMODAL_FEATURES_CSV} 不存在(需先跑 01b 脚本),跳过双峰形状对照")
        m_row = None
    print(f"  S/L 权威 D50 = {d50_auth['S']:.2f}/{d50_auth['L']:.2f} µm,"
          f" 与 M 实测双峰位置在数量级上一致(定性观察,不是统计结论)。")
    print("  注意:这里比较的是 PSD 曲线形状/峰位,不是堆积分数数值本身——"
          "(c)(d) 两个堆积分数之间的差异反映的是'把 M 当 7:3 机械混合重建'与"
          "'M 自身实测曲线'两种几何输入的差异,不涉及任何压实密度实测值。")

    out = pd.DataFrame([
        {"item": "d50_auth_S_um", "value": d50_auth["S"]},
        {"item": "d50_auth_M_um", "value": d50_auth["M"]},
        {"item": "d50_auth_L_um", "value": d50_auth["L"]},
        {"item": "d50_local_S_um", "value": d50_local["S"]},
        {"item": "d50_local_M_um", "value": d50_local["M"]},
        {"item": "d50_local_L_um", "value": d50_local["L"]},
        {"item": "f_small_blend_ratio_M", "value": f_small},
        {"item": "furnas_pred_phi_D50_2class", "value": furnas_pred},
        {"item": "lpm_pred_phi_D50_2class", "value": lpm_pred_2class},
        {"item": "lpm_pred_phi_full_psd_mechanical_mix", "value": lpm_pred_full_mix},
        {"item": "lpm_pred_phi_M_actual_full_psd", "value": lpm_pred_M_actual},
    ])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\n结果已写入 {OUT_CSV}")

    print("\n" + "=" * 70)
    print("重要:本脚本只验证模型实现本身的正确性(前驱体几何堆积计算),")
    print("不能、也不会用来替换论文 §4.1 的错误线性混合叙述。那一步需要产物")
    print("(免水)压实密度 + 产物 PSD 实测数据,目前仍是 [待测],本任务不解除这个缺口。")
    print("=" * 70)


if __name__ == "__main__":
    main()
