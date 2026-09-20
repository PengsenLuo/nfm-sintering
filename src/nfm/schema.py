# -*- coding: utf-8 -*-
"""
schema.py —— 全链路唯一事实源(v0.2)
=====================================
每列一个 ColumnSpec(name/dtype/unit/role/layer/source/bounds/description)。
role 固化"铁律":凡 TARGET 列(烧结后测得)不得进入 X,除非经 two_step 视图。

v0.2 相对 v0.1 的列级改动:
  - crystallite_size  → D_XRD            (语义明确为相干衍射畴尺寸)
  - 新增 D_XRD_WH / microstrain_WH       (Williamson–Hall 稳健性校验,派生)
  - agglomeration_index → hier_size_ratio(层级结构尺度比;去掉"一次颗粒数"误读)
  - 新增双峰特征列 B_bimodal             (来自 psd 双对数正态混合拟合)
  - 新增同炉差分量 M_D / M_B / dRho_M    (integrate 自动生成,按 condition_id 分组)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Role(Enum):
    ID = "id"
    INPUT_PROCESS = "input_process"        # 烧结前已知的工艺参数
    INPUT_PRECURSOR = "input_precursor"    # 前驱体描述符(烧结前已知)
    INPUT_DERIVED = "input_derived"        # 由工艺纯计算得到(烧结前可算)
    TARGET = "target"                      # 烧结后测得
    DERIVED_DIFF = "derived_diff"          # 同炉差分派生量(分析量,非建模 X/y 默认成员)
    QC = "qc"                              # 质控标志(烧结后测得,但非产物性能目标;
                                            # 不应进入 target_cols() 的建模目标候选,
                                            # 也非 DERIVED_DIFF 的同炉差分语义)


class Layer(Enum):
    NONE = "none"
    MORPHOLOGY = "morphology"
    STRUCTURE = "structure"
    PACKING = "packing"
    MEMORY = "memory"


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str                 # "str" | "int" | "float"
    unit: str
    role: Role
    layer: Layer = Layer.NONE
    source: str = ""           # design|supplier|derived|psd|sem_*|xrd|density|diff
    bounds: tuple | None = None
    description: str = ""


SCHEMA: list[ColumnSpec] = [
    # ---------- 标识 ----------
    ColumnSpec("sample_id", "str", "-", Role.ID, source="design",
               description="样本编号,如 C14-M"),
    ColumnSpec("condition_id", "str", "-", Role.ID, source="design",
               description="同炉条件号 C01–C27(同一炉次共享热历史,分组键)"),
    ColumnSpec("precursor", "str", "-", Role.ID, source="design",
               description="前驱体类型 S/M/L"),
    ColumnSpec("crucible_pos", "int", "-", Role.ID, source="design",
               bounds=(1, 3), description="炉内坩埚位置(区组协变量)"),

    # ---------- 输入:工艺参数 ----------
    ColumnSpec("T_C", "float", "degC", Role.INPUT_PROCESS, source="design",
               bounds=(850, 950), description="烧结温度"),
    ColumnSpec("beta", "float", "degC/min", Role.INPUT_PROCESS, source="design",
               bounds=(2, 8), description="升温速率(作用于 550→T 段)"),
    ColumnSpec("t_hold", "float", "h", Role.INPUT_PROCESS, source="design",
               bounds=(10, 20), description="保温时间"),

    # ---------- 输入:前驱体描述符 ----------
    ColumnSpec("precursor_D50", "float", "um", Role.INPUT_PRECURSOR, source="supplier",
               bounds=(2, 13),
               description="前驱体 D50(M 用实测混合 D50,无则质量加权)。上界从 12 "
                           "上调到 13:L 前驱体仪器实测 D50=12.4µm 为真实测量值"
                           "(psd_dvalues_precursors.csv),原上界与实测值冲突,已改正。"),
    ColumnSpec("bimodal", "int", "-", Role.INPUT_PRECURSOR, source="supplier",
               bounds=(0, 1), description="双峰标志(M=1,S/L=0)"),
    ColumnSpec("blend_ratio", "float", "-", Role.INPUT_PRECURSOR, source="supplier",
               bounds=(0, 1), description="小颗粒质量分数(M=0.3,S=1,L=0)"),

    # ---------- 输入:物理派生特征(features 写入) ----------
    ColumnSpec("T_K", "float", "K", Role.INPUT_DERIVED, source="derived",
               bounds=(1100, 1250), description="热力学温度"),
    ColumnSpec("inv_T_K", "float", "1/K", Role.INPUT_DERIVED, source="derived",
               description="1/T_K,Arrhenius 线性化坐标"),
    ColumnSpec("Theta", "float", "h", Role.INPUT_DERIVED, source="derived",
               bounds=(0, 200),
               description="归一化热暴露量(Arrhenius 加权积分,Q_ref;X.2.4)"),
    ColumnSpec("t_eff", "float", "h", Role.INPUT_DERIVED, source="derived",
               description="有效时间(升温段 Arrhenius 折算,Θ 计算的中间量)"),

    # ---------- 目标:形貌(激光粒度) ----------
    ColumnSpec("D10", "float", "um", Role.TARGET, Layer.MORPHOLOGY, "psd",
               bounds=(0.1, 30),
               description="体积分布 D10(下界 2026-07 第十轮下调:11 个越界样本经"
                           "仪器报告值证实为真实测量,最低见 0.286µm)"),
    ColumnSpec("D50", "float", "um", Role.TARGET, Layer.MORPHOLOGY, "psd",
               bounds=(1, 40), description="体积分布 D50 ★主目标"),
    ColumnSpec("D90", "float", "um", Role.TARGET, Layer.MORPHOLOGY, "psd",
               bounds=(2, 80), description="体积分布 D90"),
    ColumnSpec("Span", "float", "-", Role.TARGET, Layer.MORPHOLOGY, "psd",
               bounds=(0.1, 5), description="(D90-D10)/D50 ★主目标"),
    ColumnSpec("B_bimodal", "float", "-", Role.TARGET, Layer.MORPHOLOGY, "psd",
               bounds=(0, 5),
               description="双峰特征量(双对数正态混合:分离度×小峰权重等综合量)"),

    # ---------- 目标:形貌(SEM 二次颗粒 @500×) ----------
    # 定位:二次颗粒口径是激光粒度(psd)的独立原位交叉验证 + 形状重构指标,
    #       非主粒径来源(主粒径 D50 仍由 psd 给)。reactive sintering 二次颗粒
    #       为不规则碎块,以下均为投影口径,绝对值不与激光 D50 直接相等,比趋势/M_D。
    ColumnSpec("D_sec", "float", "um", Role.TARGET, Layer.MORPHOLOGY, "sem_500x",
               bounds=(1, 50),
               description="二次颗粒等效圆投影直径中位数(500×;激光 D50 的独立原位交叉验证)"),
    ColumnSpec("aspect_ratio", "float", "-", Role.TARGET, Layer.MORPHOLOGY, "sem_500x",
               bounds=(1, 5), description="二次颗粒长径比中位数(major/minor,投影)"),
    ColumnSpec("circularity", "float", "-", Role.TARGET, Layer.MORPHOLOGY, "sem_500x",
               bounds=(0, 1),
               description="4πA/P² 中位数 ★主目标(随 Θ 下降→棱角化,热重构形貌指标)"),
    ColumnSpec("convexity", "float", "-", Role.TARGET, Layer.MORPHOLOGY, "sem_500x",
               bounds=(0, 1), description="A/A_convex 中位数(凸度;棱角化随 Θ 下降)"),

    # ---------- 目标:形貌(SEM 一次颗粒 @5000×) ----------
    ColumnSpec("D_pri_sem", "float", "um", Role.TARGET, Layer.MORPHOLOGY, "sem_5kx",
               bounds=(0.05, 5),
               description="一次颗粒等效圆直径中位数(5000× 投影口径,全样本统一倍率;每样 N 见处理器 n_pri)"),

    # ---------- 目标:形貌(派生,integrate 生成) ----------
    ColumnSpec("hier_size_ratio", "float", "-", Role.TARGET, Layer.MORPHOLOGY, "derived",
               bounds=(1, 200),
               description="层级结构尺度比 = D50(激光)/D_pri_sem(非一次颗粒数!)"),

    # ---------- 目标:结构(XRD) ----------
    ColumnSpec("D_XRD", "float", "nm", Role.TARGET, Layer.STRUCTURE, "xrd",
               bounds=(20, 500),
               description="相干衍射畴尺寸(Scherrer,扣 Si 仪器宽化;≠一次颗粒)"),
    ColumnSpec("D_XRD_003", "float", "nm", Role.TARGET, Layer.STRUCTURE, "xrd",
               bounds=(20, 500),
               description="D_XRD 的 (003) 单峰 Scherrer 分量(诊断用,2026-08-15 新增;"
                           "与 D_XRD 计算方式/数值不冲突,D_XRD 仍为 (003)/(104) 平均)"),
    ColumnSpec("D_XRD_104", "float", "nm", Role.TARGET, Layer.STRUCTURE, "xrd",
               bounds=(20, 500),
               description="D_XRD 的 (104) 单峰 Scherrer 分量(诊断用,2026-08-15 新增;"
                           "与 D_XRD 计算方式/数值不冲突,D_XRD 仍为 (003)/(104) 平均)"),
    ColumnSpec("D_XRD_WH", "float", "nm", Role.TARGET, Layer.STRUCTURE, "xrd",
               bounds=(20, 500),
               description="Williamson–Hall 尺寸(稳健性校验,派生于多峰)"),
    ColumnSpec("microstrain_WH", "float", "-", Role.TARGET, Layer.STRUCTURE, "xrd",
               bounds=(0, 0.02), description="Williamson–Hall 微应变 ε"),
    ColumnSpec("lattice_a", "float", "angstrom", Role.TARGET, Layer.STRUCTURE, "xrd",
               bounds=(2.9, 3.1), description="O3 相 R-3m 晶格参数 a"),
    ColumnSpec("lattice_c", "float", "angstrom", Role.TARGET, Layer.STRUCTURE, "xrd",
               bounds=(15.8, 16.4), description="O3 相 R-3m 晶格参数 c"),
    ColumnSpec("c_a_ratio", "float", "-", Role.TARGET, Layer.STRUCTURE, "xrd",
               bounds=(5.2, 5.6), description="c/a 比"),

    # ---------- 质控(QC,非建模目标) ----------
    ColumnSpec("hydrate_index", "float", "%", Role.QC, Layer.STRUCTURE, "xrd",
               bounds=(0, 100),
               description="水合/残碱相污染质控标志(2026-08-05 新增),= 100×A_hyd/A_003:"
                           "A_hyd 为 13.8–15.2°(层间膨胀相,d≈6.14Å)扣线性基线后的积分面积,"
                           "A_003 为 16.0–17.2°((003)主峰)扣水平基线后的积分面积。度量样品"
                           "吸潮/吸CO2导致的层间膨胀相污染程度,不是产物性能目标,不得作为"
                           "建模 TARGET 使用;高值提示 D_XRD/lattice_c/lattice_a/c_a_ratio "
                           "可能受污染,应结合 hydrate_index_warn_threshold(config.yaml)判断。"),

    # ---------- 目标:堆积(核心) ----------
    ColumnSpec("compaction_density", "float", "g/cm3", Role.TARGET, Layer.PACKING, "density",
               bounds=(2.0, 4.0), description="单点压实密度 ★主目标,核心卖点"),

    # ---------- 同炉差分派生量(integrate 生成,分析用) ----------
    ColumnSpec("M_D", "float", "-", Role.DERIVED_DIFF, Layer.MEMORY, "diff",
               bounds=(-1, 2),
               description="粒径记忆系数(L−S 产物差/前驱体差,同炉;X.5)。"
                           "产物侧用激光 PSD D50(水测口径),作分散单元佐证,非主指标——"
                           "主指标见 M_D_agg。"),
    ColumnSpec("M_D_agg", "float", "-", Role.DERIVED_DIFF, Layer.MEMORY, "diff",
               bounds=(-1, 3),
               description="免水团聚体记忆系数(同炉;记忆主指标)= "
                           "(D_sec_L − D_sec_S)/(precursor_L_D50 − precursor_S_D50)。"
                           "分子为 SEM 500× 团聚体 D_sec(干态免水),分母为前驱体激光 D50"
                           "(前驱体不与水反应,可信)。与 M_D 分子口径不同,不可互相替代。"),
    ColumnSpec("M_B", "float", "-", Role.DERIVED_DIFF, Layer.MEMORY, "diff",
               bounds=(0, 1.5),
               description="双峰保留指数 = B_prod / B_prec(仅 M 组定义)"),
    ColumnSpec("dRho_M", "float", "g/cm3", Role.DERIVED_DIFF, Layer.MEMORY, "diff",
               bounds=(-1, 1),
               description="双峰压实净增益 = ρ_M − (w_S ρ_S + w_L ρ_L)(同炉,仅 M 组)"),
]

# ---------------------------------------------------------------------
# 索引与选择器
# ---------------------------------------------------------------------
BY_NAME: dict[str, ColumnSpec] = {c.name: c for c in SCHEMA}
assert len(BY_NAME) == len(SCHEMA), "schema 存在重复列名"


def cols(role: Role | None = None, layer: Layer | None = None) -> list[str]:
    out = SCHEMA
    if role is not None:
        out = [c for c in out if c.role is role]
    if layer is not None:
        out = [c for c in out if c.layer is layer]
    return [c.name for c in out]


def input_cols() -> list[str]:
    roles = {Role.INPUT_PROCESS, Role.INPUT_PRECURSOR, Role.INPUT_DERIVED}
    return [c.name for c in SCHEMA if c.role in roles]


def target_cols(layer: Layer | None = None) -> list[str]:
    out = [c for c in SCHEMA if c.role is Role.TARGET]
    if layer is not None:
        out = [c for c in out if c.layer is layer]
    return [c.name for c in out]


def diff_cols() -> list[str]:
    return [c.name for c in SCHEMA if c.role is Role.DERIVED_DIFF]


# ---------------------------------------------------------------------
# 视图校验(铁律的代码化)
# ---------------------------------------------------------------------
def validate_view(name: str, view: dict) -> None:
    """除 two_step 外,X 中出现 TARGET 列即抛错;two_step 还需 y 不与 X 同层。"""
    x_cols = list(view.get("X", []))
    y_cols = list(view.get("y", []))
    tgt = set(target_cols())
    leaked = [c for c in x_cols if c in tgt]
    if name != "view_two_step":
        if leaked:
            raise ValueError(
                f"视图 {name} 违反铁律:X 中含烧结后测得的目标列 {leaked}")
    else:
        # two_step:X 允许是形貌结构目标,但 y 不能与 X 属于同一被预测目标
        same = set(x_cols) & set(y_cols)
        if same:
            raise ValueError(f"视图 {name} 中 X 与 y 重叠:{same}(用答案预测答案)")
    return None


def validate_master_table(df) -> "ValidationReport":
    """列齐全、dtype、bounds、81 行、sample_id 唯一、缺失报告;不过不落盘。"""
    import pandas as pd  # 局部导入,避免顶层硬依赖

    problems: list[str] = []
    # 必须存在的非派生列(差分量与 WH 派生量允许缺失)
    must = [c.name for c in SCHEMA
            if c.role not in (Role.DERIVED_DIFF,) and c.source != "derived"]
    missing_cols = [c for c in must if c not in df.columns]
    if missing_cols:
        problems.append(f"缺列: {missing_cols}")
    if "sample_id" in df.columns:
        if df["sample_id"].duplicated().any():
            problems.append("sample_id 存在重复")
        if len(df) != 81:
            problems.append(f"行数为 {len(df)},应为 81")
    # bounds 检查
    for c in SCHEMA:
        if c.bounds and c.name in df.columns:
            lo, hi = c.bounds
            s = pd.to_numeric(df[c.name], errors="coerce")
            bad = ((s < lo) | (s > hi)).sum()
            if bad:
                problems.append(f"{c.name}: {bad} 个值越界 [{lo},{hi}]")
    return ValidationReport(ok=(len(problems) == 0), problems=problems)


@dataclass
class ValidationReport:
    ok: bool
    problems: list

    def raise_if_failed(self):
        if not self.ok:
            raise ValueError("master_table 校验失败:\n  - " +
                             "\n  - ".join(self.problems))
