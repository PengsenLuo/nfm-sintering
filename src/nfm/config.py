# -*- coding: utf-8 -*-
"""
config.py —— 配置加载 + 条件表 + 样本索引 (v0.2)
================================================
启动即校验所有视图(铁律前置),并对未回填的关键元数据持续提醒。
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from nfm import schema

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


@dataclass
class Config:
    raw: dict
    root: Path = field(default=Path("."))

    # —— 便捷访问 ——
    @property
    def Q_ref_J(self) -> float:
        return self.raw["physical_constants"]["Q_ref_kJmol"] * 1e3

    @property
    def R(self) -> float:
        return self.raw["physical_constants"]["R_J_per_molK"]

    @property
    def T_ref_K(self) -> float:
        return self.raw["physical_constants"]["T_ref_C"] + 273.15

    def view(self, name: str) -> dict:
        return self.raw["views"][name]

    def primary_targets(self) -> list[str]:
        return self.raw["primary_targets"]


def build_condition_map(cfg: Config):
    """由全因子设计生成 C01–C27 ↔ (T,β,t)。"""
    import pandas as pd

    f = cfg.raw["design"]["factors"]
    rows, i = [], 1
    for T in f["T_C"]:
        for b in f["beta"]:
            for t in f["t_hold"]:
                rows.append({"condition_id": f"C{i:02d}",
                             "T_C": float(T), "beta": float(b), "t_hold": float(t)})
                i += 1
    return pd.DataFrame(rows)


def load_precursor_dvalues(cfg: Config, csv_path: str | Path | None = None) -> dict[str, float]:
    """前驱体 D50 权威来源:仪器报告表(第十轮起,替代 config.yaml 里的标称值)。

    文件以 precursor_S/precursor_M/precursor_L 为 sample_id,取 D50 列;不设硬编码兜底。
    """
    import pandas as pd

    if csv_path is None:
        csv_path = cfg.raw["instruments"]["laser_psd"]["precursor_dvalues_csv"]
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(
            f"前驱体 D50 权威表缺失: {p}(precursor_D50 已改为仅从此文件读取,无标称值兜底)")
    df = pd.read_csv(p).set_index("sample_id")
    out = {}
    for letter in ("S", "M", "L"):
        key = f"precursor_{letter}"
        if key not in df.index:
            raise ValueError(f"{p} 缺少 {key} 行")
        out[letter] = float(df.loc[key, "D50"])
    return out


def build_sample_index(cfg: Config):
    """27 条件 × 3 前驱体 = 81 行的样本索引(含前驱体描述符)。

    precursor_D50 取自仪器实测权威表(见 load_precursor_dvalues),第十轮起不再用
    config.yaml 的标称值;这同时是 M_D 记忆系数分母的正式来源。
    """
    import pandas as pd

    cm = build_condition_map(cfg)
    desc = cfg.raw["design"]["precursor_desc"]
    d50 = load_precursor_dvalues(cfg)
    rows = []
    for _, r in cm.iterrows():
        for p in cfg.raw["design"]["precursors"]:
            d = desc[p]
            rows.append({
                "sample_id": f"{r.condition_id}-{p}",
                "condition_id": r.condition_id,
                "precursor": p,
                "T_C": r.T_C, "beta": r.beta, "t_hold": r.t_hold,
                "precursor_D50": d50[p],
                "bimodal": d["bimodal"],
                "blend_ratio": d["blend_ratio"],
            })
    return pd.DataFrame(rows)


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = Config(raw=raw, root=p.resolve().parents[1])

    # 启动即校验所有视图(铁律前置)
    for name, v in raw["views"].items():
        schema.validate_view(name, v)

    # 压实密度测试条件未回填的持续提醒
    cd = raw["instruments"]["compaction_density"]
    missing = [k for k in ("pressure_MPa", "die_diameter_mm", "dwell_s")
               if cd.get(k) is None]
    if missing:
        warnings.warn(f"压实密度测试条件未回填: {missing}(论文方法部分提交前必须补)")

    # Θ 起算温度提醒
    if raw["thermal_exposure"]["integrate_from_C"] == "auto":
        warnings.warn("Θ 起算温度设为 auto:将由双端点预实验反推,出图前请确认已标定")
    return cfg
