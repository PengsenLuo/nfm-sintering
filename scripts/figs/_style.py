# -*- coding: utf-8 -*-
"""
_style.py —— 图表工厂统一样式模块
====================================
所有 figs/fig_*.py 从这里拿配色、marker、字号,并用 save_fig() 统一出图
(300dpi PNG + 矢量 PDF)、统一回填 outputs/figs/captions.md。

不读数据、不做任何统计计算——纯样式 + I/O 帮手。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
MASTER_TABLE = REPO_ROOT / "data" / "processed" / "master_table.csv"
# 图片输出目录统一为 reports/figures/(不再有 outputs/figs/ 与
# reports/figures/ 两套并存的约定)。
FIGS_DIR = REPO_ROOT / "reports" / "figures"
CAPTIONS_MD = FIGS_DIR / "captions.md"
CAPTIONS_JSON = FIGS_DIR / "_captions.json"

# ---------------------------------------------------------------------
# 配色 / marker(前驱体、温度)
# ---------------------------------------------------------------------
PRECURSOR_COLOR = {"S": "#e63946", "M": "#457b9d", "L": "#2a9d8f"}
PRECURSOR_MARKER = {"S": "o", "M": "s", "L": "^"}
PRECURSOR_LABEL = {"S": "S (D50≈3µm)", "M": "M (7:3混合)", "L": "L (D50≈11µm)"}

# 温度三档:蓝(低) / 橙(中) / 红(高)
TEMP_COLOR = {850: "#1d4e89", 900: "#f4a259", 950: "#c1121f"}
TEMP_MARKER = {850: "o", 900: "s", 950: "^"}

# 图注里统一说明 Θ 的标定状态,已定稿:
# config.yaml.thermal_exposure.integrate_from_C 固定为 550°C(与预烧终温一致,
# 该 550°C/5h 预烧对全部样本相同),Θ 只积分预烧之后的升温段+保温段。
# 原 "auto"(双端点反推起算温度)标定方式已弃用,不再是当前口径。
THETA_NOTE = (
    "Θ 起算温度固定为 550°C(与预烧终温一致,2026-07-25 定稿,"
    "config.yaml.thermal_exposure.integrate_from_C=550),不再是 auto 标定。"
)
# 向后兼容旧名(若外部脚本仍引用 THETA_AUTO_NOTE,给出同样准确的文本,
# 不再传播过期的"auto/待定"表述)。
THETA_AUTO_NOTE = THETA_NOTE

# 已知图名的期望出场顺序(用于 captions.md 排版;未知名字追加到末尾)
FIGURE_ORDER = [
    "fig_dxrd_vs_theta", "fig_dxrd_vs_theta_byT",
    "fig_grain_growth_arrhenius",
    "fig_lattice_vs_theta", "fig_lattice_vs_theta_byT",
    "fig_md_vs_theta", "fig_md_vs_theta_byT",
    "fig_density_vs_theta", "fig_density_vs_theta_byT",
    "fig_density_matrix",
    "fig_doe_grid",
]

# ---------------------------------------------------------------------
# rcParams:字号、线宽、marker 尺寸统一
# ---------------------------------------------------------------------
def apply_style() -> None:
    matplotlib.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        # 中文字体候选(标题/图注若含中文时生效;数据类图本身尽量用英文标签)
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })


apply_style()


# ---------------------------------------------------------------------
# save_fig:双份出图 + 回填 captions.md
# ---------------------------------------------------------------------
def save_fig(fig, name: str, caption: str) -> None:
    """保存 fig 为 outputs/figs/{name}.png(300dpi)与 .pdf(矢量),并把
    caption 写入 outputs/figs/captions.md 对应小节(幂等,重跑会覆盖同名小节)。
    """
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGS_DIR / f"{name}.png"
    pdf_path = FIGS_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    _update_caption(name, caption)
    print(f"[save_fig] {name}: {png_path.name} + {pdf_path.name}")


def _update_caption(name: str, caption: str) -> None:
    entries = {}
    if CAPTIONS_JSON.exists():
        entries = json.loads(CAPTIONS_JSON.read_text(encoding="utf-8"))
    entries[name] = caption
    CAPTIONS_JSON.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")

    ordered_names = [n for n in FIGURE_ORDER if n in entries]
    ordered_names += [n for n in entries if n not in FIGURE_ORDER]

    lines = ["# 图表说明(captions)\n",
             "> 由 scripts/figs/*.py 通过 save_fig() 自动生成/更新,勿手改;"
             "改图注请改对应 fig_*.py 里的 caption 文本。\n"]
    for n in ordered_names:
        lines.append(f"\n## {n}\n\n{entries[n]}\n")
    CAPTIONS_MD.write_text("\n".join(lines), encoding="utf-8")


def skip(name: str, reason: str) -> None:
    """数据不足时调用,打印原因、不出图、不写 captions.md 条目。"""
    print(f"[skip] {name}: {reason}")
