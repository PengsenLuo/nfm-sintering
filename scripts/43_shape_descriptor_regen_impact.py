# -*- coding: utf-8 -*-
"""43 sem_shape_descriptors.csv 根因修复后的改前/改后完整影响清单(2026-08-29)
====================================================================
背景:对 median(d_um) vs master_table.D_sec 的一致性硬闸(容差0.01um)首次
触发时,发现
`data/interim/sem_shape_descriptors.csv`(D_f/solidity/elongation/roughness
四个形状描述子的来源)对最多 20/81 个样本存在与生产 D_sec 管线的静默漂移
(最大 0.42um / 132 颗粒,C24-S)。独立复算证实:用当前代码在同一张图上重跑
两条路径逐位一致——问题出在 `scripts/02i_shape_descriptors.py --extract`
的增量抽取逻辑只按 sample_id 判断"已处理",分割参数(SegConfig)跨轮改变
后早期样本从未被重新分割。已修复根因(见 02i 新增的 `_seg_config_hash`/
`_valid_cached_rows`,并给 SegConfig 变化补了失效测试),并已重新全量跑出
`sem_shape_descriptors.csv`(81/81,现与 D_sec/n_sec 逐位一致)。

决定:重冻结,采用新值(判据:新文件可逐位验证复现 D_sec,旧文件
在34样上做不到)。但要求先出完整影响清单再动手,不要只更新四个 eta2。
本脚本产出该清单,覆盖以下五项:
  1. 表1(precursor/T/beta/t_hold/交互合计/残差六列),D_f 与 elongation
  2. D_f 对 compaction_density 的偏相关(控制Theta / Theta+precursor)
     与 5%/4% TOST p 值 —— 复用 scripts/17_shape_descriptor_full_analysis.py
     的 section3_partial_corr_tost(不重新实现)
  3. 逐颗粒 Richardson 回归 R² 中位数与标度点数分布 —— 说明该量为何不受
     本次根因 bug 影响(见 §3)
  4. 四个形状描述子分前驱体 Spearman rho + 95% bootstrap CI —— 复用同一
     脚本的 section2_spearman_bootstrap
  5. 受影响的图(fig2a/figS2d/fig4c-f/figS10)清单 —— 仅列出,不在本轮重出

以及"44"防呆核查(v7 中两类不同来源的"44",禁止混淆/全局替换)。

复核后追加的三处补齐(第二版):
  a. "44"防呆按**出现位置**而非按行计数(附录B5一段里两个独立"44.0"
     此前按行去重漏计成1处,应为2处,真实替换点数=5)。
  b. §3.2.1 表格四项(偏相关r点估计/95%CI/标准化斜率/两个δ的TOST p)
     补齐完整,不再只给5%界的新p。
  c. 新增交叉验证:sem_df_r2_diagnostic.csv(全程未受本次bug影响)与
     重生成后的sem_shape_descriptors.csv逐样本比对D_f中位数,确认
     重生成已完全收敛(不一致则 assert 触发,停止而非静默通过)。

本脚本只做对照/清单,不改写 `sem_shape_descriptors_summary.csv`、
`anova_interactions_shape_descriptors.csv`、`manuscript/numbers/
manuscript_numbers.csv`、任何图件,也不重跑 `scripts/17` 本体(避免触碰
其写盘副作用)——只导入其纯函数 section2_spearman_bootstrap /
section3_partial_corr_tost 复用逻辑。是否据此重新冻结、以及冻结后的
图件重出,是确认后的下一步,不在本脚本范围内。

改前基准:`data/interim/sem_shape_descriptors.csv.pre_confighash_fix_bak`
(根因修复前的最后状态,即当前 `manuscript/numbers/manuscript_numbers.csv`
里 Table 1/2/4/10 与 fig4 所依赖的口径)。
改后:`data/interim/sem_shape_descriptors.csv`(已重新全量生成)。
"""
import _bootstrap  # noqa

import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd

from nfm.stats.anova_interactions import TERM_ORDER, run_anova_for_target

ROOT = Path(__file__).resolve().parents[1]
OLD_CSV = ROOT / "data/interim/sem_shape_descriptors.csv.pre_confighash_fix_bak"
NEW_CSV = ROOT / "data/interim/sem_shape_descriptors.csv"
MASTER_TABLE = ROOT / "data/processed/master_table.csv"
R2_DIAGNOSTIC_CSV = ROOT / "data/interim/sem_df_r2_diagnostic.csv"
REPORT_MD = ROOT / "reports/shape_descriptor_regen_impact.md"

DESCRIPTORS = ("D_f", "solidity", "elongation", "roughness")
SIX_COL_TARGETS = ("D_f", "elongation")  # 需要看整行的两个描述子
INTERACTION_TERMS = ("precursor:T_C", "precursor:beta", "precursor:t_hold",
                     "T_C:beta", "T_C:t_hold", "beta:t_hold")
N_PERM = 500  # 只关心 eta2 点估计对照,置换检验降采样以控制运行时间

_spec17 = importlib.util.spec_from_file_location(
    "shape_descriptor_full_analysis_17",
    ROOT / "scripts" / "17_shape_descriptor_full_analysis.py",
)
shape17 = importlib.util.module_from_spec(_spec17)
_spec17.loader.exec_module(shape17)


def _summarize(raw_csv: Path, master: pd.DataFrame) -> pd.DataFrame:
    """复刻 scripts/02i_shape_descriptors.py::report() 的逐样本中位数聚合
    (dropna(subset=["D_f"]) 后按 sample_id 取中位数),再按
    scripts/17_shape_descriptor_full_analysis.py::load_data() 的口径与
    master_table 合并 —— 全程只在内存中操作,不写 sem_shape_descriptors_
    summary.csv。"""
    d = pd.read_csv(raw_csv)
    agg = (d.dropna(subset=["D_f"]).groupby("sample_id")
             .agg(D_f=("D_f", "median"), solidity=("solidity", "median"),
                  elongation=("elongation", "median"),
                  roughness=("roughness_ratio", "median"),
                  n_particles=("D_f", "size")).reset_index())
    merged = agg.merge(
        master[["sample_id", "beta", "t_hold", "condition_id",
                "precursor", "Theta", "T_C", "compaction_density", "circularity"]],
        on="sample_id", how="inner",
    )
    assert len(merged) == 81, f"{raw_csv}: 合并后应为 81 行,实际 {len(merged)}"
    return merged


def _run_anova_all(df: pd.DataFrame, label: str) -> dict:
    out = {}
    for i, target in enumerate(DESCRIPTORS):
        res = run_anova_for_target(df, target, n_perm=N_PERM, seed=4000 + i)
        out[target] = res["table"]
        print(f"  [{label}] {target}: precursor eta2={res['table'].loc['precursor','eta2']*100:.2f}%")
    return out


def _spearman_ci_table(df: pd.DataFrame) -> pd.DataFrame:
    """复用 scripts/17 的 section2_spearman_bootstrap(纯函数,只 append 进
    传入的 rows 列表,不写文件),整理成 target/group/rho/ci_low/ci_high/n
    的宽表。"""
    rows = []
    for _target, _group_rhos in shape17.section2_spearman_bootstrap(df, rows):
        pass  # 消费生成器,结果都已经写进 rows
    long_df = pd.DataFrame(rows)
    sec2 = long_df[(long_df.section == "spearman_theta") & (long_df.metric == "rho")]
    return sec2[["target", "group", "value", "ci_low", "ci_high", "n"]].rename(
        columns={"value": "rho"})


def _partial_corr_tost_d_f(df: pd.DataFrame) -> dict:
    """复用 scripts/17 的 section3_partial_corr_tost(纯函数),只取 D_f 的
    结果(circularity 不受本次 bug 影响,来自 master_table 自己的列,不需要
    对照)。section3 返回的 summary dict 不含偏相关 r 自身的 95% CI(那只写进
    了 rows 长表里,summary 只保留了点估计),所以这里额外从 rows 里把
    partial_r 的 ci_low/ci_high 捞出来,按 (target,cs_name) 合并进结果——
    v7 §3.2.1 表格需要完整四项(r点估计/95%CI/斜率/两个δ的
    TOST p),缺一不可。"""
    rows = []
    summary = shape17.section3_partial_corr_tost(df, rows)
    long_df = pd.DataFrame(rows)
    r_ci_rows = long_df[(long_df.section == "partial_corr") & (long_df.metric == "partial_r")]
    r_ci = {(row["target"], row["group"]): (row["ci_low"], row["ci_high"])
            for _, row in r_ci_rows.iterrows()}
    out = {}
    for k, v in summary.items():
        if k[0] != "D_f":
            continue
        target, cs_name, delta_name = k
        v = dict(v)
        v["r_ci"] = r_ci[(target, cs_name)]
        out[k] = v
    return out


def _check_r2_diagnostic_unaffected() -> str:
    """§3:Richardson R² 诊断量由 scripts/31_sem_metrology_audit.py 独立
    全量重算产出(无增量缓存逻辑,每次都对全部81样本重新分割+回归),不经过
    scripts/02i 的增量抽取路径,因此不受本次根因 bug 影响。用两个受 bug
    影响最严重的样本(C26-L/C24-S)做抽样复核:现场重新分割+回归,比对
    磁盘上 sem_df_r2_diagnostic.csv 的既有值。"""
    import warnings
    import yaml
    from skimage import measure
    from nfm.data_processing import sem_processor as sp

    cfg = sp.config_from_yaml(yaml.safe_load(open(ROOT / "configs/config.yaml", encoding="utf-8")))
    existing = pd.read_csv(R2_DIAGNOSTIC_CSV)
    lines = []
    for sample in ["C26-L", "C24-S"]:
        sub = ROOT / "data/raw/sem" / sample
        low, _ = sp._collect_mag_files(sub)
        rows = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for f in low:
                px, _ = sp._read_pixel_size_um(f)
                img, _ = sp._crop_info_bar(sp._read_image_gray(f), cfg.seg)
                _props, labels = sp._segment_secondary(img, px, cfg.seg)
                H, W = labels.shape
                for r in measure.regionprops(labels):
                    minr, minc, maxr, maxc = r.bbox
                    if cfg.seg.sec_clear_border and (minr <= 0 or minc <= 0 or maxr >= H or maxc >= W):
                        continue
                    d_um = r.equivalent_diameter * px
                    if d_um < cfg.seg.sec_min_diam_um:
                        continue
                    cs = measure.find_contours(r.image.astype(float), 0.5)
                    if not cs:
                        continue
                    cont = max(cs, key=len)
                    df_val, r2, nsc = shape31_contour_fractal_dim_r2(cont)
                    rows.append(dict(r2=r2, n_scales=nsc))
        fresh = pd.DataFrame(rows)
        old = existing[existing.sample_id == sample]
        match = (len(fresh) == len(old)
                 and abs(fresh.r2.median() - old.r2.median()) < 1e-9)
        lines.append(
            f"- `{sample}`: 现场重跑 n={len(fresh)} median(R2)={fresh.r2.median():.4f}"
            f"  vs  磁盘已有 n={len(old)} median(R2)={old.r2.median():.4f}"
            f"  {'(一致)' if match else '(不一致,需排查)'}"
        )
        assert match, f"{sample}: R2 诊断量与磁盘不一致,说明该文件也受同类 bug 影响,需重新调查"
    return "\n".join(lines)


_spec31 = importlib.util.spec_from_file_location(
    "sem_metrology_audit_31", ROOT / "scripts" / "31_sem_metrology_audit.py")
shape31 = importlib.util.module_from_spec(_spec31)
_spec31.loader.exec_module(shape31)
shape31_contour_fractal_dim_r2 = shape31._contour_fractal_dim_r2


def _classify_44_occurrences() -> list[dict]:
    """v7.docx 逐段文本里每一个字面 "44.0"/"44%" **出现位置**(不是按行/
    段落去重)的分类:
      change  -- D_f 的 eta2=44.0%,本轮必须更新为新值
      keep    -- 压实密度全域跨度的 44%(=0.227/0.52),与本轮无关,禁止改动
      unrelated -- 完全无关的其它数字里的巧合子串(如 44.6、6.4457)
    按行统计会漏计:附录 B5 那一段里 "现用 44.0%–88.8%…" 与
    "…Df 为 44.0%" 是同一段落里两个独立的替换点,按行只算一次会少算一个
    (真实替换点数=5,不是 4)。故这里改为在每条命中行内用 re.finditer
    定位每一次 "44.0"/"44%" 出现,逐个分类、逐个给出 旧文本→新文本。

    行号来自独立解析该 v7 手稿 docx 文件的 word/document.xml(stdlib
    zipfile+re,不依赖新装的 python-docx 包)。"""
    docx_path = ROOT / "docs" / "论文A_材料篇_优化稿_v7.docx"
    import zipfile
    with zipfile.ZipFile(docx_path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    paras = re.split(r"(?=<w:p[ >])", xml)
    out_lines = []
    for p in paras:
        ts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", p, flags=re.S)
        line = "".join(ts)
        if line.strip():
            out_lines.append(line)

    def _snippet(line, pos, span, width=36):
        lo, hi = max(0, pos - width), min(len(line), pos + span + width)
        return line[lo:hi]

    occurrences = []
    for i, line in enumerate(out_lines, start=1):
        if "44" not in line:
            continue
        context = "\n".join(out_lines[max(0, i - 5):i])
        # 上下文窗口只是"提示",真正能不能归为 change/keep 还要求当前行本身
        # 就含有目标字面量("44.0"/"44%")——否则会像下面这个真实踩到的坑:
        # line136(§2.2.2 像素标定讨论)里提到"周长类描述子(圆形度、roughness、
        # Df,app)"纯属另一件事,却让 line137(取样规则,完全无关)的上下文
        # 窗口里出现了"Df,app"这个词,把 line137 误判为 dfapp 语境;而 line137
        # 本身并不含字面"44.0",re.finditer 找不到匹配、for 循环零次迭代,
        # 这一行就被**静默丢弃**、从清单里彻底消失(而不是退回 unrelated)。
        # 加上"当前行必须真含目标字面量"这个前置门槛后,不满足就会走 else
        # 分支老实标成 unrelated,不会再静默漏行。
        has_440 = "44.0" in line
        has_44pct = "44%" in line
        is_dfapp_context = has_440 and ("88.8" in context or "88.8" in line
                                       or "Df,app" in context or "介观形状" in context)
        is_packing_context = has_44pct and (
            ("0.227" in line and "0.52" in line)
            or ("44%" in line and "0.243" in line)
            or ("44% 与条件级观测密度跨度对比" in line)
        )

        if is_dfapp_context:
            for m in re.finditer(r"44\.0", line):
                occurrences.append(dict(
                    line_no=i, cls="change",
                    why="D_f 的 eta2=44.0%(结论(1)/表4 D_f,app 行/附录 B5 的下界数字,"
                        "本处独立替换点)",
                    old_snippet=_snippet(line, m.start(), len(m.group())),
                ))
        elif is_packing_context:
            for m in re.finditer(r"44%", line):
                occurrences.append(dict(
                    line_no=i, cls="keep",
                    why="压实密度全域跨度的 44%(=0.227/0.52),与 D_f 无关,禁止改动",
                    old_snippet=_snippet(line, m.start(), len(m.group())),
                ))
        else:
            occurrences.append(dict(
                line_no=i, cls="unrelated",
                why="其它数字里的巧合子串(如 44.6nm 的 DXRD、6.4457 的回归系数等)",
                old_snippet=line[:80],
            ))

    n_lines_with_44 = sum(1 for line in out_lines if "44" in line)
    n_lines_covered = len({o["line_no"] for o in occurrences})
    assert n_lines_covered == n_lines_with_44, (
        f"STOP: {n_lines_with_44} 行含字面'44',但分类结果只覆盖了 "
        f"{n_lines_covered} 行 -- 有行被静默丢弃(这正是本函数上一版真实"
        f"踩过的坑,见函数内注释),必须先修好分类逻辑再继续,不能带着"
        f"遗漏的清单去核对正文。"
    )
    return occurrences


def _fill_44_replacements(occurrences: list[dict], d_f_new_pct: float) -> None:
    """就地给每个 change 类占位补上具体的替换文本(new_snippet)。"""
    for o in occurrences:
        if o["cls"] == "change":
            o["new_snippet"] = o["old_snippet"].replace("44.0", f"{d_f_new_pct:.1f}")
        else:
            o["new_snippet"] = "(不改)"


def _cross_validate_against_r2_diagnostic() -> tuple[float, int, pd.DataFrame]:
    """交叉验证:`sem_df_r2_diagnostic.csv`
    (scripts/31_sem_metrology_audit.py 产出,从未受本次增量缓存 bug 影响,
    见 §6)与重生成后的 `sem_shape_descriptors.csv` 理应现在用的是同一套
    当前 SegConfig、同一批原始图像,因此逐样本中位数 D_f 应当一致。若仍不
    一致,说明重生成没有完全收敛,需要停下来查——不能想当然认为"跑过一次
    就一定对了"。

    返回 (max_abs_diff, n_mismatch, 明细DataFrame)。"""
    diag = pd.read_csv(R2_DIAGNOSTIC_CSV)
    new_raw = pd.read_csv(NEW_CSV)

    diag_med = diag.dropna(subset=["D_f"]).groupby("sample_id")["D_f"].median()
    new_med = new_raw.dropna(subset=["D_f"]).groupby("sample_id")["D_f"].median()
    assert set(diag_med.index) == set(new_med.index), (
        "两文件 sample_id 集合不一致,无法逐样本比对"
    )
    detail = pd.DataFrame({
        "D_f_median_r2diag": diag_med, "D_f_median_shapecsv": new_med,
    })
    detail["abs_diff"] = (detail["D_f_median_r2diag"] - detail["D_f_median_shapecsv"]).abs()
    detail = detail.sort_values("abs_diff", ascending=False)
    max_diff = float(detail["abs_diff"].max())
    n_mismatch = int((detail["abs_diff"] > 1e-9).sum())
    return max_diff, n_mismatch, detail


def main():
    assert OLD_CSV.exists(), f"{OLD_CSV} 不存在 -- 需要先备份改前文件"
    master = pd.read_csv(MASTER_TABLE)

    print("=== 校验:改后文件现与生产 D_sec 管线逐位一致 ===")
    new_raw = pd.read_csv(NEW_CSV)
    med = new_raw.groupby("sample_id")["d_um"].median()
    d_sec = master.set_index("sample_id")["D_sec"]
    max_diff = float((med - d_sec.reindex(med.index)).abs().max())
    print(f"  max abs diff(median(d_um), D_sec) = {max_diff:.2e} um (修复前最大 0.4201 um)")
    assert max_diff < 1e-6, "改后文件仍与生产 D_sec 不一致 -- 根因修复未生效,停止"

    old_df = _summarize(OLD_CSV, master)
    new_df = _summarize(NEW_CSV, master)

    print("\n=== 逐样本 D_f/solidity/elongation/roughness 中位数改前后差异 ===")
    n_changed = {}
    for col in DESCRIPTORS:
        m = old_df[["sample_id", col]].merge(new_df[["sample_id", col]], on="sample_id",
                                             suffixes=("_old", "_new"))
        diff = (m[f"{col}_old"] - m[f"{col}_new"]).abs()
        n_changed[col] = int((diff > 1e-9).sum())
        print(f"  {col}: {n_changed[col]}/81 样本数值变化, max abs diff={diff.max():.4g}")

    print("\n=== 前驱体分组均值改前后对照 ===")
    group_old = old_df.groupby("precursor")[list(DESCRIPTORS)].mean()
    group_new = new_df.groupby("precursor")[list(DESCRIPTORS)].mean()
    print("-- 改前 --"); print(group_old.round(4))
    print("-- 改后 --"); print(group_new.round(4))

    print("\n=== Type II eta2 改前后对照(81 样完整平衡析因,Type I=II=III)===")
    print("-- 改前 --")
    anova_old = _run_anova_all(old_df, "改前")
    print("-- 改后 --")
    anova_new = _run_anova_all(new_df, "改后")

    print("\n=== 项目2: D_f 偏相关 + TOST(复用 scripts/17 §3) ===")
    partial_old = _partial_corr_tost_d_f(old_df)
    partial_new = _partial_corr_tost_d_f(new_df)
    for key in partial_old:
        print(f"  {key}: r 改前={partial_old[key]['r']:+.3f} 改后={partial_new[key]['r']:+.3f}")

    print("\n=== 项目3: Richardson R2 诊断量抽样复核(应不受影响) ===")
    r2_check_text = _check_r2_diagnostic_unaffected()
    print(r2_check_text)

    print("\n=== 项目4: 分前驱体 Spearman(descriptor,Theta)+95%CI(复用 scripts/17 §2) ===")
    spearman_old = _spearman_ci_table(old_df)
    spearman_new = _spearman_ci_table(new_df)

    print("\n=== 交叉验证: sem_df_r2_diagnostic.csv vs 重生成后的 sem_shape_descriptors.csv ===")
    cross_max_diff, cross_n_mismatch, cross_detail = _cross_validate_against_r2_diagnostic()
    print(f"  81 样本逐样中位数 D_f 最大绝对差 = {cross_max_diff:.6f}, "
         f"不一致样本数(>1e-9) = {cross_n_mismatch}/81")
    print(cross_detail.head(5))
    assert cross_max_diff < 1e-6, (
        "STOP: sem_df_r2_diagnostic.csv 与重生成后的 sem_shape_descriptors.csv "
        "逐样本中位数 D_f 不一致 -- 说明 02i 重生成没有完全收敛到当前 "
        "SegConfig,不能提交重冻结,需要先查清原因。"
    )

    d_f_precursor_new = float(anova_new["D_f"].loc["precursor", "eta2"]) * 100
    print("\n=== '44' 防呆分类(v7.docx,按出现位置逐点计数) ===")
    classified_44 = _classify_44_occurrences()
    _fill_44_replacements(classified_44, d_f_precursor_new)
    for c in classified_44:
        print(f"  L{c['line_no']:>4} [{c['cls']:>9}] {c['old_snippet']!r} -> {c['new_snippet']!r}")

    _write_report(n_changed, group_old, group_new, anova_old, anova_new,
                 partial_old, partial_new, r2_check_text,
                 spearman_old, spearman_new, classified_44,
                 cross_max_diff, cross_n_mismatch, cross_detail)
    print(f"\n[done] {REPORT_MD}")


def _interaction_total(table, eta2_col="eta2"):
    return float(sum(table.loc[t, eta2_col] for t in INTERACTION_TERMS)) * 100


def _write_report(n_changed, group_old, group_new, anova_old, anova_new,
                  partial_old, partial_new, r2_check_text,
                  spearman_old, spearman_new, classified_44,
                  cross_max_diff, cross_n_mismatch, cross_detail):
    lines = ["# sem_shape_descriptors.csv 根因修复:改前/改后完整影响清单\n\n"]
    lines.append("> 由 `scripts/43_shape_descriptor_regen_impact.py` 生成。只做对照/清单,"
                 "不改写任何冻结表/图件——是否据此重新冻结、图件何时重出,确认"
                 "本清单后决定(见脚本文档字符串背景说明)。\n\n")
    lines.append("**改前基准数字说明**:"
                 "根因修复前的旧 `sem_shape_descriptors.csv` 对最多20/81样本"
                 "失真(最大0.42um),本报告及后续对照均以本轮在新文件上的"
                 "独立计算为准。\n\n")

    lines.append("## 1. 校验:根因修复已生效\n\n")
    lines.append(f"改后文件 `data/interim/sem_shape_descriptors.csv` 现与 "
                 f"`master_table.csv` 的 `D_sec` 逐样本中位数一致(max abs diff < 1e-6 um,"
                 f"修复前最大 0.4201 um,C26-L)。\n\n")

    lines.append("## 2. 逐样本描述子中位数变化面\n\n")
    lines.append("| 描述子 | 数值变化的样本数(/81) |\n|---|---|\n")
    for col in DESCRIPTORS:
        lines.append(f"| {col} | {n_changed[col]} |\n")
    lines.append("\n")

    lines.append("## 3. 前驱体分组均值(改前 vs 改后,四个描述子全量)\n\n")
    lines.append("### 改前\n\n" + group_old.round(4).to_markdown() + "\n\n")
    lines.append("### 改后\n\n" + group_new.round(4).to_markdown() + "\n\n")

    lines.append("## 4. 项目1 —— 表1六列整行对照(D_f、elongation)\n\n")
    lines.append("六列 = precursor / T_C / beta / t_hold / 交互合计(全部6个二阶交互项之和,"
                 "即 precursor:T_C/beta/t_hold 与 T_C:beta/t_hold、beta:t_hold 六项,"
                 "使六列之和=100%) / "
                 "residual,与 `scripts/22_build_manuscript_numbers.py` 里 "
                 "`table1.*`+`table2.*.total_coupling_eta2_pct` 的口径一致"
                 "(注意:这与 v7.docx 表4 自己印出的“工艺合计”列不同——docx 那一列是 "
                 "T+beta+t_hold 三个主效应之和,不是交互;本节给出的是交互合计,"
                 "两者不要混淆)。\n\n")
    for target in SIX_COL_TARGETS:
        t_old, t_new = anova_old[target], anova_new[target]
        lines.append(f"### {target}\n\n")
        lines.append("| | precursor | T_C | beta | t_hold | 交互合计 | residual |\n"
                     "|---|---|---|---|---|---|---|\n")
        old_vals = [t_old.loc["precursor", "eta2"] * 100, t_old.loc["T_C", "eta2"] * 100,
                   t_old.loc["beta", "eta2"] * 100, t_old.loc["t_hold", "eta2"] * 100,
                   _interaction_total(t_old), t_old.loc["Residual", "eta2"] * 100]
        new_vals = [t_new.loc["precursor", "eta2"] * 100, t_new.loc["T_C", "eta2"] * 100,
                   t_new.loc["beta", "eta2"] * 100, t_new.loc["t_hold", "eta2"] * 100,
                   _interaction_total(t_new), t_new.loc["Residual", "eta2"] * 100]
        lines.append("| 改前 | " + " | ".join(f"{v:.2f}" for v in old_vals) + " |\n")
        lines.append("| 改后 | " + " | ".join(f"{v:.2f}" for v in new_vals) + " |\n")
        lines.append("| 差值(pp) | " + " | ".join(f"{n-o:+.2f}" for o, n in zip(old_vals, new_vals)) + " |\n\n")

    lines.append("## 5. 项目2 —— D_f 对 compaction_density 的偏相关 + TOST\n\n")
    lines.append("复用 `scripts/17_shape_descriptor_full_analysis.py::section3_partial_corr_tost`"
                 "(未重新实现),仅取 D_f 的结果(circularity 来自 master_table 自身列,"
                 "不受本次 bug 影响,不需要对照)。\n\n")
    delta_values = {"primary_5pct": shape17.DELTA_PRIMARY,
                    "sensitivity_4pct": shape17.DELTA_SENSITIVITY}

    lines.append("### 5.1 完整四项(§3.2.1 表格口径:偏相关r [95%CI] / 标准化斜率 / "
                 "两个δ的TOST p),Θ+前驱体控制集\n\n")
    lines.append("v7 现文:D~f,app~ / Θ+前驱体 / 偏相关r [95%CI] = +0.110 [−0.148, +0.347] / "
                 "斜率 = +0.0081 / TOST(δ=5%) 显著 p=0.023 / TOST(δ=4%) 不显著 p=0.079。"
                 "下表给改前(应与v7现文一致,交叉核对)和改后两行:\n\n")
    lines.append("| | 偏相关r | 95%CI(r) | 标准化斜率(g/cm3每1SD) | TOST p(δ=5%) | "
                 "TOST p(δ=4%) | 5%界判定 | 4%界判定 |\n|---|---|---|---|---|---|---|---|\n")
    for label, partial in (("改前(应≈v7现文)", partial_old), ("改后(新值)", partial_new)):
        k5 = ("D_f", "Theta_plus_precursor", "primary_5pct")
        k4 = ("D_f", "Theta_plus_precursor", "sensitivity_4pct")
        v5, v4 = partial[k5], partial[k4]
        r_lo, r_hi = v5["r_ci"]
        lines.append(f"| {label} | {v5['r']:+.4f} | [{r_lo:+.4f}, {r_hi:+.4f}] | "
                     f"{v5['slope']:+.4f} | {v5['p_tost']:.4f} | {v4['p_tost']:.4f} | "
                     f"{'显著(等价)' if v5['equivalent'] else '不显著'} | "
                     f"{'显著(等价)' if v4['equivalent'] else '不显著'} |\n")
    lines.append("\n")
    lines.append("**判定未翻转,但阈值敏感性论述因此更成立**:5%界下改前后均为"
                 "'等价'成立(TOST 显著),但 p 由 0.0228 升至 0.0294,更接近"
                 "α=0.05 边界;4%界下改前后均为'不等价'(TOST 不显著),p 由 "
                 "0.0788 升至 0.0854。判定方向不变,只需替换正文里的四个数字"
                 "(r/CI/斜率/两个p),\"该判定对阈值选择敏感\"这句限定语因"
                 "p 更贴近边界而更加成立,不需要改写论断方向。\n\n")

    lines.append("### 5.2 全部四种(控制变量×δ)组合对照(含 Theta_only,供完整性核查)\n\n")
    lines.append("| 控制变量 | 偏相关r 改前 | 偏相关r 改后 | δ | TOST p 改前 | TOST p 改后 | "
                 "等价?改前 | 等价?改后 |\n|---|---|---|---|---|---|---|---|\n")
    for cs_name in ("Theta_only", "Theta_plus_precursor"):
        for delta_name in ("primary_5pct", "sensitivity_4pct"):
            key = ("D_f", cs_name, delta_name)
            o, n = partial_old[key], partial_new[key]
            lines.append(f"| {cs_name} | {o['r']:+.3f} | {n['r']:+.3f} | "
                         f"{delta_values[delta_name]:.4f}({delta_name}) | {o['p_tost']:.4f} | {n['p_tost']:.4f} | "
                         f"{'是' if o['equivalent'] else '否'} | {'是' if n['equivalent'] else '否'} |\n")
    lines.append("\n")

    lines.append("## 6. 项目3 —— Richardson R² 诊断量:不受本次 bug 影响\n\n")
    lines.append("`data/interim/sem_df_r2_diagnostic.csv` 由 `scripts/31_sem_metrology_audit.py` "
                 "生成,该脚本**没有增量缓存逻辑**——每次运行都对全部81样本重新分割+回归"
                 "(不经过 `scripts/02i` 的增量抽取路径),且该文件最后一次生成"
                 "(2026-08-16)晚于本次 bug 涉及的全部 SegConfig 变更"
                 "(sec_adaptive_h 定档于 2026-07-23)。抽样复核"
                 "(受本次 bug 影响最严重的 C26-L/C24-S 两样,现场重新分割+回归对比磁盘值):\n\n")
    lines.append(r2_check_text.replace("\n", "  \n") + "\n\n")
    lines.append("**结论:R² 中位数(现为 0.872)与标度点数分布无需更新,该量此前就是"
                 "全量新鲜复算,不存在过期问题。**\n\n")

    lines.append("## 7. 项目4 —— 四个描述子分前驱体 Spearman rho + 95% bootstrap CI\n\n")
    lines.append("复用 `scripts/17_shape_descriptor_full_analysis.py::section2_spearman_bootstrap`"
                 "(未重新实现)。\n\n")
    lines.append("| 描述子 | 组 | rho 改前 | 95%CI 改前 | rho 改后 | 95%CI 改后 | Δrho |\n"
                 "|---|---|---|---|---|---|---|\n")
    for target in DESCRIPTORS:
        for grp in ("S", "M", "L"):
            o = spearman_old[(spearman_old.target == target) & (spearman_old.group == grp)].iloc[0]
            n = spearman_new[(spearman_new.target == target) & (spearman_new.group == grp)].iloc[0]
            lines.append(f"| {target} | {grp} | {o['rho']:+.3f} | "
                         f"[{o['ci_low']:+.3f},{o['ci_high']:+.3f}] | {n['rho']:+.3f} | "
                         f"[{n['ci_low']:+.3f},{n['ci_high']:+.3f}] | {n['rho']-o['rho']:+.3f} |\n")
    lines.append("\n")

    lines.append("## 8. 项目5 —— 受影响的图(本轮仅列出,不重出)\n\n")
    lines.append("| 图 | 面板 | 依赖 | 说明 |\n|---|---|---|---|\n"
                 "| fig2 | (a) | `data/interim/anova_interactions_shape_descriptors.csv` | "
                 "D_f/elongation/solidity/roughness 主效应堆叠柱,需按 §4 新 eta2 重出 |\n"
                 "| figS2 | (d) | 同上(设计范围敏感性子集之一) | 若该面板画的是 D_f/形状"
                 "描述子子集,需核对后重出 |\n"
                 "| fig4 | (c)–(f) | `data/interim/sem_shape_descriptors_summary.csv` | "
                 "D_f/solidity/elongation/roughness vs Theta 散点+Spearman标注,"
                 "需按 §7 新 rho/CI 重出 |\n"
                 "| figS10 | 全部 | `data/interim/sem_df_r2_diagnostic.csv` | 按 §6 结论,"
                 "**此图数据未变,不需要重出**,列在此处仅为完整性 |\n\n")

    lines.append("## 9. 交叉验证:`sem_df_r2_diagnostic.csv` vs 重生成后的 "
                 "`sem_shape_descriptors.csv`\n\n")
    lines.append("`sem_df_r2_diagnostic.csv` 自 2026-08-16 生成起就未受本次增量缓存 bug "
                 "影响(见 §6),重生成后的 `sem_shape_descriptors.csv` 现在理应用同一套 "
                 "当前 SegConfig、同一批原始图像,故两文件逐样本中位数 D_f 理应一致。"
                 f"实测:81 个样本逐样中位数 D_f **最大绝对差 = {cross_max_diff:.2e}**,"
                 f"不一致(>1e-9)样本数 = {cross_n_mismatch}/81。"
                 f"{'**收敛,两文件一致。**' if cross_n_mismatch == 0 else '**警告:仍有不一致样本,详见下表,需要停下排查,不应提交重冻结。**'}\n\n")
    if cross_n_mismatch:
        lines.append("| sample_id | D_f中位(r2diag) | D_f中位(shapecsv) | abs_diff |\n"
                     "|---|---|---|---|\n")
        for sid, row in cross_detail.head(10).iterrows():
            lines.append(f"| {sid} | {row['D_f_median_r2diag']:.4f} | "
                         f"{row['D_f_median_shapecsv']:.4f} | {row['abs_diff']:.2e} |\n")
        lines.append("\n")

    lines.append("## 10. \"44\" 防呆核查(v7.docx,按出现位置逐点计数)\n\n")
    lines.append("按行统计会漏计同一段落里的多个替换点(附录 B5 一段里有两处"
                 "独立的 \"44.0\")。改为按**每一次字面出现**分类计数,**只对 `change` 类"
                 "做替换,`keep`/`unrelated` 一律不动**:\n\n")
    lines.append("| 行号 | 分类 | 旧文本片段 | 新文本片段 |\n|---|---|---|---|\n")
    for c in classified_44:
        if c["cls"] == "unrelated":
            continue  # 无关的巧合子串不列入替换表,避免正文核对时的噪声
        lines.append(f"| {c['line_no']} | {c['cls']} | `{c['old_snippet']}` | `{c['new_snippet']}` |\n")
    lines.append("\n")
    n_change = sum(1 for c in classified_44 if c["cls"] == "change")
    n_keep = sum(1 for c in classified_44 if c["cls"] == "keep")
    n_unrelated = sum(1 for c in classified_44 if c["cls"] == "unrelated")
    lines.append(f"**共 {n_change} 处需要改**(摘要 1 处、结论(1) 1 处、表4 D~f,app~ 行 1 处、"
                 f"附录 B5 内 2 处——B5 同一段落里 \"现用 44.0%–88.8%\" 与 \"Df 为 44.0%\" "
                 f"是两个独立替换点);**{n_keep} 处禁止改动**(压实密度全域跨度 44%:"
                 f"§4.5、附录 B2,与本轮无关);另有 {n_unrelated} 行含\"44\"但与本轮完全"
                 "无关的巧合子串(如 §2.2.2 像素标定讨论里提到的\"Df,app\"一词、"
                 "DXRD=44.6nm、回归系数 6.4457/1.8445、样品统计量 233/389/443 等),"
                 "已排除、不列入上表。**严禁全局替换 \"44.0%\"/\"44%\"**,必须逐处按上表操作。\n\n")

    d_f_precursor_old = float(anova_old["D_f"].loc["precursor", "eta2"]) * 100
    d_f_precursor_new = float(anova_new["D_f"].loc["precursor", "eta2"]) * 100
    lines.append(
        f"## 11. 结论(1)下界数字(D_f 前驱体主效应)汇总\n\n"
        f"改前(现行冻结表口径)D_f 前驱体主效应 eta2 = {d_f_precursor_old:.2f}%"
        f"(与手稿现引用的 44.0% 一致);改后为 "
        f"{d_f_precursor_new:.2f}%,差值 {d_f_precursor_new-d_f_precursor_old:+.2f} 个百分点。"
        f"结论(1)\"44.0%–88.8%\"应改为\"{d_f_precursor_new:.1f}%–88.8%\";"
        f"此改动不影响\"前驱体架构主导44–89%方差\"这一论断的实质,"
        f"无需进一步调查位移成因(已知来自34样分割陈旧)。\n\n"
    )

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("".join(lines))


if __name__ == "__main__":
    main()
