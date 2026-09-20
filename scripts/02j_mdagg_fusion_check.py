# -*- coding: utf-8 -*-
"""
02j_mdagg_fusion_check.py —— T2: M_D_agg 熔连伪影诊断（2026-07-31）
=================================================================
背景:M_D_agg = [D_sec(L) - D_sec(S)] / [precursor_D50(L) - precursor_D50(S)]（同炉
条件配对,27 个条件)随 Θ 上升(Spearman +0.558, p=0.0025),论文据此论证"前驱体记忆
随烧结进程被固化"。风险:高 Θ 下颗粒可能发生熔连(颈缩/相邻颗粒粘连成一个大团块),
分水岭分割器若把熔连体误判为单个大颗粒,会虚增 D_sec;若 S/L 两组熔连程度不同,
M_D_agg 的上升趋势可能是分割伪影而非真实的记忆固化。

本脚本只读调用
src/nfm/data_processing/sem_processor.py 的生产分割函数(_segment_secondary 等),
不改动该文件本体,不改 configs/config.yaml 落盘默认值(§2 的参数扫描全部在本脚本内
用 dataclasses.replace() 在 Python 对象层面覆盖 SegConfig)。

各小节对应任务书:
  §1 颗粒计数诊断(n_sec vs n_particles 口径核实 + Spearman(计数, Θ))
  §2 当前生产管线(自适应 h)参数敏感性扫描 —— 主证据,断点续跑
  §3 旧固定-h 扫描(sem_sec_sweep/metrics.csv)作为辅助交叉证据,非生产管线敏感性
  §4 solidity/roughness vs Θ 一致性检查(熔连的独立形状证据)
  §5 诊断叠加图版(C01/C14/C21 × S/L)

用法:
  python scripts/02j_mdagg_fusion_check.py --step1                 # §1 计数诊断(快)
  python scripts/02j_mdagg_fusion_check.py --sweep [--limit N]     # §2 扫描,断点续跑
  python scripts/02j_mdagg_fusion_check.py --sweep-report          # §2 汇总相关系数(需已有 sweep csv)
  python scripts/02j_mdagg_fusion_check.py --legacy                # §3 旧扫描交叉证据
  python scripts/02j_mdagg_fusion_check.py --shape                 # §4 形状一致性
  python scripts/02j_mdagg_fusion_check.py --overlays              # §5 诊断图版
  python scripts/02j_mdagg_fusion_check.py --figure                # 汇总图 fig_S_mdagg_robustness
  python scripts/02j_mdagg_fusion_check.py --diagnostic-csv        # 写 mdagg_fusion_diagnostic.csv
  python scripts/02j_mdagg_fusion_check.py --all [--limit N]       # 全跑(sweep 用 --limit 断点)
"""
from __future__ import annotations

import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfm.config import load_config, load_precursor_dvalues  # noqa: E402
from nfm.data_processing import sem_processor as sp  # noqa: E402

MASTER = ROOT / "data" / "processed" / "master_table.csv"
SHAPE_SUMMARY = ROOT / "data" / "interim" / "sem_shape_descriptors_summary.csv"
LEGACY_METRICS = ROOT / "data" / "interim" / "sem_sec_sweep" / "metrics.csv"
SWEEP_CSV = ROOT / "data" / "interim" / "mdagg_sensitivity_sweep.csv"
DIAG_CSV = ROOT / "data" / "interim" / "mdagg_fusion_diagnostic.csv"
FIG_DIR = ROOT / "reports" / "figures"

DEFAULT_PERCENTILE = 90.0
DEFAULT_SLOPE = 0.082
PERCENTILE_GRID = [75.0, 80.0, 85.0, 90.0, 95.0]
SLOPE_GRID = [0.033, 0.057, 0.082, 0.107, 0.131]

N_SEC_SMALL_SAMPLE_THRESH = 50


# ---------------------------------------------------------------------
# 公共数据加载
# ---------------------------------------------------------------------
def load_common():
    cfg = load_config()
    mt = pd.read_csv(MASTER)
    d50 = load_precursor_dvalues(cfg)  # {'S':.., 'M':.., 'L':..} um
    return cfg, mt, d50


def compute_mdagg_theta(d_sec_by_sample: pd.Series, mt: pd.DataFrame, d50: dict):
    """给定某个分割参数组合下逐样本 D_sec(Series, index=sample_id),按 condition_id
    配对 S/L 算出 27 个 M_D_agg,再算 Spearman(M_D_agg, Theta)。

    返回 (per_condition_df, rho, p, n_valid_conditions)。
    """
    df = mt[["sample_id", "condition_id", "precursor", "Theta"]].copy()
    df["D_sec"] = df["sample_id"].map(d_sec_by_sample)
    den = d50["L"] - d50["S"]
    rows = []
    for cid, g in df.groupby("condition_id"):
        s = g[g.precursor == "S"]
        l = g[g.precursor == "L"]
        theta = g["Theta"].iloc[0]
        if len(s) != 1 or len(l) != 1:
            rows.append(dict(condition_id=cid, Theta=theta, D_sec_S=np.nan, D_sec_L=np.nan, M_D_agg=np.nan))
            continue
        d_s, d_l = s["D_sec"].iloc[0], l["D_sec"].iloc[0]
        m_dagg = (d_l - d_s) / den if pd.notna(d_s) and pd.notna(d_l) and abs(den) > 1e-9 else np.nan
        rows.append(dict(condition_id=cid, Theta=theta, D_sec_S=d_s, D_sec_L=d_l, M_D_agg=m_dagg))
    out = pd.DataFrame(rows)
    valid = out.dropna(subset=["M_D_agg", "Theta"])
    if len(valid) < 4:
        return out, np.nan, np.nan, len(valid)
    rho, p = stats.spearmanr(valid["Theta"], valid["M_D_agg"])
    return out, float(rho), float(p), len(valid)


# ---------------------------------------------------------------------
# §1 颗粒计数诊断
# ---------------------------------------------------------------------
def section1_counts(mt: pd.DataFrame, shape_summary: pd.DataFrame):
    df = mt[["sample_id", "condition_id", "precursor", "Theta", "n_sec"]].copy()
    ns = shape_summary[["sample_id", "n_particles"]]
    df = df.merge(ns, on="sample_id", how="left")

    corr_rows = []
    for col in ["n_sec", "n_particles"]:
        for p in "SML":
            s = df[df.precursor == p].dropna(subset=[col, "Theta"])
            if len(s) >= 4:
                rho, pv = stats.spearmanr(s["Theta"], s[col])
            else:
                rho, pv = np.nan, np.nan
            corr_rows.append(dict(count_metric=col, precursor=p, rho=rho, p=pv, n=len(s)))
    corr_df = pd.DataFrame(corr_rows)

    ratio = (df["n_sec"] / df["n_particles"]).replace([np.inf, -np.inf], np.nan)
    small_n_sec = df[df.n_sec < N_SEC_SMALL_SAMPLE_THRESH]

    print("=== §1 颗粒计数诊断 ===")
    print(f"n_sec/n_particles 比值: median={ratio.median():.3f}  max={ratio.max():.3f}  min={ratio.min():.3f}")
    print(corr_df.to_string(index=False))
    print(f"\nn_sec < {N_SEC_SMALL_SAMPLE_THRESH} 的样本数: {len(small_n_sec)}")
    if len(small_n_sec):
        print(small_n_sec[["sample_id", "n_sec", "n_particles", "Theta"]].to_string(index=False))
    return df, corr_df, small_n_sec


# ---------------------------------------------------------------------
# §2 当前生产管线(自适应 h)参数敏感性扫描
# ---------------------------------------------------------------------
def combos_to_run():
    """9 个组合:default 一次 + percentile 4 个非默认值 + slope 4 个非默认值。"""
    combos = [("default", DEFAULT_PERCENTILE, DEFAULT_SLOPE)]
    for p in PERCENTILE_GRID:
        if abs(p - DEFAULT_PERCENTILE) < 1e-9:
            continue
        combos.append((f"percentile_{p:.0f}", p, DEFAULT_SLOPE))
    for s in SLOPE_GRID:
        if abs(s - DEFAULT_SLOPE) < 1e-9:
            continue
        combos.append((f"slope_{s:.3f}", DEFAULT_PERCENTILE, s))
    return combos


def make_seg_config(base_seg, percentile, slope):
    """在 Python 对象层面覆盖 SegConfig 的两个自适应 h 参数,不改 base_seg 本身、
    不落盘到 configs/config.yaml。"""
    return replace(base_seg, sec_adaptive_h=True,
                   sec_adaptive_h_percentile=percentile,
                   sec_adaptive_h_slope=slope)


def all_sample_dirs():
    raw = ROOT / "data" / "raw" / "sem"
    return sorted(p for p in raw.iterdir() if p.is_dir() and sp.SAMPLE_ID_RE.match(p.name))


def sample_d_sec(sub: Path, seg_cfg) -> dict:
    """对一个样本的全部 500x 图跑生产 _segment_secondary,concat 后用生产
    _aggregate_secondary 聚合(与 sem_processor._process_one_sample 的 500x 分支等价)。"""
    low, _high = sp._collect_mag_files(sub)
    if not low:
        return dict(sample_id=sub.name, n_sec=0, D_sec=np.nan, n_images=0)
    props_list = []
    for f in low:
        px, _ = sp._read_pixel_size_um(f)
        img, _ = sp._crop_info_bar(sp._read_image_gray(f), seg_cfg)
        p, _labels = sp._segment_secondary(img, px, seg_cfg)
        props_list.append(p)
    cat = pd.concat(props_list, ignore_index=True)
    agg = sp._aggregate_secondary(cat)
    return dict(sample_id=sub.name, n_sec=agg["n_sec"], D_sec=agg["D_sec"], n_images=len(low))


def run_sweep(limit: int | None):
    raw_yaml = yaml.safe_load(open(ROOT / "configs" / "config.yaml", encoding="utf-8"))
    sem_cfg = sp.config_from_yaml(raw_yaml)
    base_seg = sem_cfg.seg  # 生产默认(sec_adaptive_h=True, percentile=90, slope=0.082, ...)

    SWEEP_CSV.parent.mkdir(parents=True, exist_ok=True)
    prev = pd.read_csv(SWEEP_CSV) if SWEEP_CSV.exists() else pd.DataFrame()
    rows = prev.to_dict("records") if len(prev) else []
    have = set(zip(prev.get("combo_id", []), prev.get("sample_id", [])))

    combos = combos_to_run()
    n_new = 0
    t_start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for combo_id, percentile, slope in combos:
            seg_cfg = make_seg_config(base_seg, percentile, slope)
            for sub in all_sample_dirs():
                if (combo_id, sub.name) in have:
                    continue
                if limit is not None and n_new >= limit:
                    pd.DataFrame(rows).to_csv(SWEEP_CSV, index=False, encoding="utf-8-sig")
                    print(f"[sweep] --limit {limit} 达到,已写回 {len(rows)} 行,断点已保存")
                    return
                r = sample_d_sec(sub, seg_cfg)
                r.update(combo_id=combo_id, percentile=percentile, slope=slope)
                rows.append(r)
                have.add((combo_id, sub.name))
                n_new += 1
                dsec_str = f"{r['D_sec']:.2f}" if pd.notna(r["D_sec"]) else "NaN"
                elapsed = time.time() - t_start
                print(f"[sweep] {combo_id:14s} {sub.name}  n_sec={r['n_sec']:4d} "
                      f"D_sec={dsec_str}  ({n_new} new, {elapsed:.0f}s elapsed)", flush=True)
                if n_new % 20 == 0:
                    pd.DataFrame(rows).to_csv(SWEEP_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(SWEEP_CSV, index=False, encoding="utf-8-sig")
    total_expected = len(combos) * len(all_sample_dirs())
    print(f"[sweep] 完成,共 {len(rows)} 行(应为 {total_expected} 行) -> {SWEEP_CSV}")


def sweep_correlations(mt: pd.DataFrame, d50: dict) -> pd.DataFrame:
    if not SWEEP_CSV.exists():
        print("[sweep-report] 无 sweep csv,先跑 --sweep")
        return pd.DataFrame()
    df = pd.read_csv(SWEEP_CSV)
    out = []
    for combo_id, g in df.groupby("combo_id"):
        d_sec_map = g.set_index("sample_id")["D_sec"]
        _mdf, rho, p, n = compute_mdagg_theta(d_sec_map, mt, d50)
        row = dict(combo_id=combo_id, percentile=g["percentile"].iloc[0], slope=g["slope"].iloc[0],
                   rho=rho, p=p, n_conditions=n, n_samples_done=len(g))
        out.append(row)
    res = pd.DataFrame(out)
    n_combos_expected = len(combos_to_run())
    n_combos_done = res["combo_id"].nunique()
    print(f"[sweep-report] {n_combos_done}/{n_combos_expected} 个参数组合已跑完(每组合应含 81 样本)")
    print(res.sort_values(["combo_id"]).to_string(index=False))
    return res


# ---------------------------------------------------------------------
# §3 旧固定-h 扫描(不同算法家族,仅作交叉参考)
# ---------------------------------------------------------------------
def section3_legacy(mt: pd.DataFrame, d50: dict) -> pd.DataFrame:
    if not LEGACY_METRICS.exists():
        print(f"[legacy] {LEGACY_METRICS} 不存在,跳过")
        return pd.DataFrame()
    m = pd.read_csv(LEGACY_METRICS)
    m = m[np.isclose(m["sec_min_diam_um"], 1.0)]  # 排除 d0.6 那档(与本任务 min_diam 口径不一致)
    out = []
    for combo, g in m.groupby("combo"):
        d_sec_map = g.set_index("sample_id")["D_sec"]
        _mdf, rho, p, n = compute_mdagg_theta(d_sec_map, mt, d50)
        n_samples_with_data = int(g["D_sec"].notna().sum())
        out.append(dict(combo=combo, sec_hmaxima_rel=g["sec_hmaxima_rel"].iloc[0],
                        bin_method=g["bin_method"].iloc[0],
                        rho=rho, p=p, n_conditions=n,
                        n_samples_total=len(g), n_samples_with_data=n_samples_with_data))
    res = pd.DataFrame(out).sort_values(["bin_method", "sec_hmaxima_rel"])
    print("=== §3 旧固定-h 扫描(不同算法家族,仅交叉参考,非生产管线敏感性) ===")
    print(res.to_string(index=False))
    return res


# ---------------------------------------------------------------------
# §4 形状一致性检查(solidity/roughness vs Theta)
# ---------------------------------------------------------------------
def section4_shape(shape_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["solidity", "roughness"]:
        for p in "SML":
            s = shape_summary[shape_summary.precursor == p].dropna(subset=[col, "Theta"])
            if len(s) >= 4:
                rho, pv = stats.spearmanr(s["Theta"], s[col])
            else:
                rho, pv = np.nan, np.nan
            rows.append(dict(metric=col, precursor=p, rho=rho, p=pv, n=len(s)))
    res = pd.DataFrame(rows)
    print("=== §4 solidity/roughness vs Theta(分前驱体 Spearman) ===")
    print(res.to_string(index=False))
    return res


# ---------------------------------------------------------------------
# §5 诊断叠加图版
# ---------------------------------------------------------------------
DIAG_SAMPLES = ["C01-S", "C01-L", "C14-S", "C14-L", "C21-S", "C21-L"]


def make_diagnostic_overlays():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    raw_yaml = yaml.safe_load(open(ROOT / "configs" / "config.yaml", encoding="utf-8"))
    sem_cfg = sp.config_from_yaml(raw_yaml)
    seg_cfg = sem_cfg.seg  # 生产默认参数(percentile=90, slope=0.082)

    fig, axes = plt.subplots(len(DIAG_SAMPLES), 2, figsize=(10, 4.4 * len(DIAG_SAMPLES)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, sid in enumerate(DIAG_SAMPLES):
            sub = ROOT / "data" / "raw" / "sem" / sid
            low, _high = sp._collect_mag_files(sub)
            if not low:
                for j in range(2):
                    axes[i, j].axis("off")
                axes[i, 0].set_title(f"{sid}: 无 500x 图")
                continue
            f = low[0]
            px, _ = sp._read_pixel_size_um(f)
            img, _ = sp._crop_info_bar(sp._read_image_gray(f), seg_cfg)
            props, labels = sp._segment_secondary(img, px, seg_cfg)

            axes[i, 0].imshow(img, cmap="gray")
            axes[i, 0].set_title(f"{sid}  raw ({f.name})")
            axes[i, 0].axis("off")

            ov = np.dstack([img] * 3).astype(float) / 255
            b = find_boundaries(labels, mode="outer")
            ov[b] = [0, 1, 0]
            axes[i, 1].imshow(ov)
            med = props["equiv_diam_um"].median() if len(props) else float("nan")
            axes[i, 1].set_title(f"{sid}  segmented n={len(props)}  D_sec_med={med:.2f}um")
            axes[i, 1].axis("off")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_S_mdagg_fusion_overlays.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[overlays] wrote {out}")
    return out


# ---------------------------------------------------------------------
# 汇总图 fig_S_mdagg_robustness.png/pdf
# ---------------------------------------------------------------------
def make_robustness_figure(sweep_corr_df: pd.DataFrame, legacy_corr_df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    dfp = sweep_corr_df[np.isclose(sweep_corr_df["slope"], DEFAULT_SLOPE)].sort_values("percentile")
    ax = axes[0]
    if len(dfp):
        ax.plot(dfp["percentile"], dfp["rho"], "o-", color="#3B6FA6", lw=2, markersize=7)
        for _, r in dfp.iterrows():
            if pd.notna(r["p"]):
                ax.annotate("*" if r["p"] < 0.05 else "n.s.", (r["percentile"], r["rho"]),
                            textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(DEFAULT_PERCENTILE, color="#999999", lw=0.8, ls=":")
    ax.set_xlabel("sec_adaptive_h_percentile")
    ax.set_ylabel("Spearman rho(M_D_agg, Theta)  n=27 conditions")
    ax.set_title("percentile sweep (slope fixed = 0.082 default)")

    dfs = sweep_corr_df[np.isclose(sweep_corr_df["percentile"], DEFAULT_PERCENTILE)].sort_values("slope")
    ax = axes[1]
    if len(dfs):
        ax.plot(dfs["slope"], dfs["rho"], "o-", color="#3B6FA6", lw=2, markersize=7)
        for _, r in dfs.iterrows():
            if pd.notna(r["p"]):
                ax.annotate("*" if r["p"] < 0.05 else "n.s.", (r["slope"], r["rho"]),
                            textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(DEFAULT_SLOPE, color="#999999", lw=0.8, ls=":")
    ax.set_xlabel("sec_adaptive_h_slope")
    ax.set_title("slope sweep (percentile fixed = 90 default)")

    ax = axes[2]
    if len(legacy_corr_df):
        x = np.arange(len(legacy_corr_df))
        colors = ["#B0B0B0" if p >= 0.05 or pd.isna(p) else "#6E9BC9" for p in legacy_corr_df["p"]]
        ax.bar(x, legacy_corr_df["rho"], color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(legacy_corr_df["combo"], rotation=75, ha="right", fontsize=7)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_title("legacy FIXED-h scan (different algorithm family,\ncross-reference ONLY — not production sensitivity)")

    fig.suptitle("M_D_agg vs Theta (Spearman, n=27 conditions) robustness under segmentation "
                 "parameter perturbation", y=1.03)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / "fig_S_mdagg_robustness.png"
    pdf = FIG_DIR / "fig_S_mdagg_robustness.pdf"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] wrote {png} and {pdf}")
    return png, pdf


# ---------------------------------------------------------------------
# 汇总诊断 CSV
# ---------------------------------------------------------------------
def write_diagnostic_csv(count_df, count_corr_df, sweep_corr_df, legacy_corr_df, shape_corr_df,
                         default_mdagg_df):
    frames = []
    a = count_df.copy(); a["section"] = "s1_counts_per_sample"
    frames.append(a)
    b = count_corr_df.copy(); b["section"] = "s1_counts_spearman"
    frames.append(b)
    c = sweep_corr_df.copy(); c["section"] = "s2_adaptive_h_sensitivity"
    frames.append(c)
    d = legacy_corr_df.copy(); d["section"] = "s3_legacy_fixed_h_crossref"
    frames.append(d)
    e = shape_corr_df.copy(); e["section"] = "s4_shape_consistency"
    frames.append(e)
    f = default_mdagg_df.copy(); f["section"] = "s2_mdagg_per_condition_default_combo"
    frames.append(f)
    out = pd.concat(frames, ignore_index=True, sort=False)
    DIAG_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(DIAG_CSV, index=False, encoding="utf-8-sig")
    print(f"[diagnostic-csv] wrote {DIAG_CSV} ({len(out)} rows)")
    return out


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main():
    args = sys.argv[1:]
    limit = int(args[args.index("--limit") + 1]) if "--limit" in args else None
    do_all = "--all" in args

    cfg, mt, d50 = load_common()
    shape_summary = pd.read_csv(SHAPE_SUMMARY) if SHAPE_SUMMARY.exists() else pd.DataFrame()

    count_df = count_corr_df = small_n_sec = None
    sweep_corr_df = pd.DataFrame()
    legacy_corr_df = pd.DataFrame()
    shape_corr_df = pd.DataFrame()

    if "--step1" in args or do_all:
        count_df, count_corr_df, small_n_sec = section1_counts(mt, shape_summary)

    if "--sweep" in args or do_all:
        run_sweep(limit)

    if "--sweep-report" in args or do_all:
        sweep_corr_df = sweep_correlations(mt, d50)

    if "--legacy" in args or do_all:
        legacy_corr_df = section3_legacy(mt, d50)

    if "--shape" in args or do_all:
        shape_corr_df = section4_shape(shape_summary)

    if "--overlays" in args or do_all:
        make_diagnostic_overlays()

    if "--figure" in args or do_all:
        if sweep_corr_df.empty:
            sweep_corr_df = sweep_correlations(mt, d50)
        if legacy_corr_df.empty:
            legacy_corr_df = section3_legacy(mt, d50)
        make_robustness_figure(sweep_corr_df, legacy_corr_df)

    if "--diagnostic-csv" in args or do_all:
        if count_df is None:
            count_df, count_corr_df, small_n_sec = section1_counts(mt, shape_summary)
        if sweep_corr_df.empty:
            sweep_corr_df = sweep_correlations(mt, d50)
        if legacy_corr_df.empty:
            legacy_corr_df = section3_legacy(mt, d50)
        if shape_corr_df.empty:
            shape_corr_df = section4_shape(shape_summary)
        default_mdagg_df, rho0, p0, n0 = compute_mdagg_theta(
            mt.set_index("sample_id")["D_sec"], mt, d50)
        print(f"[sanity] 生产默认(master_table.csv 原值)Spearman(M_D_agg,Theta)="
              f"{rho0:.3f} p={p0:.4g} n={n0}(论文报告 +0.558 p=0.0025)")
        write_diagnostic_csv(count_df, count_corr_df, sweep_corr_df, legacy_corr_df,
                             shape_corr_df, default_mdagg_df)

    if not args:
        print(__doc__)


if __name__ == "__main__":
    main()
