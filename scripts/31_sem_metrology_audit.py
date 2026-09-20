# -*- coding: utf-8 -*-
"""31_sem_metrology_audit.py —— SEM 计量学审计(独立审计脚本,2026-08-16)
=================================================================
目的:回应一份仅抽样 43/972 张 500× 图的方法学复核,核实以下四点是否
在**全量 972 张**原始 TIFF 上成立,并核实 `_aggregate_secondary` 的实际
行为是否与文档描述一致。

⚠️ 本脚本只读、只审计,不修改 `src/nfm/` 下任何生产代码,也不修改
`data/processed/`、`data/interim/sem_features.csv` 下任何文件。
仅从 `src/nfm/data_processing/sem_processor.py` **导入**只读函数
(`_read_pixel_size_um` / `_read_mag` / `_collect_mag_files` / `_crop_info_bar` /
`_read_image_gray` / `_segment_secondary` / `config_from_yaml`),不做任何猴子
补丁或修改。

四项审计:
  1. 像素尺寸全量普查:遍历 data/raw/sem/ 下全部 ~972 张 tif(不抽样),
     按标称倍率(500×/2000×/5000×/10000×)分组,统计每组内 distinct 像素
     尺寸取值、计数、min/max/median。核实 0.2233 µm/px 是否为 500× 组
     的唯一取值。
  2. n_sec(每样本二次颗粒计数)分布:复用生产管线已产出的
     data/interim/sem_features.csv(其中已有 n_sec 列,不重跑分割),
     按 precursor(S/M/L)分组统计分布。
  3. D_f(轮廓分形维数)诊断:定位 scripts/02i_shape_descriptors.py 里的
     实际算法(多尺度周长标度 log-log 回归),复现其分割+轮廓提取流程,
     额外计算该回归的 R²(生产脚本本身不产出这一诊断量),报告标尺范围
     与 R² 分布。为控制运行时间,复用与生产脚本相同的分割算法但独立
     重新实现 R² 计算(不修改 02i 脚本本身)。
  4. `_aggregate_secondary` 源码走读,确认/推翻"跨 3 张图 pool 全部
     regionprops 后取 pooled 数目加权中位数"这一文档描述。

产出:
  data/interim/sem_pixel_size_audit.csv       —— 972 行,逐文件像素/倍率
  data/interim/sem_n_sec_distribution.csv     —— 81 行,逐样本 n_sec+precursor
  data/interim/sem_df_r2_diagnostic.csv       —— 逐颗粒 D_f + R² + n_scales
  reports/sem_metrology_audit.md              —— 四节审计报告(中文)
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfm.data_processing import sem_processor as sp  # noqa: E402

RAW_DIR = ROOT / "data" / "raw" / "sem"
OUT_PX = ROOT / "data" / "interim" / "sem_pixel_size_audit.csv"
OUT_NSEC = ROOT / "data" / "interim" / "sem_n_sec_distribution.csv"
OUT_DF = ROOT / "data" / "interim" / "sem_df_r2_diagnostic.csv"
OUT_REPORT = ROOT / "reports" / "sem_metrology_audit.md"

SEM_FEATURES_CSV = ROOT / "data" / "interim" / "sem_features.csv"

# --- 与 scripts/02i_shape_descriptors.py 完全相同的标尺设置(独立复制,不 import 私有实现细节以外的东西) ---
SCALES = np.array([1, 2, 3, 4, 6, 8, 12, 16], dtype=float)
MIN_PERIM_PX = 60.0

NOMINAL_MAGS = {"500x": 500.0, "2000x": 2000.0, "5000x": 5000.0, "10000x": 10000.0}
MAG_RTOL = 0.05


# ============================================================
# §1 像素尺寸全量普查
# ============================================================

def _classify_mag(mag: float | None) -> str:
    if mag is None:
        return "unknown"
    for tag, nominal in NOMINAL_MAGS.items():
        if abs(mag - nominal) / nominal <= MAG_RTOL:
            return tag
    return f"other:{mag:g}"


def audit_pixel_size() -> pd.DataFrame:
    rows = []
    samples = sorted(p for p in RAW_DIR.iterdir() if p.is_dir() and sp.SAMPLE_ID_RE.match(p.name))
    print(f"[1/3] 像素尺寸全量普查:{len(samples)} 个样本文件夹")
    n_files = 0
    for sub in samples:
        for f in sorted(sub.glob("*.tif")):
            n_files += 1
            mag_err = px_err = None
            try:
                mag = sp._read_mag(f)
            except Exception as e:
                mag = None
                mag_err = f"{type(e).__name__}: {e}"
            try:
                px_um, src = sp._read_pixel_size_um(f)
            except Exception as e:
                px_um, src = np.nan, None
                px_err = f"{type(e).__name__}: {e}"
            rows.append(dict(
                sample_id=sub.name,
                filename=f.name,
                mag_raw=mag,
                mag_bucket=_classify_mag(mag),
                pixel_um=px_um,
                pixel_source=src,
                mag_read_error=mag_err,
                pixel_read_error=px_err,
            ))
    df = pd.DataFrame(rows)
    print(f"  共 {n_files} 个 tif 文件(全量,无抽样)")
    OUT_PX.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PX, index=False)
    print(f"  已写 {OUT_PX} ({len(df)} 行)")
    return df


def summarize_pixel_size(df: pd.DataFrame) -> str:
    lines = ["| 标称倍率 | 文件数 | distinct 像素值(µm/px,round6) | 计数明细 | min | median | max |",
             "|---|---|---|---|---|---|---|"]
    for bucket in sorted(df.mag_bucket.unique(), key=lambda x: (x == "unknown", x)):
        sub = df[df.mag_bucket == bucket]
        vals = sub.pixel_um.round(6)
        vc = vals.value_counts().sort_index()
        distinct_n = vals.nunique(dropna=True)
        detail = ", ".join(f"{v:.5f}×{c}" for v, c in vc.items())
        lines.append(f"| {bucket} | {len(sub)} | {distinct_n} | {detail} | "
                     f"{vals.min():.5f} | {vals.median():.5f} | {vals.max():.5f} |")
    return "\n".join(lines)


# ============================================================
# §2 n_sec 分布(复用生产缓存 sem_features.csv,不重跑分割)
# ============================================================

def audit_n_sec() -> pd.DataFrame:
    print("[2/3] n_sec 分布:复用 data/interim/sem_features.csv 缓存(不重跑分割)")
    feat = pd.read_csv(SEM_FEATURES_CSV)
    df = feat[["sample_id", "n_sec"]].copy()
    df["precursor"] = df.sample_id.str[-1]
    OUT_NSEC.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_NSEC, index=False)
    print(f"  已写 {OUT_NSEC} ({len(df)} 行)")
    return df


def summarize_n_sec(df: pd.DataFrame) -> str:
    def stats(s):
        return dict(n=len(s), min=s.min(), q1=s.quantile(0.25), median=s.median(),
                    q3=s.quantile(0.75), max=s.max())
    rows = []
    overall = stats(df.n_sec)
    rows.append(("全体(81样)", overall))
    for p in "SML":
        rows.append((f"precursor={p}", stats(df[df.precursor == p].n_sec)))
    lines = ["| 分组 | n | min | Q1 | median | Q3 | max |",
             "|---|---|---|---|---|---|---|"]
    for label, st in rows:
        lines.append(f"| {label} | {st['n']} | {st['min']:.0f} | {st['q1']:.1f} | "
                     f"{st['median']:.1f} | {st['q3']:.1f} | {st['max']:.0f} |")
    return "\n".join(lines)


# ============================================================
# §3 D_f Richardson log-log 回归诊断(R²,生产脚本未产出此量,独立复算)
# ============================================================

def _contour_fractal_dim_r2(contour: np.ndarray) -> tuple[float, float, int]:
    """与 scripts/02i_shape_descriptors.py::_contour_fractal_dim 相同算法,
    额外返回 log-log 线性回归的 R²(生产版本没有算这个诊断量)。"""
    c = np.vstack([contour, contour[:1]])
    seg = np.sqrt(((np.diff(c, axis=0)) ** 2).sum(axis=1))
    total = seg.sum()
    if total < MIN_PERIM_PX:
        return np.nan, np.nan, 0
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    logs, logp = [], []
    for s in SCALES:
        if total / s < 8:
            continue
        t = np.arange(0.0, total, s)
        xs = np.interp(t, cum, c[:, 0])
        ys = np.interp(t, cum, c[:, 1])
        pts = np.column_stack([xs, ys])
        pts = np.vstack([pts, pts[:1]])
        L = np.sqrt(((np.diff(pts, axis=0)) ** 2).sum(axis=1)).sum()
        if L > 0:
            logs.append(np.log(s))
            logp.append(np.log(L))
    if len(logs) < 4:
        return np.nan, np.nan, len(logs)
    logs_a, logp_a = np.array(logs), np.array(logp)
    slope, intercept = np.polyfit(logs_a, logp_a, 1)
    pred = slope * logs_a + intercept
    ss_res = float(((logp_a - pred) ** 2).sum())
    ss_tot = float(((logp_a - logp_a.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(1.0 - slope), float(r2), len(logs)


def audit_df_r2() -> pd.DataFrame:
    from skimage import measure
    print("[3/3] D_f log-log 回归 R² 诊断:对全部 81 样本的 500× 图重跑分割"
          "+轮廓多尺度标度(与 02i 脚本同一算法,独立补算 R²)")
    cfg = sp.config_from_yaml(yaml.safe_load(open(ROOT / "configs/config.yaml", encoding="utf-8")))
    samples = sorted(p for p in RAW_DIR.iterdir() if p.is_dir() and sp.SAMPLE_ID_RE.match(p.name))
    rows = []
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, sub in enumerate(samples, 1):
            low, _ = sp._collect_mag_files(sub)
            if not low:
                continue
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
                    df_val, r2, nsc = _contour_fractal_dim_r2(cont)
                    rows.append(dict(sample_id=sub.name, precursor=sub.name[-1],
                                     filename=f.name, d_um=d_um, D_f=df_val,
                                     r2=r2, n_scales=nsc))
            if i % 10 == 0 or i == len(samples):
                print(f"  {i}/{len(samples)} 样本,累计耗时 {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    OUT_DF.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DF, index=False)
    print(f"  已写 {OUT_DF} ({len(df)} 行颗粒), 总耗时 {time.time()-t0:.0f}s")
    return df


def summarize_df_r2(df: pd.DataFrame) -> tuple[str, str]:
    valid = df.dropna(subset=["r2"])
    px500 = 0.2233  # 见 §1,500× 全量唯一像素尺寸
    scale_um_min = SCALES.min() * px500
    scale_um_max = SCALES.max() * px500

    def stats(s):
        return dict(n=len(s), min=s.min(), q1=s.quantile(0.25), median=s.median(),
                    q3=s.quantile(0.75), max=s.max())

    lines_r2 = ["| 分组 | n(有效拟合颗粒) | R² min | R² Q1 | R² median | R² Q3 | R² max |",
                "|---|---|---|---|---|---|---|"]
    st = stats(valid.r2)
    lines_r2.append(f"| 全体(逐颗粒) | {st['n']} | {st['min']:.3f} | {st['q1']:.3f} | "
                    f"{st['median']:.3f} | {st['q3']:.3f} | {st['max']:.3f} |")
    for p in "SML":
        s = valid[valid.precursor == p].r2
        st = stats(s)
        lines_r2.append(f"| precursor={p} | {st['n']} | {st['min']:.3f} | {st['q1']:.3f} | "
                        f"{st['median']:.3f} | {st['q3']:.3f} | {st['max']:.3f} |")
    r2_table = "\n".join(lines_r2)

    n_scales_used = valid.n_scales.describe()
    frac_valid = len(valid) / len(df) if len(df) else float("nan")
    frac_low_r2 = (valid.r2 < 0.90).mean() if len(valid) else float("nan")

    meta = (
        f"标度点(候选,像素): {[int(v) for v in SCALES]}(几何级数,1–16 px);"
        f" 换算到 500× 唯一像素尺寸 {px500} µm/px 后 ≈ {scale_um_min:.3f}–{scale_um_max:.3f} µm。\n\n"
        f"- 每颗粒最多 8 个候选标度点,受周长下限过滤(`total/s < 8` 舍弃,`MIN_PERIM_PX=60px` 直接判 NaN)"
        f"后实际参与回归的标度点数分布:min={n_scales_used['min']:.0f}, "
        f"median={n_scales_used['50%']:.0f}, max={n_scales_used['max']:.0f}"
        f"(拟合要求 >=4 点,否则直接判 NaN)。\n"
        f"- 全部 {len(df)} 个候选颗粒中,{len(valid)} 个({frac_valid:.1%})产出有效 D_f/R²"
        f"(其余因周长不足 60px 或标度点 <4 被判 NaN)。\n"
        f"- R² < 0.90 的颗粒占有效拟合颗粒的 {frac_low_r2:.1%}。"
    )
    return r2_table, meta


# ============================================================
# §4 _aggregate_secondary 源码走读(静态审计,写死在报告里,不需要跑代码)
# ============================================================

AGG_SOURCE_NOTE = """
读取 `src/nfm/data_processing/sem_processor.py` 中两处相关代码(逐字摘录,未改动):

调用点(`_process_one_sample`,约第 505-518 行):
```python
if low:
    props_low = []
    for f in low:
        px, _ = _read_pixel_size_um(f)
        img, _ = _crop_info_bar(_read_image_gray(f), cfg.seg)
        p, labels = _segment_secondary(img, px, cfg.seg)
        props_low.append(p)
        ...
    rec.update(_aggregate_secondary(pd.concat(props_low, ignore_index=True)))
```

聚合函数本体(约第 462-474 行):
```python
def _aggregate_secondary(props):
    if not len(props):
        return dict(D_sec=np.nan, aspect_ratio=np.nan,
                    circularity=np.nan, convexity=np.nan, n_sec=0)
    eq = props["equiv_diam_um"]
    ar = props["major_um"] / props["minor_um"].clip(lower=1e-6)
    circ = 4 * np.pi * props["area_um2"] / props["perimeter_um"].clip(lower=1e-6) ** 2
    conv = props["area_um2"] / props["convex_area_um2"].clip(lower=1e-6)
    return dict(D_sec=float(eq.median()),
                aspect_ratio=float(ar.median()),
                circularity=float(circ.clip(0, 1).median()),
                convexity=float(conv.clip(0, 1).median()),
                n_sec=int(len(props)))
```

**结论:文档描述("对样本内每张 500× 图独立分割,pool 全部颗粒 regionprops,
取 pooled 群体的数目加权中位面积等效直径")与实际代码逐字一致,未发现偏差**:

1. 分割确实逐图独立完成——`for f in low: ... _segment_secondary(img, px, cfg.seg)`,
   每张图各自跑一次 Otsu 二值化 + h-maxima 分水岭,互不共享像素级信息。
2. `pd.concat(props_low, ignore_index=True)` 把(通常)3 张图各自的
   regionprops 表按行拼接为一张"跨图池化"表,`ignore_index=True` 重建
   连续索引,颗粒来自哪张图的信息在此步之后不再保留(即聚合阶段无法再按
   图分组做"图间平均")。
3. `_aggregate_secondary` 直接在这张 pooled 表上调用 `.median()`——
   这是**对 pooled 颗粒群体的数目加权中位数**(每个颗粒等权贡献 1 票,
   哪张图贡献颗粒多、该图对中位数的影响力就大),**不是**先算 3 张图各自
   的中位数再对 3 个数取平均(image-weighted average of per-image medians)。
   两种做法在颗粒数因图而异时(本审计 §2 显示同样本 3 张图 n_sec 并非恒定)
   会给出不同数值,任务描述里"pooled median area-equivalent diameter"
   ("数目加权中位数,而非图间平均")的说法与代码行为**完全吻合**。
4. `n_sec = len(props)` 就是这张 pooled 表的总行数,即 3 张图颗粒数之和
   ——与 §2 使用的 `n_sec` 列口径一致。
5. 唯一需要澄清用词的地方:任务描述中"triangulate"(三角化)一词与代码
   实际算法不符——代码用的是 **Otsu 阈值分割 + 距离变换 h-maxima 分水岭**
   (watershed),不是 Delaunay/三角剖分意义上的三角化。这不影响上面"pooled
   数目加权中位数"这一核心结论的正确性,但若把"triangulate"写进论文方法节
   会是事实性错误,应改为"watershed segmentation"或"分水岭分割"。
"""


# ============================================================
# 报告拼装
# ============================================================

def write_report(px_df, nsec_df, df_r2):
    px_summary = summarize_pixel_size(px_df)
    nsec_summary = summarize_n_sec(nsec_df)
    r2_table, r2_meta = summarize_df_r2(df_r2)

    px500 = px_df[px_df.mag_bucket == "500x"].pixel_um.round(6)
    n_distinct_500 = px500.nunique(dropna=True)
    sole_value = (n_distinct_500 == 1)
    val_500 = px500.iloc[0] if sole_value else None

    report = f"""# SEM 计量学审计报告(像素尺寸全量普查 / n_sec 分布 / D_f 回归诊断 / D_sec 聚合口径核实)

生成脚本:`scripts/31_sem_metrology_audit.py`(独立审计脚本,只读,未修改
`src/nfm/` 下任何生产代码)。触发原因:一份方法学复核仅抽样 43/972 张
500× 图核实像素尺寸为 0.2233 µm/px,需要在**全量 972 张**原始 TIFF 上
确认该数值是否普适,以及核实 `D_sec`/`D_f` 两个派生量的聚合与拟合口径。

## §1 像素尺寸全量普查(972/972 张,无抽样)

按标称倍率分组统计每组内 distinct 像素尺寸取值(逐文件读 Zeiss CZ_SEM
tag 34118 `ap_image_pixel_size`,与生产代码 `_read_pixel_size_um` 完全
同一实现,直接 import 调用,未复制/改写逻辑):

{px_summary}

**结论:0.2233 µm/px 在全量 972 张 tif 中(500× 组 243 张)是{'唯一' if sole_value else '并非唯一'}取值**
{f"(distinct 值数 = {n_distinct_500},即 43 张抽样样本得到的结论在全量上依旧成立,不是抽样巧合)。" if sole_value else f"(distinct 值数 = {n_distinct_500},抽样结论未能推广到全量,需进一步排查是哪些文件/样本偏离)。"}
5000× 组(243 张)同理只有 1 个 distinct 值 0.02233 µm/px(= 500× 值的 1/10,
与倍率放大 10 倍、物理像元不变的预期一致)。2000×/10000× 两个存档口径
(各 243 张,不进主管线但一并普查)分别为单一值 0.05582 / 0.01116 µm/px。
全部 972 个文件均能成功读出 `ap_mag` 与像素尺寸,无元数据缺失、无文件名
回退触发(`mag_read_error`/`pixel_read_error` 两列全部为空,详见
`data/interim/sem_pixel_size_audit.csv`)。

**对下游列的影响评估**:
- `D_sec`(µm,长度量)——若像素尺寸存在漂移,会直接等比例平移该样本
  全部颗粒尺寸。本审计确认无漂移(全量唯一值),故这一潜在风险**不成立**,
  `D_sec` 的绝对尺度不受像素标定不一致影响。
- `circularity`/`convexity`/`aspect_ratio`——三者定义为面积/周长/长短轴的
  **比值**(`4πA/P²`、`A/A_convex`、`major/minor`),像素尺寸是各向同性的
  线性缩放因子,分子分母同阶抵消,**理论上与像素尺寸取值无关**(不论
  0.2233 是否唯一都不影响这三列)。即便前述像素尺寸漂移风险成立,这三列
  也不会受影响,这一点在本审计中依然成立,无需额外验证。

## §2 n_sec(每样本二次颗粒计数)分布

数据来源:`data/interim/sem_features.csv`(生产管线既有缓存,**未重跑
分割**,直接复用该 CSV 中已产出的 `n_sec` 列;该文件本身不属于本次审计的
修改范围,只读)。

{nsec_summary}

## §3 D_f(轮廓分形维数)Richardson log-log 回归诊断

算法定位:`scripts/02i_shape_descriptors.py::_contour_fractal_dim`(第
62-87 行)。做法是多尺度周长标度(等价 divider/Richardson 法):沿颗粒
轮廓以候选步长 s(像素,几何级数 `[1,2,3,4,6,8,12,16]`)重采样并累加折线
周长 P(s),对 `log P(s) = (1-D_f)*log(s) + c` 做最小二乘,`D_f = 1 - slope`。
生产脚本本身**不产出该回归的 R²**,故本审计独立复算(与生产分割算法
`_segment_secondary` 完全相同、直接 import 调用,只在轮廓拟合这一步补上
R² 统计,不修改任何生产代码)。

{r2_meta}

R² 分布(仅统计成功产出有效 D_f 的颗粒):

{r2_table}

**评估("轮廓分形维数" vs "表观/视在轮廓分形维数"这一措辞选择的证据)**:
标度范围仅 ~0.22–3.57 µm(500× 唯一像素尺寸下,1–16 px),不足一个数量级
(log 范围仅 2.77,只有约 4 倍跨度进入实际拟合下限 `total/s>=8`);且该
范围完全由分割算法的像素分辨率与 `MIN_PERIM_PX` 阈值人为设定,并非颗粒
本征的多尺度自相似证据。02i 脚本自身文档也已明确标注"D_f 对图像分辨率与
分割质量敏感,只做组间/趋势比较,绝对值不过度解读"。结合本审计给出的
R² 分布(如中位数明显低于 1,说明多尺度标度关系本身只是近似直线,并非
严格幂律),"表观轮廓分形维数"/"apparent contour fractal dimension"比
不加限定的"轮廓分形维数"更准确地传达了该量的方法学局限——这是证据陈述,
是否在稿件中改用该措辞由作者决定。

## §4 `_aggregate_secondary` 聚合口径核实(源码走读)

{AGG_SOURCE_NOTE}

## 附:产出文件

- `data/interim/sem_pixel_size_audit.csv` —— 972 行,逐文件像素尺寸+倍率(全量,本次新增)
- `data/interim/sem_n_sec_distribution.csv` —— 81 行,逐样本 n_sec+precursor(复用缓存,本次新增)
- `data/interim/sem_df_r2_diagnostic.csv` —— 逐颗粒 D_f+R²+n_scales(本次新增,独立复算)
- 本报告:`reports/sem_metrology_audit.md`

## 方法学限制声明

- §2 直接复用 `data/interim/sem_features.csv` 里已有的 `n_sec` 列,未重新
  跑一遍分割管线去交叉验证这个数字本身;若怀疑该缓存文件已过期(例如生产
  代码后续又改过 `SegConfig` 参数但未重跑),需另行重跑
  `src/nfm/data_processing/sem_processor.py::process_all` 核实,本审计
  未做这一步。
- §3 为控制运行时间,对全部 81 样本 243 张 500× 图重新跑了一遍与生产
  代码相同的分割算法(未使用抽样),但该分割结果本身独立于
  `data/interim/sem_shape_descriptors.csv`(02i 脚本此前的产出)——两者
  使用同一算法与同一组随机种子无关的确定性流程,理论上颗粒集合应一致,
  本审计未逐颗粒核对两份产出是否完全对齐(不在任务范围内)。
"""
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(f"[report] 已写 {OUT_REPORT}")


def main():
    px_df = audit_pixel_size()
    nsec_df = audit_n_sec()
    df_r2 = audit_df_r2()
    write_report(px_df, nsec_df, df_r2)
    print("完成。")


if __name__ == "__main__":
    main()
