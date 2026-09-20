# -*- coding: utf-8 -*-
"""
sem_processor.py —— SEM 处理器(模块二·处理器 2,v0.2 完整实现)
=================================================================
双口径策略(已定案,方案①固定倍率):
  - 低倍 500× → 二次颗粒 : D_sec / aspect_ratio / circularity(4πA/P²) / convexity(A/A_convex)
  - 高倍 5000× → 一次颗粒: D_pri_sem

一次颗粒处理走"甲-2"路线:
  - 先用【纹理可分割性判据】(不依赖分割)判断该样一次颗粒是
    "离散大单晶(可分割)" 还是 "密堆纳米晶纹理(不可逐颗粒测)";
  - 可分割 → 用 SAM(Segment Anything)做高精度实例分割,regionprops 出 D_pri_sem;
  - 不可测 → D_pri_sem = NaN,primary_segmentable=False,n_pri=0(PINN 端用 mask loss 跳过)。
  物理依据:高Θ的S烧成µm级大单晶(可分割);低Θ的L是亚微米纳米棒密堆成球(SEM下不可逐个分开)。
  "L类不可测"这一事实本身即支持"形貌记忆 vs 热重构"命题,故如实标记而非强行测量。

像素标定:读 Zeiss CZ_SEM 元数据(tag 34118)的 ap_image_pixel_size,排除人工标尺误差。
信息栏:自动检测底部近黑分隔行并裁掉(Zeiss 底部 10µm/Mag/EHT/WD/ZEISS 信息条)。

输入 : data/raw/sem/<sample>/*.tif
       倍率识别以 Zeiss CZ_SEM 元数据(tag 34118 的 ap_mag)为准,文件名无需含倍率;
       元数据缺失时回退文件名匹配(*_500x_*, *_5000x_*;兼容旧命名)。
       同一文件夹内不得混入其他样品的图(元数据不含样品编号,归属仅由文件夹决定)。
输出 : data/interim/sem_features.csv  每样本一行(列对齐 schema.py)

依赖:
  必需 : numpy pandas tifffile scikit-image scipy
  高倍 : torch + segment-anything + SAM 权重(本地下载,见 SAMConfig)
         若 SAM 不可用,可分割样本回退到传统分割(精度较低,会在日志告警)。
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage import (color, exposure, feature, filters, measure,
                     morphology, segmentation)
from scipy import ndimage as ndi

warnings.filterwarnings("ignore", message=".*min_size.*deprecated.*", category=FutureWarning)
SAMPLE_ID_RE = re.compile(r"^(C\d{2}-[SML])$")
_UNIT_TO_UM = {"pm": 1e-6, "nm": 1e-3, "um": 1.0, "µm": 1.0, "mm": 1e3}

LOW_MAG_GLOB = "*_500x_*.tif"     # 二次颗粒(仅作元数据缺失时的文件名回退)
HIGH_MAG_GLOB = "*_5000x_*.tif"   # 一次颗粒(同上)
LOW_MAG = 500.0                   # 二次颗粒口径
HIGH_MAG = 5000.0                 # 一次颗粒口径
MAG_RTOL = 0.05                   # 倍率归类相对容差
N_EXPECTED_PER_MAG = 3            # 拍摄协议:每倍率 3 张


@dataclass
class SegConfig:
    sec_min_diam_um: float = 1.0
    sec_gauss_sigma: float = 1.5
    sec_hmaxima_rel: float = 0.25
    sec_open_disk: int = 2
    sec_clear_border: bool = True   # 500× 二次颗粒:剔除触边颗粒(见 _regionprops_um docstring)
    # 形貌自适应 h(默认关闭,保证可复现旧行为):
    # sec_hmaxima_rel 不再是全库固定一个值,而是按逐图特征尺度 dt_scale 连续调整,
    # 见 _particle_scale_um / _map_h_from_scale。校准锚点为目视初值,已定档为
    # config.yaml 的 instruments.sem.seg 默认参数(见该配置块注释)。
    sec_adaptive_h: bool = False
    sec_adaptive_h_percentile: float = 90.0  # dt_scale 取 mask 内 dt 的稳健高分位
    sec_adaptive_h_x0: float = 1.6           # µm,锚点 dt_scale(S 型目视校准)
    sec_adaptive_h_intercept: float = 0.16   # 锚点处的 h
    sec_adaptive_h_slope: float = 0.082      # h 每 µm 的变化率
    sec_adaptive_h_min: float = 0.14
    sec_adaptive_h_max: float = 0.26
    pri_min_diam_um: float = 0.3
    judge_acf_scale_um_min: float = 0.30
    judge_edge_density_max: float = 0.115
    judge_require_both: bool = True
    infobar_dark_thresh: float = 5.0
    infobar_min_keep_frac: float = 0.6
    infobar_sep_max_rows: int = 8        # 近黑"分隔线"的最大厚度,超过且不贴底边则判为图像暗区
    infobar_warn_removed_frac: float = 0.35  # 切除行数占比超此值告警


@dataclass
class SAMConfig:
    enabled: bool = True
    checkpoint: str = "src/nfm/models/sam_vit_b_01ec64.pth"
    model_type: str = "vit_b"
    device: str = "cuda"
    points_per_side: int = 16
    pred_iou_thresh: float = 0.86
    stability_score_thresh: float = 0.90
    min_mask_region_area_px: int = 64


@dataclass
class SemProcessorConfig:
    seg: SegConfig = field(default_factory=SegConfig)
    sam: SAMConfig = field(default_factory=SAMConfig)
    save_overlays: bool = True
    overlay_dir: str = "data/interim/sem_overlays"


def _read_pixel_size_um(tif_path: Path) -> tuple[float, str]:
    """读取像素物理尺寸(µm/px)。优先 Zeiss CZ_SEM,退化 ap_width/图宽。"""
    with tifffile.TiffFile(tif_path) as t:
        page = t.pages[0]
        tags = page.tags
        cz = tags[34118].value if 34118 in tags else None
        if cz:
            for key in ("ap_image_pixel_size", "ap_pixel_size"):
                if key in cz:
                    entry = cz[key]
                    val, unit = float(entry[1]), str(entry[2]).strip()
                    factor = _UNIT_TO_UM.get(unit)
                    if factor:
                        return val * factor, f"meta:{key}({val}{unit})"
            if "ap_width" in cz:
                w = cz["ap_width"]
                factor = _UNIT_TO_UM.get(str(w[2]).strip())
                if factor:
                    return (float(w[1]) * factor) / int(page.imagewidth), \
                           f"meta:ap_width/{page.imagewidth}px"
    raise ValueError(f"无法从元数据标定像素:{tif_path.name}")


def _parse_mag(raw) -> float | None:
    """解析 CZ_SEM ap_mag 值:'500 X' / '10.00 K X' / 数值。失败返回 None。"""
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).upper().replace("X", "").strip()
    try:
        if "K" in s:
            return float(s.replace("K", "").strip()) * 1000.0
        return float(s)
    except ValueError:
        return None


def _read_mag(tif_path: Path) -> float | None:
    """从 CZ_SEM 元数据(tag 34118 的 ap_mag)读拍摄倍率。缺失/无法解析返回 None。"""
    with tifffile.TiffFile(tif_path) as t:
        tags = t.pages[0].tags
        cz = tags[34118].value if 34118 in tags else None
        if cz and "ap_mag" in cz:
            entry = cz["ap_mag"]
            raw = entry[1] if isinstance(entry, (tuple, list)) and len(entry) > 1 else entry
            return _parse_mag(raw)
    return None


def _collect_mag_files(sub: Path) -> tuple[list[Path], list[Path]]:
    """按元数据倍率把样本文件夹内 tif 分为 (500× 低倍, 5000× 高倍) 两组。

    - 首选 ap_mag 元数据;单文件读取失败时回退文件名(*_500x_*/*_5000x_*,兼容旧命名)。
    - 交叉校验:同一样本内 mag × pixel_size(视场基准)应为常数,偏差 >2% 告警。
    - 每口径张数 ≠ N_EXPECTED_PER_MAG 时告警(不拦截,允许补拍/缺图)。
    """
    low, high, field_refs = [], [], []
    for f in sorted(sub.glob("*.tif")):
        mag = None
        try:
            mag = _read_mag(f)
        except Exception as e:  # tif 损坏等
            warnings.warn(f"{sub.name}/{f.name}: 读元数据失败({e}),回退文件名判断")
        if mag is None:  # 文件名回退(旧命名兼容)
            name = f.name.lower()
            if "_500x_" in name:
                mag = LOW_MAG
            elif "_5000x_" in name:
                mag = HIGH_MAG
            else:
                continue  # 2000×/10000× 等存档口径:静默跳过与旧 glob 行为一致
        else:
            try:  # 视场基准交叉校验(mag × px 应为仪器常数)
                px_um, _ = _read_pixel_size_um(f)
                field_refs.append((f.name, mag * px_um))
            except Exception:
                pass
        if abs(mag - LOW_MAG) / LOW_MAG <= MAG_RTOL:
            low.append(f)
        elif abs(mag - HIGH_MAG) / HIGH_MAG <= MAG_RTOL:
            high.append(f)
        # 其余倍率(2000×/10000× 存档)不入库
    if field_refs:
        vals = np.array([v for _, v in field_refs])
        if vals.max() / vals.min() > 1.02:
            warnings.warn(f"{sub.name}: 倍率×像素尺寸不自洽(极差>2%),"
                          f"疑似元数据异常: {field_refs}")
    for tag_, files in (("500×", low), ("5000×", high)):
        if files and len(files) != N_EXPECTED_PER_MAG:
            warnings.warn(f"{sub.name}: {tag_} 共 {len(files)} 张"
                          f"(协议应为 {N_EXPECTED_PER_MAG} 张),请核对")
    return low, high


def _read_image_gray(tif_path: Path) -> np.ndarray:
    arr = tifffile.imread(str(tif_path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8)


def _crop_info_bar(img: np.ndarray, cfg: SegConfig) -> tuple[np.ndarray, int]:
    """裁掉底部 Zeiss 信息栏,返回 (图像区, 切点行号)。

    背景(2026-07-22 修复):旧实现假设信息栏本身是"近黑带",从底部向上找到
    第一条近黑行就 break。但本批 Zeiss 图的信息栏是**亮条**(行均值 230-242),
    上下各由一条近黑分隔行(行 690 / 766)夹住 —— 旧逻辑在行 766 命中即停,
    只裁掉 2 行,把 75 行白条留在了分析区。后果不是"多了一个伪颗粒"这么简单:
    白条的距离变换峰值(~76px)远大于任何真实颗粒,劫持了
    `h = sec_hmaxima_rel * dist.max()` 的全局归一化基准,使 h 虚高约一倍,
    小颗粒样本(S 型)的分水岭标记被整片抹除 —— 这正是 D_sec 出现
    "S 型反而最大"这一物理矛盾的根因。

    新策略(分辨率无关,亮/暗两种信息栏通吃):
      1. 在底部区域内找出全部近黑行,合并为连续暗带;
      2. 只接受两类暗带:细分隔线(厚度 <= infobar_sep_max_rows),
         或紧贴图像底边的厚暗带(即信息栏本身为暗条的情形);
         夹在中间的厚暗带判为图像内容(真实暗区),不接受;
      3. 取**最靠上**的合格暗带起始行作为切点。
    """
    rowmean = img.mean(axis=1)
    H = img.shape[0]
    lo = int(H * cfg.infobar_min_keep_frac)

    dark_rows = [r for r in range(lo, H) if rowmean[r] < cfg.infobar_dark_thresh]
    if not dark_rows:
        warnings.warn(
            f"_crop_info_bar: 底部 {H - lo} 行内未找到近黑分隔行"
            f"(阈值 {cfg.infobar_dark_thresh}),未做裁剪 —— 若该图含信息栏,"
            f"后续分割结果不可信,请核对图像来源"
        )
        return img, H

    # 合并为连续暗带 [(start, end_inclusive), ...]
    bands: list[tuple[int, int]] = []
    start = prev = dark_rows[0]
    for r in dark_rows[1:]:
        if r == prev + 1:
            prev = r
        else:
            bands.append((start, prev))
            start = prev = r
    bands.append((start, prev))

    for b_start, b_end in bands:  # bands 已按行号升序,首个合格者即最靠上
        thickness = b_end - b_start + 1
        if thickness <= cfg.infobar_sep_max_rows or b_end >= H - 1:
            cut = b_start
            break
    else:
        warnings.warn(
            "_crop_info_bar: 底部暗带均不符合分隔线/底边暗条特征"
            "(疑为图像内真实暗区),未做裁剪"
        )
        return img, H

    removed_frac = (H - cut) / H
    if removed_frac > cfg.infobar_warn_removed_frac:
        warnings.warn(
            f"_crop_info_bar: 切除底部 {H - cut} 行(占图高 {removed_frac:.1%}),"
            f"超过预期上限 {cfg.infobar_warn_removed_frac:.0%},请核对该图是否异常"
        )
    return img[:cut, :], cut


def _particle_scale_um(mask: np.ndarray, dt: np.ndarray, pixel_um: float,
                       percentile: float = 90.0) -> float:
    """标签无关、h 无关的特征尺度描述子:前景距离变换 dt 在 mask 内的稳健高分位
    (默认 90 分位)x pixel_um,≈ 颗粒内切半径。不做峰检测、不自我引用 h,
    只吃 Otsu mask 与像素尺寸,用于驱动 sec_hmaxima_rel 的连续自适应
    (见 SegConfig.sec_adaptive_h / _map_h_from_scale)。

    对粘连颗粒鲁棒:哑铃形粘连区 dt 仍在两个颗粒中心各自成峰,取高分位不受
    颈缩处低 dt 值拖累。mask 为空(无前景)时返回 NaN。
    """
    fg = dt[mask]
    if fg.size == 0:
        return float("nan")
    return float(np.percentile(fg, percentile) * pixel_um)


def _map_h_from_scale(dt_scale_um: float, cfg: "SegConfig") -> float:
    """dt_scale(µm)→sec_hmaxima_rel 的线性映射,系数见 SegConfig.sec_adaptive_h_*。
    起始校准锚点来自目视(S≈1.6µm→h≈0.16,L≈2.6µm→h≈0.25),已定档为默认参数。
    dt_scale 不可用(NaN,如空 mask)时退回固定的 cfg.sec_hmaxima_rel,不做外推。
    """
    if not np.isfinite(dt_scale_um):
        return cfg.sec_hmaxima_rel
    h = cfg.sec_adaptive_h_intercept + cfg.sec_adaptive_h_slope * (dt_scale_um - cfg.sec_adaptive_h_x0)
    return float(np.clip(h, cfg.sec_adaptive_h_min, cfg.sec_adaptive_h_max))


def _segment_secondary(img, pixel_um, cfg):
    img_eq = exposure.equalize_adapthist(img, clip_limit=0.02)
    img_s = filters.gaussian(img_eq, sigma=cfg.sec_gauss_sigma)
    mask = img_s > filters.threshold_otsu(img_s)
    min_area_px = (np.pi * (cfg.sec_min_diam_um / 2) ** 2) / (pixel_um ** 2)
    # area_threshold 为 skimage>=0.16 的正式参数名;旧别名 max_size 已在 0.24 移除
    mask = morphology.remove_small_holes(mask, area_threshold=int(min_area_px))
    mask = morphology.remove_small_objects(mask, min_size=int(min_area_px * 0.5))
    mask = morphology.opening(mask, morphology.disk(cfg.sec_open_disk))
    dist_raw = ndi.distance_transform_edt(mask)
    if cfg.sec_adaptive_h:
        dt_scale = _particle_scale_um(mask, dist_raw, pixel_um, cfg.sec_adaptive_h_percentile)
        hmaxima_rel = _map_h_from_scale(dt_scale, cfg)
    else:
        hmaxima_rel = cfg.sec_hmaxima_rel
    dist = filters.gaussian(dist_raw, sigma=1.0)
    h = hmaxima_rel * dist.max()
    hmax = morphology.h_maxima(dist, h)
    markers, _ = ndi.label(hmax)
    if markers.max() == 0:
        markers, _ = ndi.label(mask)
    labels = segmentation.watershed(-dist, markers, mask=mask, compactness=0.01)
    # 一次颗粒线(5000×)暂不启用 clear_border:该线的可分割性阈值正在重标定,
    # 同时改动两处会让两个效应混在一起无法归因。待阈值定稿后再统一处理。
    return _regionprops_um(labels, pixel_um, cfg.sec_min_diam_um,
                           clear_border=cfg.sec_clear_border), labels


def _radial_acf_scale_um(img_eq, pixel_um):
    f = img_eq.astype(float) - img_eq.mean()
    F = np.fft.fft2(f)
    acf = np.fft.fftshift(np.fft.ifft2(F * np.conj(F)).real)
    acf /= acf.max() + 1e-12
    cy, cx = np.array(acf.shape) // 2
    yy, xx = np.indices(acf.shape)
    r = np.hypot(yy - cy, xx - cx).astype(int)
    maxr = int(min(cy, cx))
    radial = ndi.mean(acf, labels=r, index=np.arange(maxr))
    below = np.where(radial < 0.5)[0]
    lag_px = below[0] if below.size else maxr
    return float(lag_px * pixel_um)


def _texture_metrics(img, pixel_um):
    img_eq = exposure.equalize_adapthist(img, clip_limit=0.02)
    edge_density = float(feature.canny(img_eq, sigma=2.0).mean())
    acf_scale_um = _radial_acf_scale_um(img_eq, pixel_um)
    return dict(edge_density=edge_density, acf_scale_um=acf_scale_um)


def _judge_segmentable(metrics, cfg):
    big_scale = metrics["acf_scale_um"] >= cfg.judge_acf_scale_um_min
    low_edge = metrics["edge_density"] <= cfg.judge_edge_density_max
    return (big_scale and low_edge) if cfg.judge_require_both else (big_scale or low_edge)


_SAM_GEN_CACHE: dict = {}


def _get_sam_generator(samcfg: "SAMConfig"):
    """按 SAMConfig 内容缓存已加载的 SAM 模型(2.5GB checkpoint),
    避免每张高倍图都从磁盘重新加载一次(单进程内多次调用时的纯性能优化,
    不改变分割结果)。"""
    key = (samcfg.checkpoint, samcfg.model_type, samcfg.device,
           samcfg.points_per_side, samcfg.pred_iou_thresh,
           samcfg.stability_score_thresh, samcfg.min_mask_region_area_px)
    gen = _SAM_GEN_CACHE.get(key)
    if gen is not None:
        return gen
    try:
        import torch  # noqa
        from segment_anything import (SamAutomaticMaskGenerator,
                                       sam_model_registry)
    except Exception as e:
        raise RuntimeError(
            "SAM 不可用(缺 torch/segment-anything)。本地:\n"
            "  pip install torch segment-anything\n"
            "  下载 sam_vit_h_4b8939.pth 到 SAMConfig.checkpoint。\n"
            f"原始错误:{e}")
    sam = sam_model_registry[samcfg.model_type](checkpoint=samcfg.checkpoint)
    sam.to(samcfg.device)
    gen = SamAutomaticMaskGenerator(
        sam, points_per_side=samcfg.points_per_side,
        pred_iou_thresh=samcfg.pred_iou_thresh,
        stability_score_thresh=samcfg.stability_score_thresh,
        min_mask_region_area=samcfg.min_mask_region_area_px)
    _SAM_GEN_CACHE[key] = gen
    return gen


def _segment_primary_sam(img, pixel_um, segcfg, samcfg):
    gen = _get_sam_generator(samcfg)
    masks = gen.generate(color.gray2rgb(img))
    labels = np.zeros(img.shape, dtype=np.int32)
    for i, m in enumerate(sorted(masks, key=lambda x: -x["area"]), start=1):
        labels[m["segmentation"]] = i
    return _regionprops_um(labels, pixel_um, segcfg.pri_min_diam_um), labels


def _segment_primary_classic(img, pixel_um, cfg):
    img_eq = exposure.equalize_adapthist(img, clip_limit=0.02)
    img_s = filters.gaussian(img_eq, sigma=max(1.0, 0.10 / pixel_um))
    block = max(31, int(round(2.5 / pixel_um)) | 1)
    mask = img_s > filters.threshold_local(img_s, block_size=block, offset=-0.008)
    min_area_px = (np.pi * (cfg.pri_min_diam_um / 2) ** 2) / (pixel_um ** 2)
    mask = morphology.remove_small_holes(mask, area_threshold=int(min_area_px * 3))
    mask = morphology.remove_small_objects(mask, min_size=int(min_area_px * 0.5))
    mask = morphology.closing(mask, morphology.disk(3))
    mask = morphology.opening(mask, morphology.disk(2))
    dist = filters.gaussian(ndi.distance_transform_edt(mask), sigma=1.0)
    hmax = morphology.h_maxima(dist, 0.30 * dist.max())
    markers, _ = ndi.label(hmax)
    if markers.max() == 0:
        markers, _ = ndi.label(mask)
    labels = segmentation.watershed(-dist, markers, mask=mask, compactness=0.01)
    return _regionprops_um(labels, pixel_um, cfg.pri_min_diam_um), labels


def _regionprops_um(labels, pixel_um, min_diam_um, clear_border: bool = False):
    """label 图 → µm 量纲的颗粒属性表。

    clear_border=True 时剔除与视场边界相接的颗粒(标准体视学做法)。
    理由主要在**形状而非尺寸**:被画幅切断的颗粒会获得一条人工直边,轮廓比
    真实颗粒更"光滑",系统性抬高 circularity 与 convexity。实测(2026-07-22,
    去信息栏后):触边颗粒的 circularity 中位数普遍高于内部颗粒(C14-M 为
    0.334 vs 0.245,相对差 36%),convexity 同向。而触边比例本身与颗粒尺寸
    强相关——L 组 20–25%、S 组 11–15%,约两倍之差——于是在 H3 依赖的形状
    变量上形成**前驱体相关的偏倚**,同炉差分无法抵消。

    对尺寸的影响则很小:D_sec 上移 1–3%,L/S 比值仅变动 1–2%
    (C07: 2.092→2.075),M_D 链条不受实质影响。

    残留偏倚(已量化,暂不校正):大颗粒触边概率更高,一律剔除会使数目加权
    的尺寸分布轻微偏小。Miles–Lantuéjoul 权重 1/((W-w)(H-h)) 可无偏校正,
    实测将 D_sec 再上移 1–4%(C21-S 6.84→7.10)、L/S 比值再变动约 1%。
    因量级远小于当前其它未决项(分割合并、二值化破碎),暂不引入加权中位数,
    留待形状口径定稿后一并处理。
    """
    empty_cols = ["area_um2", "perimeter_um", "equiv_diam_um",
                  "major_um", "minor_um", "convex_area_um2"]
    if labels.max() == 0:
        return pd.DataFrame(columns=empty_cols)
    props = measure.regionprops_table(
        labels, properties=("label", "area", "perimeter",
                            "equivalent_diameter", "major_axis_length",
                            "minor_axis_length", "convex_area", "bbox"))
    df = pd.DataFrame(props)
    px, px2 = pixel_um, pixel_um ** 2
    out = pd.DataFrame({
        "area_um2": df["area"] * px2,
        "perimeter_um": df["perimeter"] * px,
        "equiv_diam_um": df["equivalent_diameter"] * px,
        "major_um": df["major_axis_length"] * px,
        "minor_um": df["minor_axis_length"] * px,
        "convex_area_um2": df["convex_area"] * px2,
    })
    keep = out["equiv_diam_um"] >= min_diam_um
    if clear_border:
        H, W = labels.shape
        touches = ((df["bbox-0"] <= 0) | (df["bbox-1"] <= 0)
                   | (df["bbox-2"] >= H) | (df["bbox-3"] >= W))
        keep &= ~touches
    return out[keep].reset_index(drop=True)


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


def _aggregate_primary(props_sam, props_classic, segmentable, metrics):
    """SAM = 官方 D_pri_sem 口径,经典分割降级为 D_pri_sem_classic 诊断列
    (第十一轮任务4)。SAM 不可用/异常时,官方 D_pri_sem 退化回用经典值兜底,
    而不是直接 NaN(调用方在 warnings 里说明是哪种情况)。"""
    base = dict(primary_segmentable=bool(segmentable),
                acf_scale_um=round(metrics.get("acf_scale_um", np.nan), 4),
                edge_density=round(metrics.get("edge_density", np.nan), 4))
    if not segmentable:
        return dict(D_pri_sem=np.nan, n_pri=0,
                    D_pri_sem_classic=np.nan, n_pri_classic=0, **base)
    out = dict(**base)
    if props_sam is not None and len(props_sam):
        out.update(D_pri_sem=float(props_sam["equiv_diam_um"].median()),
                    n_pri=int(len(props_sam)))
    elif props_classic is not None and len(props_classic):
        # SAM 不可用时的兜底:官方值退化用经典分割(warnings 由调用方发出)
        out.update(D_pri_sem=float(props_classic["equiv_diam_um"].median()),
                    n_pri=int(len(props_classic)))
    else:
        out.update(D_pri_sem=np.nan, n_pri=0)
    if props_classic is not None and len(props_classic):
        out.update(D_pri_sem_classic=float(props_classic["equiv_diam_um"].median()),
                    n_pri_classic=int(len(props_classic)))
    else:
        out.update(D_pri_sem_classic=np.nan, n_pri_classic=0)
    return out


def _process_one_sample(sub, cfg):
    sid = sub.name
    rec = {"sample_id": sid}
    low, high = _collect_mag_files(sub)
    if low:
        props_low = []
        for f in low:
            px, _ = _read_pixel_size_um(f)
            img, _ = _crop_info_bar(_read_image_gray(f), cfg.seg)
            p, labels = _segment_secondary(img, px, cfg.seg)
            props_low.append(p)
            if cfg.save_overlays:
                _save_overlay(img, labels, p, px, cfg, f"{sid}_{f.stem}_sec")
        rec.update(_aggregate_secondary(pd.concat(props_low, ignore_index=True)))
    else:
        warnings.warn(f"{sid}: 无 500× 图")
        rec.update(_aggregate_secondary(pd.DataFrame()))
    if high:
        per_img_metrics, per_img_seg = [], []
        props_high_sam, props_high_classic = [], []
        for f in high:
            px, _ = _read_pixel_size_um(f)
            img, _ = _crop_info_bar(_read_image_gray(f), cfg.seg)
            m = _texture_metrics(img, px)
            seg_ok = _judge_segmentable(m, cfg.seg)
            per_img_metrics.append(m)
            per_img_seg.append(seg_ok)
            if seg_ok:
                p_classic, labels_classic = _segment_primary_classic(img, px, cfg.seg)
                props_high_classic.append(p_classic)
                p_sam, labels_sam = None, None
                if cfg.sam.enabled:
                    try:
                        p_sam, labels_sam = _segment_primary_sam(img, px, cfg.seg, cfg.sam)
                        props_high_sam.append(p_sam)
                    except Exception as e:
                        warnings.warn(f"{sid}/{f.name}: SAM 不可用({type(e).__name__}: {e}),"
                                      "官方 D_pri_sem 本张回退经典分割")
                if cfg.save_overlays:
                    if p_sam is not None:
                        _save_overlay(img, labels_sam, p_sam, px, cfg, f"{sid}_{f.stem}_pri_sam")
                    _save_overlay(img, labels_classic, p_classic, px, cfg,
                                 f"{sid}_{f.stem}_pri_classic")
        segmentable = sum(per_img_seg) >= (len(per_img_seg) / 2)
        mean_metrics = {k: float(np.mean([m[k] for m in per_img_metrics]))
                        for k in per_img_metrics[0]}
        cat_sam = pd.concat(props_high_sam, ignore_index=True) if props_high_sam else None
        cat_classic = pd.concat(props_high_classic, ignore_index=True) if props_high_classic else None
        rec.update(_aggregate_primary(cat_sam, cat_classic, segmentable, mean_metrics))
    else:
        warnings.warn(f"{sid}: 无 5000× 图")
        rec.update(_aggregate_primary(None, None, False, {}))
    return rec


def _save_overlay(img, labels, props, pixel_um, cfg, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    out_dir = Path(cfg.overlay_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title(f"{name}\n(1) cropped"); ax[0].axis("off")
    ax[1].imshow(color.label2rgb(labels, image=img, bg_label=0, alpha=0.4))
    ax[1].set_title(f"(2) labels={int(labels.max())}"); ax[1].axis("off")
    b = find_boundaries(labels, mode="outer")
    ov = np.dstack([img] * 3).astype(float) / 255; ov[b] = [0, 1, 0]
    ax[2].imshow(ov)
    med = props["equiv_diam_um"].median() if (props is not None and len(props)) else float("nan")
    ax[2].set_title(f"(3) kept={0 if props is None else len(props)} | median={med:.3f}um")
    ax[2].axis("off")
    bar_um = 10 if pixel_um > 0.05 else 1
    bar_px = bar_um / pixel_um
    ax[2].plot([20, 20 + bar_px], [img.shape[0] - 25] * 2, "y-", lw=4)
    ax[2].text(20, img.shape[0] - 40, f"{bar_um} um", color="y", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}.png", dpi=85, bbox_inches="tight")
    plt.close()


def config_from_yaml(raw_config: dict) -> SemProcessorConfig:
    """从 config.yaml 的 instruments.sem 块构建 SemProcessorConfig。

    sam/seg 均缺省时回退到 dataclass 默认值;字段名与 SegConfig/SAMConfig
    一一对应,新增映射参数需同步在 configs/config.yaml 里显式写出。
    """
    cfg = SemProcessorConfig()
    sem_yaml = raw_config.get("instruments", {}).get("sem", {})
    sam_yaml = sem_yaml.get("sam", {})
    if sam_yaml:
        cfg.sam.checkpoint = sam_yaml.get("checkpoint", cfg.sam.checkpoint)
        cfg.sam.model_type = sam_yaml.get("model_type", cfg.sam.model_type)
        cfg.sam.points_per_side = sam_yaml.get("points_per_side", cfg.sam.points_per_side)
    seg_yaml = sem_yaml.get("seg", {})
    for key, val in seg_yaml.items():
        if not hasattr(cfg.seg, key):
            raise ValueError(f"configs/config.yaml instruments.sem.seg 未知字段: {key}")
        setattr(cfg.seg, key, val)
    return cfg


def process_all(raw_dir, out_csv, cfg=None):
    cfg = cfg or SemProcessorConfig()
    raw_dir = Path(raw_dir)
    rows = []
    for sub in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        if not SAMPLE_ID_RE.match(sub.name):
            continue
        print(f"[SEM] processing {sub.name} ...")
        rows.append(_process_one_sample(sub, cfg))
    out = pd.DataFrame(rows)
    sem_targets = ["D_sec", "aspect_ratio", "circularity", "convexity", "D_pri_sem"]
    diag = ["n_sec", "n_pri", "D_pri_sem_classic", "n_pri_classic",
            "primary_segmentable", "acf_scale_um", "edge_density"]
    ordered = ["sample_id"] + [c for c in sem_targets if c in out.columns] \
              + [c for c in diag if c in out.columns]
    out = out[[c for c in ordered if c in out.columns]]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[SEM] wrote {out_csv} ({len(out)} rows)")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/sem")
    ap.add_argument("--out", default="data/interim/sem_features.csv")
    ap.add_argument("--no-sam", action="store_true")
    ap.add_argument("--no-overlay", action="store_true")
    args = ap.parse_args()
    c = SemProcessorConfig()
    if args.no_sam:
        c.sam.enabled = False
    if args.no_overlay:
        c.save_overlays = False
    process_all(args.raw, args.out, c)
