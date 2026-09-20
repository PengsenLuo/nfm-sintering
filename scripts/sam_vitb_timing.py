# -*- coding: utf-8 -*-
"""第十轮任务7:SAM ViT-B + points_per_side={16,24} 计时/显存/精度对比实验。
只做诊断,不接入主流水线、不覆盖第九轮经典分割产出的 sem_features.csv。
"""
import _bootstrap  # noqa
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nfm.data_processing import sem_processor as sp

SAMPLES = ["C16-S", "C21-S", "C24-M"]
PPS_GRID = [16, 24]
CHECKPOINT = "src/nfm/models/sam_vit_b_01ec64.pth"
OVERLAY_DIR = Path("data/interim/sem_overlays_sam_vitb_timing")


def _pick_high_mag_file(sid: str) -> Path:
    sub = Path("data/raw/sem") / sid
    _, high = sp._collect_mag_files(sub)
    if not high:
        raise RuntimeError(f"{sid}: 未找到 5000× 高倍图")
    return high[0]


def _load_classic_dpri(sid: str) -> tuple[float, int]:
    df = pd.read_csv("data/interim/sem_features.csv")
    row = df.set_index("sample_id").loc[sid]
    return float(row.get("D_pri_sem", np.nan)), int(row.get("n_pri", 0))


def main():
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    seg_cfg = sp.SegConfig()
    rows = []
    for pps in PPS_GRID:
        sam_cfg = sp.SAMConfig(enabled=True, checkpoint=CHECKPOINT, model_type="vit_b",
                               device="cuda", points_per_side=pps)
        t0 = time.perf_counter()
        sp._get_sam_generator(sam_cfg)   # 触发一次模型加载,单独计时
        load_s = time.perf_counter() - t0
        print(f"[pps={pps}] 模型加载耗时 {load_s:.2f}s")

        for sid in SAMPLES:
            f = _pick_high_mag_file(sid)
            px, _ = sp._read_pixel_size_um(f)
            img, _ = sp._crop_info_bar(sp._read_image_gray(f), seg_cfg)

            try:
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                props, labels = sp._segment_primary_sam(img, px, seg_cfg, sam_cfg)
                infer_s = time.perf_counter() - t0
                peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
            except torch.cuda.OutOfMemoryError as e:
                print(f"  {sid}: pps={pps} OOM! {e}")
                rows.append(dict(sample_id=sid, points_per_side=pps, file=f.name,
                                 load_time_s=round(load_s, 2), infer_time_s=np.nan,
                                 peak_vram_MB=np.nan,
                                 D_pri_sem_sam=np.nan, n_pri_sam=np.nan,
                                 D_pri_sem_classic=np.nan, n_pri_classic=np.nan,
                                 status=f"OOM: {e}"))
                torch.cuda.empty_cache()
                continue

            d_pri_sam = float(props["equiv_diam_um"].median()) if len(props) else np.nan
            n_pri_sam = int(len(props))
            d_pri_classic, n_pri_classic = _load_classic_dpri(sid)

            sp._save_overlay(img, labels, props, px, sp.SemProcessorConfig(overlay_dir=str(OVERLAY_DIR)),
                             f"{sid}_pps{pps}_vitb_sam")

            rows.append(dict(sample_id=sid, points_per_side=pps, file=f.name,
                             load_time_s=round(load_s, 2), infer_time_s=round(infer_s, 2),
                             peak_vram_MB=round(peak_vram_mb, 1),
                             D_pri_sem_sam=d_pri_sam, n_pri_sam=n_pri_sam,
                             D_pri_sem_classic=d_pri_classic, n_pri_classic=n_pri_classic,
                             status="ok"))
            print(f"  {sid}: infer={infer_s:.2f}s peak_vram={peak_vram_mb:.0f}MB "
                  f"D_sam={d_pri_sam:.3f}um(n={n_pri_sam}) "
                  f"D_classic={d_pri_classic:.3f}um(n={n_pri_classic})")

    out = pd.DataFrame(rows)
    out_csv = "data/interim/sam_vitb_timing_summary.csv"
    out.to_csv(out_csv, index=False)
    print(f"\n写入 {out_csv}")

    n_high_mag_imgs_per_sample = 3   # 拍摄协议:每倍率 3 张
    n_target_samples = 19            # 第九轮遗留的经典分割子集
    for pps in PPS_GRID:
        sub = out[(out.points_per_side == pps) & (out["status"] == "ok")]
        if not len(sub):
            print(f"[pps={pps}] 全部组合失败(OOM 或其它错误),无法外推")
            continue
        avg_s = sub["infer_time_s"].mean()
        est_h = avg_s * n_high_mag_imgs_per_sample * n_target_samples / 3600
        n_ok = len(sub)
        n_total = len(out[out.points_per_side == pps])
        print(f"[pps={pps}] 单张均值 {avg_s:.2f}s(基于 {n_ok}/{n_total} 张成功样本) "
              f"→ 外推 19 样({n_high_mag_imgs_per_sample}张/样)总耗时约 {est_h:.2f} 小时")


if __name__ == "__main__":
    main()
