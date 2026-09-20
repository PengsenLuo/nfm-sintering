# -*- coding: utf-8 -*-
"""02 SEM 双口径形貌。SAM 正式取代经典分割。"""
import argparse

import _bootstrap  # noqa
from nfm.config import load_config
from nfm.data_processing import sem_processor

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sam", action="store_true",
                     help="关闭 SAM,一次颗粒只跑传统阈值分割(诊断用)")
    ap.add_argument("--adaptive-h", action="store_true",
                     help="强制开启 500x 形貌自适应 h(configs/config.yaml 已默认开启,"
                          "此开关仅用于显式覆盖/兼容旧调用)")
    ap.add_argument("--no-adaptive-h", action="store_true",
                     help="关闭 500x 形貌自适应 h,复现旧的固定 sec_hmaxima_rel 行为"
                          "(诊断/回归对比用)")
    args = ap.parse_args()

    cfg = load_config()
    sem_cfg = sem_processor.config_from_yaml(cfg.raw)
    if args.no_sam:
        sem_cfg.sam.enabled = False
    if args.adaptive_h:
        sem_cfg.seg.sec_adaptive_h = True
    if args.no_adaptive_h:
        sem_cfg.seg.sec_adaptive_h = False
    out = sem_processor.process_all("data/raw/sem",
                                    "data/interim/sem_features.csv", sem_cfg)
    print(f"SEM 处理完成,{len(out)} 样本 -> data/interim/sem_features.csv")
