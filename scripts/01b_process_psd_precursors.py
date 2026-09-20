# -*- coding: utf-8 -*-
"""01b 前驱体原始曲线双峰拟合(第十一轮任务3b,解锁 M_B 分母)。
只跑三个前驱体 CSV,输出诊断表,不进入主 process_all/81行流水线。
"""
import _bootstrap  # noqa
from pathlib import Path

import pandas as pd

from nfm.data_processing import psd_processor as pp

PRECURSOR_DIR = Path("data/raw/psd/precursors")
OUT_CSV = Path("data/interim/psd_precursor_features.csv")

if __name__ == "__main__":
    rows = []
    for letter in ("S", "M", "L"):
        f = PRECURSOR_DIR / f"precursor_{letter}.csv"
        d, cum = pp.read_cumulative_csv(f)
        _, density = pp.read_density_csv(f)
        r = pp.fit_bimodal(d, cum, density)
        r["precursor"] = letter
        rows.append(r)
    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(out[["precursor", "n_modes", "mode_positions_um", "B_bimodal", "note"]]
          .to_string(index=False))
