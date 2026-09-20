# -*- coding: utf-8 -*-
"""00 环境与目录自检 + schema/视图/Θ 配置校验。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfm.config import load_config, build_sample_index
from nfm import schema


def main():
    cfg = load_config()
    print("✓ config 加载、视图铁律校验通过")
    idx = build_sample_index(cfg)
    assert len(idx) == 81, f"样本索引应为 81 行,实际 {len(idx)}"
    print(f"✓ 样本索引 {len(idx)} 行(27 条件 × 3 前驱体)")
    print(f"✓ schema 列数 {len(schema.SCHEMA)};输入列 {len(schema.input_cols())},"
          f"目标列 {len(schema.target_cols())},差分量 {len(schema.diff_cols())}")
    print("  差分量:", schema.diff_cols())
    for d in ("data/raw", "data/interim", "data/processed", "outputs"):
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✓ 目录就绪。下一步:01_process_psd.py")


if __name__ == "__main__":
    main()
