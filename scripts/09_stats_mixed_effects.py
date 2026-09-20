# -*- coding: utf-8 -*-
"""09 混合效应(炉次随机效应 + Θ×前驱体交互)+(可选)调节中介。"""
import _bootstrap  # noqa
import pandas as pd
from nfm.config import load_config
from nfm.stats.mixed_effects import fit_mixed_effects
from nfm.stats.mediation import moderated_mediation

if __name__ == "__main__":
    cfg = load_config()
    df = pd.read_csv("data/processed/master_table.csv")
    for y in ("D50", "compaction_density"):
        if y not in df.columns or df[y].notna().sum() < 10:
            print(f"⚠ {y} 数据不足,跳过混合效应"); continue
        try:
            r = fit_mixed_effects(df, y_col=y)
            print(f"=== 混合效应:{y} ===")
            print("  交互项:", r["interaction_terms"])
            print("  解读:", r["interpretation"])
        except Exception as e:  # noqa
            print(f"  {y} 混合效应跳过:{e}")
    if cfg.raw["stats"]["mediation"]["enabled"] and "M_B" in df.columns:
        try:
            med = moderated_mediation(df)
            print("=== 调节中介(支持性证据)===")
            print(" ", med)
        except Exception as e:  # noqa
            print("  中介分析跳过:", e)
