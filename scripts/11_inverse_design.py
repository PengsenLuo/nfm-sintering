# -*- coding: utf-8 -*-
"""11 形貌记忆状态图 / 工艺窗口 / 逆向设计。"""
import _bootstrap  # noqa
import numpy as np
from nfm.optimize.state_map import build_state_grid, inverse_design

if __name__ == "__main__":
    # 占位预测函数:接训练好的模型后替换为真实预测
    def fake_indices(T, beta, t):
        Theta_like = (T - 850) / 100 + (t - 10) / 10
        M_D = 1.0 / (1 + np.exp(3 * (Theta_like - 1.0)))
        M_B = max(0.0, 1.0 - 0.4 * Theta_like)
        return M_D, M_B

    grid = build_state_grid(fake_indices,
                            T_range=np.arange(850, 951, 25),
                            t_range=np.arange(10, 21, 5),
                            beta_fixed=5)
    print("形貌记忆状态图(示例,真实模型替换 fake_indices 后生效):")
    print(grid.to_string(index=False))

    def fake_targets(T, beta, t):
        rho = 3.0 + 0.002 * (T - 850) - 0.01 * abs(t - 15)
        return {"compaction_density": rho}

    cand = inverse_design(fake_targets, target_density=3.1)
    print("\n逆向设计候选(目标压实 3.1 g/cm³,示例):")
    print(cand.head().to_string(index=False))
