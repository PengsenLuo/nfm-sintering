# -*- coding: utf-8 -*-
"""依次跑全部 fig_*.py,用于一键重出图(改了数据/样式后重跑此脚本即可)。"""
import _bootstrap  # noqa

import fig_dxrd_vs_theta
import fig_grain_growth_arrhenius
import fig_lattice_vs_theta
import fig_md_vs_theta
import fig_density_vs_theta
import fig_density_matrix
import fig_doe_grid
import fig1_variance_stack
import fig3_sem_panel
import fig4_shape_descriptors
import fig7_condition_residuals

MODULES = [
    fig_dxrd_vs_theta, fig_grain_growth_arrhenius, fig_lattice_vs_theta,
    fig_md_vs_theta, fig_density_vs_theta, fig_density_matrix, fig_doe_grid,
    fig1_variance_stack, fig3_sem_panel, fig4_shape_descriptors,
    fig7_condition_residuals,
]

if __name__ == "__main__":
    for m in MODULES:
        print(f"=== {m.__name__} ===")
        m.main()
