# -*- coding: utf-8 -*-
"""08 物理一致性四项核查(Q 三方对比 / Arrhenius 还原 / 单调 / 边界)。"""
import _bootstrap  # noqa
from nfm.evaluation.physics_check import check_Q

if __name__ == "__main__":
    # 示例:训练得到 Q_learn 后,与独立动力学拟合 Q(mean±95%CI)及文献对比
    demo = check_Q(Q_learn_kJmol=205.0, Q_fit_mean_kJmol=198.0,
                   Q_fit_ci_kJmol=(182.0, 214.0))
    print("Q 三方对比示例:", demo)
    print("提示:Arrhenius 还原 / 单调满足率 / 边界合规率 需传入训练好的模型预测函数")
