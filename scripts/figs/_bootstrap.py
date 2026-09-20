# -*- coding: utf-8 -*-
"""figs 脚本公用:把 src 加进 sys.path(此文件在 scripts/figs/ 下,与顶层
scripts/_bootstrap.py 是同名但独立的两份,因为 fig_*.py 直接跑时 sys.path[0]
是 scripts/figs,不会自动看到 scripts/_bootstrap.py)。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
