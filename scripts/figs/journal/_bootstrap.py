# -*- coding: utf-8 -*-
"""journal 图脚本公用:把 src 加进 sys.path,另外把上级 scripts/figs/ 也加进
sys.path,好让 scripts/figs/journal/_journal_style.py 能 `from _style import
...` 复用内部评审图的 PRECURSOR_COLOR/PRECURSOR_MARKER/TEMP_COLOR(不重新
定义配色)。

此文件在 scripts/figs/journal/ 下,比 scripts/figs/_bootstrap.py 深一层,
sys.path[0] 是 scripts/figs/journal,既看不到 scripts/figs/_bootstrap.py
也看不到 scripts/figs/_style.py,故这里用 parents[3](而不是 _bootstrap.py
里的 parents[2])算到仓库根目录,并显式补上 parents[1] = scripts/figs。
"""
import sys
from pathlib import Path

_FIGS_DIR = Path(__file__).resolve().parents[1]  # scripts/figs
_SRC_DIR = Path(__file__).resolve().parents[3] / "src"  # repo_root/src

sys.path.insert(0, str(_SRC_DIR))
sys.path.insert(0, str(_FIGS_DIR))
