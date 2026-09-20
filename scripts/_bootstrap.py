# -*- coding: utf-8 -*-
"""脚本公用:把 src 加进 sys.path。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
