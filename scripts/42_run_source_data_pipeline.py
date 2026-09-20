# -*- coding: utf-8 -*-
"""scripts/42_run_source_data_pipeline.py -- F4 single-command entry point
("make source-data" equivalent): regenerate every journal figure's
manuscript/figures/source_data/*.csv, then run the machine cross-check.

Usage: python scripts/42_run_source_data_pipeline.py
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
JOURNAL_DIR = REPO_ROOT / "scripts" / "figs" / "journal"


def main() -> int:
    fig_scripts = sorted(JOURNAL_DIR.glob("fig*.py"))
    if not fig_scripts:
        print(f"[error] no fig*.py found under {JOURNAL_DIR}")
        return 1

    for script in fig_scripts:
        print(f"=== {script.relative_to(REPO_ROOT)} ===")
        result = subprocess.run([sys.executable, str(script)], cwd=REPO_ROOT)
        if result.returncode != 0:
            print(f"[error] {script.name} exited {result.returncode} -- stopping")
            return result.returncode

    print("=== scripts/41_verify_figure_source_data.py ===")
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "41_verify_figure_source_data.py")],
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        print(f"[error] verification script exited {result.returncode} -- "
             f"see reports/figure_source_data_verification.md")
        return result.returncode

    print(f"[ok] regenerated {len(fig_scripts)} figures' source data and "
         f"verified them; see reports/figure_source_data_verification.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
