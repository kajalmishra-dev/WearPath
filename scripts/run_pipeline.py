#!/usr/bin/env python3
"""CLI wrapper to run the WearPath batch pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable when running as a script without install.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wear_path.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
