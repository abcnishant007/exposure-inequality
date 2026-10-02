"""Shared pytest configuration for public tests.

Public tests must not depend on licensed mobility data or private CSV fixtures.
Use in-memory frames or pytest tmp_path files instead.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
