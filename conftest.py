"""Ensure the project root is importable so tests can do `from tests.fake_engine`.

Keeping the fake engine under ``tests/`` (not shipped in the wheel) means it
documents consumer wiring without becoming part of the public package.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
