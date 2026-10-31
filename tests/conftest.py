"""
Pytest configuration and environment fixture setup.
Adds project root directory to sys.path to ensure src and app can be imported.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
