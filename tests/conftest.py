"""Root conftest — ensures the project root is on sys.path for all test suites.

This is needed so that `import database`, `import src`, `import api`, etc. work
regardless of which test subdirectory pytest collects from.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
