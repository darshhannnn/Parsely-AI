"""
Pytest configuration: make the repository root importable so tests can
use `from src...` imports regardless of how pytest is invoked.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
