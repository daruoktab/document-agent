"""Jalankan seluruh tes proyek melalui pytest, termasuk fixture offline."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def run_all_tests() -> int:
    test_dir = Path(__file__).resolve().parent
    return pytest.main(["-q", str(test_dir), "-p", "no:cacheprovider"])


if __name__ == "__main__":
    sys.exit(run_all_tests())
