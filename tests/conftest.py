import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def monkey(monkeypatch):
    """Short alias: monkey(obj, name, value)."""
    return monkeypatch.setattr
