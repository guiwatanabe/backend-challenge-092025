from datetime import datetime, timezone

import pytest

# Fixed reference time matching all test timestamps (2025-09-10T10:00:00Z)
FIXED_NOW = datetime(2025, 9, 10, 10, 5, 0, tzinfo=timezone.utc)

@pytest.fixture(autouse=True)
def freeze_time(monkeypatch):
    monkeypatch.setattr("main._get_reference_time", lambda: FIXED_NOW)
