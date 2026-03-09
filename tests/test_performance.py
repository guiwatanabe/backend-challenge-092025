import json
import os
import time
import tracemalloc
from pathlib import Path
from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def _gen_dataset(n=1000):
    now = datetime(2025, 9, 10, 11, 0, 0, tzinfo=timezone.utc)
    msgs = []
    for i in range(n):
        ts = (now - timedelta(minutes=(i % 30), seconds=(i % 5))).strftime("%Y-%m-%dT%H:%M:%SZ")
        msgs.append({
            "id": f"perf_{i:04d}",
            "content": "Adorei o novo produto!" if i % 4 != 0 else "ruim",
            "timestamp": ts,
            "user_id": f"user_{i%200:03d}",
            "hashtags": ["#produto", "#teste"] if i % 10 == 0 else ["#produto"],
            "reactions": (i % 7) + 1,
            "shares": (i % 3),
            "views": ((i % 25) + 1) * 10,
        })
    return {"messages": msgs, "time_window_minutes": 30}


def test_performance_under_200ms():
    if os.getenv("RUN_PERF", "0") != "1":
        import pytest
        pytest.skip("Set RUN_PERF=1 to enable performance test")

    perf_path = Path("examples/performance_test_1000.json")
    if perf_path.exists():
        payload = json.loads(perf_path.read_text(encoding="utf-8"))
    else:
        payload = _gen_dataset(1000)

    t0 = time.perf_counter()
    r = client.post("/analyze-feed", json=payload)
    dt = (time.perf_counter() - t0) * 1000
    assert r.status_code == 200
    # Target < 200ms for 1000 messages
    assert dt < 200.0, f"Took {dt:.2f} ms"


def test_memory_under_20mb():
    if os.getenv("RUN_PERF", "0") != "1":
        import gc
        import pytest
        pytest.skip("Set RUN_PERF=1 to enable performance test")

    import gc
    gc.collect()  # free lingering allocs from previous test

    payload = _gen_dataset(10_000)
    payload_bytes = json.dumps(payload).encode()  # serialize before measuring
    del payload  # free the dict so it doesn't skew the measurement

    tracemalloc.start()
    r = client.post(
        "/analyze-feed",
        content=payload_bytes,
        headers={"Content-Type": "application/json"},
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert r.status_code == 200
    peak_mb = peak / 1024 / 1024
    assert peak_mb <= 20.0, f"Peak memory: {peak_mb:.2f} MB"

