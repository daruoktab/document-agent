#!/usr/bin/env python3
import sys
import time
import urllib.request
from datetime import UTC, datetime


def timestamp() -> str:
    """Return the current local timestamp with explicit timezone handling."""
    return f"{datetime.now(UTC).astimezone():%F %T}"

deadline = time.time() + 660
started = time.time()
last_report = 0.0
while time.time() < deadline:
    ready = False
    try:
        with urllib.request.urlopen("http://127.0.0.1:8081/health", timeout=5) as response:
            ready = response.status == 200
    except OSError:
        ready = False
    if ready:
        elapsed = int(time.time() - started)
        print(
            f"[{timestamp()}] Unlimited OCR ready on port 8081 after {elapsed}s",
            flush=True,
        )
        sys.exit(0)
    elapsed = time.time() - started
    if elapsed - last_report >= 10:
        remaining = max(0, int(deadline - time.time()))
        print(
            f"[{timestamp()}] Menunggu Unlimited OCR siap... "
            f"elapsed={int(elapsed)}s remaining={remaining}s",
            flush=True,
        )
        last_report = elapsed
    time.sleep(2)
print(
    f"[{timestamp()}] Timed out waiting for Unlimited OCR on port 8081",
    file=sys.stderr,
    flush=True,
)
sys.exit(1)
