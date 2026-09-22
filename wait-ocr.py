#!/usr/bin/env python3
import sys
import time
import urllib.request
from datetime import datetime

deadline = time.time() + 660
started = time.time()
last_report = 0.0
while time.time() < deadline:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8081/health", timeout=5) as response:
            if response.status == 200:
                elapsed = int(time.time() - started)
                print(f"[{datetime.now():%F %T}] Unlimited OCR ready on port 8081 "
                      f"after {elapsed}s", flush=True)
                sys.exit(0)
    except Exception:
        pass
    elapsed = time.time() - started
    if elapsed - last_report >= 10:
        remaining = max(0, int(deadline - time.time()))
        print(f"[{datetime.now():%F %T}] Menunggu Unlimited OCR siap... "
              f"elapsed={int(elapsed)}s remaining={remaining}s", flush=True)
        last_report = elapsed
    time.sleep(2)
print(f"[{datetime.now():%F %T}] Timed out waiting for Unlimited OCR on port 8081",
      file=sys.stderr, flush=True)
sys.exit(1)
