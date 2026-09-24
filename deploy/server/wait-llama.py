"""Readiness gate: allow up to ten minutes to load the existing local model."""
import time
import urllib.error
import urllib.request

deadline = time.monotonic() + 600
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
while time.monotonic() < deadline:
    try:
        with opener.open("http://127.0.0.1:8080/health", timeout=5) as response:
            if response.status == 200:
                print("Llama ready on port 8080", flush=True)
                break
    except (OSError, urllib.error.URLError):
        pass
    time.sleep(2)
else:
    raise SystemExit("Llama readiness timeout after 600 seconds")
