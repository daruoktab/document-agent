"""Wait for OCR readiness before systemd starts dependent services."""
import time
import urllib.error
import urllib.request

started = time.monotonic()
deadline = started + 600
next_report = started
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

while time.monotonic() < deadline:
    try:
        with opener.open("http://127.0.0.1:8081/health", timeout=5) as response:
            if response.status == 200:
                print("PaddleOCR-VL ready on port 8081", flush=True)
                break
    except (OSError, urllib.error.URLError):
        pass
    now = time.monotonic()
    if now >= next_report:
        print(f"Menunggu OCR di port 8081: {int(now - started)} detik", flush=True)
        next_report = now + 10
    time.sleep(2)
else:
    raise SystemExit("OCR readiness timeout after 600 seconds")
