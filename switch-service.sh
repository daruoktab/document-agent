#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_SYSTEMD_DIR="${HOME}/.config/systemd/user"

for required_file in llama-server.service ocr-server.service streamlit.service logged-exec.sh wait-llama.py wait-ocr.py; do
    if [[ ! -f "${PROJECT_DIR}/${required_file}" ]]; then
        echo "File startup tidak ditemukan: ${PROJECT_DIR}/${required_file}" >&2
        exit 1
    fi
done

mkdir -p "${USER_SYSTEMD_DIR}"

ln -sfn "${PROJECT_DIR}/llama-server.service" "${USER_SYSTEMD_DIR}/llama-server.service"
ln -sfn "${PROJECT_DIR}/ocr-server.service" "${USER_SYSTEMD_DIR}/ocr-server.service"
ln -sfn "${PROJECT_DIR}/streamlit.service" "${USER_SYSTEMD_DIR}/streamlit.service"

systemctl --user daemon-reload
systemctl --user enable llama-server.service ocr-server.service streamlit.service

echo "Menunggu VLM, OCR, dan Streamlit selesai direstart..."
if ! systemctl --user restart llama-server.service ocr-server.service streamlit.service; then
    echo "Restart gagal. Status service:" >&2
    systemctl --user --no-pager --full status \
        llama-server.service ocr-server.service streamlit.service || true
    echo "Log Streamlit terbaru:" >&2
    journalctl --user -u streamlit.service -n 60 --no-pager || true
    echo "Log VLM/OCR terbaru:" >&2
    journalctl --user -u llama-server.service -u ocr-server.service -n 40 --no-pager || true
    exit 1
fi

systemctl --user --no-pager --full status \
    llama-server.service ocr-server.service streamlit.service

echo "Service project v2 sedang dinyalakan: ${PROJECT_DIR}"
echo "Pantau VLM : journalctl --user -u llama-server.service -f"
echo "Pantau OCR : journalctl --user -u ocr-server.service -f"
echo "Pantau UI  : journalctl --user -u streamlit.service -f"
