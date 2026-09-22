#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_SYSTEMD_DIR="${HOME}/.config/systemd/user"

mkdir -p "${USER_SYSTEMD_DIR}"

ln -sfn "${PROJECT_DIR}/llama-server.service" "${USER_SYSTEMD_DIR}/llama-server.service"
ln -sfn "${PROJECT_DIR}/ocr-server.service" "${USER_SYSTEMD_DIR}/ocr-server.service"
ln -sfn "${PROJECT_DIR}/streamlit.service" "${USER_SYSTEMD_DIR}/streamlit.service"

systemctl --user daemon-reload
systemctl --user enable llama-server.service ocr-server.service streamlit.service
systemctl --user restart --no-block llama-server.service
systemctl --user restart --no-block ocr-server.service
systemctl --user restart --no-block streamlit.service

echo "Service project v2 sedang dinyalakan: ${PROJECT_DIR}"
echo "Pantau VLM : journalctl --user -u llama-server.service -f"
echo "Pantau OCR : journalctl --user -u ocr-server.service -f"
