"""
Konfigurasi terpusat untuk Document Vision VLM & Extraction (Ready for Chunking).

Menggunakan Pydantic-like dataclass `Settings` yang membaca environment variables
dengan fallback yang aman untuk local inference (LM Studio / Ollama / llama-server).

Peran Model:
  1. VLM utama: klasifikasi, reasoning visual, specialist, dan quality gate
  2. OCR       : ekstraksi Markdown dasar, grounding, dan region crop
  3. Logging   : konfigurasi level logging
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DPI: int = 200
PDF_PAGE_BATCH: int = 10
SUPPORTED_IMAGE_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".webp"}


def _env(name: str, default: str) -> str:
    """Ambil env var atau kembalikan default jika kosong."""
    val = os.environ.get(name, "").strip()
    return val if val else default


def _env_first(names: tuple[str, ...], default: str) -> str:
    """Ambil nilai pertama yang tidak kosong dari beberapa nama env."""
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return default


def _optional_env(name: str) -> str:
    """Ambil env opsional; string kosong berarti fitur belum dikonfigurasi."""
    return os.environ.get(name, "").strip()


def _float_env(name: str, default: str) -> float:
    """Parse float dari environment variable dengan fallback."""
    raw = os.environ.get(name, default).strip()
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _int_env(name: str, default: str) -> int:
    """Parse int dari environment variable dengan fallback."""
    raw = os.environ.get(name, default).strip()
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _bool_env(name: str, default: str) -> bool:
    """Parse boolean dari environment variable ('1'/'true'/'yes'/'on' -> True)."""
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _load_local_dotenv() -> None:
    """Load project `.env` before settings are created.

    Explicit process environment variables take priority over `.env` values.
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


_load_local_dotenv()


@dataclass(frozen=True)
class Settings:
    """Pengaturan konfigurasi LLM, VLM, OCR, dan logging."""

    # --- Endpoint server llama.cpp ---
    base_url: str = field(
        default_factory=lambda: _env_first(
            ("BASE_URL", "LLM_BASE_URL"), "http://127.0.0.1:8080/v1"
        )
    )

    # Alias lama dipertahankan untuk kompatibilitas caller.
    llm_base_url: str = field(
        default_factory=lambda: _env_first(
            ("BASE_URL", "LLM_BASE_URL"), "http://127.0.0.1:8080/v1"
        )
    )
    llm_api_key: str = field(
        default_factory=lambda: _env_first(
            ("API_KEY", "LLM_API_KEY"), "not-needed"
        )
    )

    # --- 1. VLM utama (reasoning + agent + quality gate) ---
    vlm_model: str = field(
        default_factory=lambda: _env(
            "VLM_MODEL", "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
        )
    )
    vlm_base_url: str = field(
        default_factory=lambda: _env_first(
            ("VLM_BASE_URL", "BASE_URL", "LLM_BASE_URL"),
            "http://127.0.0.1:8080/v1",
        )
    )
    vlm_api_key: str = field(
        default_factory=lambda: _env_first(
            ("VLM_API_KEY", "API_KEY", "LLM_API_KEY"), "not-needed"
        )
    )
    vlm_temperature: float = field(
        default_factory=lambda: _float_env("VLM_TEMPERATURE", "0.1")
    )
    vlm_timeout: float = field(default_factory=lambda: _float_env("VLM_TIMEOUT", "300"))
    vlm_max_tokens: int = field(
        default_factory=lambda: _int_env("VLM_MAX_TOKENS", "4096")
    )
    vlm_enable_thinking: bool = field(
        default_factory=lambda: _bool_env("VLM_ENABLE_THINKING", "false")
    )
    vlm_visual_rescue: bool = field(
        default_factory=lambda: _bool_env("VLM_VISUAL_RESCUE", "true")
    )

    # --- Spreadsheet hybrid extraction ---
    excel_native_survey: bool = field(
        default_factory=lambda: _bool_env("EXCEL_NATIVE_SURVEY", "true")
    )
    excel_region_rendering: bool = field(
        default_factory=lambda: _bool_env("EXCEL_REGION_RENDERING", "true")
    )
    excel_base_dpi: int = field(
        default_factory=lambda: _int_env("EXCEL_BASE_DPI", "300")
    )
    excel_max_dpi: int = field(
        default_factory=lambda: _int_env("EXCEL_MAX_DPI", "450")
    )
    excel_small_font_points: float = field(
        default_factory=lambda: _float_env("EXCEL_SMALL_FONT_POINTS", "8")
    )
    excel_max_region_columns: int = field(
        default_factory=lambda: _int_env("EXCEL_MAX_REGION_COLUMNS", "18")
    )
    excel_max_region_rows: int = field(
        default_factory=lambda: _int_env("EXCEL_MAX_REGION_ROWS", "80")
    )

    # --- 2. OCR terstruktur (aktif otomatis bila OCR_MODEL diisi) ---
    ocr_model: str = field(default_factory=lambda: _optional_env("OCR_MODEL"))
    ocr_base_url: str = field(
        default_factory=lambda: _env("OCR_BASE_URL", "http://127.0.0.1:8081/v1")
    )
    ocr_api_key: str = field(
        default_factory=lambda: _env_first(
            ("OCR_API_KEY", "API_KEY", "LLM_API_KEY"), "not-needed"
        )
    )
    ocr_temperature: float = field(
        default_factory=lambda: _float_env("OCR_TEMPERATURE", "0.0")
    )
    ocr_timeout: float = field(default_factory=lambda: _float_env("OCR_TIMEOUT", "300"))
    ocr_max_tokens: int = field(
        default_factory=lambda: _int_env("OCR_MAX_TOKENS", "4096")
    )
    ocr_prompt: str = field(
        default_factory=lambda: _env(
            "OCR_PROMPT", "<|grounding|>Convert the document to markdown."
        )
    )
    ocr_coordinate_size: int = field(
        default_factory=lambda: _int_env("OCR_COORDINATE_SIZE", "1024")
    )
    ocr_crop_padding: float = field(
        default_factory=lambda: _float_env("OCR_CROP_PADDING", "0.01")
    )
    ocr_min_trust_score: float = field(
        default_factory=lambda: _float_env("OCR_MIN_TRUST_SCORE", "0.72")
    )
    ocr_medium_trust_score: float = field(
        default_factory=lambda: _float_env("OCR_MEDIUM_TRUST_SCORE", "0.48")
    )
    ocr_rotation_retry: bool = field(
        default_factory=lambda: _bool_env("OCR_ROTATION_RETRY", "true")
    )
    ocr_blank_ink_ratio: float = field(
        default_factory=lambda: _float_env("OCR_BLANK_INK_RATIO", "0.0002")
    )
    ocr_sparse_ink_ratio: float = field(
        default_factory=lambda: _float_env("OCR_SPARSE_INK_RATIO", "0.015")
    )

    # --- 3. Logging Configuration ---
    log_level: str = field(
        default_factory=lambda: _env("LOG_LEVEL", "INFO").strip().upper() or "INFO"
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Singleton getter untuk Settings."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def setup_logging(
    level: str | None = None,
    log_file: str | Path | None = None,
    auto_log_stem: str | None = None,
    auto_log_dir: str | Path | None = None,
    llm_response_log_file: str | Path | None = None,
) -> None:
    """Inisialisasi logging terformat dengan timestamp ke konsol dan opsional ke file.

    Jika `log_file` tidak diberikan tetapi `auto_log_stem` ada, log otomatis
    ditulis real-time ke `{auto_log_dir}/{auto_log_stem}_latest.log` atau
    `output/{auto_log_stem}/logs/{auto_log_stem}_latest.log`.
    """
    effective_level = (level or get_settings().log_level).upper()
    log_format = "%(asctime)s | %(levelname)-7s | [%(name)s] %(message)s"
    date_format = "%H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        p = Path(log_file).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(p), mode="a", encoding="utf-8"))
    elif auto_log_stem:
        if auto_log_dir:
            log_dir = Path(auto_log_dir).resolve()
        else:
            log_dir = (Path("output") / auto_log_stem / "logs").resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        p = log_dir / f"{auto_log_stem}_latest.log"
        handlers.append(logging.FileHandler(str(p), mode="w", encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, effective_level, logging.INFO),
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True,
    )

    # Respons mentah LLM sengaja dipisahkan dari log pipeline karena dapat
    # berukuran besar dan berisi isi dokumen.
    response_log_path: Path | None = None
    if llm_response_log_file:
        response_log_path = Path(llm_response_log_file).resolve()
    elif auto_log_stem:
        response_dir = Path(auto_log_dir).resolve() if auto_log_dir else Path("output") / auto_log_stem / "logs"
        response_dir.mkdir(parents=True, exist_ok=True)
        response_log_path = response_dir / f"{auto_log_stem}_llm_responses.log"

    response_log = logging.getLogger("app.llm.response")
    response_log.handlers.clear()
    response_log.setLevel(logging.DEBUG)
    response_log.propagate = False
    if response_log_path:
        response_log_path.parent.mkdir(parents=True, exist_ok=True)
        response_handler = logging.FileHandler(str(response_log_path), mode="w", encoding="utf-8")
        response_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt=date_format,
        ))
        response_log.addHandler(response_handler)
