"""
Pembangun model chat (`ChatOpenAI`) untuk endpoint OpenAI-compatible.

Menyediakan:
  - build_chat_model(base_url, model, api_key, ...) : builder generik
  - build_vlm(settings)  : VLM utama (reasoning + agent + quality gate)
  - build_ocr(settings)  : model OCR terstruktur
  - get_vlm(settings)    : Helper singleton / factory untuk VLM
  - encode_image, encode_image_to_base64, image_data_uri : utility encoding citra

Endpoint dapat berupa LM Studio lokal, `llama-server`, atau server remote -
cukup ubah `.env`.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI

from .config import Settings, get_settings

logger = logging.getLogger("app.llm")
response_logger = logging.getLogger("app.llm.response")


class LoggingCallbackHandler(BaseCallbackHandler):
    """Callback handler transparan untuk mencatat setiap request & response LLM/VLM."""

    def __init__(self, model_name: str, base_url: str) -> None:
        self.model_name = model_name
        self.base_url = base_url
        self._start_time: float = 0.0

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        self._start_time = time.perf_counter()
        logger.info(
            "--> [LLM Request] Model: %s | URL: %s | Prompts: %d item",
            self.model_name,
            self.base_url,
            len(prompts),
        )
        for i, p in enumerate(prompts):
            preview = p[:120].replace("\n", " ")
            logger.debug("    Prompt #%d: %s...", i + 1, preview)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        elapsed = time.perf_counter() - self._start_time
        gen_count = sum(len(g) for g in response.generations)
        logger.info(
            "<-- [LLM Response] Model: %s | Waktu: %.2fs | Generasi: %d item",
            self.model_name,
            elapsed,
            gen_count,
        )

        # Simpan respons mentah di logger terpisah agar diagnosis output kosong
        # tidak tercampur dengan log pipeline utama.
        for index, generation_group in enumerate(response.generations, start=1):
            for candidate_index, generation in enumerate(generation_group, start=1):
                message = getattr(generation, "message", None)
                content = getattr(message, "content", None)
                if content is None:
                    content = getattr(generation, "text", "")
                generation_info = getattr(generation, "generation_info", None)
                message_metadata = {
                    "additional_kwargs": getattr(message, "additional_kwargs", {}),
                    "response_metadata": getattr(message, "response_metadata", {}),
                }
                response_logger.info(
                    "[LLM Response Raw] model=%s candidate=%d.%d chars=%d "
                    "generation_info=%r metadata=%r\n--- BEGIN RESPONSE ---\n%s\n--- END RESPONSE ---",
                    self.model_name,
                    index,
                    candidate_index,
                    len(str(content)),
                    generation_info,
                    message_metadata,
                    content,
                )


def encode_image(image_path: str | Path) -> str:
    """Baca file gambar dan encode ke base64 string."""
    with open(str(image_path), "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_image_to_base64(image_path: str | Path) -> tuple[str, str]:
    """
    Encode file gambar ke base64 dan tentukan tipe mime.
    Returns:
        (b64_string, mime_type) contoh: ("abc...", "image/png")
    """
    path_str = str(image_path)
    ext = os.path.splitext(path_str)[1].lower().lstrip(".")
    mime_sub = "jpeg" if ext in ("jpg", "jpeg") else ext
    mime = f"image/{mime_sub}" if mime_sub else "image/png"
    b64 = encode_image(path_str)
    return b64, mime


def image_data_uri(image_path: str | Path) -> str:
    """Format file gambar menjadi data URI (data:image/...;base64,...)."""
    b64, mime = encode_image_to_base64(image_path)
    return f"data:{mime};base64,{b64}"


def build_chat_model(
    *,
    base_url: str,
    model: str,
    api_key: str,
    temperature: float = 0.1,
    timeout: float = 300,
    enable_thinking: bool | None = None,
    callbacks: list[Any] | None = None,
    **extra_kwargs: Any,
) -> ChatOpenAI:
    """
    Buat instance `ChatOpenAI` generik untuk endpoint OpenAI-compatible.
    """
    params: dict[str, Any] = dict(extra_kwargs)

    # Dukungan model reasoning (mis. Qwen 2.5 / DeepSeek-R1 / Qwen3)
    if enable_thinking is not None:
        # llama.cpp/Qwen chat templates membaca flag ini melalui
        # chat_template_kwargs. Mengirimnya sebagai field top-level membuat
        # model tetap berpikir sampai batas token dan dapat mengembalikan
        # content Markdown kosong dengan finish_reason=length.
        params["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": enable_thinking}
        }

    return ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        timeout=timeout,
        callbacks=callbacks,
        **params,
    )


def build_vlm(settings: Settings | None = None) -> ChatOpenAI:
    """VLM normal (ekstraksi + agent), dengan enable_thinking dari config."""
    resolved = settings or get_settings()
    return build_chat_model(
        base_url=resolved.vlm_base_url,
        model=resolved.vlm_model,
        api_key=resolved.vlm_api_key,
        temperature=resolved.vlm_temperature,
        timeout=resolved.vlm_timeout,
        max_tokens=resolved.vlm_max_tokens,
        enable_thinking=resolved.vlm_enable_thinking,
        callbacks=[LoggingCallbackHandler(resolved.vlm_model, resolved.vlm_base_url)],
    )


def build_ocr(settings: Settings | None = None) -> ChatOpenAI:
    """Model OCR terstruktur; OCR_MODEL wajib terisi sebelum builder dipanggil."""
    resolved = settings or get_settings()
    if not resolved.ocr_model:
        raise ValueError("OCR_MODEL belum dikonfigurasi.")
    return build_chat_model(
        base_url=resolved.ocr_base_url,
        model=resolved.ocr_model,
        api_key=resolved.ocr_api_key,
        temperature=resolved.ocr_temperature,
        timeout=resolved.ocr_timeout,
        max_tokens=resolved.ocr_max_tokens,
        callbacks=[LoggingCallbackHandler(resolved.ocr_model, resolved.ocr_base_url)],
    )


def get_vlm(settings: Settings | None = None) -> ChatOpenAI:
    """Alias/Helper untuk mendapatkan instance VLM."""
    return build_vlm(settings)
