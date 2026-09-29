"""Penyusunan Markdown berbasis teks setelah hasilnya diaudit secara visual."""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from .multi_page import strip_page_markers, strip_thinking_process
from .tabular_db import parse_markdown_tables

logger = logging.getLogger("app.language_refiner")

_MERMAID_RE = re.compile(r"```mermaid\s*([\s\S]*?)\s*```", re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<![\w])\d[\d.,:/-]*(?![\w])")
_META_PHRASES = (
    "mari kita",
    "draft markdown yang diberikan",
    "berikut adalah markdown",
    "hasil audit visual",
)


def _unwrap_markdown(content: Any) -> str:
    text = strip_thinking_process(str(content or "").strip())
    if text.startswith("```markdown\n") and text.endswith("```"):
        text = text[len("```markdown\n") : -3].strip()
    elif text.startswith("```md\n") and text.endswith("```"):
        text = text[len("```md\n") : -3].strip()
    return strip_page_markers(text).strip()


def _safe_revision(audited: str, revised: str) -> bool:
    """Tolak perubahan pada data terstruktur dan pemangkasan/pembengkakan ekstrem."""
    if not revised or any(phrase in revised.lower() for phrase in _META_PHRASES):
        return False
    if _MERMAID_RE.findall(audited) != _MERMAID_RE.findall(revised):
        return False
    audited_table_lines = [
        line.strip() for line in audited.splitlines() if line.lstrip().startswith("|")
    ]
    revised_table_lines = [
        line.strip() for line in revised.splitlines() if line.lstrip().startswith("|")
    ]
    if audited_table_lines != revised_table_lines:
        return False
    audited_tables = parse_markdown_tables(audited)
    revised_tables = parse_markdown_tables(revised)
    if [(t["headers"], t["rows"]) for t in audited_tables] != [
        (t["headers"], t["rows"]) for t in revised_tables
    ]:
        return False
    if Counter(_NUMBER_RE.findall(audited)) != Counter(_NUMBER_RE.findall(revised)):
        return False
    return len(audited) <= 500 or 0.55 <= len(revised) / len(audited) <= 1.5


def refine_audited_markdown(
    *,
    llm: BaseChatModel,
    draft_markdown: str,
    audited_markdown: str,
    previous_page_context: str | None = None,
    native_text: str | None = None,
) -> tuple[str, str]:
    """Perbaiki struktur teks tanpa gambar; kembalikan audit Gemma bila gagal."""
    if not audited_markdown.strip():
        return audited_markdown, "skipped_empty"

    context = (previous_page_context or "").strip()[-500:]
    native = (native_text or "").strip()[:12000]
    prompt = (
        "Susun Markdown final dari hasil audit visual berikut. Hasil audit visual adalah "
        "acuan untuk semua fakta yang terlihat di dokumen. Draft awal hanya konteks. "
        "Perbaiki kejelasan struktur heading, list, paragraf, dan kesinambungan kalimat "
        "tanpa menambah atau menghilangkan fakta. Pertahankan urutan isi, kutipan, nama, "
        "angka, tanggal, setiap sel tabel, dan setiap blok Mermaid. Jangan menulis pengantar "
        "atau komentar proses. Keluarkan hanya Markdown final.\n\n"
        f"DRAFT AWAL:\n```markdown\n{draft_markdown}\n```\n\n"
        f"HASIL AUDIT VISUAL:\n```markdown\n{audited_markdown}\n```\n\n"
        f"KONTEKS HALAMAN SEBELUMNYA:\n{context}\n\n"
        f"BUKTI NATIVE:\n{native}"
    )
    try:
        response = llm.invoke(prompt)
        revised = _unwrap_markdown(getattr(response, "content", response))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[LanguageRefine] VLM bahasa gagal (%s); mempertahankan audit visual.", exc)
        return audited_markdown, "fallback_error"

    if not _safe_revision(audited_markdown, revised):
        logger.warning("[LanguageRefine] Hasil VLM bahasa melanggar integritas; mempertahankan audit visual.")
        return audited_markdown, "fallback_invalid"
    return revised, "accepted"
