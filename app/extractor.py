"""
Pipeline ekstraksi: gambar dokumen -> VLM -> Markdown Bersih Siap Chunking.
Mendukung multi-spesifikasi karakteristik tata letak dokumen secara komposit dengan logging transparan.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Literal, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .llm import image_data_uri
from .multi_page import (
    collapse_consecutive_duplicate_blocks,
    strip_page_markers,
    strip_thinking_process,
)
from .prompts import (
    CLASSIFY_PROMPT,
    CLASSIFY_SYSTEM,
    MARKDOWN_LINE_BREAK_RULES,
    MERMAID_EXTRACTION_RULES,
    SYSTEM_DOCUMENT_EXTRACTOR,
    build_extraction_prompt,
    normalize_specs,
)
from .schemas import PageInspectionResult

logger = logging.getLogger("app.extractor")


class EmptyLLMResponseError(RuntimeError):
    """LLM berhasil dipanggil tetapi tidak menghasilkan konten yang dapat dipakai."""


class VisionExtractor:
    """Ekstraktor dokumen multimodal: Mengubah gambar dokumen menjadi Markdown siap chunking."""

    def __init__(
        self,
        llm: BaseChatModel,
        system_prompt: str = SYSTEM_DOCUMENT_EXTRACTOR,
    ) -> None:
        self.llm: BaseChatModel = llm
        self.system_prompt: str = system_prompt
        self.learning_version = "baseline"
        self.learning_program: Any = None
        self.learning_lm: Any = None
        from .learning_store import LearningStore
        self.learning_store = LearningStore()
        active = self.learning_store.active_run()
        if active and system_prompt == SYSTEM_DOCUMENT_EXTRACTOR:
            from .dspy_learning import PageProgram, build_learning_lm, model_fingerprint
            if active["model_fingerprint"] == model_fingerprint():
                self.learning_program = PageProgram(active["instructions"])
                self.learning_lm = build_learning_lm()
                self.learning_version = active["id"]
            else:
                logger.warning("Konfigurasi model berubah; memakai prompt bawaan.")

    def classify(self, image_path: str) -> list[str]:
        """
        Klasifikasikan satu atau lebih karakteristik layout dokumen yang ada pada gambar:
        ['plain'], ['bilingual_journal', 'markdown_hierarchy'], ['chat_transcript'], dsb.
        """
        logger.debug(
            "[Extractor:Classify] Menyiapkan payload klasifikasi untuk: %s", image_path
        )
        content: list[dict[str, Any]] = [
            {"type": "text", "text": CLASSIFY_PROMPT},
            {"type": "image_url", "image_url": {"url": image_data_uri(image_path)}},
        ]
        messages = [
            SystemMessage(content=CLASSIFY_SYSTEM),
            HumanMessage(content=cast(Any, content)),
        ]

        resp = self.llm.invoke(messages)
        text_resp = strip_thinking_process(str(resp.content).strip())
        logger.debug("[Extractor:Classify] Respon mentah VLM: %s", text_resp)

        # Parse JSON output jika ada
        match = re.search(r"\{.*?\}", text_resp, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                raw_specs = data.get("specs") or [data.get("doc_type")]
                normalized = normalize_specs(raw_specs)
                logger.info(
                    "[Extractor:Classify] Berhasil parse JSON layout: %s", normalized
                )
                return normalized
            except (json.JSONDecodeError, TypeError, ValueError, KeyError) as e:
                logger.warning(
                    "[Extractor:Classify] Gagal parse JSON layout (%s): '%s'",
                    e,
                    match.group(0),
                )

        # Fallback multi-matching via regex
        text_lower = text_resp.lower()
        detected: list[str] = []
        if "journal" in text_lower or "bilingual" in text_lower or "2col" in text_lower:
            detected.append("bilingual_journal")
        if (
            "hierarchy" in text_lower
            or "markdown" in text_lower
            or "heading" in text_lower
        ):
            detected.append("markdown_hierarchy")
        if "slide" in text_lower or "presentation" in text_lower or "ppt" in text_lower:
            detected.append("presentation_slides")
        if (
            "chat" in text_lower
            or "whatsapp" in text_lower
            or "telegram" in text_lower
            or "percakapan" in text_lower
        ):
            detected.append("chat_transcript")
        if (
            "signature" in text_lower
            or "paraf" in text_lower
            or "approval" in text_lower
            or "persetujuan" in text_lower
            or "tanda tangan" in text_lower
        ):
            detected.append("signature_form")

        final_specs = detected if detected else ["plain"]
        logger.info("[Extractor:Classify] Fallback regex layout: %s", final_specs)
        return final_specs

    def inspect_page(
        self,
        image_path: str,
        *,
        is_first_page: bool = False,
    ) -> PageInspectionResult:
        """
        Inspeksi komprehensif layout dokumen & elemen visual (diagram, tabel, hierarki).
        Jika is_first_page=True, juga mengekstrak judul utama dokumen secara eksplisit.
        Mengembalikan PageInspectionResult berisi specs, has_diagram, diagram_type, has_table, dan document_title.
        """
        title_instruction = (
            "6. document_title: Judul utama dokumen jika ini adalah halaman 1 / sampul / halaman judul, atau null jika tidak terlihat jelas.\n"
            if is_first_page
            else ""
        )
        title_json_field = (
            '  "document_title": "string" atau null,\n' if is_first_page else ""
        )

        inspect_prompt = (
            "Analisis gambar dokumen ini secara menyeluruh untuk mendeteksi karakteristik tata letak dan elemen visual.\n\n"
            "Evaluasi hal berikut:\n"
            "1. specs: Karakteristik dokumen (pilih dari: 'plain', 'markdown_hierarchy', 'bilingual_journal', 'presentation_slides', 'chat_transcript', 'signature_form').\n"
            "   - 'plain': dokumen bisnis umum (surat, memo, pengumuman, formulir sederhana).\n"
            "   - 'markdown_hierarchy': dokumen terstruktur (SOP, SK, kebijakan, peraturan, perjanjian, laporan formal).\n"
            "   - 'bilingual_journal': artikel internal / dokumen multi-kolom / dokumen dua bahasa.\n"
            "   - 'presentation_slides': slide presentasi (PowerPoint/PDF landscape, bullet points).\n"
            "   - 'chat_transcript': screenshot percakapan chat (WhatsApp, Telegram, chat internal).\n"
            "   - 'signature_form': dokumen dengan kotak tanda tangan, paraf, approval, atau persetujuan.\n"
            "2. has_diagram: true jika terdapat diagram visual (flowchart, alur proses, sequence diagram, ERD, arsitektur blok, peta memori/register map, mindmap, state diagram, org chart, bagan teknis, figure/skema), false jika hanya teks biasa atau foto polos.\n"
            "3. diagram_type: Tipe diagram jika has_diagram=true (contoh: 'flowchart', 'sequence_diagram', 'er_diagram', 'block_architecture', 'memory_map', 'mindmap', dll., atau null jika tidak ada).\n"
            "4. has_table: true jika terdapat tabel data/baris kolom.\n"
            "5. difficulty: Tingkat kesulitan ekstraksi halaman.\n"
            "   - 'simple': teks polos, sedikit elemen, tanpa diagram/tabel/kompleksitas.\n"
            "   - 'standard': ada struktur (list, heading, tabel sederhana).\n"
            "   - 'complex': ada diagram/topologi, multi-kolom, chat, form tanda tangan, atau teks padat.\n"
            f"{title_instruction}\n"
            "Outputkan HANYA format JSON valid tanpa pengantar:\n"
            "{\n"
            '  "specs": ["spec1", "spec2"],\n'
            '  "has_diagram": true/false,\n'
            '  "diagram_type": "string" atau null,\n'
            '  "has_table": true/false,\n'
            '  "difficulty": "simple" atau "standard" atau "complex",\n'
            f"{title_json_field}"
            '  "reasoning": "penjelasan singkat"\n'
            "}"
        )

        content: list[dict[str, Any]] = [
            {"type": "text", "text": inspect_prompt},
            {"type": "image_url", "image_url": {"url": image_data_uri(image_path)}},
        ]
        messages = [
            SystemMessage(content=CLASSIFY_SYSTEM),
            HumanMessage(content=cast(Any, content)),
        ]

        try:
            resp = self.llm.invoke(messages)
            text_resp = strip_thinking_process(str(resp.content).strip())
            match = re.search(r"\{.*?\}", text_resp, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                raw_specs = data.get("specs") or [data.get("doc_type")]
                norm_specs = normalize_specs(raw_specs)
                diff_val = str(data.get("difficulty", "standard")).lower()
                clean_difficulty: Literal["simple", "standard", "complex"] = (
                    diff_val if diff_val in ("simple", "standard", "complex") else "standard"
                )
                doc_title = data.get("document_title")
                if isinstance(doc_title, str):
                    doc_title = doc_title.strip() or None
                return PageInspectionResult(
                    specs=norm_specs,
                    has_diagram=bool(data.get("has_diagram", False)),
                    diagram_type=data.get("diagram_type"),
                    has_table=bool(data.get("has_table", False)),
                    difficulty=clean_difficulty,
                    reasoning=data.get("reasoning"),
                    document_title=doc_title,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("[Extractor:Inspect] Gagal inspect JSON (%s), fallback ke classify biasa.", e)

        # Fallback
        specs = self.classify(image_path)
        return PageInspectionResult(
            specs=specs,
            has_diagram=False,
            diagram_type=None,
            has_table=False,
            difficulty="standard",
            document_title=None,
        )

    def judge_and_refine(
        self,
        image_path: str,
        draft_markdown: str,
        *,
        specs: list[str] | str | None = None,
    ) -> str:
        """
        Tahap Aggregator Judge & Self-Correction (Koreksi Ulang):
        Membandingkan draft gabungan Markdown (teks + Mermaid + tabel) terhadap citra asli dokumen.
        Memperbaiki kekurangan atau mempertahankan draft jika sudah akurat dan lengkap.
        """
        if not draft_markdown or not draft_markdown.strip():
            return draft_markdown

        judge_prompt = (
            "Periksa DRAFT MARKDOWN berikut terhadap GAMBAR ASLI DOKUMEN.\n\n"
            "Panduan verifikasi:\n"
            f"{MARKDOWN_LINE_BREAK_RULES}\n\n"
            "1. KELENGKAPAN: Pastikan seluruh teks, judul, dan data pada gambar tercakup akurat.\n"
            f"2. DIAGRAM VISUAL:\n{MERMAID_EXTRACTION_RULES}\n"
            "3. INTEGRITAS TABEL: Pastikan tabel Markdown (GFM) utuh tanpa baris kosong di tengah.\n"
            "4. KEBERSIHAN: Hapus teks duplikat berulang jika ada.\n"
            "5. JIKA DRAFT SUDAH LENGKAP & BENAR: Kembalikan teks DRAFT MARKDOWN secara persis tanpa perubahan.\n\n"
            "ATURAN KETAT:\n"
            "- DILARANG menuliskan kata pengantar, analisis, komentar proses, evaluasi, atau catatan.\n"
            "- DILARANG menuliskan frasa seperti 'Mari kita...', 'Koreksi:', 'Perbaikan teks:', 'Aturan...', 'Draft:...'.\n"
            "- Outputkan HANYA teks Markdown dokumen final tanpa embel-embel apapun.\n\n"
            f"[DRAFT MARKDOWN]:\n'''markdown\n{draft_markdown}\n'''"
        )

        content: list[dict[str, Any]] = [
            {"type": "text", "text": judge_prompt},
            {"type": "image_url", "image_url": {"url": image_data_uri(image_path)}},
        ]
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=cast(Any, content)),
        ]

        try:
            resp = self.llm.invoke(messages)
            raw_resp = str(resp.content).strip()

            # Guardrail 1: Deteksi kebocoran meta-evaluasi / reasoning CoT
            meta_leak_indicators = (
                "mari kita",
                "ini melanggar aturan",
                "aturan 10",
                "koreksi draft",
                "perbaikan teks:",
                "draft markdown yang diberikan",
                "langkah demi langkah",
            )
            lower_raw = raw_resp.lower()
            for ind in meta_leak_indicators:
                if ind in lower_raw:
                    logger.warning(
                        "[Extractor:Judge] Respon judge mengandung teks penalaran / meta-evaluasi ('%s'). Mempertahankan draft awal.",
                        ind,
                    )
                    return draft_markdown

            refined_md = strip_thinking_process(raw_resp)
            # Bersihkan wrapper code fence jika ada
            if refined_md.startswith("```markdown") and refined_md.endswith("```"):
                refined_md = refined_md[len("```markdown") : -3].strip()
            elif refined_md.startswith("```md") and refined_md.endswith("```"):
                refined_md = refined_md[len("```md") : -3].strip()
            elif (
                refined_md.startswith("```")
                and refined_md.endswith("```")
                and not refined_md.startswith("```mermaid")
            ):
                refined_md = refined_md[3:-3].strip()

            refined_md = strip_page_markers(refined_md)
            refined_md = collapse_consecutive_duplicate_blocks(refined_md)

            if not refined_md:
                logger.warning("[Extractor:Judge] Hasil judge kosong. Menggunakan draft awal.")
                return draft_markdown

            # Judge tidak boleh menghilangkan satu pun diagram dari draft.
            mermaid_fence = r"^\s*```mermaid\b"
            draft_diagrams = len(re.findall(mermaid_fence, draft_markdown, re.MULTILINE | re.IGNORECASE))
            refined_diagrams = len(re.findall(mermaid_fence, refined_md, re.MULTILINE | re.IGNORECASE))
            if refined_diagrams < draft_diagrams:
                logger.warning("[Extractor:Judge] Blok Mermaid berkurang (%d -> %d). Mempertahankan draft.", draft_diagrams, refined_diagrams)
                return draft_markdown

            # Guardrail 2: Cegah pembengkakan liar akibat halusinasi / perulangan
            len_draft = len(draft_markdown)
            len_refined = len(refined_md)
            if len_draft > 500 and len_refined > 1.8 * len_draft:
                logger.warning(
                    "[Extractor:Judge] Hasil judge membengkak secara tidak wajar (%d -> %d karakter). Menggunakan draft awal demi keselamatan data.",
                    len_draft,
                    len_refined,
                )
                return draft_markdown

            # Guardrail 3: Cegah pemangkasan drastis yang memotong dokumen (mis. kepotong token limit)
            if len_draft > 800 and len_refined < 0.45 * len_draft:
                logger.warning(
                    "[Extractor:Judge] Hasil judge terpotong drastis (%d -> %d karakter). Menggunakan draft awal.",
                    len_draft,
                    len_refined,
                )
                return draft_markdown

            return refined_md
        except Exception as e:  # noqa: BLE001
            logger.warning("[Extractor:Judge] Terjadi kesalahan pada tahap judge & refine (%s). Menggunakan draft awal.", e)
            return draft_markdown

    def extract_markdown(
        self,
        image_path: str,
        *,
        specs: list[str] | str | None = None,
        previous_page_context: str | None = None,
    ) -> str:
        """
        Ekstraksi Markdown visual sadar spesifikasi komposit.

        Args:
            image_path: Path absolut ke gambar halaman dokumen.
            specs: Spesifikasi layout aktif (mis. ['plain'], ['presentation_slides'], dll.).
            previous_page_context: Potongan teks dari halaman sebelumnya untuk kesinambungan heading/kalimat.
        """
        user_prompt = build_extraction_prompt(
            specs=specs,
            previous_page_context=previous_page_context,
        )

        logger.debug(
            "[Extractor:Markdown] Menyusun prompt ekstraksi (panjang prompt: %d karakter, konteks lalu: %s)",
            len(user_prompt),
            bool(previous_page_context),
        )

        content: list[dict[str, Any]] = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": image_data_uri(image_path)}},
        ]

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=cast(Any, content)),
        ]

        if self.learning_program is not None:
            import dspy

            from .dspy_learning import predict_page
            raw_content = predict_page(self.learning_program, self.learning_lm,
                                       dspy.Image.from_path(image_path), user_prompt)
        else:
            response = self.llm.invoke(messages)
            raw_content = str(response.content or "").strip()
        md_text = strip_thinking_process(raw_content)

        # Bersihkan pembungkus markdown block ```markdown ... ``` jika VLM membungkusnya
        if md_text.startswith("```markdown") and md_text.endswith("```"):
            md_text = md_text[len("```markdown") : -3].strip()
        elif md_text.startswith("```md") and md_text.endswith("```"):
            md_text = md_text[len("```md") : -3].strip()
        elif md_text.startswith("```") and md_text.endswith("```"):
            md_text = md_text[3:-3].strip()

        md_text = strip_page_markers(md_text)
        md_text = collapse_consecutive_duplicate_blocks(md_text)
        if not md_text.strip():
            logger.error(
                "[Extractor:Markdown] Respons LLM kosong setelah sanitasi untuk image=%s "
                "(raw_chars=%d). Proses dihentikan agar output kosong tidak dianggap sukses.",
                image_path,
                len(raw_content),
            )
            raise EmptyLLMResponseError(
                f"Respons LLM kosong saat mengekstrak Markdown: {image_path}"
            )
        if Path(image_path).is_file():
            try:
                self.learning_store.observe(image_path, md_text, user_prompt, self.learning_version)
            except (OSError, ValueError, sqlite3.Error) as exc:
                logger.warning("Metadata koreksi tidak tersimpan: %s", type(exc).__name__)
        return md_text
