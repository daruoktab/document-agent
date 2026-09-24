"""
Harness Deep Reasoning Agent (Autonomous Orchestrator) untuk ekstraksi dokumen
internal perusahaan: OCR primer + Vision VLM -> Markdown Terstruktur, SQLite tabular,
transkrip chat, form tanda tangan, & diagram Mermaid.js.

Berbeda dengan `app/agents.py` (profil prompt deterministik), file ini membangun
AI agent sungguhan via `deepagents.create_deep_agent`:
  - Master Orchestrator LLM yang MEMUTUSKAN sendiri tool/sub-agent mana yang
    dipanggil berdasarkan konteks dokumen.
  - 7 Sub-Agent terspesialisasi (klasifikasi, ekstraksi, diagram, PPT, PDF,
    SQLite) yang dapat di-summon oleh master.
  - Semua tool terintegrasi dengan flag CLI: `db_path` & `output_markdown_path`
    di-bake ke dalam tool sehingga deep agent menghormati `-o` dan `--db-path`.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from deepagents import SubAgent, create_deep_agent
from langchain_core.tools import tool

from .agents import get_agent
from .config import Settings, get_settings
from .diagram import (
    classify_diagram_convertibility,
)
from .diagram import (
    extract_diagram_to_mermaid as run_extract_diagram,
)
from .docx import process_multipage_docx
from .excel import process_multipage_excel
from .extractor import VisionExtractor
from .graph import DocumentExtractionPipeline
from .llm import build_vlm
from .multi_page import preview_markdown_chunks
from .pdf import process_multipage_pdf
from .ppt import process_presentation_vision
from .preprocess import preprocess_image
from .prompts import MARKDOWN_LINE_BREAK_RULES, MERMAID_EXTRACTION_RULES
from .tabular_db import (
    TabularDatabaseManager,
    TabularVerifier,
    classify_table_heuristic,
    extract_and_ingest_tables_from_markdown,
    parse_markdown_tables,
    query_sqlite,
)


def build_deep_agent(
    settings: Settings | None = None,
    *,
    db_path: str | Path | None = None,
    output_markdown_path: str | Path | None = None,
) -> Any:
    """
    Bangun Deep Reasoning Agent utama dengan armada Sub-Agent spesialis dua-model:
      1. `layout-classifier`          : Mengklasifikasikan multi-trait dokumen
      2. `ocr-markdown-extractor`     : Ekstraksi OCR primer ke Markdown
      3. `diagram-mermaid-specialist` : Evaluasi selektif & ekstraksi diagram ke sintaks Mermaid.js
      4. `presentation-specialist`    : Parsing file presentasi PowerPoint (.pptx / .ppt)
      5. `pdf-orchestrator`           : Orkestrasi multi-halaman PDF & heading continuity
      6. `tabular-db-specialist`      : Deteksi tabel transaksional, ingesti ke SQLite, double-verification, & eksekusi SQL

    Args:
        db_path: Path SQLite target (dari flag CLI --db-path). Semua tool tabular
                 default ke path ini sehingga deep-agent menghormati flag CLI.
        output_markdown_path: Path output Markdown (dari flag CLI -o). Tool PPT/PDF
                              menulis streaming per-halaman ke file ini.
    """
    resolved_settings = settings or get_settings()
    vlm = build_vlm(resolved_settings)
    extractor = VisionExtractor(vlm)
    pipeline = DocumentExtractionPipeline(resolved_settings, vlm=vlm)

    default_db_path = str(db_path) if db_path else None
    default_out_path = str(output_markdown_path) if output_markdown_path else None

    # --- Tool Definitions ---

    @tool
    def classify_layout(image_path: str) -> str:
        """Analisis gambar dokumen dan kembalikan daftar spesifikasi layout yang aktif (plain, markdown_hierarchy, bilingual_journal, presentation_slides, chat_transcript, signature_form)."""
        proc = preprocess_image(image_path)
        specs = extractor.classify(proc.processed_path)
        return json.dumps({"specs": specs}, ensure_ascii=False)

    @tool
    def extract_to_markdown(
        image_path: str,
        specs: str = "plain",
        previous_context: str | None = None,
    ) -> str:
        """Ekstrak gambar menjadi Markdown dengan model OCR primer; otomatis fallback ke VLM utama bila OCR belum aktif atau gagal."""
        proc = preprocess_image(image_path)
        if pipeline.ocr_extractor is not None:
            source = Path(image_path)
            region_root = (
                Path(default_out_path).resolve().parent
                if default_out_path
                else Path("output") / source.stem
            )
            ocr_result = pipeline.ocr_extractor.extract_robust(
                proc.processed_path,
                output_dir=region_root / "regions" / "deep_agent",
            )
            if (
                ocr_result.status == "success"
                and ocr_result.trust_level == "high"
                and ocr_result.decision != "blank_page"
                and ocr_result.markdown.strip()
            ):
                return ocr_result.markdown
            if ocr_result.decision == "blank_page":
                return ""

        agent = get_agent(specs)
        return agent.run(
            proc.processed_path,
            llm=vlm,
            previous_page_context=previous_context,
        )

    @tool
    def classify_diagram_suitability(image_path: str) -> str:
        """Evaluasi apakah visual merupakan keluarga flowchart yang boleh menjadi Mermaid."""
        proc = preprocess_image(image_path)
        res = classify_diagram_convertibility(proc.processed_path, llm=vlm)
        payload = res.model_dump()
        rendered_bytes = payload.pop("rendered_image_bytes", None)
        if rendered_bytes is not None:
            payload["rendered_image_size_bytes"] = len(rendered_bytes)
        return json.dumps(payload, indent=2, ensure_ascii=False)

    @tool
    def extract_diagram_to_mermaid(
        image_path: str,
        diagram_hint: str | None = None,
    ) -> str:
        """Ekstrak flowchart menjadi Mermaid atau deskripsikan visual selain flowchart."""
        proc = preprocess_image(image_path)
        res = run_extract_diagram(
            proc.processed_path, llm=vlm, forced_diagram_type=diagram_hint
        )
        payload = res.model_dump()
        rendered_bytes = payload.pop("rendered_image_bytes", None)
        if rendered_bytes is not None:
            payload["rendered_image_size_bytes"] = len(rendered_bytes)
        return json.dumps(payload, indent=2, ensure_ascii=False)

    @tool
    def extract_presentation_pptx(pptx_path: str) -> str:
        """Ekstrak dokumen presentasi PowerPoint (.pptx/.ppt) dengan merender tiap slide menjadi gambar kanvas visual lalu dianalisis oleh VLM. Output streaming per-slide ke file Markdown target."""
        res = process_presentation_vision(
            pptx_path,
            pipeline=pipeline,
            db_path=default_db_path,
            output_markdown_path=default_out_path,
        )
        return res if isinstance(res, str) else res.full_markdown

    @tool
    def extract_pdf_document(
        pdf_path: str,
        forced_specs: str | None = None,
    ) -> str:
        """Ekstrak dokumen PDF multi-halaman dengan heading continuity, ekstraksi tabel mandiri per-halaman ke SQLite, dan audit guardrail jalur ganda. Output streaming per-halaman ke file Markdown target."""
        res = process_multipage_pdf(
            pdf_path,
            pipeline=pipeline,
            forced_specs=forced_specs,
            auto_tabular_db=True,
            db_path=default_db_path,
            output_markdown_path=default_out_path,
        )
        return res.full_markdown

    @tool
    def extract_docx_document(
        docx_path: str,
        forced_specs: str | None = None,
    ) -> str:
        """Ekstrak dokumen DOCX/DOC dengan mengonversinya ke PDF, merender tiap halaman menjadi gambar, lalu menjalankan pipeline VLM dan SQLite yang sama seperti PDF."""
        res = process_multipage_docx(
            docx_path,
            pipeline=pipeline,
            forced_specs=forced_specs,
            auto_tabular_db=True,
            db_path=default_db_path,
            output_markdown_path=default_out_path,
        )
        return res.full_markdown

    @tool
    def extract_excel_document(
        excel_path: str,
        forced_specs: str | None = None,
    ) -> str:
        """Ekstrak workbook Excel/ODS dengan mengonversinya ke PDF, merender tiap halaman menjadi gambar, lalu menjalankan pipeline VLM dan SQLite yang sama seperti PDF."""
        res = process_multipage_excel(
            excel_path,
            pipeline=pipeline,
            forced_specs=forced_specs,
            auto_tabular_db=True,
            db_path=default_db_path,
            output_markdown_path=default_out_path,
        )
        return res.full_markdown

    @tool
    def preview_chunks(
        markdown_text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> str:
        """Simulasikan pemecahan teks Markdown hasil ekstraksi menjadi potongan-potongan chunk siap indeks RAG."""
        preview = preview_markdown_chunks(
            markdown_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return json.dumps(asdict(preview), indent=2, ensure_ascii=False)

    @tool
    def classify_table_storage(markdown_text: str) -> str:
        """Analisis tabel-tabel pada teks Markdown untuk membedakan mana yang bertipe transaksional/finansial (layak SQLite) vs tabel naratif kualitatif."""
        tables = parse_markdown_tables(markdown_text)
        results = []
        for idx, t in enumerate(tables, start=1):
            cls_res = classify_table_heuristic(
                headers=t["headers"],
                rows=t["rows"],
                context=t["context"],
            )
            results.append({"table_index": idx, "classification": cls_res.model_dump()})
        return json.dumps(results, indent=2, ensure_ascii=False)

    @tool
    def ingest_tables_to_sqlite(
        markdown_text: str,
        db_path: str | None = None,
        table_name_prefix: str | None = None,
        force_all: bool = False,
    ) -> str:
        """Ekstrak tabel-tabel transaksional dari teks Markdown, buat skema otomatis, ingest ke database SQLite, dan lakukan verifikasi ganda."""
        results = extract_and_ingest_tables_from_markdown(
            markdown_text=markdown_text,
            db_path=db_path or default_db_path,
            table_name_prefix=table_name_prefix,
            force_all_tables=force_all,
            llm=vlm,
        )
        return json.dumps(
            [r.model_dump() for r in results], indent=2, ensure_ascii=False
        )

    @tool
    def inspect_sqlite_tables(db_path: str | None = None) -> str:
        """Inspeksi database SQLite dokumen untuk melihat daftar seluruh tabel aktif, struktur skema kolom, jumlah baris, dan sampel data."""
        mgr = TabularDatabaseManager(db_path or default_db_path)
        info = mgr.inspect_database()
        return json.dumps(info, indent=2, ensure_ascii=False)

    @tool
    def query_sqlite_database(query: str, db_path: str | None = None) -> str:
        """Eksekusi query SELECT analitik SQL pada database SQLite dokumen secara aman."""
        res = query_sqlite(query, db_path=db_path or default_db_path)
        return json.dumps(res.model_dump(), indent=2, ensure_ascii=False)

    @tool
    def verify_table_data_integrity(
        table_name: str,
        expected_rows: int | None = None,
        db_path: str | Path | None = None,
    ) -> str:
        """Lakukan audit verifikasi ganda (double-verification) pada tabel SQLite (integritas baris, skema, agregasi SUM/AVG, dan kontinuitas saldo transaksi)."""
        mgr = TabularDatabaseManager(db_path or default_db_path)
        verifier = TabularVerifier(mgr, llm=vlm)
        report = verifier.verify_table(table_name, expected_row_count=expected_rows)
        return json.dumps(report.model_dump(), indent=2, ensure_ascii=False)

    @tool
    def judge_and_refine_markdown(
        image_path: str, draft_markdown: str, specs: str = "plain"
    ) -> str:
        """Lakukan audit verifikasi & koreksi ulang (Judge & Self-Correction) dengan membandingkan draft gabungan Markdown terhadap citra asli dokumen."""
        proc = preprocess_image(image_path)
        refined = extractor.judge_and_refine(
            proc.processed_path, draft_markdown, specs=specs.split(",")
        )
        return refined

    tools = [
        classify_layout,
        extract_to_markdown,
        classify_diagram_suitability,
        extract_diagram_to_mermaid,
        extract_presentation_pptx,
        extract_pdf_document,
        extract_docx_document,
        extract_excel_document,
        classify_table_storage,
        ingest_tables_to_sqlite,
        inspect_sqlite_tables,
        query_sqlite_database,
        verify_table_data_integrity,
        judge_and_refine_markdown,
    ]

    # --- Sub-Agent Definitions ---
    subagents = [
        SubAgent(
            name="layout-classifier",
            description="Sub-agent untuk mengidentifikasi dan mengklasifikasikan karakteristik layout dokumen.",
            system_prompt=(
                "Anda adalah Sub-Agent Spesialis Klasifikasi Dokumen Internal Perusahaan. "
                "Tugas Anda: Analisis citra dan identifikasi seluruh karakteristik dokumen. "
                "Spesifikasi valid: plain, markdown_hierarchy, bilingual_journal, presentation_slides, "
                "chat_transcript, signature_form. "
                "Dokumen perusahaan dapat memiliki beberapa spesifikasi sekaligus "
                "(mis. slide presentasi + form tanda tangan, atau artikel multi-kolom + tabel data). "
                "Gunakan tool 'classify_layout' untuk menentukan spesifikasi."
            ),
            tools=[classify_layout],
        ),
        SubAgent(
            name="ocr-markdown-extractor",
            description="Sub-agent OCR primer untuk mengekstrak halaman menjadi Markdown dan region visual/tabel.",
            system_prompt=(
                "Anda adalah Sub-Agent OCR Spesialis Ekstraksi Markdown untuk dokumen internal perusahaan. "
                "Gunakan tool 'extract_to_markdown'; model OCR adalah sumber draft primer dan VLM utama "
                "hanya menjadi fallback bila OCR belum dikonfigurasi atau gagal. "
                "Ubah citra dokumen menjadi teks Markdown bersih dan terstruktur. "
                "Pertahankan hierarki heading, list, transkrip percakapan chat, tabel form persetujuan, "
                "dan konteks antar-halaman. "
                "Spesifikasi yang mungkin aktif: plain, markdown_hierarchy, bilingual_journal, "
                "presentation_slides, chat_transcript, signature_form."
            ),
            tools=[extract_to_markdown],
        ),
        SubAgent(
            name="diagram-mermaid-specialist",
            description="Sub-agent untuk mengevaluasi kelayakan diagram dan mengekstraknya menjadi kode Mermaid.js atau deskripsi.",
            system_prompt=(
                "Anda adalah Sub-Agent Spesialis Diagram & Visual Artifacts. "
                "Tugas Anda: Konversi hanya flowchart, workflow, swimlane, dan decision tree yang jelas "
                "ke blok Mermaid.js menggunakan tool 'extract_diagram_to_mermaid'. "
                "Sequence, ERD, class/state, arsitektur, mindmap, grafik, peta, foto, dan visual lain wajib menjadi deskripsi terstruktur."
            ),
            tools=[classify_diagram_suitability, extract_diagram_to_mermaid],
        ),
        SubAgent(
            name="presentation-specialist",
            description="Sub-agent untuk memproses presentasi PowerPoint (.pptx/.ppt) secara visual per slide.",
            system_prompt=(
                "Anda adalah Sub-Agent Spesialis Presentasi PowerPoint (.pptx / .ppt). "
                "Tugas Anda: Render tiap slide menjadi gambar kanvas visual lalu ekstrak teks judul, poin peluru, diagram, dan tabel secara terstruktur."
            ),
            tools=[extract_presentation_pptx],
        ),
        SubAgent(
            name="pdf-orchestrator",
            description="Sub-agent untuk orkestrasi pemrosesan multi-halaman PDF dengan heading continuity.",
            system_prompt=(
                "Anda adalah Sub-Agent Spesialis Dokumen PDF Multi-Halaman. "
                "Tugas Anda: Proses PDF halaman demi halaman secara berurutan, jaga kesinambungan heading antar-halaman, dan gabungkan hasilnya."
            ),
            tools=[extract_pdf_document],
        ),
        SubAgent(
            name="docx-orchestrator",
            description="Sub-agent untuk orkestrasi pemrosesan DOCX/DOC multi-halaman melalui render visual per halaman.",
            system_prompt=(
                "Anda adalah Sub-Agent Spesialis Dokumen Word (.docx / .doc). "
                "Tugas Anda: Konversi dokumen ke PDF sementara, proses halaman demi halaman sebagai gambar, "
                "jaga kesinambungan heading antar-halaman, dan gabungkan hasilnya."
            ),
            tools=[extract_docx_document],
        ),
        SubAgent(
            name="excel-orchestrator",
            description="Sub-agent untuk orkestrasi pemrosesan Excel/ODS multi-halaman melalui render visual per halaman.",
            system_prompt=(
                "Anda adalah Sub-Agent Spesialis Workbook Excel/ODS. "
                "Tugas Anda: Konversi workbook ke PDF sementara, proses halaman hasil cetak spreadsheet sebagai gambar, "
                "jaga struktur sheet/tabel, dan gabungkan hasilnya."
            ),
            tools=[extract_excel_document],
        ),
        SubAgent(
            name="tabular-db-specialist",
            description="Sub-agent untuk deteksi tabel transaksional, ingesti ke SQLite, double-verification, dan eksekusi query SQL.",
            system_prompt=(
                "Anda adalah Sub-Agent Spesialis Tabular SQLite Database & Data Integrity Auditor. \n"
                "Tugas Anda:\n"
                "1. Analisis tabel pada Markdown dan pilah mana yang bertipe transaksional/finansial ('classify_table_storage').\n"
                "2. Ingest tabel transaksional ke database SQLite dokumen ('ingest_tables_to_sqlite').\n"
                "3. Lakukan audit verifikasi ganda integritas baris dan kalkulasi agregat ('verify_table_data_integrity').\n"
                "4. Lakukan inspeksi skema ('inspect_sqlite_tables') dan eksekusi query SQL bila diminta ('query_sqlite_database')."
            ),
            tools=[
                classify_table_storage,
                ingest_tables_to_sqlite,
                inspect_sqlite_tables,
                query_sqlite_database,
                verify_table_data_integrity,
            ],
        ),
    ]

    master_system_prompt = (
        "Anda adalah Master Orchestrator Deep Reasoning Agent untuk Sistem Ekstraksi Dokumen Internal Perusahaan "
        "(OCR Primer + Vision VLM -> Markdown Terstruktur, Tabular SQLite, & Mermaid).\n\n"
        "Karakteristik dokumen yang mungkin ditemui: surat/memo/pengumuman (plain), SOP/SK/kebijakan (markdown_hierarchy), "
        "artikel internal multi-kolom (bilingual_journal), slide presentasi (presentation_slides), "
        "screenshot chat (chat_transcript), form tanda tangan/paraf (signature_form).\n\n"
        "Anda mengorkestrasi 8 Sub-Agent spesialis:\n"
        "  - 'layout-classifier'         : Menentukan tipe dokumen & karakteristik komposit.\n"
        "  - 'ocr-markdown-extractor'    : Mengonversi halaman menjadi draft Markdown via OCR primer.\n"
        "  - 'diagram-mermaid-specialist': Membuat Mermaid untuk flowchart dan deskripsi untuk visual lainnya.\n"
        "  - 'presentation-specialist'   : Menangani slide PPT/PPTX visual.\n"
        "  - 'pdf-orchestrator'          : Mengelola multi-halaman PDF dengan heading continuity.\n"
        "  - 'docx-orchestrator'         : Mengelola multi-halaman DOCX/DOC lewat render visual per halaman.\n"
        "  - 'excel-orchestrator'        : Mengelola workbook Excel/ODS lewat render visual per halaman.\n"
        "  - 'tabular-db-specialist'     : Memisahkan tabel transaksional ke SQLite dan melakukan double-verification.\n\n"
        "Instruksi Kerja:\n"
        f"{MARKDOWN_LINE_BREAK_RULES}\n\n"
        f"{MERMAID_EXTRACTION_RULES}\n\n"
        "1. Identifikasi format dokumen masukan (PDF, DOCX/DOC, Excel, PPTX, gambar tunggal).\n"
        "2. Gunakan 'ocr-markdown-extractor' sebagai sumber draft teks utama, lalu delegasikan tugas lanjutan. "
        "Contoh: 'diagram-mermaid-specialist' jika ada diagram/topologi, "
        "'tabular-db-specialist' jika ada tabel data transaksional.\n"
        "3. Gabungkan hasil ekstraksi teks dengan blok Mermaid dan tabel.\n"
        "4. Lakukan tahap Judge / Koreksi Ulang ('judge_and_refine_markdown') untuk memverifikasi bahwa "
        "Markdown gabungan benar-benar merefleksikan seluruh isi visual dokumen tanpa ada yang terlewat.\n"
        "5. Sajikan hasil ekstraksi akhir yang rapi, lengkap dengan laporan database SQLite dan blok kode Mermaid bila ada.\n\n"
        "CATATAN PENTING:\n"
        "- Output Markdown dan database SQLite sudah dikonfigurasi oleh sistem berdasarkan flag CLI. "
        "Tool DOCX/Excel/PPT/PDF dan tabular akan otomatis menulis ke path tersebut.\n"
        "- Jangan menambahkan reasoning, komentar proses, atau marker halaman/slide ke dalam output."
    )

    agent = create_deep_agent(
        model=vlm,
        tools=tools,
        subagents=subagents,
        system_prompt=master_system_prompt,
    )
    return agent


def run_deep_reasoning_agent(
    prompt: str,
    settings: Settings | None = None,
    *,
    db_path: str | Path | None = None,
    output_markdown_path: str | Path | None = None,
) -> str:
    """Eksekusi Deep Reasoning Agent dengan instruksi prompt pengguna."""
    agent = build_deep_agent(
        settings,
        db_path=db_path,
        output_markdown_path=output_markdown_path,
    )
    res = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    return res.get("messages", [])[-1].content if res.get("messages") else ""
