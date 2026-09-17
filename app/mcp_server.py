"""
MCP (Model Context Protocol) Server untuk jds-magang.

Mengekspos alat-alat ekstraksi Vision VLM, PowerPoint parser, rendering slide ke gambar,
multi-page PDF stitcher, pemindaian direktori dataset, ekstraksi massal, simulasi chunking,
ingesti & double-verification tabel transaksional ke database SQLite,
serta evaluasi & ekstraksi selektif diagram ke sintaks Mermaid.js sebagai MCP Tools berstandar SDK v2.

Menjalankan server:
    python -m app.mcp_server
    python mcp_server.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from .batch import batch_extract_documents as run_batch_extract
from .batch import find_document_files, scan_document_directories
from .config import get_settings
from .deep_agent import build_deep_agent
from .diagram import (
    classify_diagram_convertibility as run_classify_diagram,
)
from .diagram import (
    extract_diagram_to_mermaid as run_extract_diagram,
)
from .docx import process_multipage_docx
from .excel import process_multipage_excel
from .extractor import VisionExtractor
from .graph import DocumentExtractionPipeline
from .llm import build_vlm
from .multi_page import preview_markdown_chunks as sim_preview_chunks
from .pdf import process_multipage_pdf
from .ppt import process_presentation_vision
from .preprocess import preprocess_image
from .tabular_db import (
    TabularDatabaseManager,
    extract_and_ingest_tables_from_markdown,
    query_sqlite,
)

# Inisialisasi Server MCP
server = MCPServer(
    name="jds-magang-vlm-agent",
    description="Vision VLM Document Extractor MCP Server: PDF, DOCX, Excel, PPTX, Scan -> Markdown Terstruktur, SQLite Tabular Engine, & Mermaid Diagrams",
    version="0.1.0",
)


@server.tool(
    name="render_presentation_slides",
    description=(
        "Render seluruh halaman/slide file presentasi PowerPoint (.pptx / .ppt) menjadi file gambar PNG beresolusi tinggi "
        "agar dapat dilihat dan dianalisis secara visual langsung oleh AI Multimodal / VLM (renderer: LibreOffice headless)."
    ),
)
def render_presentation_slides(
    pptx_path: str,
    output_dir: str | None = None,
    dpi: int = 200,
) -> str:
    """
    Render seluruh slide presentasi menjadi gambar PNG beresolusi tinggi.
    """
    path_obj = Path(pptx_path).resolve()
    if not path_obj.exists():
        return f"ERROR: File presentasi tidak ditemukan: {path_obj}"

    out_dir_obj = Path(output_dir).resolve() if output_dir else None
    try:
        from .ppt import render_presentation_slides_to_images

        img_paths = render_presentation_slides_to_images(
            path_obj, output_dir=out_dir_obj, dpi=dpi
        )
        return json.dumps(
            {
                "presentation": str(path_obj),
                "total_slides": len(img_paths),
                "slides": [str(p) for p in img_paths],
            },
            indent=2,
            ensure_ascii=False,
        )
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat render presentasi: {e}"


@server.tool(
    name="scan_document_folders",
    description="Pindai direktori input rekursif untuk menemukan dan mengelompokkan seluruh file PDF, DOCX, DOC, Excel, PPTX, PPT, dan Gambar.",
)
def scan_document_folders(base_path: str = "dataset") -> str:
    """
    Pindai folder dokumen.
    """
    path_obj = Path(base_path).resolve()
    if not path_obj.exists():
        return f"ERROR: Path tidak ditemukan: {path_obj}"

    catalog = scan_document_directories(path_obj)
    total_docs = sum(item.get("total_documents", 0) for item in catalog)
    return json.dumps(
        {
            "scanned_path": str(path_obj),
            "total_folders": len(catalog),
            "total_documents": total_docs,
            "folders": catalog,
        },
        indent=2,
        ensure_ascii=False,
    )


@server.tool(
    name="process_document_batch",
    description="Jalankan ekstraksi dokumen massal (batch) pada folder atau daftar file.",
)
def process_document_batch(
    source_dir_or_files: str,
    output_dir: str = "output/batch",
    forced_specs: str | None = None,
    dpi: int = 200,
    preview_chunks: bool = True,
) -> str:
    """
    Ekstraksi dokumen massal.
    """
    out_path = Path(output_dir).resolve()
    settings = get_settings()

    raw_items = [s.strip() for s in source_dir_or_files.split(",") if s.strip()]
    files_to_process: list[Path] = []

    for item in raw_items:
        p = Path(item).resolve()
        if p.is_dir():
            files_to_process.extend(find_document_files(p))
        elif p.is_file():
            files_to_process.append(p)

    if not files_to_process:
        return f"ERROR: Tidak ditemukan file yang valid untuk diproses dari: {source_dir_or_files}"

    results = run_batch_extract(
        files_to_process,
        output_dir=out_path,
        forced_specs=forced_specs,
        dpi=dpi,
        preview_chunks=preview_chunks,
        settings=settings,
    )

    return json.dumps(
        {
            "output_directory": str(out_path),
            "results": results,
        },
        indent=2,
        ensure_ascii=False,
    )


@server.tool(
    name="extract_document",
    description="Ekstrak satu dokumen (PDF, DOCX, Excel, PPTX, PPT, atau Gambar) menjadi teks Markdown bersih.",
)
def extract_document(
    file_path: str,
    specs: str | None = None,
    dpi: int = 200,
    use_agent: bool = False,
) -> str:
    """
    Ekstrak dokumen tunggal ke Markdown.
    """
    path_obj = Path(file_path).resolve()
    if not path_obj.exists():
        return f"ERROR: File tidak ditemukan: {path_obj}"

    settings = get_settings()
    ext = path_obj.suffix.lower()

    if use_agent:
        agent = build_deep_agent(settings)
        res = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Ekstrak dokumen berikut ke Markdown bersih: {path_obj}",
                    }
                ]
            }
        )
        return res.get("messages", [])[-1].content if res.get("messages") else ""

    try:
        pipeline = DocumentExtractionPipeline(settings)

        if ext == ".pdf":
            doc = process_multipage_pdf(
                path_obj, pipeline=pipeline, forced_specs=specs, dpi=dpi
            )
            return doc.full_markdown

        if ext in {".docx", ".doc"}:
            doc = process_multipage_docx(
                path_obj, pipeline=pipeline, forced_specs=specs, dpi=dpi
            )
            return doc.full_markdown

        if ext in {".xlsx", ".xls", ".xlsm", ".ods"}:
            doc = process_multipage_excel(
                path_obj, pipeline=pipeline, forced_specs=specs, dpi=dpi
            )
            return doc.full_markdown

        if ext in {".pptx", ".ppt"}:
            return process_presentation_vision(
                path_obj,
                pipeline=pipeline,
                dpi=dpi,
                forced_specs=specs or "presentation_slides",
            )

        if ext in {".png", ".jpg", ".jpeg", ".webp"}:
            result = pipeline.run(
                str(path_obj),
                forced_specs=specs,
                is_first_page=True,
                page_number=1,
                region_output_dir=(
                    Path("output") / path_obj.stem / "regions" / "page_0001"
                ),
            )
            return result.get("markdown_content", "")

        return f"ERROR: Ekstensi file tidak didukung: {ext}"
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat ekstraksi dokumen: {e}"


@server.tool(
    name="classify_document_layout",
    description="Analisis karakteristik visual dokumen untuk mendeteksi spesifikasi layout yang relevan (plain, markdown_hierarchy, bilingual_journal, presentation_slides, chat_transcript, signature_form).",
)
def classify_document_layout(image_path: str) -> str:
    """
    Klasifikasi layout gambar dokumen (multi-label).
    """
    path_obj = Path(image_path).resolve()
    if not path_obj.exists():
        return f"ERROR: File tidak ditemukan: {path_obj}"

    settings = get_settings()
    try:
        proc = preprocess_image(str(path_obj))
        vlm = build_vlm(settings)
        extractor = VisionExtractor(vlm)
        specs = extractor.classify(proc.processed_path)
        return json.dumps(
            {"file": str(path_obj), "specs": specs}, indent=2, ensure_ascii=False
        )
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat klasifikasi: {e}"


@server.tool(
    name="classify_diagram_convertibility",
    description="Evaluasi kelayakan diagram visual: apakah cocok dikonversi ke Mermaid (flowchart, ERD, sequence, state, class, mindmap, block diagram) atau tidak.",
)
def classify_diagram_convertibility(image_path: str) -> str:
    """
    Evaluasi kelayakan konversi diagram ke Mermaid.
    """
    path_obj = Path(image_path).resolve()
    if not path_obj.exists():
        return f"ERROR: File gambar tidak ditemukan: {path_obj}"

    settings = get_settings()
    try:
        proc = preprocess_image(str(path_obj))
        vlm = build_vlm(settings)
        res = run_classify_diagram(proc.processed_path, llm=vlm)
        payload = res.model_dump()
        # Payload MCP harus JSON; bytes PNG tersedia melalui path artefak, bukan
        # disisipkan mentah ke respons JSON.
        rendered_bytes = payload.pop("rendered_image_bytes", None)
        if rendered_bytes is not None:
            payload["rendered_image_size_bytes"] = len(rendered_bytes)
        return json.dumps(payload, indent=2, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat evaluasi diagram: {e}"


@server.tool(
    name="extract_diagram_to_mermaid",
    description="Ekstrak diagram visual pada dokumen menjadi kode Mermaid.js yang valid secara sintaks.",
)
def extract_diagram_to_mermaid(
    image_path: str,
    diagram_hint: str | None = None,
) -> str:
    """
    Ekstrak diagram visual ke kode Mermaid.js.
    """
    path_obj = Path(image_path).resolve()
    if not path_obj.exists():
        return f"ERROR: File gambar tidak ditemukan: {path_obj}"

    settings = get_settings()
    try:
        proc = preprocess_image(str(path_obj))
        vlm = build_vlm(settings)
        res = run_extract_diagram(
            proc.processed_path, llm=vlm, forced_diagram_type=diagram_hint
        )
        payload = res.model_dump()
        rendered_bytes = payload.pop("rendered_image_bytes", None)
        if rendered_bytes is not None:
            payload["rendered_image_size_bytes"] = len(rendered_bytes)
        return json.dumps(payload, indent=2, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat ekstraksi diagram Mermaid: {e}"


@server.tool(
    name="preview_markdown_chunks",
    description="[STAGING / BLUEPRINT] Simulasikan pemotongan teks Markdown hasil ekstraksi menjadi chunks untuk blueprint sistem RAG.",
)
def preview_markdown_chunks(
    markdown_text: str,
    source_file: str = "document.md",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> str:
    """
    Simulasi chunking teks Markdown.
    """
    preview = sim_preview_chunks(
        markdown_text,
        source_file=source_file,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return json.dumps(asdict(preview), indent=2, ensure_ascii=False)


@server.tool(
    name="classify_and_ingest_tables_to_sqlite",
    description="Klasifikasi dan ingest tabel-tabel transaksional dari Markdown ke basis data SQLite dokumen dengan verifikasi ganda integritas baris & kalkulasi numerik.",
)
def classify_and_ingest_tables_to_sqlite(
    markdown_text: str,
    sqlite_db_path: str,
    source_file: str = "document",
    force_all_tables: bool = False,
) -> str:
    """
    Ingesti tabel transaksional ke SQLite.
    """
    try:
        results = extract_and_ingest_tables_from_markdown(
            markdown_text=markdown_text,
            source_file=source_file,
            db_path=sqlite_db_path,
            force_all_tables=force_all_tables,
        )
        return json.dumps(
            [r.model_dump() for r in results], indent=2, ensure_ascii=False
        )
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat ingesti tabel ke SQLite: {e}"


@server.tool(
    name="query_tabular_database",
    description="Eksekusi query SQL SELECT pada basis data SQLite dokumen untuk analisis numerik / agregasi data.",
)
def query_tabular_database(
    sqlite_db_path: str,
    sql_query: str,
) -> str:
    """
    Query database SQLite dokumen.
    """
    try:
        res = query_sqlite(sql_query, db_path=sqlite_db_path)
        return json.dumps(res.model_dump(), indent=2, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat eksekusi query SQLite: {e}"


@server.tool(
    name="inspect_tabular_database",
    description="Inspeksi seluruh skema tabel aktif, jumlah baris, dan sampel record pada database SQLite dokumen.",
)
def inspect_tabular_database(
    sqlite_db_path: str,
) -> str:
    """
    Inspeksi skema dan ringkasan database SQLite dokumen.
    """
    try:
        db_mgr = TabularDatabaseManager(sqlite_db_path)
        summary = db_mgr.get_active_tables_summary()
        return json.dumps(summary, indent=2, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat inspeksi SQLite: {e}"


def run_mcp() -> None:
    """Jalankan MCP Server dengan stdio transport."""
    server.run(transport="stdio")


if __name__ == "__main__":
    run_mcp()
