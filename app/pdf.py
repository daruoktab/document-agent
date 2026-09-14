"""
Modul Pemrosesan Dokumen PDF Multi-Halaman.

Menyediakan fungsi untuk:
  - Render PDF menjadi citra (DPI tinggi) menggunakan PyMuPDF atau pypdfium2.
  - Menghitung jumlah halaman PDF.
  - Memproses seluruh halaman PDF dalam batch 10 halaman (PDF_PAGE_BATCH).
  - Ekstraksi teks native / terstruktur per-halaman langsung via PyMuPDF.
  - Menggabungkan hasil ekstraksi halaman menjadi satu dokumen Markdown utuh yang konsisten.
  - Menjalankan sub-agent SQL tabular per-halaman dan audit guardrail jalur ganda di tahap akhir.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import DEFAULT_DPI, PDF_PAGE_BATCH
from .multi_page import (
    extract_document_title,
    format_page_delimiter,
    stitch_pages_to_markdown,
    strip_page_markers,
)
from .prompts import normalize_specs
from .schemas import DocumentPage, ExtractedDocument, PageTabularEvent
from .tabular_db import (
    cross_verify_dual_track,
    process_page_tabular_agent,
    prune_document_pages,
)

if TYPE_CHECKING:
    from .graph import DocumentExtractionPipeline

logger = logging.getLogger(__name__)


def pdf_page_count(pdf_path: str | Path) -> int:
    """Hitung jumlah total halaman dalam file PDF."""
    path_obj = Path(pdf_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"File PDF tidak ditemukan: {path_obj}")

    try:
        import pymupdf

        doc = pymupdf.open(str(path_obj))
        count = len(doc)
        doc.close()
        return count
    except ImportError:
        pass

    try:
        pdfium = importlib.import_module("pypdfium2")
        pdf = pdfium.PdfDocument(str(path_obj))
        count = len(pdf)
        pdf.close()
        return count
    except (ImportError, Exception) as err:
        raise ImportError(
            "Diperlukan 'pymupdf' atau 'pypdfium2' untuk membaca file PDF. "
            "Jalankan: uv add pymupdf"
        ) from err


def pdf_to_images(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    dpi: int = DEFAULT_DPI,
    image_ext: str = "png",
    pages: list[int] | None = None,
) -> list[Path]:
    """
    Render halaman PDF menjadi file gambar (DPI tinggi).

    Args:
        pdf_path: Path ke file PDF.
        output_dir: Folder output untuk menyimpan gambar (default: folder sementara).
        dpi: Resolusi rendering (default 200 DPI).
        image_ext: Format file gambar ("png" atau "jpg").
        pages: Daftar indeks halaman 0-indexed yang akan dirender (None = semua).

    Returns:
        Daftar Path file gambar hasil render (berurutan).
    """
    path_obj = Path(pdf_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"File PDF tidak ditemukan: {path_obj}")

    if output_dir is None:
        out_path = Path("output") / path_obj.stem / "pages"
    else:
        out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rendered: list[Path] = []

    # Coba PyMuPDF terlebih dahulu (lebih cepat dan tajam)
    try:
        import pymupdf

        doc = pymupdf.open(str(path_obj))
        zoom = dpi / 72.0
        mat = pymupdf.Matrix(zoom, zoom)

        target_pages = pages if pages is not None else list(range(len(doc)))

        for pno in target_pages:
            if pno >= len(doc):
                continue
            page = doc[pno]
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_file = out_path / f"page_{pno + 1:04d}.{image_ext}"
            pix.save(str(img_file))
            rendered.append(img_file)

        doc.close()
        return rendered
    except ImportError:
        pass

    # Fallback: pypdfium2
    try:
        pdfium = importlib.import_module("pypdfium2")
        pdf = pdfium.PdfDocument(str(path_obj))
        scale = dpi / 72.0
        target_pages = pages if pages is not None else list(range(len(pdf)))

        for pno in target_pages:
            if pno >= len(pdf):
                continue
            page = pdf[pno]
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()
            img_file = out_path / f"page_{pno + 1:04d}.{image_ext}"
            pil_image.save(str(img_file))
            rendered.append(img_file)

        pdf.close()
        return rendered
    except (ImportError, Exception) as err:
        raise ImportError(
            "Tidak ditemukan library PDF renderer. Jalankan: uv add pymupdf"
        ) from err


def extract_pdf_markdown_mupdf(
    pdf_path: str | Path,
    pages: list[int] | None = None,
    page_chunks: bool = True,
    output_dir: str | Path | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """
    Ekstrak teks dan konten terstruktur dari PDF langsung menggunakan PyMuPDF4LLM.
    """
    path_obj = Path(pdf_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"File PDF tidak ditemukan: {path_obj}")

    import pymupdf

    doc = pymupdf.open(str(path_obj))
    total_pages = len(doc)
    doc.close()

    try:
        import pymupdf4llm

        target_pages = pages if pages is not None else None
        md_text = pymupdf4llm.to_markdown(
            str(path_obj),
            pages=target_pages,
            page_chunks=page_chunks,
            write_images=bool(output_dir),
            image_path=str(output_dir) if output_dir else None,
        )
    except ImportError:
        import pymupdf

        doc = pymupdf.open(str(path_obj))
        chunks = []
        target_indices = pages if pages is not None else list(range(len(doc)))
        for pno in target_indices:
            if pno < len(doc):
                chunks.append({
                    "text": doc[pno].get_text("text"),
                    "metadata": {"page": pno + 1},
                })
        doc.close()
        md_text = chunks

    result: dict[str, Any] = {
        "file_path": str(path_obj),
        "total_pages": total_pages,
        "extracted_pages_count": len(md_text)
        if isinstance(md_text, list)
        else total_pages,
        "pages": [],
    }

    if isinstance(md_text, list):
        formatted_pages = []
        for idx, item in enumerate(md_text, start=1):
            meta = item.get("metadata", {})
            page_num = meta.get("page", idx)
            formatted_pages.append({
                "page": page_num,
                "text": item.get("text", ""),
                "metadata": meta,
                "tables": item.get("tables", []),
                "images": item.get("images", []),
            })
        return formatted_pages

    return result


extract_pdf_with_pymupdf4llm = extract_pdf_markdown_mupdf


def process_multipage_pdf(
    pdf_path: str | Path,
    pipeline: DocumentExtractionPipeline | Any = None,
    *,
    llm: Any = None,
    output_dir: str | Path | None = None,
    dpi: int = DEFAULT_DPI,
    forced_specs: list[str] | str | None = None,
    forced_doc_type: str | None = None,
    db_path: str | Path | None = None,
    auto_tabular_db: bool = True,
    force_all_tables: bool = False,
    output_markdown_path: str | Path | None = None,
) -> ExtractedDocument:
    """
    Proses seluruh halaman PDF dan gabungkan hasil ekstraksi menjadi teks Markdown utuh siap chunking.
    Mengeksekusi Sub-Agent SQL mandiri per-halaman untuk memahami & meng-ingest tabel ke SQLite secara langsung,
    serta melakukan audit Guardrail jalur ganda di tahap akhir.
    """
    pdf_path = Path(pdf_path)
    total_pages = pdf_page_count(pdf_path)

    if pipeline is None:
        from .graph import DocumentExtractionPipeline

        pipeline = DocumentExtractionPipeline(vlm=llm)
    elif hasattr(pipeline, "invoke") and not hasattr(pipeline, "run"):
        from .graph import DocumentExtractionPipeline

        pipeline = DocumentExtractionPipeline(vlm=pipeline)

    pages: list[DocumentPage] = []
    pages_md: list[str] = []
    all_page_specs: list[list[str]] = []
    tabular_events: list[PageTabularEvent] = []
    previous_context: str | None = None
    total_visuals = 0
    total_tables = 0

    document_title: str | None = None
    active_forced = forced_specs or forced_doc_type

    # Tentukan path target database SQLite
    if db_path:
        resolved_db_path: Path | None = Path(db_path).resolve()
    elif output_markdown_path:
        resolved_db_path = (
            Path(output_markdown_path).resolve().parent / "databases" / f"{pdf_path.stem}.sqlite"
        )
    elif output_dir:
        resolved_db_path = (
            Path(output_dir).resolve() / "databases" / f"{pdf_path.stem}.sqlite"
        )
    else:
        resolved_db_path = (
            Path("output") / pdf_path.stem / "databases" / f"{pdf_path.stem}.sqlite"
        )

    if resolved_db_path:
        resolved_db_path.parent.mkdir(parents=True, exist_ok=True)

    stream_file: Path | None = None
    if output_markdown_path:
        stream_file = Path(output_markdown_path).resolve()
        stream_file.parent.mkdir(parents=True, exist_ok=True)
        stream_file.write_text("", encoding="utf-8")
        logger.info("Streaming output Markdown ke: %s", stream_file)

    # Tentukan folder output render gambar halaman PDF
    if output_dir:
        pages_render_dir: Path | None = Path(output_dir).resolve()
    elif output_markdown_path:
        pages_render_dir = Path(output_markdown_path).resolve().parent / "pages"
    else:
        pages_render_dir = Path("output") / pdf_path.stem / "pages"

    # Proses BERTAHAP per batch 10 halaman: render batch -> ekstrak batch -> lanjut.
    for b_start in range(0, total_pages, PDF_PAGE_BATCH):
        b_end = min(b_start + PDF_PAGE_BATCH, total_pages)
        page_images = pdf_to_images(
            pdf_path,
            output_dir=pages_render_dir,
            dpi=dpi,
            pages=list(range(b_start, b_end)),
        )

        for idx, img_path in enumerate(page_images, start=b_start + 1):
            logger.info(
                "Memproses Halaman %d / %d dari '%s'...",
                idx,
                total_pages,
                pdf_path.name,
            )

            # Jalur 1: Ekstraksi Teks Markdown VLM dengan konteks halaman sebelumnya
            res = pipeline.run(
                str(img_path),
                forced_specs=active_forced,
                previous_page_context=previous_context,
                is_first_page=(idx == 1),
            )

            from .tabular_db import sanitize_markdown_tables

            page_md: str = strip_page_markers(res["markdown_content"]).strip()
            page_md = sanitize_markdown_tables(page_md)

            # Mekanisme Judul Dokumen (diekstraksi sekali, utamanya pada halaman 1)
            if idx == 1:
                detected_title = getattr(res, "document_title", None) or extract_document_title(
                    page_md, fallback_title=pdf_path.stem
                )
                if detected_title:
                    document_title = detected_title
                    logger.info("[PDF] Judul dokumen utama teridentifikasi: '%s'", document_title)

            detected_specs: list[str] = res.get("specs") or ["plain"]
            total_visuals += int(res.get("visual_count", 0))
            total_tables += int(res.get("table_count", 0))
            if res.get("visual_count", 0) or res.get("table_count", 0):
                logger.info(
                    "[PDF] Halaman %d/%d metadata: %d visual/diagram, %d tabel",
                    idx, total_pages,
                    res.get("visual_count", 0), res.get("table_count", 0),
                )

            pages_md.append(page_md)
            all_page_specs.append(detected_specs)

            # Susun konteks kesinambungan untuk halaman berikutnya dengan referensi judul dokumen
            doc_prefix = (
                f"Konteks Dokumen: Judul: '{document_title}'. Halaman saat ini: Halaman {idx + 1}/{total_pages} (halaman lanjutan, jangan mengulang judul dokumen sebagai heading #).\n\n"
                if document_title
                else ""
            )
            tail_ctx = page_md[-400:] if len(page_md) > 400 else page_md
            previous_context = doc_prefix + tail_ctx

            pages.append(
                DocumentPage(
                    page_number=idx,
                    specs=detected_specs,
                    markdown_content=page_md,
                    image_path=str(img_path),
                )
            )

            # Jalur 2: Sub-Agent SQL Tabular Engine mandiri per-halaman
            if auto_tabular_db and resolved_db_path:
                tab_event, _ = process_page_tabular_agent(
                    page_markdown=page_md,
                    page_number=idx,
                    source_file=str(pdf_path),
                    db_path=resolved_db_path,
                    table_name_prefix=pdf_path.stem,
                    append_if_matching=True,
                    force_all_tables=force_all_tables,
                    llm=llm or getattr(pipeline, "vlm", None),
                )
                tabular_events.append(tab_event)
                if tab_event.tagged_markdown:
                    page_md = strip_page_markers(tab_event.tagged_markdown).strip()
                    pages[-1].markdown_content = page_md
                    pages_md[-1] = page_md

            if stream_file:
                delimiter = format_page_delimiter(idx, is_slide=False)
                with open(stream_file, "a", encoding="utf-8") as f:
                    f.write(f"\n{delimiter}\n\n{page_md}\n\n---\n")
                    f.flush()

    # Jahit teks seluruh halaman menjadi satu teks Markdown utuh dengan judul utama
    full_md = stitch_pages_to_markdown(
        pages_md,
        document_title=document_title,
        is_slide=False,
    )

    # Jalur 3: Supervisor Guardrail Cross-Verification (Audit Markdown vs SQLite)
    guardrail_report = None
    if auto_tabular_db and resolved_db_path and resolved_db_path.exists():
        prune_document_pages(resolved_db_path, str(pdf_path), total_pages)
        guardrail_report = cross_verify_dual_track(
            stitched_markdown=full_md,
            db_path=resolved_db_path,
            source_file=str(pdf_path),
            total_pages=total_pages,
        )
        try:
            from .tabular_db import TabularDatabaseManager

            csv_dir = (
                resolved_db_path.parent.parent / "csv"
                if resolved_db_path.parent.name == "databases"
                else resolved_db_path.parent / "csv"
            )
            TabularDatabaseManager(resolved_db_path).export_to_csv(output_dir=csv_dir)
        except Exception as e_csv:  # noqa: BLE001
            logger.warning("[PDF] Gagal mengekspor tabel SQLite ke CSV: %s", e_csv)

    # Hitung konsensus spesifikasi layout utama dokumen
    flat_specs = [
        s for page_spec in all_page_specs for s in page_spec if s != "plain"
    ]
    dominant_specs = normalize_specs(flat_specs) if flat_specs else ["plain"]
    primary_doc_type = dominant_specs[0] if dominant_specs else "plain"

    logger.info(
        "[PDF] Selesai: %d halaman | %d elemen visual/diagram | %d tabel terdeteksi",
        total_pages, total_visuals, total_tables,
    )

    return ExtractedDocument(
        source_file=str(pdf_path),
        title=document_title,
        doc_type=primary_doc_type,
        pages=pages,
        full_markdown=full_md,
        page_tabular_events=tabular_events,
        guardrail_report=guardrail_report,
        total_pages=total_pages,
        total_visuals=total_visuals,
        total_tables=total_tables,
    )


__all__ = [
    "DEFAULT_DPI",
    "PDF_PAGE_BATCH",
    "extract_pdf_markdown_mupdf",
    "extract_pdf_with_pymupdf4llm",
    "pdf_page_count",
    "pdf_to_images",
    "process_multipage_pdf",
]
