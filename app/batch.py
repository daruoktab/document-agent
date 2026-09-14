"""
Modul Pemindaian Direktori & Ekstraksi Dokumen Massal (Batch Document Processing).

Menyediakan:
  - `scan_document_directories`: Mendeteksi folder & sub-folder yang berisi dokumen (PDF, PPTX, Scan/Gambar).
  - `find_document_files`: Mendeteksi semua file dokumen yang didukung di direktori root.
  - `batch_extract_documents`: Memproses dokumen dari satu atau banyak folder/file terpilih dengan batas kuota data.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import Settings, get_settings
from .graph import DocumentExtractionPipeline
from .multi_page import preview_markdown_chunks
from .pdf import process_multipage_pdf
from .ppt import process_presentation_vision

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: set[str] = {
    ".pdf",
    ".pptx",
    ".ppt",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}


def find_document_files(
    root_dir: str | Path = ".",
) -> list[Path]:
    """
    Cari semua file dokumen yang didukung di dalam direktori root rekursif.
    """
    root_path = Path(root_dir).resolve()
    if not root_path.exists():
        return []

    ignored_patterns = {
        ".git",
        ".venv",
        "__pycache__",
        ".ruff_cache",
        ".vscode",
        ".agents",
    }

    files: list[Path] = []
    for p in root_path.rglob("*"):
        if any(part in ignored_patterns for part in p.parts):
            continue
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(p)

    return sorted(files)


def scan_document_directories(
    root_dir: str | Path = ".",
    max_depth: int = 5,
) -> list[dict[str, Any]]:
    """
    Pindai root_dir dan seluruh sub-foldernya untuk mendeteksi keberadaan file dokumen.

    Returns:
        list of dict: [
            {
                "folder_path": str,
                "relative_path": str,
                "total_documents": int,
                "extension_counts": dict[str, int],
                "sample_files": list[str],
            }
        ]
    """
    root_path = Path(root_dir).resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Direktori tidak ditemukan: {root_path}")

    folder_map: dict[Path, list[Path]] = {}

    # Abaikan folder sistem/venv/git
    ignored_patterns = {
        ".git",
        ".venv",
        "__pycache__",
        ".ruff_cache",
        ".vscode",
        ".agents",
    }

    for p in root_path.rglob("*"):
        if any(part in ignored_patterns for part in p.parts):
            continue
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            parent = p.parent
            if parent not in folder_map:
                folder_map[parent] = []
            folder_map[parent].append(p)

    results: list[dict[str, Any]] = []
    for folder, files in sorted(folder_map.items(), key=lambda x: str(x[0])):
        try:
            rel = str(folder.relative_to(root_path))
            if rel == ".":
                rel = folder.name
        except ValueError:
            rel = str(folder)

        counts: dict[str, int] = {}
        for f in files:
            ext = f.suffix.lower()
            counts[ext] = counts.get(ext, 0) + 1

        results.append(
            {
                "folder_path": str(folder),
                "relative_path": rel,
                "total_documents": len(files),
                "extension_counts": counts,
                "sample_files": [f.name for f in files[:5]],
            }
        )

    return results


def batch_extract_documents(
    folders: Sequence[str | Path] | str | Path,
    *,
    limit: int | None = None,
    limit_per_folder: int | None = None,
    specs: str = "plain",
    forced_specs: str | None = None,
    output_dir: str | Path = "output/extracted_md",
    dpi: int = 200,
    preview_chunks: bool = False,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """
    Ekstrak dokumen dari satu atau banyak folder/file terpilih dengan batas jumlah file.

    Args:
        folders: Satu path folder/file, list path, atau string dipisah koma.
        limit: Batas total maksimal dokumen yang akan diproses secara keseluruhan.
        limit_per_folder: Batas maksimal dokumen per folder yang dipilih.
        specs: Spesifikasi layout ('plain', 'markdown_hierarchy', 'bilingual_journal', 'presentation_slides', 'chat_transcript', 'signature_form', atau komposit).
        forced_specs: Alias opsional untuk specs.
        output_dir: Direktori tempat menyimpan file Markdown hasil ekstraksi.
        dpi: Resolusi rendering gambar.
        preview_chunks: Apakah menyertakan simulasi statistik chunking.
        chunk_size: Ukuran chunk untuk simulasi.
        chunk_overlap: Overlap chunk untuk simulasi.
        settings: Pengaturan aplikasi.
    """
    resolved_settings = settings or get_settings()
    active_specs = forced_specs or specs

    # Normalisasi input folders
    target_items: list[Path] = []
    if isinstance(folders, (str, Path)):
        if isinstance(folders, str):
            folder_strings = [f.strip() for f in folders.split(",") if f.strip()]
        else:
            folder_strings = [str(folders)]
    else:
        folder_strings = [str(f) for f in folders]

    for f_str in folder_strings:
        p = Path(f_str).resolve()
        if p.exists():
            target_items.append(p)

    if not target_items:
        return {
            "status": "error",
            "message": f"Tidak ada folder atau file valid yang ditemukan dari input: {folders}",
            "processed_count": 0,
            "results": [],
        }

    # Kumpulkan daftar file yang akan diproses sesuai limit
    files_to_process: list[Path] = []
    for item in target_items:
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
            files_to_process.append(item)
        elif item.is_dir():
            dir_files = [
                f
                for f in item.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
            dir_files.sort()
            if limit_per_folder is not None and limit_per_folder > 0:
                dir_files = dir_files[:limit_per_folder]
            files_to_process.extend(dir_files)

    if limit is not None and limit > 0:
        files_to_process = files_to_process[:limit]

    if not files_to_process:
        return {
            "status": "warning",
            "message": "Tidak ditemukan file dokumen yang didukung di dalam folder yang dipilih.",
            "processed_count": 0,
            "results": [],
        }

    out_base = Path(output_dir).resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    pipeline = DocumentExtractionPipeline(resolved_settings)
    processed_results: list[dict[str, Any]] = []

    for idx, doc_file in enumerate(files_to_process, start=1):
        ext = doc_file.suffix.lower()
        rel_stem = doc_file.stem
        doc_dir = out_base / rel_stem
        doc_dir.mkdir(parents=True, exist_ok=True)
        out_file = doc_dir / f"{rel_stem}.md"

        try:
            # 1. PPTX
            if ext in (".pptx", ".ppt"):
                md_content = process_presentation_vision(
                    pptx_path=doc_file,
                    pipeline=pipeline,
                    output_dir=doc_dir / "slides",
                    dpi=dpi,
                    forced_specs=active_specs,
                    output_markdown_path=out_file,
                )
            # 2. PDF
            elif ext == ".pdf":
                extracted = process_multipage_pdf(
                    pdf_path=doc_file,
                    pipeline=pipeline,
                    output_dir=doc_dir / "pages",
                    dpi=dpi,
                    forced_specs=active_specs,
                    output_markdown_path=out_file,
                )
                md_content = extracted.markdown_content
            # 3. Gambar
            else:
                res = pipeline.run(str(doc_file), forced_specs=active_specs)
                md_content = str(res["markdown_content"])

            out_file.write_text(md_content, encoding="utf-8")

            # Ekspor tabel SQLite ke CSV jika basis data terbentuk
            batch_db = doc_dir / "databases" / f"{rel_stem}.sqlite"
            if batch_db.exists():
                try:
                    from .tabular_db import TabularDatabaseManager

                    TabularDatabaseManager(batch_db).export_to_csv(
                        output_dir=doc_dir / "csv"
                    )
                except Exception as e_csv:  # noqa: BLE001
                    logger.warning("[Batch] Gagal mengekspor CSV: %s", e_csv)

            item_info: dict[str, Any] = {
                "index": idx,
                "filename": doc_file.name,
                "source_path": str(doc_file),
                "output_markdown_path": str(out_file),
                "char_count": len(md_content),
                "status": "success",
            }

            if preview_chunks:
                chunks = preview_markdown_chunks(
                    md_content, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                item_info["chunk_count"] = len(chunks.chunks)

            processed_results.append(item_info)

        except Exception as e:  # noqa: BLE001
            processed_results.append(
                {
                    "index": idx,
                    "filename": doc_file.name,
                    "source_path": str(doc_file),
                    "output_markdown_path": None,
                    "status": "error",
                    "error": str(e),
                }
            )

    successful_count = sum(1 for r in processed_results if r["status"] == "success")
    return {
        "status": "completed" if successful_count == len(files_to_process) else ("partial_failure" if successful_count else "failed"),
        "total_selected_files": len(files_to_process),
        "successful_count": successful_count,
        "error_count": len(files_to_process) - successful_count,
        "output_directory": str(out_base),
        "results": processed_results,
    }
