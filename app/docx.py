"""
Pipeline DOCX berbasis Vision VLM.

DOCX dirender lewat LibreOffice menjadi PDF sementara, lalu mengikuti pipeline PDF:
setiap halaman PDF dirender menjadi gambar dan dikirim satu per satu ke VLM.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .config import DEFAULT_DPI
from .pdf import pdf_page_count, process_multipage_pdf
from .ppt import _find_libreoffice_binary
from .schemas import ExtractedDocument

logger = logging.getLogger(__name__)


def convert_docx_to_pdf(
    docx_path: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    """Konversi DOC/DOCX menjadi PDF memakai LibreOffice headless."""
    path_obj = Path(docx_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"File DOCX tidak ditemukan: {path_obj}")

    soffice = _find_libreoffice_binary()
    if not soffice:
        raise RuntimeError(
            "LibreOffice (soffice) tidak ditemukan. Diperlukan untuk merender DOCX ke gambar halaman."
        )

    target_dir = (
        Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="docx_pdf_"))
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    user_profile = (target_dir / "lo_profile").as_uri()
    cmd = [
        soffice,
        f"-env:UserInstallation={user_profile}",
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(target_dir),
        str(path_obj),
    ]
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"LibreOffice gagal mengonversi DOCX ke PDF: {res.stderr.strip()}"
        )

    pdf_path = target_dir / f"{path_obj.stem}.pdf"
    if not pdf_path.exists():
        candidates = sorted(target_dir.glob("*.pdf"))
        if candidates:
            return candidates[0]
        raise RuntimeError(
            "LibreOffice selesai, tetapi file PDF hasil konversi tidak ditemukan."
        )
    return pdf_path


def docx_page_count(docx_path: str | Path) -> int:
    """Hitung jumlah halaman DOCX melalui hasil konversi PDF sementara."""
    with tempfile.TemporaryDirectory(prefix="docx_count_") as tmp:
        pdf_path = convert_docx_to_pdf(docx_path, tmp)
        return pdf_page_count(pdf_path)


def process_multipage_docx(
    docx_path: str | Path,
    pipeline: Any = None,
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
    resume: bool = False,
) -> ExtractedDocument:
    """
    Proses DOCX dengan jalur visual yang sama seperti PDF.

    File DOCX dikonversi ke PDF sementara, lalu halaman PDF dirender menjadi gambar
    dan diproses oleh VLM satu per satu.
    """
    path_obj = Path(docx_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"File DOCX tidak ditemukan: {path_obj}")

    with tempfile.TemporaryDirectory(prefix="docx_ingest_") as tmp:
        logger.info("[DOCX] Mengonversi '%s' ke PDF sementara...", path_obj.name)
        pdf_path = convert_docx_to_pdf(path_obj, tmp)
        logger.info(
            "[DOCX] Konversi selesai, lanjut render halaman ke gambar dan ekstraksi VLM."
        )
        return process_multipage_pdf(
            pdf_path=pdf_path,
            pipeline=pipeline,
            llm=llm,
            output_dir=output_dir,
            dpi=dpi,
            forced_specs=forced_specs,
            forced_doc_type=forced_doc_type,
            db_path=db_path,
            auto_tabular_db=auto_tabular_db,
            force_all_tables=force_all_tables,
            output_markdown_path=output_markdown_path,
            source_file_for_records=path_obj,
            resume=resume,
        )


__all__ = ["convert_docx_to_pdf", "docx_page_count", "process_multipage_docx"]
