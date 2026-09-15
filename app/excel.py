"""
Pipeline Excel berbasis Vision VLM.

Workbook dirender lewat LibreOffice menjadi PDF sementara, lalu mengikuti pipeline
PDF: setiap halaman hasil cetak spreadsheet dirender menjadi gambar dan dikirim
satu per satu ke VLM.
"""

from __future__ import annotations

import logging
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from .config import DEFAULT_DPI
from .pdf import pdf_page_count, process_multipage_pdf
from .ppt import _find_libreoffice_binary
from .schemas import ExtractedDocument

logger = logging.getLogger(__name__)


def _find_uno_python() -> str | None:
    """Temukan Python sistem yang menyediakan modul UNO LibreOffice."""
    candidates = ["/usr/bin/python3", "/usr/lib/libreoffice/program/python"]
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    return None


def _convert_excel_to_pdf_with_page_scaling(
    soffice: str,
    excel_path: Path,
    pdf_path: Path,
    profile_dir: Path,
) -> bool:
    """Ekspor Excel melalui UNO dengan semua kolom dipaskan ke satu lebar halaman."""
    uno_python = _find_uno_python()
    if uno_python is None:
        return False

    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
    except OSError as exc:
        logger.warning("Tidak dapat membuka port lokal untuk UNO: %s", exc)
        return False

    listener = subprocess.Popen(
        [
            soffice,
            f"-env:UserInstallation={profile_dir.as_uri()}",
            "--headless",
            "--norestore",
            "--nodefault",
            "--nofirststartwizard",
            f"--accept=socket,host=127.0.0.1,port={port};urp;StarOffice.ComponentContext",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    uno_script = r"""
import sys
import time
import uno
from com.sun.star.beans import PropertyValue


def prop(name, value):
    item = PropertyValue()
    item.Name = name
    item.Value = value
    return item


port = int(sys.argv[1])
source_url = uno.systemPathToFileUrl(sys.argv[2])
target_url = uno.systemPathToFileUrl(sys.argv[3])
local_context = uno.getComponentContext()
resolver = local_context.ServiceManager.createInstanceWithContext(
    "com.sun.star.bridge.UnoUrlResolver", local_context
)
context = None
for _ in range(60):
    try:
        context = resolver.resolve(
            f"uno:socket,host=127.0.0.1,port={port};urp;StarOffice.ComponentContext"
        )
        break
    except Exception:
        time.sleep(0.25)
if context is None:
    raise RuntimeError("Tidak dapat terhubung ke listener LibreOffice UNO")

desktop = context.ServiceManager.createInstanceWithContext(
    "com.sun.star.frame.Desktop", context
)
document = desktop.loadComponentFromURL(
    source_url,
    "_blank",
    0,
    (prop("Hidden", True), prop("ReadOnly", True)),
)
if document is None:
    raise RuntimeError("LibreOffice tidak dapat membuka workbook")

try:
    page_styles = document.getStyleFamilies().getByName("PageStyles")
    sheets = document.getSheets()
    for index in range(sheets.getCount()):
        sheet = sheets.getByIndex(index)
        page_style = page_styles.getByName(sheet.getPropertyValue("PageStyle"))
        page_style.setPropertyValue("ScaleToPagesX", 1)
        page_style.setPropertyValue("ScaleToPagesY", 0)

    document.storeToURL(
        target_url,
        (prop("FilterName", "calc_pdf_Export"), prop("Overwrite", True)),
    )
finally:
    document.close(True)
"""

    try:
        subprocess.run(
            [uno_python, "-c", uno_script, str(port), str(excel_path), str(pdf_path)],
            capture_output=True,
            text=True,
            timeout=180,
            check=True,
        )
        return pdf_path.exists()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning(
            "Ekspor Excel dengan pengaturan fit-to-width gagal; gunakan ekspor default: %s",
            exc,
        )
        return False
    finally:
        listener.terminate()
        try:
            listener.wait(timeout=10)
        except subprocess.TimeoutExpired:
            listener.kill()
            listener.wait()
        time.sleep(0.1)


def convert_excel_to_pdf(
    excel_path: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    """Konversi workbook Excel/ODS menjadi PDF memakai LibreOffice headless."""
    path_obj = Path(excel_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"File Excel tidak ditemukan: {path_obj}")

    soffice = _find_libreoffice_binary()
    if not soffice:
        raise RuntimeError(
            "LibreOffice (soffice) tidak ditemukan. Diperlukan untuk merender Excel ke gambar halaman."
        )

    target_dir = (
        Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="excel_pdf_"))
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = target_dir / f"{path_obj.stem}.pdf"
    user_profile = target_dir / "lo_profile"
    converted = _convert_excel_to_pdf_with_page_scaling(
        soffice, path_obj, pdf_path, user_profile
    )

    if not converted:
        cmd = [
            soffice,
            f"-env:UserInstallation={user_profile.as_uri()}",
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
                f"LibreOffice gagal mengonversi Excel ke PDF: {res.stderr.strip()}"
            )

    if not pdf_path.exists():
        candidates = sorted(target_dir.glob("*.pdf"))
        if candidates:
            return candidates[0]
        raise RuntimeError(
            "LibreOffice selesai, tetapi file PDF hasil konversi tidak ditemukan."
        )
    return pdf_path


def excel_page_count(excel_path: str | Path) -> int:
    """Hitung jumlah halaman Excel melalui hasil konversi PDF sementara."""
    with tempfile.TemporaryDirectory(prefix="excel_count_") as tmp:
        pdf_path = convert_excel_to_pdf(excel_path, tmp)
        return pdf_page_count(pdf_path)


def process_multipage_excel(
    excel_path: str | Path,
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
) -> ExtractedDocument:
    """
    Proses workbook Excel dengan jalur visual yang sama seperti PDF.

    File Excel dikonversi ke PDF sementara, lalu halaman PDF dirender menjadi
    gambar dan diproses oleh VLM satu per satu.
    """
    path_obj = Path(excel_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"File Excel tidak ditemukan: {path_obj}")

    with tempfile.TemporaryDirectory(prefix="excel_ingest_") as tmp:
        logger.info("[Excel] Mengonversi '%s' ke PDF sementara...", path_obj.name)
        pdf_path = convert_excel_to_pdf(path_obj, tmp)
        logger.info(
            "[Excel] Konversi selesai, lanjut render halaman ke gambar dan ekstraksi VLM."
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
        )


__all__ = ["convert_excel_to_pdf", "excel_page_count", "process_multipage_excel"]
