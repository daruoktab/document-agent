"""File-only HTTP interface to the Streamlit ingestion pipeline."""

from __future__ import annotations

import csv
import logging
import re
import sqlite3
import time
from io import BytesIO, StringIO
from pathlib import Path
from typing import Annotated
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from app.job_tracker import PROJECT_ROOT, JobInfo, JobManager

logger = logging.getLogger(__name__)
OUTPUT_DIR = PROJECT_ROOT / "output"
SUPPORTED_TYPES = {".pdf", ".ppt", ".pptx", ".png", ".jpg", ".jpeg", ".webp"}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

app = FastAPI(
    title="Document Ingest API",
    description="Upload satu file untuk mendapatkan ZIP berisi Markdown, SQL SQLite, dan CSV jika ada tabel.",
    version="0.1.0",
)


def build_result_zip(job: JobInfo) -> bytes:
    """Export Markdown and a consistent database snapshot, including committed WAL."""
    markdown = job.out_file.read_text(encoding="utf-8")
    sql = "BEGIN TRANSACTION;\nCOMMIT;\n"
    csv_files: dict[str, str] = {}
    if job.db_file and job.db_file.is_file():
        source = sqlite3.connect(job.db_file.resolve().as_uri() + "?mode=ro", uri=True)
        snapshot = sqlite3.connect(":memory:")
        try:
            source.backup(snapshot)
            sql = "\n".join(snapshot.iterdump()) + "\n"
            tables = snapshot.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            for (table,) in tables:
                if table.startswith("sqlite_"):
                    continue
                quoted = '"' + table.replace('"', '""') + '"'
                cursor = snapshot.execute(f"SELECT * FROM {quoted}")
                buffer = StringIO(newline="")
                writer = csv.writer(buffer)
                writer.writerow([column[0] for column in cursor.description])
                writer.writerows(cursor)
                safe_name = re.sub(r"[^\w .-]", "_", table).strip(". ") or "table"
                name = f"{safe_name}.csv"
                index = 2
                while name in csv_files:
                    name = f"{safe_name}_{index}.csv"
                    index += 1
                csv_files[name] = buffer.getvalue()
        finally:
            snapshot.close()
            source.close()

    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("document.md", markdown)
        archive.writestr("document.sql", sql)
        for name, content in csv_files.items():
            archive.writestr(f"csv/{name}", content.encode("utf-8-sig"))
    return buffer.getvalue()


@app.get("/", summary="Status API")
def root():
    return {"message": "OCR API is running", "docs": "/docs", "plan": "/plan"}


@app.get("/plan", summary="Dokumentasi alur ingest")
def plan():
    """Jelaskan alur pemrosesan tanpa menjalankan ingest atau memanggil model."""
    return {
        "endpoint": "POST /ingest",
        "input": {"field": "file", "content_type": "multipart/form-data"},
        "supported_extensions": sorted(SUPPORTED_TYPES),
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "steps": [
            "Validasi file dan simpan upload dengan ID unik.",
            "Jalankan pipeline yang sama dengan Streamlit menggunakan pengaturan bawaan.",
            "Tunggu ekstraksi Markdown dan ingest tabel ke SQLite selesai.",
            "Ekspor dump SQL dan CSV per tabel dari snapshot SQLite.",
            "Kembalikan ZIP berisi document.md, document.sql, dan CSV jika tersedia.",
        ],
        "output": {"content_type": "application/zip", "filename": "hasil_ingest.zip"},
        "docs": "/docs",
    }


@app.post(
    "/ingest",
    summary="Ingest satu file",
    response_class=Response,
    responses={
        200: {"content": {"application/zip": {"schema": {"type": "string", "format": "binary"}}}},
        400: {"description": "File kosong"},
        413: {"description": "File melebihi 100 MiB"},
        415: {"description": "Format file tidak didukung"},
        500: {"description": "Ingest atau ekspor gagal"},
    },
)
def ingest(
    file: Annotated[UploadFile, File(description="PDF, PPT/PPTX, PNG, JPG/JPEG, atau WebP; maksimum 100 MiB")],
) -> Response:
    """Tunggu ingest selesai lalu unduh ZIP. Pengaturan ekstraksi dipilih otomatis."""
    filename = Path((file.filename or "").replace("\\", "/")).name
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_TYPES:
        raise HTTPException(415, "Format file tidak didukung.")
    stem = re.sub(r"[^\w-]", "_", Path(filename).stem)[:80] or "document"
    uploads = OUTPUT_DIR / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    input_path = uploads / f"{stem}_{uuid4().hex}{extension}"
    try:
        size = 0
        with input_path.open("xb") as target:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Ukuran file melebihi 100 MiB.")
                target.write(chunk)
        if not size:
            raise HTTPException(400, "File kosong.")
    except Exception:
        input_path.unlink(missing_ok=True)
        raise
    finally:
        file.file.close()

    manager = JobManager.get_instance()
    job = manager.start_job(input_path=input_path, output_dir=OUTPUT_DIR)
    # A synchronous FastAPI handler runs in a worker thread.
    while job.status == "running":
        time.sleep(0.5)
        job = manager.get_job(job.job_id, output_dir=OUTPUT_DIR) or job
    if job.status != "completed":
        raise HTTPException(500, {"job_id": job.job_id, "message": "Ingest gagal. Periksa log di folder output."})
    try:
        archive = build_result_zip(job)
    except (OSError, sqlite3.Error):
        logger.exception("Gagal mengekspor hasil %s", job.job_id)
        raise HTTPException(500, {"job_id": job.job_id, "message": "Hasil ingest tidak dapat dibaca."}) from None
    return Response(
        archive, media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="hasil_ingest.zip"'},
    )
