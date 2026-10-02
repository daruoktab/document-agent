"""File-only and rich HTTP interface to the Streamlit ingestion pipeline and RAG knowledge base."""

from __future__ import annotations

import csv
import json
import logging
import re
import sqlite3
import threading
import time
from io import BytesIO, StringIO
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

from app.job_tracker import PROJECT_ROOT, JobInfo, JobManager
from app.streamlit_logic import (
    _cancel_jobs_before_delete,
    _get_sqlite_db_for_file,
    _save_uploaded_file,
    batch_status,
    build_document_zip,
    get_document_images,
    split_markdown_by_pages,
)
from app.upload_batches import create_batch, delete_batch, list_batches

logger = logging.getLogger(__name__)
OUTPUT_DIR = PROJECT_ROOT / "output"
SUPPORTED_TYPES = {
    ".pdf",
    ".docx",
    ".doc",
    ".xlsx",
    ".xls",
    ".xlsm",
    ".ods",
    ".ppt",
    ".pptx",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

app = FastAPI(
    title="Document Ingest & Workspace API",
    description=(
        "API backend untuk ekstraksi dokumen, manajemen batch upload, "
        "pratinjau inspeksi sebelum RAG, dan temu kembali semantik multimodal."
    ),
    version="0.2.0",
)


class _UploadedFileMemory:
    """Adapter memori untuk reuse fungsi penyimpanan file upload Streamlit."""

    def __init__(self, name: str, content: bytes) -> None:
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


def _find_sqlite_db(doc_stem: str, output_dir: Path) -> Path | None:
    """Temukan file database SQLite terkait dokumen di berbagai kemungkinan lokasi."""
    db = _get_sqlite_db_for_file(doc_stem, output_dir)
    if db and db.is_file():
        return db
    direct = output_dir / f"{doc_stem}.sqlite"
    if direct.is_file():
        return direct
    job = JobManager.get_instance().get_job(doc_stem, output_dir=output_dir)
    if job and job.db_file and job.db_file.is_file():
        return job.db_file
    return None


def _find_document_markdown(doc_stem: str, output_dir: Path) -> Path | None:
    """Temukan file Markdown hasil ekstraksi dokumen di berbagai kandidat folder."""
    candidates = [
        output_dir / doc_stem / f"{doc_stem}.md",
        output_dir / doc_stem / "document.md",
        output_dir / f"{doc_stem}.md",
    ]
    for c in candidates:
        if c.is_file():
            return c
    job = JobManager.get_instance().get_job(doc_stem, output_dir=output_dir)
    if job and job.out_file and job.out_file.is_file():
        return job.out_file
    return None


def _extract_page_number(stem: str) -> int | None:
    """Ekstraksi nomor halaman atau slide dari stem file gambar secara akurat."""
    # 1. Pola eksplisit seperti page_1, page-0001, slide_2, p01, sheet_3
    m = re.search(r"(?:page|slide|sheet|halaman|p)[_-]?(\d+)", stem, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # 2. Pola angka di akhir nama file (misal document_1, doc-02)
    m = re.search(r"(\d+)$", stem)
    if m:
        return int(m.group(1))
    # 3. Fallback: urutan angka terakhir dalam nama file
    nums = re.findall(r"\d+", stem)
    if nums:
        return int(nums[-1])
    return None


def _find_page_image(doc_stem: str, page_number: int, output_dir: Path) -> Path | None:
    """Cari file citra visual untuk nomor halaman tertentu (pages, slides, sheet_previews, atau fallback)."""
    if page_number < 1:
        return None
    clean_stem = Path(doc_stem).stem.strip() or "document"
    # 1. Folder pages (PDF)
    pages_dir = output_dir / clean_stem / "pages"
    if pages_dir.is_dir():
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            for img in sorted(pages_dir.glob(ext)):
                p = _extract_page_number(img.stem)
                if p == page_number:
                    return img
    # 2. Folder slides (PPT)
    slides_dir = output_dir / clean_stem / "slides"
    if slides_dir.is_dir():
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            for img in sorted(slides_dir.glob(ext)):
                p = _extract_page_number(img.stem)
                if p == page_number:
                    return img
    # 3. Folder sheet_previews (Excel)
    sheet_dir = output_dir / clean_stem / "sheet_previews"
    if sheet_dir.is_dir():
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            for img in sorted(sheet_dir.glob(ext)):
                p = _extract_page_number(img.stem)
                if p == page_number:
                    return img
    # 4. Fallback get_document_images
    imgs = get_document_images(clean_stem, output_dir)
    for img in imgs:
        p = _extract_page_number(img.stem)
        if p == page_number:
            return img
    if 1 <= page_number <= len(imgs):
        return imgs[page_number - 1]
    return None


def _image_media_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    return "image/png"


def _check_rag_status(doc_stem: str, output_dir: Path) -> tuple[bool, int]:
    """Periksa apakah dokumen sudah diindeks ke vector store (JSON atau Chroma) dan jumlah chunk-nya."""
    clean_stem = Path(doc_stem).stem.strip() or "document"
    chroma_dir = output_dir / clean_stem / "chroma"
    if not chroma_dir.is_dir():
        return False, 0
    json_file = chroma_dir / "vector_store.json"
    if json_file.is_file() and json_file.stat().st_size > 0:
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            chunk_count = len(data) if isinstance(data, (dict, list)) else 0
            return True, chunk_count
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return True, 0
    chroma_sqlite = chroma_dir / "chroma.sqlite3"
    if chroma_sqlite.is_file() and chroma_sqlite.stat().st_size > 0:
        try:
            conn = sqlite3.connect(
                f"{chroma_sqlite.resolve().as_uri()}?mode=ro", uri=True
            )
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {r[0] for r in cursor.fetchall()}
            count = 0
            if "embeddings" in tables:
                cursor.execute("SELECT count(*) FROM embeddings")
                count = cursor.fetchone()[0]
            elif "embedding_metadata" in tables:
                cursor.execute("SELECT count(DISTINCT id) FROM embedding_metadata")
                count = cursor.fetchone()[0]
            elif "embeddings_queue" in tables:
                cursor.execute("SELECT count(*) FROM embeddings_queue")
                count = cursor.fetchone()[0]
            conn.close()
            return True, count
        except (sqlite3.Error, OSError):
            return True, 0
    files = [f for f in chroma_dir.iterdir() if f.is_file()]
    if files:
        return True, 0
    return False, 0


def _index_document_rag(
    doc_stem: str,
    output_dir: Path,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict[str, Any]:
    """Muat Markdown & citra halaman dokumen lalu indeks ke vector store."""
    clean_stem = Path(doc_stem).stem.strip() or "document"
    if chunk_size is not None and chunk_size <= 0:
        raise HTTPException(400, "chunk_size harus bernilai lebih besar dari 0.")
    if chunk_overlap is not None and chunk_overlap < 0:
        raise HTTPException(400, "chunk_overlap tidak boleh bernilai negatif.")
    if (
        chunk_size is not None
        and chunk_overlap is not None
        and chunk_overlap >= chunk_size
    ):
        raise HTTPException(400, "chunk_overlap harus bernilai lebih kecil dari chunk_size.")

    md_file = _find_document_markdown(clean_stem, output_dir)
    if not md_file:
        raise HTTPException(
            404, f"File Markdown untuk dokumen '{clean_stem}' tidak ditemukan."
        )
    markdown_text = md_file.read_text(encoding="utf-8", errors="replace")

    page_images: dict[int, str] = {}
    pages_dir = output_dir / clean_stem / "pages"
    if pages_dir.is_dir():
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            for img_path in sorted(pages_dir.glob(ext)):
                p = _extract_page_number(img_path.stem)
                if p is not None:
                    page_images[p] = str(img_path.resolve())

    if not page_images:
        slides_dir = output_dir / clean_stem / "slides"
        if slides_dir.is_dir():
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
                for img_path in sorted(slides_dir.glob(ext)):
                    p = _extract_page_number(img_path.stem)
                    if p is not None:
                        page_images[p] = str(img_path.resolve())

    if not page_images:
        doc_imgs = get_document_images(clean_stem, output_dir)
        for idx, p in enumerate(doc_imgs, start=1):
            p_num = _extract_page_number(p.stem) or idx
            page_images[p_num] = str(p.resolve())

    from app.vector_store import index_markdown_document

    return index_markdown_document(
        markdown_text=markdown_text,
        doc_stem=clean_stem,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        page_images=page_images if page_images else None,
        persist_directory=output_dir / clean_stem / "chroma",
    )


def _schedule_auto_rag_indexing(doc_stem: str, output_dir: Path) -> None:
    """Pantau job ekstraksi di background thread dan jalankan auto-index RAG setelah selesai."""

    def _worker() -> None:
        manager = JobManager.get_instance()
        deadline = time.monotonic() + 3600
        consecutive_missing = 0
        job = None
        while time.monotonic() < deadline:
            job = manager.get_job(doc_stem, output_dir=output_dir)
            if not job:
                consecutive_missing += 1
                if consecutive_missing > 10:  # toleransi inisialisasi job hingga 5 detik
                    break
            else:
                consecutive_missing = 0
                if job.status in {"completed", "failed", "canceled"}:
                    break
            time.sleep(0.5)

        if job and job.status == "completed":
            for _ in range(10):
                if _find_document_markdown(doc_stem, output_dir) is not None:
                    break
                time.sleep(0.2)
            try:
                _index_document_rag(doc_stem, output_dir)
                logger.info("Otomatis indeks RAG berhasil untuk %s", doc_stem)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Gagal otomatis indeks RAG untuk %s: %s", doc_stem, exc)

    thread = threading.Thread(
        target=_worker, daemon=True, name=f"AutoRAG-{doc_stem}"
    )
    thread.start()


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


# ==============================================================================
# Pydantic Schemas untuk API
# ==============================================================================


class BatchUploadDocumentItem(BaseModel):
    stem: str
    source_name: str
    job_id: str
    status: str = "queued"


class BatchUploadResponse(BaseModel):
    batch_id: str
    batch_name: str
    created_at: str
    uploaded_at: str | None = None
    auto_index_rag: bool = False
    document_stems: list[str]
    job_ids: list[str]
    documents: list[BatchUploadDocumentItem]


class EnrichedDocumentStatus(BaseModel):
    stem: str
    source_name: str
    job_id: str
    status: str
    stage: str | None = None
    progress_pct: float = 0.0
    current_page: int = 0
    total_pages: int = 0
    started_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None
    error_message: str | None = None


class BatchDetailResponse(BaseModel):
    id: str
    name: str
    created_at: str
    uploaded_at: str | None = None
    status: str
    total_documents: int
    documents: list[EnrichedDocumentStatus]


class JobStatusResponse(BaseModel):
    job_id: str
    file_name: str
    status: str
    stage: str
    progress_pct: float
    current_page: int
    total_pages: int
    queue_position: int = 0
    started_at: str
    updated_at: str
    completed_at: str | None = None
    error_message: str | None = None
    last_message: str = ""
    log_snippet: str = ""
    extraction_options: dict[str, Any] = Field(default_factory=dict)


class JobActionResponse(BaseModel):
    status: str
    job_id: str
    message: str


class BatchDeleteResponse(BaseModel):
    status: str
    batch_id: str
    removed_stems: list[str]
    message: str


class DocumentSummaryResponse(BaseModel):
    doc_stem: str
    status: str
    total_pages: int
    table_count: int
    tables: list[str] = Field(default_factory=list)
    rag_indexed: bool
    rag_chunk_count: int = 0
    markdown_exists: bool
    has_images: bool


class PagePreviewSummary(BaseModel):
    page_number: int
    character_count: int
    line_count: int
    heading: str | None = None
    snippet: str
    has_table: bool
    image_filename: str | None = None
    image_url: str | None = None


class TableSummary(BaseModel):
    name: str
    columns: list[str]
    row_count: int


class DocumentPreviewResponse(BaseModel):
    doc_stem: str
    preamble: str
    total_pages: int
    page_summaries: list[PagePreviewSummary]
    chunking_preview: dict[str, Any]
    detected_tables: list[TableSummary]


class TableColumnInfo(BaseModel):
    cid: int
    name: str
    type: str
    notnull: bool
    pk: bool


class TableDetailInfo(BaseModel):
    name: str
    columns: list[str]
    column_details: list[TableColumnInfo]
    row_count: int


class DocumentTablesResponse(BaseModel):
    doc_stem: str
    total_tables: int
    tables: list[TableDetailInfo]


class TableRowsResponse(BaseModel):
    doc_stem: str
    table_name: str
    total_rows: int
    limit: int
    offset: int
    rows: list[dict[str, Any]]


class DocumentIndexRAGRequest(BaseModel):
    chunk_size: int | None = Field(default=None, description="Ukuran karakter per chunk")
    chunk_overlap: int | None = Field(default=None, description="Overlap karakter antar chunk")


def _enrich_batch(
    batch: dict[str, Any], manager: JobManager, output_dir: Path
) -> BatchDetailResponse:
    """Perkaya manifest batch dengan status real-time tiap dokumen dari JobManager."""
    enriched_docs: list[EnrichedDocumentStatus] = []
    jobs: list[Any] = []
    for doc in batch.get("documents", []):
        stem = doc.get("stem", "")
        job = manager.get_job(stem, output_dir=output_dir) if stem else None
        jobs.append(job)
        if job:
            enriched_docs.append(
                EnrichedDocumentStatus(
                    stem=stem,
                    source_name=doc.get("source_name", job.file_name),
                    job_id=job.job_id,
                    status=job.status,
                    stage=job.stage,
                    progress_pct=job.progress_percentage(),
                    current_page=job.current_page,
                    total_pages=job.total_pages,
                    started_at=job.started_at,
                    updated_at=job.updated_at,
                    completed_at=job.completed_at,
                    error_message=job.error_message,
                )
            )
        else:
            md_file = _find_document_markdown(stem, output_dir) if stem else None
            is_done = md_file is not None and md_file.is_file()
            if is_done:
                from types import SimpleNamespace
                jobs[-1] = SimpleNamespace(status="completed")
            enriched_docs.append(
                EnrichedDocumentStatus(
                    stem=stem,
                    source_name=doc.get("source_name", stem),
                    job_id=stem,
                    status="completed" if is_done else "unknown",
                    stage="Selesai" if is_done else None,
                    progress_pct=100.0 if is_done else 0.0,
                    current_page=0,
                    total_pages=0,
                    started_at=None,
                    updated_at=None,
                    completed_at=None,
                    error_message=None,
                )
            )
    return BatchDetailResponse(
        id=str(batch.get("id")),
        name=str(batch.get("name")),
        created_at=str(batch.get("created_at")),
        uploaded_at=batch.get("uploaded_at"),
        status=batch_status(jobs),
        total_documents=len(enriched_docs),
        documents=enriched_docs,
    )


# ==============================================================================
# Endpoint Status & Dokumentasi Dasar
# ==============================================================================


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


# ==============================================================================
# Endpoint Ingest Klasik (Backward Compatible)
# ==============================================================================


@app.post(
    "/ingest",
    summary="Ingest satu file",
    response_class=Response,
    responses={
        200: {
            "content": {
                "application/zip": {"schema": {"type": "string", "format": "binary"}}
            }
        },
        400: {"description": "File kosong"},
        413: {"description": "File melebihi 100 MiB"},
        415: {"description": "Format file tidak didukung"},
        500: {"description": "Ingest atau ekspor gagal"},
    },
)
def ingest(
    file: Annotated[
        UploadFile,
        File(
            description=(
                "PDF, DOC/DOCX, Excel, PPT/PPTX, PNG, JPG/JPEG, atau WebP; maksimum 100 MiB"
            )
        ),
    ],
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
    while job.status in {"queued", "running"}:
        time.sleep(0.5)
        job = manager.get_job(job.job_id, output_dir=OUTPUT_DIR) or job
    if job.status != "completed":
        raise HTTPException(
            500,
            {
                "job_id": job.job_id,
                "message": "Ingest gagal. Periksa log di folder output.",
            },
        )
    try:
        archive = build_result_zip(job)
    except (OSError, sqlite3.Error):
        logger.exception("Gagal mengekspor hasil %s", job.job_id)
        raise HTTPException(
            500, {"job_id": job.job_id, "message": "Hasil ingest tidak dapat dibaca."}
        ) from None
    return Response(
        archive,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="hasil_ingest.zip"'},
    )


# ==============================================================================
# Endpoint Batch Upload & Riwayat (Workspace Equivalence)
# ==============================================================================


@app.post(
    "/batches/upload",
    summary="Unggah satu atau banyak file dokumen sebagai batch dan mulai proses ekstraksi background",
    response_model=BatchUploadResponse,
)
def upload_batch(
    request: Request,
    files: Annotated[
        list[UploadFile] | None,
        File(
            description="Daftar file yang akan diunggah",
        ),
    ] = None,
    file: Annotated[
        UploadFile | None,
        File(
            description="Satu file dokumen (opsional jika menggunakan 'file')",
        ),
    ] = None,
    batch_name: Annotated[
        str | None,
        Form(
            description="Nama batch upload opsional",
        ),
    ] = None,
    auto_index_rag: Annotated[
        bool,
        Form(
            description="Otomatis indeks ke RAG saat ekstraksi selesai",
        ),
    ] = False,
) -> BatchUploadResponse:
    effective_name = (
        batch_name
        or request.query_params.get("batch_name")
        or "Uploaded files"
    )
    q_auto = request.query_params.get("auto_index_rag", "").lower()
    effective_auto_index = auto_index_rag or q_auto in {"1", "true", "yes"}

    all_upload_files = list(files or [])
    if file is not None:
        all_upload_files.append(file)
    if not all_upload_files:
        raise HTTPException(400, "Tidak ada file yang diunggah.")

    saved_paths: list[Path] = []
    try:
        for up_file in all_upload_files:
            filename = Path((up_file.filename or "").replace("\\", "/")).name
            if not filename:
                raise HTTPException(400, "Nama file tidak valid atau kosong.")
            extension = Path(filename).suffix.lower()
            if extension not in SUPPORTED_TYPES:
                raise HTTPException(415, f"Format file '{filename}' tidak didukung.")

            size = 0
            buffer = BytesIO()
            while chunk := up_file.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        413, f"File '{filename}' melebihi batas ukuran 100 MiB."
                    )
                buffer.write(chunk)
            up_file.file.close()

            if size == 0:
                raise HTTPException(400, f"File '{filename}' kosong.")

            content = buffer.getvalue()
            saved = _save_uploaded_file(_UploadedFileMemory(filename, content), OUTPUT_DIR)
            saved_paths.append(saved)
    except Exception:
        for path in saved_paths:
            path.unlink(missing_ok=True)
        raise

    documents = [
        {"stem": path.stem, "source_name": path.name}
        for path in saved_paths
    ]
    batch = create_batch(OUTPUT_DIR, effective_name, documents)

    manager = JobManager.get_instance()
    started_jobs = []
    for queue_pos, path in enumerate(saved_paths):
        job = manager.start_job(
            input_path=path,
            output_dir=OUTPUT_DIR,
            queue_position=queue_pos,
        )
        started_jobs.append(job)

    if effective_auto_index:
        for path in saved_paths:
            _schedule_auto_rag_indexing(path.stem, OUTPUT_DIR)

    job_map = {job.job_id: job for job in started_jobs}
    document_items = [
        BatchUploadDocumentItem(
            stem=doc["stem"],
            source_name=doc.get("source_name", doc["stem"]),
            job_id=job_map[doc["stem"]].job_id if doc["stem"] in job_map else doc["stem"],
            status=job_map[doc["stem"]].status if doc["stem"] in job_map else "queued",
        )
        for doc in batch["documents"]
    ]

    return BatchUploadResponse(
        batch_id=batch["id"],
        batch_name=batch["name"],
        created_at=batch["created_at"],
        uploaded_at=batch.get("uploaded_at"),
        auto_index_rag=effective_auto_index,
        document_stems=[doc["stem"] for doc in batch["documents"]],
        job_ids=[job.job_id for job in started_jobs],
        documents=document_items,
    )


@app.get(
    "/batches",
    summary="Daftar seluruh batch upload beserta status real-time setiap dokumen",
    response_model=list[BatchDetailResponse],
)
def get_batches() -> list[BatchDetailResponse]:
    batches = list_batches(OUTPUT_DIR)
    manager = JobManager.get_instance()
    return [_enrich_batch(b, manager, OUTPUT_DIR) for b in batches]


@app.get(
    "/batches/{batch_id}",
    summary="Detail batch upload beserta status pekerjaan seluruh dokumen di dalamnya",
    response_model=BatchDetailResponse,
)
def get_batch_detail(batch_id: str) -> BatchDetailResponse:
    batches = list_batches(OUTPUT_DIR)
    batch = next((b for b in batches if b.get("id") == batch_id), None)
    if batch is None:
        raise HTTPException(404, f"Batch '{batch_id}' tidak ditemukan.")
    return _enrich_batch(batch, JobManager.get_instance(), OUTPUT_DIR)


@app.delete(
    "/batches/{batch_id}",
    summary="Hapus batch dan artifact dokumen yang tidak lagi direferensikan",
    response_model=BatchDeleteResponse,
)
def delete_batch_endpoint(batch_id: str) -> BatchDeleteResponse:
    manager = JobManager.get_instance()
    batches = list_batches(OUTPUT_DIR)
    batch = next((b for b in batches if b.get("id") == batch_id), None)
    if batch is None:
        raise HTTPException(404, f"Batch '{batch_id}' tidak ditemukan.")
    stems = [
        doc.get("stem", "")
        for doc in batch.get("documents", [])
        if doc.get("stem")
    ]
    _cancel_jobs_before_delete(manager, stems, OUTPUT_DIR)
    try:
        result = delete_batch(OUTPUT_DIR, batch_id)
        return BatchDeleteResponse(
            status="deleted",
            batch_id=batch_id,
            removed_stems=result.get("removed_stems", []),
            message=f"Batch '{batch.get('name')}' berhasil dihapus.",
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc


# ==============================================================================
# Endpoint Manajemen Job Background
# ==============================================================================


@app.get(
    "/jobs/{stem}",
    summary="Periksa status, progres, dan cuplikan log pekerjaan ekstraksi",
    response_model=JobStatusResponse,
)
def get_job_status(stem: str) -> JobStatusResponse:
    manager = JobManager.get_instance()
    job = manager.get_job(stem, output_dir=OUTPUT_DIR)
    if not job:
        raise HTTPException(404, f"Job '{stem}' tidak ditemukan.")
    return JobStatusResponse(
        job_id=job.job_id,
        file_name=job.file_name,
        status=job.status,
        stage=job.stage,
        progress_pct=job.progress_percentage(),
        current_page=job.current_page,
        total_pages=job.total_pages,
        queue_position=job.queue_position,
        started_at=job.started_at,
        updated_at=job.updated_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
        last_message=job.last_message,
        log_snippet=manager.get_latest_logs(stem, line_count=40),
        extraction_options=job.extraction_options,
    )


@app.post(
    "/jobs/{stem}/cancel",
    summary="Batalkan pekerjaan ekstraksi yang sedang berjalan atau antre",
    response_model=JobActionResponse,
)
def cancel_job_endpoint(stem: str) -> JobActionResponse:
    manager = JobManager.get_instance()
    job = manager.get_job(stem, output_dir=OUTPUT_DIR)
    if not job:
        raise HTTPException(404, f"Job '{stem}' tidak ditemukan.")
    if job.status not in {"queued", "running", "paused"}:
        raise HTTPException(
            400,
            f"Job '{stem}' berstatus '{job.status}' dan tidak dapat dibatalkan.",
        )
    manager.cancel_job(stem)
    return JobActionResponse(
        status="canceled",
        job_id=stem,
        message=f"Job '{stem}' berhasil dibatalkan.",
    )


@app.post(
    "/jobs/{stem}/retry",
    summary="Jalankan ulang ekstraksi untuk dokumen",
    response_model=JobActionResponse,
)
def retry_job_endpoint(stem: str) -> JobActionResponse:
    manager = JobManager.get_instance()
    job = manager.get_job(stem, output_dir=OUTPUT_DIR)
    if not job:
        uploads = OUTPUT_DIR / "uploads"
        matches = (
            [p for p in uploads.iterdir() if p.is_file() and p.stem == stem]
            if uploads.exists()
            else []
        )
        if not matches:
            raise HTTPException(
                404, f"Job atau file sumber '{stem}' tidak ditemukan."
            )
    try:
        new_job = manager.restart_job(stem, output_dir=OUTPUT_DIR)
        return JobActionResponse(
            status=new_job.status,
            job_id=new_job.job_id,
            message=f"Job '{stem}' berhasil dijalankan ulang.",
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


# ==============================================================================
# Endpoint Inspeksi Dokumen (Metadata, Markdown, Preview, Tabel, Gambar, Download)
# ==============================================================================


@app.get(
    "/documents/{doc_stem}",
    summary="Ringkasan metadata dokumen hasil ekstraksi",
    response_model=DocumentSummaryResponse,
)
def get_document_summary(doc_stem: str) -> DocumentSummaryResponse:
    clean_stem = Path(doc_stem).stem.strip() or "document"
    md_file = _find_document_markdown(clean_stem, OUTPUT_DIR)
    doc_dir = OUTPUT_DIR / clean_stem
    job = JobManager.get_instance().get_job(clean_stem, output_dir=OUTPUT_DIR)
    if not doc_dir.exists() and not md_file and not job:
        raise HTTPException(
            404, f"Dokumen '{clean_stem}' tidak ditemukan."
        )

    imgs = get_document_images(clean_stem, OUTPUT_DIR)
    # Cek juga folder sheet_previews jika imgs kosong (misal dokumen Excel)
    if not imgs:
        sheet_dir = OUTPUT_DIR / clean_stem / "sheet_previews"
        if sheet_dir.is_dir():
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
                imgs.extend(sorted(sheet_dir.glob(ext)))

    if imgs:
        total_pages = len(imgs)
    elif job and job.total_pages > 0:
        total_pages = job.total_pages
    elif md_file and md_file.is_file():
        pages = split_markdown_by_pages(
            md_file.read_text(encoding="utf-8", errors="replace")
        )
        total_pages = len(pages)
    else:
        total_pages = 0

    table_names: list[str] = []
    db_file = _find_sqlite_db(clean_stem, OUTPUT_DIR)
    if db_file and db_file.is_file():
        conn = sqlite3.connect(f"{db_file.resolve().as_uri()}?mode=ro", uri=True)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            table_names = [r[0] for r in cursor.fetchall()]
        finally:
            conn.close()

    rag_indexed, rag_chunk_count = _check_rag_status(clean_stem, OUTPUT_DIR)
    status = (
        job.status
        if job
        else ("completed" if md_file and md_file.is_file() else "unknown")
    )

    return DocumentSummaryResponse(
        doc_stem=clean_stem,
        status=status,
        total_pages=total_pages,
        table_count=len(table_names),
        tables=table_names,
        rag_indexed=rag_indexed,
        rag_chunk_count=rag_chunk_count,
        markdown_exists=md_file is not None and md_file.is_file(),
        has_images=len(imgs) > 0,
    )


@app.get(
    "/documents/{doc_stem}/markdown",
    summary="Ambil teks mentah hasil ekstraksi Markdown dokumen",
    response_class=PlainTextResponse,
)
def get_document_markdown(doc_stem: str) -> PlainTextResponse:
    clean_stem = Path(doc_stem).stem.strip() or "document"
    md_file = _find_document_markdown(clean_stem, OUTPUT_DIR)
    if not md_file:
        raise HTTPException(
            404, f"File Markdown untuk dokumen '{clean_stem}' tidak ditemukan."
        )
    content = md_file.read_text(encoding="utf-8", errors="replace")
    return PlainTextResponse(content, media_type="text/markdown; charset=utf-8")


@app.get(
    "/documents/{doc_stem}/preview",
    summary="Pratinjau lengkap dokumen hasil ekstraksi sebelum indeks RAG",
    response_model=DocumentPreviewResponse,
)
def get_document_preview(
    doc_stem: str,
    chunk_size: Annotated[int, Query(ge=50, le=10000)] = 1000,
    chunk_overlap: Annotated[int, Query(ge=0, le=1000)] = 150,
) -> DocumentPreviewResponse:
    if chunk_overlap >= chunk_size:
        raise HTTPException(
            400, "chunk_overlap harus bernilai lebih kecil dari chunk_size."
        )

    clean_stem = Path(doc_stem).stem.strip() or "document"
    md_file = _find_document_markdown(clean_stem, OUTPUT_DIR)
    if not md_file:
        raise HTTPException(
            404, f"File Markdown untuk dokumen '{clean_stem}' tidak ditemukan."
        )
    markdown_text = md_file.read_text(encoding="utf-8", errors="replace")

    # 1. Preamble
    matches = list(
        re.finditer(r"<!--\s*PAGE:\s*(\d+)\s*-->", markdown_text, re.IGNORECASE)
    )
    if matches:
        preamble = markdown_text[: matches[0].start()].strip()
    else:
        paragraphs = markdown_text.strip().split("\n\n")
        preamble = paragraphs[0].strip() if paragraphs else ""

    # 2. Page summaries
    pages = split_markdown_by_pages(markdown_text)
    doc_imgs = get_document_images(clean_stem, OUTPUT_DIR)
    page_img_map: dict[int, str] = {}
    for img in doc_imgs:
        p = _extract_page_number(img.stem)
        if p is not None:
            page_img_map[p] = img.name

    page_summaries: list[PagePreviewSummary] = []
    for p_num, content in sorted(pages.items()):
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        heading = next((line for line in lines if line.startswith("#")), None)
        has_table = (
            "|" in content
            and ("|-" in content or "|:-" in content or "| ---" in content)
        )
        snippet = content[:300].strip()
        if len(content) > 300:
            snippet += "..."
        img_name = page_img_map.get(p_num)
        has_img = img_name is not None or (1 <= p_num <= len(doc_imgs))
        page_summaries.append(
            PagePreviewSummary(
                page_number=p_num,
                character_count=len(content),
                line_count=len(lines),
                heading=heading,
                snippet=snippet,
                has_table=has_table,
                image_filename=img_name
                or (
                    doc_imgs[p_num - 1].name
                    if (1 <= p_num <= len(doc_imgs))
                    else None
                ),
                image_url=f"/documents/{clean_stem}/pages/{p_num}/image"
                if has_img
                else None,
            )
        )

    # 3. Chunking simulation preview
    from app.rag import preview_markdown_chunks

    try:
        preview = preview_markdown_chunks(
            markdown_content=markdown_text,
            source_file=clean_stem,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            page_images=doc_imgs,
            doc_stem=clean_stem,
        )
        chunking_dict = preview.to_dict()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gagal melakukan simulasi chunking: %s", exc)
        chunking_dict = {
            "source_file": clean_stem,
            "total_characters": len(markdown_text),
            "total_chunks": 0,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "avg_chunk_size": 0.0,
            "multimodal_chunks_count": 0,
            "chunks": [],
        }

    # 4. Detected tables
    detected_tables: list[TableSummary] = []
    db_file = _find_sqlite_db(clean_stem, OUTPUT_DIR)
    if db_file and db_file.is_file():
        conn = sqlite3.connect(f"{db_file.resolve().as_uri()}?mode=ro", uri=True)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            for (t_name,) in cursor.fetchall():
                try:
                    quoted = '"' + t_name.replace('"', '""') + '"'
                    cursor.execute(f"PRAGMA table_info({quoted})")
                    cols = [r[1] for r in cursor.fetchall()]
                    cursor.execute(f"SELECT count(*) FROM {quoted}")
                    count = cursor.fetchone()[0]
                    detected_tables.append(
                        TableSummary(name=t_name, columns=cols, row_count=count)
                    )
                except sqlite3.Error:
                    continue
        finally:
            conn.close()

    return DocumentPreviewResponse(
        doc_stem=clean_stem,
        preamble=preamble,
        total_pages=len(pages),
        page_summaries=page_summaries,
        chunking_preview=chunking_dict,
        detected_tables=detected_tables,
    )


@app.get(
    "/documents/{doc_stem}/tables",
    summary="Daftar tabel SQLite, kolom, dan jumlah baris untuk dokumen",
    response_model=DocumentTablesResponse,
)
def get_document_tables(doc_stem: str) -> DocumentTablesResponse:
    clean_stem = Path(doc_stem).stem.strip() or "document"
    md_file = _find_document_markdown(clean_stem, OUTPUT_DIR)
    doc_dir = OUTPUT_DIR / clean_stem
    job = JobManager.get_instance().get_job(clean_stem, output_dir=OUTPUT_DIR)
    if not doc_dir.exists() and not md_file and not job:
        raise HTTPException(
            404, f"Dokumen '{clean_stem}' tidak ditemukan."
        )

    db_file = _find_sqlite_db(clean_stem, OUTPUT_DIR)
    if not db_file or not db_file.is_file():
        return DocumentTablesResponse(
            doc_stem=clean_stem, total_tables=0, tables=[]
        )

    tables: list[TableDetailInfo] = []
    conn = sqlite3.connect(f"{db_file.resolve().as_uri()}?mode=ro", uri=True)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        for (t_name,) in cursor.fetchall():
            try:
                quoted = '"' + t_name.replace('"', '""') + '"'
                cursor.execute(f"PRAGMA table_info({quoted})")
                columns: list[TableColumnInfo] = []
                for col in cursor.fetchall():
                    columns.append(
                        TableColumnInfo(
                            cid=col[0],
                            name=col[1],
                            type=col[2],
                            notnull=bool(col[3]),
                            pk=bool(col[5]),
                        )
                    )
                cursor.execute(f"SELECT count(*) FROM {quoted}")
                count = cursor.fetchone()[0]
                tables.append(
                    TableDetailInfo(
                        name=t_name,
                        columns=[c.name for c in columns],
                        column_details=columns,
                        row_count=count,
                    )
                )
            except sqlite3.Error:
                continue
    finally:
        conn.close()

    return DocumentTablesResponse(
        doc_stem=clean_stem, total_tables=len(tables), tables=tables
    )


@app.get(
    "/documents/{doc_stem}/tables/{table_name}",
    summary="Ambil baris-baris data dari tabel SQLite dokumen sebagai JSON",
    response_model=TableRowsResponse,
)
def get_document_table_rows(
    doc_stem: str,
    table_name: str,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> TableRowsResponse:
    if table_name.startswith("sqlite_"):
        raise HTTPException(
            404,
            f"Tabel '{table_name}' tidak ditemukan pada database dokumen.",
        )

    clean_stem = Path(doc_stem).stem.strip() or "document"
    db_file = _find_sqlite_db(clean_stem, OUTPUT_DIR)
    if not db_file or not db_file.is_file():
        raise HTTPException(
            404, f"Database untuk dokumen '{clean_stem}' tidak ditemukan."
        )

    conn = sqlite3.connect(f"{db_file.resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name = ? AND name NOT LIKE 'sqlite_%'",
            (table_name,),
        )
        if not cursor.fetchone():
            raise HTTPException(
                404,
                f"Tabel '{table_name}' tidak ditemukan pada database dokumen '{clean_stem}'.",
            )
        quoted = '"' + table_name.replace('"', '""') + '"'
        cursor.execute(f"SELECT count(*) FROM {quoted}")
        total_rows = cursor.fetchone()[0]
        cursor.execute(f"SELECT * FROM {quoted} LIMIT ? OFFSET ?", (limit, offset))
        raw_rows = cursor.fetchall()
        rows = []
        for r in raw_rows:
            d = {}
            for k, v in dict(r).items():
                if isinstance(v, bytes):
                    d[k] = v.decode("utf-8", errors="replace")
                else:
                    d[k] = v
            rows.append(d)
    finally:
        conn.close()

    return TableRowsResponse(
        doc_stem=clean_stem,
        table_name=table_name,
        total_rows=total_rows,
        limit=limit,
        offset=offset,
        rows=rows,
    )


@app.get(
    "/documents/{doc_stem}/pages/{page_number}/image",
    summary="Ambil citra kanvas halaman fisik dokumen",
    response_class=FileResponse,
)
def get_document_page_image(
    doc_stem: str,
    page_number: int,
) -> FileResponse:
    if page_number < 1:
        raise HTTPException(400, "Nomor halaman harus bernilai minimal 1.")
    clean_stem = Path(doc_stem).stem.strip() or "document"
    img_path = _find_page_image(clean_stem, page_number, OUTPUT_DIR)
    if not img_path or not img_path.is_file():
        raise HTTPException(
            404,
            f"Citra halaman {page_number} untuk dokumen '{clean_stem}' tidak ditemukan.",
        )
    return FileResponse(
        path=str(img_path),
        media_type=_image_media_type(img_path),
    )


@app.get(
    "/documents/{doc_stem}/download",
    summary="Unduh paket ZIP hasil ekstraksi lengkap satu dokumen",
    response_class=Response,
)
def download_document_zip(doc_stem: str) -> Response:
    clean_stem = Path(doc_stem).stem.strip() or "document"
    md_file = _find_document_markdown(clean_stem, OUTPUT_DIR)
    doc_dir = OUTPUT_DIR / clean_stem
    job = JobManager.get_instance().get_job(clean_stem, output_dir=OUTPUT_DIR)
    if not doc_dir.exists() and not md_file and not job:
        raise HTTPException(
            404, f"Dokumen '{clean_stem}' tidak ditemukan."
        )
    try:
        archive = build_document_zip(clean_stem, OUTPUT_DIR)
    except Exception as exc:
        logger.exception("Gagal membangun arsip zip dokumen %s", clean_stem)
        raise HTTPException(
            500, f"Gagal membuat paket ZIP untuk '{clean_stem}'."
        ) from exc
    return Response(
        archive,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{clean_stem}_hasil.zip"'
        },
    )


# ==============================================================================
# RAG Injection & Retrieval API (Khusus Chatbot Eksternal)
# ==============================================================================


class RAGRetrieveRequest(BaseModel):
    query: str = Field(..., description="Query pencarian semantik dari chatbot eksternal")
    doc_stem: str = Field(default="document", description="Identifier dokumen yang dicari")
    top_k: int = Field(default=4, ge=1, le=50, description="Jumlah chunk yang dikembalikan")
    filter_metadata: dict[str, Any] | None = Field(
        default=None, description="Filter metadata spesifik (mis. page_number, chapter)"
    )


class RAGRetrieveItem(BaseModel):
    chunk_id: int | str
    content: str
    score: float
    page_number: int | None = None
    image_path: str | None = None
    image_metadata: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGRetrieveResponse(BaseModel):
    query: str
    doc_stem: str
    total_results: int
    results: list[RAGRetrieveItem]


class RAGIndexRequest(BaseModel):
    markdown_text: str = Field(..., description="Teks Markdown dokumen yang akan diindeks")
    doc_stem: str = Field(default="document", description="Identifier dokumen")
    chunk_size: int | None = Field(default=None, description="Ukuran karakter per chunk")
    chunk_overlap: int | None = Field(default=None, description="Overlap karakter antar chunk")
    page_images: dict[int, str] | dict[str, str] | list[str] | None = Field(
        default=None,
        description="Pemetaan nomor halaman -> path citra visual halaman, atau list path citra berurutan",
    )


class RAGIndexResponse(BaseModel):
    status: str
    doc_stem: str
    total_chunks: int
    multimodal_chunks: int
    total_characters: int
    persist_directory: str
    backend: str
    chunk_ids: list[str]


@app.post(
    "/rag/retrieve",
    summary="Temu kembali konteks dokumen (Teks + Citra) untuk chatbot eksternal",
    response_model=RAGRetrieveResponse,
)
def rag_retrieve(payload: RAGRetrieveRequest) -> RAGRetrieveResponse:
    """
    Endpoint retrieval semantik dokumen untuk dikonsumsi oleh sistem chatbot utama.
    Mengembalikan potongan teks relevan, nomor halaman, dan path citra visual halaman.
    """
    from app.vector_store import query_document_knowledge_base

    clean_stem = Path(payload.doc_stem).stem.strip() or "document"
    results = query_document_knowledge_base(
        query=payload.query,
        doc_stem=clean_stem,
        top_k=payload.top_k,
        filter_metadata=payload.filter_metadata,
    )
    items = [
        RAGRetrieveItem(
            chunk_id=r.chunk_id,
            content=r.content,
            score=r.score,
            page_number=r.page_number,
            image_path=r.image_path,
            image_metadata=r.image_metadata,
            metadata=r.metadata,
        )
        for r in results
    ]
    return RAGRetrieveResponse(
        query=payload.query,
        doc_stem=clean_stem,
        total_results=len(items),
        results=items,
    )


@app.post(
    "/rag/index",
    summary="Injeksi & indexing dokumen Markdown ke persistent vector store",
    response_model=RAGIndexResponse,
)
def rag_index(payload: RAGIndexRequest) -> RAGIndexResponse:
    """
    Endpoint injeksi dokumen ke vector database dokumen lokal.
    Mendukung pemotongan hierarkis dan asosiasi citra halaman.
    """
    if payload.chunk_size is not None and payload.chunk_size <= 0:
        raise HTTPException(400, "chunk_size harus bernilai lebih besar dari 0.")
    if payload.chunk_overlap is not None and payload.chunk_overlap < 0:
        raise HTTPException(400, "chunk_overlap tidak boleh bernilai negatif.")
    if (
        payload.chunk_size is not None
        and payload.chunk_overlap is not None
        and payload.chunk_overlap >= payload.chunk_size
    ):
        raise HTTPException(400, "chunk_overlap harus bernilai lebih kecil dari chunk_size.")

    from app.vector_store import index_markdown_document

    clean_stem = Path(payload.doc_stem).stem.strip() or "document"
    res = index_markdown_document(
        markdown_text=payload.markdown_text,
        doc_stem=clean_stem,
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
        page_images=payload.page_images,
    )
    return RAGIndexResponse(
        status=res.get("status", "unknown"),
        doc_stem=res.get("doc_stem", clean_stem),
        total_chunks=res.get("total_chunks", 0),
        multimodal_chunks=res.get("multimodal_chunks", 0),
        total_characters=res.get("total_characters", 0),
        persist_directory=res.get("persist_directory", ""),
        backend=res.get("backend", ""),
        chunk_ids=[str(c) for c in res.get("chunk_ids", []) if c is not None],
    )


@app.post(
    "/documents/{doc_stem}/index-rag",
    summary="Indeks dokumen Markdown yang sudah diekstrak ke RAG vector store",
    response_model=RAGIndexResponse,
)
def index_document_rag_endpoint(
    doc_stem: str,
    payload: DocumentIndexRAGRequest | None = None,
) -> RAGIndexResponse:
    """
    Indeks dokumen yang telah diproses ke dalam vector store setelah pengguna
    atau sistem eksternal melihat preview dan memutuskan untuk menyimpannya ke RAG.
    """
    clean_stem = Path(doc_stem).stem.strip() or "document"
    chunk_size = payload.chunk_size if payload else None
    chunk_overlap = payload.chunk_overlap if payload else None
    res = _index_document_rag(
        doc_stem=clean_stem,
        output_dir=OUTPUT_DIR,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return RAGIndexResponse(
        status=res.get("status", "unknown"),
        doc_stem=res.get("doc_stem", clean_stem),
        total_chunks=res.get("total_chunks", 0),
        multimodal_chunks=res.get("multimodal_chunks", 0),
        total_characters=res.get("total_characters", 0),
        persist_directory=res.get("persist_directory", ""),
        backend=res.get("backend", ""),
        chunk_ids=[str(c) for c in res.get("chunk_ids", []) if c is not None],
    )
