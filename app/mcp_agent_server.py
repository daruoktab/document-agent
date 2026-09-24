"""
MCP (Model Context Protocol) Server 'Agent Mode' untuk document-agent.

Server ini dirancang untuk agent model dengan kemampuan vision bawaan
(opencode, Claude Desktop, Cursor, dsb.) dan TIDAK memanggil endpoint model
apa pun: OCR, VLM, maupun LLM tidak diaktifkan. Agent-lah yang membaca gambar
dokumen secara visual, menulis Markdown-nya sendiri, lalu menyimpannya sebagai
gold data example untuk melatih agent/LLM lokal di kemudian hari.

Alur kerja yang direkomendasikan untuk agent:
  1. scan_document_folders / select_document_batch : temukan & pilih dokumen
  2. render_presentation_slides / convert_pdf_to_images (+ preprocess_image) :
     ubah dokumen menjadi gambar dan KIRIM gambar tersebut LANGSUNG ke model
     (image content) agar model dapat melihat dan membaca dokumen secara visual.
     Tool memproses DENGAN BATCH per 10 slide/halaman untuk context window efisien:
     ulangi panggilan dengan start_slide/start_page = next_start_... hingga has_more=false.
  3. Agent menulis Markdown sesuai spesifikasi layout dari gambar yang dilihat
  4. preview_markdown_chunks : [STAGING / BLUEPRINT] validasi kesiapan chunking
  5. save_extraction_result : simpan Markdown + metadata sebagai gold data (otomatis mengidentifikasi
     dan mengekstrak tabel transaksional ke database SQLite dengan verifikasi ganda).
  6. query_tabular_database / inspect_tabular_database : jalankan query SQL untuk kalkulasi agregat (SUM, AVG, Filter).

  WAJIB: proses SATU DOKUMEN sampai SEMUA BATCH-nya selesai (has_more=false) dan
  sudah disimpan ke save_extraction_result, BARU pindah ke dokumen berikutnya.

Menjalankan server:
    python -m app.mcp_agent_server
    python mcp_agent_server.py
"""

from __future__ import annotations

import base64
import json
import logging
import random
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ContentBlock, ImageContent, TextContent

from .multi_page import preview_markdown_chunks as sim_preview_chunks
from .preprocess import preprocess_image
from .prompts import MARKDOWN_LINE_BREAK_RULES, MERMAID_EXTRACTION_RULES
from .tabular_db import (
    ROW_ROLE_SQL_GUIDANCE,
    TabularDatabaseManager,
    extract_and_ingest_tables_from_markdown,
    query_sqlite,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: set[str] = {
    ".pdf",
    ".docx",
    ".doc",
    ".xlsx",
    ".xls",
    ".xlsm",
    ".ods",
    ".pptx",
    ".ppt",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}

_IGNORED_DIR_NAMES: set[str] = {
    ".git",
    ".venv",
    "__pycache__",
    ".ruff_cache",
    ".vscode",
    ".agents",
}

# Root folder proyek (mengunci relative path agar tidak nyasar ke C:\Windows\System32)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def _resolve_path(p: str | Path) -> Path:
    """
    Konversi path relatif menjadi path absolut terhadap root proyek (bukan C:\\Windows\\System32).
    Jika sudah berupa path absolut, gunakan langsung.
    """
    path_obj = Path(p)
    if path_obj.is_absolute():
        return path_obj.resolve()
    return (PROJECT_ROOT / path_obj).resolve()


# Batas ukuran gambar mentah individual: 720 KB (Base64 encoding ~960 KB, aman di bawah limit 1 MB client tool result).
_MAX_IMAGE_BYTES: int = 720 * 1024

# Batas maksimal total payload base64 dalam satu response pesan MCP (970 KB, menyisakan margin buffer ~54 KB).
MAX_TOTAL_PAYLOAD_BYTES: int = 970 * 1024

# Batas maksimal gambar yang dikirim dalam satu tool call (default 1 gambar per step per-halaman).
DEFAULT_MAX_IMAGES: int = 1

# Resolusi & kualitas gambar ultra-tajam (2400px, quality=92, 4:4:4 lossless chroma).
DEFAULT_MAX_IMAGE_DIMENSION: int = 2400
DEFAULT_JPEG_QUALITY: int = 92


def _mime_for(image_path: Path) -> str:
    """Tentukan MIME type gambar dari ekstensi file."""
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(image_path.suffix.lower(), "image/jpeg")


def _image_to_content(
    image_path: Path,
    *,
    max_bytes: int = _MAX_IMAGE_BYTES,
    max_dimension: int = DEFAULT_MAX_IMAGE_DIMENSION,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> ImageContent | None:
    """
    Baca file gambar dan kembalikan sebagai ImageContent (base64) agar dapat
    dikirim LANGSUNG ke model sebagai image content (model bisa melihat gambar).

    Mengoptimalkan gambar dengan menjaga rasio aspek asli dan menargetkan ~700-720 KB per gambar,
    sehingga muat optimal dalam batas buffer 1 MB transport MCP (Claude Desktop).
    """
    import io

    from PIL import Image

    try:
        data = image_path.read_bytes()
    except OSError:
        return None

    if len(data) == 0:
        return None

    mime_type = _mime_for(image_path)

    # Optimasi adaptif via Pillow: pertahankan rasio aspek dan cari kualitas tertinggi yang muat <= max_bytes (720 KB)
    try:
        with Image.open(io.BytesIO(data)) as img:
            # Pastikan background putih jika ada transparansi
            if img.mode in ("RGBA", "LA") or (
                img.mode == "P" and "transparency" in img.info
            ):
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(
                    img, mask=img.split()[-1] if img.mode == "RGBA" else None
                )
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # Jika file gambar asli sudah JPEG/WebP, dimensinya <= max_dimension, dan ukurannya sudah <= max_bytes,
            # pertahankan data aslinya tanpa kompresi ulang untuk menjaga ketajaman 100%.
            orig_max_dim = max(img.width, img.height)
            if (
                len(data) <= max_bytes
                and orig_max_dim <= max_dimension
                and image_path.suffix.lower() in {".jpg", ".jpeg", ".webp"}
            ):
                encoded = base64.b64encode(data).decode("ascii")
                return ImageContent(type="image", data=encoded, mime_type=mime_type)

            # Pertahankan rasio aspek asli, coba resolusi & kualitas tertinggi terlebih dahulu
            dims = [
                min(orig_max_dim, max_dimension),
                min(orig_max_dim, 2200),
                min(orig_max_dim, 1920),
                min(orig_max_dim, 1600),
            ]
            dims = list(dict.fromkeys(dims))

            qualities = [jpeg_quality, 90, 88, 85, 82, 80]
            best_compressed: bytes | None = None

            for dim in dims:
                work_img = img.copy()
                if max(work_img.width, work_img.height) > dim:
                    work_img.thumbnail((dim, dim), Image.Resampling.LANCZOS)

                for q in qualities:
                    buf = io.BytesIO()
                    # subsampling=0 (4:4:4 chroma) menjaga teks sangat tajam tanpa artefak warna
                    work_img.save(
                        buf,
                        format="JPEG",
                        quality=q,
                        subsampling=0,
                        optimize=True,
                    )
                    compressed = buf.getvalue()
                    best_compressed = compressed

                    if len(compressed) <= max_bytes:
                        break

                if best_compressed and len(best_compressed) <= max_bytes:
                    break

            if best_compressed is not None:
                if (
                    len(data) <= max_bytes
                    and len(data) < len(best_compressed)
                    and image_path.suffix.lower() != ".png"
                ):
                    pass
                else:
                    data = best_compressed
                    mime_type = "image/jpeg"
    except Exception as exc:  # noqa: BLE001
        logger.debug("Pillow image optimization skipped for %s: %s", image_path, exc)

    if len(data) > max_bytes:
        logger.warning(
            "Ukuran gambar %s (%d bytes) melebihi batas maksimal 1 MB (%d bytes)",
            image_path,
            len(data),
            max_bytes,
        )
        return None

    encoded = base64.b64encode(data).decode("ascii")
    return ImageContent(type="image", data=encoded, mime_type=mime_type)


def _build_image_result(
    summary: dict[str, Any],
    image_paths: list[Path],
    *,
    max_images: int | None = None,
) -> list[ContentBlock]:
    """
    Susun tool result: TextContent berisi metadata ringkas, diikuti ImageContent
    dari gambar-gambar yang dikirim langsung ke model.
    Menghitung akumulasi ukuran payload base64 agar aman dari batas 32MB stdio buffer.
    """
    target_images = (
        image_paths
        if max_images is None or max_images <= 0
        else image_paths[:max_images]
    )

    image_blocks: list[ImageContent] = []
    total_payload_bytes = 0
    omitted_count = 0

    for img_path in target_images:
        ic = _image_to_content(Path(img_path))
        if ic is not None:
            block_bytes = len(ic.data)
            if total_payload_bytes + block_bytes > MAX_TOTAL_PAYLOAD_BYTES:
                omitted_count = len(target_images) - len(image_blocks)
                logger.warning(
                    "[MCP Payload Guard] Batas payload %d MB tercapai. %d gambar ditangguhkan ke batch berikutnya.",
                    MAX_TOTAL_PAYLOAD_BYTES // (1024 * 1024),
                    omitted_count,
                )
                break
            image_blocks.append(ic)
            total_payload_bytes += block_bytes

    if omitted_count > 0:
        summary["payload_warning"] = (
            f"Batas ukuran buffer transport tercapai ({total_payload_bytes / (1024 * 1024):.1f} MB). "
            f"{len(image_blocks)} dari {len(target_images)} gambar dikirimkan pada batch ini. "
            f"{omitted_count} gambar tersisa dapat diproses pada panggilan/batch berikutnya."
        )

    content: list[ContentBlock] = [
        TextContent(
            type="text", text=json.dumps(summary, indent=2, ensure_ascii=False)
        ),
        *image_blocks,
    ]
    return content


# Inisialisasi Server MCP (Agent Mode)
server = MCPServer(
    name="document-agent",
    description=(
        "Agent Mode Document Tools MCP Server (OCR/VLM/LLM endpoint NONAKTIF): "
        "alat bantu mekanis untuk agent vision — render dokumen (PPTX/PDF) menjadi gambar, "
        "seleksi acak non-duplikat dokumen, penyimpanan hasil ekstraksi Markdown buatan agent "
        "sebagai gold data example, serta pemisahan data tabular ke SQLite dengan double-verification."
    ),
    instructions=(
        "Alur kerja wajib agent saat membuat gold data example (Batch-by-Batch Pipeline):\n"
        f"{MARKDOWN_LINE_BREAK_RULES}\n\n"
        f"{MERMAID_EXTRACTION_RULES}\n\n"
        "1. Panggil 'open_file_dialog' / 'open_folder_dialog' jika user ingin memilih file/folder secara visual lewat pop-up File Explorer Windows, atau gunakan 'scan_document_folders' untuk pemindaian direktori.\n"
        "2. TANYAKAN KE USER folder/file mana yang datanya ingin diproses (user memilih, mis. 'input/ppt/english' atau via file dialog).\n"
        "3. TANYAKAN KE USER berapa banyak data random yang ingin dibuat (tanpa duplikasi).\n"
        "4. Panggil 'select_random_documents' atau 'select_document_batch' dengan pilihan folder/file dan jumlah dari user.\n"
        "5. PROSES DOKUMEN SECARA INKREMENTAL PER-BATCH (5 halaman/slide setiap kali baca & tulis):\n"
        "   a. Untuk PPTX / PDF: Panggil 'render_presentation_slides' (start_slide=1) atau\n"
        "      'convert_pdf_to_images' (start_page=1) — tool mengirim maksimal 5 gambar per batch.\n"
        "   b. Baca gambar tersebut secara visual, tulis Markdown untuk batch halaman/slide tersebut\n"
        "      (sertakan penanda standar: <!-- SLIDE: N --> atau <!-- PAGE: N -->).\n"
        "   c. LANGSUNG SIMPAN batch tersebut ke 'save_extraction_result' (file sumber = dokumen asli,\n"
        "      bukan gambar). Server secara otomatis menggabungkan (merge & append) halaman ke file disk.\n"
        "   d. Jika respon 'has_more=true', ulangi langkah (a-c) dengan start_slide / start_page berikutnya\n"
        "      (mis. start=6, lalu start=11, dst.) hingga seluruh dokumen selesai.\n"
        "   e. Setelah seluruh halaman selesai tersimpan (status=success, is_complete=true), baru pindah\n"
        "      ke dokumen berikutnya dari daftar yang dipilih.\n"
        "6. Gunakan 'preview_markdown_chunks' jika ingin memeriksa simulasi chunking Markdown.\n"
        "7. Jangan pernah memilih folder atau jumlah data sendiri tanpa persetujuan user."
    ),
    version="0.1.0",
)


def _show_open_file_dialog_native(
    title: str = "Pilih Dokumen (PDF, DOCX, Excel, PPTX, Gambar)",
    multiple: bool = True,
    initial_dir: str = ".",
) -> list[str]:
    """Membuka dialog Open File Windows standar (topmost)."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    filetypes = [
        ("Supported Documents", "*.pdf;*.pptx;*.ppt;*.png;*.jpg;*.jpeg;*.webp"),
        ("PDF Documents (*.pdf)", "*.pdf"),
        ("PowerPoint Presentations (*.pptx, *.ppt)", "*.pptx;*.ppt"),
        ("Images (*.png, *.jpg, *.jpeg, *.webp)", "*.png;*.jpg;*.jpeg;*.webp"),
        ("All Files (*.*)", "*.*"),
    ]
    try:
        init_dir = _resolve_path(initial_dir)
        init_dir_str = str(init_dir) if init_dir.exists() else str(PROJECT_ROOT)
        if multiple:
            files = filedialog.askopenfilenames(
                title=title,
                initialdir=init_dir_str,
                filetypes=filetypes,
            )
            return [str(Path(f).resolve()) for f in files]
        else:
            file = filedialog.askopenfilename(
                title=title,
                initialdir=init_dir_str,
                filetypes=filetypes,
            )
            return [str(Path(file).resolve())] if file else []
    finally:
        root.destroy()


def _show_open_folder_dialog_native(
    title: str = "Pilih Folder Dokumen",
    initial_dir: str = ".",
) -> str | None:
    """Membuka dialog Browse Folder Windows standar (topmost)."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        init_dir = _resolve_path(initial_dir)
        init_dir_str = str(init_dir) if init_dir.exists() else str(PROJECT_ROOT)
        folder = filedialog.askdirectory(
            title=title,
            initialdir=init_dir_str,
        )
        return str(Path(folder).resolve()) if folder else None
    finally:
        root.destroy()


def _scan_document_directories(root_dir: str | Path = ".") -> list[dict[str, Any]]:
    """
    Pindai root_dir dan seluruh sub-foldernya untuk mendeteksi folder berisi dokumen.
    """
    root_path = _resolve_path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Direktori tidak ditemukan: {root_path}")

    folder_map: dict[Path, list[Path]] = {}
    for p in root_path.rglob("*"):
        if any(part in _IGNORED_DIR_NAMES for part in p.parts):
            continue
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            folder_map.setdefault(p.parent, []).append(p)

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


@server.tool(
    name="open_file_dialog",
    description=(
        "Buka pop-up jendela File Explorer / File Picker Windows standar agar user dapat "
        "memilih satu atau banyak file dokumen (.pdf, .pptx, .png, dll) secara langsung lewat GUI Windows. "
        "Mengembalikan daftar path file yang dipilih user atau status 'cancelled' jika dibatalkan."
    ),
)
def open_file_dialog(
    title: str = "Pilih Dokumen (PDF, DOCX, Excel, PPTX, Gambar)",
    multiple: bool = True,
    initial_dir: str = ".",
) -> str:
    """
    Buka pop-up dialog pemilihan file standar Windows.
    """
    try:
        selected_files = _show_open_file_dialog_native(
            title=title,
            multiple=multiple,
            initial_dir=initial_dir,
        )
        if not selected_files:
            return json.dumps(
                {
                    "status": "cancelled",
                    "message": "User membatalkan pemilihan file (tidak ada file yang dipilih).",
                    "selected_files": [],
                    "selected_count": 0,
                },
                indent=2,
                ensure_ascii=False,
            )

        return json.dumps(
            {
                "status": "success",
                "message": f"Berhasil memilih {len(selected_files)} file dokumen.",
                "selected_files": selected_files,
                "selected_count": len(selected_files),
            },
            indent=2,
            ensure_ascii=False,
        )
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat membuka dialog file Windows: {e}"


@server.tool(
    name="open_folder_dialog",
    description=(
        "Buka pop-up jendela Folder Browser Windows standar agar user dapat memilih direktori/folder "
        "dokumen secara langsung lewat GUI Windows. "
        "Mengembalikan path folder terpilih dan ringkasan file di dalamnya atau status 'cancelled' jika dibatalkan."
    ),
)
def open_folder_dialog(
    title: str = "Pilih Folder Dokumen",
    initial_dir: str = ".",
) -> str:
    """
    Buka pop-up dialog pemilihan folder standar Windows.
    """
    try:
        selected_folder = _show_open_folder_dialog_native(
            title=title,
            initial_dir=initial_dir,
        )
        if not selected_folder:
            return json.dumps(
                {
                    "status": "cancelled",
                    "message": "User membatalkan pemilihan folder.",
                    "selected_folder": None,
                },
                indent=2,
                ensure_ascii=False,
            )

        folder_path = Path(selected_folder)
        dir_files = sorted(
            f
            for f in folder_path.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        return json.dumps(
            {
                "status": "success",
                "selected_folder": str(folder_path),
                "total_documents": len(dir_files),
                "sample_files": [f.name for f in dir_files[:5]],
            },
            indent=2,
            ensure_ascii=False,
        )
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat membuka dialog folder Windows: {e}"


@server.tool(
    name="scan_document_folders",
    description=(
        "Pindai direktori (misal 'dataset', 'output', 'input', atau path khusus) dan sub-subfoldernya "
        "untuk mendeteksi keberadaan folder yang berisi file dokumen (PDF, DOCX, DOC, Excel, PPTX, PPT, Gambar). "
        "Mengembalikan daftar folder yang tersedia, jumlah file per ekstensi, dan contoh nama file."
    ),
)
def scan_document_folders(
    root_dir: str = ".",
) -> str:
    """
    Pindai root_dir untuk menemukan folder-folder yang berisi file dokumen.
    """
    try:
        found_folders = _scan_document_directories(root_dir)
        return json.dumps(
            {
                "root_scanned": str(_resolve_path(root_dir)),
                "total_folders_found": len(found_folders),
                "folders": found_folders,
            },
            indent=2,
            ensure_ascii=False,
        )
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat memindai folder: {e}"


def _parse_file_indices(indices_str: str | None, total_len: int) -> list[int] | None:
    """Parse string indeks 1-based (mis. '44' atau '1, 5, 10-12') menjadi daftar indeks 0-based."""
    if not indices_str or not indices_str.strip():
        return None
    selected: set[int] = set()
    for part in indices_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            chunks = part.split("-")
            if len(chunks) == 2 and chunks[0].isdigit() and chunks[1].isdigit():
                s, e = int(chunks[0]), int(chunks[1])
                for i in range(min(s, e), max(s, e) + 1):
                    if 1 <= i <= total_len:
                        selected.add(i - 1)
        elif part.isdigit():
            i = int(part)
            if 1 <= i <= total_len:
                selected.add(i - 1)
    return sorted(selected)


@server.tool(
    name="find_documents",
    description=(
        "Cari file dokumen (PPTX, PPT, PDF, atau Gambar) secara cepat berdasarkan kata kunci nama file / nomor urut "
        "(mis. query='cindi', query='44', atau query='laporan'). "
        "Mengembalikan daftar path lengkap file yang cocok dan siap langsung diekstraksi."
    ),
)
def find_documents(
    query: str,
    root_dir: str = ".",
    limit: int = 20,
) -> str:
    """
    Cari file dokumen berdasarkan kata kunci nama file atau nomor.
    """
    import fnmatch

    if not query or not query.strip():
        return "ERROR: 'query' pencarian tidak boleh kosong."

    q = query.strip().lower()
    root_path = _resolve_path(root_dir)
    if not root_path.exists():
        return f"ERROR: Direktori '{root_dir}' tidak ditemukan."

    matches: list[dict[str, Any]] = []
    for p in root_path.rglob("*"):
        if any(part in _IGNORED_DIR_NAMES for part in p.parts):
            continue
        if (
            p.is_file()
            and p.suffix.lower() in SUPPORTED_EXTENSIONS
            and (q in p.name.lower() or fnmatch.fnmatch(p.name.lower(), f"*{q}*"))
        ):
            try:
                rel = str(p.relative_to(PROJECT_ROOT))
            except ValueError:
                rel = str(p)
            matches.append(
                {
                    "file_name": p.name,
                    "file_path": str(p),
                    "relative_path": rel,
                    "extension": p.suffix.lower(),
                    "size_bytes": p.stat().st_size,
                }
            )
            if len(matches) >= limit:
                break

    res_data = {
        "status": "success" if matches else "not_found",
        "query": query,
        "root_searched": str(root_path),
        "total_matches": len(matches),
        "matches": matches,
        "next_step": (
            f"Untuk langsung mengekstrak dokumen pertama: panggil start_document_extraction(document_path='{matches[0]['file_path']}')"
            if matches
            else "Tidak ada dokumen yang cocok. Coba kata kunci lain atau scan_document_folders."
        ),
    }
    return json.dumps(res_data, indent=2, ensure_ascii=False)


@server.tool(
    name="select_document_batch",
    description=(
        "Pilih file dokumen spesifik atau daftar file dari direktori input. "
        "Mendukung 4 cara pemilihan fleksibel: "
        "1) Path file langsung (mis. sources='input/ppt/indonesian/44_PPT UJIAN SKIRPSI CINDI FATIKASARI.pptx'), "
        "2) Path folder dengan filter nomor indeks (mis. sources='input/ppt/indonesian', file_indices='44' atau '1, 5, 10-12'), "
        "3) Path folder dengan pola nama file (mis. sources='input/ppt/indonesian', pattern='*cindi*'), "
        "4) Path folder dengan kuota limit (mis. sources='input/ppt/indonesian', limit=1)."
    ),
)
def select_document_batch(
    sources: str | None = None,
    folders: str | None = None,
    limit: int | None = None,
    limit_per_folder: int | None = None,
    file_indices: str | None = None,
    pattern: str | None = None,
) -> str:
    """
    Pilih daftar file dokumen dari folder-folder atau file-file terpilih dengan kuota, indeks, atau filter pola.

    Args:
        sources: Path file atau folder atau daftar path dipisah koma (mis. 'input/ppt/indonesian' atau 'dataset/test.pdf').
        folders: Alias untuk parameter sources (backward compatibility).
        limit: Batas total file yang dipilih.
        limit_per_folder: Batas file per folder.
        file_indices: Nomor urut file 1-based yang ingin dipilih (mis. '44' atau '1, 5, 10-12').
        pattern: Pola pencocokan nama file glob (mis. '*44*' atau '*skripsi*').
    """
    import fnmatch

    raw_input = (
        sources if (sources is not None and sources.strip()) else (folders or "")
    )
    if not raw_input.strip():
        return "ERROR: 'sources' (atau 'folders') wajib diisi dengan path file atau folder yang valid."

    if isinstance(raw_input, str):
        path_strings = [f.strip() for f in raw_input.split(",") if f.strip()]
    else:
        path_strings = [str(f).strip() for f in raw_input]

    per_source: list[dict[str, Any]] = []
    selected: list[str] = []

    for item_str in path_strings:
        p = _resolve_path(item_str)
        if not p.exists():
            # Fleksibel: cari apakah item_str adalah nama subfolder unik di dalam workspace
            sub_matches = [
                d
                for d in PROJECT_ROOT.rglob(item_str)
                if d.is_dir()
                and not any(part in _IGNORED_DIR_NAMES for part in d.parts)
            ]
            if sub_matches:
                p = sub_matches[0]

        if p.is_file():
            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                per_source.append(
                    {
                        "source": str(p),
                        "type": "file",
                        "valid": True,
                        "selected_count": 1,
                        "files": [str(p)],
                    }
                )
                selected.append(str(p))
            else:
                per_source.append(
                    {
                        "source": item_str,
                        "type": "file",
                        "valid": False,
                        "error": f"Ekstensi '{p.suffix}' tidak didukung",
                        "selected_count": 0,
                        "files": [],
                    }
                )
            continue

        if p.is_dir():
            dir_files = sorted(
                f
                for f in p.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            )

            # Filter berdasarkan pattern jika diberikan
            if pattern and pattern.strip():
                pat = pattern.strip().lower()
                dir_files = [
                    f
                    for f in dir_files
                    if fnmatch.fnmatch(f.name.lower(), pat) or pat in f.name.lower()
                ]

            # Filter berdasarkan file_indices jika diberikan (1-based)
            parsed_indices = _parse_file_indices(file_indices, len(dir_files))
            if parsed_indices is not None:
                dir_files = [dir_files[i] for i in parsed_indices]

            if limit_per_folder is not None and limit_per_folder > 0:
                dir_files = dir_files[:limit_per_folder]

            file_paths = [str(f) for f in dir_files]
            per_source.append(
                {
                    "source": str(p),
                    "type": "folder",
                    "valid": True,
                    "selected_count": len(file_paths),
                    "files": file_paths,
                }
            )
            selected.extend(file_paths)
            continue

        per_source.append(
            {
                "source": item_str,
                "type": "unknown",
                "valid": False,
                "error": "Path tidak ditemukan",
                "selected_count": 0,
                "files": [],
            }
        )

    if limit is not None and limit > 0:
        selected = selected[:limit]
        selected_set = set(selected)
        for ps in per_source:
            if ps.get("valid"):
                ps["files"] = [f for f in ps.get("files", []) if f in selected_set]
                ps["selected_count"] = len(ps["files"])

    next_step = (
        f"Gunakan dokumen terpilih dengan memanggil start_document_extraction(document_path='{selected[0]}')"
        if selected
        else "Tidak ada file terpilih. Periksa kembali filter atau path input."
    )

    return json.dumps(
        {
            "total_selected_files": len(selected),
            "selected_files": selected,
            "per_source": per_source,
            "next_step": next_step,
        },
        indent=2,
        ensure_ascii=False,
    )


@server.tool(
    name="select_random_documents",
    description=(
        "Pilih 'count' file dokumen SECARA ACAK TANPA DUPLIKASI dari path-path terpilih oleh user "
        "(bisa berupa folder, file langsung, atau campuran folder & file yang dipisahkan koma). "
        "Secara default mengecualikan dokumen yang sudah memiliki hasil gold data di output_dir agar tidak "
        "diproses ganda (anti-duplikasi antar-run). Panggil tool ini SETELAH user memilih folder/file dan "
        "menyebutkan jumlah data random yang diinginkan."
    ),
)
def select_random_documents(
    sources: str | None = None,
    folders: str | None = None,
    count: int = 1,
    seed: int | None = None,
    exclude_already_extracted: bool = True,
    output_dir: str = "output/agent_gold",
) -> str:
    """
    Pilih dokumen secara acak tanpa duplikasi dari folder atau file terpilih.
    """
    if count is None or count <= 0:
        return "ERROR: 'count' harus berupa bilangan bulat positif (jumlah data random yang diminta user)."

    raw_input = (
        sources if (sources is not None and sources.strip()) else (folders or "")
    )
    if not raw_input.strip():
        return "ERROR: 'sources' (atau 'folders') wajib diisi dengan path folder/file."

    if isinstance(raw_input, str):
        path_strings = [f.strip() for f in raw_input.split(",") if f.strip()]
    else:
        path_strings = [str(f).strip() for f in raw_input]

    resolved_out_dir = _resolve_path(output_dir)

    # Kumpulkan kandidat file unik (anti-duplikasi dalam seleksi)
    seen: set[Path] = set()
    candidates: list[Path] = []
    per_source: list[dict[str, Any]] = []

    for item_str in path_strings:
        p = _resolve_path(item_str)
        if p.is_file():
            if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
                per_source.append(
                    {
                        "source": item_str,
                        "type": "file",
                        "valid": False,
                        "error": f"Ekstensi '{p.suffix}' tidak didukung",
                        "available": 0,
                        "already_extracted": 0,
                        "selected": 0,
                    }
                )
                continue

            already_extracted = 0
            if p.resolve() in seen:
                continue
            if exclude_already_extracted and _has_existing_gold(p, resolved_out_dir):
                already_extracted = 1
            else:
                seen.add(p.resolve())
                candidates.append(p)

            per_source.append(
                {
                    "source": str(p),
                    "type": "file",
                    "valid": True,
                    "available": 1,
                    "already_extracted": already_extracted,
                    "selected": 0,
                }
            )
            continue

        if p.is_dir():
            dir_files = sorted(
                f
                for f in p.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            )

            already_extracted = 0
            for f in dir_files:
                if f.resolve() in seen:
                    continue
                if exclude_already_extracted and _has_existing_gold(
                    f, resolved_out_dir
                ):
                    already_extracted += 1
                    continue
                seen.add(f.resolve())
                candidates.append(f)

            per_source.append(
                {
                    "source": str(p),
                    "type": "folder",
                    "valid": True,
                    "available": len(dir_files),
                    "already_extracted": already_extracted,
                    "selected": 0,
                }
            )
            continue

        per_source.append(
            {
                "source": item_str,
                "type": "unknown",
                "valid": False,
                "error": "Path tidak ditemukan",
                "available": 0,
                "already_extracted": 0,
                "selected": 0,
            }
        )

    valid_sources = [ps for ps in per_source if ps["valid"]]
    if not valid_sources:
        return json.dumps(
            {
                "status": "error",
                "message": f"Tidak ada folder/file valid yang ditemukan dari input: {folders}",
                "selected_files": [],
                "selected_count": 0,
            },
            indent=2,
            ensure_ascii=False,
        )

    if not candidates:
        return json.dumps(
            {
                "status": "warning",
                "message": (
                    "Tidak ada kandidat dokumen tersisa: semua file sudah memiliki gold data "
                    "di output_dir atau folder kosong. Gunakan folder/file lain atau set "
                    "exclude_already_extracted=false untuk memproses ulang."
                ),
                "selected_files": [],
                "selected_count": 0,
                "per_source": per_source,
            },
            indent=2,
            ensure_ascii=False,
        )

    # Seleksi acak tanpa duplikasi
    rng = random.Random(seed)
    actual_count = min(count, len(candidates))
    picked = rng.sample(candidates, actual_count)
    picked_set = {p.resolve() for p in picked}

    # Distribusikan statistik terpilih per source
    for ps in per_source:
        if not ps["valid"]:
            continue
        src_path = Path(ps["source"])
        if ps["type"] == "file":
            ps["selected"] = 1 if src_path.resolve() in picked_set else 0
        else:
            ps["selected"] = sum(1 for f in picked_set if f.parent == src_path)

    return json.dumps(
        {
            "status": "success",
            "requested_count": count,
            "selected_count": actual_count,
            "excluded_already_extracted": (
                f"{sum(ps.get('already_extracted', 0) for ps in valid_sources)} dokumen dilewati "
                "karena sudah memiliki gold data"
                if exclude_already_extracted
                else 0
            ),
            "note": (
                f"Hanya {actual_count} dari {count} dokumen yang diminta tersedia."
                if actual_count < count
                else ""
            ),
            "seed_used": seed,
            "selected_files": [str(p) for p in picked],
            "per_source": per_source,
            "next_step": (
                "PROSES SATU DOKUMEN PER SATU HINGGA SELESAI SEMUA BATCH-NYA sebelum pindah ke "
                "dokumen berikutnya. Untuk PPTX: 'render_presentation_slides' (start_slide=1) ulangi "
                "dengan next_start_slide hingga has_more=false, kumulatifkan Markdown-nya. Untuk PDF: "
                "'convert_pdf_to_images' (start_page=1) ulangi dengan next_start_page hingga "
                "has_more=false. Baca gambar secara visual, tulis Markdown-nya, lalu simpan dengan "
                "'save_extraction_result' (file sumber = dokumen asli, bukan gambar)."
            ),
        },
        indent=2,
        ensure_ascii=False,
    )


def _has_existing_gold(doc_path: Path, output_dir: str | Path) -> bool:
    """Cek apakah dokumen sudah memiliki hasil gold data (untuk anti-duplikasi antar-run)."""
    out_base = _resolve_path(output_dir)
    return (out_base / f"{doc_path.stem}.md").exists()


@server.tool(
    name="start_document_extraction",
    description=(
        "Mulai ekstraksi dokumen (PPTX, PDF, atau Scan Gambar) secara bertahap per-halaman/slide "
        "menggunakan LangGraph StateGraph. Merender Halaman 1 pada resolusi ultra-tinggi (2400px, quality 92) "
        "mendekati batas 1 MB, lalu MENGIRIM gambar tersebut ke model beserta snapshot skema tabel SQLite eksisting. "
        "Gunakan 'start_page' (default 1) jika ingin memulai dari halaman tertentu."
    ),
)
def start_document_extraction(
    document_path: str,
    start_page: int = 1,
    dpi: int = 150,
    output_dir: str | None = None,
) -> list[ContentBlock] | str:
    """
    Inisialisasi ekstraksi dokumen per-halaman berbasis LangGraph StateGraph.
    """
    from .agent_graph import get_agent_document_graph

    graph = get_agent_document_graph()
    initial_state = {
        "source_path": document_path,
        "batch_start": start_page,
        "batch_size": 1,
        "dpi": dpi,
        "output_dir": output_dir or "",
    }
    result_state = graph.render_graph.invoke(initial_state)

    if result_state.get("status") == "error":
        return f"ERROR: {result_state.get('error', 'Gagal memulai ekstraksi dokumen')}"

    rendered_images = result_state.get("rendered_images", [])
    if not rendered_images:
        return "ERROR: Tidak ada gambar yang berhasil dirender dari dokumen ini."

    summary = {
        "status": "in_progress",
        "document_path": result_state["resolved_path"],
        "doc_type": result_state["doc_type"],
        "current_page": result_state["current_page"],
        "total_pages": result_state["total_items"],
        "has_more": result_state["has_more"],
        "next_start_page": result_state["next_start"],
        "active_database_tables": result_state.get("active_tables_summary", []),
        "subagent_task_directives": result_state.get("subagent_task_directives", {}),
        "instruction": (
            f"Halaman/Slide {result_state['current_page']} dari {result_state['total_items']} siap dibaca. "
            "Pilih dan jalankan peran subagent spesialis yang relevan (Mermaid/Tabular/Scientific/Slide) "
            "sesuai direktif 'subagent_task_directives', lalu kirim hasil Markdown via 'submit_page_and_get_next'."
        ),
    }
    return _build_image_result(
        summary, [Path(p) for p in rendered_images], max_images=1
    )


@server.tool(
    name="submit_page_and_get_next",
    description=(
        "Simpan hasil ekstraksi Markdown Halaman/Slide N ke disk via LangGraph (termasuk auto-ingest/append "
        "tabel transaksional ke SQLite dengan smart schema matching & verifikasi saldo aritmatika), "
        "dan LANGSUNG MERENDER serta MENGEMBALIKAN gambar Halaman N+1 dalam satu respons (memangkas round-trip)."
    ),
)
def submit_page_and_get_next(
    source_file: str,
    page_number: int,
    markdown: str,
    specs: str = "plain",
    output_dir: str = "output/agent_gold",
    dpi: int = 150,
) -> list[ContentBlock] | str:
    """
    Simpan Markdown halaman N dan langsung muat halaman N+1 berbasis LangGraph StateGraph.
    """
    if not markdown or not markdown.strip():
        return "ERROR: Konten Markdown halaman kosong; tidak ada yang disimpan."

    from .agent_graph import get_agent_document_graph

    graph = get_agent_document_graph()
    initial_state = {
        "source_path": source_file,
        "current_page": page_number,
        "incoming_markdown": markdown,
        "specs": specs,
        "output_dir": output_dir,
        "dpi": dpi,
        # Endpoint ini merepresentasikan alur halaman berurutan. Mengirim ulang
        # halaman 1 berarti memulai ekstraksi baru; jalur save batch generik
        # tetap mempertahankan chunk lama kecuali reset_existing diminta.
        "reset_existing": page_number == 1,
    }
    result_state = graph.advance_graph.invoke(initial_state)

    if result_state.get("status") == "error":
        return f"ERROR: {result_state.get('error', 'Gagal memproses halaman')}"

    is_complete = result_state.get("is_complete", False)
    rendered_images = result_state.get("rendered_images", [])

    if is_complete or not rendered_images:
        response_data = {
            "status": "success",
            "is_complete": True,
            "message": "Seluruh halaman dokumen telah selesai diekstraksi dan disimpan secara lengkap ke disk.",
            "source_file": result_state["resolved_path"],
            "markdown_path": str(
                _resolve_path(output_dir)
                / f"{Path(result_state['resolved_path']).stem}.md"
            ),
            "char_count": len(result_state.get("merged_markdown", "")),
            "total_pages": result_state["total_items"],
            "pages_saved": result_state.get("saved_pages", []),
            "active_database_tables": result_state.get("active_tables_summary", []),
            "tabular_database_tables": result_state.get("tabular_tables", []),
        }
        return json.dumps(response_data, indent=2, ensure_ascii=False)

    next_page_num = result_state.get("current_page", page_number + 1)
    summary = {
        "status": "in_progress",
        "is_complete": False,
        "page_saved": page_number,
        "current_page": next_page_num,
        "total_pages": result_state["total_items"],
        "pages_saved_so_far": result_state.get("saved_pages", []),
        "missing_pages": result_state.get("missing_pages", []),
        "active_database_tables": result_state.get("active_tables_summary", []),
        "subagent_task_directives": result_state.get("subagent_task_directives", {}),
        "instruction": (
            f"Halaman {page_number} berhasil disimpan. Sekarang Halaman {next_page_num} dari {result_state['total_items']} "
            "telah siap dibaca pada gambar terlampir. Terapkan aturan peran subagent yang sesuai (Mermaid/Tabular/Scientific) "
            "sesuai 'subagent_task_directives' lalu kirim Markdown via 'submit_page_and_get_next'."
        ),
    }
    return _build_image_result(
        summary, [Path(p) for p in rendered_images], max_images=1
    )


@server.tool(
    name="process_document_batch",
    description=(
        "Tool utama bertenaga LangGraph StateGraph untuk memproses dokumen (PPTX, PDF, atau Scan Gambar) "
        "per batch 5 slide/halaman. Secara otomatis meresolusi path, mendeteksi tipe file, merender gambar "
        "teroptimasi (< 700KB, tajam 4:4:4), dan MENGIRIM gambar-gambar tersebut LANGSUNG ke model untuk dibaca."
    ),
)
def process_document_batch(
    document_path: str,
    start_item: int = 1,
    batch_size: int = 5,
    dpi: int = 150,
    output_dir: str | None = None,
) -> list[ContentBlock] | str:
    """
    Eksekusi pipeline rendering dokumen batch 5 slide/halaman berbasis LangGraph StateGraph.
    """
    from .agent_graph import get_agent_document_graph

    graph = get_agent_document_graph()
    initial_state = {
        "source_path": document_path,
        "batch_start": start_item,
        "batch_size": batch_size,
        "dpi": dpi,
        "output_dir": output_dir or "",
    }
    result_state = graph.render_graph.invoke(initial_state)

    if result_state.get("status") == "error":
        return f"ERROR: {result_state.get('error', 'Gagal memproses dokumen')}"

    rendered_images = result_state.get("rendered_images", [])
    if not rendered_images:
        return "ERROR: Tidak ada gambar yang berhasil dirender dari dokumen ini."

    summary = {
        "document_path": result_state["resolved_path"],
        "doc_type": result_state["doc_type"],
        "dpi": result_state["dpi"],
        "total_items": result_state["total_items"],
        "batch_start": result_state["batch_start"],
        "batch_end": result_state["batch_end"],
        "batch_size": result_state["batch_size"],
        "rendered_image_paths": rendered_images,
        "images_sent_to_model": len(rendered_images),
        "has_more": result_state["has_more"],
        "next_start": result_state["next_start"],
        "instruction": result_state["instruction"],
    }
    return _build_image_result(
        summary, [Path(p) for p in rendered_images], max_images=None
    )


@server.tool(
    name="render_presentation_slides",
    description=(
        "Render slide file presentasi PowerPoint (.pptx / .ppt) menjadi file gambar beresolusi tinggi "
        "satu file per slide kanvas utuh via LangGraph, lalu KIRIM gambar-gambar tersebut LANGSUNG ke model sebagai "
        "image content teroptimasi (maksimal ~700KB per gambar, kualitas tajam 4:4:4) agar model dapat melihat dan membaca slide. "
        "Default output: output/rendered_slides/<nama_file_tanpa_ekstensi>/. "
        "Tool ini memproses DENGAN BATCH per 5 slide (default): gunakan 'start_slide' (1-based, default 1), "
        "'max_images' (default 5), dan 'dpi' (default 150)."
    ),
)
def render_presentation_slides(
    pptx_path: str,
    output_dir: str | None = None,
    start_slide: int = 1,
    max_images: int | None = None,
    dpi: int = 150,
) -> list[ContentBlock] | str:
    """
    Render slide PPTX ke file gambar per slide berbasis LangGraph dan kirim SATU BATCH gambar ke model.
    """
    from .agent_graph import get_agent_document_graph

    graph = get_agent_document_graph()
    batch_size = (
        DEFAULT_MAX_IMAGES
        if (max_images is None or max_images <= 0)
        else min(max_images, DEFAULT_MAX_IMAGES)
    )
    initial_state = {
        "source_path": pptx_path,
        "batch_start": start_slide,
        "batch_size": batch_size,
        "dpi": dpi,
        "output_dir": output_dir or "",
    }
    result_state = graph.render_graph.invoke(initial_state)

    if result_state.get("status") == "error":
        return f"ERROR: {result_state.get('error', 'Gagal merender slide presentasi')}"

    rendered_images = result_state.get("rendered_images", [])
    summary = {
        "pptx_file": result_state["resolved_path"],
        "dpi": result_state["dpi"],
        "total_slides": result_state["total_items"],
        "batch_start": result_state["batch_start"],
        "batch_end": result_state["batch_end"],
        "batch_size": result_state["batch_size"],
        "rendered_image_paths": rendered_images,
        "images_sent_to_model": len(rendered_images),
        "has_more": result_state["has_more"],
        "next_start_slide": result_state["next_start"],
        "instruction": result_state["instruction"],
    }
    return _build_image_result(
        summary, [Path(p) for p in rendered_images], max_images=None
    )


@server.tool(
    name="convert_pdf_to_images",
    description=(
        "Konversi halaman file PDF menjadi file gambar beresolusi tinggi satu file per halaman via LangGraph, "
        "lalu KIRIM gambar-gambar tersebut LANGSUNG ke model sebagai image content agar model dapat "
        "membaca halaman PDF secara visual. Default output: output/rendered_pages/<nama_file_tanpa_ekstensi>/. "
        "Gunakan 'dpi' untuk mengatur resolusi (default 150), 'start_page' (1-based, default 1) untuk "
        "memulai batch, dan 'max_images' (default 5) untuk jumlah halaman per batch."
    ),
)
def convert_pdf_to_images(
    pdf_path: str,
    output_dir: str | None = None,
    dpi: int = 150,
    start_page: int = 1,
    max_images: int | None = None,
) -> list[ContentBlock] | str:
    """
    Konversi halaman PDF ke file gambar per halaman berbasis LangGraph dan kirim SATU BATCH gambar ke model.
    """
    from .agent_graph import get_agent_document_graph

    graph = get_agent_document_graph()
    batch_size = (
        DEFAULT_MAX_IMAGES
        if (max_images is None or max_images <= 0)
        else min(max_images, DEFAULT_MAX_IMAGES)
    )
    initial_state = {
        "source_path": pdf_path,
        "batch_start": start_page,
        "batch_size": batch_size,
        "dpi": dpi,
        "output_dir": output_dir or "",
    }
    result_state = graph.render_graph.invoke(initial_state)

    if result_state.get("status") == "error":
        return f"ERROR: {result_state.get('error', 'Gagal mengonversi PDF ke gambar')}"

    rendered_images = result_state.get("rendered_images", [])
    summary = {
        "pdf_file": result_state["resolved_path"],
        "dpi": result_state["dpi"],
        "total_pages": result_state["total_items"],
        "batch_start": result_state["batch_start"],
        "batch_end": result_state["batch_end"],
        "batch_size": result_state["batch_size"],
        "rendered_image_paths": rendered_images,
        "images_sent_to_model": len(rendered_images),
        "has_more": result_state["has_more"],
        "next_start_page": result_state["next_start"],
        "instruction": result_state["instruction"],
    }
    return _build_image_result(
        summary, [Path(p) for p in rendered_images], max_images=None
    )


@server.tool(
    name="extract_pdf_markdown_mupdf",
    description=(
        "Ekstrak teks dan tabel dokumen PDF digital langsung menjadi Markdown menggunakan PyMuPDF4LLM "
        "(C++ backend sangat cepat, zero-token). Berguna untuk mengambil grounding teks mentah per-halaman, "
        "tabel GFM, dan urutan baca 2-kolom sebagai referensi bantu bagi agent. "
        "Gunakan 'max_pages' (misal 5) atau 'pages' (misal [0, 1, 2]) untuk membatasi rentang halaman pada PDF tebal."
    ),
)
def extract_pdf_markdown_mupdf(
    pdf_path: str,
    page_chunks: bool = True,
    max_pages: int | None = 5,
    pages: list[int] | None = None,
) -> str:
    """
    Ekstrak konten teks & tabel PDF digital langsung ke format Markdown menggunakan PyMuPDF4LLM.
    """
    path_obj = _resolve_path(pdf_path)
    if not path_obj.exists():
        return f"ERROR: File PDF tidak ditemukan: {path_obj}"
    if path_obj.suffix.lower() != ".pdf":
        return f"ERROR: File bukan PDF: {path_obj}"

    try:
        import importlib

        import app.pdf

        importlib.reload(app.pdf)

        target_pages = pages
        if target_pages is None and max_pages is not None and max_pages > 0:
            target_pages = list(range(max_pages))

        res = app.pdf.extract_pdf_with_pymupdf4llm(
            path_obj,
            page_chunks=page_chunks,
            pages=target_pages,
        )
        if isinstance(res, list):
            return json.dumps(
                {
                    "status": "success",
                    "pdf_file": str(path_obj),
                    "total_pages": len(res),
                    "pages": res,
                },
                indent=2,
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "status": "success",
                "pdf_file": str(path_obj),
                "markdown": res,
            },
            indent=2,
            ensure_ascii=False,
        )
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat mengekstrak PDF dengan PyMuPDF4LLM: {e}"


@server.tool(
    name="preprocess_image",
    description=(
        "Pra-pemrosesan mekanis file gambar dokumen sebelum dibaca oleh model: perbaikan orientasi "
        "berdasarkan metadata EXIF dan peningkatan kontras untuk teks pudar / scan gelap. "
        "Hasil proses (gambar yang sudah diperbaiki) dikirim LANGSUNG ke model sebagai image content "
        "agar model dapat melihat versi gambar yang sudah bersih."
    ),
)
def preprocess_image_tool(
    image_path: str,
    contrast_factor: float = 1.2,
    output_dir: str | None = None,
) -> list[ContentBlock] | str:
    """
    Pra-pemrosesan gambar dokumen dan kirim gambar hasil ke model.
    """
    path_obj = _resolve_path(image_path)
    if not path_obj.exists():
        return f"ERROR: File gambar tidak ditemukan: {path_obj}"

    resolved_out = _resolve_path(output_dir) if output_dir else None

    try:
        proc = preprocess_image(
            path_obj,
            auto_orient=True,
            enhance_contrast=True,
            contrast_factor=contrast_factor,
            output_dir=resolved_out,
        )
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat memproses gambar: {e}"

    summary = {
        "original_path": proc.original_path,
        "processed_path": proc.processed_path,
        "is_modified": proc.is_modified,
        "dimensions": {"width": proc.dimensions[0], "height": proc.dimensions[1]},
    }
    return _build_image_result(summary, [Path(proc.processed_path)], max_images=1)


@server.tool(
    name="preview_markdown_chunks",
    description=(
        "[STAGING / BLUEPRINT] Simulasikan pemecahan dokumen Markdown dengan splitter berbasis header (#, ##, ###) "
        "dan recursive character text splitter untuk blueprint sistem RAG masa depan."
    ),
)
def preview_markdown_chunks(
    markdown_text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> str:
    """
    Simulasikan chunking pada teks Markdown.
    """
    try:
        preview = sim_preview_chunks(
            markdown_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return json.dumps(
            asdict(preview),
            indent=2,
            ensure_ascii=False,
        )
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat simulasi chunking: {e}"


@server.tool(
    name="read_document_extraction",
    description=(
        "Baca isi dokumen Markdown hasil ekstraksi lengkap dan metadata dari disk server MCP "
        "(mis. document_path='44_PPT UJIAN SKIRPSI CINDI FATIKASARI' atau path file .md/.pptx/.pdf). "
        "Gunakan tool ini setelah ekstraksi selesai untuk melihat teks lengkap, memvalidasi chunking, "
        "atau memeriksa tabel dan diagram yang tersimpan di disk server."
    ),
)
def read_document_extraction(
    document_path: str,
    output_dir: str = "output/agent_gold",
) -> str:
    """
    Baca kembali file Markdown hasil ekstraksi dari output_dir.
    """
    p = Path(document_path)
    stem = (
        p.stem
        if (p.suffix.lower() in SUPPORTED_EXTENSIONS or p.suffix.lower() == ".md")
        else document_path
    )
    out_base = _resolve_path(output_dir)

    md_file = out_base / f"{stem}.md"
    if not md_file.exists():
        direct_md = _resolve_path(document_path)
        if direct_md.is_file() and direct_md.suffix.lower() == ".md":
            md_file = direct_md
        else:
            return (
                f"ERROR: File Markdown hasil ekstraksi tidak ditemukan di '{md_file}'."
            )

    content = md_file.read_text(encoding="utf-8")
    meta_file = md_file.with_suffix(".meta.json")
    meta_info: dict[str, Any] = {}
    if meta_file.exists():
        try:
            meta_info = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001, S110
            pass

    res_data = {
        "status": "success",
        "markdown_path": str(md_file),
        "char_count": len(content),
        "total_pages": meta_info.get("total_pages_detected", 0),
        "is_complete": meta_info.get("is_complete", True),
        "markdown_content": content,
        "metadata": meta_info,
    }
    return json.dumps(res_data, indent=2, ensure_ascii=False)


# --- New Tabular SQLite Tools in Agent Mode ---


@server.tool(
    name="classify_and_ingest_tables_to_sqlite",
    description=(
        "Deteksi dan ekstrak tabel-tabel data di dalam Markdown dokumen. Jika tabel bertipe transaksional "
        "(rekening koran, log mutasi keuangan, daftar tagihan, ledger), sistem otomatis membuatkan tabel SQLite "
        "dan melakukan double-verification integritas data."
    ),
)
def classify_and_ingest_tables_to_sqlite(
    markdown_text: str,
    source_file: str = "",
    db_path: str | None = None,
) -> str:
    """
    Ingest tabel transaksional dari Markdown ke database SQLite.
    """
    try:
        resolved_db = _resolve_path(db_path) if db_path else None
        results = extract_and_ingest_tables_from_markdown(
            markdown_text=markdown_text,
            source_file=source_file,
            db_path=resolved_db,
        )
        return json.dumps(
            {
                "status": "success",
                "tables_ingested_count": len(results),
                "results": [r.model_dump() for r in results],
            },
            indent=2,
            ensure_ascii=False,
        )
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat mengingest tabel ke SQLite: {e}"


@server.tool(
    name="query_tabular_database",
    description=(
        "Jalankan query SQL (misal 'SELECT SUM(debit_amount), COUNT(*) FROM ...') pada database SQLite dokumen "
        "untuk melakukan kalkulasi agregat berpresisi 100% yang andal pada basis data relasional SQLite. "
        + ROW_ROLE_SQL_GUIDANCE
    ),
)
def query_tabular_database(
    sql_query: str,
    db_path: str | None = None,
) -> str:
    """
    Eksekusi query SQL pada database SQLite dokumen.
    """
    try:
        resolved_db = _resolve_path(db_path) if db_path else None
        res = query_sqlite(sql_query, db_path=resolved_db)
        return json.dumps(res.model_dump(), indent=2, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat menjalankan query SQL: {e}"


@server.tool(
    name="inspect_tabular_database",
    description="Periksa daftar tabel, skema kolom, jumlah baris, dan contoh record pada database SQLite dokumen.",
)
def inspect_tabular_database(
    db_path: str | None = None,
) -> str:
    """
    Inspeksi skema dan status tabel pada database SQLite.
    """
    try:
        resolved_db = _resolve_path(db_path) if db_path else None
        db_mgr = TabularDatabaseManager(resolved_db)
        info = db_mgr.inspect_database()
        return json.dumps(info, indent=2, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        return f"ERROR saat menginspeksi database: {e}"


@server.tool(
    name="save_extraction_result",
    description=(
        "Simpan hasil ekstraksi Markdown yang telah ditulis sendiri oleh agent (hasil membaca gambar dokumen "
        "secara visual) sebagai gold data example via LangGraph StateGraph. Mendukung penyimpanan "
        "inkremental per batch 5 halaman/slide (otomatis di-merge dan di-append ke file <nama_dokumen>.md). "
        "Menyimpan sidecar <nama_dokumen>.meta.json serta mengekstrak data tabular ke SQLite ketika seluruh dokumen selesai."
    ),
)
def save_extraction_result(
    source_file: str,
    markdown: str,
    specs: str = "plain",
    output_dir: str = "output/agent_gold",
    ingest_transactional_tables: bool = True,
    current_page: int = 1,
    reset_existing: bool = False,
) -> str:
    """
    Simpan Markdown hasil ekstraksi agent beserta metadata gold data, struktur halaman, dan ingesti SQLite jika ada tabel transaksional.
    Mendukung penyimpanan inkremental per-batch berbasis LangGraph.

    Args:
        source_file: Path file dokumen sumber yang telah dibaca agent.
        markdown: Konten batch/dokumen Markdown hasil ekstraksi yang ditulis agent.
        specs: Spesifikasi layout yang digunakan ('plain', 'markdown_hierarchy', 'bilingual_journal', 'presentation_slides', atau komposit).
        output_dir: Direktori tujuan penyimpanan gold data.
        ingest_transactional_tables: Otomatis ingest tabel transaksional ke database SQLite jika terdeteksi.
        current_page: Nomor halaman batch bila Markdown tidak memuat penanda `<!-- PAGE: N -->`.
        reset_existing: Hapus chunk halaman lama saat memulai ulang ekstraksi penuh dari halaman 1.
    """
    if not markdown or not markdown.strip():
        return "ERROR: Konten Markdown kosong; tidak ada yang disimpan."

    from .agent_graph import get_agent_document_graph

    graph = get_agent_document_graph()
    initial_state = {
        "source_path": source_file,
        "incoming_markdown": markdown,
        "specs": specs,
        "output_dir": output_dir,
        "current_page": current_page,
        "reset_existing": reset_existing,
        "ingest_transactional_tables": ingest_transactional_tables,
    }
    result_state = graph.save_graph.invoke(initial_state)

    if result_state.get("status") == "error":
        return f"ERROR: {result_state.get('error', 'Gagal menyimpan hasil ekstraksi')}"

    response_data = {
        "status": result_state["status"],
        "is_complete": result_state["is_complete"],
        "message": result_state["instruction"],
        "source_file": result_state["resolved_path"],
        "source_extension": Path(result_state["resolved_path"]).suffix.lower(),
        "specs": [s.strip() for s in specs.split(",") if s.strip()],
        "extracted_by": "agent",
        "ocr_used": False,
        "saved_at": datetime.now(UTC).isoformat(),
        "markdown_path": str(
            _resolve_path(output_dir) / f"{Path(result_state['resolved_path']).stem}.md"
        ),
        "char_count": len(result_state.get("merged_markdown", "")),
        "total_pages_detected": len(result_state.get("page_structure", [])),
        "total_pages_in_source": result_state["total_items"],
        "pages_saved": result_state.get("saved_pages", []),
        "missing_pages": result_state.get("missing_pages", []),
        "page_structure": result_state.get("page_structure", []),
        "tabular_database_tables": result_state.get("tabular_tables", []),
    }
    return json.dumps(response_data, indent=2, ensure_ascii=False)


def main() -> None:
    """Entry point untuk menjalankan MCP Server Agent Mode via stdio transport."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
