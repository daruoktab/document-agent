"""
Modul Pemrosesan Dokumen Presentasi PowerPoint (.pptx / .ppt).

Menyediakan fungsi untuk:
  - Render slide presentasi menjadi gambar (DPI tinggi) menggunakan LibreOffice/soffice CLI atau pptx2pdf + PyMuPDF.
  - Membaca teks asli, shape, diagram, dan tabel per slide via python-pptx (opsional).
  - Ekstraksi visual slide menggunakan Vision LLM langsung dari kanvas gambar render.
  - Menjalankan sub-agent SQL tabular mandiri per-slide dan audit guardrail jalur ganda di tahap akhir.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from .config import DEFAULT_DPI
from .llm import encode_image_to_base64
from .multi_page import (
    extract_document_title,
    format_page_delimiter,
    stitch_pages_to_markdown,
    strip_page_markers,
)
from .prompts import get_vision_system_prompt
from .tabular_db import (
    cross_verify_dual_track,
    process_page_tabular_agent,
    prune_document_pages,
)

logger = logging.getLogger(__name__)


def _find_libreoffice_binary() -> str | None:
    """Temukan binary LibreOffice / soffice di sistem Windows atau Linux."""
    # 1. Path eksplisit dari konfigurasi pengguna.
    configured = os.environ.get("LIBREOFFICE_BIN")
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.is_file():
            return str(configured_path)
        logger.warning("LIBREOFFICE_BIN tidak menunjuk ke executable: %s", configured)

    # 2. Cek di PATH
    for name in ("soffice", "libreoffice", "soffice.exe", "libreoffice.exe"):
        found = shutil.which(name)
        if found:
            return found

    # 3. Lokasi standar Windows
    if platform.system() == "Windows":
        candidates = [
            Path(r"C:\Program Files\LibreOffice\program\soffice.exe"),
            Path(r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"),
            Path(os.environ.get("PROGRAMFILES", "C:\\Program Files"))
            / "LibreOffice"
            / "program"
            / "soffice.exe",
            Path(os.environ.get("LOCALAPPDATA", ""))
            / "Programs"
            / "LibreOffice"
            / "program"
            / "soffice.exe",
        ]
        for c in candidates:
            if c.exists():
                return str(c)

    # 4. Lokasi standar Linux / macOS
    for loc in (
        "/usr/bin/soffice",
        "/usr/bin/libreoffice",
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    ):
        if Path(loc).exists():
            return loc

    return None


def count_presentation_slides(presentation_path: str | Path) -> int:
    """Hitung jumlah total slide dalam file PowerPoint."""
    path_obj = Path(presentation_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"File presentasi tidak ditemukan: {path_obj}")

    # Cara 1: python-pptx (sangat cepat, tidak perlu render)
    try:
        from pptx import Presentation

        prs = Presentation(str(path_obj))
        return len(prs.slides)
    except Exception:  # noqa: BLE001, S110
        pass

    # Cara 2: Konversi ke PDF via LibreOffice lalu hitung halamannya
    from .pdf import pdf_page_count

    pdf_out = _convert_presentation_to_pdf(path_obj)
    if pdf_out and pdf_out.exists():
        return pdf_page_count(pdf_out)

    return 1


def _convert_presentation_to_pdf(
    presentation_path: Path,
    output_dir: Path | None = None,
) -> Path | None:
    """Konversi file .pptx / .ppt menjadi .pdf menggunakan LibreOffice headless."""
    soffice = _find_libreoffice_binary()
    if not soffice:
        logger.warning(
            "LibreOffice (soffice) tidak ditemukan di sistem. "
            "Rendering visual slide presentasi mungkin terbatas."
        )
        return None

    target_dir = output_dir or Path(tempfile.mkdtemp(prefix="pptx_pdf_"))
    target_dir.mkdir(parents=True, exist_ok=True)

    # Gunakan profile isolated agar bebas dari file lock jika LibreOffice GUI sedang dibuka
    user_profile = (target_dir / "lo_profile").as_uri()
    cmd = [
        soffice,
        f"-env:UserInstallation={user_profile}",
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(target_dir),
        str(presentation_path),
    ]

    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if res.returncode != 0:
            logger.error("LibreOffice konversi PDF gagal: %s", res.stderr)
            return None

        pdf_candidate = target_dir / f"{presentation_path.stem}.pdf"
        if pdf_candidate.exists():
            return pdf_candidate
    except Exception as exc:  # noqa: BLE001
        logger.error("Gagal menjalankan perintah LibreOffice: %s", exc)

    return None


def _render_slides_native_pptx(
    presentation_path: Path,
    output_dir: Path,
    slides: list[int] | None = None,
    dpi: int = 200,
    image_ext: str = "jpg",
) -> list[Path]:
    """Render slide presentasi menggunakan konversi PDF + PyMuPDF."""
    from .pdf import pdf_to_images

    # Buat sub-folder sementara untuk PDF perantara
    temp_pdf_dir = Path(tempfile.mkdtemp(prefix="pptx_render_"))
    pdf_file = _convert_presentation_to_pdf(
        presentation_path, output_dir=temp_pdf_dir
    )

    if not pdf_file or not pdf_file.exists():
        raise RuntimeError(
            f"Gagal mengonversi presentasi '{presentation_path.name}' ke PDF untuk rendering visual."
        )

    images = pdf_to_images(
        pdf_path=pdf_file,
        output_dir=output_dir,
        dpi=dpi,
        image_ext=image_ext,
        pages=slides,
    )

    # Ubah penamaan file agar sesuai konvensi slide: slide_0001.jpg
    renamed: list[Path] = []
    for img in images:
        # Contoh: page_0001.jpg -> slide_0001.jpg
        stem = img.stem
        if stem.startswith("page_"):
            slide_num = stem.replace("page_", "")
            new_name = img.parent / f"slide_{slide_num}.{image_ext}"
            img.rename(new_name)
            renamed.append(new_name)
        else:
            renamed.append(img)

    return renamed


def render_presentation_slides_to_images(
    presentation_path: str | Path,
    output_dir: str | Path | None = None,
    slides: list[int] | None = None,
    dpi: int = 200,
    image_ext: str = "jpg",
) -> list[Path]:
    """
    Render seluruh slide PowerPoint menjadi file gambar beresolusi tinggi (DPI tinggi).

    Args:
        presentation_path: Path ke file presentasi .pptx / .ppt.
        output_dir: Folder output untuk menyimpan gambar slide.
        slides: Daftar indeks slide 0-indexed yang akan dirender (None = semua slide).
        dpi: Resolusi rendering (default 200 DPI).
        image_ext: Format file gambar ("png" atau "jpg").

    Returns:
        Daftar Path file gambar hasil render (berurutan).
    """
    path_obj = Path(presentation_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"File presentasi tidak ditemukan: {path_obj}")

    if output_dir is None:
        out_path = Path("output") / path_obj.stem / "slides"
    else:
        out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    return _render_slides_native_pptx(
        presentation_path=path_obj,
        output_dir=out_path,
        slides=slides,
        dpi=dpi,
        image_ext=image_ext,
    )


def pptx_to_structured_text(
    presentation_path: str | Path,
) -> list[dict[str, Any]]:
    """
    Ekstrak teks, tabel, dan catatan pembicara (speaker notes) dari file PowerPoint secara native.
    """
    path_obj = Path(presentation_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"File presentasi tidak ditemukan: {path_obj}")

    from pptx import Presentation

    prs = Presentation(str(path_obj))
    slides_data: list[dict[str, Any]] = []

    for idx, slide in enumerate(prs.slides, start=1):
        slide_title = ""
        paragraphs: list[str] = []
        tables_extracted: list[list[list[str]]] = []

        # Ekstrak elemen bentuk (shapes)
        for shape in slide.shapes:
            if shape.has_text_frame:
                text_frame = shape.text_frame
                text = text_frame.text.strip()
                if not text:
                    continue

                if shape == slide.shapes.title:
                    slide_title = text
                else:
                    paragraphs.append(text)

            # Ekstrak tabel jika ada di dalam slide
            elif shape.has_table:
                table = shape.table
                table_matrix: list[list[str]] = []
                for row in table.rows:
                    row_vals = [cell.text.strip() for cell in row.cells]
                    table_matrix.append(row_vals)
                if table_matrix:
                    tables_extracted.append(table_matrix)

        # Ekstrak catatan pembicara (notes_slide) jika ada
        notes_text = ""
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
            notes_text = slide.notes_slide.notes_text_frame.text.strip()

        slides_data.append({
            "slide_number": idx,
            "title": slide_title,
            "paragraphs": paragraphs,
            "tables": tables_extracted,
            "notes": notes_text,
        })

    return slides_data


def process_presentation(
    presentation_path: str | Path | None = None,
    output_markdown_path: str | Path | None = None,
    source_name: str | None = None,
    force_all_tables: bool = False,
    db_path: str | Path | None = None,
    *,
    pptx_path: str | Path | None = None,
    **kwargs: Any,
) -> str:
    """
    Pipeline ekstraksi PPT native (sangat cepat, tanpa butuh Vision LLM).
    Mengembalikan string Markdown hasil ekstraksi, dan opsional menyimpan ke file jika output_markdown_path ditentukan.
    """
    path_val = presentation_path or pptx_path
    if path_val is None:
        raise ValueError("presentation_path atau pptx_path harus ditentukan.")
    path_obj = Path(path_val).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"File presentasi tidak ditemukan: {path_obj}")

    src = source_name or path_obj.name
    slides = pptx_to_structured_text(path_obj)

    pages_markdown: list[str] = []
    for s in slides:
        idx = s["slide_number"]
        title = s["title"] or f"Slide {idx}"
        lines = [f"## {title}\n"]

        if s["paragraphs"]:
            for p in s["paragraphs"]:
                lines.append(f"{p}\n")

        if s["tables"]:
            for t in s["tables"]:
                if len(t) >= 1:
                    headers = t[0]
                    lines.append("| " + " | ".join(headers) + " |")
                    lines.append(
                        "| " + " | ".join(["---"] * len(headers)) + " |"
                    )
                    for r in t[1:]:
                        lines.append("| " + " | ".join(r) + " |")
                    lines.append("")

        if s["notes"]:
            lines.append(f"> **Speaker Notes:** {s['notes']}\n")

        pages_markdown.append("\n".join(lines))

    stitched_md = stitch_pages_to_markdown(pages_markdown, source_name=src)

    # Sub-agent SQL per-slide pada teks yang terdeteksi
    if db_path:
        db_out_path = Path(db_path).resolve()
    elif output_markdown_path:
        db_out_path = (
            Path(output_markdown_path).resolve().parent
            / "databases"
            / f"{path_obj.stem}.sqlite"
        )
    else:
        db_out_path = (
            Path("output") / path_obj.stem / "databases" / f"{path_obj.stem}.sqlite"
        )
    db_out_path.parent.mkdir(parents=True, exist_ok=True)

    for page_no, p_md in enumerate(pages_markdown, start=1):
        process_page_tabular_agent(
            page_markdown=p_md,
            page_number=page_no,
            source_file=src,
            db_path=db_out_path,
            table_name_prefix=path_obj.stem,
            force_all_tables=force_all_tables,
        )

    # Dual-track guardrail verification
    prune_document_pages(db_out_path, src, len(slides))
    cross_verify_dual_track(
        stitched_markdown=stitched_md,
        db_path=db_out_path,
        source_file=src,
        total_pages=len(slides),
    )
    if db_out_path and db_out_path.exists():
        try:
            from .tabular_db import TabularDatabaseManager

            csv_dir = (
                db_out_path.parent.parent / "csv"
                if db_out_path.parent.name == "databases"
                else db_out_path.parent / "csv"
            )
            TabularDatabaseManager(db_out_path).export_to_csv(output_dir=csv_dir)
        except Exception as e_csv:  # noqa: BLE001
            logger.warning("[PPT] Gagal mengekspor tabel SQLite ke CSV: %s", e_csv)

    if output_markdown_path:
        out_file = Path(output_markdown_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(stitched_md, encoding="utf-8")
        logger.info("Hasil PPT native berhasil disimpan ke: %s", out_file)

    return stitched_md


def process_presentation_vision(
    presentation_path: str | Path | None = None,
    output_markdown_path: str | Path | None = None,
    vlm_model: BaseChatModel | None = None,
    source_name: str | None = None,
    dpi: int = DEFAULT_DPI,
    image_ext: str = "jpg",
    force_all_tables: bool = False,
    *,
    pptx_path: str | Path | None = None,
    pipeline: Any = None,
    output_dir: str | Path | None = None,
    forced_specs: str | None = None,
    db_path: str | Path | None = None,
    **kwargs: Any,
) -> str:
    """
    Pipeline ekstraksi PPT berbasis Vision VLM.
    Mengembalikan string Markdown hasil ekstraksi, dan opsional menyimpan ke file jika output_markdown_path ditentukan.
    """
    path_val = presentation_path or pptx_path
    if path_val is None:
        raise ValueError("presentation_path atau pptx_path harus ditentukan.")
    path_obj = Path(path_val).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"File presentasi tidak ditemukan: {path_obj}")

    src = source_name or path_obj.name

    target_slides_dir = (
        Path(output_dir)
        if output_dir
        else (Path(output_markdown_path).resolve().parent / "slides")
        if output_markdown_path
        else Path("output") / path_obj.stem / "slides"
    )

    logger.info("[Vision PPT] Memulai rendering slide menjadi gambar...")
    slide_images = render_presentation_slides_to_images(
        path_obj,
        output_dir=target_slides_dir,
        dpi=dpi,
        image_ext=image_ext,
    )
    logger.info(
        "[Vision PPT] Selesai render %d slide gambar", len(slide_images)
    )

    if db_path:
        db_out_path = Path(db_path).resolve()
    elif output_markdown_path:
        db_out_path = (
            Path(output_markdown_path).resolve().parent
            / "databases"
            / f"{path_obj.stem}.sqlite"
        )
    elif output_dir:
        db_out_path = (
            Path(output_dir).resolve() / "databases" / f"{path_obj.stem}.sqlite"
        )
    else:
        db_out_path = (
            Path("output") / path_obj.stem / "databases" / f"{path_obj.stem}.sqlite"
        )
    db_out_path.parent.mkdir(parents=True, exist_ok=True)

    # Dapatkan model VLM atau pipeline
    active_vlm = vlm_model
    if active_vlm is None and pipeline is not None:
        active_vlm = getattr(pipeline, "vlm", None) or getattr(
            getattr(pipeline, "extractor", None), "llm", None
        )

    if active_vlm is None and pipeline is None:
        from .llm import get_vlm

        active_vlm = get_vlm()

    doc_spec = forced_specs or "presentation_slides"
    sys_prompt = get_vision_system_prompt(doc_spec)

    pages_markdown: list[str] = []
    total_visuals = 0
    total_tables = 0

    stream_file: Path | None = None
    presentation_title: str | None = None
    if output_markdown_path:
        stream_file = Path(output_markdown_path).resolve()
        stream_file.parent.mkdir(parents=True, exist_ok=True)
        stream_file.write_text("", encoding="utf-8")
        logger.info("[Vision PPT] Streaming output Markdown ke: %s", stream_file)

    for idx, img_file in enumerate(slide_images, start=1):
        logger.info(
            "[Vision PPT] [Slide %d/%d] Memproses slide '%s'...",
            idx,
            len(slide_images),
            img_file.name,
        )

        res = None
        if pipeline is not None and hasattr(pipeline, "run"):
            res = pipeline.run(
                str(img_file),
                forced_specs=doc_spec,
                is_first_page=(idx == 1),
            )
            slide_md = res.get("markdown_content", "")
            total_visuals += int(res.get("visual_count", 0))
            total_tables += int(res.get("table_count", 0))
            if res.get("visual_count", 0) or res.get("table_count", 0):
                logger.info(
                    "[Vision PPT] [Slide %d/%d] Metadata: %d visual/diagram, %d tabel",
                    idx, len(slide_images),
                    res.get("visual_count", 0), res.get("table_count", 0),
                )
        else:
            if active_vlm is None:
                raise RuntimeError("Model VLM tidak terinisialisasi.")
            b64_data, mime = encode_image_to_base64(img_file)
            msg = HumanMessage(
                content=[
                    {"type": "text", "text": sys_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64_data}"},
                    },
                ]
            )
            response = active_vlm.invoke([msg])
            slide_md = str(response.content)

        from .tabular_db import sanitize_markdown_tables

        slide_md = strip_page_markers(slide_md).strip()
        slide_md = sanitize_markdown_tables(slide_md)

        # Mekanisme Judul Dokumen (hanya diekstrak di slide 1)
        if idx == 1:
            detected_title = (
                (getattr(res, "document_title", None) if res else None)
                or extract_document_title(slide_md, fallback_title=path_obj.stem)
            )
            if detected_title:
                presentation_title = detected_title
                logger.info("[Vision PPT] Judul presentasi teridentifikasi: '%s'", presentation_title)

        # Jalankan Sub-Agent SQL mandiri per slide
        process_page_tabular_agent(
            page_markdown=slide_md,
            page_number=idx,
            source_file=src,
            db_path=db_out_path,
            table_name_prefix=path_obj.stem,
            force_all_tables=force_all_tables,
        )

        pages_markdown.append(slide_md)

        if stream_file:
            delimiter = format_page_delimiter(idx, is_slide=True)
            with open(stream_file, "a", encoding="utf-8") as f:
                f.write(f"\n{delimiter}\n\n{slide_md}\n\n---\n")
                f.flush()

    stitched_md = stitch_pages_to_markdown(
        pages_markdown,
        document_title=presentation_title,
        source_name=src,
        is_slide=True,
    )

    # Jalankan Dual-track Guardrail Cross-Verification
    prune_document_pages(db_out_path, src, len(slide_images))
    cross_verify_dual_track(
        stitched_markdown=stitched_md,
        db_path=db_out_path,
        source_file=src,
        total_pages=len(slide_images),
    )
    if db_out_path and db_out_path.exists():
        try:
            from .tabular_db import TabularDatabaseManager

            csv_dir = (
                db_out_path.parent.parent / "csv"
                if db_out_path.parent.name == "databases"
                else db_out_path.parent / "csv"
            )
            TabularDatabaseManager(db_out_path).export_to_csv(output_dir=csv_dir)
        except Exception as e_csv:  # noqa: BLE001
            logger.warning("[Vision PPT] Gagal mengekspor tabel SQLite ke CSV: %s", e_csv)

    logger.info(
        "[Vision PPT] Selesai: %d slide | %d elemen visual/diagram | %d tabel terdeteksi",
        len(slide_images), total_visuals, total_tables,
    )

    if stream_file:
        stream_file.write_text(stitched_md, encoding="utf-8")
        logger.info("Hasil Vision PPT berhasil disimpan ke: %s", stream_file)

    return stitched_md


# Alias kompatibilitas
process_presentation_native_pipeline = process_presentation
process_presentation_vision_pipeline = process_presentation_vision
convert_presentation_to_pdf = _convert_presentation_to_pdf
