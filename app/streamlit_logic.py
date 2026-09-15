"""
Streamlit Launcher & Workspace Viewer untuk Pipeline Ekstraksi Dokumen Vision VLM,
Sub-Agent SQL Tabular, & Dual-Track Guardrail Cross-Verification.

Fitur Utama:
  - Workspace Document Explorer: Tahan refresh browser (F5) & navigasi riwayat seluruh dokumen tersimpan.
  - Side-by-Side Visual Document Inspector: Kanvas gambar halaman fisik (PDF/PPT) berdampingan dengan Markdown.
  - Interactive Mermaid.js SVG Renderer: Visualisasi diagram alur arsitektur/flowchart interaktif langsung di browser.
  - Dual-Track Guardrail Audit: Laporan rekonsiliasi otomatis teks dokumen vs database SQLite.
  - SQL Tabular Studio & Charting: Penjelajah tabel SQLite, ekspor CSV, dan grafik visual otomatis.
  - Background Job Supervision: Ekstraksi independen dari lifecycle tab browser dengan live log streaming.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol
from zipfile import ZIP_DEFLATED, ZipFile

# Pastikan root direktori proyek berada di sys.path agar impor 'from app....' selalu dikenali
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st


class UploadedFileLike(Protocol):
    name: str

    def getvalue(self) -> bytes: ...

from app.job_tracker import JobManager, is_pid_alive
from app.tabular_db import cross_verify_dual_track
from app.upload_batches import create_batch, list_batches

SUPPORTED_TYPES = ["pdf", "pptx", "ppt", "png", "jpg", "jpeg", "webp"]

SPEC_OPTIONS: dict[str, str | None] = {
    "Pilih otomatis (disarankan)": None,
    "Dokumen biasa, seperti surat atau laporan": "plain",
    "Dokumen dengan bab dan subbab": "markdown_hierarchy",
    "Jurnal dua kolom atau dua bahasa": "bilingual_journal",
    "Slide presentasi": "presentation_slides",
}


# ==============================================================================
# Helper Functions (Dapat Digunakan Kembali & Diuji Secara Independen)
# ==============================================================================


def _save_uploaded_file(uploaded_file: UploadedFileLike, output_dir: Path) -> Path:
    """Simpan file yang diunggah ke direktori output/uploads secara stabil."""
    uploads_dir = output_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    name = Path(uploaded_file.name.replace("\\", "/")).name
    content = uploaded_file.getvalue()
    suffix = Path(name).suffix.lower()
    stem = Path(name).stem

    # Rerun Streamlit tidak membuat salinan baru untuk upload yang sama persis.
    for existing in uploads_dir.glob(f"*{suffix}"):
        if existing.is_file() and existing.read_bytes() == content:
            return existing

    candidate = uploads_dir / f"{stem}{suffix}"
    ordinal = 1
    while candidate.exists():
        candidate = uploads_dir / f"{stem} ({ordinal}){suffix}"
        ordinal += 1
    candidate.write_bytes(content)
    return candidate


def _save_uploaded_files(
    uploaded_files: Sequence[UploadedFileLike],
    output_dir: Path,
) -> list[Path]:
    """Simpan beberapa file upload dan kembalikan path dalam urutan pilihan user."""
    return list(dict.fromkeys(_save_uploaded_file(uploaded_file, output_dir) for uploaded_file in uploaded_files))


def _get_sqlite_db_for_file(file_stem: str, output_dir: Path) -> Path | None:
    """Cari file database SQLite yang terkait dengan file yang diproses."""
    db_candidates = [
        output_dir / file_stem / "databases" / f"{file_stem}.sqlite",
        output_dir / file_stem / f"{file_stem}.sqlite",
        output_dir / "databases" / f"{file_stem}.sqlite",
        output_dir / "databases" / f"{file_stem}_data.sqlite",
        output_dir / "databases" / "documents_data.sqlite",
    ]
    for candidate in db_candidates:
        if candidate.exists():
            return candidate

    doc_db_dir = output_dir / file_stem / "databases"
    if doc_db_dir.exists():
        matches = sorted(doc_db_dir.glob(f"{file_stem}*.sqlite"))
        if matches:
            return matches[0]

    legacy_db_dir = output_dir / "databases"
    if legacy_db_dir.exists():
        stem_matches = sorted(legacy_db_dir.glob(f"{file_stem}*.sqlite"))
        if stem_matches:
            return stem_matches[0]
    return None


def get_document_images(stem: str, output_dir: Path) -> list[Path]:
    """Temukan seluruh gambar halaman PDF atau slide PPT terkait dokumen."""
    doc_dir = output_dir / stem
    # 1. Cek folder pages (PDF)
    pages_dir = doc_dir / "pages"
    if pages_dir.exists():
        imgs = sorted(
            list(pages_dir.glob("*.png")) + list(pages_dir.glob("*.jpg"))
        )
        if imgs:
            return imgs

    # 2. Cek folder slides (PPT)
    slides_dir = doc_dir / "slides"
    if slides_dir.exists():
        imgs = sorted(
            list(slides_dir.glob("*.png")) + list(slides_dir.glob("*.jpg"))
        )
        if imgs:
            return imgs

    # 3. Cek folder uploads jika berupa file gambar tunggal
    uploads_dir = output_dir / "uploads"
    if uploads_dir.exists():
        for ext in ("png", "jpg", "jpeg", "webp"):
            candidate = uploads_dir / f"{stem}.{ext}"
            if candidate.exists():
                return [candidate]

    return []


def split_markdown_by_pages(markdown_text: str) -> dict[int, str]:
    """Bagi teks markdown menjadi per halaman/slide berdasarkan penanda dokumen."""
    pages: dict[int, str] = {}

    # 1. Pola standar multi-halaman: <!-- PAGE: X -->
    matches = list(
        re.finditer(
            r"<!--\s*PAGE:\s*(\d+)\s*-->", markdown_text, re.IGNORECASE
        )
    )
    if matches:
        prefix_text = markdown_text[: matches[0].start()].strip()
        for i, match in enumerate(matches):
            p_num = int(match.group(1))
            start_idx = match.end()
            end_idx = (
                matches[i + 1].start()
                if i + 1 < len(matches)
                else len(markdown_text)
            )
            content = markdown_text[start_idx:end_idx].strip()
            if i == 0 and prefix_text:
                content = f"{prefix_text}\n\n{content}"
            pages[p_num] = content
        return pages

    # 2. Pola presentasi PPT: <!-- SLIDE: X --> atau ## Slide X
    slide_matches = list(
        re.finditer(
            r"(?:<!--\s*SLIDE:\s*(\d+)\s*-->|##\s*Slide\s+(\d+))",
            markdown_text,
            re.IGNORECASE,
        )
    )
    if slide_matches:
        for i, match in enumerate(slide_matches):
            s_num = int(match.group(1) or match.group(2))
            start_idx = match.start()
            end_idx = (
                slide_matches[i + 1].start()
                if i + 1 < len(slide_matches)
                else len(markdown_text)
            )
            pages[s_num] = markdown_text[start_idx:end_idx].strip()
        return pages

    return {1: markdown_text}


def read_completed_markdown_pages(markdown_file: Path) -> dict[int, str]:
    """Baca halaman yang sudah ditulis lengkap oleh proses ekstraksi aktif."""
    try:
        markdown_text = markdown_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}

    pages = split_markdown_by_pages(markdown_text) if markdown_text.strip() else {}
    return {
        page_number: content.removesuffix("---").rstrip()
        for page_number, content in pages.items()
        if content.rstrip().endswith("---")
    }


def extract_mermaid_blocks(text: str) -> list[str]:
    """Cari seluruh blok ```mermaid ... ``` dalam dokumen markdown."""
    pattern = r"```(?:mermaid)\s*\n(.*?)\n```"
    return re.findall(pattern, text, re.DOTALL | re.IGNORECASE)


def find_pages_containing(pages: dict[int, str], query: str) -> list[int]:
    """Kembalikan nomor halaman yang memuat kata pencarian tanpa membedakan kapital."""
    normalized_query = query.strip().casefold()
    if not normalized_query:
        return []
    return [
        page_number
        for page_number, content in pages.items()
        if normalized_query in content.casefold()
    ]


def build_document_zip(stem: str, output_dir: Path) -> bytes:
    """Buat paket ZIP berisi seluruh hasil yang tersedia untuk satu dokumen."""
    doc_dir = output_dir / stem
    archive_buffer = BytesIO()
    archived_paths: set[Path] = set()

    with ZipFile(archive_buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "PETUNJUK.txt",
            "Paket hasil ekstraksi dokumen.\n\n"
            "- File .md berisi teks hasil ekstraksi.\n"
            "- Folder csv dan databases berisi data tabel.\n"
            "- Folder pages atau slides berisi gambar tiap halaman.\n"
            "- Folder logs berisi catatan teknis proses.\n",
        )

        if doc_dir.exists():
            for source_path in sorted(path for path in doc_dir.rglob("*") if path.is_file()):
                archive.write(source_path, source_path.relative_to(doc_dir).as_posix())
                archived_paths.add(source_path.resolve())

        legacy_candidates = [
            output_dir / f"{stem}.md",
            output_dir / "databases" / f"{stem}.sqlite",
            output_dir / "databases" / f"{stem}_data.sqlite",
        ]
        legacy_candidates.extend(get_document_images(stem, output_dir))
        for source_path in legacy_candidates:
            resolved_path = source_path.resolve()
            if not source_path.is_file() or resolved_path in archived_paths:
                continue
            if source_path.parent == output_dir / "uploads":
                archive_name = f"dokumen_asli/{source_path.name}"
            elif source_path.suffix.lower() == ".md":
                archive_name = f"teks/{source_path.name}"
            elif source_path.suffix.lower() == ".sqlite":
                archive_name = f"databases/{source_path.name}"
            else:
                archive_name = f"halaman/{source_path.name}"
            archive.write(source_path, archive_name)
            archived_paths.add(resolved_path)

    return archive_buffer.getvalue()


def build_batch_zip(batch: dict[str, Any], output_dir: Path) -> bytes:
    """One ZIP with independent document folders and a durable batch manifest."""
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("batch.json", json.dumps(batch, ensure_ascii=False, indent=2))
        for doc in batch["documents"]:
            stem = doc["stem"]
            if not stem or stem in {".", ".."} or "/" in stem or "\\" in stem:
                raise ValueError("Identitas dokumen tidak valid")
            with ZipFile(BytesIO(build_document_zip(stem, output_dir))) as document_zip:
                for entry in document_zip.infolist():
                    archive.writestr(f"{stem}/{entry.filename}", document_zip.read(entry))
    return buffer.getvalue()


def render_batch_download(batch: dict[str, Any], output_dir: Path) -> None:
    st.subheader(batch["name"])
    st.caption(f"Dibuat {batch['created_at']} · {len(batch['documents'])} dokumen unik")
    manager = JobManager.get_instance()
    jobs = [manager.get_job(doc["stem"], output_dir=output_dir) for doc in batch["documents"]]
    busy = any(job and job.status in {"queued", "running"} for job in jobs)
    completed = sum(bool(job and job.status == "completed") for job in jobs)
    st.write(f"{completed} dari {len(jobs)} dokumen selesai")
    if busy:
        st.info("ZIP dapat disiapkan setelah semua proses batch berhenti. Klik Segarkan histori untuk memperbarui.")
    elif completed != len(jobs):
        st.warning("Sebagian dokumen belum selesai atau gagal. ZIP hanya berisi hasil yang tersedia; periksa status tiap dokumen.")
    if st.button("Siapkan ZIP seluruh hasil", key=f"prepare_{batch['id']}", disabled=busy):
        with st.spinner("Mengemas hasil batch..."):
            data = build_batch_zip(batch, output_dir)
            directory = output_dir / "batches" / batch["id"]
            temporary = directory / "hasil.zip.tmp"
            temporary.write_bytes(data)
            temporary.replace(directory / "hasil.zip")
    zip_path = output_dir / "batches" / batch["id"] / "hasil.zip"
    if zip_path.exists() and not busy:
        st.caption("ZIP adalah salinan saat terakhir disiapkan. Siapkan ulang setelah menjalankan ulang dokumen.")
        with zip_path.open("rb") as archive:
            st.download_button("Unduh ZIP batch", archive, file_name=f"batch_{batch['id']}.zip",
                               mime="application/zip", key=f"download_{batch['id']}")


def table_csv_bytes(conn: sqlite3.Connection, table: str) -> bytes:
    quoted = '"' + table.replace('"', '""') + '"'
    return pd.read_sql_query(f"SELECT * FROM {quoted}", conn).to_csv(index=False).encode("utf-8-sig")


def build_sqlite_download(db_file: Path) -> bytes:
    """Snapshot konsisten, termasuk transaksi yang telah commit di WAL."""
    source = sqlite3.connect(db_file.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        with TemporaryDirectory(prefix='sqlite_download_') as directory:
            path = Path(directory) / 'snapshot.sqlite'
            snapshot = sqlite3.connect(path)
            try:
                source.backup(snapshot)
                snapshot.execute('PRAGMA journal_mode=DELETE')
            finally:
                snapshot.close()
            return path.read_bytes()
    finally:
        source.close()


def build_all_tables_csv_zip(conn: sqlite3.Connection, tables: list[str]) -> bytes:
    buffer = BytesIO()
    used: set[str] = set()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for table in tables:
            safe_name = re.sub(r'[^\w .-]', '_', table).strip('. ') or 'table'
            name = safe_name + '.csv'
            index = 2
            while name in used:
                name = f'{safe_name}_{index}.csv'
                index += 1
            used.add(name)
            archive.writestr(name, table_csv_bytes(conn, table))
    return buffer.getvalue()


def render_mermaid_html(mermaid_code: str, height: int = 420) -> None:
    """Render diagram Mermaid interaktif dalam format SVG menggunakan Mermaid.js CDN."""
    html_code = f"""
    <!DOCTYPE html>
    <html lang="id">
    <head>
      <meta charset="utf-8">
      <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
      <script>
        mermaid.initialize({{
          startOnLoad: true,
          theme: 'default',
          securityLevel: 'loose'
        }});
      </script>
      <style>
        body {{
          margin: 0;
          padding: 12px;
          background: #f8fafc;
          border-radius: 8px;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          display: flex;
          justify-content: center;
          align-items: center;
        }}
        .mermaid {{
          width: 100%;
          display: flex;
          justify-content: center;
        }}
      </style>
    </head>
    <body>
      <div class="mermaid">
{mermaid_code}
      </div>
    </body>
    </html>
    """
    st.components.v1.html(html_code, height=height, scrolling=True)


# ==============================================================================
# Komponen Monitoring Live Real-Time (Auto-Refresh Fragment)
# ==============================================================================


@st.fragment(run_every=2)
def render_batch_monitor(stems: list[str], output_path: Path) -> None:
    """Tampilkan semua hasil upload batch agar dokumen selain yang aktif tetap terlihat."""
    job_manager = JobManager.get_instance()
    st.markdown("### Dokumen dalam batch upload")
    labels = {
        "queued": "Menunggu antrean",
        "running": "Sedang diproses",
        "completed": "Selesai",
        "failed": "Gagal",
        "canceled": "Dibatalkan",
    }
    for stem in dict.fromkeys(stems):
        job = job_manager.get_job(stem, output_dir=output_path)
        info_col, action_col = st.columns([4, 1])
        with info_col:
            status = labels.get(job.status, job.status) if job else "Belum tersedia"
            st.write(f"{job.file_name if job else stem} — {status}")
            if job and job.status == "failed" and job.error_message:
                st.caption(job.error_message)
        with action_col:
            if st.button("Buka", key=f"batch_open_{stem}", disabled=job is None):
                st.session_state["selected_stem"] = stem
                st.rerun()
    st.caption("Setiap file memiliki hasil Markdown sendiri. Pilih Buka untuk melihat hasil atau progresnya.")


@st.fragment(run_every=2)
def render_live_monitor(stem: str, output_path: Path) -> None:
    """Komponen fragment Streamlit yang memperbarui progres ekstraksi setiap 2 detik."""
    job_manager = JobManager.get_instance()
    job = job_manager.get_job(stem, output_dir=output_path)
    if not job:
        st.info("Pekerjaan tidak ditemukan.")
        return

    # Jika job telah selesai atau gagal, minta Streamlit rerun halaman penuh
    if job.status not in {"queued", "running"}:
        st.rerun()

    # Progress bar & badge
    pct = job.progress_percentage()
    st.markdown(f"### ⏳ Sedang membaca: `{job.file_name or stem}`")
    st.progress(pct / 100.0, text=f"Progres: {pct:.1f}% — {job.stage}")

    # Kartu Metrik Granular
    c1, c2, c3, c4 = st.columns(4)
    if job.total_pages > 0:
        page_str = f"{job.current_page} / {job.total_pages}"
    elif job.current_page > 0:
        page_str = f"Halaman {job.current_page}"
    else:
        page_str = "Menyiapkan..."

    c1.metric("📄 Halaman / Slide", page_str)
    c2.metric("🔄 Tahap saat ini", job.stage)
    c3.metric("📊 Kemajuan", f"{pct:.0f}%")
    is_alive = is_pid_alive(job.pid) if job.pid else True
    c4.metric(
        "🩺 Kondisi",
        (
            "🟡 Menunggu antrean"
            if job.status == "queued"
            else ("Berjalan" if is_alive else "Perlu diperiksa")
        ),
    )

    # Kotak informasi background safety & path file log fisik
    log_file_str = str(job.latest_log_path.resolve()) if job.latest_log_path else "output/logs/..."
    st.info(f"📌 Aktivitas terakhir: {job.last_message or 'Sedang menyiapkan proses...'}")

    completed_pages = read_completed_markdown_pages(job.out_file)
    if completed_pages:
        st.markdown("#### 👀 Preview halaman yang sudah selesai")
        st.caption(
            "Preview ini diperbarui otomatis. Isi akhir dapat berubah setelah pemeriksaan "
            "dan penyatuan seluruh dokumen selesai."
        )
        available_pages = sorted(completed_pages)
        selected_page = st.selectbox(
            "Pilih halaman / slide yang sudah selesai",
            available_pages,
            index=len(available_pages) - 1,
            format_func=lambda page_number: f"Halaman / Slide {page_number}",
            key=f"live_preview_page_{stem}",
        )
        images = get_document_images(stem, output_path)
        preview_text = completed_pages[selected_page]
        if 0 < selected_page <= len(images):
            image_col, text_col = st.columns([1.1, 1], gap="medium")
            with image_col:
                st.image(
                    str(images[selected_page - 1]),
                    caption=f"Dokumen asli · halaman / slide {selected_page}",
                    use_container_width=True,
                )
            with text_col:
                st.markdown(preview_text)
        else:
            st.markdown(preview_text)

    # Tampilan log real-time streaming
    with st.expander("Lihat detail teknis proses"):
        st.caption(f"Lokasi log: `{log_file_str}` · ID proses: `{job.pid or '-'}`")
        recent_log_text = job_manager.get_latest_logs(stem, line_count=35)
        st.code(recent_log_text or "(Belum ada catatan proses)", language="text")

    # Aksi kontrol
    col_a1, col_a2 = st.columns([1, 1])
    with col_a1:
        if st.button("🔄 Segarkan Tampilan Sekarang", use_container_width=True):
            st.rerun()
    with col_a2:
        if (
            st.button(
                "🛑 Batalkan Ekstraksi",
                type="secondary",
                use_container_width=True,
            )
            and job_manager.cancel_job(stem)
        ):
            st.warning("Proses ekstraksi telah dibatalkan.")
            st.rerun()


# ==============================================================================
# Komponen Tampilan Hasil Lengkap (Tabs)
# ==============================================================================


def render_completed_document_view(stem: str, output_path: Path) -> None:
    """Tampilkan antarmuka hasil ekstraksi komprehensif dokumen."""
    job_manager = JobManager.get_instance()
    job = job_manager.get_job(stem, output_dir=output_path)
    md_file = output_path / stem / f"{stem}.md"
    if not md_file.exists():
        md_file = output_path / f"{stem}.md"

    db_file = _get_sqlite_db_for_file(stem, output_path)
    images = get_document_images(stem, output_path)
    md_content = md_file.read_text(encoding="utf-8") if md_file.exists() else ""
    pages_map = split_markdown_by_pages(md_content) if md_content else {}

    # Header Ringkasan Dokumen
    col_top1, col_top2 = st.columns([3.5, 1.5])
    with col_top1:
        st.subheader(f"📄 Hasil dokumen: `{stem}`")
        badge_pages = f"{len(images)} halaman/slide" if images else "Hasil teks"
        st.caption(
            f"Selesai diproses · {badge_pages} · File hasil: `{md_file.name}`"
        )
    with col_top2:
        zip_data = build_document_zip(stem, output_path)
        st.download_button(
            "⬇️ Unduh semua hasil (.zip)",
            data=zip_data,
            file_name=f"{stem}_hasil.zip",
            mime="application/zip",
            use_container_width=True,
        )
        if st.button("🔄 Ekstrak Ulang Dokumen Ini", use_container_width=True):
            job_manager.restart_job(stem, output_dir=output_path)
            st.rerun()

    # Label utama memakai bahasa berbasis tujuan pengguna; istilah teknis ada di detail.
    tab_inspector, tab_guardrail, tab_md, tab_sql, tab_log = st.tabs([
        "🔍 Cocokkan dengan dokumen asli",
        "✅ Periksa kelengkapan data",
        "📝 Baca dan unduh teks",
        "📊 Lihat tabel dan grafik",
        "🔧 Detail proses",
    ])

    # --------------------------------------------------------------------------
    # TAB 1: VISUAL PAGE INSPECTOR (SIDE-BY-SIDE)
    # --------------------------------------------------------------------------
    with tab_inspector:
        st.subheader("🔍 Cocokkan hasil dengan dokumen asli")
        st.caption(
            "Bandingkan tampilan halaman asli di sebelah kiri dengan teks yang terbaca di sebelah kanan."
        )

        if images:
            total_imgs = len(images)
            col_ctl1, col_ctl2 = st.columns([2, 2])
            with col_ctl1:
                if total_imgs == 1:
                    selected_page_idx = 1
                    st.caption("Hanya tersedia 1 halaman / slide.")
                else:
                    selected_page_idx = st.slider(
                        "Pilih Halaman / Slide:",
                        min_value=1,
                        max_value=total_imgs,
                        value=1,
                        format="Halaman %d",
                    )
            with col_ctl2:
                view_mode = st.radio(
                    "Teks yang ditampilkan:",
                    ["Halaman ini", "Seluruh dokumen"],
                    horizontal=True,
                )

            current_img_path = images[selected_page_idx - 1]
            current_page_text = pages_map.get(
                selected_page_idx,
                f"*(Tidak ada penanda teks khusus untuk Halaman {selected_page_idx})*",
            )

            # Layout 2 Kolom Berdampingan
            col_img, col_text = st.columns([1.1, 1], gap="medium")

            with col_img:
                st.markdown(
                    f"##### 🖼️ Dokumen asli · halaman {selected_page_idx} dari {total_imgs}"
                )
                st.image(
                    str(current_img_path),
                    caption=current_img_path.name,
                    use_container_width=True,
                )
                with open(current_img_path, "rb") as f_img:
                    st.download_button(
                        f"⬇️ Unduh Gambar Halaman {selected_page_idx}",
                        data=f_img.read(),
                        file_name=current_img_path.name,
                        mime="image/png",
                    )

            with col_text:
                st.markdown(
                    f"##### ✍️ Teks hasil · {'halaman ' + str(selected_page_idx) if view_mode == 'Halaman ini' else 'seluruh dokumen'}"
                )
                text_to_show = (
                    current_page_text
                    if view_mode == "Halaman ini"
                    else md_content
                )

                # Deteksi jika halaman ini memiliki diagram Mermaid
                page_mermaids = extract_mermaid_blocks(text_to_show)
                if page_mermaids:
                    st.info(
                        f"Terdapat {len(page_mermaids)} diagram pada bagian ini."
                    )
                    with st.expander(
                        "Lihat diagram", expanded=True
                    ):
                        for m_code in page_mermaids:
                            render_mermaid_html(m_code, height=320)

                st.markdown(text_to_show)

            from app.learning_ui import render_correction_form
            render_correction_form(
                stem=stem, page=selected_page_idx, image_path=current_img_path,
                original=pages_map.get(selected_page_idx, ""), images=images,
                source_path=job.input_path if job else None,
            )
        else:
            st.info(
                "ℹ️ Gambar halaman fisik tidak ditemukan untuk dokumen ini (mungkin dokumen diproses tanpa menyimpan kanvas halaman terpisah)."
            )
            st.markdown(md_content)

    # --------------------------------------------------------------------------
    # TAB 2: DUAL-TRACK GUARDRAIL AUDIT
    # --------------------------------------------------------------------------
    with tab_guardrail:
        st.subheader("✅ Pemeriksaan kelengkapan data")
        st.caption(
            "Sistem membandingkan tabel pada teks hasil dengan tabel yang tersimpan sebagai data."
        )
        with st.expander("Bagaimana pemeriksaan ini bekerja?"):
            st.write(
                "Dokumen dibaca menjadi teks dan tabel terstruktur secara terpisah, lalu jumlah "
                "tabel serta barisnya dibandingkan. Pemeriksaan ini membantu menemukan data yang "
                "mungkin terlewat, tetapi tidak menggantikan pengecekan isi oleh pengguna."
            )

        total_pages_detected = len(images) or (
            job.total_pages if job else len(pages_map) or 1
        )
        report = cross_verify_dual_track(
            md_content,
            db_file,
            source_file=stem,
            total_pages=total_pages_detected,
        )

        # Status Banner
        if report.guardrail_status == "PASSED":
            st.success(f"✅ **Data terlihat konsisten** — {report.supervisor_notes}")
        elif report.guardrail_status == "WARNING":
            st.warning(f"⚠️ **Ada bagian yang perlu diperiksa** — {report.supervisor_notes}")
        else:
            st.error(f"❌ **Data belum konsisten** — {report.supervisor_notes}")

        # Metrik Rekonsiliasi
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tabel Markdown", report.total_markdown_tables)
        m2.metric("Tabel SQLite", report.total_sqlite_tables)
        m3.metric("Baris Data Markdown", report.total_markdown_rows)
        m4.metric("Baris Data SQLite", report.total_sqlite_rows)

        if report.discrepancies:
            with st.expander("⚠️ Bagian yang perlu diperiksa", expanded=True):
                for disc in report.discrepancies:
                    st.markdown(f"- {disc}")

        if report.table_comparisons:
            st.markdown("#### 📋 Rincian perbandingan")
            df_comp = pd.DataFrame(report.table_comparisons)
            st.dataframe(df_comp, use_container_width=True)
        else:
            st.info(
                "ℹ️ Tidak ada tabel transaksional yang ditemukan pada dokumen ini (dokumen berupa teks naratif)."
            )

    # --------------------------------------------------------------------------
    # TAB 3: MARKDOWN OUTPUT & MERMAID RENDERER
    # --------------------------------------------------------------------------
    with tab_md:
        st.subheader("📝 Baca dan unduh teks")
        if md_content:
            search_query = st.text_input(
                "Cari kata atau frasa dalam dokumen",
                placeholder="Contoh: nomor kontrak, nama kegiatan, total anggaran",
            )
            matched_pages = find_pages_containing(pages_map, search_query)
            if search_query.strip():
                if matched_pages:
                    st.info(
                        f"Ditemukan pada {len(matched_pages)} halaman/slide: "
                        + ", ".join(str(page) for page in matched_pages)
                    )
                else:
                    st.warning("Kata atau frasa tersebut tidak ditemukan.")
            st.download_button(
                "💾 Unduh File Markdown (.md)",
                data=md_content,
                file_name=f"{stem}.md",
                mime="text/markdown",
            )

            # Deteksi Diagram Mermaid di seluruh dokumen
            doc_mermaids = extract_mermaid_blocks(md_content)
            if doc_mermaids:
                st.markdown(
                    f"### 🎨 Diagram yang ditemukan ({len(doc_mermaids)})"
                )
                for idx, m_code in enumerate(doc_mermaids, 1):
                    with st.expander(
                        f"📊 Diagram {idx}",
                        expanded=True,
                    ):
                        render_mermaid_html(m_code, height=380)
                        with st.expander("Lihat kode diagram"):
                            st.code(m_code, language="mermaid")
                        st.download_button(
                            f"⬇️ Unduh Kode Diagram #{idx} (.mmd)",
                            data=m_code,
                            file_name=f"{stem}_diagram_{idx}.mmd",
                            mime="text/plain",
                            key=f"dl_mmd_{idx}",
                        )

            st.markdown("### 📄 Isi Teks Markdown")
            st.markdown(md_content)
        else:
            st.warning("Konten Markdown belum tersedia.")

    # --------------------------------------------------------------------------
    # TAB 4: SQL TABULAR STUDIO & CHARTING
    # --------------------------------------------------------------------------
    with tab_sql:
        st.subheader("📊 Lihat tabel dan buat grafik")
        if db_file is not None and db_file.exists():
            with st.expander("Informasi file data"):
                st.write(f"Lokasi database: `{db_file}`")
                st.download_button(
                    "⬇️ Unduh seluruh database (.sqlite)",
                    data=build_sqlite_download(db_file),
                    file_name=db_file.name,
                    mime="application/vnd.sqlite3",
                )
            try:
                conn = sqlite3.connect(str(db_file))
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
                )
                tables = [r[0] for r in cursor.fetchall()]

                if tables:
                    st.download_button(
                        "⬇️ Unduh Semua CSV (.ZIP)",
                        data=build_all_tables_csv_zip(conn, tables),
                        file_name=f"{stem}_csv.zip",
                        mime="application/zip",
                        key=f"all_csv_{stem}",
                    )
                    if "document_headers" in tables and "transaction_details" in tables:
                        with st.expander("📑 Tampilan Relasional Header & Detail Transaksi", expanded=False):
                            df_hdr = pd.read_sql_query("SELECT * FROM document_headers;", conn)
                            st.markdown("**Daftar Header Dokumen Terdaftar:**")
                            st.dataframe(df_hdr, use_container_width=True)
                            if not df_hdr.empty and "header_id" in df_hdr.columns:
                                def _fmt_hdr(hid: Any) -> str:
                                    matching = df_hdr.loc[df_hdr["header_id"] == hid]
                                    if matching.empty:
                                        return f"ID #{hid}"
                                    r_data = matching.iloc[0]
                                    t_val = r_data.get("doc_title")
                                    t_str = str(t_val).strip() if pd.notna(t_val) and str(t_val).strip() else "Dokumen"
                                    n_val = r_data.get("doc_number")
                                    n_str = str(n_val).strip() if pd.notna(n_val) and str(n_val).strip() else "-"
                                    return f"ID #{hid} | {t_str} ({n_str})"

                                selected_hdr = st.selectbox(
                                    "Filter Detail Transaksi Berdasarkan Header ID:",
                                    options=df_hdr["header_id"].tolist(),
                                    format_func=_fmt_hdr,
                                    key="rel_hdr_filter",
                                )
                                if selected_hdr is not None:
                                    df_rel_dtl = pd.read_sql_query(
                                        f"SELECT * FROM transaction_details WHERE header_id = {int(selected_hdr)};", conn
                                    )
                                    st.caption(f"Menampilkan {len(df_rel_dtl)} item transaksi untuk Header ID #{selected_hdr}:")
                                    st.dataframe(df_rel_dtl, use_container_width=True)

                    selected_tbl = st.selectbox(
                        "Pilih Tabel untuk Dilihat:", tables
                    )
                    df_preview = pd.read_sql_query(
                        f"SELECT * FROM '{selected_tbl}' LIMIT 100;", conn
                    )

                    col_t1, col_t2 = st.columns([3, 1])
                    with col_t1:
                        st.markdown(
                            f"**Pratinjau Tabel: `{selected_tbl}` ({len(df_preview)} baris)**"
                        )
                    with col_t2:
                        csv_data = table_csv_bytes(conn, selected_tbl)
                        st.download_button(
                            "⬇️ Unduh Tabel (.CSV)",
                            data=csv_data,
                            file_name=f"{selected_tbl}.csv",
                            mime="text/csv",
                        )

                    st.dataframe(df_preview, use_container_width=True)

                    # Fitur Deduplikasi & Pembersihan Data
                    with st.expander("🧹 Bersihkan data ganda"):
                        st.caption("Cari baris yang sama atau mirip, lalu gabungkan data yang saling melengkapi.")
                        if st.button(f"Bersihkan data ganda pada '{selected_tbl}'", key=f"btn_dedup_{selected_tbl}"):
                            from app.tabular_db import merge_and_deduplicate_tables
                            report = merge_and_deduplicate_tables(db_file, selected_tbl)
                            st.success(f"{report.details}")
                            st.rerun()

                    # Quick Visual Chart jika terdapat kolom numerik
                    num_cols = df_preview.select_dtypes(
                        include=["number"]
                    ).columns.tolist()
                    if num_cols:
                        with st.expander(
                            "📈 Buat grafik dari tabel ini",
                            expanded=False,
                        ):
                            c_type, c_x, c_y = st.columns(3)
                            with c_type:
                                chart_type = st.selectbox(
                                    "Jenis Grafik:",
                                    ["Grafik batang", "Grafik garis", "Grafik area"],
                                    key=f"ct_{selected_tbl}",
                                )
                            with c_x:
                                x_col = st.selectbox(
                                    "Sumbu X (Kategori):",
                                    [None] + list(df_preview.columns),
                                    key=f"cx_{selected_tbl}",
                                )
                            with c_y:
                                y_col = st.selectbox(
                                    "Sumbu Y (Nilai):",
                                    num_cols,
                                    key=f"cy_{selected_tbl}",
                                )

                            chart_df = df_preview.dropna(subset=[y_col])
                            if x_col:
                                chart_df = chart_df.set_index(x_col)

                            if chart_type == "Grafik batang":
                                st.bar_chart(chart_df[[y_col]])
                            elif chart_type == "Grafik garis":
                                st.line_chart(chart_df[[y_col]])
                            elif chart_type == "Grafik area":
                                st.area_chart(chart_df[[y_col]])

                    # Konsol SQL Query
                    with st.expander("🔧 Pencarian lanjutan dengan SQL"):
                        st.caption("Fitur ini ditujukan untuk pengguna yang memahami query SQL.")
                        default_query = f"SELECT * FROM '{selected_tbl}' LIMIT 10;"
                        user_query = st.text_area(
                            "Tulis query SELECT:", value=default_query, height=75
                        )
                        run_query = st.button("Jalankan query", type="primary")
                    if run_query:
                        try:
                            if not user_query.strip().upper().startswith(
                                "SELECT"
                            ):
                                st.error(
                                    "Demi keamanan sistem, hanya query SELECT yang diizinkan."
                                )
                            else:
                                df_query_res = pd.read_sql_query(
                                    user_query, conn
                                )
                                st.success(
                                    f"Query sukses — Ditemukan {len(df_query_res)} baris:"
                                )
                                st.dataframe(
                                    df_query_res, use_container_width=True
                                )
                        except Exception as q_err:  # noqa: BLE001
                            st.error(f"Error query SQL: {q_err}")
                else:
                    st.info(
                        "Database SQLite ada namun belum berisi tabel data."
                    )
                conn.close()
            except Exception as e:  # noqa: BLE001
                st.warning(f"Gagal membaca database SQLite: {e}")
        else:
            st.info(
                "Belum ada file database SQLite yang terbentuk untuk dokumen ini."
            )

    # --------------------------------------------------------------------------
    # TAB 5: RUN LOG
    # --------------------------------------------------------------------------
    with tab_log:
        st.subheader("🔧 Detail proses")
        st.caption("Catatan teknis ini berguna saat menelusuri masalah ekstraksi.")
        log_file = (
            job.latest_log_path
            if job and job.latest_log_path.exists()
            else (output_path / stem / "logs" / f"{stem}_latest.log")
        )
        if not log_file.exists():
            log_file = output_path / "logs" / f"{stem}_latest.log"

        if log_file.exists():
            log_text = log_file.read_text(encoding="utf-8", errors="replace")
            st.download_button(
                "💾 Unduh File Log (.log)",
                data=log_text,
                file_name=log_file.name,
                mime="text/plain",
            )
            st.code(log_text, language="text")
        else:
            st.info("File log belum tersedia untuk dokumen ini.")


# ==============================================================================
# Main Workspace Launcher
# ==============================================================================


def main() -> None:
    """Titik masuk utama aplikasi Streamlit."""
    st.set_page_config(
        page_title="Pengolah Dokumen AI",
        page_icon="📑",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    output_dir = PROJECT_ROOT / "output"
    job_manager = JobManager.get_instance()

    # Inisialisasi session state
    if "selected_stem" not in st.session_state:
        st.session_state["selected_stem"] = None

    # Sidebar: Konfigurasi Pipeline & Navigasi Dokumen
    with st.sidebar:
        st.title("📑 Pengolah Dokumen AI")
        st.caption("Ubah PDF, presentasi, dan gambar menjadi teks serta tabel yang siap digunakan.")

        all_docs = job_manager.list_all_documents(output_dir=output_dir)

        batches = list_batches(output_dir)
        st.markdown("### Histori batch")
        batch_query = st.text_input("Cari batch atau nama file", key="batch_search").casefold()
        for batch in batches:
            if batch_query and batch_query not in (batch["name"] + " " + " ".join(
                doc.get("source_name", doc["stem"]) for doc in batch["documents"]
            )).casefold():
                continue
            with st.expander(f"{batch['name']} · {len(batch['documents'])} file · {batch['created_at'][:10]}"):
                st.caption(batch["created_at"])
                if st.button("Buka batch", key=f"history_batch_{batch['id']}"):
                    st.session_state["selected_batch_id"] = batch["id"]
                    st.session_state["selected_stem"] = batch["documents"][0]["stem"] if batch["documents"] else None
                    st.rerun()
        if not batches:
            st.caption("Batch upload baru akan tercatat di sini, termasuk setelah restart.")

        st.markdown("### Histori dokumen")
        query = st.text_input("Cari nama dokumen", key="document_search").casefold()
        status_labels = {"queued": "Menunggu", "running": "Diproses", "completed": "Selesai",
                         "failed": "Gagal", "canceled": "Dibatalkan"}
        status_filter = st.selectbox("Filter status", ["all", *status_labels],
                                     format_func=lambda value: status_labels.get(value, "Semua status"))
        visible_docs = [doc for doc in all_docs if query in doc["stem"].casefold()
                        and (status_filter == "all" or doc["status"] == status_filter)]
        options = [None, *[doc["stem"] for doc in visible_docs]]
        current = st.session_state.get("selected_stem")
        if current and current not in options:
            options.append(current)
        doc_by_stem = {doc["stem"]: doc for doc in all_docs}

        def document_label(stem: str | None) -> str:
            if stem is None:
                return "➕ Unggah dokumen baru"
            doc = doc_by_stem.get(stem, {})
            return f"{status_labels.get(doc.get('status', ''), 'Tidak diketahui')} · {stem}"

        chosen_stem = st.selectbox("Pilih dokumen", options, index=options.index(current),
                                   format_func=document_label)
        st.caption(f"{len(visible_docs)} dari {len(all_docs)} dokumen · antrean aktif di atas, lalu terbaru")
        if chosen_stem != current:
            st.session_state["selected_stem"] = chosen_stem
            st.session_state["selected_batch_id"] = None
            st.rerun()
        if st.button("Upload baru"):
            st.session_state["selected_stem"] = None
            st.session_state["selected_batch_id"] = None
            st.rerun()

        st.markdown("---")
        st.header("⚙️ Pengaturan ekstraksi")
        spec_label = st.selectbox(
            "Bentuk dokumen:",
            list(SPEC_OPTIONS.keys()),
            help="Biarkan pilihan otomatis jika Anda tidak yakin.",
        )
        chosen_spec = SPEC_OPTIONS[spec_label]

        with st.expander("Pengaturan Lanjutan", expanded=False):
            dpi_val = st.slider(
                "Ketajaman gambar (PDF/PPT):", 100, 300, 200, 25,
                help="Nilai lebih tinggi dapat membantu dokumen kecil atau buram, tetapi prosesnya lebih lama.",
            )
            force_all_tbl = st.checkbox(
                "Simpan semua jenis tabel",
                value=False,
                help="Aktifkan jika tabel deskriptif juga perlu disimpan sebagai data terstruktur.",
            )

        st.markdown("---")
        if st.button("🔄 Segarkan histori", use_container_width=True):
            st.rerun()

    active_stem = st.session_state.get("selected_stem")
    selected_batch = next((batch for batch in batches
                           if batch["id"] == st.session_state.get("selected_batch_id")), None)
    if selected_batch:
        render_batch_download(selected_batch, output_dir)
        render_batch_monitor([doc["stem"] for doc in selected_batch["documents"]], output_dir)

    if active_stem is None:
        # MODE 1: UNGGAH DOKUMEN BARU
        st.subheader("📤 Unggah Dokumen Baru")
        st.caption(
            "1. Pilih file · 2. Sesuaikan pengaturan bila perlu · 3. Mulai ekstraksi. "
            "Mendukung PDF, PPTX/PPT, PNG, JPG, dan WebP."
        )

        uploaded_files = st.file_uploader(
            "Pilih satu atau beberapa file:",
            type=SUPPORTED_TYPES,
            accept_multiple_files=True,
            key="document_files_uploader",
            help="Pilih beberapa PDF, PPT/PPTX, PNG, JPG, atau WebP sekaligus.",
        )
        uploaded_directory = st.file_uploader(
            "Atau pilih satu folder:",
            type=SUPPORTED_TYPES,
            accept_multiple_files="directory",
            key="document_directory_uploader",
            help=(
                "Semua file yang didukung di dalam folder (termasuk subfolder) "
                "akan dimasukkan ke antrean ingest. Pilihan folder dapat digabung "
                "dengan pilihan file di atas."
            ),
        )

        # Streamlit/browser hanya menyediakan satu dialog folder per widget.
        # Gabungkan hasil folder dengan file biasa agar semuanya dapat diproses
        # melalui tombol yang sama, tanpa mengubah pipeline ingest.
        uploaded_files = list(uploaded_files or []) + list(uploaded_directory or [])

        if uploaded_files:
            saved_files = _save_uploaded_files(uploaded_files, output_dir)
            st.markdown(f"**{len(saved_files)} file unik siap diproses**")
            batch_name = st.text_input("Nama kelompok hasil", value="Upload dokumen")
            uploaded_rows = []
            for path in saved_files:
                existing_job = job_manager.get_job(path.stem, output_dir=output_dir)
                uploaded_rows.append(
                    {
                        "File": path.name,
                        "Ukuran": f"{path.stat().st_size / 1024:.1f} KB",
                        "Status": existing_job.status if existing_job else "siap",
                    }
                )
            st.dataframe(
                pd.DataFrame(uploaded_rows),
                use_container_width=True,
                hide_index=True,
            )

            if st.button(
                f"🚀 Mulai Ekstraksi {len(saved_files)} File",
                type="primary",
                use_container_width=True,
            ):
                documents = []
                for uploaded in uploaded_files:
                    path = _save_uploaded_file(uploaded, output_dir)
                    documents.append({"stem": path.stem, "source_name": uploaded.name})
                batch = create_batch(output_dir, batch_name, documents)
                st.session_state["selected_batch_id"] = batch["id"]
                started_jobs = []
                for saved_file in saved_files:
                    job = job_manager.start_job(
                        input_path=saved_file,
                        output_dir=output_dir,
                        doc_type=chosen_spec,
                        dpi=dpi_val,
                        force_all_tables=force_all_tbl,
                    )
                    started_jobs.append(job)

                # Buka monitor file pertama; semua file tetap berjalan di background.
                if started_jobs:
                    st.session_state["selected_stem"] = started_jobs[0].job_id
                    st.session_state["batch_upload_stems"] = [
                        job.job_id for job in started_jobs
                    ]
                    st.rerun()

    else:
        # MODE 2: DOKUMEN AKTIF DIPILIH
        job = job_manager.get_job(active_stem, output_dir=output_dir)

        if job is not None and job.status in {"queued", "running"}:
            render_live_monitor(active_stem, output_dir)

        elif job is not None and job.status == "completed":
            render_completed_document_view(active_stem, output_dir)

        elif job is not None and job.status in ("failed", "canceled"):
            if job.status == "canceled":
                st.warning(
                    "⚠️ **Proses ekstraksi telah dibatalkan oleh pengguna.**"
                )
            else:
                st.error("❌ **Terjadi Kesalahan saat Ekstraksi Dokumen**")
                if job.error_message:
                    st.error(f"Detail Kesalahan: {job.error_message}")
                if job.latest_log_path:
                    st.info(f"📂 **Lokasi Log Lengkap:** `{job.latest_log_path.resolve()}`")

            st.markdown("##### 📜 Log Terakhir Sebelum Berhenti:")
            st.code(
                job_manager.get_latest_logs(active_stem, line_count=40),
                language="text",
            )

            col_f1, col_f2 = st.columns([1, 1])
            with col_f1:
                if st.button("🔄 Coba Ekstrak Ulang", type="primary", use_container_width=True):
                    job_manager.restart_job(active_stem, output_dir=output_dir)
                    st.rerun()
            with col_f2:
                if st.button("➕ Beralih ke Unggah Dokumen Lain", use_container_width=True):
                    st.session_state["selected_stem"] = None
                    st.rerun()

        else:
            candidate_uploads = list(
                (output_dir / "uploads").glob(f"{active_stem}.*")
            )
            if candidate_uploads:
                input_file = candidate_uploads[0]
                st.info(f"📄 File siap diekstrak: **{input_file.name}**")
                if st.button("🚀 Mulai Ekstraksi Sekarang", type="primary"):
                    job_manager.start_job(
                        input_path=input_file,
                        output_dir=output_dir,
                        doc_type=chosen_spec,
                        dpi=dpi_val,
                        force_all_tables=force_all_tbl,
                    )
                    st.rerun()
            else:
                st.warning(
                    f"Dokumen `{active_stem}` tidak ditemukan dalam sistem."
                )
                if st.button("⬅️ Kembali ke Unggah Dokumen"):
                    st.session_state["selected_stem"] = None
                    st.rerun()


if __name__ == "__main__":
    main()
