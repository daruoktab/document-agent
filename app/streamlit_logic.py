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

import re
import sqlite3
from typing import Any
import sys
from pathlib import Path

# Pastikan root direktori proyek berada di sys.path agar impor 'from app....' selalu dikenali
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from app.job_tracker import JobManager, is_pid_alive
from app.tabular_db import cross_verify_dual_track

SUPPORTED_TYPES = ["pdf", "pptx", "ppt", "png", "jpg", "jpeg", "webp"]

SPEC_OPTIONS: dict[str, str | None] = {
    "Auto-Detect (Rekomendasi VLM)": None,
    "Plain Document (Surat/Dokumen Standar)": "plain",
    "Markdown Hierarchy (Bab, Sub-bab #, ##, ###)": "markdown_hierarchy",
    "Bilingual Journal (2 Kolom & 2 Bahasa)": "bilingual_journal",
    "Presentation Slides (Slide & Diagram Visual)": "presentation_slides",
}


# ==============================================================================
# Helper Functions (Dapat Digunakan Kembali & Diuji Secara Independen)
# ==============================================================================


def _save_uploaded_file(uploaded_file: UploadedFile, output_dir: Path) -> Path:
    """Simpan file yang diunggah ke direktori output/uploads secara stabil."""
    uploads_dir = output_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    target_path = uploads_dir / Path(uploaded_file.name).name
    target_path.write_bytes(uploaded_file.getvalue())
    return target_path


def _save_uploaded_files(
    uploaded_files: list[UploadedFile] | tuple[UploadedFile, ...],
    output_dir: Path,
) -> list[Path]:
    """Simpan beberapa file upload dan kembalikan path dalam urutan pilihan user."""
    return [_save_uploaded_file(uploaded_file, output_dir) for uploaded_file in uploaded_files]


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


def extract_mermaid_blocks(text: str) -> list[str]:
    """Cari seluruh blok ```mermaid ... ``` dalam dokumen markdown."""
    pattern = r"```(?:mermaid)\s*\n(.*?)\n```"
    return re.findall(pattern, text, re.DOTALL | re.IGNORECASE)


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
    st.markdown(f"### ⏳ Ekstraksi Berjalan: `{job.file_name or stem}`")
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
    c2.metric("🔄 Tahapan Proses", job.stage)
    c3.metric("⚙️ PID Subprocess", str(job.pid or "-"))
    is_alive = is_pid_alive(job.pid) if job.pid else True
    c4.metric(
        "🩺 Status Eksekusi",
        (
            "🟡 Menunggu antrean"
            if job.status == "queued"
            else ("🟢 Berjalan Normal" if is_alive else "⚠️ Tidak Responsif")
        ),
    )

    # Kotak informasi background safety & path file log fisik
    log_file_str = str(job.latest_log_path.resolve()) if job.latest_log_path else "output/logs/..."
    st.info(
        f"📂 **Lokasi Berkas Log Disk:** `{log_file_str}`\n\n"
        f"📌 **Aktivitas Terakhir:** *{job.last_message or 'Menunggu baris pertama...'}*"
    )

    # Tampilan log real-time streaming
    st.markdown("##### 📜 Live Log Streaming (Terminal Subprocess):")
    recent_log_text = job_manager.get_latest_logs(stem, line_count=35)
    if not recent_log_text or recent_log_text == "(Belum ada log)":
        recent_log_text = (
            "(Sedang menginisialisasi subprocess Python .venv dan memuat dependensi Vision VLM...\n"
            "Baris log pertama akan muncul di sini sesaat lagi...)"
        )
    st.code(recent_log_text, language="text")
    st.caption(
        f"💡 **Tips CLI:** Anda juga dapat memantau log langsung dari terminal PowerShell menggunakan: "
        f"`Get-Content -Path '{log_file_str}' -Wait`"
    )

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
        st.subheader(f"📄 Hasil Ekstraksi: `{stem}`")
        badge_pages = f"{len(images)} Halaman Fisik" if images else "Format Teks"
        st.caption(
            f"Status: **Selesai (Completed)** | {badge_pages} | Target Markdown: `{md_file.name}`"
        )
    with col_top2:
        if st.button("🔄 Ekstrak Ulang Dokumen Ini", use_container_width=True):
            job_manager.reset_job(stem, output_dir=output_path)
            st.rerun()

    # Tab Utama (Fokus Ekstraksi, Verifikasi Dual-Track, Mermaid, & Tabular Studio)
    tab_inspector, tab_guardrail, tab_md, tab_sql, tab_log = st.tabs([
        "🔍 Visual Page Inspector",
        "🛡️ Dual-Track Guardrail Audit",
        "📝 Markdown & Diagram Mermaid",
        "📊 Data Tabular (SQLite & Chart)",
        "📋 Run Log",
    ])

    # --------------------------------------------------------------------------
    # TAB 1: VISUAL PAGE INSPECTOR (SIDE-BY-SIDE)
    # --------------------------------------------------------------------------
    with tab_inspector:
        st.subheader("🔍 Side-by-Side Visual Document Inspector")
        st.caption(
            "Verifikasi kesesuaian antara kanvas dokumen fisik (gambar asli hasil rasterisasi) dengan teks Markdown hasil ekstraksi Vision VLM."
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
                    "Mode Teks Kolom Kanan:",
                    ["Halaman Ini Saja", "Teks Utuh Dokumen"],
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
                    f"##### 🖼️ Kanvas Asli: Halaman {selected_page_idx} / {total_imgs}"
                )
                st.image(
                    str(current_img_path),
                    caption=f"{current_img_path.name} (High-Res)",
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
                    f"##### ✍️ Hasil Ekstraksi Markdown: {'Halaman ' + str(selected_page_idx) if view_mode == 'Halaman Ini Saja' else 'Dokumen Lengkap'}"
                )
                text_to_show = (
                    current_page_text
                    if view_mode == "Halaman Ini Saja"
                    else md_content
                )

                # Deteksi jika halaman ini memiliki diagram Mermaid
                page_mermaids = extract_mermaid_blocks(text_to_show)
                if page_mermaids:
                    st.info(
                        f"🎨 Terdeteksi {len(page_mermaids)} Diagram Alur (Mermaid) pada bagian ini."
                    )
                    with st.expander(
                        "Tampilkan Render SVG Diagram", expanded=True
                    ):
                        for m_code in page_mermaids:
                            render_mermaid_html(m_code, height=320)

                st.markdown(text_to_show)
        else:
            st.info(
                "ℹ️ Gambar halaman fisik tidak ditemukan untuk dokumen ini (mungkin dokumen diproses tanpa menyimpan kanvas halaman terpisah)."
            )
            st.markdown(md_content)

    # --------------------------------------------------------------------------
    # TAB 2: DUAL-TRACK GUARDRAIL AUDIT
    # --------------------------------------------------------------------------
    with tab_guardrail:
        st.subheader("🛡️ Laporan Audit Guardrail Supervisor Jalur Ganda")
        st.caption(
            "Agent Supervisor membandingkan konsistensi numerik Jalur 1 (Teks Dokumen Markdown) vs Jalur 2 (Database Relasional SQLite)."
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
            st.success(f"✅ **STATUS GUARDRAIL: PASSED** — {report.supervisor_notes}")
        elif report.guardrail_status == "WARNING":
            st.warning(f"⚠️ **STATUS GUARDRAIL: WARNING** — {report.supervisor_notes}")
        else:
            st.error(f"❌ **STATUS GUARDRAIL: FAILED** — {report.supervisor_notes}")

        # Metrik Rekonsiliasi
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tabel Markdown", report.total_markdown_tables)
        m2.metric("Tabel SQLite", report.total_sqlite_tables)
        m3.metric("Baris Data Markdown", report.total_markdown_rows)
        m4.metric("Baris Data SQLite", report.total_sqlite_rows)

        if report.discrepancies:
            with st.expander("⚠️ Catatan Diskrepansi / Selisih", expanded=True):
                for disc in report.discrepancies:
                    st.markdown(f"- {disc}")

        if report.table_comparisons:
            st.markdown("#### 📋 Tabel Komparasi Jalur Ganda")
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
        st.subheader("📝 Teks Dokumen (Markdown) & Visualisasi Diagram")
        if md_content:
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
                    f"### 🎨 Visualisasi Diagram Mermaid ({len(doc_mermaids)} Diagram Ditemukan)"
                )
                for idx, m_code in enumerate(doc_mermaids, 1):
                    with st.expander(
                        f"📊 Diagram #{idx} (Render SVG Interaktif)",
                        expanded=True,
                    ):
                        render_mermaid_html(m_code, height=380)
                        st.caption("Kode Sumber Mermaid:")
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
        st.subheader("📊 Penjelajah Database SQLite & Visual Charting")
        if db_file is not None and db_file.exists():
            st.write(f"📁 Path Database: `{db_file}`")
            try:
                conn = sqlite3.connect(str(db_file))
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
                )
                tables = [r[0] for r in cursor.fetchall()]

                if tables:
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
                        csv_data = df_preview.to_csv(index=False).encode(
                            "utf-8"
                        )
                        st.download_button(
                            "⬇️ Unduh Tabel (.CSV)",
                            data=csv_data,
                            file_name=f"{selected_tbl}.csv",
                            mime="text/csv",
                        )

                    st.dataframe(df_preview, use_container_width=True)

                    # Fitur Deduplikasi & Pembersihan Data
                    with st.expander("🧹 Deduplikasi & Konsolidasi Data"):
                        st.caption("Deteksi baris-baris identik atau mirip dari multiple sumber, gabungkan nilai non-null, dan bersihkan duplikasi.")
                        if st.button(f"Jalankan Merge & Deduplikasi pada '{selected_tbl}'", key=f"btn_dedup_{selected_tbl}"):
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
                            "📈 Visualisasi Grafik Cepat (Auto-Chart)",
                            expanded=False,
                        ):
                            c_type, c_x, c_y = st.columns(3)
                            with c_type:
                                chart_type = st.selectbox(
                                    "Jenis Grafik:",
                                    ["Bar Chart", "Line Chart", "Area Chart"],
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

                            if chart_type == "Bar Chart":
                                st.bar_chart(chart_df[[y_col]])
                            elif chart_type == "Line Chart":
                                st.line_chart(chart_df[[y_col]])
                            elif chart_type == "Area Chart":
                                st.area_chart(chart_df[[y_col]])

                    # Konsol SQL Query
                    st.markdown("#### ⚡ Konsol Query SQL")
                    default_query = f"SELECT * FROM '{selected_tbl}' LIMIT 10;"
                    user_query = st.text_area(
                        "Tulis query SELECT:", value=default_query, height=75
                    )
                    if st.button("Jalankan Query", type="primary"):
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
        st.subheader("📋 Log Eksekusi Lengkap")
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
        page_title="Vision VLM & Dual-Track Sub-Agent Workspace",
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
        st.title("📑 Document AI Studio")
        st.caption(
            "Vision VLM, SQL Tabular & Dual-Track Guardrail Studio"
        )

        all_docs = job_manager.list_all_documents(output_dir=output_dir)

        doc_options: dict[str, str | None] = {"➕ [Unggah Dokumen Baru]": None}
        for doc in all_docs:
            stem = doc["stem"]
            status = doc["status"]
            badge = (
                "⏳"
                if status == "running"
                else ("🟢" if status == "completed" else "❌")
            )
            pg_info = (
                f" ({doc['page_count']} hal)" if doc.get("page_count") else ""
            )
            label = f"{badge} {stem[:28]}...{pg_info}"
            doc_options[label] = stem

        current_selected = st.session_state.get("selected_stem")
        if current_selected and current_selected not in doc_options.values():
            doc_options[f"⏳ {current_selected} (Aktif)"] = current_selected

        selected_label_idx = 0
        if current_selected is not None:
            for idx, (lbl, val) in enumerate(doc_options.items()):
                if val == current_selected:
                    selected_label_idx = idx
                    break

        st.markdown("### 📂 Navigasi Dokumen")
        chosen_label = st.selectbox(
            "Pilih Dokumen Aktif:",
            options=list(doc_options.keys()),
            index=selected_label_idx,
            help="Bebas F5/Reload: Anda dapat memilih dokumen yang pernah diekstrak kapan saja tanpa perlu mengunggah ulang.",
        )
        new_selected_stem = doc_options[chosen_label]

        if new_selected_stem != st.session_state.get("selected_stem"):
            st.session_state["selected_stem"] = new_selected_stem
            st.rerun()

        st.markdown("---")
        st.header("⚙️ Konfigurasi Pipeline")
        spec_label = st.selectbox(
            "Jenis Dokumen / Layout:", list(SPEC_OPTIONS.keys())
        )
        chosen_spec = SPEC_OPTIONS[spec_label]

        with st.expander("Pengaturan Lanjutan", expanded=False):
            dpi_val = st.slider(
                "DPI Raster Rendering (PDF/PPT):", 100, 300, 200, 25
            )
            force_all_tbl = st.checkbox(
                "Paksa Ingesti Seluruh Tabel ke SQLite",
                value=False,
                help="Jika dicentang, tabel naratif/kualitatif juga dimasukkan ke SQLite di samping tabel transaksional.",
            )

        st.markdown("---")
        st.markdown("#### 🕒 Riwayat Dokumen Cepat")
        if all_docs:
            for doc in all_docs[:6]:
                stem = doc["stem"]
                status = doc["status"]
                status_icon = (
                    "⏳"
                    if status in {"queued", "running"}
                    else ("🟢" if status == "completed" else "❌")
                )
                col_btn1, col_btn2 = st.columns([3.5, 1])
                with col_btn1:
                    st.caption(f"{status_icon} **{stem[:22]}...**")
                with col_btn2:
                    if st.button("Buka", key=f"btn_nav_{stem}"):
                        st.session_state["selected_stem"] = stem
                        st.rerun()
        else:
            st.caption("(Belum ada dokumen yang diproses)")

        if st.button("🔄 Segarkan Daftar Dokumen", use_container_width=True):
            st.rerun()

    # Main Area Router
    active_stem = st.session_state.get("selected_stem")

    batch_stems = st.session_state.get("batch_upload_stems", [])
    if len(batch_stems) > 1:
        render_batch_monitor(batch_stems, output_dir)

    if active_stem is None:
        # MODE 1: UNGGAH DOKUMEN BARU
        st.subheader("📤 Unggah Dokumen Baru")
        st.caption(
            "Mendukung format PDF multi-halaman, presentasi PPTX/PPT, dan gambar resolusi tinggi."
        )

        uploaded_files = st.file_uploader(
            "Pilih file dokumen:",
            type=SUPPORTED_TYPES,
            accept_multiple_files=True,
            help="File akan disimpan secara aman di output/uploads/ dan dieksekusi di latar belakang.",
        )

        if uploaded_files:
            saved_files = _save_uploaded_files(uploaded_files, output_dir)
            st.markdown(f"**{len(saved_files)} file siap diproses**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "File": path.name,
                            "Ukuran": f"{path.stat().st_size / 1024:.1f} KB",
                            "Status": (
                                job_manager.get_job(path.stem, output_dir=output_dir).status
                                if job_manager.get_job(path.stem, output_dir=output_dir)
                                else "siap"
                            ),
                        }
                        for path in saved_files
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

            if st.button(
                f"🚀 Mulai Ekstraksi {len(saved_files)} File",
                type="primary",
                use_container_width=True,
            ):
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
                    job_manager.reset_job(active_stem, output_dir=output_dir)
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
