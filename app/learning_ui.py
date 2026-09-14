"""Correction form; optimization and release privileges remain in the local CLI."""

import sqlite3
from pathlib import Path

import streamlit as st

from .learning_store import LearningStore, digest
from .prompts import build_extraction_prompt


def render_correction_form(
    *,
    stem: str,
    page: int,
    image_path: Path,
    original: str,
    images: list[Path],
    source_path: Path | None,
) -> None:
    with st.expander("✏️ Koreksi hasil halaman ini"):
        if not original.strip():
            st.info(
                "Teks halaman ini belum tersedia. Koreksi dapat disimpan setelah teks berhasil dipetakan ke halaman."
            )
            return
        st.caption(
            "Periksa kembali teks dan tabel terhadap gambar asli. Koreksi disimpan terpisah; "
            "hasil unduhan awal tetap tersedia. Koreksi yang disetujui dapat dipakai tim "
            "untuk mengevaluasi dan meningkatkan ekstraksi."
        )
        store = LearningStore()
        try:
            if source_path and source_path.is_file():
                document_hash = digest(source_path.read_bytes())
            else:
                document_hash = digest(
                    "".join(digest(p.read_bytes()) for p in images).encode()
                )
            saved = store.latest_correction(document_hash, page)
            observation = store.observation(image_path)
            key = f"correction_{document_hash}_{page}_{digest(original.encode())[:12]}"
            with st.form(key):
                corrected = st.text_area(
                    "Teks dan tabel yang sudah diperbaiki",
                    value=saved["corrected"] if saved else original,
                    key=f"{key}_text",
                    height=320,
                    help="Pertahankan susunan tabel Markdown: setiap kolom dipisahkan tanda |.",
                )
                ready = st.checkbox(
                    "Siap digunakan untuk evaluasi",
                    value=saved["ready"] if saved else False,
                    key=f"{key}_ready",
                )
                if st.form_submit_button("Simpan koreksi"):
                    store.save_correction(
                        document_hash=document_hash,
                        document_name=stem,
                        page=page,
                        image_path=image_path,
                        original=original,
                        corrected=corrected,
                        context=observation["context"]
                        if observation
                        else build_extraction_prompt(specs="plain"),
                        version=observation["version"]
                        if observation
                        else "legacy-unknown",
                        ready=ready,
                    )
                    st.success(
                        "Koreksi tersimpan untuk evaluasi."
                        if ready
                        else "Draf koreksi tersimpan dan belum masuk bahan evaluasi."
                    )
            if not observation:
                st.caption(
                    "Dokumen lama: versi prompt belum tercatat. Evaluasi memakai aturan dokumen biasa tanpa konteks halaman sebelumnya."
                )
        except (OSError, ValueError, sqlite3.Error) as exc:
            st.error(f"Koreksi belum dapat disimpan: {exc}")
