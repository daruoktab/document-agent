"""
Unit tests untuk helper UI Streamlit: pemecahan halaman Markdown, ekstraksi diagram Mermaid, dan pencarian gambar dokumen.
"""

from __future__ import annotations

import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from app.streamlit_logic import (
    build_document_zip,
    extract_mermaid_blocks,
    find_pages_containing,
    get_document_images,
    split_markdown_by_pages,
)


class TestStreamlitHelpers(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_st_helpers_"))

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_split_markdown_by_pages_pdf_format(self) -> None:
        raw_md = (
            "<!-- PAGE: 1 -->\n"
            "# Halaman 1\nTeks hal 1.\n\n"
            "<!-- PAGE: 2 -->\n"
            "# Halaman 2\nTeks hal 2.\n"
        )
        pages = split_markdown_by_pages(raw_md)
        self.assertEqual(len(pages), 2)
        self.assertIn(1, pages)
        self.assertIn(2, pages)
        self.assertIn("Teks hal 1", pages[1])
        self.assertIn("Teks hal 2", pages[2])

    def test_split_markdown_by_pages_ppt_format(self) -> None:
        raw_md = (
            "## Slide 1\nJudul Presentasi.\n\n"
            "## Slide 2\nAgenda Rapat.\n"
        )
        pages = split_markdown_by_pages(raw_md)
        self.assertEqual(len(pages), 2)
        self.assertIn("Judul Presentasi", pages[1])
        self.assertIn("Agenda Rapat", pages[2])

    def test_extract_mermaid_blocks(self) -> None:
        raw_md = (
            "# Laporan Arsitektur\n"
            "```mermaid\n"
            "graph TD;\n"
            "  A-->B;\n"
            "```\n"
            "Penjelasan diagram di atas.\n"
            "```mermaid\n"
            "sequenceDiagram\n"
            "  User->>Server: Request\n"
            "```\n"
        )
        diagrams = extract_mermaid_blocks(raw_md)
        self.assertEqual(len(diagrams), 2)
        self.assertIn("graph TD;", diagrams[0])
        self.assertIn("sequenceDiagram", diagrams[1])

    def test_find_pages_containing_is_case_insensitive(self) -> None:
        pages = {1: "Nomor Kontrak: ABC-123", 2: "Rincian anggaran kegiatan"}

        self.assertEqual(find_pages_containing(pages, "kontrak"), [1])
        self.assertEqual(find_pages_containing(pages, "ANGGARAN"), [2])
        self.assertEqual(find_pages_containing(pages, "  "), [])

    def test_get_document_images_pages_and_slides(self) -> None:
        stem = "doc_test"
        doc_dir = self.temp_dir / stem
        pages_dir = doc_dir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        (pages_dir / "page_0001.png").write_bytes(b"dummy1")
        (pages_dir / "page_0002.png").write_bytes(b"dummy2")

        imgs = get_document_images(stem, self.temp_dir)
        self.assertEqual(len(imgs), 2)
        self.assertEqual(imgs[0].name, "page_0001.png")
        self.assertEqual(imgs[1].name, "page_0002.png")

    def test_build_document_zip_contains_all_document_results(self) -> None:
        stem = "laporan"
        doc_dir = self.temp_dir / stem
        (doc_dir / "pages").mkdir(parents=True)
        (doc_dir / "databases").mkdir()
        (doc_dir / "csv").mkdir()
        (doc_dir / f"{stem}.md").write_text("# Hasil", encoding="utf-8")
        (doc_dir / "pages" / "page_0001.png").write_bytes(b"png")
        (doc_dir / "databases" / f"{stem}.sqlite").write_bytes(b"sqlite")
        (doc_dir / "csv" / "tabel.csv").write_text("nilai\n1", encoding="utf-8")

        zip_data = build_document_zip(stem, self.temp_dir)

        with ZipFile(BytesIO(zip_data)) as archive:
            self.assertEqual(
                set(archive.namelist()),
                {
                    "PETUNJUK.txt",
                    f"{stem}.md",
                    "pages/page_0001.png",
                    f"databases/{stem}.sqlite",
                    "csv/tabel.csv",
                },
            )


if __name__ == "__main__":
    unittest.main()
