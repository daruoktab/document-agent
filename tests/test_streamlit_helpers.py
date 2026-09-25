"""
Unit tests untuk helper UI Streamlit: pemecahan halaman Markdown, ekstraksi diagram Mermaid, dan pencarian gambar dokumen.
"""

from __future__ import annotations

import base64
import json
import re
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from app.streamlit_logic import (
    WORKSPACE_PAGES,
    _load_brand_assets,
    _save_uploaded_file,
    build_document_zip,
    extract_mermaid_blocks,
    find_pages_containing,
    format_timestamp,
    get_document_images,
    read_completed_markdown_pages,
    render_mermaid_html,
    split_markdown_by_pages,
)


class _UploadedFile:
    def __init__(self, name: str, content: bytes) -> None:
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


class TestStreamlitHelpers(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_st_helpers_"))

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_mermaid_renderer_passes_code_as_javascript_string(self) -> None:
        code = 'flowchart LR\nA --> B["Data primer:<br/>lokasi"]'
        with patch("app.streamlit_logic.st.components.v1.html") as component:
            render_mermaid_html(code)

        page = component.call_args.args[0]
        source = re.search(r"const diagramSource = (.*);", page)
        self.assertIsNotNone(source)
        self.assertEqual(json.loads(source.group(1)), code)
        self.assertIn("mermaid.render('mermaid-svg', diagramSource)", page)
        self.assertNotIn("--&gt;", page)
        self.assertNotIn("<br/>", page)

    def test_format_timestamp_handles_iso_and_missing_values(self) -> None:
        self.assertEqual(format_timestamp(None), "—")
        self.assertIn("17 Sep 2026", format_timestamp("2026-09-17T07:32:08+00:00"))

    def test_workspace_has_distinct_primary_destinations(self) -> None:
        self.assertEqual(WORKSPACE_PAGES, ("Dashboard", "Upload", "Histori", "Dokumen"))

    def test_local_brand_assets_are_loadable(self) -> None:
        background, logo = _load_brand_assets()

        self.assertTrue(background)
        self.assertTrue(logo)
        self.assertTrue(base64.b64decode(background).startswith(b"\xff\xd8\xff"))
        self.assertIn(b"<svg", base64.b64decode(logo))

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

    def test_read_completed_markdown_pages_ignores_page_still_being_written(self) -> None:
        markdown_file = self.temp_dir / "stream.md"
        markdown_file.write_text(
            "<!-- PAGE: 1 -->\nSelesai\n\n---\n"
            "<!-- PAGE: 2 -->\nMasih ditulis",
            encoding="utf-8",
        )

        pages = read_completed_markdown_pages(markdown_file)

        self.assertEqual(pages, {1: "Selesai"})

    def test_read_completed_markdown_pages_handles_missing_file(self) -> None:
        self.assertEqual(
            read_completed_markdown_pages(self.temp_dir / "belum-ada.md"), {}
        )

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

    def test_upload_same_stem_different_extension_gets_unique_stem(self) -> None:
        first = _save_uploaded_file(
            _UploadedFile("laporan.pdf", b"pdf"), self.temp_dir
        )
        second = _save_uploaded_file(
            _UploadedFile("laporan.docx", b"docx"), self.temp_dir
        )

        self.assertEqual(first.name, "laporan.pdf")
        self.assertEqual(second.name, "laporan (1).docx")
        self.assertNotEqual(first.stem, second.stem)


if __name__ == "__main__":
    unittest.main()
