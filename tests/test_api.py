import csv
import io
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

from fastapi.testclient import TestClient

from app import api
from app.upload_batches import create_batch


class IngestApiTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        output_patch = patch.object(api, "OUTPUT_DIR", self.root)
        output_patch.start()
        self.addCleanup(output_patch.stop)
        manager_patch = patch.object(api.JobManager, "get_instance")
        self.manager = manager_patch.start().return_value
        self.addCleanup(manager_patch.stop)
        self.manager.start_job.side_effect = self.complete_job
        self.client = TestClient(api.app)
        self.addCleanup(self.client.close)

    def complete_job(self, **kwargs):
        path = kwargs["input_path"]
        md = self.root / f"{path.stem}.md"
        md.write_text("# Hasil\nTeks.", encoding="utf-8")
        db = self.root / f"{path.stem}.sqlite"
        with closing(sqlite3.connect(db)) as conn, conn:
            conn.execute('CREATE TABLE "data/item" (nama TEXT, jumlah INTEGER)')
            conn.execute(
                'INSERT INTO "data/item" VALUES (?, ?)', ('Kopi, "A"\nBaru', 2)
            )
        return SimpleNamespace(
            job_id=path.stem, status="completed", out_file=md, db_file=db
        )

    def test_zip_sql_roundtrip_and_csv(self):
        response = self.client.post("/ingest", files={"file": ("scan.png", b"image")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/zip")
        with ZipFile(io.BytesIO(response.content)) as archive:
            self.assertEqual(
                set(archive.namelist()),
                {"document.md", "document.sql", "csv/data_item.csv"},
            )
            self.assertTrue(archive.read("document.md").decode().startswith("# Hasil"))
            with closing(sqlite3.connect(":memory:")) as conn, conn:
                conn.executescript(archive.read("document.sql").decode())
                self.assertEqual(
                    conn.execute('SELECT jumlah FROM "data/item"').fetchone(), (2,)
                )
            rows = list(
                csv.reader(
                    io.StringIO(archive.read("csv/data_item.csv").decode("utf-8-sig"))
                )
            )
            self.assertEqual(rows, [["nama", "jumlah"], ['Kopi, "A"\nBaru', "2"]])
        self.assertEqual(
            set(self.manager.start_job.call_args.kwargs), {"input_path", "output_dir"}
        )

    def test_no_database(self):
        def without_db(**kwargs):
            job = self.complete_job(**kwargs)
            job.db_file.unlink()
            return job

        self.manager.start_job.side_effect = without_db
        response = self.client.post("/ingest", files={"file": ("scan.png", b"image")})
        self.assertEqual(response.status_code, 200)
        with ZipFile(io.BytesIO(response.content)) as archive:
            self.assertEqual(set(archive.namelist()), {"document.md", "document.sql"})
            self.assertEqual(
                archive.read("document.sql"), b"BEGIN TRANSACTION;\nCOMMIT;\n"
            )

    def test_validation(self):
        for filename, content, status in [("x.exe", b"x", 415), ("x.pdf", b"", 400)]:
            self.assertEqual(
                self.client.post(
                    "/ingest", files={"file": (filename, content)}
                ).status_code,
                status,
            )
        with patch.object(api, "MAX_UPLOAD_BYTES", 2):
            self.assertEqual(
                self.client.post(
                    "/ingest", files={"file": ("x.pdf", b"123")}
                ).status_code,
                413,
            )
        self.assertEqual(self.client.post("/ingest").status_code, 422)
        self.manager.start_job.assert_not_called()
        self.assertEqual(list((self.root / "uploads").iterdir()), [])

    def test_safe_unique_uploads(self):
        for _ in range(2):
            response = self.client.post(
                "/ingest", files={"file": ("../../scan.PDF", b"pdf")}
            )
            self.assertEqual(response.status_code, 200)
        paths = [
            call.kwargs["input_path"] for call in self.manager.start_job.call_args_list
        ]
        self.assertNotEqual(paths[0], paths[1])
        self.assertTrue(all(p.parent == self.root / "uploads" for p in paths))

    def test_docx_upload_is_supported(self):
        response = self.client.post("/ingest", files={"file": ("surat.docx", b"docx")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            self.manager.start_job.call_args.kwargs["input_path"].suffix, ".docx"
        )

    def test_excel_upload_is_supported(self):
        response = self.client.post("/ingest", files={"file": ("data.xlsx", b"xlsx")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            self.manager.start_job.call_args.kwargs["input_path"].suffix, ".xlsx"
        )

    def test_wait_and_failure(self):
        self.manager.start_job.side_effect = None
        self.manager.start_job.return_value = SimpleNamespace(
            job_id="pending", status="running"
        )
        self.manager.get_job.return_value = self.complete_job(
            input_path=self.root / "pending.png"
        )
        with patch.object(api.time, "sleep"):
            response = self.client.post(
                "/ingest", files={"file": ("scan.png", b"image")}
            )
        self.assertEqual(response.status_code, 200)
        self.manager.get_job.assert_called_once_with("pending", output_dir=self.root)
        self.manager.start_job.return_value = SimpleNamespace(
            job_id="failed", status="failed"
        )
        response = self.client.post("/ingest", files={"file": ("scan.png", b"image")})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"]["job_id"], "failed")

    def test_missing_markdown(self):
        self.manager.start_job.side_effect = None
        self.manager.start_job.return_value = SimpleNamespace(
            job_id="missing", status="completed", out_file=self.root / "absent.md"
        )
        with self.assertLogs(api.logger, level="ERROR"):
            response = self.client.post(
                "/ingest", files={"file": ("scan.png", b"image")}
            )
        self.assertEqual(response.status_code, 500)

    def test_docs_and_file_only_schema(self):
        for route in ["/", "/docs", "/redoc", "/plan"]:
            self.assertEqual(self.client.get(route).status_code, 200)
        plan = self.client.get("/plan").json()
        self.assertEqual(plan["input"]["field"], "file")
        self.manager.start_job.assert_not_called()
        schema = self.client.get("/openapi.json").json()
        operation = schema["paths"]["/ingest"]["post"]
        self.assertEqual(operation.get("parameters", []), [])
        ref = operation["requestBody"]["content"]["multipart/form-data"]["schema"][
            "$ref"
        ]
        body = schema["components"]["schemas"][ref.rsplit("/", 1)[-1]]
        self.assertEqual(set(body["properties"]), {"file"})
        self.assertEqual(body["required"], ["file"])
        self.assertEqual(
            set(operation["responses"]["200"]["content"]), {"application/zip"}
        )


class WorkspaceApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

        output_patch = patch.object(api, "OUTPUT_DIR", self.root)
        output_patch.start()
        self.addCleanup(output_patch.stop)

        manager_patch = patch.object(api.JobManager, "get_instance")
        self.manager = manager_patch.start().return_value
        self.addCleanup(manager_patch.stop)

        self.jobs = {}
        self.manager.start_job.side_effect = self.fake_start_job
        self.manager.get_job.side_effect = self.fake_get_job
        self.manager.get_latest_logs.return_value = "Log baris 1\nLog baris 2"
        self.manager.cancel_job.return_value = True
        self.manager.restart_job.side_effect = self.fake_restart_job

        self.client = TestClient(api.app)
        self.addCleanup(self.client.close)

    def fake_start_job(
        self,
        input_path: Path,
        output_dir: Path,
        queue_position: int = 0,
        **kwargs,
    ):
        stem = input_path.stem
        doc_dir = output_dir / stem
        doc_dir.mkdir(parents=True, exist_ok=True)
        md = doc_dir / f"{stem}.md"
        if not md.exists():
            md.write_text(
                f"# Judul {stem}\n\n<!-- PAGE: 1 -->\nIsi halaman satu.\n\n<!-- PAGE: 2 -->\nIsi halaman dua.",
                encoding="utf-8",
            )
        job = SimpleNamespace(
            job_id=stem,
            file_name=input_path.name,
            status="completed",
            stage="Selesai",
            progress_percentage=lambda: 100.0,
            current_page=2,
            total_pages=2,
            queue_position=queue_position,
            started_at="2026-10-02T10:00:00",
            updated_at="2026-10-02T10:01:00",
            completed_at="2026-10-02T10:01:00",
            error_message=None,
            last_message="Ekstraksi sukses.",
            extraction_options={},
            out_file=md,
            db_file=None,
        )
        self.jobs[stem] = job
        return job

    def fake_get_job(self, stem: str, output_dir: Path | None = None):
        return self.jobs.get(stem)

    def fake_restart_job(self, stem: str, output_dir: Path):
        return self.fake_start_job(
            output_dir / "uploads" / f"{stem}.pdf", output_dir
        )

    def test_batch_upload_single_file(self):
        response = self.client.post(
            "/batches/upload",
            files={"files": ("laporan.pdf", b"%PDF-1.4 mock content")},
            data={"batch_name": "Batch Laporan"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["batch_name"], "Batch Laporan")
        self.assertTrue(payload["batch_id"])
        self.assertIn("laporan", payload["document_stems"])
        self.assertEqual(payload["job_ids"], ["laporan"])
        self.assertEqual(len(payload["documents"]), 1)
        self.assertTrue((self.root / "uploads" / "laporan.pdf").is_file())
        self.assertTrue(
            (self.root / "batches" / payload["batch_id"] / "manifest.json").is_file()
        )

    def test_batch_upload_multiple_files(self):
        response = self.client.post(
            "/batches/upload",
            files=[
                ("files", ("doc1.pdf", b"%PDF-1.4 file 1")),
                ("files", ("doc2.docx", b"docx file 2")),
            ],
            data={"batch_name": "Multi Batch"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["documents"]), 2)
        self.assertEqual(set(payload["document_stems"]), {"doc1", "doc2"})
        self.assertEqual(set(payload["job_ids"]), {"doc1", "doc2"})

    def test_batch_upload_validation(self):
        # Format tidak didukung
        res_unsupported = self.client.post(
            "/batches/upload", files={"files": ("bad.exe", b"malware")}
        )
        self.assertEqual(res_unsupported.status_code, 415)

        # File kosong
        res_empty = self.client.post(
            "/batches/upload", files={"files": ("empty.pdf", b"")}
        )
        self.assertEqual(res_empty.status_code, 400)

        # File melebihi batas ukuran
        with patch.object(api, "MAX_UPLOAD_BYTES", 3):
            res_toolarge = self.client.post(
                "/batches/upload", files={"files": ("large.pdf", b"123456")}
            )
            self.assertEqual(res_toolarge.status_code, 413)

        # Tanpa file sama sekali
        res_nofiles = self.client.post("/batches/upload")
        self.assertEqual(res_nofiles.status_code, 400)

    def test_batch_upload_auto_index_rag(self):
        with patch.object(api, "_schedule_auto_rag_indexing") as mock_auto:
            response = self.client.post(
                "/batches/upload",
                files={"files": ("auto.pdf", b"%PDF-1.4 auto")},
                data={"auto_index_rag": "true"},
            )
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["auto_index_rag"])
            mock_auto.assert_called_once_with("auto", self.root)

    def test_batches_list_and_detail(self):
        batch = create_batch(
            self.root,
            "Arsip Dokumen",
            [{"stem": "doc_a", "source_name": "doc_a.pdf"}],
        )
        self.fake_start_job(self.root / "uploads" / "doc_a.pdf", self.root)

        # GET /batches
        res_list = self.client.get("/batches")
        self.assertEqual(res_list.status_code, 200)
        batches_data = res_list.json()
        self.assertTrue(any(b["id"] == batch["id"] for b in batches_data))
        found = next(b for b in batches_data if b["id"] == batch["id"])
        self.assertEqual(found["name"], "Arsip Dokumen")
        self.assertEqual(len(found["documents"]), 1)
        self.assertEqual(found["documents"][0]["stem"], "doc_a")
        self.assertEqual(found["documents"][0]["status"], "completed")

        # GET /batches/{batch_id}
        res_detail = self.client.get(f"/batches/{batch['id']}")
        self.assertEqual(res_detail.status_code, 200)
        detail_data = res_detail.json()
        self.assertEqual(detail_data["id"], batch["id"])

        # Nonexistent batch
        res_missing = self.client.get("/batches/nonexistent_batch_id")
        self.assertEqual(res_missing.status_code, 404)

    def test_delete_batch(self):
        batch = create_batch(
            self.root,
            "Batch Hapus",
            [{"stem": "del_doc", "source_name": "del_doc.pdf"}],
        )
        batch_id = batch["id"]
        self.assertTrue((self.root / "batches" / batch_id).exists())

        res = self.client.delete(f"/batches/{batch_id}")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "deleted")
        self.assertFalse((self.root / "batches" / batch_id).exists())

        # Hapus batch yang sudah tidak ada
        res_404 = self.client.delete(f"/batches/{batch_id}")
        self.assertEqual(res_404.status_code, 404)

    def test_job_status_and_actions(self):
        job = self.fake_start_job(self.root / "uploads" / "job_doc.pdf", self.root)
        res_job = self.client.get("/jobs/job_doc")
        self.assertEqual(res_job.status_code, 200)
        data = res_job.json()
        self.assertEqual(data["job_id"], "job_doc")
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["log_snippet"], "Log baris 1\nLog baris 2")

        # 404 untuk job tidak dikenal
        res_unknown = self.client.get("/jobs/ghost_job")
        self.assertEqual(res_unknown.status_code, 404)

        # Batalkan job (saat running)
        job.status = "running"
        res_cancel = self.client.post("/jobs/job_doc/cancel")
        self.assertEqual(res_cancel.status_code, 200)
        self.assertEqual(res_cancel.json()["status"], "canceled")

        # Batalkan saat sudah selesai -> 400
        job.status = "completed"
        res_cancel_bad = self.client.post("/jobs/job_doc/cancel")
        self.assertEqual(res_cancel_bad.status_code, 400)

        # Batalkan job yang tidak ada -> 404
        res_cancel_404 = self.client.post("/jobs/ghost/cancel")
        self.assertEqual(res_cancel_404.status_code, 404)

        # Retry job
        res_retry = self.client.post("/jobs/job_doc/retry")
        self.assertEqual(res_retry.status_code, 200)
        self.assertEqual(res_retry.json()["status"], "completed")

    def test_document_summary_and_markdown(self):
        doc_dir = self.root / "my_doc"
        doc_dir.mkdir(parents=True, exist_ok=True)
        md_path = doc_dir / "my_doc.md"
        md_path.write_text("# Catatan Penting\nParagraf teks.", encoding="utf-8")

        # Summary
        res_sum = self.client.get("/documents/my_doc")
        self.assertEqual(res_sum.status_code, 200)
        summary = res_sum.json()
        self.assertEqual(summary["doc_stem"], "my_doc")
        self.assertTrue(summary["markdown_exists"])
        self.assertEqual(summary["table_count"], 0)
        self.assertFalse(summary["rag_indexed"])

        # Markdown
        res_md = self.client.get("/documents/my_doc/markdown")
        self.assertEqual(res_md.status_code, 200)
        self.assertEqual(res_md.text, "# Catatan Penting\nParagraf teks.")
        self.assertIn("text/markdown", res_md.headers["content-type"])

        # 404 checks
        self.assertEqual(self.client.get("/documents/unknown").status_code, 404)
        self.assertEqual(
            self.client.get("/documents/unknown/markdown").status_code, 404
        )

    def test_document_preview(self):
        doc_dir = self.root / "prev_doc"
        doc_dir.mkdir(parents=True, exist_ok=True)
        md_text = (
            "Pengantar dokumen umum.\n\n"
            "<!-- PAGE: 1 -->\n"
            "# Bagian 1\n"
            "Isi bagian satu.\n\n"
            "<!-- PAGE: 2 -->\n"
            "# Bagian 2\n"
            "| No | Nama |\n"
            "|---|---|\n"
            "| 1 | Budi |\n"
        )
        (doc_dir / "prev_doc.md").write_text(md_text, encoding="utf-8")

        response = self.client.get(
            "/documents/prev_doc/preview?chunk_size=500&chunk_overlap=50"
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["doc_stem"], "prev_doc")
        self.assertEqual(data["preamble"], "Pengantar dokumen umum.")
        self.assertEqual(data["total_pages"], 2)
        self.assertEqual(len(data["page_summaries"]), 2)
        self.assertTrue(data["page_summaries"][1]["has_table"])
        self.assertIn("chunks", data["chunking_preview"])

        self.assertEqual(
            self.client.get("/documents/unknown/preview").status_code, 404
        )

    def test_document_tables_and_rows(self):
        doc_dir = self.root / "table_doc"
        doc_dir.mkdir(parents=True, exist_ok=True)
        (doc_dir / "table_doc.md").write_text("# Tabel", encoding="utf-8")
        db_dir = doc_dir / "databases"
        db_dir.mkdir(parents=True, exist_ok=True)
        db_path = db_dir / "table_doc.sqlite"
        with closing(sqlite3.connect(db_path)) as conn, conn:
            conn.execute(
                "CREATE TABLE transaksi (id INTEGER PRIMARY KEY, item TEXT, total REAL)"
            )
            conn.execute("INSERT INTO transaksi VALUES (1, 'Buku', 50000)")
            conn.execute("INSERT INTO transaksi VALUES (2, 'Pena', 5000)")
            conn.execute("INSERT INTO transaksi VALUES (3, 'Kertas', 20000)")

        # List tables
        res_tables = self.client.get("/documents/table_doc/tables")
        self.assertEqual(res_tables.status_code, 200)
        tbl_data = res_tables.json()
        self.assertEqual(tbl_data["total_tables"], 1)
        self.assertEqual(tbl_data["tables"][0]["name"], "transaksi")
        self.assertEqual(tbl_data["tables"][0]["row_count"], 3)
        self.assertEqual(
            tbl_data["tables"][0]["columns"], ["id", "item", "total"]
        )

        # Query rows with limit and offset
        res_rows = self.client.get(
            "/documents/table_doc/tables/transaksi?limit=2&offset=0"
        )
        self.assertEqual(res_rows.status_code, 200)
        rows_data = res_rows.json()
        self.assertEqual(rows_data["total_rows"], 3)
        self.assertEqual(len(rows_data["rows"]), 2)
        self.assertEqual(rows_data["rows"][0]["item"], "Buku")

        res_rows2 = self.client.get(
            "/documents/table_doc/tables/transaksi?limit=2&offset=2"
        )
        self.assertEqual(res_rows2.status_code, 200)
        self.assertEqual(len(res_rows2.json()["rows"]), 1)
        self.assertEqual(res_rows2.json()["rows"][0]["item"], "Kertas")

        # 404 for unknown table or doc
        self.assertEqual(
            self.client.get(
                "/documents/table_doc/tables/unknown"
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client.get("/documents/ghost/tables").status_code, 404
        )

    def test_document_page_image(self):
        doc_dir = self.root / "img_doc"
        pages_dir = doc_dir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        sample_png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00"
            b"\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        (pages_dir / "page_1.png").write_bytes(sample_png)

        # Valid page image
        res_img = self.client.get("/documents/img_doc/pages/1/image")
        self.assertEqual(res_img.status_code, 200)
        self.assertEqual(res_img.headers["content-type"], "image/png")
        self.assertEqual(res_img.content, sample_png)

        # Missing page
        self.assertEqual(
            self.client.get("/documents/img_doc/pages/2/image").status_code,
            404,
        )
        # Invalid page number < 1
        self.assertEqual(
            self.client.get("/documents/img_doc/pages/0/image").status_code,
            400,
        )

    def test_document_download_zip(self):
        doc_dir = self.root / "zip_doc"
        doc_dir.mkdir(parents=True, exist_ok=True)
        (doc_dir / "zip_doc.md").write_text("# Isi dokumen zip", encoding="utf-8")

        res_zip = self.client.get("/documents/zip_doc/download")
        self.assertEqual(res_zip.status_code, 200)
        self.assertEqual(res_zip.headers["content-type"], "application/zip")
        with ZipFile(io.BytesIO(res_zip.content)) as archive:
            self.assertIn("PETUNJUK.txt", archive.namelist())
            self.assertIn("zip_doc.md", archive.namelist())

        self.assertEqual(
            self.client.get("/documents/unknown_zip/download").status_code, 404
        )

    def test_document_manual_index_rag(self):
        doc_dir = self.root / "rag_doc"
        doc_dir.mkdir(parents=True, exist_ok=True)
        (doc_dir / "rag_doc.md").write_text(
            "# Dokumen Kebijakan\n\nPasal 1: Setiap pengguna harus memiliki akun valid.\n\nPasal 2: Akses diberikan sesuai hak istimewa.",
            encoding="utf-8",
        )
        pages_dir = doc_dir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        (pages_dir / "page_1.png").write_bytes(b"mock_png_bytes")

        res_idx = self.client.post(
            "/documents/rag_doc/index-rag",
            json={"chunk_size": 200, "chunk_overlap": 20},
        )
        self.assertEqual(res_idx.status_code, 200)
        payload = res_idx.json()
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["doc_stem"], "rag_doc")
        self.assertGreater(payload["total_chunks"], 0)
        self.assertTrue(payload["persist_directory"])

        # Document summary now reflects rag_indexed
        res_sum = self.client.get("/documents/rag_doc")
        self.assertEqual(res_sum.status_code, 200)
        self.assertTrue(res_sum.json()["rag_indexed"])

        # Nonexistent doc
        self.assertEqual(
            self.client.post("/documents/ghost_doc/index-rag").status_code, 404
        )

    def test_batch_upload_partial_failure_cleanup(self):
        # File 1 valid, file 2 invalid -> upload should clean up file 1 from disk
        res = self.client.post(
            "/batches/upload",
            files=[
                ("files", ("valid.pdf", b"%PDF-1.4 content")),
                ("files", ("malware.exe", b"binary")),
            ],
        )
        self.assertEqual(res.status_code, 415)
        self.assertFalse((self.root / "uploads" / "valid.pdf").exists())

    def test_page_number_extraction_with_complex_filenames(self):
        doc_dir = self.root / "complex_doc"
        pages_dir = doc_dir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        sample_png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00"
            b"\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        # Filename has year 2025 before the page number
        (pages_dir / "laporan_2025_page_0001.png").write_bytes(sample_png)
        (pages_dir / "laporan_2025_page_0002.png").write_bytes(sample_png)
        (doc_dir / "complex_doc.md").write_text(
            "<!-- PAGE: 1 -->\nHal 1\n<!-- PAGE: 2 -->\nHal 2", encoding="utf-8"
        )

        res_p1 = self.client.get("/documents/complex_doc/pages/1/image")
        self.assertEqual(res_p1.status_code, 200)

        res_p2 = self.client.get("/documents/complex_doc/pages/2/image")
        self.assertEqual(res_p2.status_code, 200)

        res_p3 = self.client.get("/documents/complex_doc/pages/3/image")
        self.assertEqual(res_p3.status_code, 404)

    def test_sheet_previews_image_and_summary(self):
        doc_dir = self.root / "sheet_doc"
        sheet_dir = doc_dir / "sheet_previews"
        sheet_dir.mkdir(parents=True, exist_ok=True)
        sample_png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00"
            b"\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        (sheet_dir / "sheet_1.png").write_bytes(sample_png)
        (doc_dir / "sheet_doc.md").write_text("# Sheet Excel", encoding="utf-8")

        res_img = self.client.get("/documents/sheet_doc/pages/1/image")
        self.assertEqual(res_img.status_code, 200)

        res_sum = self.client.get("/documents/sheet_doc")
        self.assertEqual(res_sum.status_code, 200)
        self.assertTrue(res_sum.json()["has_images"])

    def test_chunk_validation_endpoints(self):
        doc_dir = self.root / "val_doc"
        doc_dir.mkdir(parents=True, exist_ok=True)
        (doc_dir / "val_doc.md").write_text("# Text", encoding="utf-8")

        # Preview: overlap >= size -> 400
        res_prev = self.client.get(
            "/documents/val_doc/preview?chunk_size=100&chunk_overlap=120"
        )
        self.assertEqual(res_prev.status_code, 400)

        # Index RAG: overlap >= size -> 400
        res_idx_bad = self.client.post(
            "/documents/val_doc/index-rag",
            json={"chunk_size": 100, "chunk_overlap": 100},
        )
        self.assertEqual(res_idx_bad.status_code, 400)

        # Negative overlap -> 400
        res_idx_neg = self.client.post(
            "/documents/val_doc/index-rag",
            json={"chunk_size": 100, "chunk_overlap": -5},
        )
        self.assertEqual(res_idx_neg.status_code, 400)

        # Non-positive chunk_size -> 400
        res_idx_zero = self.client.post(
            "/documents/val_doc/index-rag",
            json={"chunk_size": 0, "chunk_overlap": 10},
        )
        self.assertEqual(res_idx_zero.status_code, 400)

        # /rag/index endpoint validation
        res_rag_bad = self.client.post(
            "/rag/index",
            json={
                "markdown_text": "# Content",
                "doc_stem": "val_doc",
                "chunk_size": 100,
                "chunk_overlap": 150,
            },
        )
        self.assertEqual(res_rag_bad.status_code, 400)

    def test_tables_sqlite_internal_and_blob_serialization(self):
        doc_dir = self.root / "blob_doc"
        doc_dir.mkdir(parents=True, exist_ok=True)
        (doc_dir / "blob_doc.md").write_text("# Blob", encoding="utf-8")
        db_path = doc_dir / "blob_doc.sqlite"
        with closing(sqlite3.connect(db_path)) as conn, conn:
            conn.execute(
                "CREATE TABLE raw_data (id INTEGER PRIMARY KEY, payload BLOB)"
            )
            conn.execute(
                "INSERT INTO raw_data VALUES (1, ?)", (b"binary_payload_data",)
            )

        # Query internal sqlite table -> 404
        self.assertEqual(
            self.client.get("/documents/blob_doc/tables/sqlite_master").status_code,
            404,
        )

        # Query table with BLOB -> 200 with sanitized string, no 500 error
        res_blob = self.client.get("/documents/blob_doc/tables/raw_data")
        self.assertEqual(res_blob.status_code, 200)
        data = res_blob.json()
        self.assertEqual(data["total_rows"], 1)
        self.assertEqual(data["rows"][0]["payload"], "binary_payload_data")

    def test_delete_batch_invalid_batch_id(self):
        # Nonexistent batch id should return 404
        res = self.client.delete("/batches/invalid_id_not_found")
        self.assertEqual(res.status_code, 404)

        # When delete_batch raises ValueError (e.g. invalid path), API returns 404 instead of 500
        with (
            patch.object(api, "delete_batch", side_effect=ValueError("Identitas batch tidak valid")),
            patch.object(api, "list_batches", return_value=[{"id": "bad_id", "name": "bad_batch"}]),
        ):
            res_val = self.client.delete("/batches/bad_id")
            self.assertEqual(res_val.status_code, 404)

    def test_enriched_batch_with_untracked_completed_markdown(self):
        batch = create_batch(
            self.root,
            "Old Batch",
            [{"stem": "old_doc", "source_name": "old_doc.pdf"}],
        )
        doc_dir = self.root / "old_doc"
        doc_dir.mkdir(parents=True, exist_ok=True)
        (doc_dir / "old_doc.md").write_text("# Selesai", encoding="utf-8")

        res = self.client.get(f"/batches/{batch['id']}")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "Selesai")
        self.assertEqual(data["documents"][0]["status"], "completed")
        self.assertEqual(data["documents"][0]["progress_pct"], 100.0)

