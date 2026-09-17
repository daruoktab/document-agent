"""Regression coverage for persistent batches and history of early failures."""
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

from app.job_tracker import JobManager
from app.streamlit_logic import _save_uploaded_files, build_batch_zip
from app.upload_batches import create_batch, list_batches


class UploadedFileStub:
    def __init__(self, name: str, content: bytes) -> None:
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


class TestUploadBatches(unittest.TestCase):
    def test_same_filename_gets_ordinal_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = UploadedFileStub("report.pdf", b"one")
            second = UploadedFileStub("report.pdf", b"two")
            same = UploadedFileStub("report.pdf", b"one")
            self.assertEqual(_save_uploaded_files([first], root)[0].name, "report.pdf")
            self.assertEqual(_save_uploaded_files([second], root)[0].name, "report (1).pdf")
            self.assertEqual(_save_uploaded_files([same], root)[0].name, "report.pdf")

    def test_same_batch_name_gets_ordinal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = create_batch(root, "Folder", [{"stem": "one", "source_name": "one.pdf"}])
            second = create_batch(root, "Folder", [{"stem": "two", "source_name": "two.pdf"}])
            self.assertEqual(first["name"], "Folder")
            self.assertEqual(second["name"], "Folder (1)")

    def test_empty_batch_name_defaults_to_uploaded_files_and_records_upload_time(self):
        with tempfile.TemporaryDirectory() as directory:
            batch = create_batch(Path(directory), "", [{"stem": "one", "source_name": "one.pdf"}])
            self.assertEqual(batch["name"], "Uploaded files")
            self.assertEqual(batch["uploaded_at"], batch["created_at"])

    def test_nested_uploads_deduplicate_and_batch_survives_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uploads = [
                UploadedFileStub(name, content)
                for name, content in [
                    ("a/report.pdf", b"one"),
                    ("a/nested/report.pdf", b"two"),
                    ("b/report.pdf", b"one"),
                ]
            ]
            paths = _save_uploaded_files(uploads, root)
            self.assertEqual(len(paths), 2)
            batch = create_batch(root, "Folder reports", [{"stem": p.stem, "source_name": p.name} for p in paths])
            self.assertEqual(list_batches(root), [batch])
            for index, path in enumerate(paths):
                doc = root / path.stem
                doc.mkdir()
                (doc / "report.md").write_text(str(index))
                (doc / "csv").mkdir()
                (doc / "csv" / "table.csv").write_text("a,b\n1,2\n")
            with ZipFile(BytesIO(build_batch_zip(list_batches(root)[0], root))) as archive:
                for index, path in enumerate(paths):
                    self.assertEqual(archive.read(f"{path.stem}/report.md"), str(index).encode())
                    self.assertIn(f"{path.stem}/csv/table.csv", archive.namelist())
                self.assertIn("batch.json", archive.namelist())

    def test_failed_before_output_is_visible_in_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "failed_doc" / "logs"
            log.mkdir(parents=True)
            (log / "failed_doc_status.json").write_text(json.dumps({"status": "failed"}))
            manager = JobManager()
            with patch.object(manager, "get_job", return_value=SimpleNamespace(status="failed", stage="Gagal")):
                docs = manager.list_all_documents(root)
            self.assertEqual([d["stem"] for d in docs], ["failed_doc"])
            self.assertEqual(docs[0]["status"], "failed")

    def test_bad_manifest_does_not_hide_valid_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = create_batch(root, "Valid", [{"stem": "report", "source_name": "report.pdf"}])
            bad = root / "batches" / "bad"
            bad.mkdir()
            (bad / "manifest.json").write_text("invalid json")
            self.assertEqual(list_batches(root), [batch])

    def test_batch_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            build_batch_zip({"documents": [{"stem": "../outside"}]}, Path(directory))
