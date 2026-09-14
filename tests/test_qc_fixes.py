"""Regresi untuk kasus yang direproduksi dalam QC agent dan UI."""
import csv
import sqlite3
import tempfile
import unittest
from io import BytesIO, StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch
from zipfile import ZipFile

from app.batch import batch_extract_documents
from app.config import Settings
from app.graph import DocumentExtractionPipeline
from app.job_tracker import JobManager
from app.streamlit_logic import (
    _save_uploaded_files,
    build_all_tables_csv_zip,
    build_sqlite_download,
    table_csv_bytes,
)
from app.tabular_db import (
    TabularDatabaseManager,
    cross_verify_dual_track,
    process_page_tabular_agent,
    prune_document_pages,
)

MD = '| Tanggal | Keterangan | Debit | Kredit | Saldo |\n|---|---|---|---|---|\n| 2025-01-01 | Belanja | 100 | 0 | 900 |'


class TestQCFixes(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / 'data.sqlite'

    def ingest(self, md=MD, page=1, source='doc.pdf', **kwargs):
        return process_page_tabular_agent(md, page, source, self.db, **kwargs)

    def test_audit_detects_changed_numeric_and_text_values(self):
        self.ingest()
        self.assertEqual(cross_verify_dual_track(MD, self.db).guardrail_status, 'PASSED')
        for column, value in [('debit', 999999), ('description', 'Wrong'), ('txn_date', '2026-01-01')]:
            with self.subTest(column=column):
                self.ingest()
                with sqlite3.connect(self.db) as conn:
                    conn.execute(f'UPDATE transaction_details SET {column}=?', (value,))
                self.assertEqual(cross_verify_dual_track(MD, self.db).guardrail_status, 'WARNING')

    def test_page_replacement_preserves_other_pages_and_sources(self):
        self.ingest()
        self.ingest(page=2)
        self.ingest(source='other.pdf')
        self.ingest(MD.replace('100', '200'))
        with sqlite3.connect(self.db) as conn:
            rows = conn.execute('SELECT _source_doc, _page_number, debit FROM transaction_details ORDER BY 1,2').fetchall()
        self.assertEqual(rows, [('doc.pdf', 1, 200), ('doc.pdf', 2, 100), ('other.pdf', 1, 100)])
        self.ingest('No table')
        with sqlite3.connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM transaction_details').fetchone()[0], 2)

    def test_page_replacement_rolls_back_on_failure(self):
        self.ingest()
        with patch.object(TabularDatabaseManager, 'ingest_relational_transactions', side_effect=RuntimeError('failure')), self.assertRaises(RuntimeError):
            self.ingest(MD.replace('100', '200'))
        with sqlite3.connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT debit FROM transaction_details').fetchall(), [(100,)])

    def test_removed_pages_are_pruned_only_for_selected_document(self):
        self.ingest()
        self.ingest(page=2)
        self.ingest(page=3, source='other.pdf')
        prune_document_pages(self.db, 'doc.pdf', 1)
        with sqlite3.connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT _source_doc, _page_number FROM transaction_details ORDER BY 1,2').fetchall(), [('doc.pdf', 1), ('other.pdf', 3)])

    def test_generic_table_replacement(self):
        md = '| Produk | Warna |\n|---|---|\n| Kursi | Biru |'
        self.ingest(md, force_all_tables=True)
        self.ingest(md.replace('Biru', 'Merah'), force_all_tables=True)
        with sqlite3.connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT warna FROM doc_t1').fetchall(), [('Merah',)])

    def test_upload_collisions_and_stable_reruns(self):
        uploads = [SimpleNamespace(name=name, getvalue=lambda data=data: data) for name, data in
                   [('image.png', b'A'), ('image.png', b'B'), ('image.jpg', b'C')]]
        paths = _save_uploaded_files(cast(Any, uploads), self.root)
        self.assertEqual(len({p.stem for p in paths}), 3)
        self.assertEqual([p.read_bytes() for p in paths], [b'A', b'B', b'C'])
        self.assertEqual(paths, _save_uploaded_files(cast(Any, uploads), self.root))

    def test_retry_starts_worker_and_preserves_options(self):
        source = self.root / 'image.png'
        source.touch()
        with patch.object(JobManager, '_jobs', {}), patch.object(JobManager, '_processes', {}), patch.object(JobManager, '_threads', {}), patch('app.job_tracker.threading.Thread.start') as start:
            manager = JobManager()
            old = manager.start_job(source, self.root, dpi=250, force_all_tables=True)
            old.out_file.write_text('Partial')
            old.status = 'failed'
            new = manager.restart_job('image', self.root)
            self.assertEqual(start.call_count, 2)
            self.assertEqual(new.status, 'running')
            self.assertEqual(new.extraction_options['dpi'], 250)
            self.assertTrue(new.extraction_options['force_all_tables'])

    def test_mermaid_specialist_recovers_invalid_draft(self):
        pipeline = object.__new__(DocumentExtractionPipeline)
        pipeline.thorough = True
        pipeline.extractor = cast(Any, SimpleNamespace(judge_and_refine=lambda **kw: kw['draft_markdown']))
        state = {'image_path': 'unused', 'markdown_content': '# Flow\n```mermaid\nbroken\n```',
                 'diagram_mermaid_code': 'flowchart TD\n A["Start"] --> B["End"]'}
        result = pipeline._node_aggregate_and_judge(cast(Any, state))
        self.assertIn('A["Start"] --> B["End"]', result['markdown_content'])

    def test_csv_full_table_zip_and_sqlite_snapshot(self):
        with sqlite3.connect(self.db) as conn:
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('CREATE TABLE entries (label TEXT)')
            conn.executemany('INSERT INTO entries VALUES (?)', [(f'row_{i}',) for i in range(150)])
            conn.execute('CREATE TABLE empty_table (name TEXT)')
            conn.commit()
            data = table_csv_bytes(conn, 'entries')
            self.assertEqual(len(list(csv.reader(StringIO(data.decode('utf-8-sig'))))), 151)
            with ZipFile(BytesIO(build_all_tables_csv_zip(conn, ['entries', 'empty_table']))) as archive:
                self.assertEqual(archive.read('entries.csv'), data)
                self.assertEqual(archive.read('empty_table.csv').decode('utf-8-sig').strip(), 'name')
            snapshot = sqlite3.connect(':memory:')
            try:
                snapshot.deserialize(build_sqlite_download(self.db))
                self.assertEqual(snapshot.execute('SELECT COUNT(*) FROM entries').fetchone()[0], 150)
            finally:
                snapshot.close()

    def test_batch_failure_status(self):
        source = self.root / 'bad.png'
        source.touch()
        with patch('app.batch.DocumentExtractionPipeline') as pipeline:
            pipeline.return_value.run.side_effect = RuntimeError('model failed')
            result = batch_extract_documents([source], output_dir=self.root / 'batch', settings=Settings())
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['error_count'], 1)

    def test_cli_batch_failure_exit_code(self):
        import main
        source = self.root / 'bad.png'
        source.touch()
        with patch('sys.argv', ['main.py', '--batch', str(source)]), patch.object(main, 'migrate_legacy_output'), patch.object(main, 'setup_logging'), patch.object(main, 'get_settings', return_value=Settings()), patch.object(main, 'batch_extract_documents', return_value={'error_count': 1}):
            self.assertEqual(main.main(), 1)
