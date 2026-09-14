import csv
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

from fastapi.testclient import TestClient
from app import api


class IngestApiTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        output_patch = patch.object(api, 'OUTPUT_DIR', self.root)
        output_patch.start()
        self.addCleanup(output_patch.stop)
        manager_patch = patch.object(api.JobManager, 'get_instance')
        self.manager = manager_patch.start().return_value
        self.addCleanup(manager_patch.stop)
        self.manager.start_job.side_effect = self.complete_job
        self.client = TestClient(api.app)
        self.addCleanup(self.client.close)

    def complete_job(self, **kwargs):
        path = kwargs['input_path']
        md = self.root / f'{path.stem}.md'
        md.write_text('# Hasil\nTeks.', encoding='utf-8')
        db = self.root / f'{path.stem}.sqlite'
        with sqlite3.connect(db) as conn:
            conn.execute('CREATE TABLE "data/item" (nama TEXT, jumlah INTEGER)')
            conn.execute('INSERT INTO "data/item" VALUES (?, ?)', ('Kopi, "A"\nBaru', 2))
        return SimpleNamespace(job_id=path.stem, status='completed', out_file=md, db_file=db)

    def test_zip_sql_roundtrip_and_csv(self):
        response = self.client.post('/ingest', files={'file': ('scan.png', b'image')})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['content-type'], 'application/zip')
        with ZipFile(io.BytesIO(response.content)) as archive:
            self.assertEqual(set(archive.namelist()), {'document.md', 'document.sql', 'csv/data_item.csv'})
            self.assertTrue(archive.read('document.md').decode().startswith('# Hasil'))
            with sqlite3.connect(':memory:') as conn:
                conn.executescript(archive.read('document.sql').decode())
                self.assertEqual(conn.execute('SELECT jumlah FROM "data/item"').fetchone(), (2,))
            rows = list(csv.reader(io.StringIO(archive.read('csv/data_item.csv').decode('utf-8-sig'))))
            self.assertEqual(rows, [['nama', 'jumlah'], ['Kopi, "A"\nBaru', '2']])
        self.assertEqual(set(self.manager.start_job.call_args.kwargs), {'input_path', 'output_dir'})

    def test_no_database(self):
        def without_db(**kwargs):
            job = self.complete_job(**kwargs)
            job.db_file.unlink()
            return job
        self.manager.start_job.side_effect = without_db
        response = self.client.post('/ingest', files={'file': ('scan.png', b'image')})
        self.assertEqual(response.status_code, 200)
        with ZipFile(io.BytesIO(response.content)) as archive:
            self.assertEqual(set(archive.namelist()), {'document.md', 'document.sql'})
            self.assertEqual(archive.read('document.sql'), b'BEGIN TRANSACTION;\nCOMMIT;\n')

    def test_validation(self):
        for filename, content, status in [('x.exe', b'x', 415), ('x.pdf', b'', 400)]:
            self.assertEqual(self.client.post('/ingest', files={'file': (filename, content)}).status_code, status)
        with patch.object(api, 'MAX_UPLOAD_BYTES', 2):
            self.assertEqual(self.client.post('/ingest', files={'file': ('x.pdf', b'123')}).status_code, 413)
        self.assertEqual(self.client.post('/ingest').status_code, 422)
        self.manager.start_job.assert_not_called()
        self.assertEqual(list((self.root / 'uploads').iterdir()), [])

    def test_safe_unique_uploads(self):
        for _ in range(2):
            response = self.client.post('/ingest', files={'file': ('../../scan.PDF', b'pdf')})
            self.assertEqual(response.status_code, 200)
        paths = [call.kwargs['input_path'] for call in self.manager.start_job.call_args_list]
        self.assertNotEqual(paths[0], paths[1])
        self.assertTrue(all(p.parent == self.root / 'uploads' for p in paths))

    def test_wait_and_failure(self):
        self.manager.start_job.side_effect = None
        self.manager.start_job.return_value = SimpleNamespace(job_id='pending', status='running')
        self.manager.get_job.return_value = self.complete_job(input_path=self.root / 'pending.png')
        with patch.object(api.time, 'sleep'):
            response = self.client.post('/ingest', files={'file': ('scan.png', b'image')})
        self.assertEqual(response.status_code, 200)
        self.manager.get_job.assert_called_once_with('pending', output_dir=self.root)
        self.manager.start_job.return_value = SimpleNamespace(job_id='failed', status='failed')
        response = self.client.post('/ingest', files={'file': ('scan.png', b'image')})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()['detail']['job_id'], 'failed')

    def test_missing_markdown(self):
        self.manager.start_job.side_effect = None
        self.manager.start_job.return_value = SimpleNamespace(job_id='missing', status='completed', out_file=self.root / 'absent.md')
        with self.assertLogs(api.logger, level='ERROR'):
            response = self.client.post('/ingest', files={'file': ('scan.png', b'image')})
        self.assertEqual(response.status_code, 500)

    def test_docs_and_file_only_schema(self):
        for route in ['/', '/docs', '/redoc', '/plan']:
            self.assertEqual(self.client.get(route).status_code, 200)
        plan = self.client.get('/plan').json()
        self.assertEqual(plan['input']['field'], 'file')
        self.manager.start_job.assert_not_called()
        schema = self.client.get('/openapi.json').json()
        operation = schema['paths']['/ingest']['post']
        self.assertEqual(operation.get('parameters', []), [])
        ref = operation['requestBody']['content']['multipart/form-data']['schema']['$ref']
        body = schema['components']['schemas'][ref.rsplit('/', 1)[-1]]
        self.assertEqual(set(body['properties']), {'file'})
        self.assertEqual(body['required'], ['file'])
        self.assertEqual(set(operation['responses']['200']['content']), {'application/zip'})
