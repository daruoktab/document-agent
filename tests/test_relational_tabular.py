import tempfile
import unittest
from pathlib import Path

from app.schemas import (
    DocumentHeaderRecord,
)
from app.tabular_db import (
    TRANSACTION_CANONICAL_ALIASES,
    TabularDatabaseManager,
    defensive_map_columns,
    merge_and_deduplicate_tables,
    parse_date_value,
    process_page_tabular_agent,
)


class TestRelationalTabularDB(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "test_docs.sqlite"
        self.mgr = TabularDatabaseManager(self.db_path)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_relational_schema_creation(self):
        """Uji inisialisasi skema relasional document_headers, transaction_details, & document_signatories."""
        self.mgr.ensure_relational_schema()
        inspect = self.mgr.inspect_database()
        tables = inspect["tables"]
        self.assertIn("document_headers", tables)
        self.assertIn("transaction_details", tables)
        self.assertIn("document_signatories", tables)

    def test_defensive_mapping_and_schema_evolution(self):
        """Uji pemetaan defensif kolom model yang bervariasi dan penambahan kolom dinamis (evolusi skema)."""
        raw_headers = [
            "Tanggal Transaksi",
            "Uraian",
            "No. Ref",
            "Debet",
            "Kredit",
            "Saldo Akhir",
            "Channel Pembayaran",  # Kolom dinamis tambahan dari model
        ]
        is_valid, col_map, extra_cols = defensive_map_columns(
            raw_headers, TRANSACTION_CANONICAL_ALIASES, "transaction"
        )
        self.assertTrue(is_valid)
        self.assertIn("txn_date", col_map)
        self.assertIn("description", col_map)
        self.assertIn("debit", col_map)
        self.assertIn("credit", col_map)
        self.assertIn("balance", col_map)
        self.assertIn("channel_pembayaran", extra_cols)

    def test_parse_date_value_formats(self):
        """Uji konversi berbagai variasi format tanggal ke ISO YYYY-MM-DD secara presisi."""
        self.assertEqual(parse_date_value("15/08/2024"), "2024-08-15")
        self.assertEqual(parse_date_value("1/8/2024"), "2024-08-01")
        self.assertEqual(parse_date_value("05.12.2023"), "2023-12-05")
        self.assertEqual(parse_date_value("10/02/25"), "2025-02-10")
        self.assertEqual(parse_date_value("2024-05-20"), "2024-05-20")
        self.assertEqual(parse_date_value("20240201"), "2024-02-01")
        self.assertIsNone(parse_date_value("bukan_tanggal"))

    def test_header_detail_ingestion(self):
        """Uji alur Header-Detail: Header dibuat, lalu baris transaksi dihubungkan via header_id."""
        header = DocumentHeaderRecord(
            doc_type="bank_statement",
            doc_title="Rekening Koran Bank Mandiri",
            doc_number="123-456-7890",
            doc_date="2025-01-01 to 2025-01-31",
            parties="PT Solusi Maju Bersama",
            total_amount=50000000.0,
            currency="IDR",
            source_file="mandiri_jan.pdf",
            fingerprint_hash="fp_mandiri_001",
        )
        header_id = self.mgr.ingest_document_header(header)
        self.assertGreater(header_id, 0)

        # Ingest baris transaksi dengan FK header_id
        headers = ["Tanggal", "Keterangan", "No Ref", "Debit", "Kredit", "Saldo"]
        rows = [
            ["02/01/2025", "Setoran Modal", "REF001", "0", "50.000.000", "50.000.000"],
            ["05/01/2025", "Biaya Sewa Kantor", "REF002", "5.000.000", "0", "45.000.000"],
        ]
        inserted = self.mgr.ingest_relational_transactions(
            header_id=header_id,
            headers=headers,
            rows=rows,
            source_doc="mandiri_jan.pdf",
            page_number=1,
        )
        self.assertEqual(inserted, 2)

        # Query verifikasi FK relasi & parsing tanggal presisi
        res = self.mgr.execute_query(
            "SELECT t.*, h.doc_title, h.parties FROM transaction_details t "
            "JOIN document_headers h ON t.header_id = h.header_id WHERE t.header_id = ?;",
            (header_id,),
        )
        self.assertEqual(len(res.rows), 2)
        self.assertEqual(res.rows[0]["doc_title"], "Rekening Koran Bank Mandiri")
        self.assertEqual(res.rows[0]["txn_date"], "2025-01-02")
        self.assertEqual(res.rows[0]["balance"], 50000000.0)
        self.assertEqual(res.rows[1]["debit"], 5000000.0)

    def test_deduplication_on_insert(self):
        """Uji bahwa baris identik yang di-insert ulang diabaikan via _row_hash (INSERT OR IGNORE)."""
        header_id = self.mgr.ingest_document_header(
            DocumentHeaderRecord(doc_type="bank_statement", doc_number="111", fingerprint_hash="fp_111")
        )
        headers = ["Tanggal", "Keterangan", "Debit", "Kredit", "Saldo"]
        rows = [
            ["01/02/2025", "Pembayaran Gaji", "10000000", "0", "40000000"],
        ]
        # Insert pertama
        cnt1 = self.mgr.ingest_relational_transactions(header_id, headers, rows)
        self.assertEqual(cnt1, 1)

        # Insert kedua (data sama persis)
        cnt2 = self.mgr.ingest_relational_transactions(header_id, headers, rows)
        self.assertEqual(cnt2, 0)  # Diabaikan, tidak duplikat

        total = self.mgr.execute_query("SELECT COUNT(*) as c FROM transaction_details;").rows[0]["c"]
        self.assertEqual(total, 1)

    def test_multi_source_merge_and_deduplicate(self):
        """Uji merge_and_deduplicate_tables: menggabungkan field non-null dari dua sumber dan menghapus baris duplikat."""
        header_id = self.mgr.ingest_document_header(
            DocumentHeaderRecord(doc_type="bank_statement", doc_number="ACC-111", fingerprint_hash="fp_acc111")
        )

        # Sumber A: ada ref_no tapi info tambahan kosong
        self.mgr.ingest_relational_transactions(
            header_id=header_id,
            headers=["Tanggal", "Keterangan", "Ref No", "Debit", "Kredit", "Saldo"],
            rows=[["2025-01-10", "Transfer Bank", "TRX999", "100000", "0", "900000"]],
            source_doc="sumber_a.pdf",
        )

        # Tambah kolom dinamis 'catatan_audit' untuk simulasi sumber B
        self.mgr.ensure_columns_exist("transaction_details", {"catatan_audit": "TEXT"})

        # Sumber B: ref_no kosong tapi ada catatan_audit
        with self.mgr._get_connection() as conn:
            conn.execute(
                """INSERT INTO transaction_details (
                    header_id, txn_date, description, ref_no, debit, credit, balance, catatan_audit, _row_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);""",
                (header_id, "2025-01-10", "Transfer Bank", None, 100000, 0, 900000, "Verified OK", "dummy_hash_b"),
            )

        # Verifikasi sebelum merge ada 2 baris
        cnt_before = self.mgr.execute_query("SELECT COUNT(*) as c FROM transaction_details;").rows[0]["c"]
        self.assertEqual(cnt_before, 2)

        # Jalankan merge & deduplikasi
        report = merge_and_deduplicate_tables(self.mgr, "transaction_details")
        self.assertEqual(report.duplicates_removed, 1)
        self.assertEqual(report.deduped_rows, 1)

        # Verifikasi data baris primer telah ter-merge
        row = self.mgr.execute_query("SELECT * FROM transaction_details;").rows[0]
        self.assertEqual(row["ref_no"], "TRX999")
        self.assertEqual(row["catatan_audit"], "Verified OK")

    def test_header_deduplication_keeps_distinct_fingerprints(self):
        """Header dengan fingerprint berbeda tidak boleh digabung hanya karena nomor dokumen sama."""
        # Buat Header 1
        h1 = DocumentHeaderRecord(
            doc_type="faktur_pajak",
            doc_number="FP-2025-001",
            fingerprint_hash="fp_faktur_001",
            doc_title="Faktur Pajak Awal",
        )
        self.mgr.ingest_document_header(h1)

        # Buat Header 2 secara manual dengan nomor & tipe sama (duplikat dari sumber file lain)
        with self.mgr._get_connection() as conn:
            cur = conn.execute(
                """INSERT INTO document_headers (doc_type, doc_number, fingerprint_hash, doc_title)
                   VALUES ('faktur_pajak', 'FP-2025-001', 'fp_faktur_002', 'Faktur Pajak Sumber Lain');"""
            )
            assert cur.lastrowid is not None
            id2 = int(cur.lastrowid)

        # Kaitkan transaksi ke Header 2
        with self.mgr._get_connection() as conn:
            conn.execute(
                """INSERT INTO transaction_details (header_id, description, debit, _row_hash)
                   VALUES (?, 'Barang Modal dari File 2', 15000000, 'hash_file2');""",
                (id2,),
            )

        # Fingerprint berbeda mewakili dokumen/sumber yang berbeda.
        report = merge_and_deduplicate_tables(self.mgr, "document_headers")
        self.assertEqual(report.duplicates_removed, 0)

        # Transaksi tetap terhubung ke header sumber asal.
        tx_rows = self.mgr.execute_query(
            "SELECT * FROM transaction_details WHERE header_id = ?;", (id2,)
        ).rows
        self.assertEqual(len(tx_rows), 1)
        self.assertEqual(tx_rows[0]["description"], "Barang Modal dari File 2")

    def test_generic_table_deduplication_without_pk(self):
        """Uji deduplikasi tabel generik tanpa primary key eksplisit (memakai rowid)."""
        with self.mgr._get_connection() as conn:
            conn.execute("CREATE TABLE audit_logs (level TEXT, message TEXT, code INT);")
            conn.execute("INSERT INTO audit_logs VALUES ('INFO', 'Proses selesai', 200);")
            conn.execute("INSERT INTO audit_logs VALUES ('INFO', 'Proses selesai', 200);")
            conn.execute("INSERT INTO audit_logs VALUES ('ERROR', 'Gagal koneksi', 500);")

        report = merge_and_deduplicate_tables(self.mgr, "audit_logs")
        self.assertEqual(report.duplicates_removed, 1)
        self.assertEqual(report.deduped_rows, 2)

    def test_signatory_table_nullable_position(self):
        """Uji tabel document_signatories: jabatan/position opsional & nullable."""
        header_id = self.mgr.ingest_document_header(
            DocumentHeaderRecord(doc_type="berita_acara", doc_number="BA-001/2025", fingerprint_hash="fp_ba001")
        )
        headers = ["Pihak", "Nama", "Tanda Tangan"]  # Tanpa kolom Jabatan
        rows = [
            ["PIHAK PERTAMA", "Budi Santoso", "Signed"],
            ["PIHAK KEDUA", "Siti Rahma", "Signed"],
        ]
        inserted = self.mgr.ingest_relational_signatories(header_id, headers, rows)
        self.assertEqual(inserted, 2)

        res = self.mgr.execute_query("SELECT * FROM document_signatories WHERE header_id = ?;", (header_id,))
        self.assertEqual(len(res.rows), 2)
        self.assertIsNone(res.rows[0]["position"])  # Nullable sesuai spesifikasi
        self.assertEqual(res.rows[0]["name"], "Budi Santoso")

    def test_page_agent_relational_integration(self):
        """Uji integrasi process_page_tabular_agent memisahkan header dan detail ke tabel relasional."""
        page_md = """# REKENING KORAN PERIODE JANUARI 2025
**ACCOUNT: 987654321**
**CUSTOMER: PT MAJU MUNDUR SEJAHTERA**

| Posting Date | Value Date | Description | Ref No | Debit | Credit | Balance |
|---|---|---|---|---|---|---|
| 01/01/2025 | 01/01/2025 | Saldo Awal | REF-0 | 0 | 0 | 100.000.000 |
| 02/01/2025 | 02/01/2025 | Pembayaran Vendor | REF-101 | 15.000.000 | 0 | 85.000.000 |
| 03/01/2025 | 03/01/2025 | Penerimaan Invoice | REF-102 | 0 | 25.000.000 | 110.000.000 |
"""
        _event, results = process_page_tabular_agent(
            page_markdown=page_md,
            page_number=1,
            source_file="rekening_pt_maju.pdf",
            db_path=self.db_path,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].table_name, "transaction_details")
        self.assertEqual(results[0].total_rows_ingested, 3)

        # Verifikasi document_headers terisi otomatis dari teks dokumen
        hdr_res = self.mgr.execute_query("SELECT * FROM document_headers;").rows
        self.assertEqual(len(hdr_res), 1)
        self.assertEqual(hdr_res[0]["doc_number"], "987654321")
        self.assertEqual(hdr_res[0]["parties"], "PT MAJU MUNDUR SEJAHTERA")


if __name__ == "__main__":
    unittest.main()
