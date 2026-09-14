"""
Modul Pemrosesan Data Tabular Transaksional ke Database SQLite, Sub-Agent SQL Per-Halaman, & Dual-Track Guardrail.

Fitur:
  1. Deteksi & pemisahan tabel transaksional vs tabel naratif kualitatif.
  2. Sub-Agent SQL mandiri per-halaman/slide:
     - Memahami data tabel per halaman langsung tanpa menunggu dokumen selesai.
     - Menginspeksi skema dan tabel eksisting di SQLite.
     - Menjalankan query SQL mandiri (cek baris terakhir, continuity, dsb).
     - Meng-ingest / append data per halaman secara terpisah.
  3. Mekanisme Double-Verification (Verifikasi Ganda).
  4. Dual-Track Guardrail Cross-Verification:
     - Membandingkan hasil jalur Teks Markdown vs jalur Database SQLite.
     - Menghasilkan laporan audit kesesuaian baris, kolom, dan anomali.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger("app.tabular_db")
_active_connection: ContextVar[tuple[str, sqlite3.Connection] | None] = ContextVar("tabular_transaction", default=None)

from .schemas import (
    DedupReport,
    DocumentHeaderRecord,
    DualTrackGuardrailReport,
    PageTabularEvent,
    TableClassificationResult,
    TableColumnSchema,
    TableIngestionResult,
    TableSchema,
    TableTypeLiteral,
    TableVerificationReport,
    TabularQueryResult,
    VerificationCheck,
)

# Kata kunci header yang mengindikasikan tabel transaksional / finansial
TRANSACTIONAL_HEADER_KEYWORDS: set[str] = {
    "date",
    "tanggal",
    "posting date",
    "value date",
    "description",
    "deskripsi",
    "keterangan",
    "transaksi",
    "transaction",
    "reference",
    "ref no",
    "cheque",
    "debit",
    "debit amount",
    "credit",
    "credit amount",
    "kredit",
    "balance",
    "saldo",
    "amount",
    "nominal",
    "total",
    "subtotal",
    "price",
    "harga",
    "qty",
    "quantity",
    "unit",
    "rate",
    "tax",
    "pajak",
    "fee",
    "biaya",
    "mutasi",
    "tds",
    "dhl",
    "ph",
    "suhu",
    "temperature",
    "elevasi",
    "elevation",
    "skor",
    "score",
    "nilai",
    "value",
    "persentase",
    "percentage",
    "korelasi",
    "correlation",
    "volume",
}



# ==============================================================================
# Kamus Alias Kanonikal & Pemetaan Defensif (Defensive Model-Driven Parsing)
# ==============================================================================

TRANSACTION_CANONICAL_ALIASES: dict[str, list[str]] = {
    "txn_date": [
        "posting date",
        "posting_date",
        "post date",
        "tanggal posting",
        "tanggal",
        "date",
        "tgl",
        "tgl transaksi",
        "transaction date",
    ],
    "value_date": [
        "value date",
        "value_date",
        "valuta date",
        "tgl valuta",
        "valuta",
    ],
    "description": [
        "transaction description",
        "description",
        "deskripsi",
        "keterangan",
        "uraian",
        "transaksi",
        "uraian transaksi",
        "narasi",
        "rincian",
        "item",
        "nama item",
    ],
    "ref_no": [
        "reference/cheque no",
        "reference cheque no",
        "reference",
        "ref no",
        "ref",
        "cheque no",
        "no ref",
        "no referensi",
        "no cek",
        "nomor transaksi",
        "id transaksi",
        "doc ref",
    ],
    "detail_info": [
        "detail",
        "details",
        "keterangan tambahan",
        "info",
        "informasi",
        "penerima/pengirim",
        "tujuan",
    ],
    "debit": [
        "debit amount",
        "debit",
        "debet",
        "mutasi debet",
        "mutasi debit",
        "keluar",
        "pengeluaran",
        "dr",
    ],
    "credit": [
        "credit amount",
        "credit",
        "kredit",
        "mutasi kredit",
        "masuk",
        "penerimaan",
        "cr",
    ],
    "balance": [
        "balance",
        "saldo",
        "sisa saldo",
        "saldo akhir",
        "total balance",
    ],
}

SIGNATORY_CANONICAL_ALIASES: dict[str, list[str]] = {
    "role_party": [
        "pihak / peran",
        "pihak peran",
        "pihak",
        "peran",
        "role",
        "party",
        "pihak pertama",
        "pihak kedua",
        "instansi",
        "perusahaan",
        "organisasi",
    ],
    "name": [
        "nama",
        "nama penandatangan",
        "nama terang",
        "name",
        "signatory",
        "pejabat",
    ],
    "position": [
        "jabatan",
        "position",
        "title",
        "job title",
        "kedudukan",
        "role position",
    ],
    "date": [
        "tanggal",
        "tgl",
        "date",
        "signed date",
        "tanggal ttd",
    ],
    "signature_status": [
        "tanda tangan / paraf",
        "tanda tangan",
        "ttd",
        "paraf",
        "signature",
        "status ttd",
    ],
    "notes": [
        "keterangan",
        "notes",
        "catatan",
        "remark",
        "remarks",
    ],
}


def extract_document_header_info(
    markdown_text: str, source_file: str = ""
) -> DocumentHeaderRecord:
    """
    Ekstrak informasi metadata header dokumen secara defensif dari teks Markdown.
    Mengekstrak jenis dokumen, nomor dokumen, tanggal/periode, pihak-pihak terkait, dan total saldo/transaksi.
    Menghasilkan fingerprint hash untuk deduplikasi antar-file/sumber.
    """
    now_iso = datetime.now(UTC).isoformat()
    if not markdown_text:
        return DocumentHeaderRecord(
            source_file=source_file,
            created_at=now_iso,
            updated_at=now_iso,
        )

    text_lower = markdown_text.lower()
    doc_type = "general_document"
    if any(
        kw in text_lower
        for kw in (
            "account statement",
            "rekening koran",
            "bank statement",
            "giro",
            "tabungan",
            "statement of account",
        )
    ):
        doc_type = "bank_statement"
    elif any(
        kw in text_lower
        for kw in (
            "berita acara",
            "bast",
            "serah terima",
            "upgrade daya",
            "perjanjian",
        )
    ):
        doc_type = "berita_acara"
    elif any(
        kw in text_lower
        for kw in ("invoice", "faktur", "tagihan", "kwitansi", "receipt")
    ):
        doc_type = "invoice"

    # 1. Judul Dokumen (dari H1 atau baris tebal pertama)
    doc_title: str | None = None
    m_h1 = re.search(r"^#\s+(.+)$", markdown_text, re.MULTILINE)
    if m_h1:
        doc_title = clean_cell_text(m_h1.group(1))
    else:
        m_bold = re.search(r"\*\*([^\*\n]+)\*\*", markdown_text)
        if m_bold:
            doc_title = clean_cell_text(m_bold.group(1))

    # 2. Nomor Dokumen / Nomor Rekening / Nomor Surat
    doc_number: str | None = None
    # Cari dengan separator eksplisit : atau = terlebih dahulu
    m_acc = re.search(
        r"(?:ACCOUNT(?:\s*NO|\s*NUMBER)?|NO\.?\s*REK(?:ENING)?|NOMOR\s*REKENING)\s*[:=]\s*([0-9A-Za-z\-_/]+)",
        markdown_text,
        re.IGNORECASE,
    )
    if m_acc and not m_acc.group(1).lower() in ("koran", "giro", "tabungan", "statement"):
        doc_number = clean_cell_text(m_acc.group(1))
    else:
        m_acc_num = re.search(
            r"(?:ACCOUNT(?:\s*NO|\s*NUMBER)?|REKENING|NO\.?\s*REK(?:ENING)?)\s*[:.]?\s*([0-9][0-9A-Za-z\-_/]+)",
            markdown_text,
            re.IGNORECASE,
        )
        if m_acc_num:
            doc_number = clean_cell_text(m_acc_num.group(1))
        else:
            m_nomor = re.search(
                r"(?:NOMOR|NO)\s*[:.]\s*([0-9A-Za-z\-_/]+)",
                markdown_text,
                re.IGNORECASE,
            )
            if m_nomor and not m_nomor.group(1).lower() in ("koran", "giro", "tabungan", "statement"):
                doc_number = clean_cell_text(m_nomor.group(1))
            else:
                m_agenda = re.search(
                    r"(?:NOMOR AGENDA|ID PELANGGAN)\s*[:.]\s*([0-9]+)",
                    markdown_text,
                    re.IGNORECASE,
                )
                if m_agenda:
                    doc_number = clean_cell_text(m_agenda.group(1))

    # 3. Tanggal / Periode
    doc_date: str | None = None
    m_period = re.search(
        r"(?:PERIOD|PERIODE)\s*[:.]\s*([0-9./-]+(?:\s+(?:TO|-)\s+[0-9./-]+)?)",
        markdown_text,
        re.IGNORECASE,
    )
    if m_period:
        doc_date = clean_cell_text(m_period.group(1))
    else:
        m_date = re.search(
            r"(?:TANGGAL|DATE)\s*[:.]\s*([^\n\r]+)",
            markdown_text,
            re.IGNORECASE,
        )
        if m_date:
            raw_d = clean_cell_text(m_date.group(1))
            doc_date = parse_date_value(raw_d) or raw_d

    # 4. Pihak-pihak terkait (Customer / Nasabah / PT)
    parties: str | None = None
    m_cust = re.search(
        r"(?:CUSTOMER|NASABAH|PIHAK)\s*[:.]\s*([^\n\r]+)",
        markdown_text,
        re.IGNORECASE,
    )
    if m_cust:
        parties = clean_cell_text(m_cust.group(1))
    else:
        m_pt = re.search(r"\*\*(PT\s+[^\*]+)\*\*", markdown_text)
        if m_pt:
            parties = clean_cell_text(m_pt.group(1))

    # 5. Total Saldo / Total Amount
    total_amount: float | None = None
    m_tot = re.search(
        r"(?:BALANCE AT PERIOD START|TOTAL|SALDO AKHIR|TOTAL AMOUNT|NILAI)\s*[:=]?\s*Rp?\.?\s*([0-9,.]+)",
        markdown_text,
        re.IGNORECASE,
    )
    if m_tot:
        total_amount = parse_numeric_value(m_tot.group(1))

    # Fingerprint harus membedakan dokumen yang tidak memiliki metadata header.
    # source_file tetap sama pada setiap halaman sebuah dokumen, sehingga halaman
    # yang diproses terpisah masih mengarah ke header relasional yang sama.
    raw_fingerprint = "|".join(
        (
            doc_type,
            clean_cell_text(doc_number or ""),
            clean_cell_text(doc_date or ""),
            clean_cell_text(parties or ""),
            str(Path(source_file).resolve()) if source_file else "",
        )
    )
    if not source_file and not any((doc_number, doc_date, parties, doc_title)):
        raw_fingerprint = f"{doc_type}|{hashlib.sha256(markdown_text.encode('utf-8')).hexdigest()}"
    fingerprint_hash = hashlib.sha256(raw_fingerprint.encode("utf-8")).hexdigest()[:16]

    return DocumentHeaderRecord(
        doc_type=doc_type,
        doc_title=doc_title,
        doc_number=doc_number,
        doc_date=doc_date,
        parties=parties,
        total_amount=total_amount,
        currency="IDR",
        source_file=source_file,
        fingerprint_hash=fingerprint_hash,
        created_at=now_iso,
        updated_at=now_iso,
    )


def defensive_map_columns(
    raw_headers: list[str],
    aliases_dict: dict[str, list[str]],
    domain_type: str = "transaction",
) -> tuple[bool, dict[str, int], dict[str, str]]:
    """
    Normalisasi defensif nama kolom terhadap kamus alias kanonikal.
    Mengembalikan (is_valid, canonical_index_map, extra_columns_with_types).

    Field wajib:
      - transaction: minimal memiliki (txn_date ATAU description) DAN (debit ATAU credit ATAU balance ATAU amount)
      - signatory: minimal memiliki (name ATAU role_party) DAN (position ATAU signature_status)
    Field opsional: jika tidak ada di header sumber, akan bernilai None/NULL tanpa error.
    Kolom yang tidak dikenal dipertahankan dalam extra_columns_with_types untuk skema evolutif dinamis.
    """
    canonical_indices: dict[str, int] = {}
    matched_col_indices: set[int] = set()

    for col_idx, raw_h in enumerate(raw_headers):
        cleaned_h = clean_cell_text(raw_h).lower()
        sanitized_h = sanitize_identifier(raw_h)
        for canon_key, alias_list in aliases_dict.items():
            if canon_key in canonical_indices:
                continue
            if any(
                alias == cleaned_h
                or alias == sanitized_h
                or alias in cleaned_h
                for alias in alias_list
            ):
                canonical_indices[canon_key] = col_idx
                matched_col_indices.add(col_idx)
                break

    extra_columns: dict[str, str] = {}
    for col_idx, raw_h in enumerate(raw_headers):
        if col_idx not in matched_col_indices:
            clean_ident = sanitize_identifier(raw_h)
            if clean_ident:
                extra_columns[clean_ident] = "TEXT"

    # Validasi Anchor Wajib Defensif
    if domain_type == "transaction":
        has_anchor = ("txn_date" in canonical_indices or "description" in canonical_indices)
        has_metric = (
            "debit" in canonical_indices
            or "credit" in canonical_indices
            or "balance" in canonical_indices
        )
        is_valid = bool(has_anchor and has_metric)
    elif domain_type == "signatory":
        has_person = ("name" in canonical_indices or "role_party" in canonical_indices)
        has_signature_or_pos = (
            "position" in canonical_indices or "signature_status" in canonical_indices
        )
        is_valid = bool(has_person and has_signature_or_pos)
    else:
        is_valid = False

    return is_valid, canonical_indices, extra_columns

# ==============================================================================
# Helper Parsing & Normalisasi Nilai
# ==============================================================================


def sanitize_identifier(name: str) -> str:
    """Bersihkan nama tabel atau kolom agar valid sebagai identifier SQL."""
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"col_{cleaned}"
    return cleaned


def clean_cell_text(cell: str) -> str:
    """Bersihkan teks sel dari spasi berlebih dan karakter markdown formatting."""
    s = re.sub(r"\s+", " ", cell).strip()
    return s.strip("*_").strip()


def parse_numeric_value(val_str: str) -> float | int | None:
    """
    Ekstrak nilai numerik dari string (mata uang, pemisah ribuan, format negatif akuntansi).
    Mendukung format:
      - Standar US: 1,234,567.89
      - Standar Indo/Eropa: 1.234.567,89
      - Format negatif akuntansi: (1,234.50) -> -1234.50
      - Nilai dengan simbol mata uang: Rp 50.000, $120.50, IDR 1,000,000
    """
    if not val_str or not isinstance(val_str, str):
        return None

    raw = clean_cell_text(val_str)
    if not raw or raw in ("-", "--", "N/A", "NA", "null", "None"):
        return None

    # Cek format negatif akuntansi dalam tanda kurung, misal (1,234.50)
    is_negative = False
    if raw.startswith("(") and raw.endswith(")"):
        is_negative = True
        raw = raw[1:-1].strip()
    elif raw.startswith("-"):
        is_negative = True
        raw = raw[1:].strip()

    # Bersihkan simbol mata uang & karakter non-angka kecuali titik, koma, minus
    cleaned = re.sub(r"[^\d.,]", "", raw).strip()
    if not cleaned:
        return None

    has_dot = "." in cleaned
    has_comma = "," in cleaned
    # Jika ada dua pemisah, yang paling kanan adalah pemisah desimal.
    if has_dot and has_comma:
        decimal_sep = "." if cleaned.rfind(".") > cleaned.rfind(",") else ","
        thousands_sep = "," if decimal_sep == "." else "."
        normalized = cleaned.replace(thousands_sep, "").replace(decimal_sep, ".")
    elif has_dot or has_comma:
        sep = "." if has_dot else ","
        whole, fraction = cleaned.rsplit(sep, 1)
        # Satu pemisah dengan tepat tiga digit sesudahnya lazim dipakai sebagai
        # pemisah ribuan dalam dokumen Indonesia (Rp 50.000, 1.234).
        is_thousands = len(fraction) == 3 and len(whole.replace(sep, "")) >= 1
        normalized = cleaned.replace(sep, "") if is_thousands else f"{whole}.{fraction}"
    else:
        normalized = cleaned

    if re.match(r"^\d+(\.\d+)?$", normalized):
        try:
            num = float(normalized)
            return -num if is_negative else num
        except ValueError:
            pass

    return None


def parse_date_value(val_str: str) -> str | None:
    """
    Cek dan parsing format tanggal umum (YYYYMMDD, YYYY-MM-DD, DD.MM.YYYY, DD/MM/YYYY).
    Mengembalikan format ISO 'YYYY-MM-DD' atau None jika bukan tanggal.
    """
    if not val_str or not isinstance(val_str, str):
        return None
    raw = clean_cell_text(val_str).strip()

    # Format YYYYMMDD (misal: 20240201)
    if re.match(r"^(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])$", raw):
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"

    # Format YYYY-MM-DD
    if re.match(r"^(19|20)\d{2}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$", raw):
        return raw

    # Format DD.MM.YYYY atau DD/MM/YYYY atau DD-MM-YYYY (juga dukung 1 digit D/M misal 1/8/2024 atau 2 digit YY)
    m = re.match(
        r"^(0?[1-9]|[12]\d|3[01])[./-](0?[1-9]|1[0-2])[./-]((?:19|20)?\d{2})$", raw
    )
    if m:
        day = int(m.group(1))
        month = int(m.group(2))
        year_str = m.group(3)
        year = int(f"20{year_str}" if len(year_str) == 2 else year_str)
        return f"{year:04d}-{month:02d}-{day:02d}"

    return None


def sanitize_markdown_tables(markdown_text: str) -> str:
    """
    Normalisasi dan perbaiki tabel Markdown (GFM) yang rusak atau anomali:
    1. Perbaiki baris sub-header yang lupa diawali tanda pipa `|` (contoh: `**Bank 1** | | ...` -> `| **Bank 1** | | ...`).
    2. Hapus baris kosong yang tidak disengaja di tengah-tengah tabel sebelum baris data berikutnya.
    3. Normalisasi baris data yang mengandung pemisah pipa palsu (contoh: `|-----| |-----|`) agar jumlah kolom konsisten dengan header.
    4. Pastikan baris tabel diawali dan diakhiri dengan pipa `|`.
    """
    if not markdown_text or "|" not in markdown_text:
        return markdown_text

    lines = markdown_text.splitlines()
    result_lines: list[str] = []
    i = 0
    in_table = False
    expected_cols = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Deteksi awal tabel Markdown: header + separator |---|
        if (
            not in_table
            and stripped.startswith("|")
            and stripped.endswith("|")
            and i + 1 < len(lines)
        ):
            next_line = lines[i + 1].strip()
            if next_line.startswith("|") and re.match(r"^\|(\s*:?-+:?\s*\|)+$", next_line):
                in_table = True
                headers = [c.strip() for c in stripped.strip("|").split("|")]
                expected_cols = len(headers)
                result_lines.append(line)
                result_lines.append(lines[i + 1])
                i += 2
                continue

        if in_table:
            # Kasus 1: Baris kosong di dalam tabel
            if not stripped:
                # Intip baris berikutnya: apakah baris data tabel atau sub-header?
                peek_idx = i + 1
                while peek_idx < len(lines) and not lines[peek_idx].strip():
                    peek_idx += 1
                if peek_idx < len(lines):
                    peek_line = lines[peek_idx].strip()
                    # Header yang langsung diikuti separator menandai tabel baru;
                    # jangan gabungkan dua tabel yang hanya dipisahkan baris kosong.
                    is_new_table = (
                        peek_line.startswith("|")
                        and peek_line.endswith("|")
                        and peek_idx + 1 < len(lines)
                        and re.match(
                            r"^\|(\s*:?-+:?\s*\|)+$",
                            lines[peek_idx + 1].strip(),
                        )
                    )
                    # Jika baris berikutnya adalah baris tabel '| ... |' atau sub-header '... | ... |'
                    if (
                        (
                            (peek_line.startswith("|") and peek_line.endswith("|"))
                            or ("|" in peek_line and peek_line.endswith("|"))
                        )
                        and not is_new_table
                    ):
                        # Lewati baris kosong ini agar tabel tidak terputus
                        i += 1
                        continue
                # Jika baris berikutnya bukan tabel, maka tabel berakhir
                in_table = False
                expected_cols = 0
                result_lines.append(line)
                i += 1
                continue

            # Kasus 2: Baris sub-header yang lupa pipa di awal, misal: `**Bank 1** | | | | | | | | | | | |`
            if not stripped.startswith("|") and "|" in stripped and stripped.endswith("|"):
                stripped = "| " + stripped
                line = stripped

            # Kasus 3: Baris tabel aktif
            if stripped.startswith("|") and stripped.endswith("|"):
                # Cek apakah ini separator berulang di tengah tabel
                if re.match(r"^\|(\s*:?-+:?\s*\|)+$", stripped):
                    i += 1
                    continue

                cells = [c.strip() for c in stripped.strip("|").split("|")]

                # Normalisasi jika sel berlebih karena delimiter palsu '-----'
                if expected_cols > 0 and len(cells) > expected_cols:
                    cleaned_cells: list[str] = []
                    k = 0
                    while k < len(cells) and len(cleaned_cells) < expected_cols:
                        cleaned_cells.append(cells[k])
                        k += 1
                    if len(cells) > expected_cols and cells[-1]:
                        cleaned_cells[-1] = cells[-1]
                    cells = cleaned_cells
                elif expected_cols > 0 and len(cells) < expected_cols:
                    cells.extend([""] * (expected_cols - len(cells)))

                reconstructed = "| " + " | ".join(cells) + " |"
                result_lines.append(reconstructed)
                i += 1
                continue

            # Bukan baris tabel, maka tabel berakhir
            in_table = False
            expected_cols = 0
            result_lines.append(line)
            i += 1
            continue

        result_lines.append(line)
        i += 1

    return "\n".join(result_lines)


def parse_markdown_tables(markdown_text: str) -> list[dict[str, Any]]:
    """
    Ekstrak dan parse seluruh tabel format GFM Markdown dari teks dokumen.
    Mengembalikan daftar objek berisi header asli, baris terurai, teks konteks, dan metadata tag SQLite jika ada.
    """
    sanitized_text = sanitize_markdown_tables(markdown_text)
    lines = sanitized_text.splitlines()
    tables: list[dict[str, Any]] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Deteksi awal tabel Markdown: baris yang diawali '|' dan baris berikutnya adalah separator '|---|'
        if line.startswith("|") and line.endswith("|") and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line.startswith("|") and re.match(
                r"^\|(\s*:?-+:?\s*\|)+$", next_line
            ):
                # Header row
                header_raw = [c.strip() for c in line.strip("|").split("|")]
                rows_raw: list[list[str]] = []

                # Cek apakah ada tag metadata <!-- sqlite_table: ... --> tepat di atas tabel
                sqlite_table_hint: str | None = None
                for k in range(max(0, i - 2), i):
                    prev_line = lines[k].strip()
                    m_tag = re.search(r"<!--\s*sqlite_table:\s*([\w\-_]+).*?-->", prev_line, re.IGNORECASE)
                    if m_tag:
                        sqlite_table_hint = m_tag.group(1).strip()
                        break

                # Konteks teks sebelum tabel (3 baris sebelumnya)
                context_lines = [
                    lines[k].strip()
                    for k in range(max(0, i - 3), i)
                    if lines[k].strip() and not lines[k].strip().startswith("<!--")
                ]
                context = " ".join(context_lines)

                # Baca baris data
                j = i + 2
                while j < len(lines):
                    row_line = lines[j].strip()
                    if not row_line.startswith("|") or not row_line.endswith("|"):
                        break
                    row_cells = [c.strip() for c in row_line.strip("|").split("|")]
                    if len(row_cells) < len(header_raw):
                        row_cells.extend([""] * (len(header_raw) - len(row_cells)))
                    elif len(row_cells) > len(header_raw):
                        row_cells = row_cells[: len(header_raw)]
                    rows_raw.append(row_cells)
                    j += 1

                if rows_raw:
                    tables.append(
                        {
                            "headers": header_raw,
                            "rows": rows_raw,
                            "context": context,
                            "sqlite_table_hint": sqlite_table_hint,
                            "line_start": i,
                            "line_end": j,
                        }
                    )
                i = j
                continue
        i += 1

    return tables


def tag_markdown_tables_with_sqlite_metadata(
    markdown_text: str,
    table_mappings: list[tuple[dict[str, Any], str]],
) -> str:
    """
    Sematkan tag metadata <!-- sqlite_table: <table_name> --> di atas setiap blok tabel pada Markdown.
    Jika tag sudah ada, pertahankan atau perbarui nama tabelnya.
    """
    if not table_mappings or not markdown_text:
        return markdown_text

    lines = markdown_text.splitlines()
    # Urutkan pemetaan dari line_start paling bawah ke atas agar indeks baris tidak bergeser saat disisipkan
    sorted_mappings = sorted(table_mappings, key=lambda x: x[0].get("line_start", 0), reverse=True)

    for tbl_dict, target_table in sorted_mappings:
        line_start = tbl_dict.get("line_start", 0)
        tag_str = f"<!-- sqlite_table: {target_table} -->"

        # Cek apakah baris persis sebelumnya sudah memiliki tag
        has_existing_tag = False
        if line_start > 0:
            prev_line = lines[line_start - 1].strip()
            if prev_line.startswith("<!--") and "sqlite_table:" in prev_line:
                # Perbarui tag yang ada
                lines[line_start - 1] = tag_str
                has_existing_tag = True

        if not has_existing_tag:
            # Sisipkan baris tag baru
            lines.insert(line_start, tag_str)

    return "\n".join(lines)


# ==============================================================================
# Classifier: Heuristik & AI Classification
# ==============================================================================


def classify_table_heuristic(
    headers: list[str],
    rows: list[list[str]],
    context: str = "",
) -> TableClassificationResult:
    """
    Klasifikasi heuristik untuk menentukan apakah tabel bertipe transaksional / finansial
    (yang harus masuk SQLite) atau tabel naratif (bisa di-chunking ke Vector RAG).
    """
    total_cols = len(headers)
    total_rows = len(rows)

    if total_cols == 0 or total_rows == 0:
        return TableClassificationResult(
            table_id="empty",
            is_transactional=False,
            table_type="generic_table",
            recommended_storage="vector_rag",
            confidence=1.0,
            reasoning="Tabel kosong.",
            numeric_density=0.0,
            date_density=0.0,
            total_rows=0,
            total_columns=0,
            columns_detected=[],
        )

    # 1. Analisis Header
    matched_keywords = []
    for h in headers:
        clean_h = sanitize_identifier(h).replace("_", " ").lower()
        for kw in TRANSACTIONAL_HEADER_KEYWORDS:
            if kw in clean_h:
                matched_keywords.append(h)
                break

    keyword_ratio = len(matched_keywords) / total_cols

    # 2. Analisis Kepadatan Numerik & Tanggal pada Sel Data
    numeric_cells = 0
    date_cells = 0
    total_cells = total_rows * total_cols
    total_char_len = 0

    for r in rows:
        for cell in r:
            total_char_len += len(cell)
            if parse_numeric_value(cell) is not None:
                numeric_cells += 1
            elif parse_date_value(cell) is not None:
                date_cells += 1

    numeric_density = numeric_cells / max(1, total_cells)
    date_density = date_cells / max(1, total_cells)
    avg_cell_len = total_char_len / max(1, total_cells)

    # 3. Klasifikasi Semantik
    is_transactional = False
    table_type: TableTypeLiteral = "generic_table"
    reasoning_points: list[str] = []

    has_financial_headers = any(
        kw in [sanitize_identifier(h) for h in headers]
        for kw in [
            "debit",
            "kredit",
            "credit",
            "saldo",
            "balance",
            "amount",
            "nominal",
            "mutasi",
        ]
    )
    has_date_headers = any(
        kw in [sanitize_identifier(h) for h in headers]
        for kw in ["posting_date", "tanggal", "date", "value_date"]
    )

    if has_financial_headers and has_date_headers:
        is_transactional = True
        table_type = "financial_statement" if total_rows >= 5 else "transactional_log"
        reasoning_points.append(
            f"Header mengandung kombinasi tanggal dan finansial (Debit/Kredit/Saldo): {matched_keywords}."
        )
    elif (
        keyword_ratio >= 0.4
        and (numeric_density + date_density >= 0.3)
        and total_rows >= 3
    ):
        is_transactional = True
        table_type = "transactional_log"
        reasoning_points.append(
            f"Rasio keyword transaksional tinggi ({keyword_ratio:.1%}) dan kepadatan angka/tanggal {numeric_density + date_density:.1%}."
        )
    elif numeric_density >= 0.40 and total_rows >= 3 and avg_cell_len <= 45:
        is_transactional = True
        table_type = "transactional_log"
        reasoning_points.append(
            f"Kepadatan numerik tinggi ({numeric_density:.1%}) pada tabel terstruktur ({total_rows} baris, {total_cols} kolom)."
        )
    elif avg_cell_len > 70 or (numeric_density < 0.15 and not has_financial_headers):
        is_transactional = False
        table_type = "narrative_matrix"
        reasoning_points.append(
            f"Rata-rata teks per sel panjang ({avg_cell_len:.1f} karakter) dan dominan narasi kualitatif."
        )
    else:
        is_transactional = False
        table_type = "generic_table"
        reasoning_points.append(
            f"Tabel umum tanpa pola transaksi dominan (numeric density: {numeric_density:.1%})."
        )

    recommended_storage = "sqlite_database" if is_transactional else "vector_rag"
    confidence = min(
        0.98, max(0.65, keyword_ratio * 0.5 + (numeric_density + date_density) * 0.5)
    )

    return TableClassificationResult(
        table_id=f"table_auto_{total_rows}r_{total_cols}c",
        is_transactional=is_transactional,
        table_type=table_type,
        recommended_storage=recommended_storage,
        confidence=round(confidence, 2),
        reasoning="; ".join(reasoning_points),
        numeric_density=round(numeric_density, 3),
        date_density=round(date_density, 3),
        total_rows=total_rows,
        total_columns=total_cols,
        columns_detected=headers,
    )


def infer_table_schema(
    table_name: str,
    headers: list[str],
    rows: list[list[str]],
    source_file: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> TableSchema:
    """
    Infer skema kolom SQLite dan tipe data (TEXT, REAL, INTEGER, DATE) dari data baris.
    """
    col_schemas: list[TableColumnSchema] = []
    total_rows = len(rows)

    used_names: set[str] = set()

    for col_idx, raw_header in enumerate(headers):
        clean_name = sanitize_identifier(raw_header)
        # Hindari duplikasi nama kolom
        base_name = clean_name
        suffix = 1
        while clean_name in used_names:
            clean_name = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(clean_name)

        # Analisis sampel nilai kolom
        col_values = [
            rows[r][col_idx]
            for r in range(min(total_rows, 50))
            if col_idx < len(rows[r])
        ]
        non_empty = [v for v in col_values if clean_cell_text(v)]

        numeric_count = sum(1 for v in non_empty if parse_numeric_value(v) is not None)
        date_count = sum(1 for v in non_empty if parse_date_value(v) is not None)

        sql_type: Any = "TEXT"
        if non_empty and numeric_count / len(non_empty) >= 0.7:
            # Cek apakah integer atau real (ada desimal)
            has_float = any(
                isinstance(parse_numeric_value(v), float)
                and not float(parse_numeric_value(v) or 0).is_integer()
                for v in non_empty
            )
            sql_type = "REAL" if has_float else "NUMERIC"
        elif non_empty and date_count / len(non_empty) >= 0.7:
            sql_type = "DATE"
        elif any(
            k in clean_name
            for k in [
                "amount",
                "saldo",
                "balance",
                "debit",
                "kredit",
                "credit",
                "nominal",
                "fee",
                "harga",
                "price",
            ]
        ):
            sql_type = "REAL"
        elif any(k in clean_name for k in ["date", "tanggal", "posting"]):
            sql_type = "DATE"

        col_schemas.append(
            TableColumnSchema(
                name=clean_name,
                original_name=raw_header,
                sql_type=sql_type,
                is_nullable=True,
                sample_values=non_empty[:3],
            )
        )

    return TableSchema(
        table_name=sanitize_identifier(table_name),
        source_file=source_file or "",
        columns=col_schemas,
        metadata=metadata or {},
    )


# ==============================================================================
# Database Manager: SQLite Management & Safe Execution
# ==============================================================================


class TabularDatabaseManager:
    """
    Pengelola Database SQLite untuk Ingesti & Querying Data Tabular Transaksional Dokumen.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            default_dir = Path("output/databases").resolve()
            default_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = default_dir / "documents_data.sqlite"
        else:
            self.db_path = Path(db_path).resolve()
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Buka koneksi SQLite dengan row factory dict dan pastikan selalu ditutup."""
        active = _active_connection.get()
        if active and active[0] == str(self.db_path.resolve()):
            yield active[1]
            return
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def create_table(
        self, schema: TableSchema, if_not_exists: bool = True, replace: bool = False
    ) -> str:
        """Buat tabel SQLite berdasarkan skema terstruktur."""
        with self._get_connection() as conn:
            if replace:
                conn.execute(f'DROP TABLE IF EXISTS "{schema.table_name}";')

            col_defs = [f'"{col.name}" {col.sql_type}' for col in schema.columns]
            col_defs.insert(0, '"_row_id" INTEGER PRIMARY KEY AUTOINCREMENT')
            col_defs.append('"_source_doc" TEXT')
            col_defs.append('"_page_number" INTEGER')
            col_defs.append('"_ingested_at" TEXT')

            exist_clause = "" if replace else ("IF NOT EXISTS" if if_not_exists else "")
            sql = (
                f'CREATE TABLE {exist_clause} "{schema.table_name}" (\n  '
                + ",\n  ".join(col_defs)
                + "\n);"
            )

            conn.execute(sql)
            # Buat indeks untuk kolom tanggal atau referensi jika ada
            for col in schema.columns:
                if col.sql_type == "DATE" or "ref" in col.name or "account" in col.name:
                    idx_name = f"idx_{schema.table_name}_{col.name}"
                    conn.execute(
                        f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON "{schema.table_name}" ("{col.name}");'
                    )

        return schema.table_name

    def ingest_records(
        self,
        table_name: str,
        schema: TableSchema,
        headers: list[str],
        rows: list[list[str]],
        source_doc: str = "",
        page_number: int | None = None,
        replace: bool = True,
    ) -> int:
        """
        Ingest baris-baris data mentah ke tabel SQLite dengan transformasi tipe data otomatis.
        Jika replace=False, baris baru di-append ke tabel yang sudah ada.
        """
        self.create_table(schema, if_not_exists=True, replace=replace)

        col_names = [col.name for col in schema.columns]
        col_types = {col.name: col.sql_type for col in schema.columns}

        header_to_idx = {sanitize_identifier(h): i for i, h in enumerate(headers)}

        placeholders = ", ".join(["?"] * (len(col_names) + 3))
        insert_cols = ", ".join(
            [f'"{c}"' for c in col_names]
            + ['"_source_doc"', '"_page_number"', '"_ingested_at"']
        )
        insert_sql = (
            f'INSERT INTO "{schema.table_name}" ({insert_cols}) VALUES ({placeholders})'
        )

        ingested_at = datetime.now(UTC).isoformat()
        prepared_rows: list[tuple[Any, ...]] = []

        for _row_idx, r in enumerate(rows):
            row_values: list[Any] = []
            for col in schema.columns:
                c_idx = header_to_idx.get(col.name)
                raw_val = (
                    r[c_idx]
                    if (c_idx is not None and c_idx < len(r))
                    else (r[len(row_values)] if len(row_values) < len(r) else "")
                )
                sql_t = col_types.get(col.name, "TEXT")

                if sql_t in ("REAL", "NUMERIC", "INTEGER"):
                    num = parse_numeric_value(raw_val)
                    row_values.append(num)
                elif sql_t == "DATE":
                    dt = parse_date_value(raw_val)
                    row_values.append(dt or clean_cell_text(raw_val))
                else:
                    row_values.append(clean_cell_text(raw_val))

            row_values.append(source_doc)
            row_values.append(page_number)
            row_values.append(ingested_at)
            prepared_rows.append(tuple(row_values))

        with self._get_connection() as conn:
            conn.executemany(insert_sql, prepared_rows)

        return len(prepared_rows)

    def find_matching_table(
        self,
        headers: list[str],
        threshold: float = 0.50,
        explicit_table_name: str | None = None,
        doc_prefix: str | None = None,
    ) -> tuple[str, TableSchema] | None:
        """
        Cari tabel eksisting di database yang memiliki kecocokan skema kolom atau kelanjutan tabel.
        Mengembalikan (table_name, TableSchema) jika ditemukan, atau None jika skema baru.
        """
        valid_sql_types = {
            "TEXT",
            "INTEGER",
            "REAL",
            "NUMERIC",
            "DATE",
            "DATETIME",
        }

        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            )
            tables = [row["name"] for row in cursor.fetchall()]
            if not tables:
                return None

            # 1. Jika ada nama tabel eksplisit dari metadata tag <!-- sqlite_table: ... -->
            if explicit_table_name and explicit_table_name in tables:
                col_cur = conn.execute(f'PRAGMA table_info("{explicit_table_name}");')
                cols = [dict(c) for c in col_cur.fetchall()]
                schema_cols = [
                    TableColumnSchema(
                        name=c["name"],
                        original_name=c["name"],
                        sql_type=(
                            c["type"]
                            if c["type"].upper() in valid_sql_types
                            else "TEXT"
                        ),
                        is_nullable=not bool(c["notnull"]),
                        description="",
                        sample_values=[],
                    )
                    for c in cols
                    if not c["name"].startswith("_")
                ]
                return explicit_table_name, TableSchema(
                    table_name=explicit_table_name,
                    source_file="",
                    columns=schema_cols,
                    primary_key=None,
                    metadata={"is_continuation": True, "matched_by": "explicit_tag"},
                )

            sanitized_incoming = {sanitize_identifier(h) for h in headers if h.strip()}
            if not sanitized_incoming:
                return None

            best_match: str | None = None
            best_score: float = 0.0
            best_schema: TableSchema | None = None

            for t in tables:
                col_cur = conn.execute(f'PRAGMA table_info("{t}");')
                cols = [dict(c) for c in col_cur.fetchall()]
                existing_cols = {
                    c["name"] for c in cols if not c["name"].startswith("_")
                }
                if not existing_cols:
                    continue

                # Jaccard similarity & overlap
                intersection = sanitized_incoming.intersection(existing_cols)
                score = len(intersection) / max(
                    len(sanitized_incoming), len(existing_cols)
                )

                # Cek apakah subset kolom penting cocok (mis. date, description, debit, credit, balance, amount)
                key_financial = {"date", "tanggal", "description", "keterangan", "debit", "kredit", "credit", "balance", "saldo", "amount", "nominal"}
                matched_keys = intersection.intersection(key_financial)
                if len(matched_keys) >= 2:
                    score = max(score, 0.75)

                if score >= threshold and score > best_score:
                    best_score = score
                    best_match = t
                    schema_cols = [
                        TableColumnSchema(
                            name=c["name"],
                            original_name=c["name"],
                            sql_type=(
                                c["type"]
                                if c["type"].upper() in valid_sql_types
                                else "TEXT"
                            ),
                            is_nullable=not bool(c["notnull"]),
                            description="",
                            sample_values=[],
                        )
                        for c in cols
                        if not c["name"].startswith("_")
                    ]
                    best_schema = TableSchema(
                        table_name=t,
                        source_file="",
                        columns=schema_cols,
                        primary_key=None,
                        metadata={"is_continuation": True, "score": score},
                    )

            if best_match and best_schema:
                return best_match, best_schema
        return None

    def get_active_tables_summary(self) -> list[dict[str, Any]]:
        """Ambil snapshot ringkas seluruh tabel aktif untuk feedback konteks ke agent."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            )
            tables = [row["name"] for row in cursor.fetchall()]

            summary: list[dict[str, Any]] = []
            for t in tables:
                col_cur = conn.execute(f"PRAGMA table_info('{t}');")
                cols = [dict(c) for c in col_cur.fetchall()]
                count_cur = conn.execute(f"SELECT COUNT(*) as cnt FROM '{t}';")
                cnt = count_cur.fetchone()["cnt"]
                sample_cur = conn.execute(
                    f"SELECT * FROM '{t}' ORDER BY rowid DESC LIMIT 1;"
                )
                last_row = sample_cur.fetchone()
                last_dict = (
                    {k: v for k, v in dict(last_row).items() if not k.startswith("_")}
                    if last_row
                    else None
                )

                summary.append(
                    {
                        "table_name": t,
                        "columns": [
                            c["name"] for c in cols if not c["name"].startswith("_")
                        ],
                        "total_rows": cnt,
                        "last_row": last_dict,
                    }
                )
            return summary

    def execute_query(
        self,
        sql_query: str,
        params: tuple[Any, ...] | list[Any] | None = None,
        max_rows: int = 100,
    ) -> TabularQueryResult:
        """
        Eksekusi query SELECT aman pada database SQLite dengan dukungan parameter aman.
        """
        t0 = time.perf_counter()
        stripped = sql_query.strip().rstrip(";")
        # Keamanan: batasi hanya SELECT
        if not re.match(r"^\s*SELECT\b", stripped, re.IGNORECASE):
            return TabularQueryResult(
                query=sql_query,
                status="error",
                columns=[],
                rows=[],
                row_count=0,
                execution_time_ms=0.0,
                error_message="Hanya query 'SELECT' yang diizinkan untuk keamanan data.",
            )

        try:
            with self._get_connection() as conn:
                cursor = (
                    conn.execute(stripped, params)
                    if params is not None
                    else conn.execute(stripped)
                )
                col_names = (
                    [d[0] for d in cursor.description] if cursor.description else []
                )
                raw_rows = cursor.fetchmany(max_rows)
                rows_dict = [dict(r) for r in raw_rows]
                elapsed = round((time.perf_counter() - t0) * 1000, 2)
                return TabularQueryResult(
                    query=sql_query,
                    status="success",
                    columns=col_names,
                    rows=rows_dict,
                    row_count=len(rows_dict),
                    execution_time_ms=elapsed,
                )
        except Exception as e:  # noqa: BLE001
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            return TabularQueryResult(
                query=sql_query,
                status="error",
                columns=[],
                rows=[],
                row_count=0,
                execution_time_ms=elapsed,
                error_message=f"SQL Error: {e}",
            )


    def ensure_relational_schema(self) -> None:
        """
        Inisialisasi tabel-tabel relasional kanonikal:
          - document_headers: metadata surat/dokumen, nomor, tanggal, pihak, total
          - transaction_details: baris-baris transaksi finansial (FK header_id)
          - document_signatories: pihak-pihak penandatangan/persetujuan (FK header_id)
        """
        with self._get_connection() as conn:
            schema_sql = """
                CREATE TABLE IF NOT EXISTS document_headers (
                    header_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_type TEXT DEFAULT 'general_document',
                    doc_title TEXT,
                    doc_number TEXT,
                    doc_date TEXT,
                    parties TEXT,
                    total_amount REAL,
                    currency TEXT DEFAULT 'IDR',
                    source_file TEXT,
                    fingerprint_hash TEXT UNIQUE,
                    created_at TEXT,
                    updated_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_headers_doc_type ON document_headers(doc_type);
                CREATE INDEX IF NOT EXISTS idx_headers_doc_number ON document_headers(doc_number);
                CREATE INDEX IF NOT EXISTS idx_headers_fingerprint ON document_headers(fingerprint_hash);

                CREATE TABLE IF NOT EXISTS transaction_details (
                    detail_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    header_id INTEGER NOT NULL,
                    txn_date TEXT,
                    value_date TEXT,
                    description TEXT,
                    ref_no TEXT,
                    detail_info TEXT,
                    debit REAL,
                    credit REAL,
                    balance REAL,
                    _source_doc TEXT,
                    _page_number INTEGER,
                    _row_hash TEXT UNIQUE,
                    _ingested_at TEXT,
                    FOREIGN KEY (header_id) REFERENCES document_headers(header_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_txn_header_id ON transaction_details(header_id);
                CREATE INDEX IF NOT EXISTS idx_txn_date ON transaction_details(txn_date);
                CREATE INDEX IF NOT EXISTS idx_txn_row_hash ON transaction_details(_row_hash);

                CREATE TABLE IF NOT EXISTS document_signatories (
                    signatory_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    header_id INTEGER NOT NULL,
                    role_party TEXT,
                    name TEXT,
                    position TEXT,
                    date TEXT,
                    signature_status TEXT,
                    notes TEXT,
                    _source_doc TEXT,
                    _page_number INTEGER,
                    _row_hash TEXT UNIQUE,
                    _ingested_at TEXT,
                    FOREIGN KEY (header_id) REFERENCES document_headers(header_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_sig_header_id ON document_signatories(header_id);
                CREATE INDEX IF NOT EXISTS idx_sig_row_hash ON document_signatories(_row_hash);
            """
            # executescript melakukan commit implisit dan memutus transaksi penggantian halaman.
            for statement in schema_sql.split(";"):
                if statement.strip():
                    conn.execute(statement)

    def ensure_columns_exist(
        self, table_name: str, required_columns: dict[str, str]
    ) -> None:
        """
        Skema evolusi dinamis (ALTER TABLE ADD COLUMN):
        Menambahkan kolom-kolom baru opsional dari model ke tabel SQLite secara aman sebagai NULLable.
        """
        if not required_columns:
            return
        with self._get_connection() as conn:
            cur = conn.execute(f'PRAGMA table_info("{table_name}");')
            existing = {row["name"].lower() for row in cur.fetchall()}
            for col_name, sql_type in required_columns.items():
                clean_col = sanitize_identifier(col_name)
                if clean_col.lower() not in existing:
                    try:
                        conn.execute(
                            f'ALTER TABLE "{table_name}" ADD COLUMN "{clean_col}" {sql_type} NULL;'
                        )
                        logger.info(
                            "[Tabular:SchemaEvolution] Kolom baru opsional '%s' (%s NULL) ditambahkan ke tabel '%s'",
                            clean_col,
                            sql_type,
                            table_name,
                        )
                    except sqlite3.Error as exc:
                        logger.warning(
                            "[Tabular:SchemaEvolution] Gagal menambah kolom '%s' ke tabel '%s': %s",
                            clean_col,
                            table_name,
                            exc,
                        )

    def ingest_document_header(self, header: DocumentHeaderRecord) -> int:
        """
        Simpan atau perbarui dokumen header ke tabel document_headers.
        Mendukung deduplikasi berbasis fingerprint_hash atau nomor dokumen.
        Mengembalikan header_id integer.
        """
        self.ensure_relational_schema()
        now_iso = datetime.now(UTC).isoformat()
        with self._get_connection() as conn:
            existing_id: int | None = None
            if header.fingerprint_hash:
                cur = conn.execute(
                    'SELECT header_id FROM document_headers WHERE fingerprint_hash = ?;',
                    (header.fingerprint_hash,),
                )
                row = cur.fetchone()
                if row:
                    existing_id = int(row["header_id"])

            if existing_id is not None:
                conn.execute(
                    """UPDATE document_headers SET
                        doc_title = COALESCE(?, doc_title),
                        doc_date = COALESCE(?, doc_date),
                        parties = COALESCE(?, parties),
                        total_amount = COALESCE(?, total_amount),
                        source_file = COALESCE(?, source_file),
                        updated_at = ?
                       WHERE header_id = ?;""",
                    (
                        header.doc_title,
                        header.doc_date,
                        header.parties,
                        header.total_amount,
                        header.source_file,
                        now_iso,
                        existing_id,
                    ),
                )
                return existing_id

            cur = conn.execute(
                """INSERT INTO document_headers (
                    doc_type, doc_title, doc_number, doc_date, parties, total_amount, currency, source_file, fingerprint_hash, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
                (
                    header.doc_type,
                    header.doc_title,
                    header.doc_number,
                    header.doc_date,
                    header.parties,
                    header.total_amount,
                    header.currency or "IDR",
                    header.source_file,
                    header.fingerprint_hash,
                    now_iso,
                    now_iso,
                ),
            )
            return int(cur.lastrowid or 1)

    def ingest_relational_transactions(
        self,
        header_id: int,
        headers: list[str],
        rows: list[list[str]],
        source_doc: str = "",
        page_number: int | None = None,
    ) -> int:
        """
        Ingest baris transaksi finansial ke canonical transaction_details (FK header_id).
        Mendukung pemetaan defensif, dynamic column evolution, dan deduplikasi row-level via _row_hash.
        """
        self.ensure_relational_schema()
        _, col_idx_map, extra_cols = defensive_map_columns(
            headers, TRANSACTION_CANONICAL_ALIASES, "transaction"
        )
        if extra_cols:
            self.ensure_columns_exist("transaction_details", extra_cols)

        now_iso = datetime.now(UTC).isoformat()
        inserted_count = 0

        base_cols = [
            "header_id",
            "txn_date",
            "value_date",
            "description",
            "ref_no",
            "detail_info",
            "debit",
            "credit",
            "balance",
            "_source_doc",
            "_page_number",
            "_row_hash",
            "_ingested_at",
        ]
        extra_col_keys = list(extra_cols.keys())
        all_insert_cols = base_cols + extra_col_keys
        placeholders = ", ".join(["?"] * len(all_insert_cols))
        quoted_cols = ", ".join([f'"{c}"' for c in all_insert_cols])
        sql = f'INSERT OR IGNORE INTO transaction_details ({quoted_cols}) VALUES ({placeholders});'

        extra_indices = [
            (sanitize_identifier(h), i)
            for i, h in enumerate(headers)
            if sanitize_identifier(h) in extra_cols
        ]

        with self._get_connection() as conn:
            for r in rows:
                txn_date_raw = r[col_idx_map["txn_date"]] if "txn_date" in col_idx_map and col_idx_map["txn_date"] < len(r) else None
                value_date_raw = r[col_idx_map["value_date"]] if "value_date" in col_idx_map and col_idx_map["value_date"] < len(r) else None
                desc_raw = r[col_idx_map["description"]] if "description" in col_idx_map and col_idx_map["description"] < len(r) else ""
                ref_raw = r[col_idx_map["ref_no"]] if "ref_no" in col_idx_map and col_idx_map["ref_no"] < len(r) else None
                info_raw = r[col_idx_map["detail_info"]] if "detail_info" in col_idx_map and col_idx_map["detail_info"] < len(r) else None
                debit_raw = r[col_idx_map["debit"]] if "debit" in col_idx_map and col_idx_map["debit"] < len(r) else None
                credit_raw = r[col_idx_map["credit"]] if "credit" in col_idx_map and col_idx_map["credit"] < len(r) else None
                bal_raw = r[col_idx_map["balance"]] if "balance" in col_idx_map and col_idx_map["balance"] < len(r) else None

                txn_date = parse_date_value(str(txn_date_raw)) or (clean_cell_text(str(txn_date_raw)) if txn_date_raw else None)
                value_date = parse_date_value(str(value_date_raw)) or (clean_cell_text(str(value_date_raw)) if value_date_raw else None)
                desc = clean_cell_text(str(desc_raw or ""))
                ref_no = clean_cell_text(str(ref_raw or "")) if ref_raw else None
                detail_info = clean_cell_text(str(info_raw or "")) if info_raw else None
                debit = parse_numeric_value(str(debit_raw)) if debit_raw is not None else None
                credit = parse_numeric_value(str(credit_raw)) if credit_raw is not None else None
                balance = parse_numeric_value(str(bal_raw)) if bal_raw is not None else None

                raw_row_values = "\x1f".join(clean_cell_text(str(value)) for value in r)
                raw_hash_str = f"{source_doc}|{page_number}|{header_id}|{txn_date or ''}|{desc}|{ref_no or ''}|{debit or 0}|{credit or 0}|{balance or 0}|{raw_row_values}"
                row_hash = hashlib.sha256(raw_hash_str.encode("utf-8")).hexdigest()[:24]

                vals: list[Any] = [
                    header_id,
                    txn_date,
                    value_date,
                    desc,
                    ref_no,
                    detail_info,
                    debit,
                    credit,
                    balance,
                    source_doc,
                    page_number,
                    row_hash,
                    now_iso,
                ]
                for col_k in extra_col_keys:
                    e_val = None
                    for e_name, e_idx in extra_indices:
                        if e_name == col_k and e_idx < len(r):
                            e_val = clean_cell_text(r[e_idx])
                            break
                    vals.append(e_val)

                cur = conn.execute(sql, vals)
                if cur.rowcount > 0:
                    inserted_count += cur.rowcount

        return inserted_count

    def ingest_relational_signatories(
        self,
        header_id: int,
        headers: list[str],
        rows: list[list[str]],
        source_doc: str = "",
        page_number: int | None = None,
    ) -> int:
        """
        Ingest baris penandatangan / persetujuan dokumen ke document_signatories (FK header_id).
        Field 'position' (jabatan) bersifat nullable opsional.
        """
        self.ensure_relational_schema()
        _, col_idx_map, extra_cols = defensive_map_columns(
            headers, SIGNATORY_CANONICAL_ALIASES, "signatory"
        )
        if extra_cols:
            self.ensure_columns_exist("document_signatories", extra_cols)

        now_iso = datetime.now(UTC).isoformat()
        inserted_count = 0

        base_cols = [
            "header_id",
            "role_party",
            "name",
            "position",
            "date",
            "signature_status",
            "notes",
            "_source_doc",
            "_page_number",
            "_row_hash",
            "_ingested_at",
        ]
        extra_col_keys = list(extra_cols.keys())
        all_insert_cols = base_cols + extra_col_keys
        placeholders = ", ".join(["?"] * len(all_insert_cols))
        quoted_cols = ", ".join([f'"{c}"' for c in all_insert_cols])
        sql = f'INSERT OR IGNORE INTO document_signatories ({quoted_cols}) VALUES ({placeholders});'

        extra_indices = [
            (sanitize_identifier(h), i)
            for i, h in enumerate(headers)
            if sanitize_identifier(h) in extra_cols
        ]

        with self._get_connection() as conn:
            for r in rows:
                role_raw = r[col_idx_map["role_party"]] if "role_party" in col_idx_map and col_idx_map["role_party"] < len(r) else None
                name_raw = r[col_idx_map["name"]] if "name" in col_idx_map and col_idx_map["name"] < len(r) else ""
                pos_raw = r[col_idx_map["position"]] if "position" in col_idx_map and col_idx_map["position"] < len(r) else None
                date_raw = r[col_idx_map["date"]] if "date" in col_idx_map and col_idx_map["date"] < len(r) else None
                sig_raw = r[col_idx_map["signature_status"]] if "signature_status" in col_idx_map and col_idx_map["signature_status"] < len(r) else None
                notes_raw = r[col_idx_map["notes"]] if "notes" in col_idx_map and col_idx_map["notes"] < len(r) else None

                role = clean_cell_text(str(role_raw or "")) if role_raw else None
                name = clean_cell_text(str(name_raw or ""))
                position = clean_cell_text(str(pos_raw or "")) if pos_raw else None
                sig_date = parse_date_value(str(date_raw)) or (clean_cell_text(str(date_raw)) if date_raw else None)
                sig_status = clean_cell_text(str(sig_raw or "")) if sig_raw else None
                notes = clean_cell_text(str(notes_raw or "")) if notes_raw else None

                raw_hash_str = f"{source_doc}|{page_number}|{header_id}|{role or ''}|{name}|{position or ''}|{sig_date}|{sig_status}|{notes}"
                row_hash = hashlib.sha256(raw_hash_str.encode("utf-8")).hexdigest()[:24]

                vals: list[Any] = [
                    header_id,
                    role,
                    name,
                    position,
                    sig_date,
                    sig_status,
                    notes,
                    source_doc,
                    page_number,
                    row_hash,
                    now_iso,
                ]
                for col_k in extra_col_keys:
                    e_val = None
                    for e_name, e_idx in extra_indices:
                        if e_name == col_k and e_idx < len(r):
                            e_val = clean_cell_text(r[e_idx])
                            break
                    vals.append(e_val)

                cur = conn.execute(sql, vals)
                if cur.rowcount > 0:
                    inserted_count += cur.rowcount

        return inserted_count

    def merge_and_deduplicate_tables(
        self,
        table_name: str = "transaction_details",
        match_columns: list[str] | None = None,
    ) -> DedupReport:
        """
        Fungsi merge & deduplikasi untuk dokumen sejenis / multiple file:
        1. Mengidentifikasi baris duplikat berdasarkan kolom kunci bisnis.
        2. Mengonsolidasikan (merge) nilai non-null dari baris duplikat ke baris primer.
        3. Menghapus baris duplikat yang redundan.
        4. Mengembalikan laporan ringkasan DedupReport.
        """
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name = ?;",
                (table_name,),
            )
            if not cur.fetchone():
                return DedupReport(
                    table_name=table_name,
                    initial_rows=0,
                    deduped_rows=0,
                    duplicates_removed=0,
                    details=f"Tabel '{table_name}' tidak ditemukan di database.",
                )

            col_cur = conn.execute(f'PRAGMA table_info("{table_name}");')
            col_infos = col_cur.fetchall()
            all_cols = [r["name"] for r in col_infos]
            if not all_cols:
                return DedupReport(
                    table_name=table_name,
                    initial_rows=0,
                    deduped_rows=0,
                    duplicates_removed=0,
                    details=f"Tabel '{table_name}' tidak ditemukan atau kosong skemanya.",
                )

            initial_count = conn.execute(
                f'SELECT COUNT(*) as cnt FROM "{table_name}";'
            ).fetchone()["cnt"]

            pk_candidates = [r["name"] for r in col_infos if r["pk"] > 0]
            if pk_candidates:
                pk_col = pk_candidates[0]
                is_internal_pk = False
                select_pk_expr = f'"{pk_col}"'
                order_pk_expr = f'"{pk_col}"'
            elif "_row_id" in all_cols:
                pk_col = "_row_id"
                is_internal_pk = False
                select_pk_expr = '"_row_id"'
                order_pk_expr = '"_row_id"'
            else:
                pk_col = "_internal_pk"
                is_internal_pk = True
                select_pk_expr = 'rowid AS "_internal_pk"'
                order_pk_expr = 'rowid'

            if not match_columns:
                if table_name == "transaction_details":
                    candidates = ["header_id", "txn_date", "description", "debit", "credit", "balance"]
                    match_columns = [c for c in candidates if c in all_cols]
                elif table_name == "document_signatories":
                    candidates = ["header_id", "role_party", "name"]
                    match_columns = [c for c in candidates if c in all_cols]
                elif table_name == "document_headers":
                    match_columns = ["fingerprint_hash"] if "fingerprint_hash" in all_cols else []
                else:
                    candidates = [c for c in all_cols if not c.startswith("_") and c != pk_col]
                    match_columns = candidates[: min(4, len(candidates))]

            if not match_columns:
                return DedupReport(
                    table_name=table_name,
                    initial_rows=initial_count,
                    deduped_rows=initial_count,
                    duplicates_removed=0,
                    details="Tidak ada kolom acuan pembanding deduplikasi yang tersedia.",
                )

            group_by_str = ", ".join([f'"{c}"' for c in match_columns])
            where_clause = " AND ".join([f'"{c}" IS ?' for c in match_columns])
            non_null_clause = " AND ".join([f'"{c}" IS NOT NULL' for c in match_columns])

            dup_groups = conn.execute(f"""
                SELECT {group_by_str}, COUNT(*) as cnt
                FROM "{table_name}"
                WHERE {non_null_clause}
                GROUP BY {group_by_str}
                HAVING cnt > 1;
            """).fetchall()

            removed_count = 0
            for grp in dup_groups:
                vals = tuple(grp[c] for c in match_columns)
                rows = conn.execute(f"""
                    SELECT {select_pk_expr}, * FROM "{table_name}"
                    WHERE {where_clause}
                    ORDER BY {order_pk_expr} ASC;
                """, vals).fetchall()

                if len(rows) <= 1:
                    continue

                base_row = dict(rows[0])
                base_id = base_row[pk_col]

                for dup in rows[1:]:
                    dup_dict = dict(dup)
                    dup_id = dup_dict[pk_col]

                    for col_name, val in dup_dict.items():
                        if (
                            col_name != pk_col
                            and not col_name.startswith("_internal")
                            and base_row.get(col_name) is None
                            and val is not None
                        ):
                            if is_internal_pk:
                                conn.execute(
                                    f'UPDATE "{table_name}" SET "{col_name}" = ? WHERE rowid = ?;',
                                    (val, base_id),
                                )
                            else:
                                conn.execute(
                                    f'UPDATE "{table_name}" SET "{col_name}" = ? WHERE "{pk_col}" = ?;',
                                    (val, base_id),
                                )
                            base_row[col_name] = val

                    if table_name == "document_headers":
                        conn.execute(
                            'UPDATE transaction_details SET header_id = ? WHERE header_id = ?;',
                            (base_id, dup_id),
                        )
                        conn.execute(
                            'UPDATE document_signatories SET header_id = ? WHERE header_id = ?;',
                            (base_id, dup_id),
                        )

                    if is_internal_pk:
                        conn.execute(
                            f'DELETE FROM "{table_name}" WHERE rowid = ?;',
                            (dup_id,),
                        )
                    else:
                        conn.execute(
                            f'DELETE FROM "{table_name}" WHERE "{pk_col}" = ?;',
                            (dup_id,),
                        )
                    removed_count += 1

            final_count = conn.execute(
                f'SELECT COUNT(*) as cnt FROM "{table_name}";'
            ).fetchone()["cnt"]

            details = (
                f"Deduplikasi tabel '{table_name}' berhasil: {removed_count} baris duplikat digabung & dibersihkan "
                f"(awal: {initial_count}, akhir: {final_count} baris)."
            )
            logger.info("[Tabular:Dedup] %s", details)

            return DedupReport(
                table_name=table_name,
                initial_rows=initial_count,
                deduped_rows=final_count,
                duplicates_removed=removed_count,
                details=details,
            )

    def inspect_database(self) -> dict[str, Any]:
        """Ambil daftar seluruh tabel dan skema kolom yang ada di database."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            )
            tables = [row["name"] for row in cursor.fetchall()]

            table_details = {}
            for t in tables:
                col_cur = conn.execute(f"PRAGMA table_info('{t}');")
                cols = [dict(c) for c in col_cur.fetchall()]
                count_cur = conn.execute(f"SELECT COUNT(*) as cnt FROM '{t}';")
                cnt = count_cur.fetchone()["cnt"]
                sample_cur = conn.execute(f"SELECT * FROM '{t}' LIMIT 3;")
                samples = [dict(r) for r in sample_cur.fetchall()]
                table_details[t] = {
                    "columns": cols,
                    "row_count": cnt,
                    "samples": samples,
                }

        return {
            "database_path": str(self.db_path),
            "tables": table_details,
            "active_tables_summary": self.get_active_tables_summary(),
        }

    def export_to_csv(
        self,
        output_dir: str | Path | None = None,
        include_internal_columns: bool = False,
    ) -> list[Path]:
        """
        Ekspor seluruh tabel pengguna yang ada di basis data SQLite ke file CSV.

        Args:
            output_dir: Folder target untuk menyimpan file CSV (default: folder 'csv' per dokumen).
            include_internal_columns: Apakah kolom metadata internal (_row_id, _source_doc, dll) ikut diekspor.

        Returns:
            Daftar Path file CSV yang berhasil dibuat.
        """
        import csv

        if not self.db_path.exists():
            return []

        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            )
            tables = [row["name"] for row in cursor.fetchall()]
            if not tables:
                return []

            if output_dir:
                target_dir = Path(output_dir).resolve()
            elif self.db_path.parent.name == "databases":
                target_dir = self.db_path.parent.parent / "csv"
            else:
                target_dir = self.db_path.parent / "csv"

            target_dir.mkdir(parents=True, exist_ok=True)
            exported_files: list[Path] = []

            for t in tables:
                col_cur = conn.execute(f'PRAGMA table_info("{t}");')
                all_cols = [c["name"] for c in col_cur.fetchall()]
                if not include_internal_columns:
                    cols_to_export = [c for c in all_cols if not c.startswith("_")]
                    if not cols_to_export:
                        cols_to_export = all_cols
                else:
                    cols_to_export = all_cols

                quoted_cols = ", ".join([f'"{c}"' for c in cols_to_export])
                data_cur = conn.execute(f'SELECT {quoted_cols} FROM "{t}";')
                rows = data_cur.fetchall()

                csv_file = target_dir / f"{t}.csv"
                with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(cols_to_export)
                    for r in rows:
                        writer.writerow([r[c] for c in cols_to_export])

                exported_files.append(csv_file)
                logger.info(
                    "[Tabular:CSV] Tabel '%s' (%d baris) berhasil diekspor ke: %s",
                    t,
                    len(rows),
                    csv_file,
                )

        return exported_files



# ==============================================================================
# Double-Verification Mechanism (Integritas, Agregasi, Refleksi)
# ==============================================================================


class TabularVerifier:
    """
    Engine Verifikasi Ganda (Double-Verification):
      1. Integritas baris & skema kolom.
      2. Uji coba query kalkulasi agregat (SUM & AVG).
      3. Uji konsistensi aritmatika saldo rekening (Balance[n-1] + Credit - Debit ≈ Balance[n]).
      4. Refleksi verifikasi model LLM bila tersedia.
    """

    def __init__(
        self, db_manager: TabularDatabaseManager, llm: BaseChatModel | None = None
    ) -> None:
        self.db = db_manager
        self.llm = llm

    def verify_table(
        self,
        table_name: str,
        expected_row_count: int | None = None,
        source_markdown_sample: str | None = None,
    ) -> TableVerificationReport:
        checks: list[VerificationCheck] = []
        test_queries: list[dict[str, Any]] = []

        # 1. Pengecekan Integritas Baris (Row Count Integrity)
        count_res = self.db.execute_query(f'SELECT COUNT(*) as cnt FROM "{table_name}"')
        actual_rows = count_res.rows[0]["cnt"] if count_res.rows else 0
        test_queries.append(
            {
                "query": f'SELECT COUNT(*) FROM "{table_name}"',
                "result": actual_rows,
                "success": count_res.status == "success",
            }
        )

        if expected_row_count is not None:
            row_count_ok = actual_rows == expected_row_count
            checks.append(
                VerificationCheck(
                    check_name="row_count_integrity",
                    passed=row_count_ok,
                    details=f"SQLite memiliki {actual_rows} baris (Ekspektasi: {expected_row_count} baris).",
                    metric_value=actual_rows,
                )
            )
        else:
            checks.append(
                VerificationCheck(
                    check_name="row_count_present",
                    passed=actual_rows > 0,
                    details=f"Tabel memiliki {actual_rows} baris tersimpan.",
                    metric_value=actual_rows,
                )
            )

        # 2. Pengecekan Skema Kolom
        inspect = self.db.inspect_database()
        t_info = inspect["tables"].get(table_name, {})
        cols_info = t_info.get("columns", [])
        col_names = [c["name"] for c in cols_info if not c["name"].startswith("_")]

        checks.append(
            VerificationCheck(
                check_name="columns_registered",
                passed=len(col_names) > 0,
                details=f"Tabel memiliki {len(col_names)} kolom terdefinisi: {col_names}",
                metric_value=len(col_names),
            )
        )

        # 3. Numeric Aggregation Sanity Test (Uji SUM & AVG)
        numeric_cols = [
            c["name"]
            for c in cols_info
            if c["type"] in ("REAL", "NUMERIC", "INTEGER")
            and not c["name"].startswith("_")
        ]
        agg_passed = True
        agg_details: Any = {}

        if numeric_cols:
            agg_selects = []
            for nc in numeric_cols:
                agg_selects.append(f'SUM("{nc}") as sum_{nc}')
                agg_selects.append(f'AVG("{nc}") as avg_{nc}')
            agg_sql = f'SELECT {", ".join(agg_selects)} FROM "{table_name}";'

            agg_res = self.db.execute_query(agg_sql)
            if agg_res.status == "error":
                agg_passed = False
                agg_details = {"error": agg_res.error_message}
                test_queries.append(
                    {
                        "query": agg_sql,
                        "result": str(agg_res.error_message),
                        "success": False,
                    }
                )
            else:
                agg_details = agg_res.rows[0] if agg_res.rows else {}
                test_queries.append(
                    {"query": agg_sql, "result": agg_details, "success": True}
                )

            checks.append(
                VerificationCheck(
                    check_name="numeric_aggregation_sanity",
                    passed=agg_passed,
                    details=f"Query kalkulasi agregat (SUM & AVG) berjalan sukses tanpa error: {agg_details}"
                    if agg_passed
                    else f"Gagal menjalankan query agregat: {agg_details}",
                )
            )

        # 4. Bank Statement Balance Continuity Check (Balance[n-1] + Kredit - Debit ≈ Balance[n])
        has_debit = any("debit" in c.lower() for c in col_names)
        has_credit = any(
            "credit" in c.lower() or "kredit" in c.lower() for c in col_names
        )
        has_balance = any(
            "balance" in c.lower() or "saldo" in c.lower() for c in col_names
        )

        if has_debit and has_credit and has_balance:
            debit_col = next(c for c in col_names if "debit" in c.lower())
            credit_col = next(
                c for c in col_names if "credit" in c.lower() or "kredit" in c.lower()
            )
            balance_col = next(
                c for c in col_names if "balance" in c.lower() or "saldo" in c.lower()
            )

            sample_rows_res = self.db.execute_query(
                f'SELECT "{debit_col}", "{credit_col}", "{balance_col}" FROM "{table_name}" ORDER BY rowid ASC LIMIT 20;'
            )
            s_rows = sample_rows_res.rows
            continuity_mismatches = 0
            if len(s_rows) >= 2:
                for idx in range(1, len(s_rows)):
                    val_prev = s_rows[idx - 1].get(balance_col)
                    val_deb = s_rows[idx].get(debit_col)
                    val_crd = s_rows[idx].get(credit_col)
                    val_bal = s_rows[idx].get(balance_col)

                    prev_bal = (
                        val_prev
                        if isinstance(val_prev, (int, float))
                        else (parse_numeric_value(str(val_prev)) if val_prev is not None else 0.0)
                    ) or 0.0
                    curr_deb = (
                        val_deb
                        if isinstance(val_deb, (int, float))
                        else (parse_numeric_value(str(val_deb)) if val_deb is not None else 0.0)
                    ) or 0.0
                    curr_crd = (
                        val_crd
                        if isinstance(val_crd, (int, float))
                        else (parse_numeric_value(str(val_crd)) if val_crd is not None else 0.0)
                    ) or 0.0
                    curr_bal = (
                        val_bal
                        if isinstance(val_bal, (int, float))
                        else (parse_numeric_value(str(val_bal)) if val_bal is not None else 0.0)
                    ) or 0.0

                    expected_bal = float(prev_bal) + float(curr_crd) - float(curr_deb)
                    # Toleransi selisih floating point 1.0 (karena pembulatan sen)
                    if abs(expected_bal - float(curr_bal)) > 1.0:
                        continuity_mismatches += 1

            checks.append(
                VerificationCheck(
                    check_name="balance_arithmetic_consistency",
                    passed=continuity_mismatches == 0,
                    details=f"Pemeriksaan kontinuitas saldo (Balance[n-1] + Kredit - Debit = Balance[n]): {continuity_mismatches} ketidaksesuaian dari {len(s_rows)} baris sampel.",
                    metric_value=continuity_mismatches,
                )
            )

        # 5. LLM Reflection (Optional)
        reflection_notes: str | None = None
        if self.llm is not None and source_markdown_sample:
            try:
                sample_db_rows = self.db.execute_query(
                    f'SELECT * FROM "{table_name}" LIMIT 5;'
                ).rows
                prompt = (
                    "Lakukan verifikasi refleksi antara teks sampel Markdown asli dan record data SQLite hasil ekstraksi berikut.\n"
                    f"Sample Markdown Dokumen:\n```\n{source_markdown_sample[:1000]}\n```\n\n"
                    f"Sample Database SQLite Rows:\n```json\n{json.dumps(sample_db_rows, indent=2)}\n```\n\n"
                    "Apakah ada kolom atau angka penting yang terpotong/salah letak? Jawab ringkas 2-3 kalimat."
                )
                resp = self.llm.invoke(prompt)
                if hasattr(resp, "content"):
                    raw_c = resp.content
                    reflection_notes = (
                        raw_c
                        if isinstance(raw_c, str)
                        else json.dumps(raw_c, ensure_ascii=False)
                    )
                else:
                    reflection_notes = str(resp)
            except Exception as e:  # noqa: BLE001
                reflection_notes = f"LLM Reflection skipped: {e}"

        all_passed = all(c.passed for c in checks)
        passed_ratio = sum(1 for c in checks if c.passed) / max(1, len(checks))
        confidence_score = (
            round(0.5 + (passed_ratio * 0.5), 2)
            if all_passed
            else round(passed_ratio * 0.75, 2)
        )
        status_lit: Any = (
            "verified"
            if all_passed
            else ("needs_revision" if passed_ratio >= 0.6 else "rejected")
        )

        return TableVerificationReport(
            table_name=table_name,
            database_path=str(self.db.db_path),
            is_valid=all_passed,
            verification_status=status_lit,
            confidence_score=confidence_score,
            checks=checks,
            verified_row_count=actual_rows,
            test_queries=test_queries,
            llm_reflection=reflection_notes,
            summary=f"Verifikasi data tabel '{table_name}' selesai dengan status '{status_lit}'. Total {len(checks)} pemeriksaan dijalankan ({sum(1 for c in checks if c.passed)} passed).",
        )


# ==============================================================================
# Sub-Agent SQL Per-Halaman & Dual-Track Guardrail
# ==============================================================================


def prune_document_pages(db_path: str | Path, source_file: str, total_pages: int) -> None:
    """Buang baris halaman yang tidak lagi ada setelah ekstraksi dokumen berhasil."""
    if not source_file or total_pages < 1:
        return
    manager = TabularDatabaseManager(db_path)
    with manager._get_connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        for table in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            columns = {r[1] for r in conn.execute(f'PRAGMA table_info({quoted})')}
            if {'_source_doc', '_page_number'} <= columns:
                conn.execute(f'DELETE FROM {quoted} WHERE _source_doc=? AND _page_number>?', (source_file, total_pages))


def process_page_tabular_agent(
    page_markdown: str,
    page_number: int,
    source_file: str = "",
    db_path: str | Path | None = None,
    table_name_prefix: str | None = None,
    append_if_matching: bool = True,
    force_all_tables: bool = False,
    llm: BaseChatModel | None = None,
) -> tuple[PageTabularEvent, list[TableIngestionResult]]:
    """Ganti data halaman secara atomik; halaman dan dokumen lain tetap dipertahankan."""
    manager = TabularDatabaseManager(db_path)
    with manager._get_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        token = _active_connection.set((str(manager.db_path.resolve()), conn))
        try:
            if source_file:
                tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
                for table in tables:
                    quoted = '"' + table.replace('"', '""') + '"'
                    columns = {r[1] for r in conn.execute(f"PRAGMA table_info({quoted})")}
                    if {"_source_doc", "_page_number"} <= columns:
                        conn.execute(f"DELETE FROM {quoted} WHERE _source_doc=? AND _page_number=?", (source_file, page_number))
            return _process_page_tabular_agent(
                page_markdown, page_number, source_file, manager.db_path,
                table_name_prefix, append_if_matching, force_all_tables, llm,
            )
        finally:
            _active_connection.reset(token)


def _process_page_tabular_agent(
    page_markdown: str,
    page_number: int,
    source_file: str = "",
    db_path: str | Path | None = None,
    table_name_prefix: str | None = None,
    append_if_matching: bool = True,
    force_all_tables: bool = False,
    llm: BaseChatModel | None = None,
) -> tuple[PageTabularEvent, list[TableIngestionResult]]:
    """
    Sub-Agent SQL Tabular Engine yang berjalan mandiri per-halaman/slide:
      1. Memahami konten seluruh tabel pada halaman secara independen.
      2. Menginspeksi tabel dan skema eksisting yang sudah ada di database SQLite.
      3. Melakukan query SQL awal jika ada tabel kelanjutan (multi-page continuation table).
      4. Meng-ingest atau meng-append data baris halaman ini ke tabel SQLite yang sesuai.
      5. Menyematkan metadata tag <!-- sqlite_table: <table_name> --> pada Markdown.
      6. Menjalankan query SQL mandiri untuk memverifikasi kondisi tabel setelah ingesti.
    """
    db_manager = TabularDatabaseManager(db_path)
    verifier = TabularVerifier(db_manager, llm=llm)

    # 1. Pahami tabel pada halaman
    parsed_tables = parse_markdown_tables(page_markdown)
    if parsed_tables:
        logger.info(
            "Sub-Agent SQL [Halaman %d]: Terdeteksi %d tabel, mengevaluasi ingesti ke SQLite...",
            page_number,
            len(parsed_tables),
        )
    src_stem = Path(source_file).stem if source_file else "doc"
    base_prefix = sanitize_identifier(table_name_prefix or src_stem)

    # Inisialisasi skema relasional kanonikal & ingest header dokumen
    db_manager.ensure_relational_schema()
    header_rec = extract_document_header_info(page_markdown, source_file=source_file)
    header_id = db_manager.ingest_document_header(header_rec)

    # 2. Inspeksi tabel eksisting di SQLite
    db_info = db_manager.inspect_database()
    existing_tables = list(db_info.get("tables", {}).keys())

    if not parsed_tables:
        event = PageTabularEvent(
            page_number=page_number,
            tables_detected=0,
            existing_tables_inspected=existing_tables,
            actions_taken=["Tidak ditemukan format tabel pada halaman ini."],
            queries_executed=[],
            rows_ingested_total=0,
            status="no_tables",
            tagged_markdown=page_markdown,
        )
        return event, []

    actions: list[str] = []
    queries_run: list[dict[str, Any]] = []
    ingestion_results: list[TableIngestionResult] = []
    table_mappings_for_tagging: list[tuple[dict[str, Any], str]] = []
    total_page_rows = 0
    event_status: Any = "no_tables"

    for idx, tbl in enumerate(parsed_tables, start=1):
        headers = tbl["headers"]
        rows = tbl["rows"]
        context = tbl["context"]
        sqlite_hint = tbl.get("sqlite_table_hint")

        if not rows:
            continue

        # Klasifikasi semantik tabel (untuk logging & metadata)
        classification = classify_table_heuristic(headers, rows, context=context)

        # Deteksi domain relasional kanonikal
        is_canon_txn, _, _ = defensive_map_columns(headers, TRANSACTION_CANONICAL_ALIASES, "transaction")
        is_canon_sig, _, _ = defensive_map_columns(headers, SIGNATORY_CANONICAL_ALIASES, "signatory")

        is_relational_txn = sqlite_hint == "transaction_details" or is_canon_txn
        is_relational_sig = (
            sqlite_hint == "document_signatories"
            or is_canon_sig
            or any("pihak" in h.lower() or "tanda tangan" in h.lower() or "jabatan" in h.lower() for h in headers)
        )

        if is_relational_txn:
            target_table = "transaction_details"
            is_append = True
            rows_ingested = db_manager.ingest_relational_transactions(
                header_id=header_id,
                headers=headers,
                rows=rows,
                source_doc=source_file,
                page_number=page_number,
            )
            total_page_rows += rows_ingested
            actions.append(
                f"Tabel #{idx} teridentifikasi sebagai detail transaksi. Ingest ke 'transaction_details' (FK header_id={header_id}, +{rows_ingested} baris baru/deduped)."
            )
            inspect_t = db_manager.inspect_database()
            cols_info = inspect_t["tables"].get("transaction_details", {}).get("columns", [])
            schema = TableSchema(
                table_name="transaction_details",
                source_file=source_file,
                columns=[
                    TableColumnSchema(
                        name=c["name"],
                        original_name=c["name"],
                        sql_type=c["type"],
                        is_nullable=True,
                        description="",
                        sample_values=[],
                    )
                    for c in cols_info
                    if not c["name"].startswith("_")
                ],
            )
        elif is_relational_sig:
            target_table = "document_signatories"
            is_append = True
            rows_ingested = db_manager.ingest_relational_signatories(
                header_id=header_id,
                headers=headers,
                rows=rows,
                source_doc=source_file,
                page_number=page_number,
            )
            total_page_rows += rows_ingested
            actions.append(
                f"Tabel #{idx} teridentifikasi sebagai penandatangan/persetujuan. Ingest ke 'document_signatories' (FK header_id={header_id}, +{rows_ingested} baris baru/deduped)."
            )
            inspect_t = db_manager.inspect_database()
            cols_info = inspect_t["tables"].get("document_signatories", {}).get("columns", [])
            schema = TableSchema(
                table_name="document_signatories",
                source_file=source_file,
                columns=[
                    TableColumnSchema(
                        name=c["name"],
                        original_name=c["name"],
                        sql_type=c["type"],
                        is_nullable=True,
                        description="",
                        sample_values=[],
                    )
                    for c in cols_info
                    if not c["name"].startswith("_")
                ],
            )
        elif force_all_tables:
            matched = None
            if append_if_matching:
                matched = db_manager.find_matching_table(
                    headers=headers,
                    explicit_table_name=sqlite_hint,
                    doc_prefix=base_prefix,
                )

            if matched:
                target_table, schema = matched
                is_append = True
                prior_query = f'SELECT COUNT(*) as prev_cnt FROM "{target_table}";'
                prior_res = db_manager.execute_query(prior_query)
                prev_count = prior_res.rows[0]["prev_cnt"] if prior_res.rows else 0
                queries_run.append(
                    {
                        "query": prior_query,
                        "purpose": "Inspeksi baris sebelum penambahan data",
                        "result": prev_count,
                    }
                )
                actions.append(
                    f"Tabel #{idx} cocok dengan tabel eksisting '{target_table}' ({prev_count} baris awal). Menambahkan {len(rows)} baris halaman {page_number}."
                )
            else:
                target_table = sqlite_hint or f"{base_prefix}_t{idx}"
                schema = infer_table_schema(
                    table_name=target_table,
                    headers=headers,
                    rows=rows,
                    source_file=source_file,
                    metadata={"context": context, "page": page_number, "classification": classification.table_type},
                )
                is_append = False
                actions.append(
                    f"Tabel #{idx} adalah tabel baru ({classification.table_type}). Membuat skema tabel '{target_table}' dengan {len(schema.columns)} kolom."
                )

            # Ingest baris data untuk tabel umum
            rows_ingested = db_manager.ingest_records(
                table_name=target_table,
                schema=schema,
                headers=headers,
                rows=rows,
                source_doc=source_file,
                page_number=page_number,
                replace=not is_append,
            )
            total_page_rows += rows_ingested

        else:
            actions.append(
                f"Tabel #{idx} ({classification.table_type}) dilewati karena bukan tabel transaksional atau penandatangan."
            )
            continue

        table_mappings_for_tagging.append((tbl, target_table))

        # Jalankan query SQL mandiri untuk verifikasi baris setelah ingesti
        post_query = f'SELECT COUNT(*) as total_cnt FROM "{target_table}";'
        post_res = db_manager.execute_query(post_query)
        total_cnt = post_res.rows[0]["total_cnt"] if post_res.rows else rows_ingested
        queries_run.append(
            {
                "query": post_query,
                "purpose": "Verifikasi total baris setelah ingesti",
                "result": total_cnt,
            }
        )

        # Jalankan double-verification
        sample_md = "\n".join(["|".join(headers)] + ["|".join(r) for r in rows[:5]])
        verif_report = verifier.verify_table(
            table_name=target_table,
            expected_row_count=None if is_append else len(rows),
            source_markdown_sample=sample_md,
        )

        sample_data = db_manager.execute_query(
            f'SELECT * FROM "{target_table}" ORDER BY 1 DESC LIMIT 3;'
        ).rows

        status_str: Any = "success" if verif_report.is_valid else "warning"
        ingestion_results.append(
            TableIngestionResult(
                status=status_str,
                table_name=target_table,
                database_path=str(db_manager.db_path),
                total_rows_ingested=rows_ingested,
                columns=[col.name for col in schema.columns],
                verification_report=verif_report,
                sample_data=sample_data,
                message=(
                    f"Tabel bersambung '{target_table}' halaman {page_number} berhasil ditambah (+{rows_ingested} baris, total {total_cnt})."
                    if is_append
                    else f"Tabel '{target_table}' halaman {page_number} dibuat ({rows_ingested} baris)."
                ),
            )
        )

        event_status = "appended_existing_table" if is_append else "created_new_table"

    if total_page_rows > 0:
        logger.info(
            "Sub-Agent SQL [Halaman %d]: Sukses ingesti total %d baris ke tabel SQLite.",
            page_number,
            total_page_rows,
        )

    # Sematkan metadata tag <!-- sqlite_table: <name> --> ke Markdown halaman
    tagged_markdown = tag_markdown_tables_with_sqlite_metadata(page_markdown, table_mappings_for_tagging)

    event = PageTabularEvent(
        page_number=page_number,
        tables_detected=len(parsed_tables),
        existing_tables_inspected=existing_tables,
        actions_taken=actions,
        queries_executed=queries_run,
        rows_ingested_total=total_page_rows,
        status=event_status,
        tagged_markdown=tagged_markdown,
    )
    return event, ingestion_results


def _compare_table_values(manager: TabularDatabaseManager, md_tables: list[dict[str, Any]]) -> list[str]:
    """Bandingkan multiset nilai kolom sumber dengan data SQL, tanpa bergantung urutan baris."""
    discrepancies: list[str] = []
    groups: dict[str, list[dict[str, str]]] = {}
    with manager._get_connection() as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table in md_tables:
            headers = table['headers']
            txn, txn_map, _ = defensive_map_columns(headers, TRANSACTION_CANONICAL_ALIASES, "transaction")
            sig, sig_map, _ = defensive_map_columns(headers, SIGNATORY_CANONICAL_ALIASES, "signatory")
            target = table.get('sqlite_table_hint')
            mapping: dict[str, int] = {}
            if target == 'transaction_details' or (not target and txn):
                target, mapping = 'transaction_details', txn_map
            elif target == 'document_signatories' or (not target and sig):
                target, mapping = 'document_signatories', sig_map
            elif not target:
                match = manager.find_matching_table(headers)
                target = match[0] if match else None
            if not target or target not in tables:
                continue
            inverse = {index: name for name, index in mapping.items()}
            columns = tuple(inverse.get(i, sanitize_identifier(h)) for i, h in enumerate(headers))
            groups.setdefault(target, []).extend(
                {column: row[i] if i < len(row) else '' for i, column in enumerate(columns)}
                for row in table['rows']
            )

        for target, rows in groups.items():
            columns = tuple(dict.fromkeys(column for row in rows for column in row))
            if not columns:
                continue
            quoted = '"' + target.replace('"', '""') + '"'
            types = {r[1]: r[2].upper() for r in conn.execute(f'PRAGMA table_info({quoted})')}
            missing = set(columns) - types.keys()
            if missing:
                discrepancies.append(f"Tabel '{target}': kolom sumber tidak ditemukan di SQLite: {', '.join(sorted(missing))}.")
                continue

            def normalize(value: Any, column: str, from_sql: bool, *, column_types: dict[str, str] = types) -> str:
                if column_types[column] in ('REAL', 'NUMERIC', 'INTEGER'):
                    number = value if from_sql else parse_numeric_value(str(value))
                    return '' if number is None else format(float(number), '.12g')
                text = clean_cell_text(str(value)) if value is not None else ''
                if column_types[column] == 'DATE' or column in ('txn_date', 'value_date', 'date'):
                    text = parse_date_value(text) or text
                return text or ''

            expected = Counter(tuple(normalize(row.get(c, ''), c, False) for c in columns) for row in rows)
            selected = ', '.join('"' + c.replace('"', '""') + '"' for c in columns)
            actual = Counter(tuple(normalize(value, c, True) for value, c in zip(row, columns)) for row in conn.execute(f'SELECT {selected} FROM {quoted}'))
            if expected != actual:
                discrepancies.append(f"Tabel '{target}': isi nilai berbeda (baris sumber tidak cocok: {sum((expected - actual).values())}; baris SQL tidak cocok: {sum((actual - expected).values())}).")
    return discrepancies


def cross_verify_dual_track(
    stitched_markdown: str,
    db_path: str | Path | None,
    source_file: str = "",
    total_pages: int = 1,
) -> DualTrackGuardrailReport:
    """
    Supervisor / Master Agent Guardrail Audit:
    Membandingkan Jalur 1 (Teks Markdown hasil VLM/OCR) vs Jalur 2 (Tabel di SQLite hasil Sub-Agent Tabular).
    Memverifikasi kesesuaian baris, kolom, dan integritas perhitungan untuk memastikan tidak ada data yang terlewat atau terkorupsi.
    """
    db_manager = TabularDatabaseManager(db_path)
    db_info = db_manager.inspect_database()
    sqlite_tables = db_info.get("tables", {})

    md_tables = parse_markdown_tables(stitched_markdown)
    total_md_tables = len(md_tables)
    total_sqlite_tables = len(sqlite_tables)

    # Petakan setiap tabel Markdown ke tabel SQLite tujuannya. Tabel relasional
    # tidak selalu memiliki tag, maka gunakan pemetaan kanonikal sebelum mencoba
    # pencocokan skema generik.
    md_rows_per_sqlite_table: dict[str, int] = {}
    for tbl in md_tables:
        t_target = tbl.get("sqlite_table_hint")
        is_txn, _, _ = defensive_map_columns(
            tbl["headers"], TRANSACTION_CANONICAL_ALIASES, "transaction"
        )
        is_signatory, _, _ = defensive_map_columns(
            tbl["headers"], SIGNATORY_CANONICAL_ALIASES, "signatory"
        )
        if not t_target and is_txn:
            t_target = "transaction_details"
        elif not t_target and is_signatory:
            t_target = "document_signatories"
        if not t_target and sqlite_tables:
            matched = db_manager.find_matching_table(tbl["headers"])
            if matched:
                t_target = matched[0]
            elif len(sqlite_tables) == 1:
                t_target = next(iter(sqlite_tables.keys()))

        if t_target:
            md_rows_per_sqlite_table[t_target] = md_rows_per_sqlite_table.get(t_target, 0) + len(tbl["rows"])

    total_md_rows = sum(len(t["rows"]) for t in md_tables)
    data_table_names = {
        name
        for name in sqlite_tables
        if name not in ("document_headers", "document_signatories")
    }
    total_sqlite_rows = sum(
        sqlite_tables[name].get("row_count", 0) for name in data_table_names
    )
    logger.info(
        "Guardrail Cross-Verification: Memeriksa integritas teks Markdown vs SQLite (%d tabel MD, %d tabel SQLite)...",
        total_md_tables,
        total_sqlite_tables,
    )

    table_comparisons: list[dict[str, Any]] = []
    discrepancies: list[str] = []

    for t_name, t_info in sqlite_tables.items():
        sql_rows = t_info.get("row_count", 0)
        expected_md_rows = md_rows_per_sqlite_table.get(t_name, 0)
        cols = [c["name"] for c in t_info.get("columns", []) if not c["name"].startswith("_")]

        if t_name == "document_headers":
            expected_md_rows = 1 if stitched_markdown.strip() else 0
            is_synced = sql_rows == expected_md_rows
            status_str = "synchronized" if is_synced else "row_count_mismatch"
        elif t_name == "document_signatories":
            is_synced = sql_rows == expected_md_rows
            status_str = "synchronized" if is_synced else "row_count_mismatch"
        else:
            is_synced = (sql_rows == expected_md_rows) or (sql_rows > 0 and expected_md_rows == 0 and total_md_rows == sql_rows)
            status_str = "synchronized" if is_synced else "row_count_mismatch"

        if not is_synced:
            discrepancies.append(
                f"Tabel SQLite '{t_name}' ({sql_rows} baris) berbeda dengan data Markdown terkait ({expected_md_rows} baris)."
            )

        table_comparisons.append(
            {
                "table_name": t_name,
                "sqlite_rows": sql_rows,
                "markdown_rows_matched": expected_md_rows,
                "columns_count": len(cols),
                "columns": cols,
                "status": status_str,
            }
        )

    # Cek discrepansi jumlah tabel & baris
    if total_md_tables > 0 and total_sqlite_tables == 0:
        discrepancies.append(
            f"Ditemukan {total_md_tables} tabel pada teks Markdown, namun tidak ada tabel yang masuk ke SQLite."
        )
    elif total_md_rows > 0 and total_sqlite_rows < (total_md_rows * 0.5):
        discrepancies.append(
            f"Terdapat selisih baris signifikan: Markdown ({total_md_rows} baris) vs SQLite ({total_sqlite_rows} baris)."
        )

    discrepancies.extend(_compare_table_values(db_manager, md_tables))

    # Tentukan status Guardrail Supervisor
    if not discrepancies and (total_sqlite_tables > 0 or total_md_tables == 0):
        guardrail_status: Any = "PASSED"
        supervisor_notes = (
            f"Audit Jalur Ganda Berhasil (PASSED). Seluruh {total_sqlite_tables} tabel SQLite "
            f"dengan total {total_sqlite_rows} baris data sinkron dan terverifikasi terhadap teks dokumen ({total_pages} halaman)."
        )
    elif total_sqlite_tables > 0:
        guardrail_status = "WARNING"
        supervisor_notes = (
            f"Audit Jalur Ganda dengan Peringatan (WARNING): Data SQLite terbentuk ({total_sqlite_tables} tabel, {total_sqlite_rows} baris), "
            f"namun ada catatan: {'; '.join(discrepancies)}"
        )
    else:
        guardrail_status = "PASSED" if total_md_tables == 0 else "WARNING"
        supervisor_notes = (
            "Dokumen tidak memiliki tabel aktif untuk SQLite."
            if total_md_tables == 0
            else f"Perhatian: {'; '.join(discrepancies)}"
        )

    logger.info(
        "Guardrail Cross-Verification selesai: Status '%s', Total baris MD: %d, Total baris SQLite: %d",
        guardrail_status,
        total_md_rows,
        total_sqlite_rows,
    )

    return DualTrackGuardrailReport(
        source_file=source_file,
        database_path=str(db_manager.db_path),
        total_pages_processed=total_pages,
        total_markdown_tables=total_md_tables,
        total_sqlite_tables=total_sqlite_tables,
        total_markdown_rows=total_md_rows,
        total_sqlite_rows=total_sqlite_rows,
        guardrail_status=guardrail_status,
        table_comparisons=table_comparisons,
        discrepancies=discrepancies,
        supervisor_notes=supervisor_notes,
    )


# ==============================================================================
# High-Level Pipeline Functions
# ==============================================================================


def extract_and_ingest_tables_from_markdown(
    markdown_text: str,
    source_file: str = "",
    db_path: str | Path | None = None,
    table_name_prefix: str | None = None,
    page_number: int | None = None,
    append_if_matching: bool = True,
    force_all_tables: bool = False,
    llm: BaseChatModel | None = None,
) -> list[TableIngestionResult]:
    """
    Ekstrak semua tabel dari Markdown, filter tabel transaksional / seluruh tabel,
    simpan/append ke database SQLite, dan jalankan double-verification otomatis.
    """
    _event, results = process_page_tabular_agent(
        page_markdown=markdown_text,
        page_number=page_number or 1,
        source_file=source_file,
        db_path=db_path,
        table_name_prefix=table_name_prefix,
        append_if_matching=append_if_matching,
        force_all_tables=force_all_tables,
        llm=llm,
    )
    return results


def query_sqlite(
    sql_query: str,
    db_path: str | Path | None = None,
) -> TabularQueryResult:
    """Eksekusi query SQL pada database SQLite dokumen secara aman."""
    db_mgr = TabularDatabaseManager(db_path)
    return db_mgr.execute_query(sql_query)


__all__ = [
    "TRANSACTIONAL_HEADER_KEYWORDS",
    "TabularDatabaseManager",
    "TabularVerifier",
    "classify_table_heuristic",
    "clean_cell_text",
    "cross_verify_dual_track",
    "extract_and_ingest_tables_from_markdown",
    "infer_table_schema",
    "parse_date_value",
    "parse_markdown_tables",
    "parse_numeric_value",
    "process_page_tabular_agent",
    "query_sqlite",
    "sanitize_identifier",
    "sanitize_markdown_tables",
]


def merge_and_deduplicate_tables(
    db_manager_or_path: TabularDatabaseManager | str | Path,
    table_name: str = "transaction_details",
    match_columns: list[str] | None = None,
) -> DedupReport:
    """
    Fungsi utilitas public untuk menggabungkan (merge) dan mendeduplikasi data tabel yang mirip/duplikat
    dari multiple file atau hasil ekstraksi ganda ke dalam satu tabel bersih.
    """
    if isinstance(db_manager_or_path, TabularDatabaseManager):
        mgr = db_manager_or_path
    else:
        mgr = TabularDatabaseManager(db_manager_or_path)
    return mgr.merge_and_deduplicate_tables(
        table_name=table_name, match_columns=match_columns
    )
