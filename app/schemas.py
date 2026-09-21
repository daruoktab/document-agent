"""
Pydantic Schemas untuk Document Vision VLM Extractor (Siap Chunking RAG),
Tabular SQLite Ingestion, Diagram Mermaid Extraction, dan Master-SubAgent System.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ==============================================================================
# Model-Model Klasifikasi & Ekstraksi Dokumen Dasar
# ==============================================================================


class ClassificationResult(BaseModel):
    """Hasil klasifikasi spesifikasi karakteristik dokumen."""

    specs: list[str] = Field(
        default_factory=lambda: ["plain"],
        description="Daftar spesifikasi layout yang terdeteksi",
    )
    confidence: float = Field(
        default=1.0, description="Tingkat keyakinan klasifikasi (0.0 - 1.0)"
    )
    reasoning: str = Field(
        default="", description="Alasan atau pertimbangan klasifikasi"
    )


class PageInspectionResult(BaseModel):
    """Hasil inspeksi awal satu halaman citra dokumen (specs, diagram, tabel, difficulty)."""

    specs: list[str] = Field(
        default_factory=lambda: ["plain"],
        description="Daftar karakteristik layout yang terdeteksi",
    )
    has_diagram: bool = Field(
        default=False, description="True jika terdapat diagram/visual pada halaman"
    )
    diagram_type: str | None = Field(
        default=None, description="Tipe diagram visual jika terdeteksi"
    )
    has_table: bool = Field(
        default=False, description="True jika terdapat tabel data/baris-kolom"
    )
    difficulty: Literal["simple", "standard", "complex"] = Field(
        default="standard", description="Estimasi tingkat kesulitan pemrosesan halaman"
    )
    confidence: float = Field(
        default=1.0, description="Tingkat keyakinan inspeksi (0.0 - 1.0)"
    )
    reasoning: str | None = Field(
        default=None, description="Catatan atau penalaran inspeksi"
    )
    document_title: str | None = Field(
        default=None, description="Judul utama dokumen jika terdeteksi (terutama di halaman 1)"
    )
    rotation_degrees: Literal[0, 90, 180, 270] = Field(
        default=0,
        description="Rotasi searah jarum jam yang diperlukan agar halaman tegak",
    )
    requires_vlm_reading: bool = Field(
        default=False,
        description="True bila VLM utama perlu mengekstrak halaman secara independen karena teks atau layout sulit dibaca",
    )

    def __getitem__(self, key: str) -> Any:
        """Kompatibilitas backward untuk akses dict: insp_res['specs']."""
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """Kompatibilitas backward untuk get dict: insp_res.get('has_diagram')."""
        return getattr(self, key, default)


class OCRRegion(BaseModel):
    """Satu region grounding hasil OCR yang sudah dipetakan ke piksel sumber."""

    index: int = Field(..., ge=1)
    label: str = Field(default="unknown")
    kind: Literal["text", "table", "figure", "unknown"] = Field(default="unknown")
    text: str = Field(default="")
    bbox_model: tuple[float, float, float, float]
    bbox_pixels: tuple[int, int, int, int]
    crop_path: str | None = None


class OCRQualityAssessment(BaseModel):
    """Sinyal kualitas OCR yang dapat diaudit sebelum hasilnya dipercaya."""

    score: float = Field(default=0.0, ge=0.0, le=1.0)
    trust_level: Literal["high", "medium", "low"] = Field(default="low")
    risk_flags: list[str] = Field(default_factory=list)
    ink_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    horizontal_energy: float = Field(default=0.0, ge=0.0)
    vertical_energy: float = Field(default=0.0, ge=0.0)
    native_text_similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    is_blank: bool = False
    is_sparse: bool = False
    right_angle_suspected: bool = False


class OCRExtractionResult(BaseModel):
    """Hasil OCR halaman, termasuk Markdown bersih dan region grounding."""

    status: Literal["success", "disabled", "error"] = Field(default="success")
    markdown: str = Field(default="")
    raw_response: str = Field(default="")
    model: str = Field(default="")
    latency_ms: float = Field(default=0.0, ge=0.0)
    regions: list[OCRRegion] = Field(default_factory=list)
    manifest_path: str | None = None
    error: str | None = None
    decision: Literal[
        "accepted",
        "retried_rotated",
        "low_trust",
        "blank_page",
        "error",
        "disabled",
    ] = Field(default="accepted")
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    trust_level: Literal["high", "medium", "low"] = Field(default="low")
    risk_flags: list[str] = Field(default_factory=list)
    rotation_degrees: Literal[0, 90, 180, 270] = 0
    oriented_image_path: str | None = None
    native_text_similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    candidate_scores: dict[str, float] = Field(default_factory=dict)


class JudgeAuditDecision(BaseModel):
    """Hasil evaluasi dan keputusan koreksi tahap Judge & Refine."""

    action: Literal[
        "accepted",
        "rejected_leak",
        "rejected_bloat",
        "rejected_truncation",
        "fallback_error",
    ] = Field(
        default="accepted",
        description="Status keputusan penerimaan atau penolakan hasil judge",
    )
    final_markdown: str = Field(
        ..., description="Teks Markdown final (hasil koreksi atau fallback ke draf)"
    )
    char_count_draft: int = Field(
        default=0, description="Jumlah karakter draf sebelum judge"
    )
    char_count_refined: int = Field(
        default=0, description="Jumlah karakter teks hasil judge"
    )
    reason: str = Field(
        default="Koreksi berhasil diverifikasi",
        description="Alasan atau keterangan keputusan audit",
    )
    leak_indicator: str | None = Field(
        default=None, description="Frasa bocoran penalaran yang terdeteksi jika ditolak"
    )


class PipelinePageResult(BaseModel):
    """Hasil pemrosesan satu halaman dokumen melalui pipeline ekstraksi."""

    preprocessed_path: str = Field(
        ..., description="Path citra halaman hasil prapemrosesan"
    )
    specs: list[str] = Field(
        default_factory=lambda: ["plain"],
        description="Spesifikasi layout aktif pada halaman ini",
    )
    doc_type: str = Field(
        default="plain", description="Tipe dokumen utama"
    )
    markdown_content: str = Field(
        ..., description="Teks Markdown final yang bersih dan siap di-chunking"
    )
    has_diagram: bool = Field(
        default=False, description="True jika terdapat elemen visual/diagram"
    )
    diagram_mermaid_code: str | None = Field(
        default=None, description="Kode Mermaid.js yang berhasil dikompilasi jika ada"
    )
    difficulty: Literal["simple", "standard", "complex"] = Field(
        default="standard", description="Tingkat kesulitan halaman"
    )
    visual_count: int = Field(
        default=0, description="Jumlah elemen visual terdeteksi pada halaman"
    )
    table_count: int = Field(
        default=0, description="Jumlah tabel terdeteksi pada halaman"
    )
    document_title: str | None = Field(
        default=None, description="Judul utama dokumen jika terdeteksi"
    )
    ocr_status: Literal[
        "accepted",
        "retried_rotated",
        "corrected_by_vlm",
        "fallback_vlm",
        "vlm_visual_rescue",
        "blank_page",
        "success",
        "disabled",
        "fallback",
        "error",
    ] = Field(
        default="disabled", description="Status pemakaian model OCR pada halaman"
    )
    ocr_model: str = Field(default="", description="Nama model OCR yang dipakai")
    ocr_latency_ms: float = Field(default=0.0, ge=0.0)
    ocr_regions: list[OCRRegion] = Field(default_factory=list)
    region_manifest_path: str | None = None
    ocr_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    ocr_trust_level: Literal["high", "medium", "low"] = Field(default="low")
    ocr_risk_flags: list[str] = Field(default_factory=list)
    rotation_degrees: Literal[0, 90, 180, 270] = 0
    vlm_visual_rescue: bool = Field(
        default=False,
        description="True bila VLM utama mengambil alih pembacaan halaman yang sulit secara visual",
    )

    def __getitem__(self, key: str) -> Any:
        """Kompatibilitas backward untuk akses berbasis dict: result['markdown_content']."""
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """Kompatibilitas backward untuk get berbasis dict: result.get('specs')."""
        return getattr(self, key, default)


class DocumentPage(BaseModel):
    """Hasil ekstraksi satu halaman dokumen."""

    page_number: int = Field(..., description="Nomor halaman (1-based)")
    specs: list[str] = Field(
        default_factory=lambda: ["plain"],
        description="Spesifikasi tata letak yang aktif pada halaman ini",
    )
    markdown_content: str = Field(
        ..., description="Teks Markdown hasil ekstraksi halaman ini"
    )
    image_path: str | None = Field(
        default=None, description="Path ke citra halaman yang dirender"
    )
    confidence: float = Field(
        default=1.0, description="Tingkat keyakinan ekstraksi halaman (0.0 - 1.0)"
    )


class DocumentSection(BaseModel):
    """Bagian dokumen berbasis heading Markdown (#, ##, ###)."""

    title: str = Field(..., description="Judul heading bagian ini")
    level: int = Field(..., description="Tingkatan heading (1=H1, 2=H2, 3=H3, dst.)")
    content: str = Field(..., description="Isi teks Markdown di dalam bagian ini")
    page_start: int = Field(
        ..., description="Halaman awal di mana bagian ini dimulai"
    )
    page_end: int = Field(
        ..., description="Halaman akhir di mana bagian ini selesai"
    )
    subsections: list[DocumentSection] = Field(
        default_factory=list, description="Sub-bagian di bawah heading ini"
    )


class ChunkPreview(BaseModel):
    """Pratinjau satu potongan teks hasil simulasi chunking."""

    chunk_id: int = Field(..., description="Indeks chunk berurutan (1-based)")
    char_count: int = Field(..., description="Jumlah karakter dalam chunk")
    token_estimate: int = Field(
        ..., description="Estimasi jumlah token (~karakter / 4)"
    )
    preview: str = Field(
        ..., description="Cuplikan teks awal dan akhir dari chunk"
    )
    content: str = Field(
        default="", description="Konten teks lengkap dari potongan chunk"
    )
    start_char: int = Field(
        default=0, description="Posisi karakter awal dalam dokumen"
    )
    end_char: int = Field(
        default=0, description="Posisi karakter akhir dalam dokumen"
    )


ChunkItem = ChunkPreview


class DocumentChunkingPreview(BaseModel):
    """Hasil simulasi pemotongan teks Markdown hasil ekstraksi siap diindeks RAG."""

    source_file: str = Field(..., description="Nama file dokumen asal")
    total_characters: int = Field(
        ..., description="Jumlah total karakter seluruh dokumen"
    )
    total_chunks: int = Field(
        ..., description="Jumlah total potongan chunk yang dihasilkan"
    )
    chunk_size: int = Field(..., description="Target ukuran karakter per chunk")
    chunk_overlap: int = Field(
        ..., description="Ukuran overlap karakter antar chunk"
    )
    avg_chunk_size: float = Field(
        ..., description="Rata-rata ukuran karakter per chunk"
    )
    chunks: list[ChunkPreview] = Field(
        default_factory=list,
        description="Daftar sampel pratinjau potongan chunk",
    )


ChunkingPreview = DocumentChunkingPreview


# ==============================================================================
# Model-Model Basis Data Tabular & Verifikasi Ganda (SQLite Storage)
# ==============================================================================


class TableColumnInfo(BaseModel):
    """Metadata untuk satu kolom dalam tabel database."""

    name: str = Field(..., description="Nama kolom SQL terstandarisasi")
    data_type: str = Field(
        default="TEXT",
        description="Tipe data SQLite: TEXT, INTEGER, REAL, DATE, atau NUMERIC",
    )
    sql_type: str = Field(
        default="TEXT",
        description="Alias tipe SQL (TEXT, INTEGER, REAL, DATE, atau NUMERIC)",
    )
    original_name: str | None = Field(
        default=None, description="Nama header asli di tabel dokumen"
    )
    description: str = Field(
        default="", description="Deskripsi atau keterangan kolom"
    )
    is_nullable: bool = Field(
        default=True, description="Apakah kolom boleh bernilai NULL"
    )
    is_numeric: bool = Field(
        default=False, description="True jika kolom berisi data numerik/finansial"
    )
    sample_values: list[str] = Field(
        default_factory=list,
        description="Beberapa contoh nilai awal dari kolom ini",
    )


TableColumnSchema = TableColumnInfo


class InferredTableSchema(BaseModel):
    """Skema tabel basis data hasil inferensi otomatis dari tabel Markdown."""

    table_name: str = Field(..., description="Nama tabel SQL yang unik")
    columns: list[TableColumnInfo] = Field(
        ..., description="Daftar kolom hasil inferensi"
    )
    primary_key: str | None = Field(
        default="_row_id",
        description="Kolom primary key (default auto-increment row id)",
    )
    source_file: str = Field(
        default="", description="Path file dokumen sumber data"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata tambahan terkait tabel"
    )


TableSchema = InferredTableSchema

TableTypeLiteral = Literal[
    "transactional",
    "transactional_log",
    "financial_ledger",
    "financial_statement",
    "matrix_pivot",
    "narrative_matrix",
    "narrative_comparison",
    "generic_table",
]


class TableClassificationResult(BaseModel):
    """Hasil analisis semantik untuk membedakan tabel transaksional vs naratif."""

    table_id: str = Field(default="table_auto", description="Identifier unik tabel")
    is_transactional: bool = Field(
        ...,
        description="True jika tabel cocok disimpan di SQLite untuk query analitik SQL",
    )
    table_type: str = Field(
        default="generic_table",
        description="Kategori semantik tabel yang terdeteksi",
    )
    recommended_storage: str = Field(
        default="sqlite_database",
        description="Rekomendasi penyimpanan (sqlite_database / vector_rag)",
    )
    recommended_destination: str = Field(
        default="sqlite",
        description="Tujuan penyimpanan optimal: 'sqlite' untuk transaksional, 'markdown_rag' untuk naratif",
    )
    confidence: float = Field(
        default=1.0,
        description="Tingkat keyakinan klasifikasi (0.0 - 1.0)",
    )
    reasoning: str = Field(
        default="",
        description="Penjelasan logis di balik pemilihan strategi penyimpanan",
    )
    numeric_density: float = Field(
        default=0.0,
        description="Kepadatan numerik",
    )
    date_density: float = Field(
        default=0.0,
        description="Kepadatan tanggal",
    )
    total_rows: int = Field(
        default=0,
        description="Jumlah baris data",
    )
    total_columns: int = Field(
        default=0,
        description="Jumlah kolom data",
    )
    columns_detected: list[str] = Field(
        default_factory=list,
        description="Daftar nama kolom yang terdeteksi",
    )
    numeric_columns_ratio: float = Field(
        default=0.0,
        description="Rasio kolom numerik terhadap total kolom (0.0 - 1.0)",
    )


class VerificationCheck(BaseModel):
    """Satu unit pemeriksaan audit pada verifikasi ganda tabel."""

    check_name: str = Field(..., description="Nama pemeriksaan verifikasi")
    passed: bool = Field(..., description="Status apakah pemeriksaan lolos")
    details: str = Field(default="", description="Rincian hasil pemeriksaan")
    metric_value: Any = Field(
        default=None, description="Nilai metrik hasil perhitungan"
    )


class TableVerificationReport(BaseModel):
    """Laporan audit integritas data ganda (double-verification) tabel SQLite."""

    table_name: str = Field(..., description="Nama tabel yang diaudit")
    database_path: str = Field(
        default="", description="Path absolut file basis data SQLite"
    )
    is_valid: bool = Field(
        ..., description="True jika seluruh pemeriksaan audit lolos"
    )
    verification_status: str | None = Field(
        default=None, description="Status verifikasi tabel"
    )
    confidence_score: float = Field(
        default=1.0, description="Skor keyakinan audit (0.0 - 1.0)"
    )
    row_count_db: int = Field(
        default=0, description="Jumlah baris aktual dalam tabel SQLite"
    )
    verified_row_count: int = Field(
        default=0, description="Jumlah baris yang terverifikasi"
    )
    expected_row_count: int | None = Field(
        default=None, description="Jumlah baris yang diharapkan dari dokumen sumber"
    )
    aggregate_checks: dict[str, Any] = Field(
        default_factory=dict,
        description="Hasil perhitungan agregat SQL (mis. SUM, AVG, MIN, MAX)",
    )
    test_queries: list[Any] = Field(
        default_factory=list,
        description="Daftar query SQL pengujian dan hasilnya",
    )
    null_value_counts: dict[str, int] = Field(
        default_factory=dict, description="Jumlah nilai NULL per kolom"
    )
    discrepancies: list[str] = Field(
        default_factory=list,
        description="Daftar kejanggalan atau inkonsistensi yang ditemukan",
    )
    checks: list[VerificationCheck] = Field(
        default_factory=list, description="Daftar pemeriksaan individual"
    )
    llm_reflection: str | None = Field(
        default=None, description="Catatan hasil audit lanjutan dari LLM"
    )
    llm_audit_note: str | None = Field(
        default=None, description="Catatan hasil audit lanjutan dari LLM bila ada"
    )
    summary: str | None = Field(
        default=None, description="Ringkasan eksekutif hasil audit tabel"
    )


class TableIngestionResult(BaseModel):
    """Hasil proses ingesti tabel ke basis data SQLite dokumen."""

    status: Literal["success", "skipped_narrative", "error", "warning"] = Field(
        ..., description="Status hasil ingesti tabel"
    )
    table_name: str = Field(..., description="Nama tabel SQLite tujuan")
    database_path: str = Field(
        ..., description="Path absolut ke file basis data SQLite"
    )
    total_rows_ingested: int = Field(
        default=0, description="Jumlah baris yang berhasil di-insert ke database"
    )
    columns: list[str] = Field(
        default_factory=list, description="Daftar nama kolom tabel yang dibuat"
    )
    verification_report: TableVerificationReport | None = Field(
        default=None, description="Laporan verifikasi ganda tabel"
    )
    sample_data: list[dict[str, Any]] = Field(
        default_factory=list, description="Sampel beberapa record teratas dari database"
    )
    message: str = Field(default="", description="Pesan status atau keterangan error")


class TabularQueryResult(BaseModel):
    """Hasil eksekusi query SQL pada database SQLite dokumen."""

    query: str = Field(..., description="Query SQL yang dijalankan")
    status: Literal["success", "error"] = Field(default="success")
    columns: list[str] = Field(
        default_factory=list, description="Nama-nama kolom hasil query"
    )
    rows: list[dict[str, Any]] = Field(
        default_factory=list, description="Daftar record baris hasil query"
    )
    row_count: int = Field(
        default=0, description="Total jumlah baris yang dikembalikan"
    )
    execution_time_ms: float = Field(
        default=0.0, description="Waktu eksekusi query dalam milidetik"
    )
    error_message: str | None = Field(
        default=None, description="Pesan galat SQL jika eksekusi gagal"
    )


# ==============================================================================
# Relational Tabular & Header-Detail Deduplication Schemas
# ==============================================================================


class DocumentHeaderRecord(BaseModel):
    """Informasi metadata header dokumen untuk basis data relasional."""

    header_id: int | None = Field(
        default=None, description="ID unik primary key dokumen header"
    )
    doc_type: str = Field(
        default="general_document",
        description="Kategori dokumen (bank_statement, berita_acara, invoice, contract, dll)",
    )
    doc_title: str | None = Field(
        default=None, description="Judul utama dokumen"
    )
    doc_number: str | None = Field(
        default=None,
        description="Nomor dokumen / nomor surat / nomor rekening",
    )
    doc_date: str | None = Field(
        default=None, description="Tanggal dokumen atau rentang periode"
    )
    parties: str | None = Field(
        default=None,
        description="Pihak-pihak terkait (perusahaan, nasabah, bank, dll)",
    )
    total_amount: float | None = Field(
        default=None,
        description="Total nominal transaksi / saldo awal / saldo akhir",
    )
    currency: str = Field(
        default="IDR", description="Mata uang dokumen (IDR, USD, dll)"
    )
    source_file: str = Field(default="", description="Nama file dokumen sumber")
    fingerprint_hash: str | None = Field(
        default=None,
        description="Hash fingerprint untuk deteksi duplikasi dokumen",
    )
    created_at: str | None = Field(
        default=None, description="Timestamp ISO pembuatan record"
    )
    updated_at: str | None = Field(
        default=None, description="Timestamp ISO pembaruan record"
    )


class TransactionDetailRecord(BaseModel):
    """Record baris transaksi finansial yang berelasi FK ke DocumentHeaderRecord."""

    detail_id: int | None = Field(
        default=None, description="ID unik detail transaksi"
    )
    header_id: int = Field(
        ..., description="Foreign key ke header_id di document_headers"
    )
    txn_date: str | None = Field(
        default=None,
        description="Tanggal transaksi (ISO YYYY-MM-DD atau format string)",
    )
    value_date: str | None = Field(
        default=None, description="Tanggal valuta transaksi"
    )
    description: str = Field(
        default="", description="Deskripsi atau uraian transaksi"
    )
    ref_no: str | None = Field(
        default=None, description="Nomor referensi, cek, atau bukti transaksi"
    )
    detail_info: str | None = Field(
        default=None, description="Rincian informasi tambahan transaksi"
    )
    debit: float | None = Field(
        default=None, description="Nominal debet / mutasi keluar"
    )
    credit: float | None = Field(
        default=None, description="Nominal kredit / mutasi masuk"
    )
    balance: float | None = Field(
        default=None, description="Saldo setelah transaksi"
    )
    source_doc: str = Field(default="", description="File sumber dokumen")
    page_number: int | None = Field(
        default=None, description="Nomor halaman dokumen"
    )
    row_hash: str | None = Field(
        default=None, description="Hash baris untuk deduplikasi"
    )
    extra_fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Kolom dinamis opsional tambahan dari model",
    )


class DocumentSignatoryRecord(BaseModel):
    """Record penandatangan atau persetujuan dokumen berelasi FK ke DocumentHeaderRecord."""

    signatory_id: int | None = Field(
        default=None, description="ID unik baris penandatangan"
    )
    header_id: int = Field(
        ..., description="Foreign key ke header_id di document_headers"
    )
    role_party: str | None = Field(
        default=None, description="Pihak atau instansi penandatangan"
    )
    name: str = Field(default="", description="Nama lengkap penandatangan")
    position: str | None = Field(
        default=None, description="Jabatan penandatangan (nullable)"
    )
    date: str | None = Field(
        default=None, description="Tanggal tanda tangan"
    )
    signature_status: str | None = Field(
        default=None,
        description="Status atau keterangan tanda tangan/paraf",
    )
    notes: str | None = Field(
        default=None, description="Catatan atau keterangan tambahan"
    )
    source_doc: str = Field(default="", description="File sumber dokumen")
    page_number: int | None = Field(
        default=None, description="Nomor halaman dokumen"
    )
    row_hash: str | None = Field(
        default=None, description="Hash baris untuk deduplikasi"
    )


class DedupReport(BaseModel):
    """Laporan hasil merge dan deduplikasi data tabel."""

    table_name: str = Field(
        ..., description="Nama tabel yang dideduplikasi"
    )
    initial_rows: int = Field(
        ..., description="Jumlah baris sebelum deduplikasi"
    )
    deduped_rows: int = Field(
        ..., description="Jumlah baris setelah deduplikasi"
    )
    duplicates_removed: int = Field(
        ..., description="Jumlah duplikat yang dibersihkan/dihapus"
    )
    details: str = Field(
        default="", description="Keterangan hasil deduplikasi"
    )


# ==============================================================================
# Diagram & Visual Artifact Schemas (Mermaid.js Extraction)
# ==============================================================================

DiagramTypeLiteral = Literal[
    "flowchart",
    "sequence_diagram",
    "class_diagram",
    "state_diagram",
    "er_diagram",
    "mindmap",
    "gantt_chart",
    "block_architecture",
    "pin_diagram",
    "memory_map",
    "circuit_diagram",
    "timing_diagram",
    "git_graph",
    "unsuitable_statistical_chart",
    "unsuitable_map_or_spatial",
    "unsuitable_photo_or_illustration",
    "unsuitable_complex_schematic",
    "non_diagram",
    "generic_diagram",
]

_ALLOWED_DIAGRAM_TYPES = {
    "flowchart",
    "sequence_diagram",
    "class_diagram",
    "state_diagram",
    "er_diagram",
    "mindmap",
    "gantt_chart",
    "block_architecture",
    "pin_diagram",
    "memory_map",
    "circuit_diagram",
    "timing_diagram",
    "git_graph",
    "unsuitable_statistical_chart",
    "unsuitable_map_or_spatial",
    "unsuitable_photo_or_illustration",
    "unsuitable_complex_schematic",
    "non_diagram",
    "generic_diagram",
}


def _coerce_diagram_type_value(v: Any) -> str:
    """Normalisasi nilai diagram_type secara fleksibel agar tahan terhadap variasi VLM."""
    if not v or not isinstance(v, str):
        return "generic_diagram"
    val = v.strip().lower()
    if val in _ALLOWED_DIAGRAM_TYPES:
        return val
    if "pin" in val:
        return "pin_diagram"
    if "memory" in val or "register" in val or "map" in val:
        return "memory_map"
    if "flow" in val:
        return "flowchart"
    if "block" in val or "arch" in val:
        return "block_architecture"
    if "seq" in val:
        return "sequence_diagram"
    if "circuit" in val or "skema" in val:
        return "circuit_diagram"
    if "time" in val or "timing" in val:
        return "timing_diagram"
    return "generic_diagram"


class DiagramFormatRecommendation(BaseModel):
    """Rekomendasi format ekstraksi visual (Mermaid vs Deskripsi)."""

    diagram_type: DiagramTypeLiteral = Field(
        ..., description="Tipe visual yang diidentifikasi"
    )
    recommended_format: str = Field(
        default="mermaid_code",
        description="Format representasi output yang direkomendasikan",
    )
    is_mermaid_compatible: bool = Field(
        ..., description="True jika visual cocok dijadikan diagram Mermaid.js"
    )
    suggested_syntax: str | None = Field(
        default=None, description="Saran sintaks Mermaid (mis. flowchart TD, sequenceDiagram)"
    )
    rationale: str = Field(
        default="", description="Alasan logis pemilihan format ekstraksi"
    )

    @field_validator("diagram_type", mode="before")
    @classmethod
    def _validate_diagram_type(cls, v: Any) -> str:
        return _coerce_diagram_type_value(v)


class DiagramConvertibilityResult(BaseModel):
    """Hasil evaluasi kelayakan diagram visual untuk diekstrak menjadi kode Mermaid.js."""

    is_convertible: bool = Field(
        ...,
        description="True jika diagram memiliki simpul dan relasi diskrit yang cocok untuk Mermaid",
    )
    diagram_type: DiagramTypeLiteral = Field(
        ..., description="Kategori diagram visual yang terdeteksi"
    )
    recommended_format: str = Field(
        default="mermaid_code",
        description="Format output yang direkomendasikan (mermaid, mermaid_code, text_description, markdown_table)",
    )
    mermaid_type: str | None = Field(
        default=None,
        description="Tipe diagram Mermaid jika cocok (mis. flowchart, sequenceDiagram, erDiagram, classDiagram, stateDiagram, mindmap)",
    )
    confidence: float = Field(
        default=1.0, description="Tingkat keyakinan deteksi kelayakan (0.0 - 1.0)"
    )
    reasoning: str = Field(
        default="",
        description="Alasan mengapa visual ini cocok atau tidak cocok dikonversi ke Mermaid",
    )
    nodes_or_entities: list[str] = Field(
        default_factory=list, description="Daftar node atau entitas utama yang terdeteksi"
    )

    @field_validator("diagram_type", mode="before")
    @classmethod
    def _validate_diagram_type(cls, v: Any) -> str:
        return _coerce_diagram_type_value(v)


class DiagramExtractionResult(BaseModel):
    """Hasil ekstraksi diagram visual ke kode Mermaid.js atau deskripsi teks terstruktur."""

    status: Literal["success", "unsuitable", "error"] = Field(default="success")
    is_mermaid: bool = Field(
        ...,
        description="True jika hasil ekstraksi berupa blok kode Mermaid.js yang valid",
    )
    diagram_type: DiagramTypeLiteral = Field(
        default="flowchart", description="Tipe diagram visual"
    )
    mermaid_code: str | None = Field(
        default=None,
        description="Kode Mermaid.js lengkap (termasuk deklarasi tipe dan node/edge)",
    )
    text_description: str | None = Field(
        default=None,
        description="Deskripsi terstruktur jika visual tidak cocok untuk Mermaid (mis. grafik statistik, peta)",
    )
    text_summary: str | None = Field(
        default=None,
        description="Ringkasan atau catatan visual",
    )
    reasoning: str | None = Field(
        default=None,
        description="Alasan penentuan status ekstraksi diagram",
    )
    convertibility: DiagramConvertibilityResult | None = Field(
        default=None,
        description="Hasil evaluasi kelayakan diagram",
    )
    rendered_image_bytes: bytes | None = Field(
        default=None,
        description="Data biner PNG dari hasil render Mermaid CLI / pymmdc",
    )
    rendered_image_path: str | None = Field(
        default=None,
        description="Path file gambar PNG hasil rendering Mermaid jika disimpan",
    )
    confidence: float = Field(
        default=1.0, description="Tingkat keyakinan ekstraksi (0.0 - 1.0)"
    )
    validation_status: Literal["valid", "syntax_error", "unvalidated"] = Field(
        default="unvalidated", description="Status validasi sintaks Mermaid"
    )
    raw_response: str | None = Field(
        default=None, description="Respon mentah dari VLM untuk keperluan audit"
    )

    @field_validator("diagram_type", mode="before")
    @classmethod
    def _validate_diagram_type(cls, v: Any) -> str:
        return _coerce_diagram_type_value(v)


# ==============================================================================
# Dual-Track Multi-Page Tabular Processing & Guardrail Schemas
# ==============================================================================


class PageTabularEvent(BaseModel):
    """Log proses ekstraksi dan ingesti mandiri sub-agent SQL per-halaman/slide."""

    page_number: int = Field(..., description="Nomor halaman atau slide")
    source_file: str = Field(default="", description="Path dokumen asal")
    tables_detected: int = Field(default=0, description="Jumlah tabel yang ditemukan pada halaman ini")
    tables_ingested: list[str] = Field(
        default_factory=list, description="Daftar nama tabel SQLite yang dibuat/ditambahkan"
    )
    existing_tables_inspected: list[str] = Field(
        default_factory=list, description="Daftar tabel eksisting yang diinspeksi di SQLite"
    )
    actions_taken: list[str] = Field(
        default_factory=list, description="Daftar tindakan yang diambil sub-agent"
    )
    queries_executed: list[Any] = Field(
        default_factory=list, description="Daftar query SQL yang dieksekusi sub-agent"
    )
    rows_ingested_total: int = Field(
        default=0, description="Total baris data yang berhasil dimasukkan ke SQLite pada halaman ini"
    )
    status: str = Field(
        default="no_tables",
        description="Status hasil proses tabular pada halaman",
    )
    queries_run: list[str] = Field(
        default_factory=list, description="Daftar query SQL verifikasi mandiri yang dijalankan sub-agent"
    )
    tagged_markdown: str | None = Field(
        default=None, description="Teks Markdown halaman yang telah diperkaya tag metadata <!-- sqlite_table: ... -->"
    )


class DualTrackGuardrailReport(BaseModel):
    """Laporan audit keselarasan jalur Teks Markdown vs Jalur Database SQLite oleh Supervisor Agent."""

    source_file: str = Field(..., description="Path file dokumen yang diaudit")
    database_path: str = Field(
        default="", description="Path absolut file basis data SQLite"
    )
    total_pages_processed: int = Field(
        default=1, description="Jumlah halaman yang diaudit"
    )
    total_pages_audited: int = Field(
        default=1, description="Jumlah halaman yang diaudit"
    )
    total_markdown_tables: int = Field(
        default=0, description="Jumlah total tabel yang muncul di teks Markdown"
    )
    total_sqlite_tables: int = Field(
        default=0, description="Jumlah tabel terstruktur yang tersimpan di SQLite"
    )
    total_markdown_rows: int = Field(
        default=0, description="Jumlah total baris data dari tabel Markdown"
    )
    total_sqlite_rows: int = Field(
        default=0, description="Jumlah total baris data yang tersimpan di SQLite"
    )
    guardrail_status: str = Field(
        default="passed",
        description="Status kesesuaian jalur teks vs jalur database tabular",
    )
    table_comparisons: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Rincian perbandingan skema dan baris antar tabel",
    )
    discrepancies: list[str] = Field(
        default_factory=list,
        description="Catatan selisih atau anomali jika ditemukan ketidaksesuaian",
    )
    supervisor_notes: str = Field(
        default="", description="Catatan ringkasan dari Supervisor Agent"
    )


class ExcelCellEvidence(BaseModel):
    """Nilai dan metadata satu sel sumber yang dapat diaudit."""

    coordinate: str
    row: int = Field(..., ge=1)
    column: int = Field(..., ge=1)
    value: Any = None
    formula: str | None = None
    number_format: str = "General"
    data_type: str = "n"
    merged_range: str | None = None
    font_size: float | None = None
    text_rotation: int = 0


class ExcelRegion(BaseModel):
    """Satu blok logis pada worksheet yang dirender dan diekstrak mandiri."""

    region_id: str
    sheet_name: str
    sheet_index: int = Field(..., ge=0)
    cell_range: str
    min_row: int = Field(..., ge=1)
    max_row: int = Field(..., ge=1)
    min_column: int = Field(..., ge=1)
    max_column: int = Field(..., ge=1)
    title: str | None = None
    period: str | None = None
    kind: Literal["table", "text", "mixed"] = "mixed"
    render_dpi: int = Field(default=300, ge=72)
    requires_vlm_reading: bool = False
    native_text: str = ""
    cells: list[ExcelCellEvidence] = Field(default_factory=list)


class ExcelSheetSurvey(BaseModel):
    """Inventaris sheet dan region yang ditemukan sebelum rendering visual."""

    name: str
    index: int = Field(..., ge=0)
    visible: bool = True
    used_range: str | None = None
    regions: list[ExcelRegion] = Field(default_factory=list)


class ExcelWorkbookSurvey(BaseModel):
    """Manifest struktur workbook yang menyertai hasil ekstraksi visual."""

    source_file: str
    workbook_format: str
    sheets: list[ExcelSheetSurvey] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ExtractedDocument(BaseModel):
    """Hasil akhir dokumen lengkap dengan metadata hierarki, database tabular, & diagram Mermaid."""

    source_file: str = Field(..., description="Path file input")
    title: str | None = Field(
        default=None, description="Judul dokumen utama yang teridentifikasi (dari halaman 1/sampul)"
    )
    doc_type: str = Field(
        default="plain",
        description="Spesifikasi tata letak utama: plain, markdown_hierarchy, bilingual_journal, presentation_slides",
    )
    pages: list[DocumentPage] = Field(
        default_factory=list, description="Daftar hasil ekstraksi per halaman"
    )
    full_markdown: str = Field(
        ..., description="Teks Markdown lengkap gabungan seluruh halaman"
    )
    sections: list[DocumentSection] = Field(
        default_factory=list, description="Bagian-bagian dokumen berbasis heading"
    )
    diagrams: list[DiagramExtractionResult] = Field(
        default_factory=list,
        description="Daftar diagram visual yang berhasil diekstrak menjadi kode Mermaid",
    )
    tabular_results: list[TableIngestionResult] = Field(
        default_factory=list,
        description="Daftar tabel transaksional yang berhasil di-ingest dan diverifikasi ke SQLite",
    )
    page_tabular_events: list[PageTabularEvent] = Field(
        default_factory=list,
        description="Log per-halaman pemrosesan dan query mandiri oleh Sub-Agent SQL Tabular",
    )
    guardrail_report: DualTrackGuardrailReport | None = Field(
        default=None,
        description="Laporan audit keselarasan jalur Teks Markdown vs Database SQLite dari Supervisor Agent",
    )
    total_pages: int = Field(default=1, description="Jumlah total halaman")
    total_visuals: int = Field(
        default=0,
        description="Total elemen visual (diagram, topologi, gambar, ilustrasi) terdeteksi di seluruh dokumen",
    )
    total_tables: int = Field(
        default=0,
        description="Total tabel GFM terdeteksi di seluruh dokumen",
    )
    excel_workbook: ExcelWorkbookSurvey | None = Field(
        default=None,
        description="Manifest sheet, region, dan bukti sel untuk sumber spreadsheet",
    )
    excel_native_tables: list[str] = Field(
        default_factory=list,
        description="Nama tabel SQLite yang dibuat langsung dari nilai sel spreadsheet",
    )

    @property
    def markdown_content(self) -> str:
        """Alias untuk full_markdown agar kompatibel."""
        return self.full_markdown

    @property
    def file_path(self) -> str:
        """Alias untuk source_file agar kompatibel."""
        return self.source_file
