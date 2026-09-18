# jds-magang-document-extractor — OCR + Vision VLM Document Extractor

Sistem ekstraksi **dokumen internal perusahaan** (PDF, DOC/DOCX, Excel, PPT/PPTX, Scan Gambar, Screenshot Chat, Form Persetujuan) menjadi **Markdown bersih dan terstruktur**. Unlimited-OCR membuat draft dan grounding region, sedangkan Vision LLM utama menangani klasifikasi, reasoning, koreksi adaptif, tabel SQLite, dan ekstraksi diagram **Mermaid.js**.

> ℹ️ **Catatan Branch:** 
> Branch `experiment/ocr-model-v2` memakai dua server llama.cpp: VLM utama dan OCR pada port terpisah. Jika `OCR_MODEL` kosong atau endpoint OCR gagal, ekstraksi otomatis fallback ke VLM utama. Modul RAG staging tetap terpisah di `app/rag_staging.py`.

---

## Pembelajaran dari koreksi pengguna

Streamlit menyediakan **Koreksi hasil halaman ini** untuk menyimpan perbaikan teks dan tabel.
Contoh yang disetujui dapat dipakai admin untuk optimasi prompt dengan DSPy dan GEPA,
evaluasi pada dokumen terpisah, penerapan kandidat, dan pemulihan versi sebelumnya.
Lihat [panduan pembelajaran dan perintah admin](docs/PEMBELAJARAN.md).

## 🛠️ Instalasi & Persyaratan Sistem (Installation & Setup)

### 1. Persyaratan Sistem (Prerequisites)
- **Python**: `>= 3.12` (disarankan menggunakan Conda/venv dan manajer paket `uv`).
- **Node.js & npm**: `>= 18` (diperlukan untuk engine compiler rendering Mermaid CLI lokal).
- **LibreOffice**: Diperlukan jika memproses file presentasi PowerPoint (`.ppt`/`.pptx`) via mode headless.
- **Dua endpoint OpenAI-compatible**: server Vision LLM utama dan server Unlimited-OCR/DeepSeek-OCR-aware llama.cpp pada port berbeda.

### 2. Langkah Instalasi (Step-by-Step)

```bash
# 1. Clone repositori
git clone https://github.com/daruoktab/jds-magang-ocr-agent.git
cd jds-magang-ocr-agent

# 2. Buat & aktifkan environment (contoh menggunakan Conda)
conda create -n magang-jds python=3.12 -y
conda activate magang-jds

# 3. Instal dependensi Python (preferensikan uv pip)
uv pip install -e .

# 4. Instal engine Mermaid CLI & Puppeteer secara global
npm install -g @mermaid-js/mermaid-cli puppeteer

# 5. Pasang browser headless Chromium untuk engine rendering diagram lokal
npx puppeteer browsers install chrome-headless-shell
```

### 3. Konfigurasi Environment Variable (`.env`)
Buat berkas `.env` di direktori utama repositori. Model OCR sengaja boleh dikosongkan sampai alias di server siap:

```env
BASE_URL=http://127.0.0.1:8080/v1
VLM_MODEL=Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
VLM_ENABLE_THINKING=false
VLM_VISUAL_RESCUE=true
VLM_TEMPERATURE=0.1
VLM_TIMEOUT=300

OCR_BASE_URL=http://127.0.0.1:8081/v1
OCR_MODEL=
OCR_TEMPERATURE=0.0
OCR_TIMEOUT=300
OCR_PROMPT=<|grounding|>Convert the document to markdown.
OCR_MIN_TRUST_SCORE=0.72
OCR_MEDIUM_TRUST_SCORE=0.48
OCR_ROTATION_RETRY=true
OCR_BLANK_INK_RATIO=0.0002
OCR_SPARSE_INK_RATIO=0.015
```

Ketika `OCR_MODEL` diisi, OCR menjadi sumber draft Markdown utama hanya setelah lolos quality gate. Pipeline mengoreksi orientasi, membandingkan hasil dengan text-layer PDF bila tersedia, mencoba rotasi alternatif pada kandidat berisiko, dan memakai ekstraksi VLM independen saat trust OCR rendah. Dengan `VLM_VISUAL_RESCUE=true`, VLM juga mengambil alih pembacaan halaman berfont sangat kecil/padat, teks miring penting, anotasi teknis kecil, multi-kolom rapat, atau kontras rendah—meski OCR lolos trust tinggi. Grounding `table` dan `figure` dipotong ke `output/{dokumen}/regions/...`; crop figure dikirim ke spesialis Mermaid. `OCR_MODEL=` tetap aman untuk masa setup karena mengaktifkan fallback VLM.

---

## 🎯 6 Spesifikasi Karakteristik Dokumen (Mendukung Multi-Spesifikasi Komposit)

Sistem mendukung ekstraksi dengan satu atau **beberapa spesifikasi sekaligus secara komposit** (*Composable Prompts*), misalnya slide presentasi yang juga memuat form tanda tangan (`presentation_slides` + `signature_form`):

| No | Spesifikasi Layout | Karakteristik & Perilaku Ekstraksi | Target Output |
|:---:|:---|:---|:---|
| **1** | `plain` | **Dokumen Bisnis Umum**<br>Surat, memo, pengumuman, formulir sederhana, teks internal. Diekstrak via VLM menjadi markdown bersih dengan kop surat, key-value, dan tabel GFM. | Paragraf rapi, key-value format, tabel standar GFM |
| **2** | `markdown_hierarchy` | **Dokumen Terstruktur**<br>SOP, SK, kebijakan, peraturan, perjanjian, laporan formal bertingkat (`#`, `##`, `###`). Menjaga konsistensi judul ketika berpindah halaman (multi-page) dan menyambungkan kalimat terpotong tanpa merusak struktur. | Hierarki heading utuh, pasal/ayat dipertahankan |
| **3** | `bilingual_journal` | **Artikel Internal / Dokumen Multi-Kolom**<br>Artikel internal, buletin, dokumen 2-kolom, atau dokumen dua bahasa berdampingan. Membaca kolom kiri dari atas ke bawah sampai selesai, lalu melanjutkan ke kolom kanan. | Urutan baca logis berurutan (tidak melompat antar-kolom) |
| **4** | `presentation_slides` | **Slide Presentasi (PPT / PPTX / Slide PDF)**<br>Slide yang sarat poin-poin/bullet list, tabel ringkas, speaker notes, serta penanda visual/diagram `[Diagram/Visual]: ...`. | Markdown per-slide yang siap dipartisi per topik |
| **5** | `chat_transcript` | **Screenshot Percakapan Chat**<br>WhatsApp, Telegram, chat internal. Diekstrak sebagai transkrip urut: `- **[Waktu] Pengirim:** isi pesan`, termasuk quote/lampiran. | Transkrip percakapan terstruktur siap RAG |
| **6** | `signature_form` | **Form Tanda Tangan / Paraf / Approval**<br>Surat persetujuan dengan kotak tanda tangan. Diekstrak sebagai tabel (Pihak, Nama, Jabatan, Tanda Tangan, Tanggal) tanpa menebak nama dari tanda tangan atau status approval. | Tabel persetujuan GFM dengan konvensi `[tidak terbaca]` / `[kosong]` |

Spesifikasi komposit didukung via alias, misalnya:

```bash
python main.py laporan.pdf -t "journal,hierarchy"
python main.py dokumen.png -t "chat_transcript,signature_form"
```

---

## ⚡ Mode Adaptif Cepat (Adaptive Fast-Path)

Pipeline default berjalan dalam **mode adaptif**: setiap halaman dinilai tingkat kesulitannya, dan langkah yang tidak perlu di-skip untuk menghemat waktu (tiap VLM call lokal 30–100 detik):

| Deteksi | Cara | Biaya |
|:---|:---|:---|
| Ada diagram/visual? | Cek output ekstraksi untuk `[Diagram/Visual]`, `[Topologi]`, ` ```mermaid ` | 0 VLM call |
| Difficulty (simple/standard/complex) | Heuristic post-extraction + penilaian `inspect_page` | 0 VLM call tambahan |
| Skip judge | Hanya untuk halaman *simple* yang bersih: tidak ada diagram, tidak ada tabel, tidak ada `[tidak terbaca]`, output ≥ 20 karakter | Hemat ~45 detik/halaman |
| Skip diagram specialist | Hanya untuk halaman tanpa indikator visual | Hemat ~50 detik/halaman |

**Guard safety** — judge tetap dijalankan jika ada: diagram, tabel, `[tidak terbaca]`, output mencurigakan (hampir kosong), atau difficulty bukan `simple`.

Untuk pipeline penuh tanpa fast-path (dokumen sensitif / hasil audit):

```bash
python main.py dokumen.pdf --thorough
```

Pipeline juga menghasilkan **metadata visual & tabel** per halaman/dokumen:

```text
[Vision PPT] [Slide 3/13] Metadata: 2 visual/diagram, 1 tabel
[Vision PPT] Selesai: 13 slide | 5 elemen visual/diagram | 3 tabel terdeteksi
```

---

## 🖼️ Rendering Presentasi (PPT/PPTX)

Presentasi di-render menjadi gambar PNG per slide lalu dikirim ke model vision:

```text
PPT/PPTX -> LibreOffice headless -> PDF -> PyMuPDF -> PNG per slide
```

Pipeline memakai **LibreOffice headless** sebagai satu-satunya renderer di semua platform (Windows, Linux, macOS) — tidak ada lagi dependensi Spire. Instalasi:

- **Debian/Ubuntu:** `sudo apt-get install -y libreoffice`
- **Windows:** installer LibreOffice resmi (pipelinenya otomatis mendeteksi lokasi instalasi standar).
- Jika executable tidak berada di `PATH` atau tidak di lokasi standar, set env `LIBREOFFICE_BIN` ke path `soffice`/`libreoffice`.

Batching: rendering diproses bertahap per 10 slide (selaras strategi batch di pipeline PDF) untuk mengontrol memori, progres logging, dan keseragaman unit kerja — bukan karena batasan lisensi.

MCP tool: `render_presentation_slides` (lihat §MCP Tools di bawah).

---

## 📊 Ekstraksi Selektif Diagram ke Mermaid.js & Visual Verification (`app/diagram.py`)

Tidak semua gambar visual pada dokumen cocok diubah menjadi diagram Mermaid. Modul analisis diagram mengevaluasi kelayakan secara selektif sebelum melakukan konversi sintaks:

1. **Kategori yang Cocok untuk Mermaid (`is_convertible = True`)**:
   - `flowchart`: Alur proses bisnis, pohon keputusan, alur logika (`flowchart TD` / `LR`).
   - `sequence_diagram`: Interaksi antar aktor, sistem, API, atau servis (`sequenceDiagram`).
   - `class_diagram`: Diagram kelas UML, struktur atribut & method (`classDiagram`).
   - `state_diagram`: State machine, siklus hidup status (`stateDiagram-v2`).
   - `er_diagram`: Entity Relationship Diagram data relasional (`erDiagram`).
   - `mindmap`: Peta konsep hierarkis bertingkat (`mindmap`).
   - `gantt_chart`: Jadwal linimasa proyek & milestone (`gantt`).
   - `block_architecture`: Arsitektur blok/komponen sistem (`flowchart` dengan `subgraph`).

2. **Kategori yang Tidak Cocok untuk Mermaid (`is_convertible = False`)**:
   - Grafik numerik kontinu padat (scatter plot, multi-series line chart, histogram, heatmap).
   - Peta geografis / denah spasial.
   - Foto realistis, gambar anatomi biologis, atau seni bebas.
   - Skematik sirkuit elektrik mikro atau CAD mekanik rumit.
   - *Penanganan:* Dikonversi ke format representasi yang lebih tepat (tabel Markdown atau ringkasan struktural), bukan dipaksakan ke kode Mermaid yang rusak.

3. **Rendering Lokal ke Gambar PNG (`pymmdc` + Mermaid CLI)**:
   - Setiap diagram yang dihasilkan diuji kompilasi secara lokal langsung ke biner gambar PNG menggunakan `pymmdc` dan `mmdc.cmd` dengan engine headless Chromium (`chrome-headless-shell`).
   - Mencegah kode Mermaid yang rusak atau syntax error lolos ke hasil akhir. Data biner gambar PNG yang terverifikasi disimpan pada atribut `rendered_image_bytes` di objek `DiagramExtractionResult`.

4. **Compiler Error Feedback Loop**:
   - Jika kompilasi Mermaid CLI gagal (misal: siklus hierarki DAG `Setting UserMem as parent of UserMem would create a cycle` atau duplikasi edge/label konflik), pesan error presisi langsung diumpankan kembali ke Vision LLM bersama gambar asli dokumen sumber untuk diperbaiki secara mandiri (self-correction hingga 2x percobaan).

5. **Multimodal Visual Verification Loop**:
   - Saat render PNG sukses, sistem menyajikan perbandingan visual multimodal ke Vision LLM:
     - **Gambar 1:** Potongan diagram asli dari dokumen sumber.
     - **Gambar 2:** Gambar PNG hasil render kode Mermaid.
     - **Teks Kode:** Kode Mermaid yang dihasilkan.
   - Vision LLM membandingkan kedua gambar secara visual. Jika diagram hasil render sudah lengkap dan akurat, model mengonfirmasi `[CONFIRMED]`. Jika terdapat simpul atau teks yang terpotong/hilang, model merevisi kode Mermaid hingga sempurna.

---

## 🗄️ Mekanisme Tabular Transaksional vs Narrative Chunking (SQLite + Double-Verification)

Dokumen seperti **rekening koran, mutasi bank (Doc 8), ledger kas, slip transaksi, dan tabel keuangan** tidak cocok di-chunking ke Vector DB karena Vector Search tidak dapat melakukan kalkulasi agregat (SUM, AVG, filter tanggal).

Sistem kini dilengkapi modul cerdas ([`app/tabular_db.py`](app/tabular_db.py)):
1. **Deteksi & Klasifikasi Heuristik + LLM**:
   - Membedakan tabel transaksional (`transactional_log`, `financial_statement`) vs tabel naratif (`narrative_matrix`).
   - Menganalisis rasio numerik, format tanggal ISO, keyword perbankan/akuntansi, dan kepadatan teks sel.
2. **Pydantic Validation**:
   - Skema tabel divalidasi ketat menggunakan Pydantic (`TableClassificationResult`, `TableSchema`, `TableColumnSchema`, `TableVerificationReport`, `TabularQueryResult`).
3. **SQLite Ingestion**:
   - Data otomatis dikonversi ke tipe data SQL murni (`REAL`, `INTEGER`, `DATE`, `TEXT`), membersihkan pemisah ribuan internasional (`4,218,640,517.32`) maupun format Indonesia (`4.218.640.517,32`).
4. **Double-Verification (Verifikasi Ganda)**:
   - **Row Count Integrity Check**: Memastikan tidak ada baris yang terlewat.
   - **Numeric & Aggregation Sanity Test**: Mengeksekusi query `SELECT SUM(...)` & `SELECT AVG(...)` untuk membuktikan query SQL bebas syntax/type error.
   - **Balance Continuity Check**: Menguji konsistensi aritmatika saldo berjalan ($Saldo_{n-1} + Kredit - Debit = Saldo_n$).
   - **LLM Reflection**: Mengonfirmasi akurasi ekstraksi terhadap sampel data mentah.

---

## 🤖 Arsitektur Sub-Agent (Deep Agents Harness — via MCP)

Proyek ini dilengkapi dengan **Master Agent dan 6 Sub-Agent Spesialis** ([app/deep_agent.py](app/deep_agent.py)) berbasis Vision Language Model murni. Deep Agent tersedia via **MCP Server** untuk use case conversational (instruksi bebas, query SQLite interaktif) — bukan via CLI, karena pipeline default CLI sudah mencakup seluruh kemampuan ekstraksi secara lebih cepat dan deterministik.

Semua tool Deep Agent terintegrasi dengan konfigurasi caller: `db_path` dan `output_markdown_path` di-bake ke dalam tool, sehingga hasil ekstraksi menulis ke path yang ditentukan.

| Nama Sub-Agent | Peran & Spesialisasi | Tool Utama |
|:---|:---|:---|
| `layout-classifier` | Deteksi multi-trait tata letak dokumen (kolom, hierarki, slide, tabel) | `classify_layout` |
| `markdown-extractor` | Ekstraksi gambar multimodal ke Markdown bersih berbasis spesifikasi komposit via VLM | `extract_to_markdown` |
| `diagram-mermaid-specialist` | Evaluasi selektif & ekstraksi diagram visual ke kode Mermaid.js yang valid | `classify_diagram_suitability`, `extract_diagram_to_mermaid` |
| `presentation-specialist` | Ekstraksi slide PowerPoint (.pptx/.ppt): render gambar per slide, lalu dibaca VLM menjadi Markdown | `extract_presentation_pptx` |
| `pdf-orchestrator` | Orkestrasi pemrosesan PDF multi-halaman & penyambungan kontinuitas heading | `extract_pdf_document` |
| `tabular-db-specialist` | Deteksi tabel transaksional, simpan ke SQLite, verifikasi ganda, & eksekusi query SQL | `classify_table_storage`, `ingest_table_to_sqlite`, `verify_sqlite_table`, `query_tabular_database` |

---

## 🔌 Model Context Protocol (MCP) Server & Batch Document Discovery

Server MCP berbasis **MCP Python SDK** (`mcp>=1.0.0`; [app/mcp_server.py](app/mcp_server.py)) menyediakan MCP Tools lengkap untuk AI Assistant:

### Daftar MCP Tools:
1. **`scan_document_folders`**: Pindai direktori (mis. `dataset`, `input`, `output`) dan seluruh subfolder untuk mendeteksi folder dokumen.
2. **`process_document_batch`**: Ekstraksi dokumen massal dari folder terpilih.
3. **`extract_document`**: Ekstraksi file dokumen tunggal (PDF, DOCX, Excel, PPTX, gambar) ke Markdown siap chunking.
4. **`classify_document_layout`**: Deteksi multi-trait spesifikasi tata letak dokumen.
5. **`classify_diagram_convertibility`**: Evaluasi kelayakan visual diagram untuk konversi ke Mermaid.js.
6. **`extract_diagram_to_mermaid`**: Ekstraksi diagram visual ke kode Mermaid.js yang valid secara sintaks.
7. **`render_presentation_slides`**: Render slide presentasi (.pptx/.ppt) menjadi file gambar PNG beresolusi tinggi (LibreOffice headless).
8. **`preview_markdown_chunks`**: Simulasi partisi teks Markdown dengan header metadata.
9. **`classify_and_ingest_tables_to_sqlite`**: Ingesti otomatis tabel transaksional ke SQLite dengan double-verification.
10. **`query_tabular_database`**: Eksekusi SQL query untuk kalkulasi numerik (SUM, AVG, COUNT, date-filter) berpresisi 100%.
11. **`inspect_tabular_database`**: Inspeksi daftar tabel, skema kolom, dan jumlah baris di SQLite.

### Konfigurasi `mcp_config.json`:
```json
{
  "mcpServers": {
    "jds-magang-doc-agent": {
      "command": "python",
      "args": ["-m", "app.mcp_server"],
      "env": {
        "BASE_URL": "http://127.0.0.1:8080/v1",
        "VLM_MODEL": "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf",
        "OCR_BASE_URL": "http://127.0.0.1:8081/v1",
        "OCR_MODEL": "unlimited-ocr"
      }
    }
  }
}
```

> Kedua endpoint memakai API OpenAI-compatible. `BASE_URL` khusus VLM utama; `OCR_BASE_URL` khusus proses OCR.

---

## 🚀 Penggunaan CLI

### Perilaku Default
- **Output otomatis disimpan ke file** `output/{nama_file}/{nama_file}.md` (bukan dump ke terminal) — pakai `-o` untuk override
- **Streaming per-halaman**: Markdown ditulis ke file output sambil dokumen diproses, bukan di akhir
- **Log real-time otomatis** ke `output/{nama_file}/logs/{nama_file}_latest.log` — monitor dengan `Get-Content -Wait` (PowerShell) atau `tail -f` (Linux)
- **Database SQLite otomatis** di `output/{nama_file}/databases/{nama_file}.sqlite`

### 1. Ekstraksi Dokumen Tunggal (PDF / Gambar / Scan)
```bash
# Ekstraksi otomatis — output ke output/sample/sample.md, SQLite ke output/sample/databases/
python main.py dataset/sample.pdf

# Dengan path output eksplisit
python main.py dataset/sample.pdf -o output/sample.md

# Screenshot chat WhatsApp + form tanda tangan dalam satu dokumen
python main.py laporan.png -t "chat_transcript,signature_form"

# Spesifikasi komposit artikel multi-kolom + hierarki bab
python main.py dataset/artikel.pdf -t "bilingual_journal,markdown_hierarchy"

# Tampilkan juga hasil di terminal (selain file)
python main.py dataset/sample.pdf --stdout

# Pratinjau statistik chunking untuk validasi RAG
python main.py dataset/sample.pdf --preview-chunks --chunk-size 800 --chunk-overlap 100

# Klasifikasi layout saja (tanpa ekstraksi)
python main.py dataset/sample.pdf --classify-only
```

### 2. Ekstraksi Presentasi PowerPoint (.pptx / .ppt)
```bash
# Pipeline VLM (default) — slide dirender ke gambar via LibreOffice headless
python main.py dataset/presentation.pptx

# Mode native teks (hanya membaca shape & text frame, cepat, tanpa VLM)
python main.py dataset/presentation.pptx --ppt-native

# Pipeline penuh: judge & diagram specialist selalu aktif di tiap slide
python main.py dataset/presentation.pptx --thorough
```

### 3. Database Tabular & Verifikasi
```bash
# Path SQLite kustom
python main.py rekening.pdf --db-path output/db/rekening.sqlite

# Paksa seluruh tabel (termasuk naratif) di-ingest ke SQLite
python main.py dokumen.pdf --force-all-tables
```

### 4. Pemindaian Dataset & Ekstraksi Massal (Batch Mode)
```bash
# Pindai struktur folder dataset
python main.py --scan-folders dataset

# Ekstraksi batch seluruh dokumen di dalam subfolder terpilih
python main.py --batch "dataset/01_Kamus,dataset/02_Jurnal" -o output/batch_results

# Daftar seluruh spesifikasi yang didukung
python main.py --list-types
```

### 5. Mode Adaptif vs Thorough
```bash
# Default: mode adaptif — halaman simple di-skip judge & diagram specialist
python main.py dokumen.pdf

# Thorough: selalu full pipeline (judge + diagram di setiap halaman)
python main.py dokumen.pdf --thorough
```

### 6. Logging
```bash
# Log otomatis real-time ke output/{nama}/logs/{nama}_latest.log (default)
python main.py dokumen.pdf

# Path log kustom
python main.py dokumen.pdf --log-file logs/run.txt

# Level DEBUG
python main.py dokumen.pdf --debug
```

> ℹ️ **Deep Agent (`--agent`) telah dihapus dari CLI** karena redundan — pipeline default sudah melakukan semua hal yang sama (auto-klasifikasi, diagram, SQLite, judge) lebih cepat dan deterministik. Deep Agent tetap tersedia via **MCP Server** untuk use case conversational (instruksi bebas, query SQLite interaktif).

---

## API ingest dengan FastAPI

Jalankan `uv sync`, lalu `uv run uvicorn app.api:app --port 8000`.
`POST /ingest` menerima hanya field upload `file` dan mengembalikan ZIP berisi
Markdown, SQL SQLite, serta CSV jika ada tabel. Buka `/docs` untuk mencoba API
dan `/plan` untuk melihat alur ingest. Lihat [panduan API](docs/API.md).

## 🖥️ Antarmuka Interaktif Streamlit Workspace Studio

Aplikasi web interaktif Streamlit untuk monitoring proses ekstraksi, inspeksi visual side-by-side, preview chunking RAG, dan analisis data tabular SQLite.

### Menjalankan Streamlit Studio
```bash
uv run streamlit run app/streamlit_logic.py
```

### Fitur Utama Streamlit:
1. **Workspace Document Explorer (Bebas Reset F5):**
   - Mendeteksi secara otomatis semua dokumen yang pernah diproses atau sedang berjalan dari folder `output/`.
   - Pengguna dapat beralih antar dokumen secara instan tanpa perlu mengunggah ulang file.
2. **Side-by-Side Visual Document Inspector:**
   - Menampilkan kanvas visual dokumen fisik (gambar render resolusi tinggi per halaman PDF / slide PPT) berdampingan langsung dengan hasil ekstraksi Markdown.
   - Pilihan inspeksi teks per halaman atau seluruh dokumen dengan tombol unduh gambar halaman.
3. **Interactive Mermaid.js SVG Diagram Renderer:**
   - Mendeteksi blok sintaks diagram Mermaid secara otomatis dan merendernya menjadi visualisasi diagram interaktif berbasis SVG langsung di peramban.
   - Dilengkapi penampil kode sumber Mermaid dan tombol unduh `.mmd`.
4. **Dual-Track Guardrail Audit:**
   - Laporan rekonsiliasi matematis antara tabel teks Markdown dan database SQLite dengan status `PASSED`, `WARNING`, atau `FAILED`.
5. **SQL Tabular Studio & Quick Visual Chart:**
   - Pratinjau tabel SQLite, ekspor CSV 1-klik, auto-chart (Bar Chart, Line Chart, Area Chart) untuk kolom numerik, dan konsol SQL query interaktif.
6. **Background Job Supervision:**
   - Ekstraksi dijalankan di background thread independen dari siklus tab/browser dengan streaming log real-time dan proteksi pembatalan proses.
