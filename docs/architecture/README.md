# Pengolah Dokumen AI: kemampuan, arsitektur, dan penggunaan

Dokumentasi ini mengikuti kode **main** pada commit
`cfa10b8fe1e5a03ac2992a84662d9dcdfafc2cdb`. Diagram dibuat ulang dengan Archify.
Tujuannya menjelaskan apa yang dapat dikerjakan sistem, cara pemrosesan dokumen,
dan cara pengguna memperoleh hasil melalui Streamlit.

## Mulai membaca

| Diagram | Pertanyaan yang dijawab | Sumber yang dapat diperbarui |
| --- | --- | --- |
| [Arsitektur sistem](system-architecture.html) | Bagian apa yang bekerja bersama dan siapa yang menyimpan hasil? | [JSON](system-architecture.json) |
| [Proses dokumen lengkap](document-processing.html) | Bagaimana dokumen dibaca, diperiksa, dan disatukan menjadi hasil? | [JSON](document-processing.json) |
| [Perjalanan pengguna Streamlit](streamlit-user-journey.html) | Bagaimana mengunggah, memantau, memeriksa, dan mengunduh hasil? | [JSON](streamlit-user-journey.json) |

Buka HTML dengan browser. Setiap diagram berdiri sendiri, menyediakan tema
terang/gelap, penjelajahan komponen, referensi sumber, serta ekspor visual.
Label dan penjelasan menggunakan bahasa Indonesia; kontrol bawaan viewer Archify
menggunakan bahasa Inggris. Diagram tidak memerlukan layanan model untuk dibuka.

## Dokumen yang dapat diproses

Format file menentukan cara menyiapkan sumber. Profil dokumen menentukan aturan
membaca isinya; keduanya merupakan pilihan yang berbeda.

| Format | Ekstensi | Cara utama menyiapkan dokumen |
| --- | --- | --- |
| PDF | `.pdf` | Render halaman menjadi gambar; gunakan text-layer sebagai bukti pembanding bila tersedia. |
| Word | `.doc`, `.docx` | Konversi ke PDF melalui LibreOffice, lalu proses halaman secara berurutan. |
| Spreadsheet | `.xls`, `.xlsx`, `.xlsm`, `.ods` | Survei sheet, sel, formula, dan sumber grafik; pertahankan data native dan render bagian visual sesuai kebutuhan. |
| PowerPoint | `.ppt`, `.pptx` | Render slide menjadi gambar dan baca isi visualnya. CLI juga menyediakan pilihan ekstraksi native. |
| Gambar / scan | `.png`, `.jpg`, `.jpeg`, `.webp` | Proses gambar sebagai satu halaman. Screenshot termasuk dalam jalur ini. |

Daftar unggahan berasal dari [Streamlit](../../app/streamlit_logic.py#L64).
Pemilihan pemroses format ada pada [CLI](../../main.py#L346).
Konversi Office dan OCR memerlukan dependensi runtime yang sesuai; daftar format
ini tidak berarti semua file mempunyai struktur atau kualitas yang sama.

### Profil isi dan tata letak

| Profil | Contoh dokumen | Fokus hasil |
| --- | --- | --- |
| `plain` | Surat, memo, pengumuman, formulir sederhana | Teks, identitas, pasangan kunci–nilai, dan tabel. |
| `markdown_hierarchy` | SOP, SK, kebijakan, perjanjian, laporan | Bab, subbab, heading, penomoran, dan daftar bertingkat. |
| `bilingual_journal` | Artikel multikolom atau dokumen dua bahasa | Urutan baca per kolom dan bahasa asli. Tidak otomatis menerjemahkan. |
| `presentation_slides` | Slide presentasi dan materi sosialisasi | Judul, poin, tabel, serta penjelasan visual per slide. |
| `chat_transcript` | Screenshot percakapan | Pengirim, waktu bila terlihat, pesan, kutipan, dan lampiran. |
| `signature_form` | Formulir persetujuan, tanda tangan, paraf | Informasi dan posisi persetujuan yang terlihat; status tidak ditebak. |

Sistem dapat mengenali lebih dari satu profil pada halaman yang sama.
Di sidebar Streamlit tersedia **otomatis dan empat profil manual pertama**;
percakapan dan formulir tanda tangan tetap termasuk kemampuan klasifikasi sistem.
Definisi profil: [prompts](../../app/prompts.py#L184).
Pilihan sidebar: [SPEC_OPTIONS](../../app/streamlit_logic.py#L81).

## Ide utama pemrosesan

1. **Gunakan sumber yang paling sesuai.** Workbook menyediakan angka dan struktur
   secara native. PDF dapat menyediakan text-layer. Halaman visual tetap dibaca
   dari gambar untuk memahami susunan dan unsur yang tidak terdapat pada teks.
2. **OCR menghasilkan draft, VLM memahami visual.** Main memakai satu VLM utama
   untuk inspeksi, klasifikasi, pembacaan ulang, diagram, dan judge. OCR aktif jika
   dikonfigurasi, menggunakan PaddleOCR-VL 1.6: layout berjalan lokal dan recognizer
   memakai endpoint `llama-cpp-server`. [Konfigurasi](../../app/config.py#L111),
   [pemilihan OCR](../../app/graph.py#L188), [runtime Paddle](../../app/paddle_ocr.py#L162).
3. **Keputusan mengikuti kondisi halaman.** Halaman kosong dapat berhenti lebih
   awal. OCR dinilai memakai sinyal gambar, struktur, dan bukti native bila ada.
   Hasil yang tidak dipercaya atau tidak tersedia memicu pembacaan VLM; visual
   rescue dapat meminta pembacaan VLM meskipun OCR dipercaya.
4. **Spesialis hanya digunakan saat relevan.** Flowchart, workflow, swimlane, dan
   pohon keputusan dapat menjadi Mermaid yang diperiksa sintaksnya. Grafik,
   arsitektur blok, foto, dan visual lain menjadi deskripsi terstruktur.
5. **Teks dan data memiliki keluaran berbeda.** Markdown menyimpan susunan bacaan;
   SQLite menyimpan tabel yang dipilih untuk pencarian dan perhitungan data.
   Pemeriksaan konsistensi membantu menemukan perbedaan, bukan menjamin kebenaran
   isi terhadap dokumen asli.

### Jalur halaman visual

Render / siapkan gambar → praproses → inspeksi VLM → normalisasi orientasi → OCR
opsional dan quality gate → pilih draft → spesialis visual bila diperlukan →
gabungkan dan periksa → simpan tabel serta hasil halaman → lanjutkan halaman →
satukan dokumen → audit / ekspor hasil.

- Jika retry rotasi aktif dan trust OCR belum tinggi, main mencoba 90/180/270° dan
  memilih kandidat terbaik. Kondisi ini juga dapat berlaku setelah OCR error.
- Draft OCR yang dipercaya dapat disusun ulang dari region melalui TextReFlow.
  Tabel OCR yang lebih lengkap dapat dipertahankan setelah pembacaan VLM.
- Judge dilewati untuk halaman kosong; tabel OCR panjang dengan sedikitnya 20
  baris tanpa hasil Mermaid; atau fast-path bersih yang memenuhi syarat.
  Di main, pengecualian tabel panjang juga berlaku pada mode `thorough`.
- Pada PDF, ingest tabel dan checkpoint berlangsung **per halaman**, sebelum
  stitching akhir. Konteks halaman sebelumnya dipakai untuk kesinambungan isi.
- Audit akhir PDF membandingkan Markdown dengan SQLite bila database tersedia.
  Streamlit menghitung laporan kembali ketika tab pemeriksaan dibuka.

Sumber: [graph](../../app/graph.py#L215),
[retry OCR](../../app/paddle_ocr.py#L263),
[penyimpanan per halaman dan audit PDF](../../app/pdf.py#L554).

### Jalur Excel

Survei workbook → identifikasi peran sheet dan region → ambil data native →
render region visual bila diperlukan → jalankan pipeline visual → simpan data
native / artefak → gabungkan Markdown per sheet.

Nilai sel, formula yang tersedia, dan sumber grafik dipertahankan dari workbook.
Region besar dapat dibagi menjadi tile, dengan DPI menyesuaikan kebutuhan.
Workbook tanpa unit visual dapat selesai melalui jalur native. Cabang native pada
diagram merangkum penyimpanan tabel native dan komposisi sheet, tanpa loop halaman
PDF. Jika survei gagal,
pemrosesan beralih ke render sheet. Preview sheet untuk pengguna merupakan
artefak pembanding; keberadaannya tidak berarti semua sheet dibaca ulang oleh VLM.
[Pemroses Excel](../../app/excel.py#L3230).

## Menggunakan Streamlit

Jalankan dari root repositori sesuai lingkungan yang telah disiapkan:

```powershell
uv run streamlit run app/streamlit_logic.py
```

### Unggah dan mulai

1. Buka **Upload** dari Dashboard atau navigasi workspace.
2. Biarkan **Bentuk dokumen** pada pilihan otomatis bila belum mengetahui profilnya.
   Pengaturan lanjutan menyediakan ketajaman gambar 100–300 DPI dan
   **Simpan semua jenis tabel**.
3. Pilih satu/banyak file atau satu folder. Subfolder ikut dibaca; struktur folder
   tidak disimpan. Pilihan file dan folder digabung ke antrean yang sama.
4. Tinjau daftar, tentukan file prioritas bila ada beberapa file, dan isi nama batch.
5. Klik **Mulai ekstraksi**. Pekerjaan menunggu slot, kemudian diproses di background.

### Pantau pekerjaan

Halaman **Dokumen** menampilkan status, progres, aktivitas terakhir, serta preview
halaman yang sudah selesai. Preview sementara masih dapat berubah ketika hasil
disatukan. Pengguna dapat memprioritaskan antrean, menjeda/melanjutkan pekerjaan,
membatalkan, atau mencoba ulang setelah gagal. Log terakhir membantu meninjau
kendala. Navigasi dan refresh dapat membuka kembali dokumen yang dipilih melalui
workspace; pekerjaan dikelola di luar lifecycle tab browser.

### Periksa hasil dan unduh

Kelima tab berikut bisa dibuka dalam urutan apa pun.

| Tab | Fitur dan tujuan |
| --- | --- |
| **Cocokkan dengan dokumen asli** | Bandingkan gambar halaman/slide atau preview sheet dengan hasil; pilih bagian yang ingin dilihat dan unduh gambar yang tersedia. Koreksi halaman tersedia ketika teksnya dapat dipetakan. |
| **Periksa kelengkapan data** | Lihat perbandingan tabel/baris Markdown dan SQLite, status konsistensi, serta bagian yang perlu diperiksa. |
| **Baca dan unduh teks** | Cari kata/frasa, baca hasil Markdown, unduh teks, dan lihat/unduh diagram yang tersedia. |
| **Lihat tabel dan grafik** | Jelajahi tabel, relasi data transaksi, ekspor CSV/database/SQL, buat grafik, jalankan query SELECT, dan bersihkan data ganda bila diperlukan. |
| **Detail proses** | Baca dan unduh log untuk peninjauan proses. |

**Siapkan ZIP hasil** mengemas artefak dokumen yang tersedia. **Ekstrak Ulang
Dokumen Ini** memulai ulang pemrosesan. Pada **Histori**, cari nama, filter status,
buka dokumen/batch, pantau batch, unduh hasil kelompok, atau hapus file/batch
melalui konfirmasi UI. [Tab hasil](../../app/streamlit_logic.py#L1999),
[histori](../../app/streamlit_logic.py#L430).

### Koreksi dan peningkatan hasil

Koreksi halaman disimpan terpisah dari hasil unduhan awal. Pengguna dapat menyimpan
draf atau menandainya **Siap digunakan untuk evaluasi**. Evaluasi, optimasi prompt
DSPy/GEPA, serta aktivasi kandidat dilakukan melalui CLI lokal; tombol koreksi
tidak menjalankan pelatihan atau mengubah bobot model.
[Form koreksi](../../app/learning_ui.py#L12),
[optimasi](../../app/dspy_learning.py#L214),
[pengelolaan kandidat](../../app/learning_cli.py).

## Jalur integrasi selain Streamlit

- **CLI** memakai pipeline ekstraksi dokumen; batch dan pilihan pemrosesan tersedia
  melalui argumen CLI. [main](../../main.py#L53).
- **FastAPI** menerima satu file di `POST /ingest`, menjalankan pekerjaan ekstraksi,
  lalu mengembalikan ZIP berisi Markdown, dump SQL, dan CSV jika tersedia.
  [API](../../app/api.py#L98).
- **MCP lokal** menyediakan tool ekstraksi, klasifikasi, diagram, dan database.
  [server](../../app/mcp_server.py#L173).
- **MCP agent** mengirim gambar dan arahan kepada aplikasi klien, kemudian menerima
  hasil halaman untuk disimpan. Model yang membaca gambar dipilih oleh klien MCP,
  bukan oleh konfigurasi VLM lokal. [server agent](../../app/mcp_agent_server.py#L1024).
- **Deep Agent** merupakan harness tersendiri dengan master dan delapan subagent:
  layout, Markdown/OCR, diagram, presentasi, PDF, Word, Excel, dan tabel/database.
  Mereka memilih tool; profil prompt pada `agents.py` merupakan aturan ekstraksi
  deterministik. Deep Agent tidak menjadi worker default Streamlit.
  [harness](../../app/deep_agent.py#L315), [profil](../../app/agents.py#L1).

Pratinjau chunk tersedia melalui CLI/tool sebagai bahan persiapan pemakaian RAG.
Diagram ini tidak menggambarkan layanan pencarian RAG yang sudah diterapkan.
Surveyor geometri dan audit hierarchy merupakan modul tambahan; keduanya tidak
diposisikan sebagai tahap wajib pipeline halaman.

## Memperbarui dan memeriksa diagram

Perbarui JSON setelah menelusuri perilaku sumber, lalu sesuaikan referensi baris
dan commit pada `meta.repository`. Jalankan Archify dari root repositori:

```powershell
node '<lokasi-skill-archify>/bin/archify.mjs' finalize architecture docs/architecture/system-architecture.json docs/architecture/system-architecture.html --repo-root . --quality showcase --json
node '<lokasi-skill-archify>/bin/archify.mjs' finalize workflow docs/architecture/document-processing.json docs/architecture/document-processing.html --repo-root . --quality showcase --json
node '<lokasi-skill-archify>/bin/archify.mjs' finalize workflow docs/architecture/streamlit-user-journey.json docs/architecture/streamlit-user-journey.html --repo-root . --quality showcase --json
```

Receipt finalisasi dan bukti browser harus sesuai dengan HTML yang diserahkan.
Jika kandidat berubah setelah pemeriksaan browser, gunakan direktori bukti baru
dengan `--out-dir`. Pemeriksaan diagram tidak menjalankan pipeline ekstraksi atau
memanggil endpoint VLM/OCR. Hasil validasi dan pemeriksaan visual dicatat terpisah
di [VALIDATION.md](VALIDATION.md).
