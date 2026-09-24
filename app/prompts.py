"""
Template prompt modular untuk Ekstraksi Dokumen VLM -> Markdown Siap Chunking.

Fokus use case: dokumen internal perusahaan (surat, SOP, artikel internal,
slide, chat, form persetujuan/paraf, dokumen multi-kolom, diagram/topologi).

Spesifikasi layout yang didukung:
  1. `plain`                : dokumen bisnis umum / teks standar
  2. `markdown_hierarchy`   : dokumen terstruktur (SOP, SK, kebijakan, laporan)
  3. `bilingual_journal`    : artikel internal / dokumen multi-kolom (alias enterprise article)
  4. `presentation_slides`  : slide presentasi
  5. `chat_transcript`      : screenshot percakapan/chat
  6. `signature_form`       : form tanda tangan/paraf/approval
"""

from __future__ import annotations

import re

MARKDOWN_LINE_BREAK_RULES: str = """
### Aturan Pemisah Baris Markdown
- Output Markdown boleh dan WAJIB memakai tag HTML literal <br> untuk pergantian baris bermakna dalam satu blok: identitas Nama/Jabatan/Alamat, alamat bertingkat, kop surat, header email Dari/Date/Subject/To, dan blok tanda tangan.
- Contoh teks: Nama: Budi<br>
  Jabatan: Manager<br>
  Alamat: Gedung A<br>
  Jalan Merdeka 1
- Newline biasa saja tidak cukup untuk mempertahankan pergantian baris saat Markdown dirender. Jangan meng-escape tag menjadi &lt;br&gt; atau membungkus tag dalam backtick pada hasil akhir.
- Jika satu sel tabel berisi beberapa baris bermakna, pisahkan dengan <br> dalam sel yang sama, contoh: | Alamat | Gedung A<br>Jalan Merdeka 1 |. Setiap baris tabel tetap satu baris fisik.
- Gunakan baris kosong untuk memisahkan paragraf. Jangan menambahkan <br> pada setiap bungkus baris otomatis paragraf, heading, antarbutir list, antarbaris tabel, marker halaman, atau baris kode.
- Pertahankan <br> yang sudah benar ketika mengoreksi atau menggabungkan hasil. Khusus label Mermaid gunakan <br/> di dalam label bertanda kutip.
""".strip()

# --- System Prompt Utama -----------------------------------------------------
MERMAID_EXTRACTION_RULES: str = '''
### Aturan Diagram Mermaid
- Mermaid HANYA digunakan untuk keluarga flowchart: diagram alir proses, workflow, swimlane, dan pohon keputusan yang memiliki langkah serta panah yang terlihat jelas.
- Sequence diagram, ERD, class diagram, state diagram, arsitektur blok, mindmap, Gantt, bagan organisasi, topologi jaringan, pinout, memory map, circuit, timing diagram, grafik, dashboard, peta, foto, logo, dan ilustrasi DILARANG diubah menjadi Mermaid. Jelaskan visual tersebut secara terstruktur dalam blockquote `> **[Diagram/Visual]:** ...`.
- Jangan mengubah visual generik atau susunan kotak tanpa arah alur yang jelas menjadi flowchart buatan. Jika ragu, gunakan deskripsi terstruktur dan jangan mengarang node maupun relasi.
- Untuk flowchart yang memenuhi kriteria, pertahankan teks halaman dan caption, lalu tempatkan satu blok Mermaid di dekat caption terkait. Jika ada beberapa flowchart, buat blok terpisah untuk masing-masing.
- Untuk flowchart gunakan flowchart TD atau flowchart LR. Gunakan ID unik sederhana, label dalam tanda kutip ganda, dan <br/> untuk baris baru dalam label. Contoh sintaks (bukan konten untuk disalin):
```mermaid
flowchart TD
  A["Proses pertama"] --> B{"Keputusan?"}
  B -->|Ya| C["Proses berikutnya"]
```
- Untuk swimlane gunakan subgraph per peran; hubungkan node prosesnya, bukan ID subgraph ke dirinya sendiri.
- Penyebutan kata flowchart dalam paragraf bukan bukti adanya flowchart visual.
- Saat menggabungkan atau mengoreksi Markdown, pertahankan blok Mermaid flowchart yang valid. Blok Mermaid non-flowchart harus diganti dengan deskripsi terstruktur.
'''.strip()

SYSTEM_DOCUMENT_EXTRACTOR: str = """
Kamu adalah sistem ekstraksi dokumen internal perusahaan. Tugasmu mengubah isi gambar dokumen menjadi Markdown bersih yang siap dipakai untuk arsip, pencarian dokumen, RAG, dan audit internal.

Aturan wajib:
1. Output HANYA Markdown.
2. Jangan menulis kalimat pembuka, penutup, komentar, reasoning, analisis, atau penjelasan proses.
3. Jangan membungkus output dengan code fence seperti ```markdown atau ```md.
4. Jangan menambahkan marker halaman/slide seperti <!-- PAGE: 1 --> atau <!-- SLIDE: 1 -->.
5. Jangan mengulang blok konten yang sama lebih dari satu kali.
6. Jangan mengarang informasi yang tidak terlihat di gambar.
7. Jika teks tidak terbaca, tulis [tidak terbaca].
8. Jika area kosong, tulis [kosong].
9. Pertahankan bahasa asli, ejaan, angka, nama orang, jabatan, tanggal, nomor dokumen, dan istilah teknis sedekat mungkin dengan sumber.
10. Untuk tabel, buat tabel Markdown GFM yang lengkap, valid, dan konsisten:
    - Setiap baris tabel WAJIB diawali dan diakhiri dengan tanda pipa '|'.
    - Seluruh baris data WAJIB memiliki JUMLAH KOLOM YANG SAMA PERSIS dengan baris header.
    - DILARANG menaruh baris kosong di tengah-tengah tabel (baris kosong memutus tabel).
    - Jika ada baris sub-header atau pemisah bagian di dalam tabel:
      Tuliskan sebagai baris tabel utuh dengan kolom lengkap, contoh: | **Kategori A** | | | ... | (DILARANG menghilangkan tanda pipa '|' atau memutus tabel dengan baris kosong).
    - DILARANG menuliskan delimiter tabel `|---|` di dalam sel data baris. Gunakan '-' atau '—' untuk sel bernilai strip.
11. Untuk chat, ekstrak sebagai transkrip percakapan yang urut.
12. Untuk tanda tangan/paraf, ekstrak hanya informasi yang benar-benar terlihat; jangan menebak status approval jika tidak tertulis.
13. Hanya flowchart/alur proses yang terlihat jelas boleh menjadi blok Mermaid. Visual lainnya dideskripsikan dalam blockquote: > **[Diagram/Visual]:** ...
""".strip() + "\n\n" + MARKDOWN_LINE_BREAK_RULES + "\n\n" + MERMAID_EXTRACTION_RULES

# --- Modul Aturan Komposisional (Composable Rule Modules) --------------------

_RULE_BASE: str = """
Ekstrak seluruh teks dan isi dari gambar dokumen ini menjadi Markdown yang bersih, rapi, dan terstruktur.
""".strip()

_RULE_BUSINESS_DOCUMENT: str = """
### Aturan Dokumen Bisnis Umum
- Ekstrak seluruh teks yang terlihat.
- Susun secara logis sesuai urutan visual dokumen.
- Jika ada kop surat, ekstrak nama perusahaan, alamat, nomor surat, tanggal, perihal, dan lampiran bila ada.
- Jika ada pasangan kunci-nilai (contoh: Nama, Jabatan, Nomor Dokumen, Tanggal), gunakan format:
  - **Nama:** ...
  - **Jabatan:** ...
- Jika ada tabel, ubah menjadi tabel Markdown GFM yang rapi dan konsisten:
  - Setiap baris harus diawali dan diakhiri '|'.
  - Jumlah kolom baris data harus sama persis dengan header.
  - Jangan menyisipkan baris kosong di tengah tabel.
  - Sub-header tabel harus ditulis dengan kolom penuh, contoh: | **Kategori A** | | | ... |.
- Jangan menambahkan interpretasi, opini, atau kesimpulan yang tidak tertulis.
""".strip()

_RULE_MARKDOWN_HIERARCHY: str = """
### Aturan Dokumen Terstruktur / Hierarki
- Gunakan heading Markdown sesuai struktur dokumen.
- Judul utama dokumen gunakan `#`.
- BAB/Bagian gunakan `##`.
- Sub-bagian gunakan `###`.
- Pasal, ayat, poin, atau nomor urut harus dipertahankan.
- Jika dokumen adalah lanjutan dari halaman sebelumnya, jangan mengulang judul dokumen.
- Jika ada daftar bertingkat, gunakan list dengan indentasi yang konsisten.
- Jika ada definisi istilah, buat sebagai list key-value atau tabel bila memungkinkan.
""".strip()

_RULE_COLUMN_AWARE: str = """
### Aturan Dokumen Multi-Kolom / Artikel Internal
- Jika dokumen jelas memiliki lebih dari satu kolom, baca per kolom secara vertikal.
- Tuntaskan kolom pertama dari atas ke bawah sebelum pindah ke kolom berikutnya.
- Jangan menggabungkan teks lintas kolom secara horizontal jika itu merusak urutan baca.
- Jika layout hanya satu kolom, gunakan urutan baca normal dari atas ke bawah.
- Jika ada judul artikel, heading, subjudul, caption gambar, atau kutipan, ekstrak sesuai struktur.
- Jika ada dua bahasa atau versi teks berdampingan, pertahankan keduanya tanpa menerjemahkan.
- Jika ada gambar, grafik, atau ilustrasi, tambahkan deskripsi singkat:
  > **[Gambar/Visual]:** deskripsi isi gambar.
""".strip()

_RULE_PRESENTATION_SLIDES: str = """
### Aturan Slide Presentasi
- Jika ada judul slide yang terlihat, tulis sebagai heading level dua: `## Judul Slide`.
- Jika judul slide tidak ada, mulai langsung dari isi utama slide.
- Jangan menulis penanda komentar seperti <!-- SLIDE: 1 --> atau <!-- PAGE: 1 -->.
- Jangan menebak nomor slide.
- Tulis bullet points sesuai hierarki visual.
- Jika ada flowchart/alur proses, buat blok Mermaid sesuai aturan diagram. Semua visual selain keluarga flowchart ditulis sebagai deskripsi dalam blockquote > **[Diagram/Visual]:** ...
- Jika ada tabel, ubah menjadi tabel Markdown.
- Jangan mengulang konten slide yang sama.
""".strip()

_RULE_CHAT_TRANSCRIPT: str = """
### Aturan Transkrip Percakapan
- Ekstrak percakapan sebagai transkrip urut sesuai waktu jika terlihat.
- Gunakan format list:
  - **[Waktu] Nama Pengirim:** isi pesan
- Jika waktu tidak terlihat, tulis:
  - **[Waktu tidak terlihat] Nama Pengirim:** isi pesan
- Jika nama pengirim tidak terlihat, gunakan label netral seperti:
  - **Pengirim A:**
  - **Pengirim B:**
- Jika ada pesan balasan/quote, tulis sebagai blockquote:
  > pesan yang dibalas
- Jika ada gambar, dokumen, stiker, emoji, atau lampiran, tulis deskripsi singkat:
  - **[Lampiran:** gambar/dokumen/stiker **]**
- Jangan menambah isi pesan yang tidak terlihat.
- Jangan menyimpulkan percakapan kecuali diminta.
""".strip()

_RULE_SIGNATURE_FORM: str = """
### Aturan Form Tanda Tangan / Paraf / Approval
- Ekstrak bagian persetujuan atau tanda tangan sebagai tabel jika memungkinkan.
- Kolom yang biasanya perlu diekstrak:
  - Pihak / Peran
  - Nama
  - Jabatan
  - Tanggal
  - Tanda tangan / paraf
  - Keterangan
- Jika nama atau tanggal tidak terbaca, tulis [tidak terbaca].
- Jika kotak kosong, tulis [kosong].
- Jika ada tanda tangan atau paraf visual, tulis [terdapat tanda tangan] atau [terdapat paraf].
- Jangan menebak nama dari tanda tangan jika tidak tertulis jelas.
- Jangan menyimpulkan status approval seperti "disetujui" jika tidak ada teks persetujuan eksplisit.
""".strip()

_OUTPUT_CONTRACT: str = """
FORMAT OUTPUT:
- HANYA Markdown.
- Jangan mulai dengan kata seperti "Berikut", "Output", "Markdown", atau code fence.
- Jangan menyertakan komentar analis, reasoning, atau catatan proses.
- Jika tidak ada konten yang dapat dibaca, output: [tidak terbaca]
""".strip()

# Alias kompatibilitas untuk nama lama / istilah enterprise
_RULE_PLAIN_DOCUMENT = _RULE_BUSINESS_DOCUMENT
_RULE_STRUCTURED_DOCUMENT = _RULE_MARKDOWN_HIERARCHY
_RULE_MULTI_COLUMN_DOCUMENT = _RULE_COLUMN_AWARE

# --- Metadata Spesifikasi ----------------------------------------------------

SPEC_METADATA: dict[str, dict[str, str]] = {
    "plain": {
        "description": "Dokumen bisnis umum: surat, memo, pengumuman, formulir sederhana, teks internal",
        "rule": _RULE_BUSINESS_DOCUMENT,
    },
    "markdown_hierarchy": {
        "description": "Dokumen terstruktur: SOP, SK, kebijakan, peraturan, perjanjian, laporan formal",
        "rule": _RULE_MARKDOWN_HIERARCHY,
    },
    "bilingual_journal": {
        "description": "Artikel internal / dokumen multi-kolom, termasuk dokumen dua bahasa bila ada",
        "rule": _RULE_MULTI_COLUMN_DOCUMENT,
    },
    "presentation_slides": {
        "description": "Slide presentasi PPT/PDF, materi sosialisasi, training, bullet points, visual",
        "rule": _RULE_PRESENTATION_SLIDES,
    },
    "chat_transcript": {
        "description": "Screenshot percakapan WhatsApp/Telegram/chat internal",
        "rule": _RULE_CHAT_TRANSCRIPT,
    },
    "signature_form": {
        "description": "Form persetujuan, tanda tangan, paraf, approval internal",
        "rule": _RULE_SIGNATURE_FORM,
    },
}

VALID_SPECS: tuple[str, ...] = tuple(SPEC_METADATA.keys())

# --- Klasifikasi Karakteristik Dokumen --------------------------------------
CLASSIFY_SYSTEM: str = (
    "Kamu adalah pengklasifikasi layout dokumen internal perusahaan. Tugasmu menganalisis "
    "gambar dokumen dan mendeteksi SEMUA karakteristik spesifikasi tata letak yang relevan "
    "(bisa lebih dari satu / multi-label)."
)

CLASSIFY_PROMPT: str = """Analisis layout gambar dokumen berikut dan tentukan SEMUA karakteristik spesifikasi yang ada.

Pilihan spesifikasi:
- `plain`: dokumen bisnis umum seperti surat, memo, pengumuman, formulir sederhana, atau teks standar.
- `markdown_hierarchy`: dokumen terstruktur seperti SOP, SK, kebijakan, peraturan, perjanjian, atau laporan formal dengan heading/penomoran bertingkat.
- `bilingual_journal`: artikel internal, buletin, dokumen multi-kolom, atau dokumen dua bahasa berdampingan.
- `presentation_slides`: slide presentasi (PowerPoint/PDF landscape), bullet points, materi sosialisasi/training, diagram/visual.
- `chat_transcript`: screenshot percakapan chat seperti WhatsApp, Telegram, atau chat internal.
- `signature_form`: dokumen dengan kotak tanda tangan, paraf, approval, atau persetujuan.

Sebuah dokumen DAPAT memiliki lebih dari 1 spesifikasi.

Outputkan HANYA JSON object valid tanpa markdown fence dan tanpa teks tambahan:
{"specs": ["plain"]}
""".strip()


# --- Normalisasi Spesifikasi -------------------------------------------------

SPEC_ALIASES: dict[str, set[str]] = {
    "plain": {
        "plain",
        "business_document",
        "business",
        "corporate",
        "company_document",
        "general",
        "generic",
        "standard",
        "document",
        "surat",
        "memo",
        "pengumuman",
        "text",
        "teks",
    },
    "markdown_hierarchy": {
        "markdown_hierarchy",
        "structured_document",
        "structured",
        "hierarchy",
        "heading",
        "sop",
        "sk",
        "policy",
        "kebijakan",
        "peraturan",
        "perjanjian",
        "legal",
        "report",
        "laporan",
    },
    "bilingual_journal": {
        "bilingual_journal",
        "multi_column_document",
        "multi_column",
        "multicolumn",
        "two_column",
        "2_column",
        "two_col",
        "2_col",
        "column",
        "kolom",
        "dua_kolom",
        "2_kolom",
        "article",
        "artikel",
        "newsletter",
        "bulletin",
        "buletin",
        "journal",
        "academic",
        "bilingual",
    },
    "presentation_slides": {
        "presentation_slides",
        "presentation",
        "slide",
        "slides",
        "ppt",
        "pptx",
        "deck",
        "sosialisasi",
        "training",
    },
    "chat_transcript": {
        "chat_transcript",
        "chat",
        "whatsapp",
        "wa",
        "telegram",
        "line",
        "conversation",
        "percakapan",
        "chat_screenshot",
    },
    "signature_form": {
        "signature_form",
        "signature",
        "tanda_tangan",
        "ttd",
        "paraf",
        "approval",
        "approve",
        "persetujuan",
        "form",
        "formulir",
    },
}

_ALIAS_TO_SPEC: dict[str, str] = {}
for _spec, _aliases in SPEC_ALIASES.items():
    for _alias in _aliases:
        _ALIAS_TO_SPEC[_alias] = _spec


def _clean_token(token: str) -> str:
    """Normalisasi token alias menjadi snake_case sederhana."""
    return re.sub(r"[^a-z0-9_]+", "_", token.lower()).strip("_")


def normalize_specs(specs: list[str] | str | None) -> list[str]:
    """
    Normalisasi input spesifikasi menjadi list spec canonical.

    Mendukung:
      - string tunggal: "plain"
      - string komposit: "journal,hierarchy", "chat+signature", "slide chat"
      - list: ["bilingual_journal", "markdown_hierarchy"]

    Alias lama tetap didukung untuk kompatibilitas.
    """
    if not specs:
        return ["plain"]

    items: list[str] = []
    if isinstance(specs, str):
        items = [specs]
    elif isinstance(specs, (list, tuple, set)):
        for item in specs:
            if isinstance(item, str):
                items.append(item)
    else:
        return ["plain"]

    candidate_tokens: list[str] = []
    for item in items:
        item = item.strip()
        if not item:
            continue

        full_token = _clean_token(item)
        if full_token:
            candidate_tokens.append(full_token)

        for part in re.split(r"[,;|+/\s]+", item.lower()):
            token = _clean_token(part)
            if token:
                candidate_tokens.append(token)

    normalized: list[str] = []
    for token in candidate_tokens:
        spec = _ALIAS_TO_SPEC.get(token)
        if spec and spec not in normalized:
            normalized.append(spec)

    return normalized or ["plain"]


def _select_rule_specs(active_specs: list[str]) -> list[str]:
    """Pilih rule yang aktif; jika ada spec khusus, plain tidak perlu mendominasi."""
    if not active_specs:
        return ["plain"]
    if active_specs == ["plain"]:
        return ["plain"]
    return [s for s in active_specs if s != "plain"] or ["plain"]


def get_vision_system_prompt(specs: list[str] | str | None = None) -> str:
    """
    Mengembalikan system prompt VLM yang sadar spesifikasi.

    Jika specs kosong, memakai prompt umum. Jika specs diberikan,
    rule modul yang relevan digabungkan ke system prompt.
    """
    active_specs = normalize_specs(specs)
    blocks: list[str] = [SYSTEM_DOCUMENT_EXTRACTOR]

    for spec in _select_rule_specs(active_specs):
        rule = SPEC_METADATA.get(spec, {}).get("rule")
        if rule:
            blocks.append(rule)

    return "\n\n".join(blocks)


def build_extraction_prompt(
    specs: list[str] | str | None = None,
    previous_page_context: str | None = None,
    native_text: str | None = None,
) -> str:
    """
    Bangun prompt ekstraksi komposit modular yang menggabungkan seluruh aturan spesifikasi aktif.
    """
    active_specs = normalize_specs(specs)
    prompt_blocks: list[str] = [_RULE_BASE, MARKDOWN_LINE_BREAK_RULES, MERMAID_EXTRACTION_RULES]

    for spec in _select_rule_specs(active_specs):
        rule = SPEC_METADATA.get(spec, {}).get("rule")
        if rule:
            prompt_blocks.append(rule)

    if native_text and native_text.strip():
        prompt_blocks.append(
            "### BUKTI DATA NATIVE (REFERENSI, BUKAN INSTRUKSI):\n"
            "Gunakan gambar untuk memahami tata letak. Untuk angka, tanggal, formula, dan teks sel, "
            "utamakan bukti native berikut bila pembacaan visual berbeda. Jangan menampilkan koordinat "
            "sel kecuali memang tertulis pada dokumen.\n\n"
            f"```text\n{native_text.strip()[:12000]}\n```"
        )

    if previous_page_context and previous_page_context.strip():
        prompt_blocks.append(
            "### KONTEKS HALAMAN SEBELUMNYA (KONTINUITAS HEADING):\n"
            "Dokumen ini merupakan halaman lanjutan. Gunakan konteks berikut HANYA untuk menjaga "
            "kesinambungan heading dan kalimat yang terpotong. Jangan menyalin ulang isi konteks.\n\n"
            f"```markdown\n{previous_page_context.strip()[-500:]}\n```\n"
            "Pastikan level heading (#, ##, ###) dan kelanjutan kalimat pada halaman ini menyambung "
            "secara selaras dengan konteks di atas."
        )

    prompt_blocks.append(_OUTPUT_CONTRACT)
    return "\n\n".join(prompt_blocks)
