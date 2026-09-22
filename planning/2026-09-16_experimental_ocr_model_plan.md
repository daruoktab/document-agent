# Implementasi Eksperimental Unlimited-OCR

**Tanggal:** 2026-09-17  
**Branch:** `experiment/ocr-model-v2`  
**Status:** Arsitektur inti sudah diimplementasikan; menunggu alias model dan pengujian server

## Sasaran

Memisahkan pekerjaan dokumen di antara dua model nyata:

- **Unlimited-OCR** membuat draft Markdown, grounding bounding box, serta crop tabel/figure.
- **Qwen3.8-27B** tetap menjadi model utama untuk klasifikasi, reasoning, specialist Mermaid, dan quality gate adaptif.
- Kegagalan atau konfigurasi OCR yang belum lengkap tidak menggagalkan dokumen; ekstraksi draft kembali ke VLM utama.

OCR bukan mode, shadow process, atau sekadar referensi. Ketika `OCR_MODEL` terisi dan endpoint sehat, hasil OCR benar-benar menjadi draft utama pipeline.

## Topologi Runtime

| Peran | Env | Default endpoint | Catatan |
|---|---|---|---|
| VLM utama | `BASE_URL`, `VLM_MODEL` | `http://127.0.0.1:8080/v1` | Qwen; agent, reasoning, judge, Mermaid |
| OCR | `OCR_BASE_URL`, `OCR_MODEL` | `http://127.0.0.1:8081/v1` | Server llama.cpp terpisah; `OCR_MODEL` boleh kosong selama setup |

Server OCR membutuhkan build llama.cpp yang memahami DeepSeek-OCR serta pasangan model GGUF + `mmproj`. Alias final model ditentukan pada server dan dimasukkan ke `OCR_MODEL` tanpa perubahan kode.

## Alur yang Diimplementasikan

1. Preprocess gambar halaman dan lakukan inspeksi layout + orientasi melalui VLM utama.
2. Putar halaman 0/90/180/270 derajat agar tegak sebelum OCR.
3. Panggil Unlimited-OCR menggunakan prompt grounding.
4. Nilai trust OCR dari bentuk output, grounding, kepadatan tinta, indikasi halaman menyamping, repetisi, dan kemiripan dengan text-layer PDF bila tersedia.
5. Untuk kandidat berisiko, coba orientasi alternatif dan pilih skor terbaik. Halaman benar-benar kosong dihentikan tanpa memanggil OCR.
6. Parse `<|det|>label [x1,y1,x2,y2]<|/det|>` secara defensif dan simpan crop `table`/`figure` beserta manifest audit.
7. Gunakan Markdown OCR hanya bila trust `high` dan inspeksi VLM tidak meminta *visual rescue*. Untuk trust `medium`/`low`, error, OCR nonaktif, atau halaman berfont sangat kecil/padat, teks miring penting, anotasi teknis kecil, multi-kolom rapat, maupun kontras rendah, jalankan ekstraksi VLM independen—bukan memperbaiki draft OCR yang mungkin salah.
8. Kirim crop figure ke specialist Mermaid dan jalankan judge VLM pada halaman non-kosong. Koreksi judge dicatat sebagai `corrected_by_vlm`.
9. Lanjutkan jalur SQLite/tabular dan penyatuan multi-halaman yang sudah ada.

Alur tersebut digunakan oleh pipeline deterministik serta tool dokumen di Deep Agent.

## Kontrak Output

`PipelinePageResult` membawa:

- `ocr_status`: `accepted`, `retried_rotated`, `corrected_by_vlm`, `fallback_vlm`, `vlm_visual_rescue`, `blank_page`, `disabled`, atau `error`;
- nama model dan latency OCR;
- skor/trust OCR, risk flags, serta rotasi final yang diterapkan;
- daftar region dengan label, jenis, koordinat model, koordinat piksel, teks region, dan path crop;
- path manifest region;
- hitungan tabel/visual yang menggabungkan sinyal Markdown dan region OCR.

Isi respons tidak ditulis ke log pipeline umum. Manifest berada di direktori output dokumen untuk audit lokal.

## Konfigurasi Patokan

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
OCR_MAX_TOKENS=4096
OCR_PROMPT=<|grounding|>Convert the document to markdown.
OCR_COORDINATE_SIZE=1024
OCR_CROP_PADDING=0.01
OCR_MIN_TRUST_SCORE=0.72
OCR_MEDIUM_TRUST_SCORE=0.48
OCR_ROTATION_RETRY=true
OCR_BLANK_INK_RATIO=0.0002
OCR_SPARSE_INK_RATIO=0.015
```

## Pekerjaan Setelah Server OCR Siap

1. Isi `OCR_MODEL` dengan alias yang benar dan pastikan `/v1/chat/completions` menerima image data URL.
2. Uji satu gambar, satu PDF multi-halaman, dan satu PPT dengan tabel/diagram.
3. Validasi apakah koordinat model benar-benar memakai ruang `1024`; ubah `OCR_COORDINATE_SIZE` bila berbeda.
4. Ukur OCR accuracy, fidelity tabel, kualitas crop, kualitas Mermaid, latency p50/p95, serta penggunaan VRAM.
5. Tambahkan batas concurrency apabila dua server berbagi GPU yang sama.
6. Putuskan quant produksi setelah perbandingan minimal Q4_K_M dan Q6_K/Q8_0.

## Kriteria Penerimaan

- VLM dan OCR memakai port terpisah dan dapat hidup/mati secara independen.
- `OCR_MODEL=` mempertahankan jalur lama melalui fallback VLM.
- OCR aktif menghasilkan Markdown primer hanya ketika quality gate memberi trust tinggi.
- Error/timeout OCR menghasilkan dokumen lewat fallback, bukan kegagalan total.
- PDF dengan text-layer membandingkan bukti native dengan hasil OCR; mismatch memicu VLM independen.
- Halaman 90/180/270 derajat dapat dinormalisasi atau dicoba ulang, dan halaman kosong tidak dianggap error.
- Crop figure dipakai specialist diagram; tabel/figure tidak perlu dianalisis dari seluruh halaman bila bbox tersedia.
- Tidak ada tag grounding mentah di Markdown final.
- Unit test konfigurasi, parser bbox, persist crop, fallback, dan integrasi pipeline lulus.

## Risiko Terbuka

| Risiko | Mitigasi |
|---|---|
| Koordinat output model berbeda dari skala 1024 | `OCR_COORDINATE_SIZE` dapat dikonfigurasi dan bbox selalu di-clamp |
| OCR mengulang/loop pada halaman padat | temperature 0, token limit, timeout; evaluasi repetition penalty di server |
| Dua model berebut VRAM | jalankan server pada port/proses terpisah dan atur concurrency/deployment GPU |
| OCR mengeluarkan teks masuk akal tetapi salah | skor trust + text-layer PDF + ekstraksi VLM independen + judge citra asli |
| PDF hasil render kehilangan EXIF orientasi | inspeksi VLM mengeluarkan rotasi eksplisit; OCR punya retry rotasi cadangan |
| Jumlah figure besar memicu banyak call Mermaid | ukur beban; tambahkan cap/prioritas region setelah data nyata tersedia |
