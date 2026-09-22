# Migrasi PaddleOCR-VL 1.6 dan TextReflow

Branch implementasi: `experiment/paddleocr-vl-textreflow`.

## Keputusan

- Gemma 4 26B-A4B tetap menjadi VLM utama pada port 8080.
- Unlimited-OCR digantikan oleh pipeline resmi PaddleOCR-VL 1.6; recognizer GGUF
  tetap dilayani llama.cpp pada port 8081, sedangkan layout analyzer berjalan di
  proses Python aplikasi.
- TextReflow berjalan lokal tanpa backend tambahan dan hanya diterapkan pada
  halaman prosa satu/dua kolom yang tidak memuat tabel, formula, atau figure.
- Kegagalan PaddleOCR-VL atau TextReflow selalu kembali ke jalur Gemma atau
  Markdown Paddle asli.
- File systemd tidak diubah karena konfigurasi deployment akan diselesaikan
  terpisah.

## Validasi

- Unit test mencakup normalisasi region Paddle, crop/manifest, error recovery,
  pemilihan backend, reading order TextReflow, de-hyphenation, dan guard halaman.
- Validasi akhir menggunakan environment Conda `magang-jds`, unit test jalur OCR
  dan pipeline terkait, `ruff check --fix`, serta `ty check`.
