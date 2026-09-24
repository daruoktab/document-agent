# Rencana Riset Pengolahan Excel Heterogen

## Pertanyaan utama

Bagaimana memperkuat pipeline ekstraksi Excel agar mampu menangani workbook acak dengan banyak sheet, banyak region, tabel renggang, judul bertingkat, merged cells, formula, named table/range, objek visual, dan variasi tipe data tanpa menurunkan ketepatan hasil native maupun visual?

## Subtopik

1. **Semantik workbook dan sumber struktur native**
   - Dokumentasi primer Open XML, Microsoft Excel, openpyxl, dan LibreOffice.
   - Informasi yang dapat dipercaya untuk batas sheet, Excel Table, defined names, merged cells, hidden state, formula, dan cached values.

2. **Deteksi tabel dan segmentasi region spreadsheet**
   - Paper atau metodologi primer tentang table detection, layout inference, header detection, dan pemisahan beberapa tabel dalam satu worksheet.
   - Kelemahan connected-components murni serta fitur tambahan yang terbukti berguna.

3. **Arsitektur ekstraksi dan validasi produksi**
   - Praktik resmi untuk memilih jalur native, visual, dan fallback.
   - Penanganan tipe data campuran, batas ukuran, error isolation per sheet/region, provenance, dan verifikasi hasil.

## Sintesis

Temuan akan dibandingkan dengan implementasi `app/excel.py` dan pengujian yang ada. Perubahan dipilih berdasarkan dampak terhadap variasi workbook nyata, kompatibilitas data lama, serta kemampuan diuji secara deterministik tanpa benchmark performa.
