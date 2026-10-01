# Hasil pemeriksaan dokumentasi

Tanggal: **1 Oktober 2026**. Acuan: branch **main**, commit
`cfa10b8fe1e5a03ac2992a84662d9dcdfafc2cdb`.
Generator: Archify **3.0.1**, profil **showcase**. Browser: Microsoft Edge
(Chromium), melalui `ARCHIFY_CHROME`.

## Artefak terbaru

| Diagram | Jenis | Validasi | Browser | Review visual |
| --- | --- | --- | --- | --- |
| [Arsitektur sistem](system-architecture.html) | `architecture` | 9/9, 0 error, 0 warning | passed | passed |
| [Proses dokumen](document-processing.html) | `workflow` v2 | 9/9, 0 error, 0 warning | passed | passed |
| [Penggunaan Streamlit](streamlit-user-journey.html) | `workflow` v2 | 9/9, 0 error, 0 warning | passed | passed |

`finalize` berhasil dengan exit code 0 untuk ketiganya. Pemeriksaan meliputi
validasi sumber/skema, delivery, pemeriksaan artefak dengan provenance, dan
pemeriksaan browser pada byte HTML yang diserahkan.

### Receipt yang menjadi acuan

| Diagram | Finalisasi terbaru | Bukti browser | Screenshot dan contact sheet |
| --- | --- | --- | --- |
| Arsitektur | [ringkasan](evidence/system-route-2/system-architecture.finalize-summary.json), [lengkap](evidence/system-route-2/system-architecture.finalize.json) | [browser](evidence/system-route-2/system-architecture.browser-check.json) | [visual receipt](evidence/system-route-2/system-architecture.visual-check.json), [contact sheet](evidence/system-route-2/system-architecture.visual-check.html) |
| Proses | [ringkasan](document-processing.finalize-summary.json), [lengkap](document-processing.finalize.json) | [browser](document-processing.browser-check.json) | [visual receipt](evidence/document-processing-visual/document-processing.visual-check.json), [contact sheet](evidence/document-processing-visual/document-processing.visual-check.html) |
| Streamlit | [ringkasan](streamlit-user-journey.finalize-summary.json), [lengkap](streamlit-user-journey.finalize.json) | [browser](streamlit-user-journey.browser-check.json) | [visual receipt](evidence/streamlit-user-journey-visual/streamlit-user-journey.visual-check.json), [contact sheet](evidence/streamlit-user-journey-visual/streamlit-user-journey.visual-check.html) |

Folder `evidence` hanya menyimpan bukti pemeriksaan untuk artefak terbaru.
Sebanyak 14 file dari empat direktori percobaan lama telah dihapus. Receipt
delivery dan finalisasi yang masih digunakan dipertahankan agar provenance
dan hasil pemeriksaan dapat ditelusuri. Tabel di atas menunjuk bukti yang
sesuai hash saat ini.

## Pemeriksaan browser dan visual

- Browser otomatis memeriksa 1440×900, 1600×1000, 1920×1080, dan 2048×1320.
  Containment, keterbacaan, kontrol viewer, dan keadaan tema lolos.
- Screenshot terang dan gelap pada 1440×900 serta 2048×1320 dihasilkan dan
  **seluruh 12 screenshot artefak terbaru ditinjau secara visual**.
- Node, panah, serta label dapat dibaca; tidak ditemukan label bertumpuk atau
  persilangan panah pada komposisi terakhir. Teks pendukung pada diagram proses
  cukup padat di 1440×900; gunakan zoom viewer atau buka README untuk penjelasan
  panjang.
- Cabang native dan pengulangan halaman memakai koridor luar agar terpisah dari
  jalur utama. Archify masih memberikan saran mengenai panjang/belokan beberapa
  rute; saran ini bukan error atau warning validasi. Rute diperiksa melalui
  screenshot, dan maknanya tetap dapat diikuti.
- Review visual merupakan penilaian terpisah. Field `visualReview` pada receipt
  `visual-check` tetap `pending` karena tool hanya mengambil screenshot; status
  `passed` pada dokumen ini dicatat setelah inspeksi gambar oleh agen.
- Arsitektur menjalani satu perbaikan rute setelah inspeksi screenshot
  (`correction_rounds: 1`). Proses dan Streamlit tidak berubah setelah inspeksi
  screenshot (`correction_rounds: 0`); perbaikan compiler dilakukan sebelumnya.

## Ketepatan isi terhadap main

| Skenario / klaim | Hasil pencocokan sumber |
| --- | --- |
| Konfigurasi model | Satu VLM utama. OCR opsional; backend normal PaddleOCR-VL 1.6 dengan layout lokal dan recognizer server. |
| Format dan profil | Format unggahan sesuai Streamlit. Enam profil sistem dibedakan dari empat profil manual di sidebar. |
| Jalur normal | Persiapan format, inspeksi/orientasi, OCR, pemilihan draft, spesialis, pemeriksaan, dan hasil sesuai pipeline. |
| OCR tidak tersedia / kurang dipercaya | VLM membaca gambar sebagai fallback; visual rescue dapat meminta pembacaan independen. |
| Retry dan halaman kosong | Retry mengikuti kondisi trust main, termasuk hasil error. Halaman kosong tidak melewati pembacaan isi atau judge. |
| Tabel OCR panjang | Pengecualian judge untuk sedikitnya 20 baris tanpa Mermaid dicatat, termasuk ketika thorough aktif. |
| Excel native | Native dan visual dipisahkan; sumber angka/formula/grafik berasal dari workbook. Penyimpanan native dan komposisi sheet dijelaskan. |
| PDF multi-halaman | Ingest tabel dan checkpoint per halaman mendahului stitching; konteks dilanjutkan antarhalaman. |
| Diagram | Hanya keluarga flowchart menjadi Mermaid; visual lain berupa deskripsi. Spesialis dipanggil sesuai indikator. |
| Streamlit | Unggah/batch, prioritas, jeda/lanjut, batal/ulang, tab hasil, ekspor, koreksi, dan histori dicocokkan dengan UI. |
| Integrasi tambahan | Deep Agent, MCP lokal, dan MCP agent dibedakan dari worker default Streamlit. Koreksi tidak langsung menjalankan optimasi prompt. |

Ini merupakan pemeriksaan statis dan pemeriksaan browser dokumentasi, bukan uji
ekstraksi dokumen. Tidak ada endpoint VLM/OCR yang dipanggil dan tidak ada kode
aplikasi atau konfigurasi runtime yang diubah.

## Identitas artefak

| Diagram | SHA-256 spesifikasi JSON | SHA-256 HTML |
| --- | --- | --- |
| Arsitektur | `d29f7d5758253c882f5be30ed16781c3c1ae2761c8bdea522e28f555d4475dcf` | `a8123a42fac0768ac4f8f814da8543fdd9ab835c4c58c2ad135f4809da1d0967` |
| Proses | `d3059a1d65850340fd1aafc61700fda0b57caab5cc5c1cd9314434995f1db5c7` | `8b759d62c230c5c7804006d1cf364051fd95de44649b5e80fb4771c7ea6a4217` |
| Streamlit | `c9c311f9b381df5528f57b97ba63015f62a19938e43b7e430a91f0c7c65d8804` | `c436af51d51addd6a0070435a846fb07663f6293a9d7f1d37c3a25081a71fcf0` |

Saat spesifikasi berubah, finalisasi dan pemeriksaan visual perlu diulang agar
receipt, screenshot, dan hash kembali mengacu pada byte yang sama.

Aturan `.gitattributes` lokal mempertahankan byte HTML dan JSON saat checkout,
sehingga konversi akhir baris Git tidak mengubah hash provenance. Spasi akhir
bawaan template HTML generator dipertahankan bersama artefak yang divalidasi.
