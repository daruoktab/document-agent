# Pembelajaran dari koreksi dokumen

DSPy 3.3.1 dan GEPA 0.1.4 memperbaiki **instruksi ekstraksi**, bukan bobot model.
Koreksi dikumpulkan selama penggunaan aplikasi; optimasi serta penerapan versi tetap dilakukan admin dari terminal lokal.
Fitur ini ditujukan untuk tim internal. Streamlit tidak menambahkan akun atau autentikasi admin baru.

## Penggunaan oleh tim

1. Buka hasil dokumen, pilih tab **Cocokkan dengan dokumen asli**, lalu pilih halaman.
2. Buka **Koreksi hasil halaman ini**, perbaiki teks dan tabel Markdown berdasarkan gambar asli.
3. Centang **Siap digunakan untuk evaluasi** jika isi sudah diperiksa, lalu simpan.
4. Untuk menarik contoh dari dataset berikutnya, hapus centang dan simpan kembali.

Setiap simpan menjadi revisi baru; hanya revisi terbaru yang disetujui ikut evaluasi.
Hasil awal tidak ditimpa. Koreksi dan salinan gambar tersimpan di `data/learning/learning.sqlite`,
terpisah dari hasil ZIP dan pembersihan `output/`. Cadangkan direktori tersebut untuk mempertahankan dataset dan versi prompt.
Lokasi dapat diubah dengan `LEARNING_DATA_DIR` (gunakan lokasi yang sama untuk aplikasi dan CLI).
Dokumen lama yang belum memiliki metadata prompt diberi label `legacy-unknown`; evaluasinya memakai aturan `plain` tanpa konteks halaman sebelumnya.

## Perintah admin

Aktifkan lingkungan untuk seluruh perintah berikut:

```powershell
.venv\Scripts\Activate.ps1
python -m app.learning_cli status
python -m app.learning_cli probe
python -m app.learning_cli optimize --max-metric-calls 50
python -m app.learning_cli activate ID_PERCOBAAN
python -m app.learning_cli rollback
```

`probe` mengirim satu gambar buatan bertuliskan BUKU 100 untuk memeriksa kompatibilitas endpoint.
`optimize` memakai pengaturan `VLM_*` yang sama dengan aplikasi, termasuk URL, model, kunci API, dan pengaturan thinking.
Data gambar dan koreksi terpilih dikirim ke endpoint tersebut saat optimasi, termasuk untuk refleksi multimodal.
Proses berjalan di terminal; tutup terminal atau interupsi untuk menghentikannya. Percobaan yang terputus paksa dapat tetap bertanda `running`, tetapi tidak bisa diterapkan.

Minimum awal adalah enam kelompok dokumen berbeda. Pembagian awal sekitar 60/20/20 untuk optimasi/validasi/pengujian
(enam dokumen menghasilkan 4/1/1). Ini batas kelayakan teknis percobaan, bukan ukuran dataset yang menjamin kualitas.
Kelompok pengujian harus mengandung tabel. Status saat ini menampilkan jumlah contoh dan seluruh laporan percobaan.

Identitas dokumen menggunakan hash isi berkas; halaman identik juga mengikat dokumen salinan ke kelompok yang sama.
Pembagian disimpan permanen lintas percobaan. Dokumen baru ditambahkan secara deterministik tanpa memindahkan contoh lama.
Jika salinan baru menghubungkan dua kelompok berbeda, optimasi dihentikan agar tim meninjau koreksinya.
Hash halaman membandingkan byte gambar; dokumen yang dipindai ulang atau dikompresi berbeda perlu ditinjau sebagai potensi duplikat.

Anggaran 50 adalah parameter `max_metric_calls` GEPA. GEPA menghentikan pencarian pada batas iterasi sehingga jumlah evaluasi
dapat sedikit melewati anggaran. Refleksi model serta pengujian baseline dan kandidat juga memerlukan panggilan API;
angka ini bukan batas biaya mata uang. Tidak ada optimasi otomatis setiap unggahan.

## Evaluasi dan penerapan

Skor teks membandingkan isi di luar tabel setelah normalisasi spasi. Skor tabel membandingkan baris, urutan, isi sel, dan jumlah kolom.
Skor gabungan adalah rata-rata skor teks dan tabel. Umpan balik menyertakan perbedaan angka dan koreksi acuan untuk refleksi GEPA.
Format Markdown yang berbeda secara substantif dapat menurunkan skor walaupun tampak mirip; gunakan koreksi yang konsisten.
Penilaian ini berfokus pada tahap ekstraksi per halaman, belum mengukur kualitas akhir diagram, penyatuan halaman, atau database.

Kandidat diuji terhadap prompt aktif pada kelompok pengujian yang tidak diberikan ke GEPA.
Pada percobaan pertama baseline adalah pemanggilan ekstraktor lama melalui LangChain.
Laporan menyimpan versi baseline, konfigurasi model tanpa kredensial, ID revisi contoh, pembagian dataset,
instruksi kandidat, prediksi pengujian, skor teks/tabel/gabungan, dan kelayakan penerapan.
Salinan koreksi serta gambar pada revisi tersebut tetap tersedia walaupun sumber di `output/` terhapus.

`activate` hanya menerima kandidat selesai yang meningkatkan skor gabungan tanpa menurunkan skor teks atau tabel.
Jika koreksi, prompt aktif, atau konfigurasi model berubah setelah evaluasi, penerapan ditolak dan diperlukan percobaan baru.
Penerapan berlangsung atomik. Ekstraktor baru memakai versi tersebut; ekstraktor yang sudah berjalan tetap memakai versi sebelumnya.
Untuk server yang mempertahankan instance ekstraktor, buat ulang instance atau mulai ulang server setelah penerapan/pemulihan.
Perubahan konfigurasi model membuat ekstraktor kembali ke prompt bawaan dengan peringatan log sampai versi yang sesuai tersedia.
`rollback` mengembalikan rilis sebelumnya; pemulihan dari rilis pertama kembali ke baseline.
Gangguan API saat optimasi tidak mengubah rilis aktif. Admin tetap perlu memeriksa laporan sebelum menerapkan kandidat.

## Pemeriksaan pengembangan

```powershell
.venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING='utf-8'
python tests\run_tests.py
ruff check --fix; ty check
uv pip check
uv lock --check
```

Tes menggunakan dokumen sintetis dan model tiruan melalui adapter DSPy serta mesin GEPA asli.
Tes ini memverifikasi mekanisme; peningkatan pada dokumen pengguna harus dibuktikan dengan koreksi nyata.
