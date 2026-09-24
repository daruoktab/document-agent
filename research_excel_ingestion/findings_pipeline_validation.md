# Temuan: Arsitektur Ingestion dan Validasi Excel Heterogen

Tanggal riset: 24 September 2026
Ruang lingkup: routing native-vs-visual, isolasi per sheet, inferensi tipe, provenance, batas sumber daya, formula, named table/range, chart/image, dan validasi.
Kebijakan sumber: hanya dokumentasi resmi proyek/platform dan makalah penelitian primer.

## Ringkasan keputusan

Pipeline produksi sebaiknya tidak memilih antara parser native **atau** VLM secara global untuk satu workbook. Pilihan dibuat per objek/region:

1. **Native-first** untuk nilai sel, formula, tipe, tabel resmi Excel, named range statis, dan hubungan antar-sheet.
2. **Native + visual** untuk dashboard, sheet dengan tata letak campuran, chart, shape/image, merged-cell kompleks, atau region yang batasnya ambigu.
3. **Visual-only sebagai fallback**, bukan sumber kebenaran numerik, ketika objek memang hanya tersedia sebagai raster atau parser tidak dapat memaknai representasi objeknya.

Alasannya: satu sheet dapat memiliki beberapa tabel yang berdekatan, baris/kolom kosong di dalam tabel, tipe heterogen, dan artefak presentasi. TableSense menunjukkan region-growth sederhana rapuh pada kondisi tersebut; fitur yang relevan tidak hanya isi sel, tetapi juga tipe, format, style, merge, dan keberadaan formula. Makalah Mondrian juga memodelkan spreadsheet sebagai kumpulan region tabular dan metadata, lalu memakai segmentasi, clustering, dan hubungan spasial untuk membedakan region yang berdekatan. [TableSense (AAAI 2019)](https://www.microsoft.com/en-us/research/wp-content/uploads/2019/01/TableSense_AAAI19.pdf), [Mondrian / PVLDB 2021](https://arxiv.org/abs/2109.06630)

## Arsitektur yang disarankan

### 1. Intake dan artefak sumber yang tidak berubah

Simpan file unggahan sebagai artefak immutable sebelum parsing. Buat `workbook_id`, hash SHA-256, nama asli, ukuran, format, waktu ingest, dan versi parser. Semua keluaran selanjutnya menunjuk ke identitas ini. Jangan mengubah file sumber untuk memaksa recalculation atau memperbaiki format.

Terapkan batas eksplisit sebelum parsing: ukuran file terkompresi, total ukuran hasil dekompresi, jumlah entry ZIP, jumlah sheet, estimasi sel, jumlah drawing, waktu proses, dan memori. Untuk workbook besar, gunakan pembacaan streaming/lazy pada tahap inventarisasi. Dokumentasi openpyxl menyatakan mode read-only memakai lazy loading dan harus ditutup eksplisit; metadata dimensi dari pembuat file juga dapat salah sehingga `calculate_dimension()` perlu diperiksa dan, bila tidak masuk akal, dimensinya dihitung ulang. Apache POI juga membedakan event model yang hemat memori dari user model yang lebih lengkap namun lebih berat. [openpyxl optimized modes](https://openpyxl.readthedocs.io/en/stable/optimized.html), [Apache POI spreadsheet overview](https://poi.apache.org/components/spreadsheet/)

**Kriteria uji:** file yang melebihi salah satu batas berhenti dengan status `REJECTED_LIMIT`, bukan exception generik; file di bawah batas tetap bisa diproses.

### 2. Inventory pass sebelum ekstraksi

Buat manifest workbook tanpa langsung membentuk tabel akhir. Untuk setiap sheet catat:

- indeks, nama, visibility, declared dimension dan observed non-empty bounds;
- merged ranges, formula/error cells, hyperlinks, comments, format angka, hidden rows/columns;
- Excel structured tables (`ws.tables`) beserta nama dan `ref`;
- defined names global/lokal beserta destination yang dapat di-resolve;
- chartsheet, chart, image/drawing, pivot/relationship yang terdeteksi;
- indikator risiko: external link, formula tanpa cached value, dynamic named range, dimensi tidak wajar, dan unsupported object.

Structured table dan named range harus diperiksa sebelum pencarian region heuristik. openpyxl menyediakan enumerasi tabel dan rentangnya melalui `ws.tables`; defined names dapat menunjuk constant, formula, satu range, beberapa range, bahkan lintas sheet. Namun dynamic defined name tidak selalu dapat di-resolve oleh openpyxl dan akan dilewati dengan warning, sehingga manifest harus mempertahankan ekspresi aslinya dan menandainya `UNRESOLVED_DYNAMIC_NAME`, bukan menghilangkannya. [openpyxl worksheet tables](https://openpyxl.readthedocs.io/en/stable/worksheet_tables.html), [openpyxl defined names](https://openpyxl.readthedocs.io/en/stable/defined_names.html)

**Kriteria uji:** fixture berisi satu table resmi, satu named range global, satu lokal, satu multi-range, dan satu dynamic name; empat objek resolvable masuk manifest, sedangkan dynamic name tetap tercatat dengan warning.

### 3. Eksekusi dan isolasi kegagalan per sheet

Jadikan tiap sheet sebuah unit kerja dengan state sendiri: `DISCOVERED -> INVENTORIED -> EXTRACTED -> VALIDATED`, atau `FAILED`/`PARTIAL`. Kesalahan saat membaca atau merender satu sheet tidak boleh membatalkan hasil sheet lain. Setelah semua unit selesai, status workbook diturunkan dari agregat:

- `SUCCESS`: seluruh sheet relevan berhasil;
- `PARTIAL_SUCCESS`: minimal satu berhasil dan minimal satu gagal/parsial;
- `FAILED`: tidak ada sheet relevan yang berhasil;
- `REJECTED_LIMIT`: gagal sebelum tahap sheet karena batas keamanan.

Checkpoint hasil inventory dan region secara atomik agar retry hanya mengulang sheet/region gagal. Setiap kegagalan menyimpan `stage`, exception class yang disanitasi, kode penyebab, dan apakah retry aman.

**Kriteria uji:** paksa parser atau renderer gagal pada sheet kedua dari tiga sheet; sheet pertama dan ketiga tetap tersimpan, workbook berstatus `PARTIAL_SUCCESS`, dan retry hanya menjadwalkan sheet kedua.

### 4. Deteksi region berlapis, bukan sekadar connected non-empty cells

Urutan proposal region:

1. Excel structured table sebagai boundary berkepercayaan tinggi.
2. Named range statis yang berbentuk rectangle atau kumpulan rectangle.
3. Region kandidat dari kombinasi isi, tipe, number format, formula, style/border/fill/bold, merge, alignment, dan kedekatan spasial.
4. Region visual dari chart, image, shape/drawing, serta dashboard layout.
5. Reconciliation untuk menggabungkan proposal tumpang tindih, memisahkan tabel berdekatan, dan mempertahankan title/footnote sebagai region metadata terpisah.

Jangan menjadikan satu baris/kolom kosong sebagai hard boundary. TableSense mendokumentasikan blank internal rows/columns, tipe heterogen di dalam tabel, missing data, dan tabel yang disusun sangat berdekatan. Cell feature set mereka mencakup isi, pola angka/tanggal/waktu, fill/font/border, merge, dan formula. Mondrian menunjukkan connected component perlu dipotong lalu dikelompokkan kembali agar region berdekatan dapat dibedakan. [TableSense](https://www.microsoft.com/en-us/research/wp-content/uploads/2019/01/TableSense_AAAI19.pdf), [Mondrian](https://arxiv.org/abs/2109.06630)

Simpan `boundary_confidence` dan alasan proposal (`excel_table`, `named_range`, `layout_inferred`, `visual_object`). Bila beberapa proposal konflik dan confidence rendah, render crop dengan margin kecil untuk pemeriksaan VLM, tetapi angka tetap dibaca dari sel native.

**Kriteria uji:** fixture mencakup tabel dengan blank row internal, dua tabel berjarak satu kolom, title dan footnote, merge header, serta tabel bertumpang tindih dengan named range. Assert boundary setiap region dan gunakan toleransi berbasis Error-of-Boundary: exact untuk structured table/named range, maksimum dua sel hanya untuk region inferred yang ditandai perlu tinjauan.

### 5. Router native-vs-visual per region

Gunakan kontrak routing berikut:

| Kondisi | Jalur | Sumber kebenaran |
|---|---|---|
| Structured table, named range, tabel detail biasa | native | nilai dan metadata sel |
| Dashboard atau summary dengan layout semantik | native + render sheet/crop | native untuk fakta; visual untuk hubungan dan label |
| Chart | native series/range + render | sel sumber untuk angka; render untuk judul/legend/appearance |
| Embedded image/logo/screenshot | visual | hasil deskripsi visual dengan confidence |
| Parser objek tidak mendukung pivot/chart tertentu | visual + warning | deskripsi visual, tanpa mengklaim angka sebagai terverifikasi |
| Batas region ambigu | native candidate + visual arbitration | nilai native dalam boundary terpilih |

Chart merupakan objek yang series-nya mengacu ke cell ranges; openpyxl mendokumentasikan chart sebagai kumpulan series yang tersusun dari reference ke range. Namun dukungan library tidak identik untuk semua objek: Apache POI menyatakan dukungan chart dan pivot terbatas. Karena itu, inventaris native dan render harus saling melengkapi, serta unsupported object wajib tampak sebagai warning. [openpyxl charts](https://openpyxl.readthedocs.io/en/stable/charts/introduction.html), [openpyxl images](https://openpyxl.readthedocs.io/en/stable/images.html), [Apache POI limitations](https://poi.apache.org/components/spreadsheet/limitations.html)

**Kriteria uji:** workbook dengan chart yang merujuk data detail menghasilkan provenance series dan deskripsi visual; workbook dengan image menghasilkan deskripsi tanpa membuat tabel angka fiktif; unsupported pivot menghasilkan warning tetapi sheet lain tetap selesai.

### 6. Formula: simpan ekspresi dan cached value secara terpisah

Lakukan dua pembacaan logis untuk `.xlsx/.xlsm`:

- `data_only=False` untuk formula asli;
- `data_only=True` untuk nilai cache terakhir.

Open XML menyimpan formula pada elemen `<f>` dan cached result pada `<v>`. openpyxl juga menegaskan bahwa `data_only=True` hanya mengembalikan nilai yang terakhir kali disimpan Excel. Cached value bukan bukti bahwa kalkulasi masih mutakhir. Formula dapat mengacu ke sheet lain, workbook eksternal, named range, dan user-defined function; evaluator pihak ketiga juga tidak selalu mendukung semuanya. [Microsoft Open XML formulas](https://learn.microsoft.com/en-us/office/open-xml/spreadsheet/working-with-formulas), [openpyxl load_workbook](https://openpyxl.readthedocs.io/en/stable/api/openpyxl.reader.excel.html), [Apache POI formula evaluation](https://poi.apache.org/components/spreadsheet/eval.html)

Untuk setiap formula simpan `formula_text`, `cached_value`, `cached_type`, `error_code`, external dependencies, dan status:

- `CACHED_PRESENT_UNVERIFIED`;
- `MISSING_CACHE`;
- `FORMULA_ERROR` untuk `#VALUE!`, `#REF!`, `#DIV/0!`, `#N/A`, dan error lain;
- `EXTERNAL_DEPENDENCY_MISSING`;
- `UNSUPPORTED_FUNCTION` bila dapat diidentifikasi.

Jangan otomatis mengubah error menjadi kosong atau nol. Jangan mengevaluasi ulang lalu menimpa sumber. Jika recalculation engine ditambahkan kemudian, hasilnya disimpan sebagai nilai turunan dengan nama engine dan versi.

**Kriteria uji:** formula normal dengan cache, formula tanpa cache, error literal, referensi silang-sheet, external workbook yang hilang, dan fungsi yang tidak didukung semuanya menghasilkan status berbeda dan tidak menggagalkan sel nonformula.

### 7. Inferensi tipe konservatif per region dan per kolom

Inferensi dilakukan setelah boundary region diputuskan, bukan dari seluruh sheet. Pertahankan dua lapisan:

- `raw_value` serta metadata native (`cell.data_type`, `number_format`, formula/error);
- `normalized_value` dan `logical_type` hasil inferensi.

Aturan aman:

1. Nilai kosong diabaikan saat memilih tipe tetapi tetap dipertahankan sebagai null.
2. Boolean dipisahkan dari integer.
3. Tanggal/waktu ditentukan dari kombinasi nilai native dan number format, bukan pola teks saja.
4. Kolom numeric yang memiliki satu nilai tekstual non-null tidak boleh dipaksa `float`; pilih `TEXT` atau tipe union/variant dan catat konflik.
5. Identifier dengan leading zero, kode panjang, nomor telepon, NIK, atau kode akun dipertahankan sebagai teks bila style/format atau distribusi mendukungnya.
6. Normalisasi locale (`decimal`, `thousands`, true/false, NA markers) merupakan konfigurasi eksplisit, bukan asumsi global.
7. Schema hasil memiliki `inference_confidence`, jumlah contoh per tipe, dan daftar cell konflik.

pandas mendukung pembacaan multi-sheet dan inferensi tipe, tetapi juga menyediakan `dtype`, `converters`, `decimal`, `thousands`, `true_values`, `false_values`, dan `na_values` karena interpretasi yang benar memang bergantung konteks. Dokumentasinya juga menunjukkan `sheet_name=None` mengembalikan seluruh sheet sebagai dictionary. [pandas.read_excel](https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html)

**Kriteria uji:** satu suite fixture memuat integer, desimal, persen, mata uang, serial date, ISO date text, boolean, leading-zero ID, blank, error, angka berformat locale, dan satu teks di kolom angka. Assert tidak ada exception konversi, raw value tetap tersedia, serta konflik ditandai.

### 8. Provenance sampai level cell/region

Setiap tabel/region keluaran minimal menyimpan:

- `workbook_id`, source hash, parser/render/model version;
- sheet name/index/visibility;
- `region_id`, boundary A1 dan koordinat numerik;
- source kind dan source object name (`table`, `defined_name`, `inferred`, `chart`, `image`);
- extraction path (`native`, `visual`, `hybrid`) dan confidence;
- untuk cell: coordinate, raw value, normalized value, native data type, logical type, number format, formula, cached value, error, merged anchor;
- warning dan validation result.

Lineage turunan harus menunjuk cell sumber. Contoh: angka pada deskripsi chart menunjuk chart series lalu ke source ranges; summary hasil VLM menunjuk region crop dan daftar range native yang dipakai untuk verifikasi. Ini membuat koreksi dapat dilakukan tanpa mengulang seluruh workbook.

**Kriteria uji:** pilih acak nilai hasil normalisasi dan telusuri balik sampai file hash, sheet, region, dan cell; nilai dari chart harus dapat ditelusuri ke series/range atau secara eksplisit berlabel `VISUAL_ONLY_UNVERIFIED`.

### 9. Validasi berlapis dan status yang dapat ditindaklanjuti

Validasi tidak cukup berupa keberhasilan insert database. Terapkan lapisan berikut:

1. **Package validation:** workbook dapat dibuka dan inventory selesai dalam batas.
2. **Structural validation:** range valid, header unik setelah canonicalization, tidak ada overlap tanpa alasan, merged cells konsisten, table/name destinations resolvable.
3. **Content validation:** row/column count, null/error count, distribusi tipe, duplicate header, leading/trailing empty area, dan checksum/canonical digest per region.
4. **Formula validation:** jumlah formula, missing cache, error, external dependency, unsupported function.
5. **Cross-representation validation:** nilai native di Markdown/DB sama setelah canonical normalization; VLM tidak menambah angka yang tidak ditemukan pada native evidence.
6. **Cross-region validation:** bila summary menyatakan total/subtotal, bandingkan dengan agregasi detail ketika relasi kolom dapat dibuktikan; mismatch menjadi warning, bukan koreksi otomatis.
7. **Object validation:** chart series ranges dan named ranges resolve; image/chart yang gagal dirender dicatat.

Hasil validasi harus berupa kode terstruktur (`BOUNDARY_AMBIGUOUS`, `TYPE_CONFLICT`, `FORMULA_CACHE_MISSING`, `EXTERNAL_LINK`, `UNRESOLVED_NAME`, `VISUAL_NATIVE_MISMATCH`) beserta severity dan evidence, bukan teks bebas saja.

**Kriteria uji:** setelah hasil persist dibuat, ubah satu nilai DB, satu boundary, satu formula status, dan satu angka pada ringkasan visual; masing-masing harus memicu kode discrepancy yang spesifik tanpa crash.

## Urutan implementasi yang paling rendah risiko

1. Tambahkan manifest dan provenance tanpa mengubah output pengguna.
2. Prioritaskan structured tables dan named ranges sebelum detector saat ini.
3. Tambahkan status per-sheet dan `PARTIAL_SUCCESS` serta retry granular.
4. Pisahkan formula/cached value dan perluas warning formula/error.
5. Perkuat inferensi tipe konservatif beserta conflict reporting.
6. Tambahkan router hybrid untuk dashboard/chart/image.
7. Tambahkan reconciliation dan confidence untuk region ambigu.
8. Aktifkan validator silang native/DB/Markdown/visual.

Urutan ini memungkinkan peningkatan bertahap tanpa mengganti hasil ekstraksi sekaligus. Semua perubahan dapat dijaga dengan fixture regresi, bukan benchmark performa.

## Sumber primer

- openpyxl, Optimised Modes: https://openpyxl.readthedocs.io/en/stable/optimized.html
- openpyxl, Workbook Reader: https://openpyxl.readthedocs.io/en/stable/api/openpyxl.reader.excel.html
- openpyxl, Worksheet Tables: https://openpyxl.readthedocs.io/en/stable/worksheet_tables.html
- openpyxl, Defined Names: https://openpyxl.readthedocs.io/en/stable/defined_names.html
- openpyxl, Charts: https://openpyxl.readthedocs.io/en/stable/charts/introduction.html
- openpyxl, Images: https://openpyxl.readthedocs.io/en/stable/images.html
- pandas, `read_excel`: https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html
- Microsoft Learn, Working with formulas in SpreadsheetML: https://learn.microsoft.com/en-us/office/open-xml/spreadsheet/working-with-formulas
- Apache POI, Spreadsheet APIs: https://poi.apache.org/components/spreadsheet/
- Apache POI, Formula Evaluation: https://poi.apache.org/components/spreadsheet/eval.html
- Apache POI, HSSF/XSSF Limitations: https://poi.apache.org/components/spreadsheet/limitations.html
- Dong et al., TableSense: https://www.microsoft.com/en-us/research/wp-content/uploads/2019/01/TableSense_AAAI19.pdf
- Vitagliano, Jiang, Naumann, Detecting Layout Templates in Complex Multiregion Files: https://arxiv.org/abs/2109.06630
