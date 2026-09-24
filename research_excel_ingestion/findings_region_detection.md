# Temuan: Deteksi Tabel dan Region Spreadsheet Heterogen

## Ringkasan eksekutif

Connected-components atas sel yang tidak kosong cocok sebagai **pembentuk kandidat awal**, tetapi tidak cukup sebagai batas tabel final. Pendekatan yang lebih tahan terhadap workbook acak adalah pipeline bertingkat:

1. gunakan struktur eksplisit Excel sebagai batas berpresisi tinggi;
2. bentuk elemen atomik dari nilai, formula, merge, dan formatting;
3. pecah komponen yang terlalu besar atau berbentuk tidak beraturan;
4. gabungkan kembali elemen menggunakan skor jarak, alignment, tipe data, style, formula, dan kecocokan header;
5. klasifikasikan bagian region menjadi judul, header atas/kiri, body, total, dan catatan;
6. simpan confidence serta alasan keputusan dan gunakan render/VLM hanya untuk kandidat ambigu.

Rekomendasi ini sesuai untuk implementasi deterministik dengan Python/openpyxl dan tidak mengharuskan model machine learning baru.

## Bukti dari sumber primer

### 1. Sel spreadsheet memiliki sinyal lebih kaya daripada sekadar isi

TableSense menggunakan 20 fitur dari empat kelompok: string nilai, format data, format sel, dan formula. Fitur yang digunakan mencakup status non-empty, panjang string, rasio digit/huruf, pola angka/tanggal/waktu, fill, warna font, bold, empat sisi border, hubungan merge horizontal/vertikal, dan keberadaan formula. Paper tersebut juga menyatakan bahwa kohesi di dalam tabel dan kontras pada batas tabel dapat dilihat dari konsistensi format data, formula, serta statistik sel. Sumber: [TableSense, AAAI 2019](https://www.microsoft.com/en-us/research/uploads/prod/2019/01/TableSense_AAAI19.pdf).

Implikasi: occupancy mask tidak boleh hanya berasal dari `cell.value is not None`. Sel formula, sel kosong yang diberi border/fill sebagai bagian grid, dan merge harus menjadi evidence tersendiri. Style tidak boleh dianggap bukti tunggal karena workbook sering memiliki formatting berlebih.

### 2. Connected-components gagal pada dua arah yang berlawanan

Mondrian menjelaskan dua kegagalan penting:

- Satu tabel dapat terdiri dari beberapa komponen terpisah akibat missing values, baris kosong, atau kolom kosong.
- Dua region independen yang berdempetan dapat menjadi satu connected component dengan bentuk tidak beraturan.

Mondrian mengatasi masalah kedua dengan memotong komponen pada sisi konkaf menjadi elemen yang lebih kecil, lalu mengatasi masalah pertama dengan clustering berbasis kepadatan. Jarak custom-nya menggabungkan jarak sel terdekat, perbedaan ukuran elemen, dan besar alignment horizontal/vertikal. Alignment dipakai untuk mengompensasi ruang kosong di dalam region. Sumber: [Detecting Layout Templates in Complex Multiregion Files, PVLDB 2021](https://arxiv.org/pdf/2109.06630).

Eksperimen Mondrian juga menunjukkan bahwa radius tetap terlalu kecil berubah menjadi connected-components biasa, sedangkan radius lebih besar membantu tabel yang terpisah tetapi dapat menyatukan region non-tabel. Radius dinamis per file lebih baik daripada satu ambang global. Ini mendukung threshold adaptif berdasarkan karakter sheet, bukan konstanta tunggal.

### 3. Header harus dipahami melalui isi, style, posisi, dan relasi

Tasi mengidentifikasi struktur header memakai fitur isi, style, dan lokasi spasial. Fitur yang relevan bagi implementasi rule-based meliputi kata agregasi seperti total/sum/max, formula agregasi, ukuran merged cell, indentasi, posisi relatif dalam tabel, jumlah header pada baris sekitar, border tebal, blank row/column, dan hubungan referensi formula. Tasi membagi header menjadi value name, index, index name, aggregation, dan other. Sumber: [Semantic Table Structure Identification in Spreadsheets, ISSTA 2021](https://www.microsoft.com/en-us/research/wp-content/uploads/2020/08/Semantic-Table-Structure-Identification-in-Spreadsheets.pdf).

Paper tersebut menemukan bahwa hampir semua tabel yang diamati memiliki header di bagian atas dan/atau kiri, tetapi tetap mencatat layout header kanan/bawah sebagai kasus langka. Ini layak dijadikan prior, bukan aturan mutlak.

### 4. Deteksi batas dan struktur internal sebaiknya saling mengoreksi

Riset multi-task Microsoft memformulasikan ekstraksi struktur secara coarse-to-fine: deteksi tabel, pengenalan top header/left header/value region, lalu klasifikasi tipe sel. Hasil tahap struktur juga dipakai kembali sebagai channel untuk tahap berikutnya, dan paper menekankan risiko error propagation bila tahap dijalankan secara buta. Sumber: [Semantic Structure Extraction for Spreadsheet Tables, NeurIPS DI 2019](https://www.microsoft.com/en-us/research/wp-content/uploads/2019/12/TableStructure_DI19.pdf).

Implikasi: kandidat batas yang menghasilkan body tanpa pola kolom, header yang berada di luar batas, atau formula family yang terpotong perlu diperluas/diperkecil lalu dinilai ulang.

### 5. Detail representasi openpyxl yang memengaruhi deteksi

- `ws.tables` menyediakan Excel Table dan `Table.ref` sebagai batas native yang eksplisit; ini harus diprioritaskan daripada inferensi. [Dokumentasi resmi openpyxl: Worksheet Tables](https://openpyxl.readthedocs.io/en/3.1/worksheet_tables.html).
- Pada merged range, hanya sel kiri-atas yang menyimpan nilai dan style utama; sel lain menjadi `MergedCell` dan bernilai `None`. [Dokumentasi resmi openpyxl: merge/unmerge](https://openpyxl.readthedocs.io/en/3.1/editing_worksheets.html) dan [merged ranges](https://openpyxl.readthedocs.io/en/3.1/api/openpyxl.worksheet.merge.html).
- Style dapat diterapkan di tingkat sel, baris, atau kolom dan mempunyai perilaku berbeda; karena itu style perlu dibaca sebagai evidence, bukan occupancy absolut. [Dokumentasi resmi openpyxl: styles](https://openpyxl.readthedocs.io/en/3.1/styles.html).
- openpyxl tidak menghitung formula. Formula dan cached value harus diperlakukan sebagai dua sumber berbeda; cached value yang kosong tidak berarti sel formula kosong secara struktural. [Dokumentasi resmi openpyxl](https://openpyxl.readthedocs.io/en/2.6/usage.html).
- SpreadsheetML menyimpan sel secara sparse dan mendukung merged cells sebagai unit. Artinya bounding box worksheet dapat jauh lebih besar daripada data bermakna dan tidak aman diperlakukan sebagai matriks padat tanpa pembatas. [ECMA-376 / Office Open XML](https://ecma-international.org/publications-and-standards/standards/ecma-376/).

## Audit singkat terhadap logika saat ini

Pada `app/excel.py`, `_connected_components()` memakai konektivitas delapan arah terhadap `data_coordinates`, sementara `data_coordinates` hanya berisi sel dengan nilai. `_cell_has_visible_style()` sudah tersedia tetapi belum digunakan dalam pembentukan kandidat.

Konsekuensinya:

- dua tabel yang hanya bersentuhan diagonal dapat menyatu;
- satu tabel sparse dengan baris/kolom kosong dapat pecah;
- sel formula tetap terdeteksi karena nilai formula ada, tetapi cached value kosong dapat menyulitkan penilaian isi;
- blank cells yang membentuk grid melalui border/fill tidak berkontribusi;
- merged title yang lebar dapat memperluas atau menghubungkan region secara tidak semantik;
- dua tabel yang menempel tanpa separator cenderung menjadi satu region;
- pemisahan judul, multi-level header, body, total, dan footnote belum menjadi bagian eksplisit dari segmentasi batas.

## Rekomendasi algoritma untuk Python/openpyxl

### Tahap A — Inventaris sumber berpresisi tinggi

Untuk setiap worksheet:

1. Catat semua `ws.tables.values()` dan parse `table.ref`. Jadikan range ini **locked regions**; kandidat lain tidak boleh menggabungkan dua Excel Table berbeda.
2. Catat merged ranges sebagai objek `(anchor, bounds)`, bukan dengan menyalin nilai anchor ke seluruh sel.
3. Catat formula asli dari workbook `data_only=False` dan cached values dari workbook `data_only=True`.
4. Catat charts/images dan anchor-nya sebagai region visual terpisah.
5. Batasi scanning ke bounding box sparse dari instantiated cells, table refs, merges, dan drawing anchors. Hindari membuat matriks sampai `ws.max_row/ws.max_column` tanpa guard karena formatting sisa dapat memperbesar dimensi.

### Tahap B — Bentuk vektor fitur sel

Bangun record ringan hanya untuk sel yang memiliki salah satu evidence berikut:

- nilai literal;
- formula;
- bagian dari merged range;
- border/fill/bold/indent/number format yang berbeda dari default;
- bagian dari Excel Table;
- comment/hyperlink bila ingin mempertahankan metadata.

Fitur minimum yang disarankan:

```text
row, col, has_value, has_formula, value_type,
number_format_family, style_id, fill_key, border_mask,
bold, indent, merged_range_id, table_id,
formula_family, text_shape
```

`formula_family` sebaiknya berupa formula relatif/R1C1-normalized sehingga formula yang disalin ke baris atau kolom lain tetap dianggap satu pola. `text_shape` cukup berupa kategori seperti empty, integer, decimal, date, time, uppercase, titlecase, generic text, dan aggregation keyword.

### Tahap C — Elemen atomik, bukan langsung region final

1. Gunakan konektivitas empat arah untuk seed cells agar sentuhan diagonal tidak otomatis menyatukan tabel.
2. Buat komponen tambahan dari run horizontal/vertikal yang memiliki type/style/formula family serupa.
3. Untuk komponen dengan bounding box berlubang, berbentuk L/T, atau density rendah, potong pada separator kandidat:
   - concavity pada mask;
   - garis border tebal penuh/sebagian besar lebar;
   - perubahan mendadak style/type/formula family;
   - pengulangan pola header di tengah region;
   - blank gap yang diikuti schema baru.
4. Jangan jadikan merged title sebagai jembatan. Merge di atas beberapa kandidat boleh diasosiasikan sebagai title bersama, tetapi tidak otomatis menggabungkan body di bawahnya.

### Tahap D — Merge kandidat dengan skor multi-sinyal

Untuk dua elemen `a` dan `b`, hitung fitur pasangan:

```text
gap_rows / gap_cols
row_overlap_ratio / column_overlap_ratio
nearest_boundary_distance
value_type_similarity
style_similarity
formula_family_similarity
header_body_compatibility
separator_penalty
explicit_table_conflict
```

Skor awal yang praktis:

```text
merge_score =
    2.0 * max(row_overlap_ratio, column_overlap_ratio)
  + 1.5 * formula_family_similarity
  + 1.0 * value_type_similarity
  + 0.8 * style_similarity
  + 1.2 * header_body_compatibility
  - 0.8 * normalized_gap
  - 2.0 * separator_penalty
  - 10.0 * explicit_table_conflict
```

Angka tersebut adalah starting heuristic, bukan angka dari paper. Kalibrasikan dengan fixture internal. Ambang harus adaptif:

- gunakan median jarak antarelemen pada sheet;
- gunakan median tinggi baris/lebar kolom aktif;
- izinkan gap lebih besar hanya bila alignment, schema, atau formula continuity kuat;
- larang merge bila kandidat masing-masing memiliki header kuat atau berada pada dua `Table.ref` berbeda.

DBSCAN tidak wajib. Agglomerative merging dengan priority queue lebih mudah dijelaskan, diuji, dan diberi hard constraints. Namun, distance function-nya sebaiknya mengikuti prinsip Mondrian: jarak batas terdekat, alignment, dan ukuran elemen.

### Tahap E — Rekonsiliasi blank gap dan sparse table

Blank row/column jangan menjadi keputusan tunggal.

Gabungkan melintasi blank gap bila minimal dua kondisi kuat terpenuhi:

- overlap proyeksi tinggi, misalnya `>= 0.6`;
- tipe kolom sebelum dan sesudah gap konsisten;
- formula family berlanjut;
- border/fill table berlanjut;
- header hanya muncul di sisi pertama dan sisi kedua tampak seperti body lanjutan.

Pertahankan sebagai region berbeda bila salah satu kondisi berikut ada:

- kedua sisi memiliki header kuat;
- terdapat border pemisah tebal;
- schema/type sequence berubah tajam;
- ada Excel Table berbeda;
- gap mengandung title/note yang merujuk kandidat berikutnya;
- dua kandidat sejajar berdampingan dan masing-masing membentuk body rectangular yang valid.

### Tahap F — Pisahkan title, header, body, total, dan footnote

Setelah region terbentuk, cari struktur internal:

1. **Title/preamble**: satu atau sedikit sel teks, sering merged melintasi banyak kolom, font lebih besar/bold, density baris rendah, berada di atas header.
2. **Top header**: 1–5 baris awal dengan rasio teks tinggi, style/border serupa, dan body bertipe stabil di bawahnya. Dukung multi-level header melalui merged-span dan parent-child containment.
3. **Left header/index**: satu atau beberapa kolom kiri bertipe teks/date dengan body numerik/formula di kanan.
4. **Body**: pola tipe per kolom dan/atau formula family stabil pada beberapa baris.
5. **Aggregation rows/columns**: keyword total/subtotal/sum/max atau formula yang mereferensikan body.
6. **Footnote**: teks panjang/density rendah setelah body, sering setelah gap atau melebar lintas kolom.

Header detection harus menghasilkan `header_rows`, `left_header_columns`, dan confidence; jangan hanya memilih baris pertama.

### Tahap G — Refinement dan confidence

Nilai ulang boundary berdasarkan struktur internal:

- perluas bila header terkait tepat di luar kandidat atau formula family terpotong;
- perkecil bila baris/kolom tepi hanya berisi title/footnote yang tidak dibutuhkan sebagai tabel;
- pecah bila ditemukan dua header/body pair di dalam satu boundary;
- tandai ambigu bila dua segmentasi memiliki skor berdekatan.

Simpan provenance per keputusan, misalnya:

```json
{
  "method": "inferred",
  "confidence": 0.78,
  "signals": ["column_alignment", "formula_continuity", "single_header"],
  "warnings": ["blank_row_inside_body"]
}
```

Kandidat dengan confidence rendah, density sangat rendah, merge kompleks, atau konflik antar-sinyal dapat dirender per crop untuk validasi VLM. Native coordinates tetap menjadi sumber batas utama agar hasil dapat diaudit.

## Failure modes yang perlu menjadi regression tests

1. Dua tabel menyentuh diagonal; harus tetap terpisah.
2. Dua tabel berdampingan tanpa kolom kosong tetapi berbeda header/style; harus terpisah.
3. Satu tabel memiliki satu atau dua baris kosong internal; harus tetap satu region.
4. Satu tabel memiliki kolom separator kosong tetapi formula/schema berlanjut; harus tetap satu region.
5. Merged title membentang di atas dua tabel; title boleh terkait keduanya tetapi body tidak boleh menyatu.
6. Header dua atau tiga tingkat dengan merged spans.
7. Left header hierarkis dengan indentasi.
8. Baris total/subtotal yang dipisahkan border tebal.
9. Formula ada tetapi cached value kosong; sel tetap aktif secara struktural.
10. Grid kosong yang hanya memiliki border/fill; dapat memperluas tabel, tetapi tidak menciptakan tabel tanpa body.
11. Style diterapkan ke seluruh kolom/baris; tidak boleh memperbesar region ke batas worksheet.
12. Notes atau source text tepat di bawah tabel; harus menjadi footnote/metadata, bukan body.
13. Excel Table resmi bersebelahan dengan range informal; batas `Table.ref` harus dipertahankan.
14. Region berbentuk L atau T akibat dua tabel berdempetan; harus dipartisi sebelum clustering.
15. Sheet sangat sparse dengan koordinat jauh; algoritma tidak boleh mengalokasikan matriks padat penuh.

## Prioritas implementasi

1. **Dampak tinggi, risiko rendah:** prioritaskan `ws.tables`, ubah seed menjadi four-neighbor, perlakukan formula/merge/style sebagai evidence terpisah, dan tambahkan confidence/provenance.
2. **Dampak tinggi, risiko sedang:** tambahkan pairwise merge lintas blank gap menggunakan alignment + type/formula/style continuity serta hard separator rules.
3. **Dampak sedang:** klasifikasi title/header/body/total/footnote dan lakukan boundary refinement.
4. **Dampak lanjutan:** partition komponen konkaf dan clustering adaptif ala Mondrian; gunakan hanya setelah fixture tahap awal stabil.

Pendekatan ini mempertahankan determinisme dan keterlacakan openpyxl, sambil memakai VLM sebagai pemeriksa kasus ambigu alih-alih menggantikan struktur native workbook.
