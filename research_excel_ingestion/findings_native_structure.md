# Temuan: Struktur Native Workbook Excel yang Tangguh

## Ringkasan audit kode saat ini

Implementasi `survey_excel_workbook()` sudah memiliki fondasi yang tepat: workbook dibuka dua kali untuk mempertahankan formula dan membaca cache nilainya, batas data dihitung dari sel yang benar-benar memiliki nilai (bukan sekadar `max_row`/`max_column`), merged range dipetakan, dan status sheet dicatat. Ini menghindari sebagian besar masalah `UsedRange` yang membesar akibat format kosong.

Kesenjangan utama untuk workbook acak adalah: sheet hidden/very-hidden hanya dicatat lalu dilewati; Excel Table, defined name, dan print area belum dipakai sebagai batas region berotoritas; sheet non-worksheet seperti chartsheet belum diinventarisasi; serta cache formula belum dibedakan secara eksplisit antara “tersedia” dan “terjamin mutakhir”.

## Fakta primer dan implikasi

### 1. Jangan menjadikan dimensi/used range sebagai kebenaran tunggal

- Microsoft menyatakan `UsedRange` juga mencakup sel kosong yang hanya memiliki format. Openpyxl juga memperingatkan bahwa produsen file dapat menulis metadata dimensi yang salah, khususnya pada mode read-only.
- Implikasi: pertahankan pemindaian sel bernilai sebagai sumber batas aktual. Gunakan `calculate_dimension()` hanya sebagai petunjuk/peringatan, bukan batas final. Untuk file besar, gunakan read-only streaming dengan pemeriksaan kewajaran dimensi dan `reset_dimensions()` bila metadata jelas salah.
- Sumber: https://learn.microsoft.com/en-us/office/vba/excel/concepts/cells-and-ranges/select-a-range
- Sumber: https://openpyxl.readthedocs.io/en/stable/optimized.html

### 2. Excel Table harus menjadi seed region prioritas tertinggi

- Excel Table memiliki `ref` eksplisit, header, filter, kolom, dan kemungkinan totals row. Openpyxl menyediakan `ws.tables` beserta nama dan range tabel.
- Implikasi: sebelum connected-component detection, ambil seluruh `ws.tables.values()` sebagai region berotoritas. Jangan pecah tabel hanya karena ada sel kosong di dalamnya. Validasi header sebagai string, tangani totals row secara terpisah, lalu keluarkan sel tabel dari pencarian region heuristik agar tidak diduplikasi.
- Sumber: https://openpyxl.readthedocs.io/en/stable/worksheet_tables.html
- Sumber API: https://openpyxl.readthedocs.io/en/stable/api/openpyxl.worksheet.table.html

### 3. Defined name dan print area adalah landmark, bukan selalu tabel

- Defined name dapat menunjuk konstanta, formula, satu range, banyak range, atau range lintas sheet; cakupannya bisa global atau lokal per sheet. Print area/title merupakan defined name khusus. Openpyxl tidak selalu dapat menyelesaikan defined name dinamis yang berbasis formula atau Table.
- Implikasi: inventarisasi global dan local defined names; klasifikasikan hanya destinasi range statis sebagai kandidat region. Gunakan print area untuk meningkatkan prioritas render, bukan memangkas data di luar area. Simpan defined name dinamis sebagai metadata/warning dan jangan menganggapnya gagal atau kosong.
- Sumber: https://openpyxl.readthedocs.io/en/stable/defined_names.html
- Sumber: https://learn.microsoft.com/en-us/office/open-xml/spreadsheet/how-to-retrieve-a-dictionary-of-all-named-ranges-in-a-spreadsheet

### 4. Hidden dan very-hidden tetap penting bagi lineage

- Excel mendukung `Hidden` dan `VeryHidden`. Sheet tersembunyi sering menjadi lookup, staging, atau sumber formula meskipun bukan keluaran yang perlu ditampilkan kepada pengguna.
- Implikasi: jangan melewati ekstraksi native sepenuhnya. Buat mode `support_only`: inventarisasi nilai, table/name, formula dependency, dan schema tanpa render VLM secara default. Tandai tingkat visibilitas secara utuh (`visible`, `hidden`, `veryHidden`) dan cegah sheet pendukung muncul sebagai laporan utama kecuali direferensikan atau diminta.
- Sumber: https://learn.microsoft.com/en-us/office/open-xml/spreadsheet/how-to-retrieve-a-list-of-the-hidden-worksheets-in-a-spreadsheet

### 5. Formula dan nilai cache adalah dua fakta berbeda

- SpreadsheetML menyimpan formula pada elemen `<f>` dan boleh menyimpan nilai hasil kalkulasi terakhir pada `<v>`. Nilai cache dapat hilang dan secara semantik hanya mencerminkan kalkulasi terakhir oleh aplikasi spreadsheet. Openpyxl tidak menghitung formula; `data_only=True` hanya membaca nilai tersimpan terakhir.
- Implikasi: simpan `formula`, `cached_value`, dan `cache_status` secara terpisah. Status minimal: `not_formula`, `cached`, `missing_cache`; jangan mengubah missing cache menjadi nol/kosong. Beri peringatan bila keputusan numerik memakai cache formula, terutama untuk formula lintas workbook/external link.
- Sumber: https://learn.microsoft.com/en-us/office/open-xml/spreadsheet/working-with-formulas
- Sumber: https://openpyxl.readthedocs.io/en/stable/tutorial.html

### 6. Merged cell bukan sekumpulan nilai kosong biasa

- Pada merged range, hanya sel kiri-atas yang mempertahankan nilai; sel lainnya menjadi `MergedCell` dan nilainya `None`.
- Implikasi: simpan anchor dan span merge sebagai struktur. Untuk header, nilai anchor boleh diproyeksikan sebagai label rentang ketika membangun hierarki kolom. Untuk badan data, jangan forward-fill otomatis karena merge dapat bersifat dekoratif atau menandai kelompok dan berisiko mengubah makna.
- Sumber: https://openpyxl.readthedocs.io/en/stable/editing_worksheets.html

### 7. Tipe tanggal bergantung pada epoch dan number format

- Workbook XLSX dapat memakai sistem tanggal 1900 atau 1904; angka serial yang sama dapat bermakna berbeda. Excel juga tidak menyimpan zona waktu, dan durasi memakai number format tertentu.
- Implikasi: simpan nilai Python, `data_type`, `number_format`, serta epoch workbook. Normalisasikan tanggal ke ISO 8601 hanya setelah openpyxl menafsirkannya; pertahankan durasi sebagai durasi, bukan tanggal. Jangan menginfer tanggal hanya dari besar angka atau tampilan string.
- Sumber: https://openpyxl.readthedocs.io/en/stable/datetime.html

### 8. Inventaris workbook harus mencakup lebih dari Worksheet

- OOXML membedakan worksheet, chartsheet, dan dialogsheet; chart/drawing merupakan bagian tersendiri yang berelasi dengan sheet.
- Implikasi: buat manifest seluruh sheet part. Chartsheet perlu masuk antrean visual/metadata walaupun tidak memiliki grid sel. Dialogsheet atau objek yang tidak didukung harus dicatat sebagai unsupported evidence, bukan diam-diam hilang.
- Sumber: https://learn.microsoft.com/en-us/office/open-xml/spreadsheet/working-with-sheets

## Urutan implementasi yang disarankan

1. Tambahkan manifest lengkap: jenis sheet, `sheet_state`, epoch, tables, defined names, print area/title, hidden rows/columns, dan formula-cache status.
2. Bangun region berlapis: **Excel Table -> static named/print ranges -> pivot/filter range -> heuristik connected components -> drawings/charts**. Deduplicasi region yang saling bertumpuk dengan provenance, bukan hanya koordinat.
3. Untuk area heuristik, gunakan konektivitas adaptif: izinkan gap kecil jika header/schema/style/formula pattern konsisten, tetapi pertahankan pemisahan ketika ada judul baru, perubahan lebar kolom besar, atau blank band yang kuat.
4. Ekstrak hidden/very-hidden sebagai support-only untuk dependency dan verifikasi; render hanya sheet yang terlihat kecuali diminta.
5. Terapkan logical type inference per kolom dengan confidence dan exception count: boolean, integer, real/decimal, date, datetime, duration, text, error, mixed. Fallback selalu `TEXT/mixed`, bukan memaksa numerik.
6. Simpan provenance tiap nilai: sheet, koordinat, region/table/name, formula, cache status, number format, merge anchor, dan visibility.

## Matriks uji minimum

- Dua tabel berdampingan; tabel dipisah satu blank row; tabel dengan blank row internal; dua tabel tanpa separator tetapi schema berbeda.
- Excel Table dengan totals row, filter, kolom kosong, calculated column, dan table yang berada di sheet hidden.
- Defined name global/lokal, multi-area, dinamis, print area berbasis table, serta nama yang menunjuk konstanta/formula.
- Formula dengan cache valid, cache hilang, external link, error (`#N/A`, `#DIV/0!`), array/dynamic spill.
- Merge untuk judul, header bertingkat, dan group label pada badan data.
- Workbook epoch 1900/1904, tanggal, waktu, durasi lebih dari 24 jam, angka berformat tanggal tetapi sebenarnya identifier.
- Sheet visible/hidden/veryHidden, hidden row/column, chartsheet, dan workbook yang dimensi XML-nya salah atau membesar karena formatting.

## Batas pustaka

Openpyxl sendiri menyatakan read-only mode tidak menyediakan semua fitur, tidak menghitung formula, dan tidak membaca seluruh kemungkinan objek Excel. Karena itu, parsing native sebaiknya tetap menjadi sumber data utama, sedangkan LibreOffice/headless rendering atau VLM dipakai sebagai jalur visual/fallback, bukan untuk menggantikan struktur Table, formula, tipe, dan provenance.
