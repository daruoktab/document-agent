# Audit dan Keputusan Pengolahan Excel Heterogen

## Kesimpulan

Pipeline yang paling aman adalah **native-first dan visual-selective**. Nilai sel,
formula, tipe, Excel Table, dan named range dibaca dari struktur workbook. VLM
dipakai untuk dashboard, chart, gambar, serta region yang batas atau maknanya
ambigu. Angka hasil visual tidak menggantikan bukti native.

## Temuan audit

- Connected-components delapan arah dapat menyatukan tabel yang hanya
  bersentuhan diagonal dan dapat memecah tabel renggang.
- `ws.tables`, named range statis, dan AutoFilter dua dimensi memberikan batas
  yang lebih kuat daripada inferensi berbasis sel tidak kosong.
- Blank cell berformat berguna sebagai bukti struktur, tetapi tidak aman bila
  dianggap data tanpa syarat karena format sering diterapkan ke area berlebih.
- Sheet `hidden` dan `veryHidden` sering menyimpan lookup atau sumber formula.
  Isinya perlu dipertahankan untuk lineage native, tetapi tidak ditampilkan atau
  dirender secara default.
- Formula dan cached value merupakan fakta berbeda. Cached value dapat hilang
  atau sudah tidak mutakhir; kondisi tersebut harus menjadi warning, bukan nol.
- Inferensi tipe harus konservatif. Satu nilai teks pada kolom numerik membuat
  kolom disimpan sebagai `TEXT`, bukan dipaksa menjadi angka.

## Keputusan implementasi

1. Prioritas region: Excel Table, named range statis, AutoFilter valid, region
   heuristik, lalu objek visual.
2. Batas native yang tumpang tindih dideduplicasi agar satu sel tidak masuk ke
   beberapa tabel hasil.
3. Region heuristik memakai konektivitas empat arah.
4. Blank styled cell hanya dipakai bila berada dalam batas native atau benar-benar
   menjembatani data di kedua sisinya.
5. Header dipilih berdasarkan cakupan, keunikan, body tepat di bawahnya, pola
   numerik, dan penalti formula atau teks sangat panjang.
6. Hidden sheet diproses sebagai `support` native dan dikeluarkan dari Markdown,
   visual plan, extraction order, serta jumlah tabel pengguna.
7. Database lama tetap diverifikasi walaupun kolom berafinitas `NUMERIC`
   mengandung label teks.

## Sumber primer

- [Microsoft Open XML: formulas](https://learn.microsoft.com/en-us/office/open-xml/spreadsheet/working-with-formulas)
- [Microsoft Open XML: hidden worksheets](https://learn.microsoft.com/en-us/office/open-xml/spreadsheet/how-to-retrieve-a-list-of-the-hidden-worksheets-in-a-spreadsheet)
- [openpyxl: worksheet tables](https://openpyxl.readthedocs.io/en/stable/worksheet_tables.html)
- [openpyxl: defined names](https://openpyxl.readthedocs.io/en/stable/defined_names.html)
- [openpyxl: optimized modes](https://openpyxl.readthedocs.io/en/stable/optimized.html)
- [TableSense, AAAI 2019](https://www.microsoft.com/en-us/research/uploads/prod/2019/01/TableSense_AAAI19.pdf)
- [Mondrian, PVLDB 2021](https://arxiv.org/abs/2109.06630)
- [Semantic Table Structure Identification, ISSTA 2021](https://www.microsoft.com/en-us/research/wp-content/uploads/2020/08/Semantic-Table-Structure-Identification-in-Spreadsheets.pdf)

Rincian bukti dan skenario uji tersedia pada tiga berkas `findings_*.md` dalam
folder ini.
