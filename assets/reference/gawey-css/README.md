# Referensi warna Gawey!

Diambil dari halaman utama [gawey.id](https://gawey.id/) pada 23 September 2026. Ketiga file CSS bernama hash adalah stylesheet yang ditautkan oleh HTML halaman; `inline.css` berisi blok `<style>` pada halaman tersebut. Situs dapat mengganti hash dan isi CSS sewaktu-waktu.

| Peran | Warna | Sumber |
| --- | --- | --- |
| Biru utama | `#1B75BB` | `--gawey-blue` |
| Ungu aksen | `#3F3F97` | `--blueberry` dan gradien Gawey |
| Biru gelap | `#20204C` | `--gawey-dark-blue` |
| Biru muda | `#76ACD6` | `--moonstone-blue` |
| Toska indikator | `#13B5C8` | indikator kemajuan di `inline.css` |
| Teks | `#3B3B3B` | `--monochrome` |
| Teks sekunder | `#757575` | `--gawey-grey` |
| Garis/abu muda | `#CDCDCD` | `--gawey-light-grey` |
| Permukaan abu | `#EAEAEA` | `--green-white` |

Token siap pakai ada di `palette.css`. Gradien merek pada situs bergerak dari `#1B75BB` ke `#3F3F97`. Warna yang sama diterapkan pada antarmuka Streamlit di `app/streamlit_logic.py`. `.streamlit/config.toml` menjadikan tema terang sebagai bawaan, dengan latar putih, panel biru pucat, dan aksen widget biru. Tema gelap tetap tersedia. File referensi CSS asli tidak dimuat oleh aplikasi.

Sumber stylesheet: [7be137465b0bc017.css](https://gawey.id/_next/static/css/7be137465b0bc017.css), [d51b7b271f0a7c71.css](https://gawey.id/_next/static/css/d51b7b271f0a7c71.css), [cb74a29fa9526037.css](https://gawey.id/_next/static/css/cb74a29fa9526037.css).
