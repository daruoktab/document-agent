"""
Surveyor — Pemetaan Deterministik Dokumen Scan Berskala Besar.

Modul ini TIDAK memanggil LLM dan TIDAK memakai OCR sebagai sumber teks.
Seluruh keluarannya diturunkan dari geometri citra halaman, lalu dipakai
sebagai perancah (scaffold) yang memandu VLM agent saat menulis ulang isi
dokumen secara sekuensial per batch halaman.

Yang dihasilkan per halaman:
  - orientasi, DPI efektif, dan mode (teks / tabel)
  - grid kolom tabel (posisi sekat) beserta tanda tangan schema-nya
  - peta keterisian kolom: kolom mana yang berisi, kosong, atau ragu
  - penanda awal baris logis
  - sudut kemiringan scan bila perlu dikoreksi

Yang dihasilkan per dokumen:
  - segmentasi halaman menjadi span (satu span = satu schema tabel)
  - rencana batch yang batasnya digeser ke awal baris terdekat

Catatan soal text layer: `_detect_row_start` memang membaca
`page.get_text("words")`, tetapi HANYA memanfaatkan posisi token dan ada
tidaknya digit — bukan isi karakternya. Pada dokumen hasil scan, karakter
OCR sering salah ("L2" untuk "12") sementara posisinya tetap benar.
"""

from __future__ import annotations

import argparse
import io
import json
import re
from collections.abc import Sequence

import numpy as np
import pymupdf
from PIL import Image
from pydantic import BaseModel, Field

# --- Konstanta kalibrasi -----------------------------------------------------
# Dikalibrasi pada dokumen peraturan hasil scan bilevel 300-400 DPI.

INK_THRESHOLD = 128  # ambang biner abu-abu -> tinta
H_LINE_FRAC = 0.35  # garis horizontal "panjang" = menutupi >=35% lebar halaman
V_LINE_FRAC = 0.85  # garis vertikal di pita header = menutupi >=85% tinggi pita
GROUP_GAP = 8  # piksel berdekatan digabung jadi satu garis
BODY_MARGIN = 20  # jarak aman dari garis grid saat mengukur badan tabel
COL_PAD_FRAC = 0.010  # padding kiri-kanan tiap kolom, buang sisa garis vertikal
EDGE_MARGIN_FRAC = 0.01  # abaikan garis di tepi citra (bekas tepi hitam scan)
ROW_TOKEN_TOLERANCE_PT = 4.0  # nomor baris kerap tercetak sedikit di atas garis

# Ambang keterisian kolom (kerapatan tinta x1000). Zona di antara keduanya
# sengaja dibiarkan "ragu" agar sistem mengeskalasi, bukan menebak.
DENSITY_EMPTY_MAX = 9.0
DENSITY_FILLED_MIN = 13.0

SIGNATURE_TOL = 0.03  # toleransi pencocokan posisi sekat (relatif lebar halaman)
SKEW_MAX_DEG = 1.2
SKEW_STEP_DEG = 0.1
RENDER_DPI = 300  # dipakai hanya bila halaman bukan satu citra utuh


# --- Model ------------------------------------------------------------------


class ColumnGrid(BaseModel):
    """Grid kolom sebuah halaman tabel, hasil deteksi garis pada citra."""

    n_cols: int = Field(..., description="Jumlah kolom = jumlah sekat dikurangi satu")
    separators_px: list[int] = Field(
        ..., description="Posisi x sekat kolom dalam piksel citra"
    )
    separators_rel: list[float] = Field(
        ...,
        description="Posisi x sekat relatif terhadap lebar halaman (tahan beda DPI)",
    )
    header_top_px: int = Field(..., description="y garis atas pita header")
    header_bottom_px: int = Field(..., description="y garis bawah pita header")
    body_top_px: int = Field(
        ..., description="y awal badan tabel (di bawah baris nomor kolom)"
    )
    body_bottom_px: int = Field(..., description="y akhir badan tabel")
    source: str = Field(
        default="deteksi",
        description="'deteksi' bila diukur di halaman ini, 'warisan' bila memakai grid halaman sebelumnya",
    )


class PageSurvey(BaseModel):
    """Hasil pemetaan satu halaman."""

    page_no: int = Field(..., description="Nomor halaman PDF, mulai 1")
    orientation: str = Field(..., description="'landscape' atau 'portrait'")
    dpi: int = Field(..., description="DPI efektif citra halaman")
    mode: str = Field(..., description="'tabel' bila bergrid, selain itu 'teks'")
    has_ruled_grid: bool = Field(
        default=False, description="True bila grid tabel bergaris berhasil ditemukan"
    )
    skew_deg: float = Field(
        default=0.0, description="Sudut kemiringan scan yang dikoreksi"
    )
    grid: ColumnGrid | None = Field(
        default=None, description="Grid kolom bila halaman bertabel"
    )
    col_density: list[float] = Field(
        default_factory=list, description="Kerapatan tinta per kolom (x1000)"
    )
    occupancy: list[str] = Field(
        default_factory=list,
        description="Status per kolom: 'isi', 'kosong', atau 'ragu'",
    )
    is_row_start: bool = Field(
        default=False, description="Apakah baris logis baru dimulai di halaman ini"
    )
    row_tokens: list[str] = Field(
        default_factory=list,
        description="Token di kolom nomor; posisinya yang dipercaya, bukan isinya",
    )
    warnings: list[str] = Field(
        default_factory=list, description="Catatan yang perlu ditinjau"
    )


class Batch(BaseModel):
    """Satu unit kerja untuk VLM agent."""

    page_start: int
    page_end: int
    snapped: bool = Field(
        default=False,
        description="True bila batas batch digeser agar jatuh di awal baris",
    )
    opens_at_row_start: bool = Field(
        default=False, description="True bila batch dimulai tepat di awal baris logis"
    )


class Span(BaseModel):
    """Rentang halaman berturut-turut yang memakai satu schema tabel yang sama."""

    index: int
    page_start: int
    page_end: int
    mode: str
    n_cols: int | None = None
    separators_rel: list[float] = Field(default_factory=list)
    row_start_pages: list[int] = Field(default_factory=list)
    batches: list[Batch] = Field(default_factory=list)

    @property
    def n_pages(self) -> int:
        """Jumlah halaman dalam span."""
        return self.page_end - self.page_start + 1


class SurveyReport(BaseModel):
    """Peta lengkap dokumen: bahan baku semua briefing agent."""

    source: str
    total_pages: int
    surveyed_pages: int
    spans: list[Span] = Field(default_factory=list)
    pages: list[PageSurvey] = Field(default_factory=list)


# --- Primitif citra ---------------------------------------------------------


def _group(indices: Sequence[int], gap: int = GROUP_GAP) -> list[int]:
    """Gabungkan indeks piksel yang berdekatan menjadi satu posisi garis."""
    runs: list[list[int]] = []
    for i in indices:
        if runs and i - runs[-1][-1] <= gap:
            runs[-1].append(int(i))
        else:
            runs.append([int(i)])
    return [int(np.mean(r)) for r in runs]


def _page_ink(doc: pymupdf.Document, page: pymupdf.Page) -> tuple[np.ndarray, float]:
    """
    Ambil peta tinta halaman sebagai array boolean, plus skala piksel-per-poin.

    Bila halaman berupa satu citra scan utuh, citra aslinya diambil langsung
    tanpa render ulang: jauh lebih cepat dan tidak menurunkan resolusi.
    """
    imgs = page.get_images()
    if len(imgs) == 1:
        raw = doc.extract_image(imgs[0][0])["image"]
        arr = np.array(Image.open(io.BytesIO(raw)).convert("L"))
        sx = arr.shape[1] / page.rect.width
        sy = arr.shape[0] / page.rect.height
        # Pastikan citra benar-benar sebidang dengan halaman sebelum dipercaya.
        if abs(sx - sy) / max(sx, sy) < 0.02:
            return arr < INK_THRESHOLD, sx

    pix = page.get_pixmap(dpi=RENDER_DPI, colorspace=pymupdf.csGRAY)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    return arr < INK_THRESHOLD, arr.shape[1] / page.rect.width


def _estimate_skew(ink: np.ndarray) -> float:
    """
    Perkirakan kemiringan scan dari garis grid yang berulang di tiap halaman.

    Dokumen ini mencetak ulang garis header di setiap halaman, jadi kita
    mengukur objek yang bentuknya sudah pasti lurus, bukan menebak dari isi.
    """
    small = Image.fromarray((ink[::4, ::4] * 255).astype(np.uint8))
    best_deg, best_score = 0.0, -1.0
    for deg in np.arange(-SKEW_MAX_DEG, SKEW_MAX_DEG + 1e-9, SKEW_STEP_DEG):
        rot = np.array(
            small.rotate(float(deg), resample=Image.Resampling.NEAREST, fillcolor=0)
        )
        score = float((rot > 127).sum(axis=1).max())
        if score > best_score:
            best_score, best_deg = score, float(deg)
    return round(best_deg, 2)


def _deskew(ink: np.ndarray, deg: float) -> np.ndarray:
    """Putar peta tinta sebesar `deg` derajat."""
    im = Image.fromarray((ink * 255).astype(np.uint8))
    return (
        np.array(im.rotate(deg, resample=Image.Resampling.BILINEAR, fillcolor=0)) > 127
    )


def _has_long_h_lines(ink: np.ndarray) -> bool:
    """Apakah halaman punya garis horizontal panjang sama sekali.

    Dipakai sebagai penjaga murah sebelum koreksi kemiringan: halaman tanpa
    satu pun garis panjang jelas bukan tabel bergaris, jadi tidak perlu diputar.
    """
    _height, width = ink.shape
    return bool((ink.sum(axis=1) > H_LINE_FRAC * width).any())


def _detect_grid(ink: np.ndarray) -> ColumnGrid | None:
    """
    Deteksi grid kolom lewat projection profile.

    Sekat kolom diukur HANYA di dalam pita header (antara dua garis horizontal
    panjang pertama). Pita itu pendek dan bersih sehingga garis vertikalnya
    utuh; mengukur di sepanjang badan tabel gagal karena garisnya terputus.
    """
    height, width = ink.shape
    h_lines = _group(np.where(ink.sum(axis=1) > H_LINE_FRAC * width)[0].tolist())
    if len(h_lines) < 3:
        return None

    # Blok header selalu berupa beberapa garis berdekatan (batas atas, baris
    # judul kolom, baris nomor kolom). Badan tabel dimulai di celah besar yang
    # PERTAMA, bukan yang terlebar: pada halaman berisi awal baris, badan
    # terpotong sekat sub-baris sehingga celah terlebar justru jatuh di dalamnya.
    min_body_gap = max(150, int(0.055 * height))
    body_idx = next(
        (
            i
            for i in range(len(h_lines) - 1)
            if h_lines[i + 1] - h_lines[i] >= min_body_gap
        ),
        None,
    )
    if body_idx is None or body_idx == 0:
        return None

    # Blok header = seluruh wilayah dari garis teratas sampai garis yang membuka
    # badan tabel. Diukur sekaligus, bukan per pasang garis: pada halaman miring
    # satu garis kerap terdeteksi ganda, dan pita tipis di antaranya membuat
    # goresan huruf ikut terhitung sebagai sekat kolom.
    header_top, header_bottom = h_lines[0], h_lines[body_idx]
    if header_bottom - header_top < 20:
        return None

    header = ink[header_top:header_bottom, :]
    v_lines = _group(
        np.where(header.sum(axis=0) > V_LINE_FRAC * header.shape[0])[0].tolist()
    )
    # Buang tepi hitam hasil scan: garis di batas citra bukan sekat kolom.
    v_lines = [
        v
        for v in v_lines
        if EDGE_MARGIN_FRAC * width < v < (1 - EDGE_MARGIN_FRAC) * width
    ]
    if len(v_lines) < 2:
        return None

    body_top = h_lines[body_idx] + BODY_MARGIN
    body_bottom = h_lines[-1] - BODY_MARGIN
    if body_bottom <= body_top + 50:
        body_bottom = height - int(0.06 * height)

    return ColumnGrid(
        n_cols=len(v_lines) - 1,
        separators_px=v_lines,
        separators_rel=[round(v / width, 3) for v in v_lines],
        header_top_px=header_top,
        header_bottom_px=header_bottom,
        body_top_px=body_top,
        body_bottom_px=body_bottom,
    )


def _column_density(ink: np.ndarray, grid: ColumnGrid) -> list[float]:
    """Kerapatan tinta tiap kolom di badan tabel, dikali 1000 agar terbaca."""
    width = ink.shape[1]
    pad = max(4, int(COL_PAD_FRAC * width))
    body = ink[grid.body_top_px : grid.body_bottom_px, :]
    if body.size == 0:
        return [0.0] * grid.n_cols

    out: list[float] = []
    for i in range(grid.n_cols):
        x0 = grid.separators_px[i] + pad
        x1 = grid.separators_px[i + 1] - pad
        out.append(round(float(body[:, x0:x1].mean() * 1000), 1) if x1 > x0 else 0.0)
    return out


def _classify_occupancy(densities: Sequence[float]) -> list[str]:
    """
    Terjemahkan kerapatan menjadi status keterisian kolom.

    Zona antara dua ambang dilabeli 'ragu' agar Auditor mengeskalasi. Menebak
    di zona ini merusak sifat deterministik perancah.
    """
    status: list[str] = []
    for d in densities:
        if d >= DENSITY_FILLED_MIN:
            status.append("isi")
        elif d <= DENSITY_EMPTY_MAX:
            status.append("kosong")
        else:
            status.append("ragu")
    return status


def _detect_row_start(page: pymupdf.Page, grid: ColumnGrid, scale: float) -> list[str]:
    """
    Cari token berangka di kolom pertama, dibatasi pada badan tabel saja.

    Pembatasan y penting: kode pengaman "SK No 122554 C" di kaki halaman berada
    pada rentang x yang sama dengan kolom nomor, dan akan salah terbaca sebagai
    awal baris bila ikut terjaring.
    """
    x0 = grid.separators_px[0] / scale - 2
    x1 = grid.separators_px[1] / scale + 2
    # Mulai tepat dari garis pembuka badan tabel, bukan dari batas aman
    # pengukuran tinta, lalu beri toleransi kecil ke atas.
    y0 = (grid.body_top_px - BODY_MARGIN) / scale - ROW_TOKEN_TOLERANCE_PT
    y1 = grid.body_bottom_px / scale
    return [
        w[4]
        for w in page.get_text("words")
        if x0 <= w[0] <= x1 and y0 <= w[1] <= y1 and re.search(r"\d", w[4])
    ]


# --- Pemetaan per halaman ---------------------------------------------------


def survey_page(
    doc: pymupdf.Document,
    page_no: int,
    *,
    inherited_grid: ColumnGrid | None = None,
    allow_deskew: bool = True,
) -> PageSurvey:
    """Petakan satu halaman. `inherited_grid` dipakai bila deteksi di halaman ini gagal."""
    page = doc[page_no - 1]
    landscape = page.rect.width > page.rect.height
    ink, scale = _page_ink(doc, page)

    result = PageSurvey(
        page_no=page_no,
        orientation="landscape" if landscape else "portrait",
        dpi=round(scale * 72),
        mode="teks",
    )

    if not landscape:
        return result

    grid = _detect_grid(ink)

    # Kemiringan scan adalah penyebab kegagalan grid yang paling sering, tetapi
    # koreksinya mahal. Baru dikerjakan setelah percobaan murah gagal, dan hanya
    # bila halaman memang punya garis panjang sehingga masuk akal bergrid.
    if grid is None and allow_deskew and _has_long_h_lines(ink):
        deg = _estimate_skew(ink)
        if abs(deg) >= 0.05:
            ink = _deskew(ink, deg)
            grid = _detect_grid(ink)
            result.skew_deg = deg
            if grid is not None:
                result.warnings.append(
                    f"grid ditemukan setelah koreksi miring {deg} derajat"
                )

    if grid is None and inherited_grid is not None:
        grid = inherited_grid.model_copy(update={"source": "warisan"})
        result.warnings.append("grid diwarisi dari halaman sebelumnya")

    if grid is None:
        # Halaman landscape tanpa grid itu wajar: slide presentasi, gambar
        # selebar halaman, atau teks biasa. Bukan anomali, jadi tidak ditandai.
        # Yang ditandai hanya bila halaman sebelumnya bergrid (lihat cabang di atas),
        # sebab di situ grid memang seharusnya ada.
        return result

    result.grid = grid
    result.mode = "tabel"
    result.has_ruled_grid = grid.source == "deteksi"
    result.col_density = _column_density(ink, grid)
    result.occupancy = _classify_occupancy(result.col_density)
    result.row_tokens = _detect_row_start(page, grid, scale)
    result.is_row_start = bool(result.row_tokens)

    if "ragu" in result.occupancy:
        ragu = [i + 1 for i, s in enumerate(result.occupancy) if s == "ragu"]
        result.warnings.append(f"keterisian kolom {ragu} tidak pasti")

    return result


# --- Segmentasi span --------------------------------------------------------


def _signature_matches(page: PageSurvey, n_cols: int, sep_rel: Sequence[float]) -> bool:
    """Dua halaman dianggap satu schema bila jumlah kolom sama dan sekatnya berimpit."""
    if page.grid is None or page.grid.n_cols != n_cols:
        return False
    if len(page.grid.separators_rel) != len(sep_rel):
        return False
    return (
        max(abs(x - y) for x, y in zip(page.grid.separators_rel, sep_rel))
        <= SIGNATURE_TOL
    )


def segment_spans(pages: Sequence[PageSurvey]) -> list[Span]:
    """
    Pecah daftar halaman menjadi span. Satu span = satu schema tabel.

    Span inilah yang memicu pemanggilan agent penentu schema: sekali per span,
    bukan sekali per halaman. Header tabel dicetak ulang di setiap halaman,
    jadi kemunculan header BUKAN penanda tabel baru.
    """
    spans: list[Span] = []
    current: Span | None = None

    for page in pages:
        if current is not None:
            same_mode = page.mode == current.mode
            same_schema = page.mode != "tabel" or _signature_matches(
                page,
                current.n_cols if current.n_cols is not None else -1,
                current.separators_rel,
            )
            if same_mode and same_schema:
                current.page_end = page.page_no
                if page.is_row_start:
                    current.row_start_pages.append(page.page_no)
                continue
            spans.append(current)

        current = Span(
            index=len(spans),
            page_start=page.page_no,
            page_end=page.page_no,
            mode=page.mode,
            n_cols=page.grid.n_cols if page.grid else None,
            separators_rel=list(page.grid.separators_rel) if page.grid else [],
            # Halaman pertama sebuah span selalu dianggap awal baris: di halaman
            # judul lampiran, kolom nomor sering tidak terjaring deteksi biasa.
            row_start_pages=[page.page_no],
        )

    if current is not None:
        spans.append(current)
    return spans


def plan_batches(span: Span, size: int = 10, snap: int = 2) -> list[Batch]:
    """
    Susun rencana batch untuk satu span.

    Ukuran nominal tetap `size` halaman, tetapi batasnya digeser sampai `snap`
    halaman agar jatuh tepat di awal baris logis. Batch yang dimulai di awal
    baris membuka jauh lebih sedikit scope, dan bisa dijalankan ulang sendirian
    tanpa bergantung pada hasil batch sebelumnya.
    """
    row_starts = set(span.row_start_pages)
    batches: list[Batch] = []
    cursor = span.page_start

    while cursor <= span.page_end:
        nominal_end = cursor + size - 1
        snapped = False

        if nominal_end >= span.page_end:
            end = span.page_end
        else:
            candidates = [
                p
                for p in range(nominal_end + 1 - snap, nominal_end + 1 + snap + 1)
                if p in row_starts and cursor < p <= span.page_end
            ]
            if candidates:
                best = min(candidates, key=lambda p: abs(p - (nominal_end + 1)))
                end = best - 1
                snapped = end != nominal_end
            else:
                end = nominal_end

        batches.append(
            Batch(
                page_start=cursor,
                page_end=end,
                snapped=snapped,
                opens_at_row_start=cursor in row_starts,
            )
        )
        cursor = end + 1

    return batches


# --- API utama --------------------------------------------------------------


def survey(
    pdf_path: str,
    *,
    page_range: tuple[int, int] | None = None,
    batch_size: int = 10,
    snap: int = 2,
    allow_deskew: bool = True,
    progress_every: int = 0,
) -> SurveyReport:
    """Petakan dokumen (atau sebagian halamannya) menjadi `SurveyReport`."""
    doc = pymupdf.open(pdf_path)
    total_pages = doc.page_count

    first, last = page_range or (1, total_pages)
    first = max(1, first)
    last = min(total_pages, last)

    pages: list[PageSurvey] = []
    inherited: ColumnGrid | None = None

    for page_no in range(first, last + 1):
        result = survey_page(
            doc, page_no, inherited_grid=inherited, allow_deskew=allow_deskew
        )
        # Hanya grid hasil deteksi asli yang layak diwariskan. Warisan diputus
        # saat orientasi berubah, bukan saat satu halaman gagal terdeteksi:
        # kegagalan satu halaman di tengah tabel justru alasan utama adanya warisan.
        if result.grid is not None and result.grid.source == "deteksi":
            inherited = result.grid
        elif result.orientation == "portrait":
            inherited = None
        pages.append(result)

        if progress_every and (page_no - first + 1) % progress_every == 0:
            print(f"  ... {page_no - first + 1}/{last - first + 1} halaman", flush=True)

    doc.close()

    spans = segment_spans(pages)
    for span in spans:
        span.batches = plan_batches(span, size=batch_size, snap=snap)

    return SurveyReport(
        source=pdf_path,
        total_pages=total_pages,
        surveyed_pages=len(pages),
        spans=spans,
        pages=pages,
    )


# --- CLI --------------------------------------------------------------------


def format_report(report: SurveyReport) -> str:
    """Ringkasan laporan yang enak dibaca di terminal."""
    lines = [
        f"sumber            : {report.source}",
        f"halaman dokumen   : {report.total_pages}",
        f"halaman dipetakan : {report.surveyed_pages}",
        f"span ditemukan    : {len(report.spans)}",
        "",
        f"{'span':>4}  {'halaman':>15}  {'mode':<12} {'kolom':>5}  {'baris':>6}  {'batch':>5}",
    ]
    for span in report.spans:
        lines.append(
            f"{span.index:>4}  {f'{span.page_start}-{span.page_end}':>15}  {span.mode:<12} "
            f"{span.n_cols or '-'!s:>5}  {len(span.row_start_pages):>6}  {len(span.batches):>5}"
        )

    flagged = [p for p in report.pages if p.warnings]
    lines += ["", f"halaman bertanda  : {len(flagged)}"]
    for page in flagged[:15]:
        lines.append(f"  p.{page.page_no}: {'; '.join(page.warnings)}")
    if len(flagged) > 15:
        lines.append(f"  ... {len(flagged) - 15} lainnya")
    return "\n".join(lines)


def _cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.surveyor",
        description="Petakan dokumen scan menjadi perancah deterministik untuk VLM agent.",
    )
    parser.add_argument("pdf", help="Path file PDF")
    parser.add_argument("--pages", default=None, help="Rentang halaman, mis. 384-425")
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Ukuran batch nominal"
    )
    parser.add_argument(
        "--snap", type=int, default=2, help="Toleransi geser batas batch"
    )
    parser.add_argument(
        "--no-deskew", action="store_true", help="Matikan koreksi kemiringan"
    )
    parser.add_argument(
        "--out", default=None, help="Simpan laporan lengkap ke file JSON"
    )
    parser.add_argument(
        "--detail", action="store_true", help="Cetak rincian per halaman"
    )
    args = parser.parse_args(argv)

    page_range = None
    if args.pages:
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", args.pages.strip())
        if not match:
            parser.error("--pages harus berformat AWAL-AKHIR, mis. 384-425")
        page_range = (int(match.group(1)), int(match.group(2)))

    report = survey(
        args.pdf,
        page_range=page_range,
        batch_size=args.batch_size,
        snap=args.snap,
        allow_deskew=not args.no_deskew,
        progress_every=50,
    )

    print(format_report(report))

    if args.detail:
        symbol = {"isi": "#", "kosong": ".", "ragu": "?"}
        print("\nrincian halaman:")
        for page in report.pages:
            mark = "AWAL-BARIS" if page.is_row_start else "          "
            occupancy = "".join(symbol[s] for s in page.occupancy)
            print(
                f"  p.{page.page_no:>6} {page.mode:<12} {mark} "
                f"kolom[{occupancy}] {page.row_tokens}"
            )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report.model_dump(), handle, ensure_ascii=False, indent=2)
        print(f"\nlaporan lengkap -> {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
