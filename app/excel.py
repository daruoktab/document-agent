"""
Pipeline Excel berbasis Vision VLM.

Workbook dirender lewat LibreOffice menjadi PDF sementara, lalu mengikuti pipeline
PDF: setiap halaman hasil cetak spreadsheet dirender menjadi gambar dan dikirim
satu per satu ke VLM.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import socket
import sqlite3
import subprocess
import tempfile
import time
import unicodedata
from collections import deque
from datetime import date, datetime
from datetime import time as datetime_time
from pathlib import Path
from typing import Any

from .config import DEFAULT_DPI, Settings, get_settings
from .pdf import pdf_page_count, process_multipage_pdf
from .ppt import _find_libreoffice_binary
from .schemas import (
    ExcelCellEvidence,
    ExcelRegion,
    ExcelSheetSurvey,
    ExcelWorkbookSurvey,
    ExtractedDocument,
)

logger = logging.getLogger(__name__)


def _find_uno_python(soffice: str | None = None) -> str | None:
    """Temukan Python sistem yang menyediakan modul UNO LibreOffice."""
    candidates = []
    if soffice:
        candidates.append(str(Path(soffice).resolve().with_name("python.exe")))
    candidates.extend(
        [
            r"C:\Program Files\LibreOffice\program\python.exe",
            r"C:\Program Files (x86)\LibreOffice\program\python.exe",
            "/usr/bin/python3",
            "/usr/lib/libreoffice/program/python",
            "/Applications/LibreOffice.app/Contents/Resources/python",
        ]
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    return None


def _convert_excel_to_pdf_with_page_scaling(
    soffice: str,
    excel_path: Path,
    pdf_path: Path,
    profile_dir: Path,
) -> bool:
    """Ekspor Excel melalui UNO dengan semua kolom dipaskan ke satu lebar halaman."""
    uno_python = _find_uno_python(soffice)
    if uno_python is None:
        return False

    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
    except OSError as exc:
        logger.warning("Tidak dapat membuka port lokal untuk UNO: %s", exc)
        return False

    listener = subprocess.Popen(
        [
            soffice,
            f"-env:UserInstallation={profile_dir.as_uri()}",
            "--headless",
            "--norestore",
            "--nodefault",
            "--nofirststartwizard",
            f"--accept=socket,host=127.0.0.1,port={port};urp;StarOffice.ComponentContext",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    uno_script = r"""
import sys
import time
import uno
from com.sun.star.beans import PropertyValue


def prop(name, value):
    item = PropertyValue()
    item.Name = name
    item.Value = value
    return item


port = int(sys.argv[1])
source_url = uno.systemPathToFileUrl(sys.argv[2])
target_url = uno.systemPathToFileUrl(sys.argv[3])
local_context = uno.getComponentContext()
resolver = local_context.ServiceManager.createInstanceWithContext(
    "com.sun.star.bridge.UnoUrlResolver", local_context
)
context = None
for _ in range(60):
    try:
        context = resolver.resolve(
            f"uno:socket,host=127.0.0.1,port={port};urp;StarOffice.ComponentContext"
        )
        break
    except Exception:
        time.sleep(0.25)
if context is None:
    raise RuntimeError("Tidak dapat terhubung ke listener LibreOffice UNO")

desktop = context.ServiceManager.createInstanceWithContext(
    "com.sun.star.frame.Desktop", context
)
document = desktop.loadComponentFromURL(
    source_url,
    "_blank",
    0,
    (prop("Hidden", True), prop("ReadOnly", True)),
)
if document is None:
    raise RuntimeError("LibreOffice tidak dapat membuka workbook")

try:
    page_styles = document.getStyleFamilies().getByName("PageStyles")
    sheets = document.getSheets()
    for index in range(sheets.getCount()):
        sheet = sheets.getByIndex(index)
        page_style = page_styles.getByName(sheet.getPropertyValue("PageStyle"))
        page_style.setPropertyValue("ScaleToPagesX", 1)
        page_style.setPropertyValue("ScaleToPagesY", 0)

    document.storeToURL(
        target_url,
        (prop("FilterName", "calc_pdf_Export"), prop("Overwrite", True)),
    )
finally:
    document.close(True)
    desktop.terminate()
"""

    try:
        subprocess.run(
            [uno_python, "-c", uno_script, str(port), str(excel_path), str(pdf_path)],
            capture_output=True,
            text=True,
            timeout=180,
            check=True,
        )
        return pdf_path.exists()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning(
            "Ekspor Excel dengan pengaturan fit-to-width gagal; gunakan ekspor default: %s",
            exc,
        )
        return False
    finally:
        listener.terminate()
        try:
            listener.wait(timeout=10)
        except subprocess.TimeoutExpired:
            listener.kill()
            listener.wait()
        time.sleep(0.5)


def convert_excel_to_pdf(
    excel_path: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    """Konversi workbook Excel/ODS menjadi PDF memakai LibreOffice headless."""
    path_obj = Path(excel_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"File Excel tidak ditemukan: {path_obj}")

    soffice = _find_libreoffice_binary()
    if not soffice:
        raise RuntimeError(
            "LibreOffice (soffice) tidak ditemukan. Diperlukan untuk merender Excel ke gambar halaman."
        )

    target_dir = (
        Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="excel_pdf_"))
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = target_dir / f"{path_obj.stem}.pdf"
    user_profile = target_dir / "lo_profile"
    converted = _convert_excel_to_pdf_with_page_scaling(
        soffice, path_obj, pdf_path, user_profile
    )

    if not converted:
        cmd = [
            soffice,
            f"-env:UserInstallation={user_profile.as_uri()}",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(target_dir),
            str(path_obj),
        ]
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if res.returncode != 0:
            raise RuntimeError(
                f"LibreOffice gagal mengonversi Excel ke PDF: {res.stderr.strip()}"
            )

    if not pdf_path.exists():
        candidates = sorted(target_dir.glob("*.pdf"))
        if candidates:
            return candidates[0]
        raise RuntimeError(
            "LibreOffice selesai, tetapi file PDF hasil konversi tidak ditemukan."
        )
    return pdf_path


def _slug(value: str, *, fallback: str = "sheet") -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_value).strip("_").lower()
    return cleaned[:48] or fallback


def _json_safe_cell_value(value: Any) -> Any:
    if isinstance(value, (datetime, date, datetime_time)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _extract_period(text: str) -> str | None:
    months = (
        "jan(?:uari|uary)?|feb(?:ruari|ruary)?|mar(?:et|ch)?|apr(?:il)?|"
        "mei|may|jun(?:i|e)?|jul(?:i|y)?|agu(?:stus)?|aug(?:ust)?|"
        "sep(?:tember)?|okt(?:ober)?|oct(?:ober)?|nov(?:ember)?|des(?:ember)?|dec(?:ember)?"
    )
    patterns = (
        rf"\b(?:{months})\s+20\d{{2}}\b",
        rf"\b\d{{1,2}}\s+(?:{months})\s+20\d{{2}}\b",
        r"\b(?:0?[1-9]|1[0-2])[/\-]20\d{2}\b",
        r"\b20\d{2}[/\-](?:0?[1-9]|1[0-2])\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return None


def _cell_has_visible_style(cell: Any) -> bool:
    fill = getattr(cell, "fill", None)
    if fill is not None and getattr(fill, "fill_type", None):
        return True
    border = getattr(cell, "border", None)
    if border is not None:
        for side_name in ("left", "right", "top", "bottom"):
            if getattr(getattr(border, side_name, None), "style", None):
                return True
    font = getattr(cell, "font", None)
    alignment = getattr(cell, "alignment", None)
    return bool(
        getattr(font, "bold", False)
        or getattr(font, "italic", False)
        or int(getattr(alignment, "text_rotation", 0) or 0)
    )


def _connected_components(cells: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    remaining = set(cells)
    components: list[set[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        component = {start}
        queue: deque[tuple[int, int]] = deque([start])
        while queue:
            row, column = queue.popleft()
            for row_offset in (-1, 0, 1):
                for column_offset in (-1, 0, 1):
                    if row_offset == 0 and column_offset == 0:
                        continue
                    candidate = (row + row_offset, column + column_offset)
                    if candidate in remaining:
                        remaining.remove(candidate)
                        component.add(candidate)
                        queue.append(candidate)
        components.append(component)
    return components


def _drawing_regions(worksheet: Any) -> list[tuple[str, int, int, int, int]]:
    """Ambil batas cell chart/gambar; gunakan ukuran aman untuk anchor satu sel."""
    drawings: list[tuple[str, int, int, int, int]] = []
    objects = [
        *(('chart', item) for item in getattr(worksheet, "_charts", [])),
        *(('image', item) for item in getattr(worksheet, "_images", [])),
    ]
    for kind, drawing in objects:
        anchor = getattr(drawing, "anchor", None)
        if anchor is None:
            continue
        start = getattr(anchor, "_from", None)
        end = getattr(anchor, "to", None)
        if start is not None:
            min_row = int(start.row) + 1
            min_column = int(start.col) + 1
            max_row = int(end.row) + 1 if end is not None else min_row + 14
            max_column = int(end.col) + 1 if end is not None else min_column + 7
        elif isinstance(anchor, str):
            try:
                from openpyxl.utils.cell import coordinate_to_tuple

                min_row, min_column = coordinate_to_tuple(anchor)
            except (TypeError, ValueError):
                continue
            max_row = min_row + 14
            max_column = min_column + 7
        else:
            continue
        drawings.append((kind, min_row, max_row, min_column, max_column))
    return drawings


def _region_native_text(
    *,
    sheet_name: str,
    cell_range: str,
    cells: list[ExcelCellEvidence],
) -> str:
    lines = [f"Sheet: {sheet_name}", f"Range: {cell_range}"]
    by_row: dict[int, list[ExcelCellEvidence]] = {}
    for cell in cells:
        by_row.setdefault(cell.row, []).append(cell)
    for row_number in sorted(by_row):
        items = []
        for cell in sorted(by_row[row_number], key=lambda item: item.column):
            value = cell.value
            if value is None and cell.formula is None:
                continue
            rendered = "" if value is None else str(value)
            if cell.formula:
                rendered = f"{rendered} [formula={cell.formula}]"
            items.append(f"{cell.coordinate}={rendered}")
        if items:
            lines.append(" | ".join(items))
    return "\n".join(lines)


def survey_excel_workbook(
    excel_path: str | Path,
    *,
    settings: Settings | None = None,
) -> ExcelWorkbookSurvey:
    """Inventaris region workbook tanpa mengubah file sumber."""
    try:
        from openpyxl import load_workbook
        from openpyxl.utils import get_column_letter, range_boundaries
    except ImportError as exc:  # pragma: no cover - dependency installation guard
        raise RuntimeError(
            "openpyxl diperlukan untuk survei struktur workbook. Jalankan instalasi dependensi proyek."
        ) from exc

    resolved_settings = settings or get_settings()
    source = Path(excel_path).resolve()
    keep_vba = source.suffix.lower() == ".xlsm"
    formula_book = load_workbook(
        source,
        read_only=False,
        data_only=False,
        keep_vba=keep_vba,
        keep_links=True,
    )
    value_book = load_workbook(
        source,
        read_only=False,
        data_only=True,
        keep_vba=keep_vba,
        keep_links=True,
    )
    warnings: list[str] = []
    surveyed_sheets: list[ExcelSheetSurvey] = []
    try:
        for sheet_index, worksheet in enumerate(formula_book.worksheets):
            visible = worksheet.sheet_state == "visible"
            if not visible:
                surveyed_sheets.append(
                    ExcelSheetSurvey(
                        name=worksheet.title,
                        index=sheet_index,
                        visible=False,
                    )
                )
                continue

            cached_sheet = value_book[worksheet.title]
            raw_cells = list(getattr(worksheet, "_cells", {}).values())
            drawing_specs = _drawing_regions(worksheet)
            data_coordinates = {
                (cell.row, cell.column)
                for cell in raw_cells
                if cell.value is not None
            }
            if not data_coordinates and not drawing_specs:
                surveyed_sheets.append(
                    ExcelSheetSurvey(
                        name=worksheet.title,
                        index=sheet_index,
                        visible=True,
                    )
                )
                continue

            occupied = set(data_coordinates)
            if data_coordinates:
                min_data_row = min(row for row, _ in data_coordinates)
                max_data_row = max(row for row, _ in data_coordinates)
                min_data_column = min(column for _, column in data_coordinates)
                max_data_column = max(column for _, column in data_coordinates)
            else:
                min_data_row = min(spec[1] for spec in drawing_specs)
                max_data_row = max(spec[2] for spec in drawing_specs)
                min_data_column = min(spec[3] for spec in drawing_specs)
                max_data_column = max(spec[4] for spec in drawing_specs)

            for cell in raw_cells:
                if (
                    min_data_row - 1 <= cell.row <= max_data_row + 1
                    and min_data_column - 1 <= cell.column <= max_data_column + 1
                    and _cell_has_visible_style(cell)
                ):
                    occupied.add((cell.row, cell.column))

            merged_by_coordinate: dict[tuple[int, int], str] = {}
            for merged_range in worksheet.merged_cells.ranges:
                min_column, min_row, max_column, max_row = range_boundaries(
                    str(merged_range)
                )
                if (max_row - min_row + 1) * (max_column - min_column + 1) > 10_000:
                    warnings.append(
                        f"Merge sangat besar di {worksheet.title}!{merged_range} diabaikan sebagian."
                    )
                    continue
                anchor_value = worksheet.cell(min_row, min_column).value
                for row in range(min_row, max_row + 1):
                    for column in range(min_column, max_column + 1):
                        merged_by_coordinate[(row, column)] = str(merged_range)
                        if anchor_value is not None:
                            occupied.add((row, column))

            regions: list[ExcelRegion] = []
            components = sorted(
                _connected_components(occupied),
                key=lambda component: (
                    min(row for row, _ in component),
                    min(column for _, column in component),
                ),
            )
            for component in components:
                if not component.intersection(data_coordinates):
                    continue
                min_row = min(row for row, _ in component)
                max_row = max(row for row, _ in component)
                min_column = min(column for _, column in component)
                max_column = max(column for _, column in component)
                evidence: list[ExcelCellEvidence] = []
                text_values: list[str] = []
                minimum_font_size: float | None = None
                has_rotation = False
                formula_without_cache = False

                for row, column in sorted(component):
                    source_cell = worksheet.cell(row, column)
                    formula = (
                        str(source_cell.value)
                        if source_cell.data_type == "f"
                        else None
                    )
                    cached_value = cached_sheet.cell(row, column).value
                    value = cached_value if formula else source_cell.value
                    if formula and cached_value is None:
                        formula_without_cache = True
                    if value is None and formula is None:
                        continue
                    font_size_raw = getattr(source_cell.font, "sz", None)
                    font_size = float(font_size_raw) if font_size_raw else None
                    if font_size is not None:
                        minimum_font_size = (
                            font_size
                            if minimum_font_size is None
                            else min(minimum_font_size, font_size)
                        )
                    rotation = int(source_cell.alignment.text_rotation or 0)
                    has_rotation = has_rotation or rotation != 0
                    safe_value = _json_safe_cell_value(value)
                    if isinstance(safe_value, str) and safe_value.strip():
                        text_values.append(safe_value.strip())
                    evidence.append(
                        ExcelCellEvidence(
                            coordinate=source_cell.coordinate,
                            row=row,
                            column=column,
                            value=safe_value,
                            formula=formula,
                            number_format=source_cell.number_format or "General",
                            data_type=source_cell.data_type,
                            merged_range=merged_by_coordinate.get((row, column)),
                            font_size=font_size,
                            text_rotation=rotation,
                        )
                    )

                if not evidence:
                    continue
                cell_range = (
                    f"{get_column_letter(min_column)}{min_row}:"
                    f"{get_column_letter(max_column)}{max_row}"
                )
                first_rows = [
                    str(cell.value).strip()
                    for cell in evidence
                    if cell.value not in (None, "") and cell.row <= min_row + 2
                ]
                title = " ".join(first_rows[:3]).strip() or None
                period = _extract_period(" ".join(text_values[:20]))
                row_count = len({cell.row for cell in evidence})
                column_count = len({cell.column for cell in evidence})
                kind = "table" if row_count >= 2 and column_count >= 2 else "text"
                width = max_column - min_column + 1
                height = max_row - min_row + 1
                small_font = bool(
                    minimum_font_size is not None
                    and minimum_font_size <= resolved_settings.excel_small_font_points
                )
                requires_vlm = bool(
                    small_font
                    or has_rotation
                    or width > resolved_settings.excel_max_region_columns
                    or height > resolved_settings.excel_max_region_rows
                )
                render_dpi = resolved_settings.excel_base_dpi
                if requires_vlm:
                    render_dpi = min(
                        resolved_settings.excel_max_dpi,
                        max(render_dpi, 400),
                    )
                if minimum_font_size is not None and minimum_font_size <= 6:
                    render_dpi = resolved_settings.excel_max_dpi

                region = ExcelRegion(
                    region_id=f"s{sheet_index + 1:03d}_r{len(regions) + 1:03d}",
                    sheet_name=worksheet.title,
                    sheet_index=sheet_index,
                    cell_range=cell_range,
                    min_row=min_row,
                    max_row=max_row,
                    min_column=min_column,
                    max_column=max_column,
                    title=title,
                    period=period,
                    kind=kind,
                    render_dpi=render_dpi,
                    requires_vlm_reading=requires_vlm,
                    cells=evidence,
                )
                region.native_text = _region_native_text(
                    sheet_name=worksheet.title,
                    cell_range=cell_range,
                    cells=evidence,
                )
                regions.append(region)
                if formula_without_cache:
                    warnings.append(
                        f"Nilai cache formula tidak tersedia di {worksheet.title}!{cell_range}."
                    )

            for kind, min_row, max_row, min_column, max_column in drawing_specs:
                drawing_number = (
                    sum(
                        1
                        for region in regions
                        if (region.title or "").startswith(f"{kind.title()} ")
                    )
                    + 1
                )
                cell_range = (
                    f"{get_column_letter(min_column)}{min_row}:"
                    f"{get_column_letter(max_column)}{max_row}"
                )
                regions.append(
                    ExcelRegion(
                        region_id=f"s{sheet_index + 1:03d}_r{len(regions) + 1:03d}",
                        sheet_name=worksheet.title,
                        sheet_index=sheet_index,
                        cell_range=cell_range,
                        min_row=min_row,
                        max_row=max_row,
                        min_column=min_column,
                        max_column=max_column,
                        title=f"{kind.title()} {drawing_number}",
                        kind="mixed",
                        render_dpi=min(
                            resolved_settings.excel_max_dpi,
                            max(resolved_settings.excel_base_dpi, 400),
                        ),
                        requires_vlm_reading=True,
                        native_text=(
                            f"Sheet: {worksheet.title}\nRange: {cell_range}\n"
                            f"Visual object: {kind}"
                        ),
                    )
                )

            used_range = (
                f"{get_column_letter(min_data_column)}{min_data_row}:"
                f"{get_column_letter(max_data_column)}{max_data_row}"
            )
            surveyed_sheets.append(
                ExcelSheetSurvey(
                    name=worksheet.title,
                    index=sheet_index,
                    visible=True,
                    used_range=used_range,
                    regions=regions,
                )
            )
    finally:
        formula_book.close()
        value_book.close()

    return ExcelWorkbookSurvey(
        source_file=str(source),
        workbook_format=source.suffix.lower().lstrip("."),
        sheets=surveyed_sheets,
        warnings=list(dict.fromkeys(warnings)),
    )


def _convert_workbook_to_xlsx_copy(
    source: Path,
    output_dir: Path,
) -> Path:
    """Buka, hitung ulang, dan simpan salinan XLSX sementara untuk survei."""
    soffice = _find_libreoffice_binary()
    if not soffice:
        raise RuntimeError("LibreOffice diperlukan untuk menyiapkan salinan survei XLSX.")
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = output_dir / "lo_profile_xlsx"
    result = subprocess.run(
        [
            soffice,
            f"-env:UserInstallation={profile.as_uri()}",
            "--headless",
            "--convert-to",
            "xlsx",
            "--outdir",
            str(output_dir),
            str(source),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    target = output_dir / f"{source.stem}.xlsx"
    if result.returncode != 0 or not target.exists():
        raise RuntimeError(
            f"Konversi workbook ke XLSX gagal: {result.stderr.strip()}"
        )
    return target


def _convert_excel_regions_to_pdf(
    excel_path: Path,
    pdf_path: Path,
    profile_dir: Path,
    regions: list[ExcelRegion],
) -> bool:
    """Ekspor setiap region sebagai PDF terpisah lalu gabungkan sesuai urutan survei."""
    soffice = _find_libreoffice_binary()
    if not soffice or not regions:
        return False
    uno_python = _find_uno_python(soffice)
    if uno_python is None:
        return False

    profile_dir.mkdir(parents=True, exist_ok=True)
    region_dir = profile_dir.parent / "region_pdfs"
    region_dir.mkdir(parents=True, exist_ok=True)
    spec_path = profile_dir.parent / "excel_regions.json"
    spec_path.write_text(
        json.dumps(
            [
                {
                    "sheet_index": region.sheet_index,
                    "min_row": region.min_row - 1,
                    "max_row": region.max_row - 1,
                    "min_column": region.min_column - 1,
                    "max_column": region.max_column - 1,
                    "output": str(region_dir / f"{index:04d}.pdf"),
                }
                for index, region in enumerate(regions, start=1)
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
    except OSError as exc:
        logger.warning("Tidak dapat membuka port lokal untuk render region Excel: %s", exc)
        return False

    listener = subprocess.Popen(
        [
            soffice,
            f"-env:UserInstallation={profile_dir.as_uri()}",
            "--headless",
            "--norestore",
            "--nodefault",
            "--nofirststartwizard",
            f"--accept=socket,host=127.0.0.1,port={port};urp;StarOffice.ComponentContext",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    uno_script = r"""
import json
import sys
import time
import uno
from com.sun.star.beans import PropertyValue


def prop(name, value):
    item = PropertyValue()
    item.Name = name
    item.Value = value
    return item


port = int(sys.argv[1])
source_url = uno.systemPathToFileUrl(sys.argv[2])
specs = json.load(open(sys.argv[3], "r", encoding="utf-8"))
local_context = uno.getComponentContext()
resolver = local_context.ServiceManager.createInstanceWithContext(
    "com.sun.star.bridge.UnoUrlResolver", local_context
)
context = None
for _ in range(60):
    try:
        context = resolver.resolve(
            f"uno:socket,host=127.0.0.1,port={port};urp;StarOffice.ComponentContext"
        )
        break
    except Exception:
        time.sleep(0.25)
if context is None:
    raise RuntimeError("Tidak dapat terhubung ke listener LibreOffice UNO")

desktop = context.ServiceManager.createInstanceWithContext(
    "com.sun.star.frame.Desktop", context
)
document = desktop.loadComponentFromURL(
    source_url, "_blank", 0, (prop("Hidden", True),)
)
if document is None:
    raise RuntimeError("LibreOffice tidak dapat membuka workbook")

try:
    sheets = document.getSheets()
    page_styles = document.getStyleFamilies().getByName("PageStyles")
    for spec in specs:
        for sheet_index in range(sheets.getCount()):
            sheet = sheets.getByIndex(sheet_index)
            sheet.setPrintAreas(())
            try:
                sheet.setPropertyValue("AutomaticPrintArea", False)
            except Exception:
                pass
        selected = sheets.getByIndex(int(spec["sheet_index"]))
        address = uno.createUnoStruct("com.sun.star.table.CellRangeAddress")
        address.Sheet = int(spec["sheet_index"])
        address.StartColumn = int(spec["min_column"])
        address.EndColumn = int(spec["max_column"])
        address.StartRow = int(spec["min_row"])
        address.EndRow = int(spec["max_row"])
        selected.setPrintAreas((address,))
        page_style = page_styles.getByName(selected.getPropertyValue("PageStyle"))
        page_style.setPropertyValue("ScaleToPagesX", 1)
        page_style.setPropertyValue("ScaleToPagesY", 1)
        document.calculateAll()
        document.storeToURL(
            uno.systemPathToFileUrl(spec["output"]),
            (prop("FilterName", "calc_pdf_Export"), prop("Overwrite", True)),
        )
finally:
    document.close(True)
    desktop.terminate()
"""
    try:
        subprocess.run(
            [uno_python, "-c", uno_script, str(port), str(excel_path), str(spec_path)],
            capture_output=True,
            text=True,
            timeout=max(180, 60 * len(regions)),
            check=True,
        )
        import pymupdf

        merged = pymupdf.open()
        try:
            for index in range(1, len(regions) + 1):
                region_pdf = region_dir / f"{index:04d}.pdf"
                if not region_pdf.exists():
                    raise RuntimeError(f"PDF region tidak ditemukan: {region_pdf}")
                source_pdf = pymupdf.open(region_pdf)
                try:
                    if len(source_pdf) != 1:
                        raise RuntimeError(
                            f"Region {index} menghasilkan {len(source_pdf)} halaman."
                        )
                    merged.insert_pdf(source_pdf)
                finally:
                    source_pdf.close()
            merged.save(pdf_path)
        finally:
            merged.close()
        return pdf_path.exists() and pdf_page_count(pdf_path) == len(regions)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError) as exc:
        logger.warning("Render region Excel gagal; gunakan render seluruh sheet: %s", exc)
        return False
    finally:
        listener.terminate()
        try:
            listener.wait(timeout=10)
        except subprocess.TimeoutExpired:
            listener.kill()
            listener.wait()
        time.sleep(0.5)


def _detect_header_row(region: ExcelRegion) -> tuple[int, list[str]] | None:
    by_row: dict[int, list[ExcelCellEvidence]] = {}
    for cell in region.cells:
        by_row.setdefault(cell.row, []).append(cell)
    candidates: list[tuple[int, int, list[str]]] = []
    for row_number in sorted(by_row)[:8]:
        cells = sorted(by_row[row_number], key=lambda item: item.column)
        text_count = sum(
            isinstance(cell.value, str) and bool(cell.value.strip()) for cell in cells
        )
        nonempty = [cell for cell in cells if cell.value not in (None, "")]
        if text_count >= 2 and len(nonempty) >= 2:
            headers_by_column = {cell.column: str(cell.value).strip() for cell in nonempty}
            headers = [
                headers_by_column.get(column, f"column_{column - region.min_column + 1}")
                for column in range(region.min_column, region.max_column + 1)
            ]
            candidates.append((text_count, -row_number, headers))
    if not candidates:
        return None
    _, negative_row, headers = max(candidates, key=lambda item: (item[0], item[1]))
    return -negative_row, headers


def _unique_sql_headers(headers: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    result: list[str] = []
    for index, header in enumerate(headers, start=1):
        base = _slug(header, fallback=f"column_{index}")
        seen[base] = seen.get(base, 0) + 1
        result.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    return result


def persist_excel_native_data(
    survey: ExcelWorkbookSurvey,
    db_path: str | Path,
) -> list[str]:
    """Simpan bukti sel dan dataset berulang tanpa mengandalkan hasil OCR/VLM."""
    target = Path(db_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    created_tables: list[str] = []
    grouped_regions: dict[tuple[str, tuple[str, ...]], list[tuple[ExcelRegion, int]]] = {}

    with sqlite3.connect(target) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS excel_regions (
                source_file TEXT NOT NULL,
                sheet_name TEXT NOT NULL,
                region_id TEXT NOT NULL,
                cell_range TEXT NOT NULL,
                title TEXT,
                period TEXT,
                kind TEXT NOT NULL,
                render_dpi INTEGER NOT NULL,
                requires_vlm_reading INTEGER NOT NULL,
                PRIMARY KEY (source_file, region_id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS excel_cells (
                source_file TEXT NOT NULL,
                sheet_name TEXT NOT NULL,
                region_id TEXT NOT NULL,
                coordinate TEXT NOT NULL,
                row_number INTEGER NOT NULL,
                column_number INTEGER NOT NULL,
                value_text TEXT,
                value_number REAL,
                formula TEXT,
                number_format TEXT,
                data_type TEXT,
                merged_range TEXT,
                PRIMARY KEY (source_file, sheet_name, region_id, coordinate)
            )
            """
        )
        connection.execute("DELETE FROM excel_regions WHERE source_file = ?", (survey.source_file,))
        connection.execute("DELETE FROM excel_cells WHERE source_file = ?", (survey.source_file,))

        for sheet in survey.sheets:
            for region in sheet.regions:
                connection.execute(
                    """
                    INSERT INTO excel_regions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        survey.source_file,
                        region.sheet_name,
                        region.region_id,
                        region.cell_range,
                        region.title,
                        region.period,
                        region.kind,
                        region.render_dpi,
                        int(region.requires_vlm_reading),
                    ),
                )
                for cell in region.cells:
                    numeric = (
                        float(cell.value)
                        if isinstance(cell.value, (int, float))
                        and not isinstance(cell.value, bool)
                        else None
                    )
                    connection.execute(
                        """
                        INSERT INTO excel_cells VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            survey.source_file,
                            region.sheet_name,
                            region.region_id,
                            cell.coordinate,
                            cell.row,
                            cell.column,
                            None if cell.value is None else str(cell.value),
                            numeric,
                            cell.formula,
                            cell.number_format,
                            cell.data_type,
                            cell.merged_range,
                        ),
                    )

                header = _detect_header_row(region)
                if header:
                    header_row, raw_headers = header
                    sql_headers = _unique_sql_headers(raw_headers)
                    grouped_regions.setdefault(
                        (region.sheet_name, tuple(sql_headers)), []
                    ).append((region, header_row))

        for (sheet_name, headers), group in grouped_regions.items():
            signature = hashlib.sha1(
                (sheet_name + "|" + "|".join(headers)).encode("utf-8")
            ).hexdigest()[:8]
            table_name = f"excel_{_slug(sheet_name)}_{signature}"
            quoted_columns = ", ".join(f'"{header}"' + " NUMERIC" for header in headers)
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS "{table_name}" (
                    source_file TEXT NOT NULL,
                    sheet_name TEXT NOT NULL,
                    region_id TEXT NOT NULL,
                    period TEXT,
                    source_row INTEGER NOT NULL,
                    {quoted_columns}
                )
                """
            )
            connection.execute(
                f'DELETE FROM "{table_name}" WHERE source_file = ?',
                (survey.source_file,),
            )
            insert_columns = [
                "source_file",
                "sheet_name",
                "region_id",
                "period",
                "source_row",
                *headers,
            ]
            column_sql = ", ".join(f'"{column}"' for column in insert_columns)
            placeholders = ", ".join("?" for _ in insert_columns)
            for region, header_row in group:
                values_by_coordinate = {
                    (cell.row, cell.column): cell.value for cell in region.cells
                }
                for row in range(header_row + 1, region.max_row + 1):
                    values = [
                        values_by_coordinate.get((row, column))
                        for column in range(region.min_column, region.max_column + 1)
                    ]
                    if not any(value not in (None, "") for value in values):
                        continue
                    connection.execute(
                        f'INSERT INTO "{table_name}" ({column_sql}) VALUES ({placeholders})',
                        (
                            survey.source_file,
                            region.sheet_name,
                            region.region_id,
                            region.period,
                            row,
                            *values,
                        ),
                    )
            created_tables.append(table_name)
        connection.commit()
    return created_tables


def excel_page_count(excel_path: str | Path) -> int:
    """Hitung jumlah halaman Excel melalui hasil konversi PDF sementara."""
    with tempfile.TemporaryDirectory(prefix="excel_count_") as tmp:
        pdf_path = convert_excel_to_pdf(excel_path, tmp)
        return pdf_page_count(pdf_path)


def process_multipage_excel(
    excel_path: str | Path,
    pipeline: Any = None,
    *,
    llm: Any = None,
    output_dir: str | Path | None = None,
    dpi: int = DEFAULT_DPI,
    forced_specs: list[str] | str | None = None,
    forced_doc_type: str | None = None,
    db_path: str | Path | None = None,
    auto_tabular_db: bool = True,
    force_all_tables: bool = False,
    output_markdown_path: str | Path | None = None,
    resume: bool = False,
) -> ExtractedDocument:
    """
    Proses workbook Excel dengan jalur visual yang sama seperti PDF.

    File Excel dikonversi ke PDF sementara, lalu halaman PDF dirender menjadi
    gambar dan diproses oleh VLM satu per satu.
    """
    path_obj = Path(excel_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"File Excel tidak ditemukan: {path_obj}")

    with tempfile.TemporaryDirectory(prefix="excel_ingest_") as tmp:
        temporary_root = Path(tmp)
        settings = getattr(pipeline, "settings", None) or get_settings()
        survey: ExcelWorkbookSurvey | None = None
        regions: list[ExcelRegion] = []
        survey_source = path_obj
        if settings.excel_native_survey:
            try:
                try:
                    survey_source = _convert_workbook_to_xlsx_copy(
                        path_obj, temporary_root / "survey"
                    )
                except RuntimeError:
                    if path_obj.suffix.lower() in {".xls", ".ods"}:
                        raise
                    survey_source = path_obj
                survey = survey_excel_workbook(survey_source, settings=settings)
                survey.source_file = str(path_obj)
                survey.workbook_format = path_obj.suffix.lower().lstrip(".")
                regions = [
                    region
                    for sheet in survey.sheets
                    if sheet.visible
                    for region in sheet.regions
                ]
                logger.info(
            "[Excel] Survei native menemukan %d region pada %d sheet terlihat.",
                    len(regions),
                    sum(sheet.visible for sheet in survey.sheets),
                )
            except (OSError, ValueError, RuntimeError, TypeError) as exc:
                logger.warning(
                    "[Excel] Survei struktur gagal; lanjutkan render seluruh sheet: %s",
                    exc,
                )

        pdf_path = temporary_root / f"{path_obj.stem}.pdf"
        region_rendered = False
        if settings.excel_region_rendering and regions:
            region_rendered = _convert_excel_regions_to_pdf(
                path_obj,
                pdf_path,
                temporary_root / "lo_region_profile",
                regions,
            )
        if not region_rendered:
            logger.info("[Excel] Mengonversi '%s' ke PDF sementara...", path_obj.name)
            pdf_path = convert_excel_to_pdf(path_obj, tmp)
        logger.info(
            "[Excel] Konversi selesai, lanjut render halaman ke gambar dan ekstraksi VLM."
        )
        native_by_page = (
            {index: region.native_text for index, region in enumerate(regions, start=1)}
            if region_rendered
            else None
        )
        dpi_by_page = (
            {index: region.render_dpi for index, region in enumerate(regions, start=1)}
            if region_rendered
            else None
        )
        rescue_by_page = (
            {
                index: region.requires_vlm_reading
                for index, region in enumerate(regions, start=1)
            }
            if region_rendered
            else None
        )
        result = process_multipage_pdf(
            pdf_path=pdf_path,
            pipeline=pipeline,
            llm=llm,
            output_dir=output_dir,
            dpi=dpi,
            forced_specs=forced_specs,
            forced_doc_type=forced_doc_type,
            db_path=db_path,
            auto_tabular_db=auto_tabular_db,
            force_all_tables=force_all_tables,
            output_markdown_path=output_markdown_path,
            source_file_for_records=path_obj,
            resume=resume,
            native_text_by_page_override=native_by_page,
            dpi_by_page_override=dpi_by_page,
            force_vlm_reading_by_page_override=rescue_by_page,
        )

        if survey is not None:
            result.excel_workbook = survey
            if output_dir:
                output_root = Path(output_dir).resolve().parent
            elif output_markdown_path:
                output_root = Path(output_markdown_path).resolve().parent
            else:
                output_root = Path("output") / path_obj.stem
            manifest_path = output_root / "regions" / "excel_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(survey.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            if db_path:
                native_db_path = Path(db_path).resolve()
            elif output_markdown_path:
                native_db_path = (
                    Path(output_markdown_path).resolve().parent
                    / "databases"
                    / f"{path_obj.stem}.sqlite"
                )
            elif output_dir:
                native_db_path = (
                    Path(output_dir).resolve()
                    / "databases"
                    / f"{path_obj.stem}.sqlite"
                )
            else:
                native_db_path = (
                    Path("output")
                    / path_obj.stem
                    / "databases"
                    / f"{path_obj.stem}.sqlite"
                )
            if auto_tabular_db:
                result.excel_native_tables = persist_excel_native_data(
                    survey, native_db_path
                )
                try:
                    from .tabular_db import TabularDatabaseManager

                    csv_dir = (
                        native_db_path.parent.parent / "csv"
                        if native_db_path.parent.name == "databases"
                        else native_db_path.parent / "csv"
                    )
                    TabularDatabaseManager(native_db_path).export_to_csv(
                        output_dir=csv_dir
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[Excel] Ekspor data native ke CSV gagal: %s", exc)

        if region_rendered and len(result.pages) == len(regions):
            blocks: list[str] = []
            current_sheet: str | None = None
            for page, region in zip(result.pages, regions, strict=True):
                page_parts: list[str] = []
                if region.sheet_name != current_sheet:
                    page_parts.append(f"## Sheet: {region.sheet_name}")
                    current_sheet = region.sheet_name
                label = region.period or region.title or region.cell_range
                page_parts.append(f"### {label}")
                page_parts.append(
                    f"<!-- excel_region: {region.region_id}; range: "
                    f"'{region.sheet_name}'!{region.cell_range} -->"
                )
                page_parts.append(page.markdown_content.strip())
                page.markdown_content = "\n\n".join(
                    part for part in page_parts if part
                ).strip()
                blocks.append(page.markdown_content)
            title = result.title or path_obj.stem
            result.title = title
            result.full_markdown = f"# {title}\n\n" + "\n\n---\n\n".join(blocks)
            if output_markdown_path:
                Path(output_markdown_path).write_text(
                    result.full_markdown, encoding="utf-8"
                )

        return result


__all__ = [
    "convert_excel_to_pdf",
    "excel_page_count",
    "persist_excel_native_data",
    "process_multipage_excel",
    "survey_excel_workbook",
]
