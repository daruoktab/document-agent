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
from contextlib import closing
from datetime import date, datetime
from datetime import time as datetime_time
from pathlib import Path
from typing import Any

from .config import DEFAULT_DPI, Settings, get_settings
from .pdf import pdf_page_count, process_multipage_pdf
from .ppt import _find_libreoffice_binary
from .schemas import (
    ExcelCellEvidence,
    ExcelChartEvidence,
    ExcelChartSeries,
    ExcelNativeArtifact,
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


def _formula_dependencies(worksheet: Any, sheet_names: set[str]) -> list[str]:
    dependencies: set[str] = set()
    pattern = re.compile(r"(?:'((?:[^']|'')+)'|([A-Za-z_][A-Za-z0-9_. -]*))!")
    for cell in getattr(worksheet, "_cells", {}).values():
        if getattr(cell, "data_type", None) != "f":
            continue
        for quoted, plain in pattern.findall(str(cell.value)):
            candidate = (quoted or plain).replace("''", "'").strip()
            if candidate in sheet_names and candidate != worksheet.title:
                dependencies.add(candidate)
    return sorted(dependencies)


def _chart_title(chart: Any, fallback: str) -> str:
    try:
        paragraphs = chart.title.tx.rich.p
        value = "".join(
            run.t or "" for paragraph in paragraphs for run in paragraph.r
        ).strip()
        return value or fallback
    except (AttributeError, TypeError):
        return fallback


def _cache_values(reference: Any) -> tuple[str | None, list[Any]]:
    if reference is None:
        return None, []
    formula = getattr(reference, "f", None)
    cache = getattr(reference, "strCache", None) or getattr(
        reference, "numCache", None
    )
    points = getattr(cache, "pt", None) or []
    if not points:
        return formula, []
    point_count = int(getattr(cache, "ptCount", 0) or 0)
    max_index = max(int(point.idx) for point in points)
    values: list[Any] = [None] * max(point_count, max_index + 1)
    for point in points:
        value: Any = point.v
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        values[int(point.idx)] = value
    return formula, values


def _reference_values(formula: str | None, workbook: Any) -> list[Any]:
    if not formula or "!" not in formula:
        return []
    sheet_token, cell_range = formula.rsplit("!", 1)
    sheet_name = sheet_token.strip("'").replace("''", "'")
    if sheet_name not in workbook.sheetnames:
        return []
    try:
        cells = workbook[sheet_name][cell_range.replace("$", "")]
    except (KeyError, TypeError, ValueError):
        return []
    if not isinstance(cells, tuple):
        return [cells.value]
    values: list[Any] = []
    for row in cells:
        if isinstance(row, tuple):
            values.extend(cell.value for cell in row)
        else:
            values.append(row.value)
    return values


def _series_title(series: Any, index: int, workbook: Any = None) -> str:
    text = getattr(series, "tx", None)
    reference = getattr(text, "strRef", None)
    formula, cached = _cache_values(reference)
    if cached and cached[0] not in (None, ""):
        return str(cached[0])
    referenced = _reference_values(formula, workbook) if workbook is not None else []
    if referenced and referenced[0] not in (None, ""):
        return str(referenced[0])
    literal = getattr(text, "v", None)
    return str(literal).strip() if literal else f"Series {index}"


def _category_family(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return "date"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "number"
    text = str(value or "").strip()
    if re.fullmatch(
        r"(?:jan|feb|mar|apr|may|mei|jun|jul|aug|agu|sep|oct|okt|nov|dec|des)[a-z]*[- /]\d{2,4}",
        text,
        flags=re.IGNORECASE,
    ):
        return "date"
    return "text"


def _extract_chart_evidence(
    worksheet: Any,
    *,
    value_workbook: Any = None,
) -> list[ExcelChartEvidence]:
    try:
        from openpyxl.utils import get_column_letter
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("openpyxl diperlukan untuk membaca grafik workbook.") from exc

    charts: list[ExcelChartEvidence] = []
    for chart_index, chart in enumerate(getattr(worksheet, "_charts", []), start=1):
        anchor = getattr(chart, "anchor", None)
        start = getattr(anchor, "_from", None)
        end = getattr(anchor, "to", None)
        if start is not None:
            min_row = int(start.row) + 1
            min_column = int(start.col) + 1
            max_row = int(end.row) + 1 if end is not None else min_row + 14
            max_column = int(end.col) + 1 if end is not None else min_column + 7
        else:
            min_row = min_column = 1
            max_row, max_column = 15, 8
        cell_range = (
            f"{get_column_letter(min_column)}{min_row}:"
            f"{get_column_letter(max_column)}{max_row}"
        )
        evidence_series: list[ExcelChartSeries] = []
        warnings: list[str] = []
        for series_index, series in enumerate(chart.series, start=1):
            category_source = getattr(series, "cat", None)
            category_reference = getattr(category_source, "strRef", None) or getattr(
                category_source, "numRef", None
            )
            value_source = getattr(series, "val", None)
            value_reference = getattr(value_source, "numRef", None)
            category_formula, categories = _cache_values(category_reference)
            value_formula, raw_values = _cache_values(value_reference)
            if not categories and value_workbook is not None:
                categories = _reference_values(category_formula, value_workbook)
            if not raw_values and value_workbook is not None:
                raw_values = _reference_values(value_formula, value_workbook)
            values = [
                value if isinstance(value, (int, float)) else None
                for value in raw_values
            ]
            series_name = _series_title(series, series_index, value_workbook)
            if len(categories) != len(values):
                warnings.append(
                    f"Seri {series_name} memiliki jumlah kategori "
                    "dan nilai yang berbeda."
                )
            families = {
                _category_family(category)
                for category, value in zip(categories, values, strict=False)
                if category not in (None, "") and value is not None
            }
            if "date" in families and len(families) > 1:
                warnings.append(
                    f"Seri {series_name} mencampur kategori "
                    "periode dengan kategori nonperiode."
                )
            if any(value is None for value in values):
                warnings.append(f"Seri {series_name} memiliki nilai kosong.")
            evidence_series.append(
                ExcelChartSeries(
                    name=series_name,
                    category_reference=category_formula,
                    value_reference=value_formula,
                    categories=[_json_safe_cell_value(value) for value in categories],
                    values=values,
                )
            )
        charts.append(
            ExcelChartEvidence(
                chart_id=f"s{worksheet._parent.index(worksheet) + 1:03d}_c{chart_index:03d}",
                title=_chart_title(chart, f"Chart {chart_index}"),
                chart_type=type(chart).__name__.removesuffix("Chart").lower(),
                cell_range=cell_range,
                series=evidence_series,
                warnings=list(dict.fromkeys(warnings)),
            )
        )
    return charts


def _classify_excel_sheets(sheets: list[ExcelSheetSurvey]) -> list[str]:
    dashboards = {sheet.name for sheet in sheets if sheet.charts and sheet.visible}
    dashboard_dependencies = {
        dependency
        for sheet in sheets
        if sheet.name in dashboards
        for dependency in sheet.dependencies
    }
    for sheet in sheets:
        if not sheet.visible:
            continue
        formula_ratio = sheet.formula_count / max(1, sheet.nonempty_cell_count)
        largest = max(
            (region for region in sheet.regions if region.kind == "table"),
            key=lambda region: len(region.cells),
            default=None,
        )
        tall_dataset = bool(
            largest
            and largest.max_row - largest.min_row + 1
            >= max(12, 3 * (largest.max_column - largest.min_column + 1))
            and len(largest.cells)
            >= 0.55
            * (largest.max_row - largest.min_row + 1)
            * (largest.max_column - largest.min_column + 1)
        )
        if sheet.charts:
            sheet.role = "dashboard"
            sheet.role_confidence = 0.98
            sheet.role_reasons = ["memiliki grafik native"]
            sheet.render_strategy = "hybrid"
        elif (
            sheet.name in dashboard_dependencies
            and formula_ratio >= 0.35
            and sheet.dependencies
        ):
            sheet.role = "support"
            sheet.role_confidence = 0.9
            sheet.role_reasons = [
                "menjadi sumber dashboard",
                "didominasi formula agregasi",
            ]
            sheet.render_strategy = "native"
        elif formula_ratio >= 0.35 and sheet.dependencies:
            sheet.role = "summary"
            sheet.role_confidence = min(0.95, 0.65 + formula_ratio / 3)
            sheet.role_reasons = ["merangkum sheet lain melalui formula"]
            if len([region for region in sheet.regions if region.kind == "table"]) > 1:
                sheet.role_reasons.append("memiliki beberapa blok tabel ringkasan")
            sheet.render_strategy = "native"
        elif tall_dataset:
            sheet.role = "detail"
            sheet.role_confidence = 0.9
            sheet.role_reasons = [
                "memiliki header dan pola baris berulang",
                "bentuk data lebih tinggi daripada lebar",
            ]
            sheet.render_strategy = "native"
        else:
            sheet.role = "plain"
            sheet.role_confidence = 0.65
            sheet.role_reasons = ["tidak menunjukkan pola dashboard atau agregasi"]
            sheet.render_strategy = "hybrid"

        for region in sheet.regions:
            if sheet.role in {"detail", "summary", "support"} or sheet.role == "dashboard":
                region.render_strategy = "native"
            else:
                region.render_strategy = "hybrid"
            region.persist_native = region.kind == "table" and sheet.role != "support"

    role_order = {"dashboard": 0, "summary": 1, "plain": 2, "detail": 3, "support": 4}
    return [
        sheet.name
        for sheet in sorted(
            (item for item in sheets if item.visible),
            key=lambda item: (role_order[item.role], item.index),
        )
    ]


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
            chart_evidence = _extract_chart_evidence(
                worksheet, value_workbook=value_book
            )
            for chart in chart_evidence:
                warnings.extend(
                    f"{worksheet.title} - {chart.title}: {message}"
                    for message in chart.warnings
                )
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

            merged_by_coordinate: dict[tuple[int, int], str] = {}
            merged_bounds_by_anchor: dict[tuple[int, int], tuple[int, int, int, int]] = {}
            bounds: list[tuple[int, int, int, int]] = [
                (row, row, column, column) for row, column in data_coordinates
            ]
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
                if anchor_value is not None:
                    merged_bounds_by_anchor[(min_row, min_column)] = (
                        min_row,
                        max_row,
                        min_column,
                        max_column,
                    )
                    bounds.append((min_row, max_row, min_column, max_column))
                for row in range(min_row, max_row + 1):
                    for column in range(min_column, max_column + 1):
                        merged_by_coordinate[(row, column)] = str(merged_range)

            bounds.extend(
                (min_row, max_row, min_column, max_column)
                for _, min_row, max_row, min_column, max_column in drawing_specs
            )
            min_data_row = min(item[0] for item in bounds)
            max_data_row = max(item[1] for item in bounds)
            min_data_column = min(item[2] for item in bounds)
            max_data_column = max(item[3] for item in bounds)

            used_width = max_data_column - min_data_column + 1
            isolated_merged_anchors = {
                anchor
                for anchor, (min_row, max_row, min_column, max_column) in merged_bounds_by_anchor.items()
                if min_row == max_row
                and max_column - min_column + 1 >= 8
                and max_column - min_column + 1 >= used_width * 0.5
            }
            connected_coordinates = data_coordinates - isolated_merged_anchors

            regions: list[ExcelRegion] = []
            components = sorted(
                [
                    *_connected_components(connected_coordinates),
                    *({anchor} for anchor in isolated_merged_anchors),
                ],
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
                for anchor in component:
                    merged_bounds = merged_bounds_by_anchor.get(anchor)
                    if merged_bounds is None:
                        continue
                    merged_min_row, merged_max_row, merged_min_col, merged_max_col = (
                        merged_bounds
                    )
                    min_row = min(min_row, merged_min_row)
                    max_row = max(max_row, merged_max_row)
                    min_column = min(min_column, merged_min_col)
                    max_column = max(max_column, merged_max_col)
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
                    nonempty_cell_count=len(data_coordinates),
                    formula_count=sum(
                        cell.data_type == "f" for cell in raw_cells
                    ),
                    row_count=max_data_row - min_data_row + 1,
                    column_count=max_data_column - min_data_column + 1,
                    dependencies=_formula_dependencies(
                        worksheet, set(formula_book.sheetnames)
                    ),
                    charts=chart_evidence,
                    regions=regions,
                )
            )
    finally:
        formula_book.close()
        value_book.close()

    extraction_order = _classify_excel_sheets(surveyed_sheets)
    return ExcelWorkbookSurvey(
        source_file=str(source),
        workbook_format=source.suffix.lower().lstrip("."),
        sheets=surveyed_sheets,
        extraction_order=extraction_order,
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


def _markdown_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).replace("|", "\\|").replace("\n", "<br>").strip()


def _region_markdown(region: ExcelRegion, *, max_rows: int = 200) -> str:
    if region.kind == "text":
        values = [
            _markdown_cell(cell.value)
            for cell in sorted(region.cells, key=lambda item: (item.row, item.column))
            if cell.value not in (None, "")
        ]
        return "\n\n".join(dict.fromkeys(values))

    header = _detect_header_row(region)
    if header is None:
        return _region_native_text(
            sheet_name=region.sheet_name,
            cell_range=region.cell_range,
            cells=region.cells,
        )
    header_row, raw_headers = header
    headers = [header or f"Kolom {index}" for index, header in enumerate(raw_headers, 1)]
    values_by_coordinate = {
        (cell.row, cell.column): cell.value for cell in region.cells
    }
    rows: list[list[str]] = []
    for row_number in range(header_row + 1, region.max_row + 1):
        values = [
            _markdown_cell(values_by_coordinate.get((row_number, column)))
            for column in range(region.min_column, region.max_column + 1)
        ]
        if any(values):
            rows.append(values)
    visible_rows = rows[:max_rows]
    lines = [
        "| " + " | ".join(_markdown_cell(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in visible_rows)
    if len(rows) > max_rows:
        lines.append(
            f"\n> Tabel diringkas: {max_rows} dari {len(rows)} baris ditampilkan. "
            "Data lengkap tersedia pada artefak CSV/SQLite."
        )
    return "\n".join(lines)


def _chart_points(
    series: ExcelChartSeries,
) -> list[tuple[Any, float | int]]:
    return [
        (category, value)
        for category, value in zip(series.categories, series.values, strict=False)
        if category not in (None, "") and value is not None
    ]


def _chart_narrative(chart: ExcelChartEvidence) -> list[str]:
    narratives: list[str] = []
    for series in chart.series:
        points = _chart_points(series)
        if not points:
            continue
        date_points = [
            (category, value)
            for category, value in points
            if _category_family(category) == "date"
        ]
        if len(date_points) >= 2 and len(date_points) >= len(points) / 2:
            start_category, start_value = date_points[0]
            end_category, end_value = date_points[-1]
            max_category, max_value = max(date_points, key=lambda item: item[1])
            min_category, min_value = min(date_points, key=lambda item: item[1])
            change = end_value - start_value
            narratives.append(
                f"Seri **{series.name}** berubah dari {_markdown_cell(start_value)} "
                f"pada {_markdown_cell(start_category)} menjadi {_markdown_cell(end_value)} "
                f"pada {_markdown_cell(end_category)} (perubahan {_markdown_cell(change)}). "
                f"Nilai tertinggi {_markdown_cell(max_value)} pada "
                f"{_markdown_cell(max_category)} dan terendah {_markdown_cell(min_value)} "
                f"pada {_markdown_cell(min_category)}."
            )
        else:
            max_category, max_value = max(points, key=lambda item: item[1])
            min_category, min_value = min(points, key=lambda item: item[1])
            narratives.append(
                f"Pada seri **{series.name}**, nilai terbesar adalah "
                f"{_markdown_cell(max_value)} untuk {_markdown_cell(max_category)}, "
                f"sedangkan nilai terkecil adalah {_markdown_cell(min_value)} untuk "
                f"{_markdown_cell(min_category)}."
            )
    return narratives


def _chart_markdown(chart: ExcelChartEvidence, *, max_points: int = 200) -> str:
    max_length = max((len(series.categories) for series in chart.series), default=0)
    headers = ["Kategori", *(series.name for series in chart.series)]
    rows: list[list[str]] = []
    for index in range(min(max_length, max_points)):
        category = next(
            (
                series.categories[index]
                for series in chart.series
                if index < len(series.categories)
                and series.categories[index] not in (None, "")
            ),
            "",
        )
        values = [
            series.values[index] if index < len(series.values) else None
            for series in chart.series
        ]
        if category not in (None, "") or any(value is not None for value in values):
            rows.append([_markdown_cell(category), *map(_markdown_cell, values)])
    parts = [
        f"### Grafik: {chart.title}",
        "| " + " | ".join(_markdown_cell(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]
    narratives = _chart_narrative(chart)
    if narratives:
        parts.append("\n".join(f"> {item}" for item in narratives))
    if max_length > max_points:
        parts.append(
            f"> Grafik memiliki {max_length} titik; hanya {max_points} titik pertama "
            "ditampilkan di Markdown."
        )
    if chart.warnings:
        parts.append("\n".join(f"> Peringatan: {item}" for item in chart.warnings))
    return "\n".join(parts)


def _reference_cell_range(reference: str | None) -> str | None:
    if not reference or "!" not in reference:
        return None
    return reference.rsplit("!", 1)[1].replace("$", "")


def _reference_sheet_name(reference: str | None) -> str | None:
    if not reference or "!" not in reference:
        return None
    return reference.rsplit("!", 1)[0].strip("'").replace("''", "'")


def _range_contains(outer: str, inner: str) -> bool:
    try:
        from openpyxl.utils.cell import range_boundaries

        outer_min_col, outer_min_row, outer_max_col, outer_max_row = range_boundaries(
            outer
        )
        inner_min_col, inner_min_row, inner_max_col, inner_max_row = range_boundaries(
            inner
        )
    except (ImportError, TypeError, ValueError):
        return False
    return (
        outer_min_col <= inner_min_col <= inner_max_col <= outer_max_col
        and outer_min_row <= inner_min_row <= inner_max_row <= outer_max_row
    )


def _sheet_markdown(
    sheet: ExcelSheetSurvey,
    *,
    artifacts: list[ExcelNativeArtifact],
    visual_markdown: str | None = None,
) -> str:
    parts = [f"## Sheet: {sheet.name}"]
    if sheet.role == "detail":
        sheet_artifacts = [
            artifact for artifact in artifacts if artifact.sheet_name == sheet.name
        ]
        if sheet_artifacts:
            data_rows = sum(artifact.row_count for artifact in sheet_artifacts)
            columns = next(
                (artifact.columns for artifact in sheet_artifacts if artifact.columns),
                [],
            )
        else:
            largest = max(
                (region for region in sheet.regions if region.kind == "table"),
                key=lambda region: len(region.cells),
                default=None,
            )
            header = _detect_header_row(largest) if largest else None
            data_rows = (
                max(0, largest.max_row - header[0])
                if largest is not None and header is not None
                else sheet.row_count
            )
            columns = header[1] if header else []
        parts.append(f"Data detail terdiri dari {data_rows} baris dan {len(columns)} kolom.")
        if columns:
            parts.append("Kolom: " + ", ".join(f"`{column}`" for column in columns))
        for artifact in sheet_artifacts:
            location = artifact.csv_path or artifact.table_name
            parts.append(
                f"- Data lengkap: `{location}` ({artifact.row_count} baris)."
            )
        return "\n\n".join(parts)
    if sheet.role == "support":
        dependencies = ", ".join(sheet.dependencies) or "tidak ada"
        parts.append(
            "Sheet pendukung dipertahankan dalam manifest untuk audit formula. "
            f"Sumber yang dirujuk: {dependencies}."
        )
        return "\n\n".join(parts)

    chart_source_ranges = {
        cell_range
        for chart in sheet.charts
        for series in chart.series
        for reference, cell_range in (
            (
                series.category_reference,
                _reference_cell_range(series.category_reference),
            ),
            (series.value_reference, _reference_cell_range(series.value_reference)),
        )
        if cell_range and _reference_sheet_name(reference) == sheet.name
    }
    for region in sheet.regions:
        if region.kind == "mixed":
            continue
        if sheet.role == "dashboard" and any(
            _range_contains(region.cell_range, source_range)
            for source_range in chart_source_ranges
        ):
            continue
        rendered = _region_markdown(region)
        if rendered:
            if region.kind == "table" and region.title:
                parts.append(f"### {region.title}\n\n{rendered}")
            else:
                parts.append(rendered)
    parts.extend(_chart_markdown(chart) for chart in sheet.charts)
    if visual_markdown and visual_markdown.strip():
        parts.append(f"### Konteks visual\n\n{visual_markdown.strip()}")
    return "\n\n".join(parts)


def _workbook_title(survey: ExcelWorkbookSurvey, fallback: str) -> str:
    ordered_names = survey.extraction_order or [sheet.name for sheet in survey.sheets]
    sheets = {sheet.name: sheet for sheet in survey.sheets}
    for name in ordered_names:
        sheet = sheets[name]
        if sheet.role != "dashboard":
            continue
        for region in sheet.regions:
            for cell in sorted(region.cells, key=lambda item: (item.row, item.column)):
                if isinstance(cell.value, str) and cell.value.strip():
                    return cell.value.strip()
    return fallback


def compose_excel_markdown(
    survey: ExcelWorkbookSurvey,
    *,
    fallback_title: str,
    artifacts: list[ExcelNativeArtifact] | None = None,
    visual_by_sheet: dict[str, str] | None = None,
) -> tuple[str, list[tuple[str, str]]]:
    artifacts = artifacts or []
    visual_by_sheet = visual_by_sheet or {}
    title = _workbook_title(survey, fallback_title)
    sheets_by_name = {sheet.name: sheet for sheet in survey.sheets}
    sections: list[tuple[str, str]] = []
    extraction_order = survey.extraction_order or [
        sheet.name for sheet in survey.sheets if sheet.visible
    ]
    for sheet_name in extraction_order:
        sheet = sheets_by_name[sheet_name]
        markdown = _sheet_markdown(
            sheet,
            artifacts=artifacts,
            visual_markdown=visual_by_sheet.get(sheet_name),
        )
        sections.append((sheet_name, markdown))
    if survey.warnings:
        warning_text = "## Peringatan Workbook\n\n" + "\n".join(
            f"- {warning}" for warning in survey.warnings
        )
        sections.append(("Peringatan Workbook", warning_text))
    full_markdown = f"# {title}\n\n" + "\n\n---\n\n".join(
        markdown for _, markdown in sections if markdown.strip()
    )
    return full_markdown.strip(), sections


def plan_excel_visual_regions(
    survey: ExcelWorkbookSurvey,
    *,
    settings: Settings | None = None,
) -> list[ExcelRegion]:
    """Pilih unit visual yang benar-benar memerlukan pembacaan VLM."""
    resolved_settings = settings or get_settings()
    planned: list[ExcelRegion] = []
    for sheet in survey.sheets:
        if not sheet.visible:
            continue
        if sheet.role == "dashboard" and sheet.used_range:
            cells_by_coordinate = {
                cell.coordinate: cell
                for region in sheet.regions
                for cell in region.cells
            }
            min_row = min((region.min_row for region in sheet.regions), default=1)
            max_row = max((region.max_row for region in sheet.regions), default=1)
            min_column = min(
                (region.min_column for region in sheet.regions), default=1
            )
            max_column = max(
                (region.max_column for region in sheet.regions), default=1
            )
            planned.append(
                ExcelRegion(
                    region_id=f"sheet_{sheet.index + 1:03d}_dashboard",
                    sheet_name=sheet.name,
                    sheet_index=sheet.index,
                    cell_range=sheet.used_range,
                    min_row=min_row,
                    max_row=max_row,
                    min_column=min_column,
                    max_column=max_column,
                    title=sheet.name,
                    kind="mixed",
                    render_dpi=resolved_settings.excel_base_dpi,
                    requires_vlm_reading=True,
                    render_strategy="visual",
                    persist_native=False,
                    native_text=_sheet_markdown(sheet, artifacts=[]),
                    cells=list(cells_by_coordinate.values()),
                )
            )
            continue
        if sheet.role in {"detail", "support"}:
            continue
        planned.extend(
            region
            for region in sheet.regions
            if region.render_strategy in {"visual", "hybrid"}
            and (region.requires_vlm_reading or region.kind == "mixed")
        )
    return planned


def _describe_excel_native_artifacts(
    survey: ExcelWorkbookSurvey,
    db_path: Path,
    table_names: list[str],
    csv_paths: list[Path],
) -> list[ExcelNativeArtifact]:
    csv_by_stem = {path.stem: path for path in csv_paths}
    artifacts: list[ExcelNativeArtifact] = []
    with closing(sqlite3.connect(db_path)) as connection:
        for table_name in table_names:
            quoted = table_name.replace('"', '""')
            columns = [
                row[1]
                for row in connection.execute(
                    f'PRAGMA table_info("{quoted}")'
                ).fetchall()
                if row[1]
                not in {"source_file", "sheet_name", "region_id", "period", "source_row"}
            ]
            row_count = connection.execute(
                f'SELECT COUNT(*) FROM "{quoted}" WHERE source_file = ?',
                (survey.source_file,),
            ).fetchone()[0]
            sheet_row = connection.execute(
                f'SELECT sheet_name FROM "{quoted}" WHERE source_file = ? LIMIT 1',
                (survey.source_file,),
            ).fetchone()
            artifacts.append(
                ExcelNativeArtifact(
                    sheet_name=sheet_row[0] if sheet_row else "",
                    table_name=table_name,
                    row_count=row_count,
                    columns=columns,
                    csv_path=(
                        str(csv_by_stem[table_name])
                        if table_name in csv_by_stem
                        else None
                    ),
                )
            )
    return artifacts


def persist_excel_native_data(
    survey: ExcelWorkbookSurvey,
    db_path: str | Path,
) -> list[str]:
    """Simpan bukti sel dan dataset berulang tanpa mengandalkan hasil OCR/VLM."""
    target = Path(db_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    created_tables: list[str] = []
    grouped_regions: dict[tuple[str, tuple[str, ...]], list[tuple[ExcelRegion, int]]] = {}

    with closing(sqlite3.connect(target)) as connection:
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

                header = _detect_header_row(region) if region.persist_native else None
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


def convert_excel_for_extraction(
    excel_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Render rencana visual adaptif, dengan konversi penuh sebagai fallback."""
    source = Path(excel_path).resolve()
    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    target = target_dir / f"{source.stem}.pdf"
    try:
        survey = survey_excel_workbook(source, settings=settings)
        visual_regions = plan_excel_visual_regions(survey, settings=settings)
        if visual_regions and _convert_excel_regions_to_pdf(
            source,
            target,
            target_dir / "lo_region_profile",
            visual_regions,
        ):
            return target
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        logger.warning("Survei adaptif gagal; gunakan konversi penuh: %s", exc)
    return convert_excel_to_pdf(source, target_dir)


def excel_page_count(excel_path: str | Path) -> int:
    """Hitung jumlah unit visual dari rencana ekstraksi adaptif."""
    with tempfile.TemporaryDirectory(prefix="excel_count_") as tmp:
        pdf_path = convert_excel_for_extraction(excel_path, tmp)
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
    Proses workbook Excel dengan jalur native dan visual yang adaptif.

    Dashboard atau blok campuran dirender untuk VLM, sedangkan ringkasan, formula,
    grafik native, dan data detail dipertahankan langsung dari struktur workbook.
    """
    path_obj = Path(excel_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"File Excel tidak ditemukan: {path_obj}")

    with tempfile.TemporaryDirectory(prefix="excel_ingest_") as tmp:
        temporary_root = Path(tmp)
        settings = getattr(pipeline, "settings", None) or get_settings()
        survey: ExcelWorkbookSurvey | None = None
        visual_regions: list[ExcelRegion] = []
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
                visual_regions = plan_excel_visual_regions(
                    survey, settings=settings
                )
                logger.info(
                    "[Excel] Survei native merencanakan %d unit visual pada %d sheet terlihat.",
                    len(visual_regions),
                    sum(sheet.visible for sheet in survey.sheets),
                )
            except (OSError, ValueError, RuntimeError, TypeError) as exc:
                logger.warning(
                    "[Excel] Survei struktur gagal; lanjutkan render seluruh sheet: %s",
                    exc,
                )

        pdf_path = temporary_root / f"{path_obj.stem}.pdf"
        region_rendered = False
        if settings.excel_region_rendering and visual_regions:
            region_rendered = _convert_excel_regions_to_pdf(
                path_obj,
                pdf_path,
                temporary_root / "lo_region_profile",
                visual_regions,
            )
        needs_visual_pipeline = bool(visual_regions) or survey is None
        if not region_rendered and needs_visual_pipeline:
            logger.info("[Excel] Mengonversi '%s' ke PDF sementara...", path_obj.name)
            pdf_path = convert_excel_to_pdf(path_obj, tmp)
        elif not needs_visual_pipeline:
            logger.info("[Excel] Tidak ada unit visual; ekstraksi diselesaikan secara native.")
        native_by_page = (
            {
                index: region.native_text
                for index, region in enumerate(visual_regions, start=1)
            }
            if region_rendered
            else None
        )
        dpi_by_page = (
            {
                index: region.render_dpi
                for index, region in enumerate(visual_regions, start=1)
            }
            if region_rendered
            else None
        )
        rescue_by_page = (
            {
                index: region.requires_vlm_reading
                for index, region in enumerate(visual_regions, start=1)
            }
            if region_rendered
            else None
        )
        if needs_visual_pipeline:
            logger.info(
                "[Excel] Konversi selesai, lanjut render unit visual dan ekstraksi VLM."
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
                output_markdown_path=None,
                source_file_for_records=path_obj,
                resume=resume,
                native_text_by_page_override=native_by_page,
                dpi_by_page_override=dpi_by_page,
                force_vlm_reading_by_page_override=rescue_by_page,
            )
        else:
            result = ExtractedDocument(
                source_file=str(path_obj),
                title=path_obj.stem,
                doc_type="spreadsheet",
                full_markdown="",
                total_pages=1,
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
                csv_paths: list[Path] = []
                try:
                    from .tabular_db import TabularDatabaseManager

                    csv_dir = (
                        native_db_path.parent.parent / "csv"
                        if native_db_path.parent.name == "databases"
                        else native_db_path.parent / "csv"
                    )
                    csv_paths = TabularDatabaseManager(native_db_path).export_to_csv(
                        output_dir=csv_dir
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[Excel] Ekspor data native ke CSV gagal: %s", exc)
                result.excel_native_artifacts = _describe_excel_native_artifacts(
                    survey,
                    native_db_path,
                    result.excel_native_tables,
                    csv_paths,
                )

            visual_by_sheet: dict[str, str] = {}
            if region_rendered and len(result.pages) == len(visual_regions):
                for page, region in zip(result.pages, visual_regions, strict=True):
                    content = page.markdown_content.strip()
                    if content:
                        visual_by_sheet[region.sheet_name] = "\n\n".join(
                            part
                            for part in (
                                visual_by_sheet.get(region.sheet_name),
                                content,
                            )
                            if part
                        )
            result.full_markdown, _ = compose_excel_markdown(
                survey,
                fallback_title=result.title or path_obj.stem,
                artifacts=result.excel_native_artifacts,
                visual_by_sheet=visual_by_sheet,
            )
            result.title = _workbook_title(survey, result.title or path_obj.stem)
            result.total_tables = sum(
                region.kind == "table"
                for sheet in survey.sheets
                for region in sheet.regions
                if sheet.role != "support"
            )
            if output_markdown_path:
                markdown_target = Path(output_markdown_path).resolve()
                markdown_target.parent.mkdir(parents=True, exist_ok=True)
                markdown_target.write_text(result.full_markdown, encoding="utf-8")

        return result


__all__ = [
    "compose_excel_markdown",
    "convert_excel_for_extraction",
    "convert_excel_to_pdf",
    "excel_page_count",
    "persist_excel_native_data",
    "plan_excel_visual_regions",
    "process_multipage_excel",
    "survey_excel_workbook",
]
