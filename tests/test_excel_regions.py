import gc
import sqlite3
import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from app.config import Settings
from app.excel import (
    ExcelTileBudgetError,
    _convert_excel_regions_to_pdf,
    _convert_workbook_to_xlsx_copy,
    _estimate_native_tokens,
    persist_excel_native_data,
    split_excel_regions_into_tiles,
    survey_excel_workbook,
)
from app.pdf import pdf_page_count
from app.ppt import _find_libreoffice_binary


class TestExcelRegionSurvey(unittest.TestCase):
    def test_token_budget_splits_dense_cells_without_losing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dense.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            for row in range(1, 15):
                sheet.cell(row, 1, f"Row {row}")
                sheet.cell(row, 2, "=SUM(" + ",".join(["12345"] * 30) + ")")
            workbook.save(path)
            survey = survey_excel_workbook(path)
            settings = Settings(excel_tile_max_native_tokens=450)
            tiled = split_excel_regions_into_tiles(survey, settings)
            regions = tiled.sheets[0].regions
            self.assertGreater(len(regions), 1)
            self.assertTrue(all(_estimate_native_tokens(r.native_text) <= 450 for r in regions))
            expected = {c.coordinate: c.formula for r in survey.sheets[0].regions for c in r.cells}
            actual = {c.coordinate: c.formula for r in regions for c in r.cells}
            self.assertEqual(actual, expected)
            with self.assertRaises(ExcelTileBudgetError):
                split_excel_regions_into_tiles(survey, Settings(excel_tile_max_native_tokens=10))

    def test_merged_caption_does_not_bridge_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "caption.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.merge_cells("A1:N1")
            sheet["A1"] = "Shared caption"
            for column, marker in ((1, "LEFT"), (9, "RIGHT")):
                for row in range(2, 40):
                    for offset in range(4):
                        sheet.cell(row, column + offset, f"{marker}{row}_{offset}")
            workbook.save(path)
            survey = survey_excel_workbook(path)
            regions = survey.sheets[0].regions
            self.assertEqual([r.cell_range for r in regions], ["A1:N1", "A2:D39", "I2:L39"])
            self.assertNotIn("RIGHT", regions[1].native_text)
            self.assertNotIn("LEFT", regions[2].native_text)

    @unittest.skipUnless(_find_libreoffice_binary(), "LibreOffice tidak tersedia")
    def test_vertical_tiles_render_only_their_rows(self) -> None:
        import pymupdf

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "tall.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.column_dimensions["A"].width = 24
            for row in range(1, 45):
                sheet.cell(row, 1, f"MARKER{row:03d}")
                sheet.cell(row, 2, row)
            workbook.save(path)
            settings = Settings(excel_tile_max_columns=8, excel_tile_max_rows=20)
            survey = split_excel_regions_into_tiles(
                survey_excel_workbook(path, settings=settings), settings,
            )
            regions = survey.sheets[0].regions
            pdf = root / "tiles.pdf"
            self.assertTrue(_convert_excel_regions_to_pdf(
                path, pdf, root / "profile", regions,
            ))
            with pymupdf.open(pdf) as document:
                self.assertEqual(len(document), len(regions))
                for page, region in zip(document, regions, strict=True):
                    text = page.get_text()
                    expected_rows = {cell.row for cell in region.cells}
                    for row in range(1, 45):
                        self.assertEqual(
                            f"MARKER{row:03d}" in text, row in expected_rows,
                            f"{region.region_id}: wrong presence of row {row}",
                        )

    def _build_side_by_side_workbook(self, path: Path) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Billing"

        for start_column, month, values in (
            (2, "FEB 2026", (3287, 1_036_330_654, 113_993_234, 1_150_323_888)),
            (8, "MAR 2026", (3035, 951_306_917, 104_641_630, 1_055_948_547)),
        ):
            sheet.merge_cells(
                start_row=2,
                start_column=start_column,
                end_row=2,
                end_column=start_column + 4,
            )
            title = sheet.cell(2, start_column, f"INVOICE 1 {month}")
            title.font = Font(bold=True, color="FFFFFF", size=7)
            title.fill = PatternFill("solid", fgColor="000000")
            headers = ["Category", "Subs", "Current Charges", "Tax", "Charges + Tax"]
            for offset, header in enumerate(headers):
                cell = sheet.cell(3, start_column + offset, header)
                cell.font = Font(bold=True, color="FFFFFF", size=7)
                cell.fill = PatternFill("solid", fgColor="000000")
            sheet.cell(4, start_column, "New Sub MANUAL")
            for offset, value in enumerate(values, start=1):
                sheet.cell(4, start_column + offset, value)
            sheet.cell(5, start_column, "TOTAL")
            for offset in range(1, 5):
                source = sheet.cell(4, start_column + offset).coordinate
                sheet.cell(5, start_column + offset, f"={source}")

        sheet["N2"] = "Rotated note"
        sheet["N2"].alignment = Alignment(text_rotation=45)
        chart = BarChart()
        chart.add_data(Reference(sheet, min_col=3, min_row=3, max_row=5), titles_from_data=True)
        chart.anchor = "N8"
        sheet.add_chart(chart)
        workbook.save(path)

    def test_side_by_side_tables_become_separate_regions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "billing.xlsx"
            self._build_side_by_side_workbook(path)

            survey = survey_excel_workbook(
                path,
                settings=Settings(
                    excel_small_font_points=8,
                    excel_base_dpi=300,
                    excel_max_dpi=450,
                ),
            )

            regions = survey.sheets[0].regions
            invoice_regions = [region for region in regions if region.kind == "table"]
            self.assertEqual(len(invoice_regions), 2)
            self.assertEqual(
                {region.period for region in invoice_regions},
                {"FEB 2026", "MAR 2026"},
            )
            self.assertTrue(all(region.render_dpi >= 400 for region in invoice_regions))
            self.assertTrue(
                all(region.requires_vlm_reading for region in invoice_regions)
            )
            self.assertNotIn("N2", {cell.coordinate for region in invoice_regions for cell in region.cells})
            chart_regions = [region for region in regions if region.title == "Chart 1"]
            self.assertEqual(len(chart_regions), 1)
            self.assertTrue(chart_regions[0].requires_vlm_reading)

    def test_native_tables_group_matching_monthly_regions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workbook_path = root / "billing.xlsx"
            database_path = root / "billing.sqlite"
            self._build_side_by_side_workbook(workbook_path)
            survey = survey_excel_workbook(workbook_path)

            created = persist_excel_native_data(survey, database_path)

            self.assertEqual(len(created), 1)
            connection = sqlite3.connect(database_path)
            try:
                periods = {
                    row[0]
                    for row in connection.execute(
                        f'SELECT DISTINCT period FROM "{created[0]}"'
                    ).fetchall()
                }
                source_rows = connection.execute(
                    f'SELECT COUNT(*) FROM "{created[0]}"'
                ).fetchone()[0]
                evidence_count = connection.execute(
                    "SELECT COUNT(*) FROM excel_cells"
                ).fetchone()[0]
            finally:
                connection.close()
            gc.collect()

            self.assertEqual(periods, {"FEB 2026", "MAR 2026"})
            self.assertEqual(source_rows, 4)
            self.assertGreater(evidence_count, 20)

    def test_wide_region_is_tiled_and_can_be_combined(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wide.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Wide"
            for column in range(1, 25):
                sheet.cell(1, column, f"H{column}")
                for row_number in range(2, 35):
                    sheet.cell(row_number, column, row_number * column)
            workbook.save(path)

            survey = survey_excel_workbook(path, settings=Settings(
                excel_tile_max_columns=12,
                excel_tile_max_rows=40,
            ))
            tiled = split_excel_regions_into_tiles(survey, Settings(
                excel_tile_max_columns=12,
                excel_tile_max_rows=40,
            ))
            regions = tiled.sheets[0].regions
            self.assertGreaterEqual(len(regions), 2)
            self.assertTrue(all(region.is_tile for region in regions))
            self.assertEqual({region.parent_region_id for region in regions}, {"s001_r001"})
            self.assertEqual(
                {cell.coordinate for region in regions for cell in region.cells},
                {f"{get_column_letter(column)}{row}" for column in range(1, 25) for row in range(1, 35)},
            )

    @unittest.skipUnless(_find_libreoffice_binary(), "LibreOffice tidak tersedia")
    def test_region_renderer_produces_one_page_per_region(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workbook_path = root / "billing.xlsx"
            pdf_path = root / "billing.pdf"
            self._build_side_by_side_workbook(workbook_path)
            survey = survey_excel_workbook(workbook_path)
            invoice_regions = [
                region
                for region in survey.sheets[0].regions
                if region.kind == "table"
            ]

            rendered = _convert_excel_regions_to_pdf(
                workbook_path,
                pdf_path,
                root / "lo_profile",
                invoice_regions,
            )

            self.assertTrue(rendered)
            self.assertEqual(pdf_page_count(pdf_path), 2)

    @unittest.skipUnless(_find_libreoffice_binary(), "LibreOffice tidak tersedia")
    def test_survey_copy_recalculates_formula_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workbook_path = root / "billing.xlsx"
            self._build_side_by_side_workbook(workbook_path)

            recalculated = _convert_workbook_to_xlsx_copy(
                workbook_path, root / "recalculated"
            )
            survey = survey_excel_workbook(recalculated)
            formula_cells = [
                cell
                for region in survey.sheets[0].regions
                for cell in region.cells
                if cell.formula
            ]

            self.assertTrue(formula_cells)
            self.assertTrue(all(cell.value is not None for cell in formula_cells))


if __name__ == "__main__":
    unittest.main()
